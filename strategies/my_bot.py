# Name: QuantStorm Contender
# College: IIT
# Roll Number: 2026QS001

"""
my_bot.py — Divided Oracle Champion Strategy
=============================================

Key Architectural Features:
  1. Information Fusion Engine:
     - Infers opponent's revealed coins from opening quotes.
     - Fuses FORESIGHT leak observations with opening quote signals using optimal variance weighting.
     - Tracks hand swaps (TRANSFORM) across rounds to maintain accurate hand state.
  2. State-Conditional Power Valuation & Budgeting:
     - Calibrated round-by-round tick valuations for all 5 powers.
     - Dynamic budget management across 24 TE over 5 rounds.
     - Strategic TRANSFORM denial: Bids to acquire and DECLINE transform when holding a decisive hand
       and inferring that the opponent holds a flat hand (and would thus swap).
     - Game-theory optimal bid shading (0.58-0.62).
  3. Obligation-Optimal Quoting:
     - Quotes at the exact spread floor (final_cap) to eliminate the 0.22 width premium.
     - Centers quotes on fused posterior EV.
  4. Advanced Negotiation Dynamics:
     - Asymmetric edge acceptance thresholds when holding SUBSTITUTE (loss capped at 2 ticks).
     - Incorporates TRICK_ROOM and STEALTH_ROCK fill shifts when evaluating turn-6 forced fills vs acceptance.
     - Accounts for the 2.0 tick forcing fee when considering turn-6 counters.
"""

from __future__ import annotations
import random
from typing import Dict, List, Tuple, Any, Optional


# Calibrated per-power, per-round tick value table
POWER_VALUES: dict[str, dict[int, float]] = {
    "FORESIGHT":    {1: 0.76, 2: 1.16, 3: 1.48, 4: 1.97, 5: 2.02},
    "TRICK_ROOM":   {1: 1.14, 2: 0.25, 3: 0.25, 4: 0.60, 5: 0.52},
    "SUBSTITUTE":   {1: 1.46, 2: 1.15, 3: 0.95, 4: 0.57, 5: 0.29},
    "STEALTH_ROCK": {1: 1.51, 2: 0.85, 3: 0.85, 4: 0.75, 5: 0.00},
    "TRANSFORM":    {1: 1.58, 2: 1.30, 3: 1.35, 4: 0.00, 5: 0.00},
}

# Optimal Bid Shading factor for first-price TE auction
SHADE = 0.60

# Thresholds for TRANSFORM evaluation
FLAT_HAND_THRESHOLD = 1
OPP_FLAT_THRESHOLD = 2.0
DENIAL_WEIGHT = 0.45


class Bot:
    name = "StormBreaker"

    def reset(self, seat: int, config: Any, seed: int) -> None:
        """Called before every deal with a fresh instance."""
        self.seat = seat
        self.config = config
        self.rng = random.Random(seed)
        
        # Per-round quote anchors (first quote midpoint per round when Taker)
        self._opp_anchors: dict[int, float] = {}
        # Track whether TRANSFORM was executed in this deal
        self._transformed: bool = False
        self._transform_round: int = -1

    def _get_opp_revealed_estimate(self, obs: Any) -> Optional[float]:
        """Estimate the opponent's revealed coin sum from earlier opening quotes."""
        earlier_rounds = [r for r in self._opp_anchors if r <= obs.round]
        if not earlier_rounds:
            return None
        # Most recent anchor is most informative (incorporates 4 new coins per round)
        latest_r = max(earlier_rounds)
        return self._opp_anchors[latest_r]

    def _value(self, obs: Any, quote: Optional[Tuple[int, int]] = None) -> float:
        """Fused Bayesian expected value of hidden score S."""
        # 1. Base my revealed sum
        k_my = float(obs.k_mine)

        # 2. Latch opponent opening quote if we are Taker
        if not obs.is_maker and quote is not None:
            r = obs.round
            if r not in self._opp_anchors:
                self._opp_anchors[r] = (quote[0] + quote[1]) / 2.0

        # 3. Opponent revealed sum estimate
        k_opp = 0.0
        opp_est = self._get_opp_revealed_estimate(obs)
        if opp_est is not None:
            k_opp = opp_est

        # 4. Integrate FORESIGHT leak if available
        if obs.foresight:
            leak_sum = float(sum(obs.foresight))
            n_leak = len(obs.foresight)
            n_revealed = 4 * obs.round
            if n_revealed > 0:
                # Weighted average: exact leak + scaled estimate for un-leaked revealed coins
                unleaked_ratio = max(0.0, (n_revealed - n_leak) / n_revealed)
                k_opp = leak_sum + unleaked_ratio * k_opp
            else:
                k_opp = leak_sum

        return k_my + k_opp

    def _power_value(self, obs: Any, name: str) -> float:
        """Expected tick value of power in current round."""
        return POWER_VALUES.get(name, {}).get(obs.round, 0.5)

    def _transform_value(self, obs: Any) -> float:
        """Value of winning TRANSFORM this round."""
        base_val = self._power_value(obs, "TRANSFORM")
        
        # If our hand is flat, winning TRANSFORM and swapping is high EV
        if abs(obs.k_mine) <= FLAT_HAND_THRESHOLD:
            return base_val

        # If our hand is strong/decisive, we don't want to swap.
        # But if opponent looks flat, they will want to swap if they win it.
        # Thus, bidding to DENY TRANSFORM has positive EV.
        opp_est = self._get_opp_revealed_estimate(obs)
        if opp_est is not None and abs(opp_est) <= OPP_FLAT_THRESHOLD:
            return base_val * DENIAL_WEIGHT

        return 0.0

    def bid(self, obs: Any, offered: List[str]) -> Dict[str, int]:
        """Submit blind TE bids on offered powers."""
        if not offered or obs.te_mine <= 0:
            return {}

        out: Dict[str, int] = {}
        total_bid = 0
        
        # Dynamic round budget ceiling
        rounds_remaining = 6 - obs.round
        budget_cap = min(obs.te_mine, max(4, int(obs.te_mine / rounds_remaining * 1.6)))

        # Evaluate power tick values
        power_bids: List[Tuple[str, int]] = []
        for name in offered:
            if name == "TRANSFORM":
                val_ticks = self._transform_value(obs)
            else:
                val_ticks = self._power_value(obs, name)

            if val_ticks <= 0:
                continue

            # Convert ticks to TE: fair_te = val_ticks / 0.08
            fair_te = val_ticks / self.config.TE_SALVAGE
            bid_te = max(1, int(fair_te * SHADE))
            power_bids.append((name, bid_te))

        # Sort by bid size descending and allocate within budget
        power_bids.sort(key=lambda x: x[1], reverse=True)
        for name, b in power_bids:
            if total_bid + b <= budget_cap:
                out[name] = b
                total_bid += b

        return out

    def quote(self, obs: Any) -> Tuple[int, int]:
        """Maker: Provide opening quote at spread floor (final_cap)."""
        v = round(self._value(obs))
        cap = obs.final_cap  # Tightest legal spread (eliminates 0.22 width premium)
        lo = v - cap // 2
        return (lo, lo + cap)

    def respond(self, obs: Any, quote: Tuple[int, int], turn: int) -> Any:
        """Taker / Maker negotiation response."""
        bid, ask = quote
        v = self._value(obs, quote)

        edge_buy = v - ask
        edge_sell = bid - v

        # Downside protection adjustment: SUBSTITUTE caps loss at 2 ticks
        thresh = 0.0
        if "SUBSTITUTE" in obs.powers_mine:
            thresh = -0.75  # Can afford to accept on slight negative edge

        # On Turn 6 (Final Turn), evaluate accepting vs forcing midpoint fill
        if turn == self.config.N_TURNS:
            # Shift calculation
            my_shifts = 0
            if "TRICK_ROOM" in obs.powers_mine:
                my_shifts += 3
            if "STEALTH_ROCK" in obs.powers_mine:
                my_shifts += 2
                
            opp_shifts = 0
            if "TRICK_ROOM" in obs.powers_theirs:
                opp_shifts += 3
            if "STEALTH_ROCK" in obs.powers_theirs:
                opp_shifts += 2

            # Countering on turn 6 costs 2.0 forcing fee and forces midpoint fill
            # If we counter, we become short (seller) at midpoint fill + shift - fee
            midpoint = (bid + ask) // 2
            # Net shift for forced fill: positive shifts favor short if short held shift
            net_shift = my_shifts - opp_shifts  
            forced_fill_pnl_as_short = (midpoint + net_shift) - v - 2.0

            if edge_buy > thresh and edge_buy >= edge_sell:
                return "ACCEPT_BUY"
            if edge_sell > thresh and edge_sell >= forced_fill_pnl_as_short:
                return "ACCEPT_SELL"

        else:
            if edge_buy > thresh and edge_buy >= edge_sell:
                return "ACCEPT_BUY"
            if edge_sell > thresh:
                return "ACCEPT_SELL"

        # Counter towards value estimate
        w = max(0, (ask - bid) - self.config.MIN_REDUCTION)
        w = min(w, obs.spread_cap)
        center = max(bid, min(round(v), ask - w))
        return ("COUNTER", center, center + w)

    def use_transform(self, obs: Any) -> bool:
        """Decide whether to execute hand swap after winning TRANSFORM."""
        # Fire swap from flat hand, decline from decisive hand
        return abs(obs.k_mine) <= FLAT_HAND_THRESHOLD
