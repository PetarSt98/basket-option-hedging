"""Mechanical absolute holdings; scheduling and hedge choice stay outside the engine."""

from typing import Mapping

from .contracts import DataGap
from .engine import PolicyView


def no_hedge(view: PolicyView) -> Mapping[str, float]:
    """Close permitted hedges; keep the nontradable basket liability."""
    return {key: 0.0 for key in view.hedge_universe}


def stock_delta_hedge(view: PolicyView) -> Mapping[str, float]:
    """Offset all held option deltas; option targets are zero for this stock baseline.

    Include option hedges until they actually close. For a stock-only book this
    equals the final pseudocode's signed basket delta. Round share targets to
    the nearest configured lot (ties to even), never cash or option marks.
    """
    targets = dict(no_hedge(view))
    stocks = {key for key in view.hedge_universe if view.instruments[key].kind == "stock"}
    for key, quantity in view.positions.items():
        instrument = view.instruments[key]
        if not instrument.is_option or quantity == 0:
            continue
        unit_deltas = view.valuation.deltas.get(key, {})
        for stock in instrument.risk_stocks:
            if stock not in stocks:
                continue
            if stock not in unit_deltas:
                raise DataGap(f"missing unit delta of {key} with respect to {stock}")
            targets[stock] -= quantity * instrument.multiplier * unit_deltas[stock]
    for stock in stocks:
        lot = view.rules.get("lot_sizes", {}).get(stock, 0.0)
        if lot:
            targets[stock] = round(targets[stock] / lot) * lot
    return targets


class UnhedgedPolicy:
    """Named callable adapter for reports and the CLI."""

    name = "unhedged"

    def targets(self, view: PolicyView) -> Mapping[str, float]:
        return no_hedge(view)


class ScheduledDeltaPolicy:
    """Offset all nonspot deltas with absolute stock targets at engine decisions."""

    name = "delta"

    def targets(self, view: PolicyView) -> Mapping[str, float] | None:
        if not view.decision_event:
            return None
        return stock_delta_hedge(view)
