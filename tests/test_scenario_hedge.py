from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from basket_hedging.contracts import InputError, Instrument
from basket_hedging.engine import PolicyView
from basket_hedging.market import ValuationPacket, readonly
from basket_hedging.orders import Order
from basket_hedging.scenario_hedge import ScenarioHedge
from basket_hedging.scenarios import ScenarioChanges


TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)


def view(stock=0, *, decision=True):
    registry = {"S": Instrument("S", "stock"),
                "B": Instrument("B", "basket_call", multiplier=100, strike=100,
                                expiry=TIME + timedelta(days=30), coefficients=(("S", 1),))}
    marks = {"S": 100, "B": 2}
    packet = ValuationPacket(TIME, marks, {"unchanged": marks}, observed_at={key: TIME for key in marks})
    return PolicyView("event", TIME, 0, readonly(registry), readonly({"B": -1, "S": stock}),
                      1200, 200, 0, 1000, packet, (), (), ("S",), False, readonly({}),
                      decision_event=decision)


class FixedScenarios:
    def generate(self, state, fixed_positions, hedge_ids):
        assert fixed_positions == {"B": -1}  # Current hedge must not enter the response twice.
        x = np.array([-1., 1.])
        return ScenarioChanges(hedge_ids, 3 + 2 * x, x[:, None], TIME + timedelta(days=1), 42)


def test_strategy_returns_absolute_targets_and_does_not_mutate_view():
    state = view(stock=1)
    pending = Order("old", "S", .5, TIME - timedelta(days=1), TIME, TIME + timedelta(days=1), False)
    state = replace(state, pending_orders=(pending,))
    policy = ScenarioHedge(FixedScenarios(), ("S",), {"S": .02})
    assert policy(state) == pytest.approx({"S": 2})  # Not the trade of +1, or pending-adjusted +.5.
    assert state.positions == {"B": -1, "S": 1}
    assert state.cash == 1200 and state.pending_orders == (pending,)
    assert policy.gross_scale == 200  # Absolute marked basket value, not 1200 cash or risk capital.
    assert policy.diagnostics[0]["intercept"] == pytest.approx(3)


def test_actual_holdings_anchor_penalty_and_gross_scale_is_fixed():
    policy = ScenarioHedge(FixedScenarios(), ("S",), {"S": 1}, alpha=.01)
    initial = view(stock=1, decision=False)
    assert policy(initial) is None
    assert policy.gross_scale == 200
    marked_later = replace(view(stock=1), valuation=replace(initial.valuation, marks={"S": 100, "B": 9}))
    # lambda=.01*200*1=2: beta=soft(1,1)=0, so retain one actual share.
    assert policy(marked_later) == {"S": 1}
    assert policy.gross_scale == 200
    assert policy.diagnostics[0]["previous_holdings"] == {"S": 1}


@pytest.mark.parametrize("ids,costs", [(('B',), {'B': .1}), (('unknown',), {'unknown': .1})])
def test_strategy_cannot_optimize_unpermitted_instruments(ids, costs):
    policy = ScenarioHedge(FixedScenarios(), ids, costs)
    with pytest.raises(InputError, match="permitted hedges"):
        policy(view())


@pytest.mark.parametrize("kwargs", [dict(alpha=-1), dict(gross_scale=0)])
def test_invalid_strategy_configuration(kwargs):
    with pytest.raises(InputError):
        ScenarioHedge(FixedScenarios(), ('S',), {'S': .02}, **kwargs)
