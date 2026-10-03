"""Known review calendars set scenario horizons without reading future prices."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from basket_hedging.contracts import DataGap, InputError, Instrument
from basket_hedging.engine import Engine, PolicyView
from basket_hedging.market import Observation, ValuationPacket
from basket_hedging.replay import load_case
from basket_hedging.risk import UnnettedLiabilityStressMargin
from basket_hedging.scenario_hedge import ScenarioHedge
from basket_hedging.scenarios import ConditionalGBMScenarios
from basket_hedging.synthetic import generate_synthetic_case
from basket_hedging.valuation import PricingParameters


MONDAY = datetime(2026, 1, 5, 16, tzinfo=timezone.utc)
FRIDAY = datetime(2026, 1, 9, 16, tzinfo=timezone.utc)


def state(time=MONDAY, *, review_hours=24, execution_hours=17, expiry_hours=720):
    instruments = {"S": Instrument("S", "stock"),
                   "B": Instrument("B", "basket_put", multiplier=10, strike=110,
                                   expiry=time + timedelta(hours=expiry_hours), coefficients=(("S", 1),))}
    marks = {"S": 100.0, "B": 8.0}
    observation = Observation("initial-S", "S", 100, time, time)
    packet = ValuationPacket(time, marks, {"unchanged": marks}, observed_at={key: time for key in marks})
    return PolicyView("decision", time, 0, instruments, {"B": -2}, 10000, 0, 0, 10000,
                      packet, (observation,), (), ("S",), False,
                      {"earliest_execution": time + timedelta(hours=execution_hours),
                       "next_rebalance_time": time + timedelta(hours=review_hours)},
                      market={"S": observation})


def generator():
    pricing = PricingParameters(("S",), (0.,), ((1.,),), 0, MONDAY, 71, samples=4)
    return ConditionalGBMScenarios(pricing, replay_volatilities=(.2,),
                                   replay_correlation=((1.,),), samples=4, seed=123)


def test_weekday_horizon_preserves_prechange_24_hour_draws():
    generated = generator().generate(state(), {"B": -2}, ("S",))
    assert generated.horizon == MONDAY + timedelta(hours=24)
    # Frozen before changing the horizon API, with the same seed/model inputs.
    np.testing.assert_array_equal(generated.delta_h[:, 0],
        [1.0370823420762747, -1.1161819136176376, -0.43968847384829246, -0.6239738527241059])
    np.testing.assert_array_equal(generated.delta_v,
        [19.258353158474506, 62.32363827235275, 48.79376947696585, 52.47947705448212])


@pytest.mark.parametrize("hours,execution_hours", [(24, 17), (72, 65)])
def test_elapsed_time_changes_gbm_drift_and_diffusion(hours, execution_hours):
    generated = generator().generate(state(review_hours=hours, execution_hours=execution_hours),
                                      {"B": -2}, ("S",))
    # Independent standard-normal references for PCG64 seeds derived from seed
    # 123, Monday's UTC time, model tag and row indices 0..3. No market_path call.
    z = np.array([.990803958707767, -1.0669922302374446, -.41570367306356953, -.5926833173131765])
    years = hours / (365 * 24)
    future_spots = 100 * np.exp(-.5 * .2**2 * years + .2 * np.sqrt(years) * z)
    np.testing.assert_allclose(generated.delta_h[:, 0], future_spots - 100, rtol=0, atol=1e-12)
    np.testing.assert_allclose(generated.delta_v, 20 * (np.maximum(110-future_spots, 0)-8),
                               rtol=0, atol=1e-12)


def test_friday_horizon_is_monday_close_after_monday_open_eligibility():
    current = state(FRIDAY, review_hours=72, execution_hours=65)
    generated = generator().generate(current, {"B": -2}, ("S",))
    assert current.rules["earliest_execution"] == datetime(2026, 1, 12, 9, tzinfo=timezone.utc)
    assert generated.horizon == datetime(2026, 1, 12, 16, tzinfo=timezone.utc)
    assert generated.horizon > current.rules["earliest_execution"] > FRIDAY + timedelta(hours=24)


@pytest.mark.parametrize("timing", [
    {}, {"next_rebalance_time": MONDAY + timedelta(hours=24)},
    {"earliest_execution": MONDAY + timedelta(hours=17)},
    {"earliest_execution": MONDAY + timedelta(hours=17), "next_rebalance_time": MONDAY + timedelta(hours=16)},
    {"earliest_execution": MONDAY + timedelta(hours=17), "next_rebalance_time": MONDAY + timedelta(hours=17)},
])
def test_missing_or_misaligned_timing_is_a_gap(timing):
    with pytest.raises(DataGap):
        generator().generate(replace(state(), rules=timing), {"B": -2}, ("S",))


@pytest.mark.parametrize("execution_hours,review_hours", [(17, 0), (17, -1), (-1, 24)])
def test_nonfuture_review_or_past_execution_timing_is_malformed(execution_hours, review_hours):
    with pytest.raises(InputError):
        generator().generate(state(execution_hours=execution_hours, review_hours=review_hours),
                             {"B": -2}, ("S",))


@pytest.mark.parametrize("expiry_hours", [16, 17])
def test_expiry_at_or_before_order_eligibility_cannot_be_a_hedging_horizon(expiry_hours):
    with pytest.raises(DataGap):
        generator().generate(state(expiry_hours=expiry_hours), {"B": -2}, ("S",))


def test_relevant_expiry_after_eligibility_truncates_to_exact_payoff():
    generated = generator().generate(state(expiry_hours=20), {"B": -2}, ("S",))
    assert generated.horizon == MONDAY + timedelta(hours=20)
    future_spots = 100 + generated.delta_h[:, 0]
    np.testing.assert_allclose(generated.delta_v, 20*(np.maximum(110-future_spots, 0)-8),
                               rtol=0, atol=1e-12)


@pytest.mark.parametrize("delay_hours", [0, 70])
def test_released_calendar_and_effective_eligibility_reach_policy_view(tmp_path, delay_hours):
    case = load_case(generate_synthetic_case(tmp_path / "calendar.json", days=6, samples=16))
    events = list(case.events())
    friday = next(event for event in events if event.time == FRIDAY)
    assert friday.next_rebalance_time == datetime(2026, 1, 12, 16, tzinfo=timezone.utc)
    seen = {}

    def observe(view):
        seen[view.time] = view
        return None

    settings = replace(case.rules, execution_delay_seconds=delay_hours*3600)
    result = Engine(case.instruments, case.opening, settings, UnnettedLiabilityStressMargin()).run(events, observe)
    assert result.status == "COMPLETED"
    assert seen[MONDAY].rules["next_rebalance_time"] == MONDAY + timedelta(hours=24)
    assert seen[FRIDAY].rules["next_rebalance_time"] == friday.next_rebalance_time
    assert seen[FRIDAY].rules["earliest_execution"] == FRIDAY + timedelta(hours=max(65, delay_hours))


def test_future_price_mutation_preserves_friday_scenarios_and_financial_prefix(tmp_path):
    case = load_case(generate_synthetic_case(tmp_path / "case.json", days=6, samples=32))
    events = list(case.events())
    changed = []
    for event in events:
        if event.time <= FRIDAY:
            changed.append(event)
            continue

        def move(marks):
            return {key: value*1.01 if case.instruments[key].kind == "stock" else value+.2
                    for key, value in marks.items()}

        changed.append(replace(event,
            valuation=replace(event.valuation, marks=move(event.valuation.marks),
                              stress_marks={key: move(marks) for key, marks in event.valuation.stress_marks.items()}),
            observations=tuple(replace(row, value=row.value*1.01) for row in event.observations),
            quotes=tuple(replace(row, bid=row.bid*1.01, ask=row.ask*1.01) for row in event.quotes),
            fixings=tuple(replace(row, spots={key: spot*1.01 for key, spot in row.spots.items()}) for row in event.fixings)))
    raw = dict(case.metadata["pricing_parameters"])
    raw["available_at"] = datetime.fromisoformat(raw["available_at"])
    pricing = PricingParameters(**raw)
    policies, results = [], []
    for replay in (events, changed):
        source = ConditionalGBMScenarios(pricing,
            replay_volatilities=tuple(case.metadata["replay_volatilities"]),
            replay_correlation=tuple(tuple(row) for row in case.metadata["replay_correlation"]), samples=8)
        policy = ScenarioHedge(source, case.rules.hedge_universe, {key: .02 for key in case.rules.hedge_universe})
        results.append(Engine(case.instruments, case.opening, case.rules,
                              UnnettedLiabilityStressMargin()).run(replay, policy))
        policies.append(policy)
    assert next(row for row in policies[0].diagnostics if row["time"] == FRIDAY)["horizon"] == FRIDAY + timedelta(hours=72)
    assert [row for row in policies[0].diagnostics if row["time"] <= FRIDAY] == [
        row for row in policies[1].diagnostics if row["time"] <= FRIDAY]
    for attribute in ("snapshots", "order_outcomes"):
        assert [row for row in getattr(results[0], attribute) if row.time <= FRIDAY] == [
            row for row in getattr(results[1], attribute) if row.time <= FRIDAY]
    assert [post for post in results[0].ledger.postings if post.time <= FRIDAY] == [
        post for post in results[1].ledger.postings if post.time <= FRIDAY]
    assert results[0].ledger.equity != results[1].ledger.equity
