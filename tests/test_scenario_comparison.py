from dataclasses import replace
from datetime import datetime
import hashlib
import json

import pytest

from basket_hedging.engine import Engine
from basket_hedging.replay import load_case
from basket_hedging.risk import UnnettedLiabilityStressMargin
from basket_hedging.scenario_comparison import run_scenario_comparison
from basket_hedging.scenario_hedge import ScenarioHedge
from basket_hedging.scenarios import ConditionalGBMScenarios
from basket_hedging.synthetic import generate_synthetic_case
from basket_hedging.valuation import PricingParameters


def test_four_policy_comparison_reproduces_inputs_metrics_and_hand_examples(tmp_path):
    path = generate_synthetic_case(tmp_path / "case.json", days=2, samples=128)
    before = (path.read_bytes(), path.with_suffix('.jsonl').read_bytes())
    first = run_scenario_comparison(path, tmp_path / "first", samples=16, pricing_samples=64)
    second = run_scenario_comparison(path, tmp_path / "second", samples=16, pricing_samples=64)
    assert first['outcomes'] == second['outcomes']
    assert first['diagnostics'] == second['diagnostics']
    assert [row['policy'] for row in first['outcomes']] == ['unhedged', 'delta', 'variance', 'cont_vuletic_l1']
    assert all(row['status'] == 'COMPLETED' for row in first['outcomes'])
    assert first['outcomes'][0]['accepted_hedge_fill_count'] == 0
    for row in first['outcomes']:
        saved = json.loads((tmp_path / 'first' / row['policy'] / 'summary.json').read_text())
        components = saved['current']['pnl_components']
        # Gross tracking residual plus actual costs/financing reconciles to net account P&L.
        assert row['net_pnl'] == pytest.approx(
            -row['gross_hedge_error_currency'] - row['transaction_costs'] - row['financing_costs'])
        assert row['gross_hedge_error_currency'] == -(components['liability_pnl'] + components['hedge_pnl'])
        assert row['accepted_hedge_fill_count'] == saved['accepted_hedge_fill_count']
        assert row['margin_label'] == 'uncalibrated research margin proxy'
    assert before == (path.read_bytes(), path.with_suffix('.jsonl').read_bytes())
    assert first['provenance']['files'][0]['sha256'] == hashlib.sha256(before[0]).hexdigest()
    examples = json.loads((tmp_path / 'first' / 'optimization_examples.json').read_text())
    assert examples['perfect']['solution']['holdings'] == pytest.approx([2])
    assert examples['perfect']['solution']['mse'] == pytest.approx(0, abs=1e-25)
    assert examples['redundant']['solution']['holdings'] == pytest.approx([1.5, 0])


@pytest.mark.parametrize('alpha', [0, .1])
def test_future_market_mutation_cannot_change_scenario_hedge_prefix(tmp_path, alpha):
    case = load_case(generate_synthetic_case(tmp_path / 'case.json', days=3, samples=128))
    raw = dict(case.metadata['pricing_parameters'])
    raw['available_at'] = datetime.fromisoformat(raw['available_at'])
    parameters = PricingParameters(**raw)
    events = list(case.events())
    cutoff = events[2].time
    modified = []
    for event in events:
        if event.time <= cutoff:
            modified.append(event)
            continue
        def move(marks):
            return {key: mark * 1.01 if case.instruments[key].kind == 'stock' else mark + .2
                    for key, mark in marks.items()}
        valuation = replace(event.valuation, marks=move(event.valuation.marks),
                            stress_marks={name: move(marks) for name, marks in event.valuation.stress_marks.items()})
        modified.append(replace(event, valuation=valuation,
            observations=tuple(replace(o, value=o.value * 1.01) for o in event.observations),
            quotes=tuple(replace(q, bid=q.bid * 1.01, ask=q.ask * 1.01) for q in event.quotes),
            fixings=tuple(replace(f, spots={key: spot * 1.01 for key, spot in f.spots.items()})
                          for f in event.fixings)))
    results, policies = [], []
    for replay in (events, modified):
        source = ConditionalGBMScenarios(parameters,
            replay_volatilities=tuple(case.metadata['replay_volatilities']),
            replay_correlation=tuple(tuple(row) for row in case.metadata['replay_correlation']), samples=16)
        policy = ScenarioHedge(source, case.rules.hedge_universe,
                               {key: .02 for key in case.rules.hedge_universe}, alpha=alpha)
        results.append(Engine(case.instruments, case.opening, case.rules,
                               UnnettedLiabilityStressMargin()).run(replay, policy))
        policies.append(policy)
    left, right = results
    assert [s for s in left.snapshots if s.time <= cutoff] == [s for s in right.snapshots if s.time <= cutoff]
    assert [p for p in left.ledger.postings if p.time <= cutoff] == [p for p in right.ledger.postings if p.time <= cutoff]
    assert [o for o in left.order_outcomes if o.time <= cutoff] == [o for o in right.order_outcomes if o.time <= cutoff]
    assert [d for d in policies[0].diagnostics if d['time'] <= cutoff] == [d for d in policies[1].diagnostics if d['time'] <= cutoff]
    assert left.ledger.equity != right.ledger.equity
