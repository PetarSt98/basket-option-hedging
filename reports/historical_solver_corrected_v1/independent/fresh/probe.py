"""Read-only complete first historical episode, causal mutation, IV sensitivity."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
import json, csv, sys, math
import numpy as np
ROOT=Path(r'C:/Users/petar/Desktop/basket-option-hedging')
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from basket_hedging import historical as h
from basket_hedging import historical_hedge as hh
from basket_hedging.etf_data import load_prepared
from basket_hedging.volgan import VolGANModel
from basket_hedging.learned_model import ConditionalGaussianModel
from basket_hedging.engine import Engine
from basket_hedging.risk import UnnettedLiabilityStressMargin
from basket_hedging.scenario_hedge import ScenarioHedge
from basket_hedging.optimization import solve_hedge
from basket_hedging.policies import UnhedgedPolicy,ScheduledDeltaPolicy
from basket_hedging.valuation import StressScenario,MODEL_VERSION
from basket_hedging.surfaces import PAPER_GRID
from basket_hedging.replay import jsonable,Case
from basket_hedging.reporting import write_reports

def write(name,obj): (OUT/name).write_text(json.dumps(jsonable(obj),indent=2,allow_nan=False),encoding='utf-8')
def hashed(p): return sha256(p.read_bytes()).hexdigest()
paths=sorted((ROOT/'src/basket_hedging').glob('*.py'))+sorted((ROOT/'reports/historical').rglob('*.*'))
paths=[p for p in paths if p.is_file()]
prior={str(p):hashed(p) for p in paths}
write('frozen_hashes_before.json',prior)
summary=json.loads((ROOT/'reports/historical/benchmark/summary.json').read_text())
config=h.HistoricalConfig(**summary['config'])
arrays,manifest=load_prepared(Path.home()/'.cache/basket-hedging/prepared-etf-v1-actions')
mean,cov,correlation=h.training_dynamics(arrays)
arrays=h.expand_calendar(arrays,manifest)
start=int(np.searchsorted(arrays['dates'],h.TEST_START))
stocks=tuple(map(str,arrays['symbols']))
volgan=VolGANModel.load(ROOT/'reports/historical/models/volgan.json')
gaussian=ConditionalGaussianModel.load(ROOT/'reports/historical/models/gaussian.json')
valuator=hh.SurfaceBasketValuator(stocks,correlation,h.MODEL_AVAILABLE,samples=config.pricing_samples,pricing_seed=config.pricing_seed,
    stress_scenarios=(StressScenario('unchanged'),StressScenario('spot_down_20',.8),StressScenario('spot_up_20',1.2),StressScenario('vol_up_25',1,1.25)))
instruments,opening,rules,events=h.episode(arrays,start,config,valuator)
weights={key:config.half_spread*float(arrays['close'][start,j]) for j,key in enumerate(stocks)}
captures={}; current_name=None; capturing=True
original_reprice=h.reprice_basket_scenarios

def observed_reprice(view,fixed_positions,hedge_ids,**kwargs):
    if capturing:
        captures[(current_name,view.event_id)]=dict(view=view,fixed=dict(fixed_positions),hedge_ids=hedge_ids,
                  kwargs={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in kwargs.items()})
    return original_reprice(view,fixed_positions,hedge_ids,**kwargs)
h.reprice_basket_scenarios=observed_reprice

def policy_for(name):
    if name=='unhedged': return UnhedgedPolicy()
    if name=='delta': return ScheduledDeltaPolicy()
    provider=h.HistoricalScenarios(name,valuator,config,mean=mean,covariance=cov,model=gaussian if name=='gaussian' else volgan)
    p=ScenarioHedge(provider,stocks,weights,alpha=config.alpha);p.name=name
    return p

results={}; policies={}; rows=[]
frozenrows=list(csv.DictReader((ROOT/'reports/historical/benchmark/episodes.csv').open()))
for name in h.POLICIES:
    current_name=name
    policy=policy_for(name);policies[name]=policy
    result=Engine(instruments,opening,rules,UnnettedLiabilityStressMargin()).run(iter(events),policy)
    results[name]=result
    c=result.ledger.components()
    row=dict(episode=str(arrays['dates'][start]),policy=name,status=result.status,reason=result.reason,
        hedge_error=-(c['liability_pnl']+c['hedge_pnl']),net_pnl=c['net_pnl'],turnover=c['stock_turnover'],trading_costs=c['trading_costs'],fills=len(result.fills),
        rejected_orders=sum(o.outcome=='rejected' for o in result.order_outcomes),skipped_decisions=sum(o['status']=='skipped' for o in result.decisions),
        dividend_cashflow=sum(p.cash_change for p in result.ledger.postings if p.kind=='dividend'),borrow_charges=c['borrow_charges'])
    expected=next(r for r in frozenrows if r['policy']==name and r['episode']==row['episode'])
    mismatch={k:(v,expected[k]) for k,v in row.items() if (v!=float(expected[k]) if isinstance(v,(float,int)) else v!=expected[k])}
    assert not mismatch,mismatch
    rows.append(row)
    print(name,result.status,row['net_pnl'],row['fills'],flush=True)
case=Case(instruments,opening,rules,ROOT/'reports/historical/benchmark/representative.jsonl',{'audit':'fresh first fixed episode; frozen input'})
write_reports(list(results.values()),OUT/'reports',case=case)
# Compare all generated event/order/ledger CSVs with frozen first-episode files.
report_checks=[]
for name in h.POLICIES:
    for file in sorted((OUT/'reports'/name).glob('*.csv')):
        frozen=ROOT/'reports/historical/benchmark/representative'/name/file.name
        assert frozen.exists()
        a=list(csv.DictReader(file.open())); b=list(csv.DictReader(frozen.open()))
        assert a==b,(name,file.name)
        report_checks.append(dict(policy=name,file=file.name,rows=len(a),exact_row_match=True))
write('fresh_results.json',dict(rows=rows,all_frozen_row_fields_equal=True,csv_checks=report_checks))
# Observe weekday and weekend scenario transformation without changing generated inputs.
clocks=[]
for name in ('volgan','gaussian','parametric'):
    for event_id in ('2020-01-02/close','2020-01-03/close'):
        record=captures[(name,event_id)];view=record['view'];kw=record['kwargs'];state=view.valuation.metadata['historical_state']
        days=(kw['horizon']-view.time).total_seconds()/86400
        spots=np.array([view.valuation.marks[k] for k in stocks]);rng=np.random.default_rng(kw['seed'])
        if name=='volgan':
            independent_future,independent_surface=volgan.sample(np.asarray(state['volgan_features']),spots,rng.normal(size=(config.scenarios,volgan.config.noise_dim)))
            assert np.array_equal(independent_future,kw['future_spots']) and np.array_equal(independent_surface,kw['future_surfaces'])
            surface_rule='generated, unrescaled one trading-session transition'
            scaling_error=0.
        else:
            z=rng.normal(size=(config.scenarios,len(stocks)))
            if name=='gaussian':
                feature=np.asarray(state['gaussian_features']);mu1,fac1=gaussian._parameters(feature,1.)
                expected=spots*np.exp(days*mu1+np.sqrt(days)*z@fac1.T)
                actualmean,actualfac=gaussian._parameters(feature,days)
                assert np.allclose(actualmean,days*mu1,rtol=0,atol=1e-16)
                assert np.allclose(actualfac,np.sqrt(days)*fac1,rtol=0,atol=1e-16)
            else: expected=spots*np.exp(days*mean+np.sqrt(days)*z@np.linalg.cholesky(cov).T)
            scaling_error=float(np.max(abs(expected-kw['future_spots'])))
            assert scaling_error<1e-10
            assert np.array_equal(kw['future_surfaces'],np.broadcast_to(np.asarray(state['iv']),kw['future_surfaces'].shape))
            surface_rule='current surface carried forward unchanged'
        clocks.append(dict(policy=name,decision=view.time,eligible=view.rules['earliest_execution'],endpoint=kw['horizon'],calendar_days=days,
                           execution_delay_hours=(view.rules['earliest_execution']-view.time).total_seconds()/3600,
                           surface_rule=surface_rule,independent_scaling_max_error=scaling_error))
write('clock_transforms.json',clocks)
# Sensitivity experiment changes only hypothetical generated surfaces. No trades.
record=captures[('volgan','2020-01-02/close')];view=record['view'];kw=record['kwargs'];base=original_reprice(view,record['fixed'],record['hedge_ids'],**kw)
perturbed_kw=dict(kw,future_surfaces=kw['future_surfaces']*1.1)
changed=original_reprice(view,record['fixed'],record['hedge_ids'],**perturbed_kw)
assert np.array_equal(base.delta_h,changed.delta_h)
assert np.max(abs(base.delta_v-changed.delta_v))>1
priorholding=[view.positions.get(k,0) for k in stocks]
base_solution=solve_hedge(base.delta_v,base.delta_h,priorholding,[weights[k] for k in stocks],alpha=config.alpha,gross_scale=policies['volgan'].gross_scale)
changed_solution=solve_hedge(changed.delta_v,changed.delta_h,priorholding,[weights[k] for k in stocks],alpha=config.alpha,gross_scale=policies['volgan'].gross_scale)
assert np.max(abs(base_solution.holdings-changed_solution.holdings))>1e-6
# Independently price first generated state: direct ATM interpolation, seed,
# antithetic normals, correlated GBM terminal shares, arithmetic basket put.
basket=instruments['B'];T=(basket.expiry-kw['horizon']).total_seconds()/(365*86400)
terms=np.asarray(PAPER_GRID.maturities);j=int(np.searchsorted(terms,T)-1);w=(T-terms[j])/(terms[j+1]-terms[j])
mindex=list(PAPER_GRID.moneyness).index(1.)
vols=(1-w)*kw['future_surfaces'][0,:,mindex,j]+w*kw['future_surfaces'][0,:,mindex,j+1]
material=json.dumps({'seed':config.pricing_seed,'time_utc':kw['horizon'].astimezone(timezone.utc).isoformat(timespec='microseconds'),'model':MODEL_VERSION},sort_keys=True,separators=(',',':')).encode()
seed=int.from_bytes(sha256(material).digest()[:16],'big')
z=np.random.Generator(np.random.PCG64(seed)).standard_normal(((config.pricing_samples+1)//2,3))
z=np.concatenate((z,-z),axis=0)[:config.pricing_samples]
eig,U=np.linalg.eigh(np.asarray(correlation));normal=z@(U@np.diag(np.sqrt(eig))).T
coeff=np.array([dict(basket.coefficients)[k] for k in stocks])
prices=[]
for sample in normal:
    terminal=kw['future_spots'][0]*np.exp(-.5*vols*vols*T+vols*math.sqrt(T)*sample)
    prices.append(max(basket.strike-float(np.dot(coeff,terminal)),0.))
scalar_price=math.fsum(prices)/len(prices)
expected_dv=basket.multiplier*(scalar_price-view.valuation.marks['B'])
assert abs(expected_dv-base.delta_v[0])<1e-10
write('surface_sensitivity.json',dict(note='Audit-only sensitivity, not a new benchmark run or accepted holdings. Model and accounting settings unchanged.',
    same_spots=True,same_noise_seed=True,same_horizon=True,same_delta_h=True,surface_scale=1.1,
    maximum_response_change=float(np.max(abs(base.delta_v-changed.delta_v))),
    original_optimizer_holdings=base_solution.holdings.tolist(),perturbed_optimizer_holdings=changed_solution.holdings.tolist(),
    independent_scenario0_unit_put_price=scalar_price,independent_scenario0_delta_v=expected_dv,actual_delta_v=float(base.delta_v[0]),
    unit_delta_v_error=abs(expected_dv-base.delta_v[0])))
np.savez(OUT/'first_decision_surface_probe.npz',future_spots=kw['future_spots'],future_surfaces=kw['future_surfaces'],delta_h=base.delta_h,base_delta_v=base.delta_v,perturbed_delta_v=changed.delta_v)
# Mutate future values only, preserve dates and contract terms; no file rewritten.
cutoff=datetime.fromisoformat('2020-01-09T21:00:00+00:00')
mutated={k:v.copy() for k,v in arrays.items()};future=mutated['dates']>'2020-01-09'
mutated['open'][future]*=1.15;mutated['close'][future]*=.92;mutated['iv'][future]*=1.2;mutated['dividends'][future]+=.123
_,_,_,mutated_events=h.episode(mutated,start,config,valuator)
original_prefix=[jsonable(e) for e in events if e.time<=cutoff]
mutated_prefix=[jsonable(e) for e in mutated_events if e.time<=cutoff]
assert original_prefix==mutated_prefix
capturing=False
mutation_checks=[]
def prefix(result):
    return dict(postings=[jsonable(p) for p in result.ledger.postings if p.time<=cutoff],
        order_outcomes=[jsonable(o) for o in result.order_outcomes if o.time<=cutoff],
        snapshots=[jsonable(s) for s in result.snapshots if s.time<=cutoff],
        decisions=[jsonable(d) for d in result.decisions if d['time']<=cutoff],
        valuations=[jsonable(v) for v in result.valuations if v['as_of']<=cutoff])
for name in ('delta','volgan'):
    current_name=name
    changed_result=Engine(instruments,opening,rules,UnnettedLiabilityStressMargin()).run(iter(mutated_events),policy_for(name))
    a,b=prefix(results[name]),prefix(changed_result)
    assert a==b,(name,[k for k in a if a[k]!=b[k]])
    mutation_checks.append(dict(policy=name,cutoff=cutoff,exact_before_cutoff=True,counts={k:len(v) for k,v in a.items()}))
    write(f'{name}_mutation_prefix.json',a)
write('future_mutation.json',dict(events_exactly_equal_before_cutoff=True,event_count=len(original_prefix),checks=mutation_checks,
    mutation='Only dates after Jan09: opens*1.15, closes*.92, IV*1.2, all constituent dividends+.123. Same clocks, terms, model, prior state.'))
after={str(p):hashed(p) for p in paths};assert prior==after
write('frozen_hashes_after.json',after)
print('DONE:',len(rows),'complete fixed-policy cases;',len(report_checks),'CSV matches;',len(after),'unchanged hashes',flush=True)
