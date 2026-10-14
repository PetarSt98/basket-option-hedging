"""Fresh fixed March2020 episode: official IWM ex-date cash before fills."""
from pathlib import Path
from hashlib import sha256
import sys,json,csv,math
import numpy as np
ROOT=Path(r'C:/Users/petar/Desktop/basket-option-hedging');OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from basket_hedging import historical as h
from basket_hedging.etf_data import load_prepared
from basket_hedging.historical_hedge import SurfaceBasketValuator
from basket_hedging.volgan import VolGANModel
from basket_hedging.valuation import StressScenario
from basket_hedging.engine import Engine
from basket_hedging.risk import UnnettedLiabilityStressMargin
from basket_hedging.policies import ScheduledDeltaPolicy
from basket_hedging.scenario_hedge import ScenarioHedge
from basket_hedging.replay import jsonable,Case
from basket_hedging.reporting import write_reports
config=h.HistoricalConfig(**json.loads((ROOT/'reports/historical/benchmark/summary.json').read_text())['config'])
paths=list((ROOT/'src/basket_hedging').glob('*.py'))+[ROOT/'reports/historical/models/volgan.json',ROOT/'data/iwm_corporate_actions/distributions.csv',ROOT/'reports/historical/benchmark/episodes.csv']
before={str(p):sha256(p.read_bytes()).hexdigest() for p in paths}
arrays,manifest=load_prepared(Path.home()/'.cache/basket-hedging/prepared-etf-v1-actions')
mean,cov,correlation=h.training_dynamics(arrays);arrays=h.expand_calendar(arrays,manifest)
stocks=tuple(map(str,arrays['symbols']));start=int(np.where(arrays['dates']=='2020-03-04')[0][0])
valuator=SurfaceBasketValuator(stocks,correlation,h.MODEL_AVAILABLE,samples=config.pricing_samples,pricing_seed=config.pricing_seed,stress_scenarios=(StressScenario('unchanged'),StressScenario('spot_down_20',.8),StressScenario('spot_up_20',1.2),StressScenario('vol_up_25',1,1.25)))
instruments,opening,rules,events=h.episode(arrays,start,config,valuator)
model=VolGANModel.load(ROOT/'reports/historical/models/volgan.json')
frozen=list(csv.DictReader((ROOT/'reports/historical/benchmark/episodes.csv').open()))
action=next(r for r in csv.DictReader((ROOT/'data/iwm_corporate_actions/distributions.csv').open()) if r['ex_date'].startswith('2020-03'))
assert action['ex_date']=='2020-03-25'
per_share=float(action['cash_per_share']);weights={k:config.half_spread*float(arrays['close'][start,j]) for j,k in enumerate(stocks)}
replay=OUT/'march_episode.jsonl';replay.write_text('\n'.join(json.dumps(jsonable(e),allow_nan=False) for e in events)+'\n',encoding='utf-8')
records=[];results=[]
for name in ('delta','volgan'):
    policy=ScheduledDeltaPolicy() if name=='delta' else ScenarioHedge(h.HistoricalScenarios('volgan',valuator,config,model=model),stocks,weights,alpha=config.alpha)
    policy.name=name
    result=Engine(instruments,opening,rules,UnnettedLiabilityStressMargin()).run(iter(events),policy);results.append(result)
    comp=result.ledger.components();expected=next(r for r in frozen if r['episode']=='2020-03-04' and r['policy']==name)
    assert result.status==expected['status']=='COMPLETED'
    assert comp['net_pnl']==float(expected['net_pnl'])
    assert comp['stock_turnover']==float(expected['turnover'])
    assert comp['trading_costs']==float(expected['trading_costs'])
    assert len(result.fills)==int(expected['fills'])
    dividends=[p for p in result.ledger.postings if p.kind=='dividend' and p.instrument_id=='IWM']
    assert len(dividends)==1
    paid=dividends[0];assert paid.time==h.session_time(action['ex_date'],opening=True)
    q=sum(p.quantity for p in result.ledger.postings if p.kind=='fill' and p.instrument_id=='IWM' and p.time<paid.time)
    assert paid.quantity==q and paid.price==per_share and paid.cash_change==q*per_share
    same_event_fills=[p for p in result.ledger.postings if p.kind=='fill' and p.instrument_id=='IWM' and p.time==paid.time]
    i=list(result.ledger.postings).index(paid)
    assert all(list(result.ledger.postings).index(fill)>i for fill in same_event_fills)
    assert not any(p.kind=='dividend' and p.instrument_id=='IWM' and p.time.date().isoformat()==action['payable_date'] for p in result.ledger.postings)
    records.append(dict(policy=name,status=result.status,net_pnl=comp['net_pnl'],all_summary_fields_equal_frozen=True,
        official_action=action,opening_dividend_post=jsonable(paid),independently_summed_pre_ex_holdings=q,
        same_open_later_IWM_fills=jsonable(same_event_fills),no_payable_date_recredit=True,
        total_dividend_cashflow=sum(p.cash_change for p in result.ledger.postings if p.kind=='dividend')))
write_reports(results,OUT/'march_reports',case=Case(instruments,opening,rules,replay,{'audit':'unchanged March2020 historical episode'}))
after={str(p):sha256(p.read_bytes()).hexdigest() for p in paths};assert before==after
(OUT/'ex_dividend_check.json').write_text(json.dumps(dict(unchanged_hashes=after,checks=records),indent=2),encoding='utf-8')
print(json.dumps(records,indent=2))
