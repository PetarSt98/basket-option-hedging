"""Recompute corrected historical benchmark independently, without engine imports."""
from pathlib import Path
from hashlib import sha256
from collections import Counter
import csv,json,math,statistics,argparse

ROOT=Path(r'C:/Users/petar/Desktop/basket-option-hedging')
OUT=Path(__file__).resolve().parent
POLICIES=('unhedged','delta','parametric','gaussian','volgan')
NUMERIC=('hedge_error','net_pnl','turnover','trading_costs','fills','rejected_orders','skipped_decisions','dividend_cashflow','borrow_charges')
checks=[]

def rows(path):
    values=list(csv.DictReader(path.open(encoding='utf-8')))
    for r in values:
        for k in NUMERIC:
            if k in r:r[k]=float(r[k]) if r[k] else None
    return values

def mean(values):return math.fsum(values)/len(values)
def check(a,b,label):
    if a is None:assert b is None or b=='',(label,a,b)
    else:assert math.isclose(a,float(b),rel_tol=1e-12,abs_tol=2e-7),(label,a,b)
    checks.append(label)

def stats(selected):
    errors=[r['hedge_error'] for r in selected];pnl=[r['net_pnl'] for r in selected];tail=math.ceil(.05*len(pnl))
    return dict(hedge_error_sd=statistics.stdev(errors),hedge_error_rmse=math.sqrt(mean([x*x for x in errors])),
        net_pnl_sd=statistics.stdev(pnl),cvar95=mean(sorted([-x for x in pnl])[-tail:]),cvar95_tail_episodes=tail,
        mean_turnover=mean([r['turnover'] for r in selected]),mean_transaction_costs=mean([r['trading_costs'] for r in selected]),
        mean_fills=mean([r['fills'] for r in selected]),mean_hedge_error=mean(errors),mean_net_pnl=mean(pnl))

parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path,default=ROOT/'reports/historical_solver_corrected_v1');args=parser.parse_args()
report=args.report
current=rows(report/'episodes.csv');old=rows(ROOT/'reports/historical/benchmark/episodes.csv')
keys={(r['episode'],r['policy']) for r in current};assert len(keys)==len(current)==355
ids=sorted({r['episode'] for r in current});assert len(ids)==71
assert all({r['policy'] for r in current if r['episode']==e}==set(POLICIES) for e in ids)
bykey={(r['episode'],r['policy']):r for r in current};oldbykey={(r['episode'],r['policy']):r for r in old}
common=[e for e in ids if all(bykey[e,p]['status']=='COMPLETED' for p in POLICIES)]
gapsets={p:{r['episode'] for r in current if r['policy']==p and r['status']=='STOP_DATA_GAP'} for p in POLICIES}
assert all(s==gapsets['unhedged'] for s in gapsets.values())
assert len(gapsets['unhedged'])==5
completedsets={p:{r['episode'] for r in current if r['policy']==p and r['status']=='COMPLETED'} for p in POLICIES}
summary=json.loads((report/'summary.json').read_text());original_summary=json.loads((ROOT/'reports/historical/benchmark/summary.json').read_text())
for field in ('config','split','model_hashes','data_manifest','frozen_pricing_correlation','parametric_daily_log_mean','parametric_daily_covariance'):
    assert summary[field]==original_summary[field],field
metrics={r['policy']:r for r in summary['metrics']}
metriccsv={r['policy']:r for r in csv.DictReader((report/'metrics.csv').open())}
allstats=[]
for p in POLICIES:
    allrows=[r for r in current if r['policy']==p];selected=[bykey[e,p] for e in common];own=[r for r in allrows if r['status']=='COMPLETED']
    calculated=stats(selected);ownstats=stats(own)
    counts=dict(episodes=len(allrows),matched_completed=len(selected),completion_rate=len(own)/len(allrows),
                margin_breaches=sum(r['status']=='STOP_WITH_BREACH' for r in allrows),
                data_gaps=sum(r['status']=='STOP_DATA_GAP' for r in allrows),
                optimizer_failures=sum(r['status']=='FAILED_OPTIMIZER' for r in allrows),
                total_turnover_all_runs=math.fsum(r['turnover'] for r in allrows),
                total_trading_costs_all_runs=math.fsum(r['trading_costs'] for r in allrows),total_fills_all_runs=sum(r['fills'] for r in allrows),
                rejected_orders_all_runs=sum(r['rejected_orders'] for r in allrows),skipped_decisions_all_runs=sum(r['skipped_decisions'] for r in allrows))
    for k,v in dict(calculated,**counts).items():
        if k in metrics[p]:check(v,metrics[p][k],f'summary/{p}/{k}')
        if k in metriccsv[p]:check(v,metriccsv[p][k],f'metrics.csv/{p}/{k}')
    for r in allrows:
        if r['status']!='COMPLETED':assert r['hedge_error'] is None and r['net_pnl'] is None
    allstats.append(dict(policy=p,status_counts=dict(Counter(r['status'] for r in allrows)),common_stats=calculated,own_completed_stats=ownstats,**counts))
# Successful old results remain exact; previously failed episodes are not compared at unequal horizon.
unchanged=0;recovered=[]
for r in current:
    oldrow=oldbykey[r['episode'],r['policy']]
    if oldrow['status']=='FAILED_OPTIMIZER':
        recovered.append(dict(episode=r['episode'],policy=r['policy'],old_status=oldrow['status'],new_status=r['status'],new_terminal_pnl=r['net_pnl'],new_terminal_error=r['hedge_error']))
        continue
    for key in oldrow:
        if key=='seconds':continue
        assert r[key]==oldrow[key],(r['episode'],r['policy'],key,r[key],oldrow[key])
    unchanged+=1
assert unchanged==351 and len(recovered)==4
# Paired metrics use the same complete population; differences may include nonlinear SD/RMSE differences.
paired=[]
if (report/'paired_differences.csv').exists():
    for row in csv.DictReader((report/'paired_differences.csv').open()):
        p,q=row['policy'],row['reference'];a=[bykey[e,p] for e in common];b=[bykey[e,q] for e in common];sa,sb=stats(a),stats(b)
        values=dict(matched_completed=len(common))
        for outkey,inputkey in [('mean_hedge_error_difference','hedge_error'),('mean_net_pnl_difference','net_pnl'),('mean_turnover_difference','turnover'),('mean_trading_cost_difference','trading_costs'),('mean_fills_difference','fills')]:
            values[outkey]=mean([x[inputkey]-y[inputkey] for x,y in zip(a,b)])
        values['mean_absolute_error_difference']=mean([abs(x['hedge_error'])-abs(y['hedge_error']) for x,y in zip(a,b)])
        values['mean_squared_error_difference']=mean([x['hedge_error']**2-y['hedge_error']**2 for x,y in zip(a,b)])
        for k in ('hedge_error_sd','hedge_error_rmse','net_pnl_sd','cvar95'):values[k+'_difference']=sa[k]-sb[k]
        for k,v in values.items():check(v,row[k],f'paired/{p}/{q}/{k}')
        paired.append(dict(policy=p,reference=q,**values))
# Original historical results/models have not been modified during this audit.
protected=json.loads((OUT/'frozen_original_hashes.json').read_text())
assert all(sha256(Path(path).read_bytes()).hexdigest()==expected for path,expected in protected.items())
result=dict(scheduled_episodes=len(ids),rows=len(current),common_completed=len(common),common_ids=common,shared_data_gap_ids=sorted(gapsets['unhedged']),
            statistics=allstats,unchanged_old_rows=unchanged,recovered_failed_episodes=recovered,paired=paired,
            numeric_checks=len(checks),frozen_original_hashes_unchanged=len(protected),configuration_models_inputs_unchanged=True)
(OUT/'independent_metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
