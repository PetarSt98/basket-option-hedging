"""Independent population, moving-block uncertainty and figure-input audit."""
from pathlib import Path
from hashlib import sha256
from collections import Counter
import csv,json,math,statistics
import numpy as np
ROOT=Path(r'C:/Users/petar/Desktop/basket-option-hedging');OUT=Path(__file__).resolve().parent;REPORT=ROOT/'reports/historical_solver_corrected_v1'
POLICIES=('unhedged','delta','parametric','gaussian','volgan');checks=0

def load(path):
 rows=list(csv.DictReader(path.open()))
 for row in rows:
  for key in ('hedge_error','net_pnl','turnover','trading_costs','fills'):
   row[key]=float(row[key]) if row.get(key) else None
 return {(r['episode'],r['policy']):r for r in rows}

def avg(values):return math.fsum(values)/len(values)
def check(a,b,label):
 global checks
 assert math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=2e-7),(label,a,b)
 checks+=1

def statistics_for(rows):
 e=[r['hedge_error'] for r in rows];p=[r['net_pnl'] for r in rows];tail=math.ceil(.05*len(rows))
 return dict(completed_episodes=len(rows),scheduled_episodes=71,mean_hedge_error=avg(e),mean_absolute_hedge_error=avg(list(map(abs,e))),
  hedge_error_sd=statistics.stdev(e),hedge_error_rmse=math.sqrt(avg([x*x for x in e])),mean_net_pnl=avg(p),net_pnl_sd=statistics.stdev(p),
  cvar95=avg(sorted([-x for x in p])[-tail:]),cvar95_tail_episodes=tail,mean_turnover=avg([r['turnover'] for r in rows]),
  mean_transaction_costs=avg([r['trading_costs'] for r in rows]),mean_fills=avg([r['fills'] for r in rows]))

new=load(REPORT/'episodes.csv');old=load(ROOT/'reports/historical/benchmark/episodes.csv');episodes=sorted({k[0] for k in new});n=len(episodes)
common=[e for e in episodes if all(new[e,p]['status']=='COMPLETED' for p in POLICIES)]
frozen=[e for e in episodes if all(old[e,p]['status']=='COMPLETED' for p in POLICIES)]
raw=json.loads((OUT/'independent_raw_checks.json').read_text());sufficient=raw['sufficient_history_common']
rel=json.loads((REPORT/'reliability_populations.json').read_text());plan=json.loads((REPORT/'plan.json').read_text())
assert rel['common_completed_episode_ids']==common
assert rel['frozen_common_episode_ids']==frozen
assert rel['common_sufficient_history_episode_ids']==sufficient
assert rel['sufficient_history_episode_ids']==raw['sufficient_history_all']
populations={'common_completed':(new,common),'common_sufficient_history':(new,sufficient),'frozen_common_original':(old,frozen),
 'frozen_common_matched_original':(old,frozen),'frozen_common_matched_corrected':(new,frozen)}
metrics=[]
for group,reportedrows in rel['metrics'].items():
 for reported in reportedrows:
  policy=reported['policy']
  if group=='own_completed':source=new;members=[e for e in episodes if new[e,policy]['status']=='COMPLETED']
  else:source,members=populations[group]
  assert reported['episode_ids']==members
  values=statistics_for([source[e,policy] for e in members])
  for k,v in values.items():check(v,reported[k],(group,policy,k))
  metrics.append(dict(population=group,policy=policy,**values))
flat={(r['population'],r['policy']):r for r in csv.DictReader((REPORT/'population_metrics.csv').open())}
for r in metrics:
 for k,v in r.items():
  if k not in ('population','policy'):check(v,flat[r['population'],r['policy']][k],('population_csv',r['population'],r['policy'],k))
# Generate identical preregistered full-calendar block indices independently.
seed=plan['bootstrap']['seed'];B=plan['bootstrap']['repetitions'];length=plan['bootstrap']['block_length'];block_count=math.ceil(n/length)
assert (seed,B,length)==(20261004,5000,3)
rng=np.random.Generator(np.random.PCG64(seed));indices=[]
for _ in range(B):
 starts=rng.integers(0,n-length+1,size=block_count)
 indices.append([int(start+offset) for start in starts for offset in range(length)][:n])
uncertainty={(r['population'],r['policy'],r['reference'],r['metric']):r for r in csv.DictReader((REPORT/'uncertainty.csv').open())}
# Percentile interpolation explicitly uses order-statistic positions, no report helper.
def percentile(values,prob):
 ordered=sorted(values);index=(len(ordered)-1)*prob;lo=math.floor(index);hi=math.ceil(index);w=index-lo
 return ordered[lo]*(1-w)+ordered[hi]*w
results=[]
for group,(source,members) in populations.items():
 if group=='frozen_common_original':continue
 member_set=set(members);distribution={'mean_absolute_error_difference':[],'hedge_error_sd_difference':[],'cvar95_difference':[]};counts=[]
 for draw in indices:
  sampled=[episodes[j] for j in draw if episodes[j] in member_set];counts.append(len(sampled))
  p=[source[e,'volgan'] for e in sampled];q=[source[e,'delta'] for e in sampled]
  pe=[x['hedge_error'] for x in p];qe=[x['hedge_error'] for x in q];tail=math.ceil(.05*len(sampled))
  distribution['mean_absolute_error_difference'].append(avg([abs(a)-abs(b) for a,b in zip(pe,qe)]))
  distribution['hedge_error_sd_difference'].append(statistics.stdev(pe)-statistics.stdev(qe))
  distribution['cvar95_difference'].append(avg(sorted([-x['net_pnl'] for x in p])[-tail:])-avg(sorted([-x['net_pnl'] for x in q])[-tail:]))
 for metric,values in distribution.items():
  row=uncertainty[group,'volgan','delta',metric];lower,upper=percentile(values,.025),percentile(values,.975)
  check(lower,row['lower_95'],(group,metric,'lower'));check(upper,row['upper_95'],(group,metric,'upper'))
  assert int(row['valid_replicates'])==B==int(row['requested_replicates'])
  assert int(row['minimum_resampled_completed'])==min(counts) and int(row['maximum_resampled_completed'])==max(counts)
  assert int(row['observed_completed'])==len(members) and int(row['observed_cvar95_tail'])==math.ceil(.05*len(members))
  results.append(dict(population=group,policy='volgan',reference='delta',metric=metric,lower_95=lower,upper_95=upper,
                      draws=B,min_resampled_count=min(counts),max_resampled_count=max(counts)))
# Each plotted coordinate comes from saved raw rows or independent stats.
fig=json.loads((REPORT/'figures/figure_data.json').read_text())
assert fig['scheduled_episode_ids']==episodes and fig['common_completed_episode_ids']==common
assert fig['excluded_episode_ids']==[e for e in episodes if e not in common]
for name,digest in fig['input_sha256'].items():assert sha256((REPORT/name).read_bytes()).hexdigest()==digest
for p in POLICIES:
 assert fig['distribution'][p]==[new[e,p]['hedge_error'] for e in common]
 assert fig['time']['values'][p]==[new[e,p]['hedge_error'] if e in common else None for e in episodes]
assert fig['time']['dates']==episodes
for point in fig['risk_cost']:
 policy=point['policy'];stats=statistics_for([new[e,policy] for e in common])
 for k in ('mean_turnover','mean_transaction_costs','hedge_error_sd'):check(stats[k],point[k],('figure/risk_cost',policy,k))
assert fig['representative']['episode']==episodes[0]=='2020-01-02'
assert fig['representative']['rule']=='first scheduled test episode, selected before results'
representative=list(csv.DictReader((REPORT/'representative.csv').open()))
assert len(representative)==len(fig['representative']['rows'])
for csvrow,point in zip(representative,fig['representative']['rows']):
 assert csvrow['policy']==point['policy'] and csvrow['time']==point['time']
 for k in ('basket','liability_value','net_pnl','SPY','QQQ','IWM'):check(csvrow[k],point[k],('figure/representative',k))
(OUT/'independent_population_bootstrap_figures.json').write_text(json.dumps(dict(numeric_checks=checks,population_metrics=metrics,
 bootstrap_comparisons=results,bootstrap_index_sha256=sha256(np.asarray(indices,dtype=np.int64).tobytes()).hexdigest(),
 bootstrap_first_three_indices=indices[:3],figure_values_verified=True,figure_source_hashes_verified=True,
 note='Statistics use stdlib math/statistics, explicit scalar block resampling and linear empirical percentiles. No production reporting helper imported.'),indent=2),encoding='utf-8')
print('PASSED',checks,'population/figure/interval checks;',len(results),'bootstrap intervals across4populations.')
