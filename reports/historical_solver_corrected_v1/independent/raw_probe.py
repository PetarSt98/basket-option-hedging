"""Independent corrected benchmark raw-fill, accounting, and history audit."""
from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
from hashlib import sha256
import csv,json,math
import numpy as np
ROOT=Path(r'C:/Users/petar/Desktop/basket-option-hedging');OUT=Path(__file__).resolve().parent
REPORT=ROOT/'reports/historical_solver_corrected_v1'
PREP=Path.home()/'.cache/basket-hedging/prepared-etf-v1-actions'
rows=list(csv.DictReader((REPORT/'episodes.csv').open()));bykey={(r['episode'],r['policy']):r for r in rows}
config=json.loads((REPORT/'summary.json').read_text())['config']
with np.load(PREP/'surfaces.npz',allow_pickle=False) as z: data={k:z[k].copy() for k in z.files}
manifest=json.loads((PREP/'manifest.json').read_text())
calendar=sorted(set(map(str,data['dates']))|{d for gap in manifest['common_date_gaps'] for d in gap['missing_dates']})
date_index={str(d):i for i,d in enumerate(data['dates'])};cal_index={d:i for i,d in enumerate(calendar)}
POLICIES=('unhedged','delta','parametric','gaussian','volgan')

def time(day,opening=False):return datetime.fromisoformat(day).replace(hour=9 if opening else 16,minute=30 if opening else 0,tzinfo=ZoneInfo('America/New_York')).astimezone(timezone.utc)
checks=0
def close(a,b,label):
 global checks
 assert math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=2e-7),(label,a,b)
 checks+=1

def conditioning(j):
 if j<21 or not all(d in date_index and bool(data['consecutive'][date_index[d]]) for d in calendar[j-20:j+1]):return False
 d=calendar[j]
 return d in date_index and bool(data['valid_surface'][date_index[d]].all()) and bool(np.isfinite(data['close'][date_index[d]]).all())

# Schedule reconstructed from fixed raw calendar; no strategy or production history helper.
expected_schedules={}
first=cal_index['2020-01-02'];starts=list(range(first,len(calendar)-config['holding_sessions']-1,config['holding_sessions']+1))
assert len(starts)==71
for start in starts:
 s=[]
 for j in range(start,start+21):
  terminal=j==start+20
  s.append(dict(event_id=calendar[j]+'/close',time=time(calendar[j]).isoformat(),terminal=terminal,conditioning_available=True if terminal else conditioning(j)))
 expected_schedules[calendar[start]]=s
assert set(expected_schedules)=={r['episode'] for r in rows}
decisions=json.loads((REPORT/'decisions.json').read_text());assert len(decisions)==355
bydecision={};availability=set();skip_totals=Counter();notreached=Counter();historybad=[]
for record in decisions:
 key=(record['episode'],record['policy']);assert key not in bydecision;bydecision[key]=record
 expected=expected_schedules[record['episode']]
 assert len(record['scheduled_records'])==len(expected)==21
 for actual,wanted in zip(record['scheduled_records'],expected):
  for k,v in wanted.items():assert actual[k]==v,(key,k,actual[k],v)
 sufficient=all(e['conditioning_available'] for e in expected)
 assert sufficient==record['sufficient_history_by_data']
 if sufficient:availability.add(record['episode'])
 scheduled={e['event_id']:e for e in expected};actual=[d for d in record['records'] if d['event_id'] in scheduled]
 reached={d['event_id'] for d in actual};assert len(reached)==len(actual)
 counts=Counter(d['status'] for d in actual)
 assert record['decisions_made']==counts['targets']
 assert record['decisions_skipped']==counts['skipped']
 assert record['decisions_no_target']==counts['no_decision']
 assert record['decisions_not_reached']==21-len(actual)
 assert record['decisions_made']+record['decisions_skipped']+record['decisions_no_target']+record['decisions_not_reached']==21
 assert counts['skipped']==int(bykey[key]['skipped_decisions'])
 for d in actual:
  if d['status']=='skipped':assert not scheduled[d['event_id']]['conditioning_available'],(key,d)
  if record['policy'] in ('parametric','gaussian','volgan') and not scheduled[d['event_id']]['conditioning_available']:
   assert d['status']=='skipped',(key,d)
 skip_totals[record['policy']]+=counts['skipped'];notreached[record['policy']]+=21-len(actual)
 if counts['skipped']:historybad.append(dict(episode=key[0],policy=key[1],skips=counts['skipped'],status=record['status']))

# Fill outcomes reconstructed from recorded executable prices and observed raw opens.
fills=list(csv.DictReader((REPORT/'fills.csv').open()));fgroups=defaultdict(list)
for fill in fills:
 assert fill['kind']=='fill' and fill['instrument_id'] in ('SPY','QQQ','IWM')
 fgroups[fill['episode'],fill['policy']].append(fill)
orders=json.loads((REPORT/'order_outcomes.json').read_text());ogroups=defaultdict(list)
for order in orders:ogroups[order['episode'],order['policy']].append(order)
for key,row in bykey.items():
 fs=fgroups[key];assert len(fs)==int(row['fills'])
 turnover=math.fsum(abs(float(f['quantity'])*float(f['price'])) for f in fs)
 spread=math.fsum(float(f['quantity'])*(float(f['price'])-float(f['reference_mark'])) for f in fs)
 fees=math.fsum(float(f['fee']) for f in fs)
 close(turnover,row['turnover'],(key,'turnover'));close(spread+fees,row['trading_costs'],(key,'costs'))
 submitted={};capacity=defaultdict(float);seenfills={f['id']:f for f in fs}
 for o in ogroups[key]:
  if o['outcome']=='submitted':submitted[o['order_id']]=o
  if o['outcome']!='filled':continue
  original=submitted[o['order_id']];t=datetime.fromisoformat(o['time']);ts=datetime.fromisoformat(original['time']);assert t>ts
  decision=next(d for d in bydecision[key]['records'] if d['event_id']==original['event_id'])
  assert t>=datetime.fromisoformat(decision['earliest_execution'])
  sidx=cal_index[original['event_id'].split('/')[0]];epi=cal_index[key[0]]
  deadline=time(calendar[sidx+1],opening=(sidx==epi+20))
  assert t<=deadline if sidx==epi+20 else t<deadline
  date=o['event_id'].split('/')[0];assert t==time(date,opening=True)
  j=list(data['symbols']).index(o['instrument_id']);opening=float(data['open'][date_index[date],j]);dq=float(o['quantity'])
  close(o['price'],opening*(1+config['half_spread'] if dq>0 else 1-config['half_spread']),(key,'observed open plus spread'))
  assert o['quote_id']=='assumed-execution/'+o['event_id']+'/'+o['instrument_id']
  f=seenfills['fill/'+o['order_id']];close(f['cash_change'],-dq*float(o['price']),'fillcash')
  close(f['reference_mark'],opening,'fillmark');close(f['fee'],o['fee'],'fillfee')
  side='buy' if dq>0 else 'sell';capkey=(o['quote_id'],side);capacity[capkey]+=abs(dq)
  assert capacity[capkey]<=config['quote_size']+1e-8
 assert sum(o['outcome']=='filled' for o in ogroups[key])==len(fs)
 assert sum(o['outcome']=='rejected' for o in ogroups[key])==int(row['rejected_orders'])

components=list(csv.DictReader((REPORT/'accounting_components.csv').open()));seen=set()
for r in components:
 key=(r['episode'],r['policy']);assert key not in seen;seen.add(key);episode=bykey[key]
 for field,ref in [('stock_turnover','turnover'),('trading_costs','trading_costs'),('borrow_charges','borrow_charges')]:close(r[field],episode[ref],(key,field))
 close(float(r['execution_cost'])+float(r['fees']),r['trading_costs'],(key,'fee once'))
 close(float(r['borrow_charges'])-float(r['interest']),r['financing_costs'],(key,'financing'))
 if episode['status']=='COMPLETED':
  assert r['terminal']=='True'
  liability=float(r['liability_pnl']);hedge=float(r['hedge_pnl']);net=float(r['net_pnl'])
  close(-(liability+hedge),episode['hedge_error'],(key,'hedgeerror'))
  close(liability+hedge-float(r['trading_costs'])-float(r['financing_costs']),net,(key,'pnl decomposition'))
  close(net,episode['net_pnl'],(key,'terminalpnl'))
 else:
  assert r['terminal']=='False'
  # Revised exporter deliberately refuses stale valuation-dependent components.
  assert all(r[k]=='' for k in ('net_pnl','liability_pnl','hedge_pnl','reconciliation_error')),('stale valuations exported',key)
for key,row in bykey.items():
 if key not in seen:assert row['originated']=='False' and float(row['fills'])==0
common={e for e in expected_schedules if all(bykey[e,p]['status']=='COMPLETED' for p in POLICIES)}
history_common=sorted(common&availability)
polish=json.loads((REPORT/'solver_diagnostics.json').read_text())
assert all(s['kkt_residual']<=s['solver_tolerance']==1e-9 for s in polish)
assert all(s['max_iterations']==20000 for s in polish)
assert all(s['iterations']<=20000 for s in polish)
assert all(not s['polish_succeeded'] or s['polish_attempted'] for s in polish)
result=dict(independent_numeric_checks=checks,fills_independently_checked=len(fills),accounting_rows=len(components),scheduled_episodes=71,
 scheduled_decisions_per_episode=21,all_schedules_verified_from_prepared_data=True,sufficient_history_all=sorted(availability),
 sufficient_history_common=history_common,sufficient_history_common_count=len(history_common),skipped_decisions_totals=dict(skip_totals),
 decisions_not_reached_totals=dict(notreached),skipped_episode_records=historybad,solver_calls=len(polish),
 polishing_calls=[s for s in polish if s['polish_attempted']],all_solver_optimality_residuals_pass=True)
(OUT/'independent_raw_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print('PASSED',checks,'numeric checks;',len(fills),'fills;',len(history_common),'sufficient-history common episodes')
