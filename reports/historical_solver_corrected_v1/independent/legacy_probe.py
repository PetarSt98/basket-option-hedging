from pathlib import Path
import json,csv,hashlib
ROOT=Path(r'C:/Users/petar/Desktop/basket-option-hedging')
OUT=Path(r'C:/Users/petar/AppData/Local/Temp/historical-corrected-audit-0zslhkz2')
results=[]
for case in ('deterministic','synthetic','gap','missing_exit','rejected_exit'):
 for policy in ('unhedged','delta'):
  old=ROOT/'reports'/case/policy;new=ROOT/'reports/historical_solver_corrected_v1/legacy'/case/policy
  checks=[]
  for f in new.glob('*.csv'):
   assert (old/f.name).read_bytes()==f.read_bytes(),(case,policy,f.name)
   checks.append(f.name)
  a=json.loads((old/'summary.json').read_text());b=json.loads((new/'summary.json').read_text())
  added_timing_fields=[]
  if a.get('decisions')!=b.get('decisions'):
   assert len(a['decisions'])==len(b['decisions'])
   for original,fresh in zip(a['decisions'],b['decisions']):
    assert all(fresh[k]==v for k,v in original.items()),(case,policy,'decision')
    extra=set(fresh)-set(original)
    assert extra <= {'earliest_execution','next_rebalance_time'},extra
    added_timing_fields.extend(extra)
   b['decisions']=a['decisions']
  diffs=[k for k in a if a[k]!=b.get(k)]
  assert not diffs,(case,policy,diffs)
  results.append(dict(case=case,policy=policy,status=a['status'],marked_pnl=a['current']['marked_pnl'],identical_csvs=checks,summary_original_fields_exact=True,added_decision_metadata=sorted(set(added_timing_fields))))
(OUT/'legacy_exact_comparison.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print('10 summaries preserve every original field; only extra timing metadata. ',sum(len(r['identical_csvs']) for r in results),'CSVs byte-identical.')
