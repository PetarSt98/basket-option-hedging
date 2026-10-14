"""Independent arithmetic checks of fresh first-episode reports; no engine helpers."""
import csv,json,math
from pathlib import Path
from datetime import datetime
ROOT=Path(r'C:/Users/petar/Desktop/basket-option-hedging')
OUT=Path(__file__).resolve().parent
E={r['id']:r for r in map(json.loads,(ROOT/'reports/historical/benchmark/representative.jsonl').read_text().splitlines())}

def rows(path): return list(csv.DictReader(path.open()))
def close(a,b,label):
    assert math.isclose(a,b,rel_tol=0,abs_tol=2e-7),(label,a,b)

results=[]
for policy in ('unhedged','delta','parametric','gaussian','volgan'):
    folder=OUT/'reports'/policy
    posts=rows(folder/'postings.csv');orders=rows(folder/'orders.csv');accounts=rows(folder/'account.csv')
    summary=json.loads((folder/'summary.json').read_text())
    instrument=summary['provenance']['configuration']['instruments']['B']
    coeff=dict(instrument['coefficients']);mult=instrument['multiplier'];strike=instrument['strike']
    assert len({p['id'] for p in posts})==len(posts)
    pbytime={}
    for p in posts:pbytime.setdefault(p['time'],[]).append(p)
    cash=100000.;positions={};R=Q=costs=fees=interest=borrow=turnover=liabcash=hedgemarkcash=0.
    previous_time=None;previous_marks=None;count=0
    for snapshot in accounts:
        if snapshot['phase']!='event':continue
        t=snapshot['time'];event=E[snapshot['event_id']];marks=event['valuation']['marks']
        oldpositions=dict(positions)
        for p in pbytime.get(t,[]):
            kind=p['kind'];key=p['instrument_id'];dq=float(p['quantity']);delta=float(p['cash_change'])
            cash+=delta
            if kind in ('fill','origination'):
                multiplier=mult if key=='B' else 1.
                close(delta,-dq*multiplier*float(p['price']),('trade',p['id']))
                positions[key]=positions.get(key,0.)+dq
                if kind=='origination':liabcash+=delta
                else:
                    turnover+=abs(dq*float(p['price']))
                    costs+=dq*(float(p['price'])-float(p['reference_mark']))
                    hedgemarkcash-=dq*float(p['reference_mark'])
            elif kind=='settlement':
                assert key=='B'
                payoff=max(strike-sum(coeff[k]*event['fixings'][0]['spots'][k] for k in coeff),0.)
                close(delta,positions[key]*mult*payoff,'settlement')
                liabcash+=delta;positions.pop(key)
            elif kind=='collateral':
                assert delta==0;R+=float(p['restriction_change'])
            elif kind=='short_proceeds':
                assert delta==0;Q+=float(p['restriction_change'])
            elif kind=='fee':fees-=delta
            elif kind=='interest':
                close(delta,0.,'zero configured interest');interest+=delta
            elif kind=='borrow':
                dt=(datetime.fromisoformat(t)-previous_time).total_seconds()/(365*86400)
                expected=-max(-oldpositions.get(key,0),0)*previous_marks[key]*.01*dt
                close(delta,expected,('borrow',p['id']));borrow-=delta
            elif kind=='dividend':hedgemarkcash+=delta
            else:raise AssertionError(('unexpected posting kind',kind))
        positions={k:v for k,v in positions.items() if abs(v)>1e-10}
        marked=sum(q*(mult if k=='B' else 1)*marks[k] for k,q in positions.items())
        liab=liabcash+positions.get('B',0)*mult*marks.get('B',0)
        hedge=hedgemarkcash+sum(q*marks[k] for k,q in positions.items() if k!='B')
        equity=cash+marked
        expected=dict(cash=cash,equity=equity,posted_margin_cash=R,restricted_short_proceeds=Q,available_cash=cash-Q-R,
                      liability_pnl=liab,hedge_pnl=hedge,execution_cost=costs,fees=fees,trading_costs=costs+fees,
                      interest=interest,borrow_charges=borrow,financing_costs=borrow-interest,
                      stock_turnover=turnover,net_pnl=liab+hedge-costs-fees+interest-borrow)
        for k,v in expected.items():close(v,float(snapshot[k]),(policy,t,k))
        close(equity-100000,expected['net_pnl'],'economic decomposition')
        reported=json.loads(snapshot['positions'])
        for k in set(positions)|set(reported):close(positions.get(k,0),reported.get(k,0),'position')
        previous_time=datetime.fromisoformat(t);previous_marks=marks;count+=len(expected)+1
    submitted={};used={};fillcount=0
    for o in orders:
        if o['outcome']=='submitted':submitted[o['order_id']]=o
        if o['outcome']!='filled':continue
        submission=submitted[o['order_id']]
        assert datetime.fromisoformat(o['time'])>datetime.fromisoformat(submission['time'])
        event=E[o['event_id']];decision=E[submission['event_id']]
        assert event['time']>=decision['earliest_execution'] and event['time']<=decision['order_deadline']
        quote=next(q for q in event['quotes'] if q['id']==o['quote_id'])
        assert quote['time']==o['time'] and quote['available_at']<=o['time']
        dq=float(o['quantity']);side='ask' if dq>0 else 'bid';px=float(o['price'])
        assert px==quote[side]
        spot=next(obs['value'] for obs in event['observations'] if obs['instrument_id']==o['instrument_id'])
        close(px,spot*(1.0005 if dq>0 else .9995),'assumed 5bp from observed open')
        capacity_key=(quote['id'],side);used[capacity_key]=used.get(capacity_key,0)+abs(dq)
        assert used[capacity_key]<=quote[side+'_size']
        fillcount+=1
    assert positions=={};close(R,0,'final released margin');close(Q,0,'final released proceeds')
    settlements=rows(folder/'settlements.csv');assert len(settlements)==1
    results.append(dict(policy=policy,event_snapshots=sum(a['phase']=='event' for a in accounts),independent_numeric_checks=count,
                        fills_checked=fillcount,terminal_cash=cash,terminal_pnl=cash-100000,
                        one_exact_settlement=True,all_restrictions_released=True,unexplained_cash_injections=0))
(OUT/'independent_financial_checks.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print('Checked',sum(r['independent_numeric_checks'] for r in results),'event account identities;',sum(r['fills_checked'] for r in results),'eligible fills.')
