"""2018 diagnostic: generated IV enters repricing; no financial backtest."""
from datetime import timedelta
from hashlib import sha256
from pathlib import Path
import json
import numpy as np
from basket_hedging.volgan import VolGANModel,prepared_training_pairs
from basket_hedging.historical import training_dynamics,session_time
from basket_hedging.historical_hedge import SurfaceBasketValuator,reprice_basket_scenarios
from basket_hedging.contracts import Instrument
from basket_hedging.engine import PolicyView
from basket_hedging.market import Observation
from basket_hedging.surfaces import IVSurface

ROOT=Path(r'C:\Users\petar\Desktop\basket-option-hedging')
OUT=Path(__file__).parent
path=ROOT/'reports/historical/models/volgan.json'
model=VolGANModel.load(path)
with np.load(Path.home()/'.cache/basket-hedging/prepared-etf-v1-actions/surfaces.npz',allow_pickle=False) as source:
    dates=source['dates']; count=np.searchsorted(dates,'2019-01-01')
    data={key:source[key][:count].copy() if source[key].ndim and len(source[key])==len(dates) else source[key].copy() for key in source.files}
features,_,indices=prepared_training_pairs(data,model.grid,split='validation')
target=indices[0];decision=target-1
time=session_time(data['dates'][decision]);horizon=session_time(data['dates'][target])
earliest=session_time(data['dates'][target],opening=True)
stocks=model.asset_ids;current=data['close'][decision]
_,_,correlation=training_dynamics(data)
pricing=SurfaceBasketValuator(stocks,correlation,session_time('2017-12-29'),samples=512,pricing_seed=731)
instruments={stock:Instrument(stock,'stock') for stock in stocks}
instruments['B']=Instrument('B','basket_put',multiplier=100,strike=300,expiry=time+timedelta(days=30),
                             coefficients=tuple(zip(stocks,map(float,100/current))))
surfaces={stock:IVSurface(model.grid,data['iv'][decision,a],time,time,'2018 diagnostic current surface') for a,stock in enumerate(stocks)}
packet=pricing.packet(time,dict(zip(stocks,map(float,current))),instruments,surfaces)
observations=tuple(Observation(f'close-{stock}',stock,float(price),time,time) for stock,price in zip(stocks,current))
view=PolicyView('surface-dependency-diagnostic',time,0,instruments,{'B':-1},100000,0,0,100000,
                packet,observations,(),stocks,False,
                {'earliest_execution':earliest,'next_rebalance_time':horizon},
                market={o.instrument_id:o for o in observations})
noise=np.random.Generator(np.random.PCG64(1730)).standard_normal((256,model.config.noise_dim))
future_spots,future_iv=model.sample(features[0],current,noise)
frozen_iv=np.broadcast_to(data['iv'][decision],future_iv.shape)
def reprice(iv):
    return reprice_basket_scenarios(view,view.positions,stocks,future_spots=future_spots,
        future_surfaces=iv,grid=model.grid,horizon=horizon,valuator=pricing,seed=1730,
        input_ids=('frozen-volgan','2018-current-state'))
joint=reprice(future_iv);fixed_iv=reprice(frozen_iv)
np.testing.assert_array_equal(joint.delta_h,fixed_iv.delta_h)
assert np.max(abs(joint.delta_v-fixed_iv.delta_v))>1e-6
result=dict(scope='Read-only generated-surface dependency diagnostic, no trades/P&L/backtest; probe basket only',
    checkpoint_sha256=sha256(path.read_bytes()).hexdigest(),
    decision=time.isoformat(),earliest_execution=earliest.isoformat(),endpoint=horizon.isoformat(),
    pricing_calibration_end='2017-12-29',pricing_samples=512,pricing_seed=731,scenario_seed=1730,scenarios=256,
    contract=dict(kind='European arithmetic basket put',expiry=instruments['B'].expiry.isoformat(),
                  strike=300,multiplier=100,quantity=-1,coefficients=instruments['B'].coefficients),
    current_basket_put_unit_mark=packet.marks['B'],
    generated_spot_hash=sha256(future_spots.astype('<f8').tobytes()).hexdigest(),
    stock_hedge_changes_identical=True,
    max_liability_change_difference=float(np.max(abs(joint.delta_v-fixed_iv.delta_v))),
    mean_abs_liability_change_difference=float(np.mean(abs(joint.delta_v-fixed_iv.delta_v))),
    liability_change_sd_generated_iv=float(joint.delta_v.std(ddof=1)),
    liability_change_sd_carried_iv=float(fixed_iv.delta_v.std(ddof=1)),
    conclusion='The actual frozen generator surfaces change the liability response supplied to the unchanged optimizer; stock hedge regressors stay exactly fixed.',
    caveat='Comparing generated versus carried IV here is a dependency probe, not a changed benchmark or model-selection metric.')
(OUT/'valuation_dependency_probe.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
