"""Independent formulas on manufactured observations, not confirmation data."""
from datetime import date,timedelta
from pathlib import Path
from types import SimpleNamespace
from math import sqrt
import json
import numpy as np
from basket_hedging.surfaces import SurfaceGrid
from basket_hedging.volgan import validate_prepared_volgan

OUT=Path(__file__).parent
grid=SurfaceGrid((.8,1.2),(.1,.5))
days=[str(date(year,1,1)+timedelta(days=i)) for year in (2018,2019) for i in range(24)]
n=len(days)
prices=np.full((n,3),100.)
actual_r=np.array([[.03,-.01,.015],[-.025,.04,-.01]])
prices[22]=100*np.exp(actual_r[0]);prices[23]=prices[22]*np.exp(actual_r[1])
prices[46]=100*np.exp(actual_r[0]);prices[47]=prices[46]*np.exp(actual_r[1])
iv=np.full((n,3,2,2),.2)
iv[22:24]=.2*np.exp(np.array([[[[.1,-.02],[.03,.02]],[[.04,.02],[.01,.03]],[[.03,.04],[.02,.06]]],
                            [[[.02,.04],[.03,.07]],[[.03,-.02],[.01,.08]],[[.04,.01],[.01,.05]]]]))
iv[46:48]=iv[22:24]
consecutive=np.ones(n,dtype=bool);consecutive[[0,24]]=False
data=dict(dates=np.array(days),symbols=np.array(['SPY','QQQ','IWM']),close=prices,iv=iv,
          valid_surface=np.ones((n,3),bool),consecutive=consecutive)
model=SimpleNamespace(asset_ids=('SPY','QQQ','IWM'),grid=grid,config=SimpleNamespace(noise_dim=2),metadata={})
generated=np.array([[[-.02,.01,-.01],[0.,-.01,-.02],[.01,.02,.03],[.02,.03,.01]],
                    [[-.03,.03,.01],[-.01,-.02,-.01],[.015,.01,.02],[.04,.02,.025]]])
surface_changes=np.array([[[-.03,.02,.01],[.02,.015,-.01],[.04,-.01,.03],[.06,-.02,.02]],
                          [[-.02,-.01,.03],[.02,.03,.01],[.06,.02,-.01],[.04,.01,.02]]])
node_offset=np.array([[-.01,.02],[.015,-.025]])
seen={}
def sampler(feature,target,normals):
    seen[target]=normals.copy()
    idx=target%24-22
    future_spots=data['close'][target-1]*np.exp(generated[idx])
    future_iv=data['iv'][target-1]*np.exp(surface_changes[idx,:,:,None,None]+node_offset)
    return future_spots,future_iv
development=validate_prepared_volgan(model,data,year=2018,samples=4,seed=882,sampler=sampler)
seen2018={k:v.copy() for k,v in seen.items()};seen.clear()
# Only artificial 2019 observations; no historical confirmation is accessed.
confirmation_fixture=validate_prepared_volgan(model,data,year=2019,samples=4,seed=882,sampler=sampler)
seen2019={k:v.copy() for k,v in seen.items()};seen.clear()
combined=validate_prepared_volgan(model,data,samples=4,seed=882,sampler=sampler)
assert set(seen2018)=={22,23} and set(seen2019)=={46,47}
assert all(np.array_equal(v,seen[k]) for k,v in seen2018.items())
assert all(np.array_equal(v,seen[k]) for k,v in seen2019.items())
rng=np.random.Generator(np.random.PCG64(882))
for idx in (22,23,46,47):
    assert np.array_equal(seen[idx],rng.standard_normal((4,2)))

expected_energies=[];interval_width=[];interval_score=[];coverage=[]
for draws,truth in zip(generated,actual_r):
    distance=lambda a,b:sqrt(sum(float(x-y)**2 for x,y in zip(a,b)))
    expected_energies.append(sum(distance(row,truth) for row in draws)/4
        -sum(distance(a,b) for a in draws for b in draws)/32)
    width=[];score=[];covered=[]
    for asset in range(3):
        ordered=sorted(draws[:,asset]);low=.85*ordered[0]+.15*ordered[1]
        high=.15*ordered[2]+.85*ordered[3]
        width.append(high-low)
        penalty=20*(low-truth[asset]) if truth[asset]<low else 20*(truth[asset]-high) if truth[asset]>high else 0
        score.append(high-low+penalty);covered.append(low<=truth[asset]<=high)
    interval_width.append(width);interval_score.append(score);coverage.append(covered)
np.testing.assert_allclose(development['energy_score'],np.mean(expected_energies),atol=1e-14)
np.testing.assert_allclose(development['return_interval_width_90'],np.mean(interval_width,axis=0),atol=1e-14)
np.testing.assert_allclose(development['return_interval_score_90'],np.mean(interval_score,axis=0),atol=1e-14)
np.testing.assert_array_equal(development['return_5_95_percentile_coverage'],np.mean(coverage,axis=0))

def covariance(rows,ddof):
    means=rows.mean(axis=0)
    return np.array([[sum((r[i]-means[i])*(r[j]-means[j]) for r in rows)/(len(rows)-ddof)
                      for j in range(3)] for i in range(3)])
within=np.mean([covariance(rows,1) for rows in generated],axis=0)
means_cov=covariance(generated.mean(axis=1),0)
mixture=covariance(generated.reshape(-1,3),0)
np.testing.assert_allclose(development['mean_generated_return_covariance'],within,atol=1e-16)
np.testing.assert_allclose(development['covariance_of_conditional_means'],means_cov,atol=1e-16)
np.testing.assert_allclose(development['generated_mixture_covariance'],mixture,atol=1e-16)
np.testing.assert_allclose(mixture,within*.75+means_cov,atol=1e-16)

def scalar_corr(a,b):
    ac=a-a.mean();bc=b-b.mean()
    return float(sum(ac*bc)/sqrt(sum(ac*ac)*sum(bc*bc)))
expected_pair=[(i,j) for i in range(3) for j in range(i+1,3)]
for i,j in expected_pair:
    corrs=[scalar_corr(rows[:,i],rows[:,j]) for rows in generated]
    key=f'{model.asset_ids[i]}/{model.asset_ids[j]}'
    np.testing.assert_allclose(development['generated_pairwise_correlations'][key]['quantiles_05_50_95'],np.quantile(corrs,[.05,.5,.95]),atol=1e-13)
np.testing.assert_allclose(development['surface_changes']['generated_mean'],surface_changes.mean(axis=(0,1))+node_offset.mean(),atol=1e-14)
np.testing.assert_allclose(development['surface_changes']['mean_conditional_std'],surface_changes.std(axis=1,ddof=1).mean(axis=0),atol=1e-14)
surface_rmse=[]
for a in range(3):
    se=[]
    for state,target in enumerate((22,23)):
        mean_generated_log=np.log(iv[target-1,a])+surface_changes[state,:,a].mean()+node_offset
        se.extend((mean_generated_log-np.log(iv[target,a])).ravel()**2)
    surface_rmse.append(sqrt(sum(se)/len(se)))
np.testing.assert_allclose(development['log_iv_mean_forecast_rmse'],surface_rmse,atol=1e-14)
for a in range(3):
    corrs=[scalar_corr(generated[s,:,a],surface_changes[s,:,a]) for s in range(2)]
    np.testing.assert_allclose(development['surface_changes']['return_mean_log_iv_change_correlations'][a]['quantiles_05_50_95'],np.quantile(corrs,[.05,.5,.95]),atol=1e-13)
result={'status':'PASS','scope':'Manufactured observations only, no real 2019 data',
        'checks':['direct Euclidean energy pair sum','central interval score and coverage',
                  'sample within-state vs population mixture covariance','pair correlations',
                  'grid-average surface changes and full-grid forecast RMSE','return/surface correlation',
                  'selected target-year and consistent original noise stream'],
        'manufactured_development':development}
(OUT/'metrics_probe.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({'status':'PASS','independent_checks':result['checks']},indent=2))
