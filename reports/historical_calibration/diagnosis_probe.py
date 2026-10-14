"""Read-only training/development diagnosis; no model training candidate."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from basket_hedging.volgan import (
    VolGANModel, VolGANGenerator, VolGANDiscriminator, prepared_training_pairs,
    smoothness_penalties, RETURN_SCALE,
)

ROOT = Path(r'C:\Users\petar\Desktop\basket-option-hedging')
OUT = Path(__file__).parent
CHECKPOINT = ROOT/'reports/historical/models/volgan.json'
PREPARED = Path.home()/'.cache/basket-hedging/prepared-etf-v1-actions/surfaces.npz'
before = {str(p): sha256(p.read_bytes()).hexdigest() for p in (CHECKPOINT, PREPARED)}
torch.set_num_threads(1)
torch.manual_seed(4401)
model = VolGANModel.load(CHECKPOINT)
with np.load(PREPARED, allow_pickle=False) as stored:
    dates = stored['dates']
    stop = np.searchsorted(dates, '2019-01-01')
    data = {key: (stored[key][:stop].copy() if stored[key].ndim and len(stored[key]) == len(dates)
                  else stored[key].copy()) for key in stored.files}
x, y, train_idx = prepared_training_pairs(data, model.grid, split='train')
vx, vy, dev_idx = prepared_training_pairs(data, model.grid, split='validation')
assets, grid = len(model.asset_ids), model.grid
report = {'purpose': 'train and 2018 diagnosis only; no candidate training',
          'development_dates': [str(data['dates'][dev_idx[0]]), str(data['dates'][dev_idx[-1]])],
          'development_rows': len(dev_idx), 'training_rows': len(x),
          'training_sha256': sha256(x.astype('<f8').tobytes()+y.astype('<f8').tobytes()).hexdigest(),
          'checkpoint_config': vars(model.config), 'metadata': dict(model.metadata)}

# Independent direct flattening and next-label formulas at every retained row.
errors = {'features': 0., 'labels': 0., 'reconstruction_spots': 0., 'reconstruction_iv': 0.}
for features, targets, indices in ((x, y, train_idx), (vx, vy, dev_idx)):
    for feature, label, target in zip(features, targets, indices):
        past = np.diff(np.log(data['close'][target-22:target]), axis=0)
        condition = np.concatenate([
            np.r_[np.sqrt(252)*past[-1,a], np.sqrt(252)*past[-2,a],
                  np.sqrt(12*np.sum(past[:,a]**2)), np.log(data['iv'][target-1,a]).T.ravel()]
            for a in range(assets)])
        expected = np.concatenate([
            np.r_[np.sqrt(252)*(np.log(data['close'][target,a])-np.log(data['close'][target-1,a])),
                  (np.log(data['iv'][target,a])-np.log(data['iv'][target-1,a])).T.ravel()]
            for a in range(assets)])
        errors['features'] = max(errors['features'], float(np.max(abs(feature-condition))))
        errors['labels'] = max(errors['labels'], float(np.max(abs(label-expected))))
        label_block = label.reshape(assets, grid.size+1)
        spot_back = data['close'][target-1]*np.exp(label_block[:,0]/np.sqrt(252))
        iv_back = np.exp(feature.reshape(assets, grid.size+3)[:,3:]+label_block[:,1:])
        errors['reconstruction_spots'] = max(errors['reconstruction_spots'], float(np.max(abs(spot_back-data['close'][target]))))
        errors['reconstruction_iv'] = max(errors['reconstruction_iv'], float(np.max(abs(iv_back-np.stack([v.T.ravel() for v in data['iv'][target]])))))
report['alignment_errors'] = errors

# Frozen checkpoint noise response on every developmental state. The noise is
# common across assets and identical across states, not generated per asset.
noise = np.random.default_rng(4402).standard_normal((512, model.config.noise_dim))
noise_tensor = torch.tensor(noise, dtype=torch.float32)
returns, spread_stds, correlations, surface_std = [], [], [], []
for feature in vx:
    with torch.no_grad():
        output = model.generator(torch.tensor(feature, dtype=torch.float32).repeat(len(noise),1), noise_tensor)
    blocks = output.numpy().reshape(len(noise), assets, grid.size+1)
    r = blocks[:,:,0]/RETURN_SCALE
    returns.append(r)
    spread_stds.append((r[:,1]-r[:,2]).std(ddof=1))
    correlations.append(np.corrcoef(r,rowvar=False)[np.triu_indices(assets,1)])
    surface_std.append(blocks[:,:,1:].std(axis=0,ddof=1).mean(axis=1))
returns = np.array(returns, dtype=float)
truth = vy.reshape(len(vy),assets,grid.size+1)[:,:,0]/RETURN_SCALE
means = returns.mean(axis=1)
conditional = np.array([np.cov(row,rowvar=False,ddof=0) for row in returns])
pooled = np.cov(returns.reshape(-1, assets), rowvar=False,ddof=0)
between = np.cov(means,rowvar=False,ddof=0)
report['development_variation'] = {
    'assets': model.asset_ids, 'pair_order': ['SPY/QQQ','SPY/IWM','QQQ/IWM'],
    'conditional_return_std_quantiles': np.quantile(returns.std(axis=1,ddof=1),[0,.1,.5,.9,1],axis=0).tolist(),
    'conditional_correlations_quantiles': np.quantile(correlations,[0,.1,.5,.9,1],axis=0).tolist(),
    'conditional_mean_return_std_over_dates': means.std(axis=0,ddof=1).tolist(),
    'observed_return_std_over_dates': truth.std(axis=0,ddof=1).tolist(),
    'mean_conditional_covariance': conditional.mean(axis=0).tolist(),
    'between_state_conditional_mean_covariance': between.tolist(),
    'generated_pooled_covariance': pooled.tolist(),
    'total_covariance_identity_error': float(np.max(abs(pooled-conditional.mean(axis=0)-between))),
    'conditional_qqq_iwm_spread_std_quantiles': np.quantile(spread_stds,[0,.1,.5,.9,1]).tolist(),
    'observed_unconditional_qqq_iwm_spread_std': float((truth[:,1]-truth[:,2]).std(ddof=1)),
    'generated_conditional_log_iv_std_quantiles': np.quantile(surface_std,[0,.1,.5,.9,1],axis=0).tolist(),
    'return_90_coverage': ((truth>=np.quantile(returns,.05,axis=1))&(truth<=np.quantile(returns,.95,axis=1))).mean(axis=0).tolist(),
    'warning': 'Conditional and unconditional covariances are distinct; no matching constraint is inferred.'}

# Latent Jacobian at a single state: nonzero effects and separately parameterized
# outputs. This is sensitivity evidence, not proof of calibrated dispersion.
condition = torch.tensor(vx[0],dtype=torch.float32)[None,:]
latent = torch.tensor(noise[0],dtype=torch.float32, requires_grad=True)
def generated(z):
    return model.generator(condition,z[None,:]).reshape(assets,grid.size+1)
jac = torch.autograd.functional.jacobian(generated,latent).detach().numpy()
report['latent_jacobian'] = {
    'return_row_norm': np.linalg.norm(jac[:,0],axis=1).tolist(),
    'log_iv_block_norm': np.linalg.norm(jac[:,1:],axis=(1,2)).tolist(),
    'return_jacobian_equal_pairwise': [bool(np.array_equal(jac[a,0],jac[b,0])) for a,b in ((0,1),(0,2),(1,2))],
    'output_return_weight_rows_equal_pairwise': [bool(torch.equal(model.generator.layers[-1].weight[a*(grid.size+1)],model.generator.layers[-1].weight[b*(grid.size+1)])) for a,b in ((0,1),(0,2),(1,2))]}

# One isolated copied-model update checks graph connectivity and parameter
# changes. It is never saved or used for model comparison or inference.
batch_x = torch.tensor(x[:100],dtype=torch.float32)
batch_y = torch.tensor(y[:100],dtype=torch.float32)
batch_z = torch.tensor(noise[:100],dtype=torch.float32)
bce = nn.BCELoss()
weights = (model.metadata['smoothness_m'],model.metadata['smoothness_t'])
def gradient_diagnosis(generator,discriminator):
    fake = generator(batch_x,batch_z)
    real_p, fake_p = discriminator(batch_x,batch_y),discriminator(batch_x,fake.detach())
    d_loss = .5*(bce(real_p,torch.ones_like(real_p))+bce(fake_p,torch.zeros_like(fake_p)))
    d_grads = torch.autograd.grad(d_loss,tuple(discriminator.parameters()),retain_graph=True)
    probability = discriminator(batch_x,fake)
    adversarial = .5*bce(probability,torch.ones_like(probability))
    future = batch_x.reshape(100,assets,grid.size+3)[:,:,3:]+fake.reshape(100,assets,grid.size+1)[:,:,1:]
    surfaces = future.reshape(100,assets,len(grid.maturities),len(grid.moneyness)).transpose(-1,-2)
    m_loss,t_loss = smoothness_penalties(surfaces,grid)
    grad_vectors = [torch.cat([g.flatten() for g in torch.autograd.grad(loss,tuple(generator.parameters()),retain_graph=True)]) for loss in (adversarial,m_loss,t_loss)]
    norms = [float(g.norm()) for g in grad_vectors]
    combined = grad_vectors[0]+weights[0]*grad_vectors[1]+weights[1]*grad_vectors[2]
    return {'losses': [float(v.detach()) for v in (d_loss,adversarial,m_loss,t_loss)],
            'generator_raw_grad_norms_adversarial_m_t': norms,
            'weighted_smooth_to_adversarial_ratios': [weights[i]*norms[i+1]/norms[0] for i in range(2)],
            'generator_combined_grad_norm': float(combined.norm()),
            'discriminator_grad_norm': float(torch.sqrt(sum(g.square().sum() for g in d_grads))),
            'discriminator_mean_real_fake': [float(real_p.mean().detach()),float(fake_p.mean().detach())],
            'finite': all(bool(torch.isfinite(g).all()) for g in grad_vectors+list(d_grads))}
with torch.random.fork_rng():
    torch.manual_seed(model.config.seed)
    initial_g = VolGANGenerator(x.shape[1],y.shape[1],model.config)
    initial_d = VolGANDiscriminator(x.shape[1],y.shape[1],model.config)
report['gradient_diagnosis'] = {'scope': 'one fixed train-only batch; not evidence of every historical update',
    'weights': weights,
    'initial_networks': gradient_diagnosis(initial_g,initial_d),
    'final_checkpoint': gradient_diagnosis(model.generator,model.discriminator)}
g,d = deepcopy(model.generator),deepcopy(model.discriminator)
g_before = [p.detach().clone() for p in g.parameters()]
d_before = [p.detach().clone() for p in d.parameters()]
g_opt = torch.optim.RMSprop(g.parameters(),lr=model.config.learning_rate)
d_opt = torch.optim.RMSprop(d.parameters(),lr=model.config.learning_rate)
fake = g(batch_x,batch_z)
d_opt.zero_grad(set_to_none=True)
real_p,fake_p = d(batch_x,batch_y),d(batch_x,fake.detach())
d_loss = .5*(bce(real_p,torch.ones_like(real_p))+bce(fake_p,torch.zeros_like(fake_p)))
d_loss.backward(); d_opt.step()
g_opt.zero_grad(set_to_none=True)
fake = g(batch_x,batch_z)
p = d(batch_x,fake)
future = batch_x.reshape(100,assets,grid.size+3)[:,:,3:]+fake.reshape(100,assets,grid.size+1)[:,:,1:]
m,t = smoothness_penalties(future.reshape(100,assets,len(grid.maturities),len(grid.moneyness)).transpose(-1,-2),grid)
(.5*bce(p,torch.ones_like(p))+weights[0]*m+weights[1]*t).backward();g_opt.step()
report['isolated_update'] = {
    'label': 'one copied-network connectivity probe; fresh optimizer state; discarded, not a training candidate',
    'generator_changed_parameter_tensors': sum(not torch.equal(a,b) for a,b in zip(g_before,g.parameters())),
    'discriminator_changed_parameter_tensors': sum(not torch.equal(a,b) for a,b in zip(d_before,d.parameters())),
    'generator_parameter_tensors': len(g_before), 'discriminator_parameter_tensors':len(d_before)}
report['training_history'] = [dict(row) for row in model.training_history]
report['original_files_unchanged'] = {str(p): before[str(p)]==sha256(p.read_bytes()).hexdigest() for p in (CHECKPOINT,PREPARED)}
(OUT/'diagnosis.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({key: report[key] for key in ('development_dates','development_rows','training_rows','training_sha256','alignment_errors','latent_jacobian','gradient_diagnosis','isolated_update','original_files_unchanged')},indent=2))
