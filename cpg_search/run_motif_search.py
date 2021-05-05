import nengo
import multiprocessing
import numpy as np
from itertools import product
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from hashids import Hashids
import os
from tqdm import tqdm
import nengolib
from cpg_search.utils import create_adj_matrix, sin_fit, sin_fit_jac
from scipy import optimize
from cpg_search import MotorNode
negative_alpha = lambda  tau,alpha : nengo.LinearFilter([-alpha,1], [tau ** 2, 2*tau, 1])
strong_alpha = lambda  tau,alpha : nengo.LinearFilter([alpha,1], [tau ** 2, 2*tau, 1])

def generate_module_matrices(m, lim=4, scale=1):
    for i in product("01", repeat=m**2):
        ret = np.reshape(i*scale,(m,m)).astype(int)
        if np.trace(ret) > 0:
            continue
        elif np.sum(ret.ravel()) > lim or np.sum(ret.ravel()) == 0:
            continue
        
        else:
            yield ret

def generate_lr_matrices(m, lim=4,scale=-1):
    for i in product("01", repeat=m**2):
        ret = np.reshape(i*scale,(m,m)).astype(int)
        if np.sum(ret.ravel()) > lim or np.sum(ret.ravel()) == 0:
            continue
        
        else:
            yield ret
def generate_intermodule_matrices(m, scale=1):
    for i in product("01", repeat=m):
        ret = np.zeros((m,m))
        np.fill_diagonal(ret,i)
        ret = ret*scale
        yield ret
def make_hash(metadata):
    st = []
    for met in metadata:
        st.extend(list(met.ravel().astype(int)))
    hsh = Hashids()
    
    hsh = hsh.encode(*st)
    return hsh
def save_return(ret, path='.'):
    metadata = ret['_metadata']
    hsh = make_hash(metadata)
    fn = os.path.join(path, '{}.pkl'.format(hsh))
    with open(fn, 'wb') as fp:
        pickle.dump(ret,fp,protocol=pickle.HIGHEST_PROTOCOL )

    return hsh

def fit_curve(ret,tslc,motor_id=0):
    try:
        y = np.array(ret['motor_values'])[tslc,motor_id].ravel()
    except IndexError:
        y = np.array(ret['motor_values'])[tslc].ravel()
    x = np.array(ret['time'][tslc]).ravel()
    
    ff = np.fft.fftfreq(len(x), (x[1]-x[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(y))
    guess_amp = np.std(y) * 2.**0.5
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])*2*np.pi # excluding the zero frequency "peak", which is related to offset
    guess_offset = np.mean(y)
    
    init_guess = [guess_amp,guess_freq,0,guess_offset]
    try:
        params, cov = optimize.curve_fit(
            sin_fit, 
            x, 
            y,                                      
            p0=init_guess,
            jac=sin_fit_jac,
        )
        yhat = sin_fit(x,*params)
        err = np.sum(np.sqrt(((y-yhat)**2)))
    except RuntimeError:
        err = np.inf
        params = np.ones(4)*np.nan
        cov = np.ones((4,4))*np.nan
        yhat = np.ones(len(x))*np.nan
    return params, cov, err,yhat




def generate_nengo_model(module_matrix,
    lr_matrix,
    intermodule_matrix=None, 
    modules=1, 
    generate_probes=True,
    crop=None,
    defaults={},
    motor_filter=nengo.Alpha(0.05),
    motor_probe_filter=nengo.Alpha(0.05),
    w_synapse=0.01, 
    imm_synapse=0.01,
    inp_conn_type=None,
    tau=None,
    alpha=None
):
    # Create complete module matrix
    n = len(module_matrix)
    W = np.zeros((n*2,n*2))
    W[:n,:n] = module_matrix
    W[n:,n:] = module_matrix
    W[:n,n:] = lr_matrix
    W[n:,:n] = lr_matrix
    mn = MotorNode(modules,crop=crop)
    rng = np.random.RandomState(0)
    

    def create_single_module(i,model, motor_node):
        biases = np.zeros(n*2)
        biases = nengo.dists.Uniform(5,5).sample(n*2, rng=rng)
        # biases[0] = nengo.dists.Uniform(5,5).sample(1, rng=rng)
        # biases[n] = nengo.dists.Uniform(5,5).sample(1, rng=rng)
        edefaults = dict(
            gain = nengo.dists.Uniform(5,5),
            bias = biases,
    #         noise=nengo.processes.WhiteNoise(),
            # intercepts=nengo.dists.Uniform(0.1, 0.1),
            # neuron_type=nengo.LIF(tau_rc=0.02, tau_ref=0.01),
            # neuron_type=nengo.Izhikevich(tau_recovery=0.02, coupling=0.2, reset_voltage=-65., reset_recovery=8.),
            neuron_type=nengo.AdaptiveLIF(tau_n=1, inc_n=0.5, tau_rc=0.02, tau_ref=0.002, min_voltage=0, amplitude=1)
        )
        edefaults.update(defaults)
        with model:
            pop = nengo.Ensemble(n*2,2, label='module_{}'.format(i),**edefaults)
            module_conns = nengo.Connection(pop.neurons,pop.neurons, transform=-W, synapse=w_synapse)
            # The first neuron is always the measured one -- since we iterate through all matrices, 
            # this should cover every motif
            motor_conn_l = nengo.Connection(pop.neurons[0], motor_node[i*2],synapse=motor_filter)
            motor_conn_r = nengo.Connection(pop.neurons[n], motor_node[i*2+1], synapse=motor_filter)
            if inp_conn_type=='standard':
                inv_motor_conn_l = nengo.Connection(motor_node[i*2],pop.neurons[0], synapse=motor_filter)
                inv_motor_conn_r = nengo.Connection(motor_node[i*2+1],pop.neurons[n], synapse=motor_filter)
            elif inp_conn_type=='advanced':
                if tau is None or alpha is None:
                    raise ValueError('tau or alpha cannot be None for advanced input connection')
                fast_conn_l = nengo.Connection(motor_node[4*i],pop.neurons[0], transform=-1, synapse=negative_alpha(tau,alpha))
                fast_conn_r = nengo.Connection(motor_node[4*i+1],pop.neurons[n], transform=-1, synapse=negative_alpha(tau,alpha))
                slow_conn_l = nengo.Connection(motor_node[4*i+2],pop.neurons[0], transform=-1, synapse=strong_alpha(tau,alpha))
                slow_conn_r = nengo.Connection(motor_node[4*i+3],pop.neurons[n], transform=-1, synapse=strong_alpha(tau,alpha))
            elif inp_conn_type is None:
                pass
            else:
                raise ValueError('Unsupported motor input connection type {}'.format(str(inp_conn_type)))
            if generate_probes:
                spike_probe = nengo.Probe(pop.neurons, 'spikes', synapse=motor_probe_filter)
                val_probe = nengo.Probe(pop,synapse=nengo.Triangle(0.5))
                probes = {'spikes': spike_probe, 'values' : val_probe}
            else:
                probes = {}
        return pop, probes
    probes = []
    ens = []
    with nengo.Network(seed=0) as model:
        motor_node = nengo.Node(size_in=2*modules, output=mn.integrate, label="motor_node")
        if inp_conn_type == 'standard':
            motor_node.size_out = 2*modules
        elif inp_conn_type == 'advanced':
            motor_node.size_out = 4*modules
        if modules == 1:
            pop, mprobes = create_single_module(0,model,motor_node)
            probes.append(mprobes)
            ens.append(pop)
        elif intermodule_matrix is not None:
            im_transform = np.eye(n*2,n*2)
      
            im_transform[:n,:n] = intermodule_matrix
            im_transform[n:,n:] = intermodule_matrix

            
            prev_module, mprobes = create_single_module(0,model, motor_node)
            ens.append(prev_module)
            probes.append(mprobes)
            for module in range(1, modules):
                next_module, mprobes = create_single_module(module, model, motor_node)
                nengo.Connection(prev_module.neurons, next_module.neurons, transform=im_transform, synapse=imm_synapse)
                
                prev_module = next_module
                
                probes.append(mprobes)
                ens.append(next_module)

        else:
            raise ValueError('intermodule matrix is not set, but number of modules is >1')
                
    model.ens = ens
    return model,mn, probes
def create_and_run_model(module_matrix,lr_matrix,intermodule_matrix=None, modules=1, metadata=None, **kwargs):
    model, mn, probes = generate_nengo_model(module_matrix, lr_matrix, intermodule_matrix, modules, **kwargs)
    n = mn.hs
    with nengo.Simulator(model, progress_bar=False, seed=0) as sim:
        for ens in model.ens:
            signal = sim.model.sig[ens.neurons]
            sim.signals[signal['voltage']][:n] = np.ones(n)
        sim.run(6)
    spikedata = [sim.data[probes[i]['spikes']] for i in range(len(probes))]
    valuedata = [sim.data[probes[i]['values']] for i in range(len(probes))]
    ret = {
        'time': sim.trange(),
        'motor_values': mn.rets[1:],
        'spikes' : spikedata,
        'values' : valuedata,
        '_metadata' :metadata
    }
    
    
    return ret

def run_model(model,mn,probes, metadata, **kwargs):
    
    n = mn.hs
    with nengo.Simulator(model, progress_bar=False, seed=0) as sim:
        for ens in model.ens:
            signal = sim.model.sig[ens.neurons]
            sim.signals[signal['voltage']][:n] = np.ones(n)
        sim.run(6)
    spikedata = [sim.data[probes[i]['spikes']] for i in range(len(probes))]
    valuedata = [sim.data[probes[i]['values']] for i in range(len(probes))]
    ret = {
        'time': sim.trange(),
        'motor_values': mn.rets[1:],
        'spikes' : spikedata,
        'values' : valuedata,
        '_metadata' :metadata
    }
    
    
    return ret




