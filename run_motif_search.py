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


def generate_module_matrices(m, lim=4):
    for i in product("01", repeat=m**2):
        ret = np.reshape(i,(m,m)).astype(int)
        if np.trace(ret) > 0:
            continue
        elif np.sum(ret.ravel()) > lim or np.sum(ret.ravel()) == 0:
            continue
        
        else:
            yield ret

def generate_lr_matrices(m, lim=4):
    for i in product("01", repeat=m**2):
        ret = np.reshape(i,(m,m)).astype(int)
        if np.sum(ret.ravel()) > lim or np.sum(ret.ravel()) == 0:
            continue
        
        else:
            yield ret
def generate_intermodule_matrices(m):
    for i in product("01", repeat=m):
        ret = np.zeros((m,m))
        np.fill_diagonal(ret,i)
        yield ret
class MotorNode(object):
    """
    Reads motoneuron output and torque input into the system. 
    
    
    Has the same input and output dimensionality.
    
    The first half of the input corresponds to positive angle changes, the second half is negative angle changes
    The output should be considered inhibitory to both? neurons in the oscillator
    
    """
    def __init__(self, sz):
        self.hs = sz
        self.l = slice(0, self.hs)
        self.r = slice(self.hs, None)
        self.rets = [np.zeros(sz)]
        self._t = None
    def integrate(self, t,x):
        if self._t is None:
            dt = 0.001
        else:
            dt = t - self._t
        self._t = t        
        ret = self.rets[-1] + (x[self.l] - x[self.r])*dt
        
        self.rets.append(ret)
        return ret

def generate_nengo_model(module_matrix,lr_matrix,intermodule_matrix=None, modules=1):
    # Create complete module matrix
    n = len(module_matrix)
    W = np.zeros((n*2,n*2))
    W[:n,:n] = module_matrix
    W[n:,n:] = module_matrix
    W[:n,n:] = lr_matrix
    W[n:,:n] = lr_matrix
    mn = MotorNode(modules)
    rng = np.random.RandomState(0)
    biases = np.zeros(n*2)
    
    biases[0] = nengo.dists.Uniform(4.99,5).sample(1,rng=rng)
    biases[n] = nengo.dists.Uniform(4.99,5).sample(1, rng=rng)
    defaults = dict(
        gain = nengo.dists.Uniform(4.99,5),
        bias = biases,
#         noise=nengo.processes.WhiteNoise(),
        # intercepts=nengo.dists.Uniform(0.1, 0.1),
        # neuron_type=nengo.LIF(tau_rc=0.02, tau_ref=0.01),
        # neuron_type=nengo.Izhikevich(tau_recovery=0.02, coupling=0.2, reset_voltage=-65., reset_recovery=8.),
        neuron_type=nengo.AdaptiveLIF(tau_n=1, inc_n=0.5, tau_rc=0.02, tau_ref=0.002, min_voltage=0, amplitude=1)
    )
    motor_filter = nengo.Alpha(0.05)

    def create_single_module(i,model, motor_node):
        with model:
            pop = nengo.Ensemble(n*2,2, **defaults)
            module_conns = nengo.Connection(pop.neurons,pop.neurons, transform=-W, synapse=0.01)
            # The first neuron is always the measured one -- since we iterate through all matrices, 
            # this should cover every motif
            motor_conn_l = nengo.Connection(pop.neurons[0], motor_node[i*2],synapse=motor_filter)
            motor_conn_r = nengo.Connection(pop.neurons[n], motor_node[i*2+1], synapse=motor_filter)
            spike_probe = nengo.Probe(pop.neurons, 'spikes', synapse=motor_filter)
            val_probe = nengo.Probe(pop,synapse=nengo.Triangle(0.5))
            probes = {'spikes': spike_probe, 'values' : val_probe}
        return pop, probes
    probes = []
    with nengo.Network(seed=0) as model:
        motor_node = nengo.Node(size_in=2*modules, size_out=modules, output=mn.integrate)
        if modules == 1:
            pop, mprobes = create_single_module(0,model,motor_node)
            probes.append(mprobes)
        elif intermodule_matrix is not None:
            im_transform = np.eye(n*2,n*2)
      
            im_transform[:n,:n] = intermodule_matrix
            im_transform[n:,n:] = intermodule_matrix

            pops = []
            prev_module, mprobes = create_single_module(0,model, motor_node)
            pops.append(pops)
            probes.append(mprobes)
            for module in range(1, modules):
                next_module, mprobes = create_single_module(module, model, motor_node)
                nengo.Connection(prev_module.neurons, next_module.neurons, transform=-im_transform, synapse=0.01)
                probes.append(mprobes)
        else:
            raise ValueError('intermodule matrix is not set, but number of modules is >1')
                

    return model,mn, probes
def run_model(module_matrix,lr_matrix,intermodule_matrix=None, modules=1, metadata=None):
    model, mn, probes = generate_nengo_model(module_matrix, lr_matrix, intermodule_matrix, modules)
    with nengo.Simulator(model, progress_bar=False, seed=0) as sim:
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

from scipy import optimize

def sin_fit(x,a, b,c,e):
   
    ret = a * np.sin(b * x + c) + e
    
    return ret
def sin_fit_jac(x,a,b,c,e):
    ret = np.zeros((len(x), 4))
    ret[:,0] = np.sin(b*x+c)
    ret[:,1] = a*x*np.cos(b*x + c)
    ret[:,2] = a*np.cos(b*x + c)
    ret[:,3] = 1
    
    return ret