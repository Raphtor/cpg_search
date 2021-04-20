import numpy as np
import networkx
import itertools
from cpg_search.utils import create_adj_matrix
import matplotlib.pyplot as plt
import networkx as nx
from scipy.signal import find_peaks
def draw_graph(ax, adj2,modules=2,m=4,with_labels=True,colors=['tab:blue', 'tab:orange'], node_positions=None, **kwargs):
    
    
    
        
   
    # Create graph
    G = nx.from_numpy_matrix(adj2, create_using=nx.DiGraph)
    # Draw graph
    if node_positions is None:
        node_positions = np.array([
            [-2,1.5], # L1
            [-0.5,0], # L2
            [-2.5,-1], # L3
            [-1,-2], # L4
            [2,1.5], # R1
            [0.5,0], # R2
            [2.5,-1], # R3
            [1,-2] # R4
        ])
    if modules > 1:
        node_positions = np.vstack([node_positions]*modules)
    olabels = ['L{}'.format(i) for i in range(1,m+1)]
    olabels.extend(['R{}'.format(i) for i in range(1,m+1)])
    labels = olabels
    
    for i in range(1,modules):
        node_positions[m*2*i:m*2*(i+1),1] -= 4.5*i
        labels.extend(olabels)
    # print(node_positions)
    nodecolors = list(itertools.chain(*[[colors[i]]*m*2 for i in range(modules)]))
    labels = dict(zip(range(m*2*modules),labels))
#     nodecolors = [colors[0]]*8 + [colors[1]]*8
    nx.draw_networkx(G, pos=node_positions,ax=ax, with_labels=with_labels, labels=labels, node_color=nodecolors , **kwargs)
    ax.set_yticks(-np.arange(modules)*4.5)
    ax.set_yticklabels(['M{}'.format(i) for i in range(1,modules+1)])
    ax.set_xticks([])

    
    
def draw_adj(ax, adj2, **kwargs):
    
    
    im = ax.matshow(adj2, aspect='equal', **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel('Presynaptic')
    ax.set_xlabel('Postsynaptic')
#     ax.set_xticks(np.arange(-0.5,15,4), minor=True)
#     ax.set_yticks(np.arange(-0.5,15,4), minor=True)
#     ax.set_xticks([1.75,5.75,9.75,13.75])
#     ax.set_yticks([1.75,5.75,9.75,13.75])
#     ax.set_xticklabels(['M1L' ,'M1R','M2L','M2R'])
#     ax.set_yticklabels(['M1L' ,'M1R','M2L','M2R'])
#     ax.axvline([7.5],color='r', ls='--',lw=2)
#     ax.axhline([7.5],color='r', ls='--', lw=2)
# #     ax.grid(which='minor', color='r', linestyle='--', linewidth=1)
#     ax.grid(which='minor', color='w', linestyle='-', linewidth=1)

    return im
def plot_limit_cycle(ret,ax,tslc=slice(0,None),N=10):
    x = ret['motor_values'][tslc,0]
    y = ret['motor_values'][tslc,1]
    _i = 0
    for i in np.linspace(0,x.shape[0],N+1,dtype=int):
        alpha = alpha=float(i)/x.shape[0]
        ax.plot(x[_i:i],y[_i:i],alpha=alpha, color='tab:blue')
        _i = i
        print(_i,i,alpha)
    ax.set_xlabel('M1')
    ax.set_ylabel('M2')
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.title.set_text('Limit cycle')
    ax.set_aspect('equal')


def plot_spikes(axs, sim, probes1,it,begin_time,insert_time=None):
    T = sim.trange()
    tslc = np.where(T>begin_time)[0]
    spike1 = [sim.data[probes1[i]['spikes']] for i in range(len(probes1))]

    labels = ['M{}{}{}']
    labels = []

    ims = []
    for i in it:
        ims.append(spike1[i][np.ix_(tslc,[0,4])])
        labels.extend(['M%iL1'% (i+1),'M%iR1'%(i+1)])
    ims = np.hstack(ims).T
    bt = T[tslc][0]
    et = T[tslc][-1]
    bi = -0.5
    ei = ims.shape[0]-0.5
    im = axs.matshow(ims,aspect='auto',extent=[bt,et,ei,bi])
    # im = axs[0].matshow(ims,aspect='auto')

    axs.set_xlabel('Time (s)')
    axs.set_ylabel('Neuron')
    axs.set_yticks(range(len(it)*2))
    axs.xaxis.tick_bottom()
    axs.set_yticklabels(labels)
    cbar = plt.colorbar(im,ax=axs)
    cbar.ax.title.set_text('Hz')
    labels = ['M1L1', 'M1R1','M2L1','M2R1','M3L1','M3R1']
    if insert_time is not None:
        axs.axvline(insert_time, ls='--', color='r')
    plt.tight_layout()
def plot_output_compliance(ax, mn,begin_time,sim,it,insert_time=None,before_ls='-', after_ls='-',show_peaks=False):
    T = sim.trange()
    tslc = np.where(T>begin_time)

    out = np.array(mn.rets[1:])
    if insert_time is None:
        insert_time = T[-1]
        
    in_slc = np.where(T>insert_time)[0]
    out_slc = np.where(np.logical_and(begin_time<T,T<insert_time))[0]
    tslc = np.where(T>begin_time)[0]
    lcolors = ['tab:green', 'tab:orange','tab:purple','tab:brown','tab:cyan']
    sc = 5
    for i in it:
        if i == 0:
            ax.plot(T[out_slc], out[out_slc,i]/sc,color=lcolors[i],ls=before_ls)
            ax.plot(T[in_slc], out[in_slc,i]/sc,color=lcolors[i],ls=after_ls)
        else:
            ax.plot(T[tslc], out[tslc,i]/sc,color=lcolors[i],alpha=0.3)
    if insert_time != T[-1]:
        ax.axvline(insert_time,color='r', ls='--')
    if show_peaks:
        peaks,_ = find_peaks(out[in_slc,0])
        npeaks,_ = find_peaks(-out[in_slc,0])
        peaks = peaks[:2]
        npeaks = npeaks[:2]
        ax.scatter(T[in_slc][peaks], out[in_slc,0][peaks]/sc,marker='x')
        ax.scatter(T[in_slc][npeaks], out[in_slc,0][npeaks]/sc,marker='x')
    ax.set_xlabel('Time')
    ax.set_ylabel('Motor value')

def plot_before_after_bar(ax,mn,begin_time, sim,insert_time,it,baseline_output):
    T = sim.trange()
    tslc = np.where(T>begin_time)

    out = np.array(mn.rets[1:])
    if insert_time is None:
        insert_time = T[-1]
        
    in_slc = np.where(T>insert_time)[0]
    out_slc = np.where(np.logical_and(begin_time<T,T<insert_time))[0]
    tslc = np.where(T>begin_time)[0]
    baseline_amps = []
    after_amps = []
    sc=5
    def get_amp(out):
        
        
        peaks,_ = find_peaks(out)
        npeaks,_ = find_peaks(-out)
        
        return np.mean(out[peaks[:2]]-out[npeaks[:2]]/2)
        
    for i in it:
        baseline_amps.append(get_amp(baseline_output[in_slc,i])/sc)


        
        after_amps.append(get_amp(out[in_slc,i])/sc)
    M_r = it
    ax.bar(M_r*2, baseline_amps)
    # ax.bar(M_r*2+1, bbefore_amps)
    ax.bar(M_r*2+1, after_amps)
    ax.set_xticks(M_r*2+0.5,minor=True)
    ax.set_xticks(M_r*2-0.5)
    ax.tick_params(axis='x',which='minor',bottom=False)
    ax.tick_params(axis='x',which='major',labelbottom=False)
    ax.set_xticklabels(M_r,minor=True)
    ax.legend(['Before','After'])
    ax.set_xlabel('Motor #')
    ax.set_ylabel('Amplitude')

def plot_input_currents(ax,mn, sim,begin_time,inp_probes,insert_time):
    T = sim.trange()
    tslc = np.where(T>begin_time)[0]

    ax.plot(T[tslc], sim.data[inp_probes[0]][tslc,0]+5)
    ax.plot(T[tslc], sim.data[inp_probes[0]][tslc,4]+5)
    ax.legend(['L1','R1'])
    ax.set_yscale('symlog')
    ax.axvline(insert_time, ls='--', color='r')
    ax.set_ylabel('Input current (u)')
    ax.set_xlabel('Time (s)')
