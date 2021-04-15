import numpy as np
import networkx
import itertools
from cpg_search.utils import create_adj_matrix
import networkx as nx
def draw_graph(ax, adj2,modules=2,colors=['tab:blue', 'tab:orange']):
    
    
    
        
   
    # Create graph
    G = nx.from_numpy_matrix(adj2, create_using=nx.DiGraph)
    # Draw graph
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
    node_positions = np.vstack([node_positions]*modules)
    olabels = ['L1','L2','L3','L4','R1','R2','R3','R4']
    labels = olabels
    
    for i in range(1,modules):
        node_positions[8*i:8*(i+1),1] -= 4*i
        labels.extend(olabels)
    print(node_positions.shape)
    nodecolors = list(itertools.chain(*[[colors[i]]*8 for i in range(modules)]))
    labels = dict(zip(range(8*modules),labels))
#     nodecolors = [colors[0]]*8 + [colors[1]]*8
    nx.draw_networkx(G, pos=node_positions,ax=ax, with_labels=True, labels=labels, node_color=nodecolors )
    
    
def draw_adj(ax, adj2):
    
    
    im = ax.matshow(adj2, aspect='equal')
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