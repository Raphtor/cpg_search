import numpy as np
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

def create_adj_matrix(mm,lr,imm=None,modules=1):
    """
    Makes a full size adjacency matrix out of a module matrix, left-right matrix and an intermodule matrix.
    params:
    -------
    mm: module matrix
    lr: left-right matrix (same size as mm)
    imm: intermodule matrix (same size as mm)
    modules: number of modules connected by intermodule matrix
    """
    n = len(mm)
    nn = n*2
    if imm is not None:
        imadj = np.zeros((nn,nn))
        imadj[n:,n:] = imm
        imadj[:n,:n] = imm
    else:
        assert modules == 1,'Modules > 1, but no intermodule matrix given'
    adj = np.zeros((nn,nn))
    adj[:n, :n] = mm
    adj[n:nn, n:nn] = mm
    adj[n:nn,:n] = lr
    adj[:n,n:nn] = lr
    
    
    
    
    
    adj2 = np.zeros((nn*modules,nn*modules))
    adj2[:nn,:nn] = adj
    
    for i in range(1,modules):
        bi = (i-1)*nn
        mi = nn*i
        ei = nn*(i+1)
        adj2[bi:mi,mi:ei] = imadj
        adj2[mi:ei, mi:ei] = adj
    return adj2


def eval_cfg(cfg, context):
    for key,val in cfg.items():
        if isinstance(val,dict):
            cfg[key] = eval_cfg(val,context)
        else:
            cfg[key] = eval(str(val),context) 
    return cfg