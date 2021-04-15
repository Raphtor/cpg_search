import numpy as np
class MotorNode(object):
    """
    Reads motoneuron output and torque input into the system. 
    
    
    Has the same input and output dimensionality.
    
    The first half of the input corresponds to positive angle changes, the second half is negative angle changes
    The output should be considered inhibitory to both? neurons in the oscillator
    
    """
    def __init__(self, sz,crop=None):
        self.hs = sz
        self.l = slice(0, None,2)
        self.r = slice(1, None,2)
        self.rets = [np.zeros(sz)]
        self.crop = crop
        self._t = None
    def integrate(self, t,x):
        if self._t is None:
            dt = 0.001
        else:
            dt = t - self._t
        self._t = t        
        ret = self.rets[-1] + (-self.rets[-1] + x[self.l] - x[self.r])*dt
        if self.crop is not None:
            ret = np.clip(ret,-self.crop, self.crop)
        self.rets.append(ret)
        return ret