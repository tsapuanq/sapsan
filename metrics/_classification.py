#---------------------------------------------#
        #All classification metrics by Sapsan#

def accuracy(y_true, y_pred, *, normalize=True, sample_weight=None):
    y_true, y_pred = to_numpy(y_true, y_pred)
    
    if y_true.ndim > 1:
        correct = np.all(y_true == y_pred, axis=1).astype(float)

    else:
        correct = (y_true == y_pred).astype(float)

    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=float)

        if w.ndim != 1:
            raise ValueError('sample_weight must be 1D array')
        
        if correct.shape[0] != w.shape[0]:
            raise ValueError('sample_weight must have shape (n_samples, )')
        
        weighted = (correct * w).sum()
        if normalize:
            total_w = w.sum()
            if total_w == 0:
                raise ValueError('Must be > 0')
            else:
                return float(weighted / total_w)
        
        else:
            return float(weighted)
        
    else:
        return float(correct.mean()) if normalize else float(correct.sum())
    

def recall():
    pass

def precision():
    pass

def s(): pass


from ..utils.helper import to_numpy
import numpy as np