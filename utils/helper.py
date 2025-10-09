import numpy as np



# function to convert inputs to numpy arrays and check for shape mismatch
def to_numpy(y_true, y_pred):

    # to be sure inputs are numpy arrays
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    # Check for shape mismatch
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred do not match.")
    
    return y_true, y_pred
    