#---------------------------------------------#
        #All regression metrics by Sapsan#


# ------ R2 score ------
def r2_score(y_true, y_pred, round_digit=None):

    # just to be sure that inputs are numpy arrays
    y_true, y_pred = to_numpy(y_true, y_pred)

    # calculating r2 score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot

    # rounding if round_digit is provided
    if round_digit is not None:
        return round(r2, round_digit)
    return r2



# ------ Mean Squared Error ------
def mean_squared_error(y_true, y_pred, round_digit=None):
    
    # just to be sure that inputs are numpy arrays
    y_true, y_pred = to_numpy(y_true, y_pred)
    
    # calculating MSE
    mse = np.mean((y_true - y_pred) ** 2)

    # rounding if round_digit is provided
    if round_digit is not None:
        return round(mse, round_digit) 
    
    return mse

MSE = mean_squared_error

# ------ Mean Absolute Error ------
def mean_absolute_error(y_true, y_pred, round_digit=None):

    # just to be sure that inputs are numpy arrays
    y_true, y_pred = to_numpy(y_true, y_pred)

    # calculating MAE
    mae = np.mean(np.abs(y_true - y_pred))

    # rounding if round_digit is provided
    if round_digit is not None:
        return round(mae, round_digit)
    return mae

MAE = mean_absolute_error

# ------ Root Mean Squared Error ------
def root_mean_squared_error(y_true, y_pred, round_digit=None):

    # calculating RMSE using MSE function defined above
    rmse = np.sqrt(MSE(y_true, y_pred))

    # rounding if round_digit is provided
    if round_digit is not None:
        return round(rmse, round_digit)
    return rmse


RMSE = root_mean_squared_error

# ------ Mean Absolute Percentage Error ------
def mean_absolute_percentage_error(y_true, y_pred, round_digit=None):

    # just to be sure that inputs are numpy arrays
    y_true, y_pred = to_numpy(y_true, y_pred)
    
    # calculating MAPE
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # rounding if round_digit is provided
    if round_digit is not None:
        return round(mape, round_digit)
    return mape


MAPE = mean_absolute_percentage_error



# # ------ Mean Absolute Percentage Error ------
# def mean_squared_least_error(y_true, y_pred, round_digit=None):

#     # just to be sure that inputs are numpy arrays
#     y_true, y_pred = to_numpy(y_true, y_pred)
    
#     # calculating MAPE
#     mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

#     # rounding if round_digit is provided
#     if round_digit is not None:
#         return round(mape, round_digit)
#     return mape

# MSLE = mean_squared_least_error


# ----- Imports ------
import numpy as np
from ..utils.helper import to_numpy