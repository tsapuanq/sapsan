from ._regression import (
    r2_score,
    mean_squared_error,
    MSE,
    mean_absolute_error,
    MAE,
    root_mean_squared_error,
    RMSE,
    mean_absolute_percentage_error,
    MAPE
)

from ._classification import (
    accuracy,
    recall,
    precision
)


__all__ = [
    'r2_score',
    'mean_squared_error', 'MSE',
    'mean_absolute_error', 'MAE',
    'root_mean_squared_error', 'RMSE',
    'mean_absolute_percentage_error', 'MAPE',
    'accuracy', 'recall', 'precision'
]
