import importlib
import math
import numpy as np

def test_import_module_and_public_api():
    mod = importlib.import_module("sapsan.metrics")
    assert mod is not None

    required = [
        "mean_squared_error","MSE",
        "root_mean_squared_error","RMSE",
        "mean_absolute_error","MAE",
        "r2_score","mean_absolute_percentage_error","MAPE"
    ]
    for name in required:
        assert hasattr(mod, name), f"missing {name} in sapsan.metrics"

def test_star_import_works_and_all_export():

    g = {}
    exec("from sapsan.metrics import *", g)
    for name in ("MSE", "RMSE", "mean_squared_error", "mean_absolute_error"):
        assert name in g, f"{name} not found after star import"

def test_functions_callable_and_math_relations():
    from sapsan.metrics import MSE, RMSE, mean_squared_error, root_mean_squared_error, mean_absolute_error

    y = np.array([1.0, 2.0, 3.0])
    yp = np.array([1.1, 1.9, 2.8])

    mse_val = MSE(y, yp)
    assert isinstance(mse_val, float)

    assert math.isclose(mean_squared_error(y, yp), mse_val, rel_tol=1e-12)

    # RMSE == sqrt(MSE)
    rmse_val = RMSE(y, yp)
    assert math.isclose(rmse_val, math.sqrt(mse_val), rel_tol=1e-12)

    # MAE works
    mae_val = mean_absolute_error(y, yp)
    assert isinstance(mae_val, float)

def test_shape_mismatch_raises_value_error():
    from sapsan.metrics import MSE
    with np.testing.assert_raises(ValueError):
        MSE([1,2,3], [1,2])

def test_production_like_pipeline_usage():
    from sapsan import metrics
    import logging
    logging.basicConfig(level=logging.INFO)
    y = np.random.RandomState(0).randn(100)
    yp = y + np.random.RandomState(1).normal(scale=0.1, size=100)

    mse = metrics.MSE(y, yp)
    rmse = metrics.RMSE(y, yp)
    r2 = metrics.r2_score(y, yp)

    assert mse >= 0.0
    assert rmse >= 0.0
    assert isinstance(r2, float)