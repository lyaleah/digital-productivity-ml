from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from config import RANDOM_SEED


def get_models():
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(),
        "RandomForest": RandomForestRegressor(
            random_state=RANDOM_SEED
        ),
        "XGBoost": XGBRegressor(
            random_state=RANDOM_SEED,
            verbosity=0
        )
    }
