# train.py  – baseline Linear Regression
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

def load_data():
    """Fetch Boston Housing data from OpenML."""
    X, y = fetch_openml("Boston", version=1, as_frame=False, return_X_y=True)
    return X.astype(np.float32), y.astype(np.float32)

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(f"R² on test: {r2_score(y_test, preds):.3f}")

    # save model
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/linear_regression.joblib")
    print("Model saved to models/linear_regression.joblib")

if __name__ == "__main__":
    main()
