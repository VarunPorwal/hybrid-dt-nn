import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from hybrid_dt_nn import HybridTreeRegressor

def test_fixed_architecture():
    print("\n--- Testing Fixed Architecture (use_hpo=False) ---")
    X, y = make_regression(n_samples=500, n_features=10, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Using very small nn_epochs and tiny trees for fast testing
    model = HybridTreeRegressor(
        dt_max_depth=3, 
        dt_min_samples_leaf=20, 
        nn_min_samples=20, 
        use_hpo=False, 
        nn_epochs=5,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    score = r2_score(y_test, preds)
    print(f"Fixed Architecture R2 Score: {score:.4f}")
    assert len(preds) == len(X_test), "Output shape mismatch!"
    print("Fixed Architecture Test PASSED.\n")

def test_hpo_architecture():
    print("\n--- Testing Optuna HPO Architecture (use_hpo=True) ---")
    X, y = make_regression(n_samples=500, n_features=10, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Using very few trials for fast testing
    model = HybridTreeRegressor(
        dt_max_depth=2, 
        dt_min_samples_leaf=30, 
        nn_min_samples=30, 
        use_hpo=True, 
        hpo_trials=3,
        nn_epochs=5,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    score = r2_score(y_test, preds)
    print(f"HPO Architecture R2 Score: {score:.4f}")
    assert len(preds) == len(X_test), "Output shape mismatch!"
    print("HPO Architecture Test PASSED.\n")

if __name__ == "__main__":
    print("Starting tests for HybridTreeRegressor library...")
    test_fixed_architecture()
    test_hpo_architecture()
    print("All tests completed successfully!")
