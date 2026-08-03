import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from hybrid_dt_nn import HybridTreeRegressor

def run_real_world_test():
    print("="*50)
    print("  Testing HybridTreeRegressor on REAL Data")
    print("  Dataset: Diabetes Progression")
    print("="*50)
    
    # 1. Load a real dataset
    print("\n1. Loading dataset...")
    data = load_diabetes()
    X, y = data.data, data.target
    print(f"   Loaded {X.shape[0]} rows and {X.shape[1]} features.")
    
    # 2. Split and Scale
    print("2. Splitting and Scaling data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 3. Initialize our custom library
    print("\n3. Initializing HybridTreeRegressor...")
    # We will use realistic parameters for a real dataset
    model = HybridTreeRegressor(
        dt_max_depth=3,              # 3 levels deep
        dt_min_samples_leaf=20,      # At least 20 per leaf
        nn_min_samples=20,           # Only train NN if leaf has 20+
        use_hpo=False,               # Keep it true if want to use Optuna
        nn_epochs=50,                # Train NN for 50 epochs
        verbose=1
    )
    
    # 4. Train
    print("\n4. Training model (This might take a minute)...")
    model.fit(X_train, y_train)
    
    # 5. Predict and Evaluate
    print("\n5. Evaluating on Test Set...")
    preds = model.predict(X_test)
    
    score = r2_score(y_test, preds)
    print("\n" + "="*50)
    print(f"  FINAL REAL-WORLD R² SCORE: {score:.4f}")
    print("="*50)

if __name__ == "__main__":
    run_real_world_test()
