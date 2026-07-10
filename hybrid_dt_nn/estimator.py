import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.tree import DecisionTreeRegressor
import tensorflow as tf
from tensorflow import keras
import optuna

# Disable optuna output to keep it clean unless user wants verbose
optuna.logging.set_verbosity(optuna.logging.WARNING)

class HybridTreeRegressor(BaseEstimator, RegressorMixin):
    """
    A Hybrid Decision Tree + Neural Network Regressor.
    
    This model first fits a Decision Tree to partition the data into leaves.
    For each leaf that contains at least `nn_min_samples`, it trains a specialized
    Keras Neural Network. If a leaf has fewer samples, it falls back to the DT's prediction.
    """
    def __init__(self, 
                 dt_max_depth=5, 
                 dt_min_samples_leaf=50, 
                 nn_min_samples=50,
                 use_hpo=False,
                 hpo_trials=20,
                 nn_hidden_layers=2,
                 nn_units=64,
                 nn_epochs=30,
                 nn_batch_size=32,
                 random_state=None,
                 verbose=0):
        """
        Parameters:
        -----------
        dt_max_depth : int, default=5
            Maximum depth of the Decision Tree router.
        dt_min_samples_leaf : int, default=50
            Minimum samples required to be at a leaf node in the DT.
        nn_min_samples : int, default=50
            Minimum samples required in a leaf to train a Neural Network. If fewer, falls back to DT.
        use_hpo : bool, default=False
            If True, uses Optuna to optimize the Neural Network architecture per leaf.
            If False, uses the fixed architecture defined by nn_hidden_layers and nn_units.
        hpo_trials : int, default=20
            Number of Optuna trials to run per leaf (only used if use_hpo=True).
        nn_hidden_layers : int, default=2
            Number of hidden layers in the NN (only used if use_hpo=False).
        nn_units : int, default=64
            Number of units per hidden layer in the NN (only used if use_hpo=False).
        nn_epochs : int, default=30
            Number of epochs to train each NN.
        nn_batch_size : int, default=32
            Batch size for NN training.
        random_state : int, default=None
            Random state for the Decision Tree and Optuna.
        verbose : int, default=0
            Verbosity level. 0=silent, 1=progress.
        """
        self.dt_max_depth = dt_max_depth
        self.dt_min_samples_leaf = dt_min_samples_leaf
        self.nn_min_samples = nn_min_samples
        self.use_hpo = use_hpo
        self.hpo_trials = hpo_trials
        self.nn_hidden_layers = nn_hidden_layers
        self.nn_units = nn_units
        self.nn_epochs = nn_epochs
        self.nn_batch_size = nn_batch_size
        self.random_state = random_state
        self.verbose = verbose
        
        # Internal state
        self.dt_ = None
        self.leaf_models_ = {}

    def fit(self, X, y):
        """Fit the model to the training data."""
        # Ensure numpy arrays
        X = np.asarray(X)
        y = np.asarray(y)
        
        if self.verbose > 0:
            print("1. Fitting Decision Tree...")
            
        self.dt_ = DecisionTreeRegressor(
            max_depth=self.dt_max_depth,
            min_samples_leaf=self.dt_min_samples_leaf,
            random_state=self.random_state
        )
        self.dt_.fit(X, y)
        
        # Get leaf ids for each training sample
        leaf_ids = self.dt_.apply(X)
        unique_leaves = np.unique(leaf_ids)
        
        if self.verbose > 0:
            print(f"Decision Tree built with {len(unique_leaves)} leaves.")
            print(f"2. Training Neural Networks (use_hpo={self.use_hpo})...")
            
        for leaf_id in unique_leaves:
            idx = np.where(leaf_ids == leaf_id)[0]
            n_samples = len(idx)
            
            if n_samples >= self.nn_min_samples:
                X_leaf = X[idx]
                y_leaf = y[idx]
                
                if self.verbose > 0:
                    print(f"  -> Leaf {leaf_id} ({n_samples} samples): Training NN...")
                
                if self.use_hpo:
                    nn = self._train_with_hpo(X_leaf, y_leaf)
                else:
                    nn = self._build_fixed_nn(X.shape[1])
                    nn.fit(X_leaf, y_leaf, epochs=self.nn_epochs, batch_size=self.nn_batch_size, verbose=0)
                
                self.leaf_models_[leaf_id] = nn
            else:
                if self.verbose > 0:
                    print(f"  -> Leaf {leaf_id} ({n_samples} samples): Skipping NN (fallback to DT).")
                    
        if self.verbose > 0:
            print("Training complete!")
            
        return self

    def predict(self, X):
        """Predict target values for X."""
        X = np.asarray(X)
        
        # Base predictions from DT
        dt_preds = self.dt_.predict(X)
        
        # Get leaf assignments
        leaf_ids = self.dt_.apply(X)
        
        final_preds = np.zeros(len(X))
        
        for i in range(len(X)):
            leaf = leaf_ids[i]
            if leaf in self.leaf_models_:
                # Route to specific Neural Network
                nn = self.leaf_models_[leaf]
                # reshape for keras inference
                final_preds[i] = nn.predict(X[i].reshape(1, -1), verbose=0)[0][0]
            else:
                # Fallback
                final_preds[i] = dt_preds[i]
                
        return final_preds
        
    def _build_fixed_nn(self, input_dim, hidden_layers=None, units=None, activation='relu', optimizer='adam'):
        """Builds a basic Keras Sequential model."""
        if hidden_layers is None:
            hidden_layers = self.nn_hidden_layers
        if units is None:
            units = self.nn_units
            
        model = keras.Sequential()
        model.add(keras.layers.InputLayer(input_shape=(input_dim,)))
        
        for _ in range(hidden_layers):
            model.add(keras.layers.Dense(units, activation=activation))
            
        model.add(keras.layers.Dense(1)) # Regression output
        model.compile(optimizer=optimizer, loss='mse')
        return model

    def _train_with_hpo(self, X_leaf, y_leaf):
        """Runs Optuna HPO to find the best NN architecture for this specific leaf."""
        from sklearn.metrics import mean_squared_error
        
        def objective(trial):
            # Hyperparameter search space
            optimizer = trial.suggest_categorical("optimizer", ["adam", "rmsprop"])
            activation = trial.suggest_categorical("activation", ["relu", "tanh", "elu"])
            hidden_layers = trial.suggest_int("hidden_layers", 1, 3)
            units = trial.suggest_categorical("units", [16, 32, 64, 128])
            
            model = self._build_fixed_nn(
                input_dim=X_leaf.shape[1], 
                hidden_layers=hidden_layers, 
                units=units, 
                activation=activation, 
                optimizer=optimizer
            )
            
            # Simple validation split for tuning (80/20)
            split_idx = int(len(X_leaf) * 0.8)
            X_t, X_v = X_leaf[:split_idx], X_leaf[split_idx:]
            y_t, y_v = y_leaf[:split_idx], y_leaf[split_idx:]
            
            # If leaf is too small for split, just use training error
            if len(X_v) == 0:
                X_t, y_t = X_leaf, y_leaf
                X_v, y_v = X_leaf, y_leaf
                
            model.fit(X_t, y_t, epochs=self.nn_epochs, batch_size=self.nn_batch_size, verbose=0)
            preds = model.predict(X_v, verbose=0).flatten()
            
            mse = mean_squared_error(y_v, preds)
            return mse
            
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.hpo_trials)
        
        best_params = study.best_trial.params
        
        # Re-train on full leaf data using best parameters
        final_model = self._build_fixed_nn(
            input_dim=X_leaf.shape[1],
            hidden_layers=best_params["hidden_layers"],
            units=best_params["units"],
            activation=best_params["activation"],
            optimizer=best_params["optimizer"]
        )
        final_model.fit(X_leaf, y_leaf, epochs=self.nn_epochs, batch_size=self.nn_batch_size, verbose=0)
        
        return final_model
