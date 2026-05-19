"""
CyberSentinel Ensemble Model
Random Forest + XGBoost with hyperparameter tuning
"""

import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')


class OptimizedEnsemble:
    """Random Forest + XGBoost ensemble with GridSearch tuning"""
    
    def __init__(self, tune_hyperparams=True, n_jobs=-1):
        self.tune_hyperparams = tune_hyperparams
        self.n_jobs = n_jobs
        self.rf_model = None
        self.xgb_model = None
        self.ensemble = None
        self.is_fitted = False
        self.best_params_ = {}
        
    def fit(self, X_train, y_train):
        """Train ensemble with optional hyperparameter tuning"""
        print("\n   Training Ensemble Model...")
        start_time = time.time()
        
        if self.tune_hyperparams:
            self._tune_and_fit(X_train, y_train)
        else:
            self._fit_default(X_train, y_train)
        
        train_time = time.time() - start_time
        print(f"   ✅ Training complete in {train_time:.2f}s")
        
        return self
    
    def _tune_and_fit(self, X_train, y_train):
        """Grid search for best hyperparameters"""
        
        # Use subset for faster tuning
        n_tune = min(20000, len(X_train))
        indices = np.random.choice(len(X_train), n_tune, replace=False)
        X_tune, y_tune = X_train[indices], y_train[indices]
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        # Random Forest tuning
        print("   🔵 Tuning Random Forest...")
        rf_params = {
            'n_estimators': [100, 200],
            'max_depth': [15, 20, 25],
        }
        rf_grid = GridSearchCV(
            RandomForestClassifier(random_state=42, n_jobs=self.n_jobs),
            rf_params, cv=cv, scoring='f1', n_jobs=self.n_jobs, verbose=0
        )
        rf_grid.fit(X_tune, y_tune)
        self.rf_model = rf_grid.best_estimator_
        self.best_params_['rf'] = rf_grid.best_params_
        print(f"      Best RF: {rf_grid.best_params_}, F1: {rf_grid.best_score_:.3f}")
        
        # XGBoost tuning
        print("   🟢 Tuning XGBoost...")
        xgb_params = {
            'n_estimators': [100, 200],
            'max_depth': [8, 10, 12],
            'learning_rate': [0.05, 0.1],
        }
        xgb_grid = GridSearchCV(
            XGBClassifier(random_state=42, use_label_encoder=False, 
                         eval_metric='logloss', n_jobs=self.n_jobs),
            xgb_params, cv=cv, scoring='f1', n_jobs=self.n_jobs, verbose=0
        )
        xgb_grid.fit(X_tune, y_tune)
        self.xgb_model = xgb_grid.best_estimator_
        self.best_params_['xgb'] = xgb_grid.best_params_
        print(f"      Best XGB: {xgb_grid.best_params_}, F1: {xgb_grid.best_score_:.3f}")
        
        # Create voting ensemble
        self.ensemble = VotingClassifier(
            estimators=[('rf', self.rf_model), ('xgb', self.xgb_model)],
            voting='soft'
        )
        self.ensemble.fit(X_train, y_train)
        self.is_fitted = True
    
    def _fit_default(self, X_train, y_train):
        """Fit with default parameters"""
        self.rf_model = RandomForestClassifier(
            n_estimators=200, max_depth=20, random_state=42, n_jobs=self.n_jobs
        )
        self.xgb_model = XGBClassifier(
            n_estimators=200, max_depth=10, learning_rate=0.1,
            random_state=42, use_label_encoder=False, eval_metric='logloss'
        )
        self.ensemble = VotingClassifier(
            estimators=[('rf', self.rf_model), ('xgb', self.xgb_model)],
            voting='soft'
        )
        self.ensemble.fit(X_train, y_train)
        self.is_fitted = True
    
    def predict(self, X):
        """Predict labels"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self.ensemble.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self.ensemble.predict_proba(X)