"""
CyberSentinel AI - Main Application
Production-ready intrusion detection system using UNSW-NB15 dataset
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from core.real_data_loader import load_unsw_data, OptimizedPreprocessor
from models.ensemble_model import OptimizedEnsemble
from engine.evaluator import ModelEvaluator

class CyberSentinelSystem:
    """Main CyberSentinel intrusion detection system"""
    
    def __init__(self, fast_mode=False):
        self.fast_mode = fast_mode
        self.train_df = None
        self.test_df = None
        self.preprocessor = None
        self.ensemble = None
        self.evaluator = None
        self.selected_features = None
        self.X_test = None
        self.y_test = None
        self.is_trained = False
        
    def load_data(self):
        """Load UNSW-NB15 dataset"""
        print("\n" + "=" * 60)
        print("📥 Loading UNSW-NB15 Dataset")
        print("=" * 60)
        
        self.train_df, self.test_df = load_unsw_data()
        
        if self.train_df is None:
            print("❌ Dataset not found!")
            print("Download from: https://research.unsw.edu.au/projects/unsw-nb15-dataset")
            print("Place CSV files in the 'data/' folder")
            return False
        
        if self.fast_mode:
            print("⚡ Fast mode: sampling 30,000 training records...")
            self.train_df = self.train_df.sample(n=30000, random_state=42)
            self.test_df = self.test_df.sample(n=15000, random_state=42)
        
        return True
    
    def preprocess(self):
        """Preprocess data with feature selection"""
        print("\n" + "=" * 60)
        print("⚙️  Preprocessing Pipeline")
        print("=" * 60)
        
        self.preprocessor = OptimizedPreprocessor(n_features=75)
        X_train, X_test, y_train, y_test, self.selected_features = \
            self.preprocessor.fit_transform(self.train_df, self.test_df)
        
        self.X_test = X_test
        self.y_test = y_test
        
        return X_train, y_train
    
    def train(self, X_train, y_train):
        """Train optimized ensemble model"""
        print("\n" + "=" * 60)
        print("🧠 Training Ensemble Model")
        print("=" * 60)
        
        print("Model: Random Forest + XGBoost Voting Ensemble")
        print("Tuning: GridSearchCV with 3-fold stratified CV")
        print(f"Training samples: {len(X_train):,}")
        print(f"Features: {X_train.shape[1]}")
        print(f"Attack ratio: {sum(y_train)/len(y_train)*100:.1f}%")
        
        start_time = time.time()
        self.ensemble = OptimizedEnsemble(tune_hyperparams=not self.fast_mode, n_jobs=-1)
        self.ensemble.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        self.is_trained = True
        
        print(f"\n✅ Training complete in {train_time:.1f}s")
        
        if self.ensemble.best_params_:
            print(f"   Best RF params: {self.ensemble.best_params_.get('rf', 'N/A')}")
            print(f"   Best XGB params: {self.ensemble.best_params_.get('xgb', 'N/A')}")
    
    def evaluate(self):
        """Evaluate on test set"""
        print("\n" + "=" * 60)
        print("📊 Model Evaluation")
        print("=" * 60)
        
        if not self.is_trained:
            print("❌ Model not trained yet!")
            return None
        
        print(f"Test samples: {len(self.X_test):,}")
        print(f"Test attacks: {sum(self.y_test):,}")
        
        start_time = time.time()
        y_pred = self.ensemble.predict(self.X_test)
        y_proba = self.ensemble.predict_proba(self.X_test)
        infer_time = (time.time() - start_time) * 1000
        
        self.evaluator = ModelEvaluator()
        self.evaluator.evaluate(self.y_test, y_pred, y_proba)
        
        self.evaluator.print_report(
            test_size=len(self.X_test),
            train_time=0,  # Will be updated
            infer_time=infer_time
        )
        
        return self.evaluator.metrics
    
    def predict_single(self, data):
        """Predict on new data"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        processed = self.preprocessor.transform_single(data)
        prediction = self.ensemble.predict(processed)
        probability = self.ensemble.predict_proba(processed)
        
        return prediction, probability
    
    def get_summary(self):
        """Get system summary for dashboard"""
        if not self.is_trained or self.evaluator is None:
            return None
        
        metrics = self.evaluator.get_resume_metrics()
        
        return {
            'dataset': 'UNSW-NB15',
            'train_records': len(self.train_df),
            'test_records': len(self.test_df),
            'features_used': len(self.selected_features) if self.selected_features is not None else 75,
            'total_features': 194,
            'attack_types': 9,
            'metrics': metrics,
            'best_params': self.ensemble.best_params_ if self.ensemble else {},
            'confusion_matrix': self.evaluator.confusion_mat.tolist() if self.evaluator else None,
            'classification_report': self.evaluator.report if self.evaluator else None,
        }
    
    def run_full_pipeline(self):
        """Run complete pipeline from load to evaluate"""
        print("\n" + "=" * 60)
        print("🛡️  CyberSentinel AI - Intrusion Detection System")
        print("=" * 60)
        print(f"Fast Mode: {self.fast_mode}")
        
        # Step 1: Load
        if not self.load_data():
            return None
        
        # Step 2: Preprocess
        X_train, y_train = self.preprocess()
        
        # Step 3: Train
        self.train(X_train, y_train)
        
        # Step 4: Evaluate
        metrics = self.evaluate()
        
        # Step 5: Summary
        if metrics:
            print("\n" + "=" * 60)
            print("📝 FINAL SUMMARY")
            print("=" * 60)
            print(f"""
   Dataset:      UNSW-NB15 (public benchmark)
   Model:        Random Forest + XGBoost Ensemble
   Accuracy:     {metrics['accuracy']*100:.1f}%
   Precision:    {metrics['precision']*100:.1f}%
   Recall:       {metrics['recall']*100:.1f}%
   F1 Score:     {metrics['f1']*100:.1f}%
   ROC-AUC:      {metrics.get('roc_auc', 0)*100:.1f}%
            """)
        
        return metrics


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='CyberSentinel AI - Intrusion Detection')
    parser.add_argument('--fast', action='store_true', 
                       help='Fast mode (smaller sample, no tuning)')
    parser.add_argument('--predict', type=str, 
                       help='Path to CSV file for prediction')
    
    args = parser.parse_args()
    
    # Initialize system
    system = CyberSentinelSystem(fast_mode=args.fast)
    
    if args.predict:
        # Prediction mode
        if not system.load_data():
            return
        
        X_train, y_train = system.preprocess()
        system.train(X_train, y_train)
        
        # Load prediction data
        pred_df = pd.read_csv(args.predict)
        predictions, probabilities = system.predict_single(pred_df)
        
        print("\n📊 Predictions:")
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            label = "ATTACK" if pred == 1 else "NORMAL"
            confidence = max(prob) * 100
            print(f"   Record {i}: {label} (confidence: {confidence:.1f}%)")
    else:
        # Full pipeline mode
        system.run_full_pipeline()


if __name__ == "__main__":
    main()