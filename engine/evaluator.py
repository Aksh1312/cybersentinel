"""
CyberSentinel Evaluation Engine
Professional metric calculation and reporting
"""

import numpy as np
import time
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score
)


class ModelEvaluator:
    """Professional model evaluation and reporting"""
    
    def __init__(self):
        self.metrics = {}
        self.report = None
        self.confusion_mat = None
        
    def evaluate(self, y_true, y_pred, y_proba=None):
        """Calculate all metrics"""
        
        self.metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
        }
        
        if y_proba is not None:
            self.metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
        
        self.confusion_mat = confusion_matrix(y_true, y_pred)
        self.report = classification_report(y_true, y_pred, 
                                            target_names=['Normal', 'Attack'])
        
        return self.metrics
    
    def print_report(self, test_size, train_time, infer_time):
        """Print formatted evaluation report"""
        
        m = self.metrics
        cm = self.confusion_mat
        
        print(f"\n{'='*60}")
        print("📊 FINAL RESULTS ON UNSW-NB15 DATASET")
        print(f"{'='*60}")
        print(f"   Test Samples: {test_size:,}")
        print(f"   Accuracy:     {m['accuracy']*100:.2f}%")
        print(f"   Precision:    {m['precision']*100:.2f}%")
        print(f"   Recall:       {m['recall']*100:.2f}%")
        print(f"   F1 Score:     {m['f1']*100:.2f}%")
        
        if 'roc_auc' in m:
            print(f"   ROC-AUC:      {m['roc_auc']*100:.2f}%")
        
        print(f"\n   ⚡ Training: {train_time:.2f}s | Inference: {infer_time:.1f}ms")
        
        print(f"\n📝 Classification Report:")
        print(self.report)
        
        print(f"📋 Confusion Matrix:")
        print(f"                    Predicted Normal  Predicted Attack")
        print(f"   Actual Normal         {cm[0][0]:>6}          {cm[0][1]:>6}")
        print(f"   Actual Attack         {cm[1][0]:>6}          {cm[1][1]:>6}")
        
        return self
    
    def get_resume_metrics(self):
        """Return metrics formatted for resume"""
        m = self.metrics
        return {
            'Accuracy': f"{m['accuracy']*100:.1f}%",
            'Precision': f"{m['precision']*100:.1f}%",
            'Recall': f"{m['recall']*100:.1f}%",
            'F1 Score': f"{m['f1']*100:.1f}%",
        }