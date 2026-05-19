"""
Real Dataset Loader for UNSW-NB15
Optimized preprocessing pipeline with feature selection
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import os


def load_unsw_data():
    """Load UNSW-NB15 training and testing sets"""
    
    train_path = 'data/UNSW_NB15_training-set.csv'
    test_path = 'data/UNSW_NB15_testing-set.csv'
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("❌ Dataset files not found in data/ folder")
        print("Download from: https://research.unsw.edu.au/projects/unsw-nb15-dataset")
        return None, None
    
    print("Loading UNSW-NB15 dataset...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"✅ Training set: {train_df.shape}")
    print(f"✅ Test set: {test_df.shape}")
    
    # Show attack distribution
    if 'attack_cat' in train_df.columns:
        print("\n📊 Attack types in training:")
        print(train_df['attack_cat'].value_counts())
    
    return train_df, test_df


class OptimizedPreprocessor:
    """
    Complete preprocessing pipeline:
    1. Drop ID column
    2. Encode categorical features
    3. Align train/test columns
    4. Scale features
    5. Select best features
    """
    
    def __init__(self, n_features=75):
        self.n_features = n_features
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(mutual_info_classif, k=n_features)
        self.train_columns = None
        self.is_fitted = False
        
    def fit_transform(self, train_df, test_df):
        """Fit preprocessing on training data, transform both"""
        
        # Drop ID column if exists
        if 'id' in train_df.columns:
            train_df = train_df.drop('id', axis=1)
            test_df = test_df.drop('id', axis=1)
        
        # Separate features and labels
        X_train = train_df.drop(['label', 'attack_cat'], axis=1, errors='ignore')
        y_train = train_df['label'].values
        X_test = test_df.drop(['label', 'attack_cat'], axis=1, errors='ignore')
        y_test = test_df['label'].values
        
        # Encode categorical features
        X_train_encoded = pd.get_dummies(X_train)
        X_test_encoded = pd.get_dummies(X_test)
        
        # Align columns (handle missing dummies in test)
        X_train_encoded, X_test_encoded = X_train_encoded.align(
            X_test_encoded, join='left', axis=1, fill_value=0
        )
        
        self.train_columns = X_train_encoded.columns
        print(f"   Encoded features: {len(self.train_columns)}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_encoded)
        X_test_scaled = self.scaler.transform(X_test_encoded)
        
        # Feature selection
        X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = self.feature_selector.transform(X_test_scaled)
        
        # Get selected feature names
        selected_indices = self.feature_selector.get_support(indices=True)
        selected_features = [self.train_columns[i] for i in selected_indices]
        
        self.is_fitted = True
        
        print(f"   Selected top {self.n_features} features")
        print(f"   Training samples: {len(X_train_selected)}")
        print(f"   Test samples: {len(X_test_selected)}")
        print(f"   Training attacks: {sum(y_train)} ({sum(y_train)/len(y_train)*100:.1f}%)")
        print(f"   Test attacks: {sum(y_test)} ({sum(y_test)/len(y_test)*100:.1f}%)")
        
        return X_train_selected, X_test_selected, y_train, y_test, selected_features
    
    def transform_single(self, df):
        """Transform a single dataframe (for prediction)"""
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        if 'id' in df.columns:
            df = df.drop('id', axis=1)
        
        df = df.drop(['label', 'attack_cat'], axis=1, errors='ignore')
        df_encoded = pd.get_dummies(df)
        df_encoded = df_encoded.reindex(columns=self.train_columns, fill_value=0)
        df_scaled = self.scaler.transform(df_encoded)
        df_selected = self.feature_selector.transform(df_scaled)
        
        return df_selected