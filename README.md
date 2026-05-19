# 🛡️ CyberSentinel AI

## Real-Time Network Intrusion Detection System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-green.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.10+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An AI-powered cybersecurity system that detects network intrusions using a **Random Forest + XGBoost voting ensemble** trained on the **UNSW-NB15 benchmark dataset**. Features a professional SOC dashboard for real-time threat visualization.

---

## 📊 Performance

| Metric | Score |
|--------|-------|
| **F1 Score** | 89.5% |
| **Recall** | 98.7% |
| **Precision** | 81.9% |
| **ROC-AUC** | 98.4% |
| **Accuracy** | 87.3% |

- **Dataset:** UNSW-NB15 (175,341 training, 82,332 test records)
- **Attack Types Detected:** 9 (Generic, Exploits, Fuzzers, DoS, Reconnaissance, Analysis, Backdoor, Shellcode, Worms)
- **Attacks Caught:** 44,748 out of 45,332 (only 584 missed)
- **False Alarms:** 9,879 (acceptable rate for SOC analyst triage)

---

## 🏗️ Project Structure
cybersentinel/
│
├── main.py # Entry point - run full pipeline
├── requirements.txt # Dependencies
├── README.md # Documentation
│
├── core/
│ ├── init.py
│ └── real_data_loader.py # Data loading & preprocessing
│ • load_unsw_data() # Load CSV files
│ • OptimizedPreprocessor # Encoding, scaling, feature selection
│
├── models/
│ ├── init.py
│ └── ensemble_model.py # ML model with hyperparameter tuning
│ • OptimizedEnsemble # Train, tune, predict
│
├── engine/
│ ├── init.py
│ └── evaluator.py # Evaluation & reporting
│ • ModelEvaluator # Metrics, confusion matrix, reports
│
├── dashboard/
│ ├── init.py
│ └── app.py # Streamlit SOC dashboard
│ • Performance gauges # Interactive metric visualization
│ • Attack analysis # Distribution & detection stats
│ • Data explorer # Raw data & feature inspection
│ • Resume summary # Copy-paste ready results
└── data/ # Dataset directory
├── UNSW_NB15_training-set.csv # 175,341 records
└── UNSW_NB15_testing-set.csv # 82,332 records

## 🔧 Tech Stack

| Category | Technology |
|----------|------------|
| **Language** | Python 3.8+ |
| **ML Frameworks** | Scikit-learn, XGBoost |
| **Data Processing** | Pandas, NumPy, Scipy |
| **Hyperparameter Tuning** | GridSearchCV with 3-fold stratified cross-validation |
| **Feature Selection** | Mutual Information (SelectKBest, top 75 of 194 features) |
| **Preprocessing** | Label Encoding, One-Hot Encoding, StandardScaler |
| **Dashboard** | Streamlit, Plotly |
| **Dataset** | UNSW-NB15 (public benchmark) |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### 1. Clone the Repository

```bash
git clone https://github.com/Aksh1312/cybersentinel.git
cd cybersentinel
2. Install Dependencies
bash
pip install -r requirements.txt
3. Download the Dataset
Download the UNSW-NB15 dataset from the official source:

🔗 UNSW-NB15 Dataset Download Page

Required files:

UNSW_NB15_training-set.csv (175,341 records)

UNSW_NB15_testing-set.csv (82,332 records)

Place both files in the data/ directory.

4. Run the Pipeline
bash
# Full training with hyperparameter tuning
python main.py

# Fast mode (smaller sample, no tuning) for quick testing
python main.py --fast
5. Launch the Dashboard
bash
streamlit run dashboard/app.py
The dashboard will open at http://localhost:8501

📈 Pipeline Details
Data Preprocessing
Step	Description	Result
1. Load	Read CSV files	175K train, 82K test records
2. Clean	Drop ID column	44 features remain
3. Split	Separate features from labels	42 features + 1 label
4. Encode	One-hot encode categoricals	42 → 194 features
5. Align	Match train/test columns	Consistent feature set
6. Scale	StandardScaler (mean=0, std=1)	Normalized features
7. Select	Mutual Information (top 75)	194 → 75 best features
Model Architecture
Ensemble: Random Forest + XGBoost (Soft Voting)

Model	Parameters	Tuning Method
Random Forest	n_estimators=100, max_depth=25	GridSearchCV
XGBoost	n_estimators=200, max_depth=8, learning_rate=0.1	GridSearchCV
Ensemble	Soft voting classifier	Combines probabilities
Training Performance
Metric	Value
Training samples	175,341
Features used	75 (selected from 194)
Cross-validation	3-fold stratified
Training time	~52 seconds
Inference time	683ms (82,332 records)
Random Forest CV F1	96.3%
XGBoost CV F1	96.3%
📊 Dashboard Features
The Streamlit dashboard has four tabs:

📊 Model Performance
Gauge charts for Accuracy, Precision, Recall, and F1 Score

Color-coded performance indicators (green > 90%, yellow > 80%, red < 80%)

Interactive confusion matrix heatmap

Detailed classification report

🔥 Attack Analysis
Bar chart of attack type distribution (9 attack categories)

Detection capability statistics (True/False Positives/Negatives)

Attack category breakdown with percentages

📋 Data Explorer
Raw training data preview (first 100 rows)

Top 15 selected features ranked by importance

Dataset statistics (records, features, class balance)

📝 Resume Summary
Copy-paste ready bullet points for your resume

All metrics formatted professionally

Complete project description

📝 Sample Output
text
============================================================
🛡️  CyberSentinel AI - Intrusion Detection System
============================================================

📥 Loading UNSW-NB15 Dataset
Loading UNSW-NB15 dataset...
✅ Training set: (175341, 45)
✅ Test set: (82332, 45)

📊 Attack types in training:
attack_cat
Normal            56000
Generic           40000
Exploits          33393
Fuzzers           18184
DoS               12264
Reconnaissance    10491
Analysis           2000
Backdoor           1746
Shellcode          1133
Worms               130

⚙️  Preprocessing Pipeline
   Encoded features: 194
   Selected top 75 features
   Training samples: 175,341
   Test samples: 82,332

🧠 Training Ensemble Model
   🔵 Tuning Random Forest...
      Best RF: {'max_depth': 25, 'n_estimators': 100}, F1: 0.963
   🟢 Tuning XGBoost...
      Best XGB: {'learning_rate': 0.1, 'max_depth': 8, 'n_estimators': 200}, F1: 0.963
   ✅ Training complete in 51.79s

📊 Model Evaluation
============================================================
   Test Samples: 82,332
   Accuracy:     87.29%
   Precision:    81.92%
   Recall:       98.71%
   F1 Score:     89.53%
   ROC-AUC:      98.36%

📝 Classification Report:
              precision    recall  f1-score   support
      Normal       0.98      0.73      0.84     37000
      Attack       0.82      0.99      0.90     45332
    accuracy                           0.87     82332

📋 Confusion Matrix:
                    Predicted Normal  Predicted Attack
   Actual Normal          27121            9879
   Actual Attack            584           44748

📝 FINAL SUMMARY
============================================================
   Dataset:      UNSW-NB15 (public benchmark)
   Model:        Random Forest + XGBoost Ensemble
   Accuracy:     87.3%
   Precision:    81.9%
   Recall:       98.7%
   F1 Score:     89.5%
   ROC-AUC:      98.4%
🎯 Key Features
✅ End-to-end ML pipeline — Data loading → Preprocessing → Feature selection → Training → Evaluation

✅ Ensemble learning — Random Forest + XGBoost with soft voting for robust predictions

✅ Automated feature engineering — Reduces 194 features to 75 using mutual information

✅ Hyperparameter tuning — GridSearchCV with 3-fold stratified cross-validation

✅ Production-ready metrics — Precision, Recall, F1, ROC-AUC, confusion matrix

✅ Interactive SOC dashboard — Professional visualization for security analysts

✅ Modular architecture — Clean separation of concerns (data, model, evaluation, dashboard)

✅ Real benchmark dataset — UNSW-NB15 with 9 modern attack types

✅ Reproducible results — All metrics verifiable from publicly available data

✅ Resume-ready — Real metrics you can confidently discuss in interviews

🎓 Research Context
The UNSW-NB15 dataset is a widely-cited benchmark in network intrusion detection research, created by the Australian Centre for Cyber Security. It contains realistic modern network traffic with nine attack families.

Published results on UNSW-NB15 typically range from 85-97% F1 score depending on methodology, feature engineering, and model complexity.

Our ensemble approach achieves 89.5% F1 with 98.7% recall — prioritizing maximum attack detection while maintaining acceptable precision for SOC analyst workflows.

Why High Recall Matters
In cybersecurity, missing an attack (false negative) is far more costly than a false alarm (false positive). Our model:

Catches 98.7% of all attacks (misses only 584 out of 45,332)

Generates false alarms at an acceptable rate (9,879 out of 37,000 normal records)

Enables SOC analysts to focus investigation on flagged events

🔮 Future Improvements
Additional datasets — Evaluate on CICIDS2017, NSL-KDD for cross-dataset validation

Deep learning models — Add CNN, LSTM, or Transformer architectures for comparison

Real-time streaming — Integrate with Apache Kafka for live network monitoring

REST API — Deploy model as a service using FastAPI or Flask

Explainability — Add SHAP or LIME for individual prediction explanations

Online learning — Implement concept drift detection for adaptive retraining

Docker support — Containerize for easy deployment

MLOps integration — Add MLflow for experiment tracking and model versioning

📄 License
This project is licensed under the MIT License. See the LICENSE file for details.

👤 Author
Aksh1312

GitHub: @Aksh1312

Acknowledgments
UNSW-NB15 Dataset: Moustafa, N., & Slay, J. (2015). UNSW-NB15: a comprehensive data set for network intrusion detection systems.

Scikit-learn: Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python.

XGBoost: Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System.

Streamlit: For making ML dashboard creation simple and beautiful.
