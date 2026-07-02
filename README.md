# Customer Churn Prediction — Leakage-Safe Pipeline with MLflow

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AsserGharib1/CustomerChurnPrediction/blob/main/customer_churn_prediction.ipynb)
[![View on nbviewer](https://img.shields.io/badge/view%20full%20notebook-nbviewer-F37626?logo=jupyter&logoColor=white)](https://nbviewer.org/github/AsserGharib1/CustomerChurnPrediction/blob/main/customer_churn_prediction.ipynb)


End-to-end churn classification on an imbalanced business dataset, with every experiment tracked in MLflow.

## Highlights

- **Leakage-safe preprocessing**: KNN imputation, encoding, and scaling fit on the training split only.
- **Six classifiers benchmarked** across three imbalance strategies: SMOTE, random oversampling, and undersampling.
- **MLflow experiment tracking** for every run; model selection on F1 and ROC-AUC rather than accuracy.
- Undersampling raised true churn detections **from 16 to 162** on the test set.
- EDA: outlier analysis (IQR + Isolation Forest), K-Means clustering, feature selection with SelectKBest.

## Sample outputs

Final model comparison across imbalance strategies:

![Model comparison](figures/model_comparison_1.png)

![Model comparison 2](figures/model_comparison_2.png)

K-Means clustering for data understanding:

![Clusters](figures/kmeans_clusters.png)

## Repository contents

- `customer_churn_prediction.ipynb` — full pipeline with preserved outputs (44 documented sections).
- `data/customer_churn_business_dataset.csv` — dataset used by the notebook.

## Running

```bash
pip install -r requirements.txt
jupyter notebook customer_churn_prediction.ipynb
mlflow ui   # browse tracked experiments
```
