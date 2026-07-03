# Customer Churn Prediction: Leakage-Safe Pipeline with MLflow

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AsserGharib1/CustomerChurnPrediction/blob/main/customer_churn_prediction.ipynb)
[![View on nbviewer](https://img.shields.io/badge/view%20full%20notebook-nbviewer-F37626?logo=jupyter&logoColor=white)](https://nbviewer.org/github/AsserGharib1/CustomerChurnPrediction/blob/main/customer_churn_prediction.ipynb)

End-to-end churn classification on the **Customer Churn Business** dataset (OpenML), centered on one question: what does class imbalance do to real churn detection, and which resampling strategy actually fixes it?

## Highlights

- **Leakage-safe preprocessing**: KNN imputation, encoding, and scaling fit on the training split only.
- **Six classifiers benchmarked**: Random Forest, Bagging, Stacking, Voting, Logistic Regression, SVM, across **three imbalance strategies**: SMOTE, random oversampling, and undersampling.
- **Every experiment tracked in MLflow**, selection on F1 and ROC-AUC, not raw accuracy.

## The imbalance story (test set, 204 true churns)

| Setup | Accuracy | F1 | Churns caught |
|---|---|---|---|
| Best model, original imbalanced data (Stacking) | 0.8880 | 0.8505 | **7 / 204** |
| Best model on SMOTE data (Random Forest) | 0.8715 | 0.8471 | 16 / 204 (churn recall 0.08) |
| Best model with **undersampling** | 0.7220 | 0.7755 | **162 / 204** |

The headline: the highest-*accuracy* model was nearly useless at its actual job (7 churns caught), SMOTE barely moved the needle (16 caught, churn-class recall 0.08), and only undersampling traded a little accuracy for a **23× jump in churn detection**. F1 above is the weighted score, which is why recall on the minority class had to drive the final selection.

## Sample outputs

![Model comparison](figures/model_comparison_1.png)

![Model comparison 2](figures/model_comparison_2.png)

![Clusters](figures/kmeans_clusters.png)

## Pipeline, exactly as implemented in the notebook

1. **Load and audit.** Null counts checked, then nulls are injected in a controlled way so the imputation stage can be exercised and measured honestly.
2. **Split first.** Train/test split happens before any fitting. Every transformer below is fit on the training split only.
3. **KNN imputation (k=8), train-fit only:**

```python
knn_imputer = KNNImputer(n_neighbors=8)
X_train_num_imputed = pd.DataFrame(knn_imputer.fit_transform(X_train_raw[num_cols]), ...)
X_test_num_imputed  = pd.DataFrame(knn_imputer.transform(X_test_raw[num_cols]), ...)
```

   Mean and variance are audited before vs after imputation to confirm the distributions were not distorted.
4. **Outliers.** IQR analysis, then Isolation Forest for handling.
5. **Scaling and encoding**, fit on train, applied to test.
6. **EDA.** Target distribution, feature correlation with churn, revenue and monthly-fee patterns, discount effect, and K-Means clustering for data understanding.
7. **Feature selection** with SelectKBest.
8. **Three balancing variations** of the same training data:

```python
smote = SMOTE(random_state=RND)
X_smote, y_smote = smote.fit_resample(X_train_final, y_train_final)
```

   plus `RandomOverSampler` and `RandomUnderSampler` variants. Resampling touches the training split only, never the test set.
9. **Hyperparameter tuning** for Random Forest, Logistic Regression, and SVM.
10. **Train 6 classifiers × 3 data variations**, every run logged:

```python
mlflow.set_experiment(EXPERIMENT_NAME)
with mlflow.start_run(run_name=run_name):
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
```

11. **Compare per variation, then a final cross-variation comparison** on the untouched test set (204 true churns), which produces the table above.

## Running

```bash
pip install -r requirements.txt
jupyter notebook customer_churn_prediction.ipynb
mlflow ui
```

Dataset: `data/customer_churn_business_dataset.csv` (source: OpenML).
