import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

#download heart disease data
heart_disease = fetch_ucirepo(id=45)
X = heart_disease.data.features
y = heart_disease.data.targets['num']

#binarize dataset (0 is healthy, 1,2,3,4 is sick so we convert all to 1)
y_bin = (y > 0).astype(int)

# clean dataset with median for some null values
X = X.fillna(X.median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

activations = ['logistic', 'tanh', 'relu']
hidden_layers = [(50,), (100,), (50, 50)]


def evaluate(X_data, y_data):
    results = []
    # 5 fold cross validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for act in activations:
        for hl in hidden_layers:
            acc_folds, f1_folds = [], []
            for train_idx, test_idx in kf.split(X_data):
                X_train, X_test = X_data[train_idx], X_data[test_idx]
                y_train, y_test = y_data.iloc[train_idx], y_data.iloc[test_idx]

                mlp = MLPClassifier(hidden_layer_sizes=hl, activation=act, max_iter=500, random_state=42)
                mlp.fit(X_train, y_train)
                preds = mlp.predict(X_test)

                acc_folds.append(accuracy_score(y_test, preds))
                f1_folds.append(f1_score(y_test, preds, zero_division=0))

            results.append({
                'Activation': act,
                'Hidden Layers': str(hl),
                'Accuracy': np.mean(acc_folds),
                'F1-Score': np.mean(f1_folds)
            })
    return pd.DataFrame(results)


print("\nExperiment 1")
df_results_all = evaluate(X_scaled, y_bin)
print(df_results_all.to_markdown(index=False))

print("\nExperiment 2")
# chosen features: age, sex, chest pain, blood pressure and cholesterol
features_5 = ['age', 'sex', 'cp', 'trestbps', 'chol']
X_5 = X[features_5]
X_5_scaled = scaler.fit_transform(X_5)

df_results_5 = evaluate(X_5_scaled, y_bin)
print(df_results_5.to_markdown(index=False))