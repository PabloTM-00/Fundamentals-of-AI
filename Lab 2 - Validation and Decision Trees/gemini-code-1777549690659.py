import pandas as pd
import numpy as np
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix

# 1. Carga de datos
df = pd.read_csv('decision_tree_dataset.csv')
X = df.drop('target', axis=1) # Variables predictoras
y = df['target'] # Variable a predecir

# Función auxiliar para TNR (Especificidad)
def calcular_tnr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0

print("=== INICIANDO EXPERIMENTOS ===")

# --- EXPERIMENTO 1: HOLD-OUT (10 Iteraciones) ---
print("\n--- Método Hold-out (10% Test) ---")
for i in range(10):
    start = time.time()
    # OJO: No usamos random_state para que cada partición sea distinta
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)
    
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    tnr = calcular_tnr(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Manejo de error si el test set (12 filas) aleatoriamente solo tiene 1 clase
    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = np.nan
        
    t = time.time() - start
    print(f"Exp {i+1} | Tiempo: {t:.4f}s | Acc: {acc:.2f} | F1: {f1:.2f} | AUC: {auc:.2f}")

# --- EXPERIMENTO 2: 10-FOLD CROSS-VALIDATION (10 Iteraciones) ---
print("\n--- Método 10-Fold Cross-Validation ---")
for i in range(10):
    start = time.time()
    # shuffle=True garantiza particiones distintas y aleatorias por experimento
    kf = KFold(n_splits=10, shuffle=True)
    
    # Listas temporales para guardar las métricas de los 10 folds
    acc_folds, f1_folds, auc_folds = [], [], []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]
        
        acc_folds.append(accuracy_score(y_test, y_pred))
        f1_folds.append(f1_score(y_test, y_pred, zero_division=0))
        try:
            auc_folds.append(roc_auc_score(y_test, y_prob))
        except ValueError:
            pass
            
    t = time.time() - start
    # Mostramos la media de los 10 folds para este experimento
    print(f"Exp {i+1} | Tiempo: {t:.4f}s | Media Acc: {np.mean(acc_folds):.2f} | Media F1: {np.mean(f1_folds):.2f} | Media AUC: {np.mean(auc_folds):.2f}")