import pandas as pd
import numpy as np

def classificationMetrics(fileName, modelName):
    # load data
    dataFrame = pd.read_csv(fileName)

    # using threshold
    dataFrame['y_pred'] = (dataFrame['pred'] >= 0.5).astype(int)

    TP = ((dataFrame['gt'] == 1) & (dataFrame['y_pred'] == 1)).sum()
    TN = ((dataFrame['gt'] == 0) & (dataFrame['y_pred'] == 0)).sum()
    FP = ((dataFrame['gt'] == 0) & (dataFrame['y_pred'] == 1)).sum()
    FN = ((dataFrame['gt'] == 1) & (dataFrame['y_pred'] == 0)).sum()

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    # manage division by 0 errors
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0
    F1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    dataFrame['rank'] = dataFrame['pred'].rank(method='average')
    R1 = dataFrame[dataFrame['gt'] == 1]['rank'].sum()
    n1 = (dataFrame['gt'] == 1).sum()
    n0 = (dataFrame['gt'] == 0).sum()

    AUC_ROC = (R1 - (n1 * (n1 + 1)) / 2) / (n1 * n0) if (n1 * n0) > 0 else 0

    print(f"\n--- {modelName} Results ---")
    print(f"TP: {TP} | TN: {TN} | FP: {FP} | FN: {FN}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"TNR:       {TNR:.4f}")
    print(f"F1-Score:  {F1:.4f}")
    print(f"AUC-ROC:   {AUC_ROC:.4f}")

models = [
    ('models_results/Model_A.csv', 'Model A'),
    ('models_results/Model_B.csv', 'Model B'),
    ('models_results/Model_C.csv', 'Model C'),
    ('models_results/Model_D.csv', 'Model D')
]

for path, name in models:
    classificationMetrics(path, name)

########################################

def regressionMetrics(fileName, modelName):
    df = pd.read_csv(fileName)

    real = df['gt']
    pred = df['pred']

    absError = (real - pred).abs()

    # mean of absolute errors
    mae = absError.mean()

    # mean squared error
    mse = ((real - pred) ** 2).mean()

    # root mean squared error
    rmse = np.sqrt(mse)

    # absolute error median
    medae = absError.median()

    print(f"\n--- {modelName} Regression Results ---")
    print(f"MAE:   {mae:.4f}")
    print(f"MSE:   {mse:.4f}")
    print(f"RMSE:  {rmse:.4f}")
    print(f"MedAE: {medae:.4f}")


modelos_reg = [
    ('models_results/Reg_0.csv', 'Model 0'),
    ('models_results/Reg_1.csv', 'Model 1'),
    ('models_results/Reg_2.csv', 'Model 2'),
    ('models_results/Reg_3.csv', 'Model 3'),
    ('models_results/Reg_4.csv', 'Model 4')
]

for path, name in modelos_reg:
    regressionMetrics(path, name)