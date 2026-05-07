import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

df0 = pd.read_csv('data_0.csv').sort_values('input')
df1 = pd.read_csv('data_1.csv').sort_values('input')

datasets = {
    'Periodic Activity Monitoring': (df0[['input']].values, df0['output'].values, 10),
    'Increase in Accuracy according to use': (df1[['input']].values, df1['output'].values, 4)
}


def evaluate_and_plot(X, y, title, poly_degree):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Basic Linear': LinearRegression(),
        'L1 (Lasso)': Lasso(alpha=0.1),
        'L2 (Ridge)': Ridge(alpha=1.0),
        f'Polynomial (Degree {poly_degree})': make_pipeline(PolynomialFeatures(degree=poly_degree), LinearRegression())
    }

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='gray', alpha=0.5, label='Real Data')

    print(f"\n {title} ")
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        print(f"Model {name}: MSE = {mse:.4f} | R2 = {r2:.4f}")

        X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_plot = model.predict(X_plot)
        plt.plot(X_plot, y_plot, label=f'{name} (R2: {r2:.2f})', linewidth=2)

    plt.title(f'Model Fitting: {title}')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.legend()
    plt.grid(True)
    plt.show()


for title, (X, y, degree) in datasets.items():
    evaluate_and_plot(X, y, title, degree)