from typing import Tuple

from lip.utils.paths import MOVIE_DATA_FILE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # READ DATA
    movies_data_df: pd.DataFrame = pd.read_csv(MOVIE_DATA_FILE)
    movies_budget_X: np.ndarray = movies_data_df['budget'].values
    movies_budget_X = movies_budget_X[:, np.newaxis]
    movies_revenue_y: np.ndarray = movies_data_df['revenue'].values
    # DIVIDE BY 10-e6
    movies_budget_X /= 1000000
    movies_revenue_y /= 1000000

    # WE SPLIT TO MAKE A LITTLE TEST LATER ON
    X_train, X_test, y_train, y_test = train_test_split(movies_budget_X, movies_revenue_y,
                                                        test_size=0.1,
                                                        random_state=42)

    # FIT
    linear_regression: LinearRegression = LinearRegression()
    linear_regression.fit(X_train, y_train)

    print('Coefficients: \n', linear_regression.coef_)

    # MAKE PREDICTION ON TEST
    y_predicted: np.ndarray = linear_regression.predict(X_test)

    print("Mean squared error: %.2f"
          % mean_squared_error(y_test, y_predicted))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(y_test, y_predicted))

    # PLOT
    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots()

    ax.scatter(X_test, y_test, color='black')
    ax.plot(X_test, y_predicted, color='blue', linewidth=3)

    ax.set_xlabel('Budget (USD)')
    ax.set_ylabel('Revenue (USD')

    plt.show()
