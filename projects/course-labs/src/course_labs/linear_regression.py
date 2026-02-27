import numpy as np
import matplotlib.pyplot as plt
import copy
import math




def compute_cost_single_variable(x, y, w, b):
    """
    Computes the cost function for linear regression single variable

    Args:
        x (ndarray): Shape (m,) Input to the model (Population of cities)
        y (ndarray): Shape (m,) Label (Actual profits for the cities)
        w, b (scalar): Parameters of the model

    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # number of training examples
    m = x.shape[0]

    total_cost = 0

    for i in range(m):
        fwb_xi = w * x[i] + b
        total_cost += (fwb_xi - y[i]) ** 2

    total_cost /= 2 * m

    return total_cost


def compute_cost_multi_variable(x, y, w, b):
    """
    Computes the cost function for linear regression with multiple variables.
    """
    m = x.shape[0]

    # Calculate all predictions at once f_wb = Xw + b
    # np.dot handles the multiplication of features and weights
    f_wb = np.dot(x, w) + b

    # Calculate squared errors
    err_sq = (f_wb - y) ** 2

    # Sum and divide by 2m
    total_cost = np.sum(err_sq) / (2 * m)

    # Use .item() to ensure it returns a standard Python float
    return total_cost.item()



def compute_gradient(X, y, w, b):
    """
    X: ndarray (m, n) - Matrix of m examples with n features
    y: ndarray (m,)  - Target values
    w: ndarray (n,)  - Parameters (weights)
    b: scalar        - Parameter (bias)
    """
    m, n = X.shape

    # 1. Calculate prediction error for all examples at once
    # (m,n) @ (n,) results in (m,)
    f_wb = np.dot(X, w) + b
    err = f_wb - y

    # 2. Calculate gradients
    # (n,m) @ (m,) results in (n,)
    dj_dw = (1 / m) * np.dot(X.T, err)
    dj_db = np.mean(err)

    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking
    num_iters gradient steps with learning rate alpha

    Args:
      x :    (ndarray): Shape (m,)
      y :    (ndarray): Shape (m,)
      w_in, b_in : (scalar) Initial values of parameters of the model
      cost_function: function to compute cost
      gradient_function: function to compute the gradient
      alpha : (float) Learning rate
      num_iters : (int) number of iterations to run gradient descent
    Returns
      w : (ndarray): Shape (1,) Updated values of parameters of the model after
          running gradient descent
      b : (scalar)                Updated value of parameter of the model after
          running gradient descent
    """

    # number of training examples
    m = len(x)

    # An array to store cost J and w's at each iteration — primarily for graphing later
    J_history = []
    w_history = []
    w = copy.deepcopy(w_in)  # avoid modifying global w within function
    b = b_in

    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_dw, dj_db = gradient_function(x, y, w, b)

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion
            cost = cost_function(x, y, w, b)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0:
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")

    return w, b, J_history, w_history  # return w and J,w history for graphing


def gradient_descent_multi_variable(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    # number of training examples
    m = len(x)

    # Arrays to store history for graphing
    j_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        # 1. Calculate the gradient
        dj_dw, dj_db = gradient_function(x, y, w, b)

        # 2. Update Parameters
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # 3. Save cost J at each iteration
        cost = cost_function(x, y, w, b)
        j_history.append(cost)

        # 4. Print progress
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4}: Cost {np.array(j_history[-1]).item():8.2f}")

    return w, b, j_history


def zscore_normalize_features(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma
