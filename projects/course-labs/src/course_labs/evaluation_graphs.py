import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def plot_model_performance(y_actual, y_pred, metrics=['mse', 'r2', 'mae']):
    """
    Standardised dashboard for regression model evaluation.

    Args:
        y_actual (ndarray): The true target values.
        y_pred (ndarray): The values predicted by the model.
        metrics (list): Choices are ['mse', 'r2', 'mae']. Default is all three.
    """
    num_metrics = len(metrics)
    if num_metrics == 0:
        print("No metrics selected for plotting.")
        return

    # Create dynamic subplot layout based on number of metrics chosen
    fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 6))

    # Ensure axes is an iterable even if only 1 plot is requested
    if num_metrics == 1:
        axes = [axes]

    # Calculate values once
    mse_val = mean_squared_error(y_actual, y_pred)
    r2_val = r2_score(y_actual, y_pred)
    mae_val = mean_absolute_error(y_actual, y_pred)

    plot_idx = 0

    # --- Plot 1: MSE Focus (Actual vs Predicted) ---
    if 'mse' in metrics:
        ax = axes[plot_idx]
        ax.scatter(y_actual, y_pred, alpha=0.3, color='blue')
        ax.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--', lw=2)
        ax.set_title(f"MSE Analysis\nMSE: {mse_val:,.0f}")
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        plot_idx += 1

    # --- Plot 2: R2 Focus (Model Fit) ---
    if 'r2' in metrics:
        ax = axes[plot_idx]
        ax.scatter(y_actual, y_pred, alpha=0.3, color='green')
        ax.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--', lw=2)
        ax.set_title(f"Variance Analysis\nR² Score: {r2_val:.4f}")
        ax.set_xlabel("Actual Values")
        plot_idx += 1

    # --- Plot 3: MAE/Residual Focus (Error Signature) ---
    if 'mae' in metrics:
        ax = axes[plot_idx]
        residuals = y_actual - y_pred
        ax.scatter(y_pred, residuals, alpha=0.3, color='purple')
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        ax.set_title(f"Residual Analysis\nAvg Error (MAE): ${mae_val:,.2f}")
        ax.set_xlabel("Predicted Values")
        ax.set_ylabel("Residual (Error)")
        plot_idx += 1

    plt.tight_layout()
    plt.show()
