#!/usr/bin/env python3
"""Linear regression example: fit a model on the diabetes dataset and visualize results.

Uses pynabled.linear_regression which fits y = X @ coef[1:] + coef[0] (intercept added
automatically). Returns (coefficients, r_squared).

Requires: numpy, pynabled, scikit-learn, matplotlib

Run from repo root:
    python python/examples/regression/linear_regression.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

import pynabled


def main():
    # Load diabetes dataset (10 features, 442 samples)
    data = load_diabetes()
    X = data["data"].astype(np.float64)
    y = data["target"].astype(np.float64)
    feature_names = data["feature_names"]

    # Fit linear regression (intercept added automatically)
    coef, r_squared = pynabled.linear_regression(X, y)

    # Predictions: y_hat = intercept + X @ slopes
    intercept = coef[0]
    slopes = coef[1:]
    y_pred = intercept + X @ slopes

    # Print summary
    print("Linear Regression (pynabled) - Diabetes dataset")
    print("-" * 50)
    print(f"Data shape: X {X.shape}, y {y.shape}")
    print(f"R² = {r_squared:.4f}")
    print(f"\nCoefficients:")
    print(f"  Intercept: {intercept:.4f}")
    for name, b in zip(feature_names, slopes):
        print(f"  {name}: {b:.4f}")

    # Figure: predicted vs actual, residuals
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Predicted vs actual
    ax1.scatter(y, y_pred, alpha=0.6, edgecolors="black", linewidth=0.3)
    lims = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
    ax1.plot(lims, lims, "k--", alpha=0.5, label="y = y_pred")
    ax1.set_xlabel("Actual")
    ax1.set_ylabel("Predicted")
    ax1.set_title("Predicted vs Actual")
    ax1.legend()
    ax1.set_aspect("equal")

    # Residuals
    residuals = y - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.6, edgecolors="black", linewidth=0.3)
    ax2.axhline(y=0, color="k", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Residual")
    ax2.set_title("Residual plot")

    plt.tight_layout()
    out_path = Path(__file__).parent / "linear_regression.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
