import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    os.makedirs("figures", exist_ok=True)

    # Metrics
    metrics = ["MSE", "RMSE", "MAE"]

    # Values
    lstm_values = [0.001812, 0.042562, 0.030458]
    transformer_values = [0.001619, 0.040237, 0.028655]

    x = np.arange(len(metrics))
    width = 0.3

    plt.figure(figsize=(8, 5))

    plt.bar(x - width/2, lstm_values, width, label="LSTM")
    plt.bar(x + width/2, transformer_values, width, label="Transformer")

    plt.xticks(x, metrics)
    plt.ylabel("Error")
    plt.title("Model Comparison by Metric")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    plt.savefig("figures/model_comparison_grouped.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()