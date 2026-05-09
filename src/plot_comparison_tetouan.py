import matplotlib.pyplot as plt
import numpy as np

metrics = ["MSE", "RMSE", "MAE"]

lstm = [
    0.011035,
    0.105046,
    0.092032
]

transformer = [
    0.004146,
    0.064386,
    0.050492
]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(10, 5))

plt.bar(x - width/2, lstm, width, label="LSTM")
plt.bar(x + width/2, transformer, width, label="Transformer")

plt.xticks(x, metrics)
plt.ylabel("Error")
plt.title("Tetouan Dataset: Model Comparison by Metric")
plt.legend()

plt.savefig(
    "figures/tetouan_model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()