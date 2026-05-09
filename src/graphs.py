import matplotlib.pyplot as plt

y_true = y_test[0]
y_lstm = lstm_model.predict(X_test[:1])[0]
y_transformer = transformer_model.predict(X_test[:1])[0]

plt.figure(figsize=(10,5))
plt.plot(y_true, label="Actual", linewidth=3)
plt.plot(y_lstm, label="LSTM", linestyle="--")
plt.plot(y_transformer, label="Transformer", linestyle="--")

plt.title("Forecast Comparison: LSTM vs Transformer")
plt.xlabel("Hour Ahead")
plt.ylabel("Normalized Energy")
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig("figures/comparison_forecast.png", dpi=300)
plt.show()