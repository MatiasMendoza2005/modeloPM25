import matplotlib.pyplot as plt

def plot_real_vs_pred(y_true, y_pred, title="Predicción PM2.5"):
    plt.figure(figsize=(10,4))
    plt.plot(y_true, label="Real")
    plt.plot(y_pred, label="Predicho")
    plt.legend()
    plt.title(title)
    plt.show()
