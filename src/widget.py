import matplotlib.pyplot as plt

# Visualize the raw data
def plot_raw_signal(data_set, i: int = 0):
    X = data_set.X[i]

    plt.figure(figsize=(10,10))
    for j in range(X.shape[0]):
        x = np.linspace(0,X.shape[1]-1,X.shape[1])
        y = X[j] + j
        plt.plot(x,y)
    plt.show()