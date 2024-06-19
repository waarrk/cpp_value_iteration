import numpy as np
import matplotlib.pyplot as plt


def load_max_values(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    data = []
    for line in lines:
        data.append([float(value) for value in line.split()])

    return np.array(data)


def plot_heatmap(data):
    plt.figure(figsize=(8, 6))
    plt.imshow(data, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title('Heatmap of Maximum Values')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()


# ファイルから最大の価値を読み込む
max_values_data = load_max_values('max_values.txt')

# ヒートマップをプロットする
plot_heatmap(max_values_data)
