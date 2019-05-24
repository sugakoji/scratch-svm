import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


def viz(x, y, model, var_1, var_2):
    fig = plt.figure()
    ax = plt.axes()

    resolution = 0.01
    markers = ('s', 'x', 'o')
    cmap = ListedColormap(('red', 'blue', 'green'))

    ## 2変数の入力データの最小値から最大値まで引数resolutionの幅でメッシュを描く
    x1_min, x1_max = x[:, var_1].min() - 1, x[:, var_1].max() + 1
    x2_min, x2_max = x[:, var_2].min() - 1, x[:, var_2].max() + 1
    x1_mesh, x2_mesh = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                                   np.arange(x2_min, x2_max, resolution))

    ## メッシュデータ全部を学習モデルで分類
    z = model.predict(np.array([x1_mesh.ravel(), x2_mesh.ravel()]).T)
    z = z.reshape(x1_mesh.shape)

    ## メッシュデータと分離クラスを使って決定境界を描いている
    plt.contourf(x1_mesh, x2_mesh, z, alpha=0.3, cmap=cmap)
    plt.xlim(x1_mesh.min(), x1_mesh.max())
    plt.ylim(x2_mesh.min(), x2_mesh.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=x[y == cl, var_1],
                    y=x[y == cl, var_2],
                    alpha=0.6,
                    c=cmap(idx),
                    edgecolors='black',
                    marker=markers[idx],
                    label=cl)

    plt.tight_layout()

    plt.show()
