from TrivialAugment.ac_func.experiment_01 import Func_01, Func_02, Func_03, Func_04
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == '__main__':
    df = {"x": [], "y": [], "func": [], "b_0": [], "b_1": [], "r": []}
    with torch.no_grad():
        for idx, F in enumerate([Func_01, Func_02, Func_03, Func_04]):
            for r in [3, 100]:
                for b_0 in torch.linspace(-0.5, 0.5, 3):
                    for b_1 in torch.linspace(-0.5, 0.5, 3):

                        x = torch.linspace(-r, r, 1000)
                        func = F()
                        if idx in [0, 1]:
                            func.beta_mix -= 0.5
                            func.beta_mix -= b_0

                            if idx in [1]:
                                func.beta -= 1
                                func.beta -= b_1
                        y = func.forward(x).detach()
                        if idx == 2 or idx == 3:
                            y = y
                        else:
                            y = y[0][0]
                        df["x"] += x.tolist()
                        df["y"] += y.tolist()
                        df["func"] += [idx] * len(x)
                        df["b_0"] += [float(b_0)] * len(x)
                        df["b_1"] += [float(b_1)] * len(x)
                        df["r"] += [r] * len(x)

    df = pd.DataFrame(df)
    sns.set_theme()

    df_tmp = df[(df["func"] == 3) & (df["r"] == 3)]
    df_tmp = df_tmp.drop(['b_1', 'b_0'], axis=1)
    sns.lineplot(df_tmp, x="x", y="y", hue=df_tmp[['func']].apply(tuple, axis=1))
    plt.show()

    df_tmp = df[(df["func"] == 3) & (df["r"] == 100)]
    df_tmp = df_tmp.drop(['b_1', 'b_0'], axis=1)
    sns.lineplot(df_tmp, x="x", y="y", hue=df_tmp[['func']].apply(tuple, axis=1))
    plt.show()

    df_tmp = df[(df["func"] == 2) & (df["r"] == 3)]
    df_tmp = df_tmp.drop(['b_1', 'b_0'], axis=1)
    sns.lineplot(df_tmp, x="x", y="y", hue=df_tmp[['func']].apply(tuple, axis=1))
    plt.show()

    df_tmp = df[(df["func"] == 2) & (df["r"] == 100)]
    df_tmp = df_tmp.drop(['b_1', 'b_0'], axis=1)
    sns.lineplot(df_tmp, x="x", y="y", hue=df_tmp[['func']].apply(tuple, axis=1))
    plt.show()

    df_tmp = df[(df["func"] == 0) & (df["r"] == 3)]
    df_tmp = df_tmp.drop(['b_1'], axis=1)
    sns.lineplot(df_tmp, x="x", y="y", hue=df_tmp[['func', 'b_0']].apply(tuple, axis=1))
    plt.show()

    df_tmp = df[(df["func"] == 0) & (df["r"] == 100)]
    df_tmp = df_tmp.drop(['b_1'], axis=1)
    sns.lineplot(df_tmp, x="x", y="y", hue=df_tmp[['func', 'b_0']].apply(tuple, axis=1))
    plt.show()

    df_tmp = df[(df["func"] == 1) & (df["r"] == 3)]
    # df_tmp.drop(['b_1'], axis=1, inplace=True)
    sns.lineplot(df_tmp, x="x", y="y", hue=df_tmp[['func', 'b_0', 'b_1']].apply(tuple, axis=1))
    plt.show()

    df_tmp = df[(df["func"] == 1) & (df["r"] == 100)]
    # df_tmp.drop(['b_1'], axis=1, inplace=True)
    sns.lineplot(df_tmp, x="x", y="y", hue=df_tmp[['func', 'b_0', 'b_1']].apply(tuple, axis=1))
    plt.show()