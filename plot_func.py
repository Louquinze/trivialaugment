from TrivialAugment.ac_func.experiment_only_topk import *
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == '__main__':
    df = {"x": [], "y": [], "func": [], "r": []}
    r_lst = [1, 5, 15, 50]
    with torch.no_grad():
        for idx, F in enumerate(
                [Func_01, Func_02, Func_03, Func_04, Func_05, Func_06, Func_07, Func_08, Func_09, Func_10]):
            for r in r_lst:
                # for b_0 in torch.linspace(-0.5, 0.5, 3):
                #     for b_1 in torch.linspace(-0.5, 0.5, 3):

                x = torch.linspace(-r, r, 10000)
                func = F()
                if sum("beta" in key for key in func._parameters):
                    y = func.forward(x).detach()[0][0]
                else:
                    y = func.forward(x).detach()

                df["x"] += x.tolist()
                df["y"] += y.tolist()
                df["func"] += [f"func_{idx + 1}"] * len(x)
                # df["b_0"] += [float(b_0)] * len(x)
                # df["b_1"] += [float(b_1)] * len(x)
                df["r"] += [r] * len(x)

    print({key: len(df[key]) for key in df.keys()})
    df = pd.DataFrame(df)
    sns.set_theme()

    for r in r_lst:
        df_tmp = df[df["r"] == r]
        g = sns.FacetGrid(df_tmp, col="func", col_wrap=5)
        g.map(sns.lineplot, "x", "y")
        plt.savefig(f"new_ac_func_r_{r}.png")
        plt.show()
