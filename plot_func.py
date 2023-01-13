from TrivialAugment.ac_func.experiment_final import *
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == '__main__':
    df = {"x": [], "y": [], "func": [], "r": []}
    r_lst = [0.5, 1, 5, 15]
    with torch.no_grad():
        # for b_1 in np.linspace(-0.5, 0.5, 4):
        #     for b_2 in np.linspace(-0.5, 0.5, 4):
        #         for b_3 in np.linspace(-0.5, 0.5, 4):
        for idx, F in enumerate(
                [Func_01, Func_02, Func_03, Func_04, Func_05, Func_06, Func_07, Func_08]):
            for r in r_lst:
                # for b_0 in torch.linspace(-0.5, 0.5, 3):
                #     for b_1 in torch.linspace(-0.5, 0.5, 3):

                x = torch.linspace(-r, r, 1000)
                # if idx != 10:
                func = F()
                y = func.forward(x).detach()
                print(func, len(y))
                # else:
                #     y = F(x)
                #     print(y)

                df["x"] += x.tolist()
                df["y"] += y.tolist()
                df["func"] += [f"Function {idx + 1}"] * len(x)
                df["r"] += [r] * len(x)

        print({key: len(df[key]) for key in df.keys()})
        df = pd.DataFrame(df)
        sns.set_theme()

for r in r_lst:
    # g = sns.FacetGrid(df_tmp, col="func", col_wrap=5)
    # g.map(sns.lineplot, "x", "y")
    # plt.savefig(f"new_ac_func_r_{r}.png")
    # plt.show()
    for idx, func in enumerate(df["func"].unique()):
        # if idx == 0:
        df_tmp = df[df["r"] == r]
        df_tmp = df_tmp[df_tmp["func"] == func]
        print(len(df_tmp))
        df_tmp = df_tmp.append(pd.DataFrame(
                {"x": torch.linspace(-r, r, 1000),
                 "y": torch.relu(torch.linspace(-r, r, 1000)),
                 "func": ["ReLU"] * 1000}
            )
        , ignore_index=True)

        df_tmp = df_tmp.append(pd.DataFrame(
            {"x": torch.linspace(-r, r, 1000),
             "y": torch.sigmoid(torch.linspace(-r, r, 1000)) * torch.linspace(-r, r, 1000),
             "func": ["SiLU"] * 1000}
        )
            , ignore_index=True)

        sns.lineplot(df_tmp, x="x", y="y", hue="func")
        plt.title(f"Function_{idx}; x-range: {r}")
        plt.savefig(f"Function_{idx}_x-range_{r}.png")
        plt.close()
