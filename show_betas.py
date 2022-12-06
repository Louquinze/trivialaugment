import numpy as np
import os

if __name__ == '__main__':
    path = "/home/lukas/PycharmProjects/Evaluation/eval_func/logs"
    os.chdir(path)
    for func in os.listdir():
        # open torch
        os.chdir(func)
        print(func)
        for file_name in ["layer1_1.npy", "layer1_2.npy", "layer2_1.npy", "layer2_2.npy", "layer3_1.npy",
                          "layer3_2.npy", "ac_func_layer.npy"]:
            try:
                with open(file_name, 'rb') as f:
                    betas = np.load(f)
                print(f"{file_name}: {betas}")
            except Exception as e:
                print(e)
        print()
        os.chdir('..')
