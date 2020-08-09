# similarity class
import operator
import os
import pandas as pd
from pathlib import Path


class Sim:
    def similarity_LBP(self, output_file, top_k, data_path):
        a = 0
        des_queryimage = pd.read_csv(output_file, sep=',', header=None)
        matching_score = {}
        for file in os.listdir(data_path):
            if file.endswith(".csv"):
                print(os.path.join(data_path, file))
                des = pd.read_csv(os.path.join(data_path, file), sep=',', header=None)
                sum_hist = ((((des_queryimage.subtract(des)) ** 2).sum(axis=1)) ** 0.5).values.sum()
                image_name = Path(file).stem
                matching_score[image_name] = sum_hist

        sorted_matchingscore = sorted(matching_score.items(), key=operator.itemgetter(1))
        for i in range(0, int(top_k)):
            print(sorted_matchingscore[i])
