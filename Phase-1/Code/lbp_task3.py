# Implement a program which, given an image ID, a model, and a value “k”, returns and visualizes the most
# similar k images based on the corresponding visual descriptors. For each match, also list the overall matching score.

import getopt
import sys
import cv2
import os
import numpy as np
from scipy.spatial import distance
from skimage.feature import local_binary_pattern
from scipy.stats import itemfreq
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt
import csv
from lbp import LBP
from similarity import Sim
import pandas as pd
from pathlib import Path


class Task3:
    def main(self):
        # Get the arguments from the command-line except the filename
        argv = sys.argv[1:]

        try:
            # Define the getopt parameters
            opts, args = getopt.getopt(argv, 'i:m:k:d:', ['foperand', 'soperand', 'thoperand'])
            if len(opts) == 0 and len(opts) > 4:
                print('usage: add.py -i <first_operand> -m <second_operand>')
            else:
                # call model function and pass argument
                t3.query_image_model_function(opts[0][1], opts[1][1], opts[2][1], opts[3][1])


        except getopt.GetoptError:
            # Print something useful
            print('usage: main.py -i <first_operand> -m <second_operand>')
            sys.exit(2)

    def query_image_model_function(self, image_path, model_name, top_k, data_path):

        image_name = Path(image_path).stem
        print(data_path)
        if model_name == 'LBP':
            lbp_object = LBP()
            task3_lbp_histogram = lbp_object.model_lbp(image_path)

        # call similairty function for top k images
        sim_object = Sim()

        if not os.path.exists(os.path.join(os.path.dirname(image_path), "task3output")):
            os.mkdir(os.path.join(os.path.dirname(image_path), "task3output"))
        output_file = os.path.join(os.path.join(os.path.dirname(image_path), "task3output"), image_name + ".csv")
        df = pd.DataFrame(task3_lbp_histogram)
        df.to_csv(output_file, index=False, header=False)
        sim_object.similarity(output_file, top_k, data_path)


if __name__ == '__main__':
    t3 = Task3()
    t3.main()
