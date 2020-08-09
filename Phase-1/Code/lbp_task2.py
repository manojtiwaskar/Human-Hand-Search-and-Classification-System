# Implement a program which, given a folder with images, extracts and stores feature descriptors for all the images
# in the folder
import getopt
import sys
import cv2
import os
from lbp import LBP
import pandas as pd
from pathlib import Path


class Task2:

    def main(self):
        a = 0
        argv = sys.argv[1:]

        try:
            # Define the getopt parameters
            opts, args = getopt.getopt(argv, 'd:m:', ['foperand', 'soperand'])
            if len(opts) == 0 and len(opts) > 2:
                print('usage: add.py -d <first_operand> -m <second_operand>')
            else:
                # call model function and pass argument
                t2.acess_all_images(opts[0][1], opts[1][1])


        except getopt.GetoptError:
            # Print something useful
            print('usage: main.py -d <first_operand> -m <second_operand>')
            sys.exit(2)

    def acess_all_images(self, folder_path, model_name):
        for file in os.listdir(folder_path):
            if file.endswith(".jpg"):
                print(os.path.join(folder_path, file))
                t2.model_call_function(os.path.join(folder_path, file), model_name)

    def model_call_function(self, image_path, model_name):
        image_name = Path(image_path).stem
        if model_name == 'LBP':
            lbp_object = LBP()
            task2_lbp_histogram = lbp_object.model_lbp(image_path)

        if not os.path.exists(os.path.join(os.path.dirname(image_path), "task2ouput")):
            os.mkdir(os.path.join(os.path.dirname(image_path), "task2ouput"))
        output_file = os.path.join(os.path.join(os.path.dirname(image_path), "task2ouput"), image_name + ".csv")

        df = pd.DataFrame(task2_lbp_histogram)
        df.to_csv(output_file, index=False, header=False)


if __name__ == '__main__':
    t2 = Task2()
    t2.main()
