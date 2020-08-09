import getopt
import sys
import os
from lbp import LBP
import pandas as pd
from pathlib import Path


class Task1:

    def main(self):
        # Get the arguments from the command-line except the filename
        argv = sys.argv[1:]

        try:
            # Define the getopt parameters
            opts, args = getopt.getopt(argv, 'i:m:', ['foperand', 'soperand'])
            if len(opts) == 0 and len(opts) > 2:
                print('usage: add.py -i <first_operand> -m <second_operand>')
            else:
                # call model function and pass argument
                t1.model_call_function(opts[0][1], opts[1][1])


        except getopt.GetoptError:
            # Print something useful
            print('usage: main.py -i <first_operand> -m <second_operand>')
            sys.exit(2)

    def model_call_function(self, image_path, model_name):
        image_name = Path(image_path).stem
        if model_name == 'LBP':
            lbp_object = LBP()
            task1_lbp_histogram = lbp_object.model_lbp(image_path)

        if not os.path.exists(os.path.join(os.path.dirname(image_path), "task1output")):
            os.mkdir(os.path.join(os.path.dirname(image_path), "task1output"))
        output_file = os.path.join(os.path.join(os.path.dirname(image_path), "task1output"), image_name + ".csv")
        df = pd.DataFrame(task1_lbp_histogram)
        df.to_csv(output_file, index=False, header=False)


if __name__ == '__main__':
    t1 = Task1()
    t1.main()
