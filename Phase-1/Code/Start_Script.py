import argparse
import os
import glob
import cv2
import pandas as pd
# from LBP_Feature import LBP_Feature
from Sift_Feature import Sift_Feature

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model',action="store", dest="model",help="Provide any of these model: LBP, SIFT, CM, HOG", default="SIFT")
parser.add_argument('-d', '--dir',action="store", dest="dir",help="Provide directory name", default="None")
parser.add_argument('-r', '--ranking',action="store_true", dest="ranking",help="Enable ranking of images")
parser.add_argument('-i', '--imageLoc',action="store", dest="imageLoc",help="Provide image name", default="None")
parser.add_argument('-k', '--kimage',type=int, dest="kimage",help="Provide k value to get k similar images", default=-1)
parser.add_argument('-s', '--single_task',action="store_true", dest="single_task",help="Enable task 1 for single image")

args = parser.parse_args()

if args.dir == "None":
    print("Please provide directory name")
    exit(1)

curpath = os.path.dirname(os.path.abspath(__file__))
print(curpath)

dirpath = os.path.join(curpath, args.dir)
print(dirpath)

if not os.path.exists(dirpath):
    print("Please provide proper directory location")
    exit(1)

imagePath = os.path.join(curpath, args.dir, args.imageLoc)
print(imagePath)

topn =args.kimage

if args.ranking:
    if args.imageLoc == "None" or not os.path.exists(imagePath):
        print("Please provide proper image location using '-i'")
        exit(1)
    if (int(args.kimage) == -1):
        print("Please provide k value using '-k'")
        exit(1)

if args.single_task:
   if args.imageLoc == "None" or not os.path.exists(imagePath):
       print("Please provide proper directory location")
       exit(1)


   if args.model == 'SIFT':
       m1 = Sift_Feature('./Extracted_Features/')
       descriptor = m1.extract_features(imagePath)
       descriptor = pd.DataFrame(descriptor)
       descriptor.to_excel(curpath + "/Sift_" + str(args.imageLoc) + '.xlsx')
   else:
       print("Please provide proper model name")
       exit(1)
   exit(0)


if args.model == 'SIFT':
    if glob.glob(os.path.join(curpath+"/Extracted_Features/", "*.xlsx")):
        ma = Sift_Feature('./Extracted_Features/')
        ma.batch_extractor(dirpath)

if args.model == 'SIFT' and args.ranking:
    ma1 = Sift_Feature('./Extracted_Features/')
    ma1.match(imagePath,dirpath, topn)
