import pandas as pd
import numpy as np
import os
from numpy.linalg import norm
import cv2



class Sift_Feature(object):

    def __init__(self, pickled_db_path):
        self.db_path = pickled_db_path

    def extract_features(self, image_path, vector_size=70):
        image = cv2.imread(image_path)
        try:
            # converting image into gray scale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            #
            sift = cv2.xfeatures2d.SIFT_create()
            # finding image keypoints
            kps = sift.detect(gray, None)
            # Getting first 32 of them.
            # Number of keypoints is varies depend on image size and color pallet
            # Sorting them based on keypoint response value(bigger is better)
            kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
            # computing descriptors vector
            kps, dsc = sift.compute(image, kps)
            # Making descriptor of same size
            dsc1 = dsc.flatten()
            # Descriptor vector size is 128

            if len(dsc) < vector_size:
                # if we have less the 50 descriptors then just adding zeros at the
                # end of our feature vector
                concat = vector_size - len(dsc)
                dsc = np.concatenate((dsc, np.zeros((concat, 128))), axis=0)
            descriptor = dsc
        except cv2.error as e:
            print('Error: ', e)
            return None
        # print(descriptor)
        return descriptor

    def batch_extractor(self, images_path):
        files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]

        result = {}
        for f in files:
            print('Extracting features from image %s', f)
            name = f.split("\\")[-1].lower()
            name = name.split(".")[0]
            print(name)
            descriptor = self.extract_features(f, 70)
            df = pd.DataFrame(descriptor)
            df.to_excel(self.db_path+"/SiftDes_" + name + ".xlsx", header=None, index=False)

    def Similarity_Metric(self, vector, Q_filename):
        # getting cosine distance between search image and images database
        Desc_files = [os.path.join(self.db_path, p) for p in sorted(os.listdir(self.db_path))]
        Best_Matches = []
        Similarity_score = []
        all_dist = []
        Query_Descriptor = vector
        Q_filename = Q_filename.split('\\')[-1]

        # vect1 is the the each feature descriptors of each image in the database
        for desc_file in Desc_files:
            filename = str(desc_file)
            temp = filename.split('/')[-1].split('.')[0].split('_')
            temp = temp[1]+'_'+temp[2]
            if temp == Q_filename:
                continue
            print(filename)
            if filename == './Extracted_Features/~$SiftDes_hand_0000002.xlsx':
                break
            df_desc = pd.read_excel(filename)
            vect1 = df_desc.values
            all_dist.clear()
            Best_Matches.clear()
            # print(desc_file)
            # D_des is the each descriptor of each image in the database
            for D_des in vect1:
                # Q_des is the each descriptor of query Image
                for Q_des in Query_Descriptor:
                    distance = (sum([(a - b) ** 2 for a, b in zip(D_des, Q_des)])) ** 0.5
                    all_dist.append(distance)
                    # all_dist.append(dot(D_des, Q_des)/(norm(D_des)*norm(Q_des)))
                min_dist = min(all_dist)
                Best_Matches.append(min_dist)
            Avg_sum_sim = int(sum(Best_Matches)/70)
            db_file = desc_file.split('/')[-1].split('.')[0]
            db_file = db_file.split('_')
            db_file = db_file[1]+'_'+db_file[2]
            Similarity_score.append([Q_filename, db_file, Avg_sum_sim])

        cos_distance = np.asarray(Similarity_score)
        Similarity_score_df = pd.DataFrame(cos_distance, columns= ['Query_Image','Database_Image','Matching_score'])
        convert_dict = {'Matching_score': int}
        Similarity_score_df = Similarity_score_df.astype(convert_dict)
        Similarity_score_df = Similarity_score_df.sort_values(by='Matching_score', ascending=True)
        Similarity_score_df.to_excel("./Sift_Output/Sift_Matching_Score.xlsx", columns=['Query_Image', 'Database_Image', 'Matching_score'], index=True)
        return Similarity_score_df

    def match(self, image_path, current_folder ,topn):
        features = self.extract_features(image_path)
        Query_filename = image_path.split('/')[-1].lower()
        Query_filename = Query_filename.split(".")[0]
        Similarity_score_df1 = self.Similarity_Metric(features, Query_filename)
        # Similarity_score_df1 = Similarity_score_df1.sort_values(by='Matching_score', ascending=False)
        # getting top K records
        Top_results = Similarity_score_df1.head(topn)
        nearest_img_paths = []
        similarity_score = []
        for index, row in Top_results.iterrows():
            nearest_img_paths.append(row['Database_Image'])
            similarity_score.append(row['Matching_score'])

        final_names = []
        for name in nearest_img_paths:
            name = name.split('/')[-1]
            name = name.split('.')[0]
            name = name.split('_')[-1]
            name = "Hand_" + name
            final_names.append(name)


        print("Result images ========================================")
        for i in range(topn):
            # we got cosine distance, less cosine distance between vectors
            # more they similar, thus we subtruct it from 1 to get match value
            print("Image : ", final_names[i], 'Match : ', similarity_score[i])

            img1 = cv2.imread(os.path.join(current_folder, final_names[i] + '.jpg'))
            cv2.imwrite('./Sift_Output/' + str(final_names[i]) + '.jpg', img1)


