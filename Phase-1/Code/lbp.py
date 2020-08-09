import cv2
import numpy as np

from localBinaryPattern import Local_Binary_Pattern


class LBP:

    def model_lbp(self, image_path_id):

        img_rgb = cv2.imread(image_path_id)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        #dimensions = img_gray.shape
        img_gray_height = img_gray.shape[0]
        img_gray_width = img_gray.shape[1]
        print(img_gray_height, img_gray_width)
        height_window = 100
        width_window = 100

        # object of class LocalBinaryPattern in which LBP algo is implemented in file localBinaryPattern.py
        mylbp = LocalBinaryPattern()

        histogram = []
        # cv2.imshow('image',img_gray)
        for h in range(0, img_gray_height, height_window):
            for w in range(0, img_gray_width, width_window):
                x1 = h + height_window
                y1 = w + width_window
                cropped_tiles = img_gray[h:x1, w:y1]
                # Uniform LBP is used
                lbp = mylbp.lbp(cropped_tiles)
                # x = itemfreq(lbp.ravel())
                (hist, bin_edges) = np.histogram(lbp.ravel(), bins=256)
                histogram.append(hist)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return histogram
