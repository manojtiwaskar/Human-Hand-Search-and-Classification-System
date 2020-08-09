import numpy as np
from copy import copy, deepcopy


class Local_Binary_Pattern:
    m = 0
    n = 0

    def __init__(self):
        a = 0

    # lbp function using uniform method
    def lbp(self, array):
        LocalBinaryPattern.m, LocalBinaryPattern.n = np.shape(array)
        new_array = deepcopy(array)

        for i in range(0, LocalBinaryPattern.m):
            for j in range(0, LocalBinaryPattern.n):
                binary_code = ""
                binary_code = self.assign_binary_values(array, i, j)
                new_array[i][j] = self.binary_to_decimal(binary_code)
        return new_array

    # This function assigns binary value to the eight surrounding pixels when compared to threshold value of center
    # pixel
    def assign_binary_values(self, array, x1, y1):
        threshold = array[x1][y1]
        binary_string = ""
        for tempx in range(x1 - 1, x1 + 1 + 1):
            for tempy in range(y1 - 1, y1 + 1 + 1):
                if tempx != x1 or tempy != y1:
                    binary_string = binary_string + self.get_binary_values(tempx, tempy, array, threshold)

        return binary_string

    # Compares the cell value with threshold and returns 0 or 1 and also covers the edge cases and corner cases
    def get_binary_values(self, x, y, array, threshold):
        m = LocalBinaryPattern.m
        n = LocalBinaryPattern.n
        if x < 0 and y < 0:
            return "0"
        elif x >= m and y < 0:
            return "0"
        elif x >= m and y >= n:
            return "0"
        elif x < 0 and y >= n:
            return "0"
        elif x < 0 and 0 <= y < n:
            return "0"
        elif 0 <= x < m and y < 0:
            return "0"
        elif x >= m and 0 <= y < n:
            return "0"
        elif 0 <= x < m and y >= n:
            return "0"
        else:
            if array[x][y] >= threshold:
                return "1"
            else:
                return "0"

    def binary_to_decimal(self, bin_str):
        return int(bin_str, 2)
