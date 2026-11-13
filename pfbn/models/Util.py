import numpy as np
import pandas as pd

class Util:

    @staticmethod
    def z_norm(z):
        mean = np.mean(z, axis=0, keepdims=True)
        std = np.std(z, axis=0, keepdims=True)
        z = (z - mean) / std
        return z

    @staticmethod
    def norm_norm(z):
        norms = np.linalg.norm(z, axis=0)
        norms[norms == 0] = 1
        z = z / norms[np.newaxis, :]
        return z

    @staticmethod
    def norm_minmax_allmatrix(z):
        min_val = np.min(z)
        max_val = np.max(z)
        z = (z - min_val) / (max_val - min_val)
        return z

    @staticmethod
    def save2csv(paths, output_path):
        df = pd.DataFrame(paths)
        df.to_csv(output_path, index=False, header=False, encoding='utf-8')

    @staticmethod
    def rand_intarr_generate(total_length, range_low, range_high):
        one_to_hundred = np.arange(range_low, range_high + 1)
        remaining_length = total_length - len(one_to_hundred)
        random_elements = np.random.randint(range_low, range_high + 1, size=remaining_length)
        combined_array = np.concatenate((one_to_hundred, random_elements))
        np.random.shuffle(combined_array)
        return combined_array
