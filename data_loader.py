import os
import cv2
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

RANDOM_STATE = 42

class DataLoader:
    def __init__(self, dataset: str, verbose: int = 0):
        self._dataset = dataset
        self._verbose = verbose

        self._datasets_folder = 'datasets'

    def load(self):
        if self._dataset == '1':
            return self._load_1()
        elif self._dataset == '2':
            return self._load_2()
        elif self._dataset == '3':
            return self._load_3()
        elif self._dataset == '4':
            return self._load_4()
        elif self._dataset == 'breast_cancer':
            return self._load_bc()
        elif self._dataset == 'chinese_mnist':
            return self._load_cm()
        else:
            return None

    def _load_1(self):
        centers = [[-700, -700], [0, 0], [500, -800]]
        X, y = make_blobs(n_samples=100, n_features=2,
                          centers=centers, cluster_std=1.0,
                          random_state=RANDOM_STATE)

        return X, y

    def _load_2(self):
        centers = [[0, 0], [4, 4], [-5, 4]]
        X, y = make_blobs(n_samples=500, n_features=2,
                          centers=centers, cluster_std=1.0,
                          random_state=RANDOM_STATE)

        return X, y

    def _load_3(self):
        centers = [[0, 0], [2, 2], [-3, 2]]
        X, y = make_blobs(n_samples=500, n_features=2,
                          centers=centers, cluster_std=1.0,
                          random_state=RANDOM_STATE)

        return X, y

    def _load_4(self):
        centers = [[0, 0], [1, 0]]
        X, y = make_blobs(n_samples=500, n_features=2,
                          centers=centers, cluster_std=1.0,
                          random_state=RANDOM_STATE)

        return X, y

    def _load_bc(self):
        filename = 'breast-cancer.csv'
        filepath = os.path.join(self._datasets_folder, filename)

        df = pd.read_csv(filepath)

        X = df.iloc[:,1:]
        y = df.iloc[:,:1]

        dtype_dict: dict = {}
        for col in X.columns:
            dtype_dict[col] = 'category'

        X = X.astype(dtype_dict)

        cat_columns = X.select_dtypes(['category']).columns
        X[cat_columns] = X[cat_columns].apply(lambda x: x.cat.codes)

        y['class'] = y['class'].astype('category')
        y = y['class'].cat.codes

        return X, y

    def _load_cm(self):
        DATASET = 'chinese_mnist'
        DATA_FILE = 'chinese_mnist.csv'

        DATA_FOLDER = 'data/data'
        FILE_FORMATTER = 'input_{}_{}_{}.jpg'

        DATA_FILE_PATH = os.path.join(self._datasets_folder, DATASET, DATA_FILE)
        DATA_FOLDER_PATH = os.path.join(self._datasets_folder, DATASET, DATA_FOLDER)

        df = pd.read_csv(DATA_FILE_PATH)
        # df = df.sample(frac = 1)

        data_size = df.shape[0]
        n_samples_per_class = 100
        samples_table: dict = {1: 0, 2: 0}

        X = []
        y = []

        if self._verbose > 0:
            print("Loading data...")

        for i in range(data_size):
            if self._verbose > 0:
                print("Progress:", int(100 * i / data_size), "%", end='\r', flush=True)

            input_data = df.iloc[i]

            value = int(input_data['code'])
            if value not in samples_table:
                continue
            if samples_table[value] >= n_samples_per_class:
                continue
            else:
                samples_table[value] += 1

            input_image_path = os.path.join(DATA_FOLDER_PATH,
                                            FILE_FORMATTER
                                            .format(input_data['suite_id'],
                                                    input_data['sample_id'],
                                                    value))

            input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

            input_image = cv2.resize(input_image, (32, 32))

            input_image_vector = input_image.flatten()

            X.append(input_image_vector)
            y.append(value)

        X = pd.DataFrame(np.array(X))
        y = np.array(y)

        if self._verbose > 0:
            print("\nDone!")

        return X, y
