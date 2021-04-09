# Code modified from: https://github.com/hsjeong5/MNIST-for-Numpy

import numpy as np
from urllib import request
import gzip
import pickle
import pathlib

filename = [
    ["training_images", "train-images-idx3-ubyte.gz"],
    ["test_images", "t10k-images-idx3-ubyte.gz"],
    ["training_labels", "train-labels-idx1-ubyte.gz"],
    ["test_labels", "t10k-labels-idx1-ubyte.gz"]
]
filename2url = {
    "t10k-images-idx3-ubyte.gz": "https://folk.ntnu.no/haakohu/original_mnist/t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz": "https://folk.ntnu.no/haakohu/original_mnist/t10k-labels-idx1-ubyte.gz",
    "train-images-idx3-ubyte.gz": "https://folk.ntnu.no/haakohu/original_mnist/train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz": "https://folk.ntnu.no/haakohu/original_mnist/train-labels-idx1-ubyte.gz"
}
SAVE_PATH = pathlib.Path("data/original_mnist")


def download_mnist():
    SAVE_PATH.mkdir(exist_ok=True, parents=True)
    for name in filename:
        filepath = SAVE_PATH.joinpath(name[1])
        if filepath.is_file():
            continue
        print("Downloading "+name[1]+"...")
        request.urlretrieve(filename2url[name[1]], filepath)[1]


def extract_mnist():
    save_path = SAVE_PATH.joinpath("mnist.pkl")
    if save_path.is_file():
        return
    mnist = {}
    # Load images
    for name in filename[:2]:
        path = SAVE_PATH.joinpath(name[1])
        with gzip.open(path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
            print(data.shape)
            mnist[name[0]] = data.reshape(-1,   28*28)
    # Load labels
    for name in filename[2:]:
        path = SAVE_PATH.joinpath(name[1])
        with gzip.open(path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
            mnist[name[0]] = data
    with open(save_path, 'wb') as f:
        pickle.dump(mnist, f)


def load():
    download_mnist()
    extract_mnist()
    dataset_path = SAVE_PATH.joinpath("mnist.pkl")
    with open(dataset_path, 'rb') as f:
        mnist = pickle.load(f)
    X_train, Y_train, X_test, Y_test = mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]
    return X_train.reshape(-1, 28, 28), Y_train, X_test.reshape(-1, 28, 28), Y_test


if __name__ == '__main__':
    init()