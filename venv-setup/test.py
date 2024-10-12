from pathlib import Path
from matplotlib import pyplot
import numpy as np
import requests
import pickle
import gzip
import torch

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents = True, exist_ok = True)

URL = "https://github.com/pytorch/tutorials/raw/main/_static/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
    content = requests.get(URL + FILENAME).content
    (PATH / FILENAME).open("wb").write(content)

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

pyplot.imshow(x_train[0].reshape((28,28)), cmap="gray")

try:
    import google.colab

except ImportError:
    pyplot.show()
print(x_train.shape)

#x_train, y_train, x_valid, y_valid = map(torch.sensor, (x_train))

