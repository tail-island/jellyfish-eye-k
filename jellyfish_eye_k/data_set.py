import numpy as np
import os

from funcy import mapcat, rpartial
from operator import attrgetter, contains, methodcaller


def load_data(data_path='./data'):
    def rgbz(line_string):
        return map(float, line_string.split())

    def load_image(image_path):
        with open(image_path) as file:
            return np.reshape(np.array(tuple(mapcat(rgbz, file))), (100, 100, 4))

    def load_object(object_path):
        return map(load_image, filter(rpartial(contains, '.txt'), map(attrgetter('path'), os.scandir(object_path))))

    def load_class(class_path):
        return np.array(tuple(mapcat(load_object, map(attrgetter('path'), os.scandir(class_path)))))

    def train_validate_and_test(x):
        np.random.shuffle(x)

        return x[20:], x[:20], x[:20]  # データ数が少ないので、検証データとテスト・データは同じにします。

    def dataset(xs):
        xs = tuple(xs)

        x = np.concatenate(xs)
        y = np.concatenate([np.full(x.shape[0], i, np.int32) for i, x in enumerate(xs)])

        i = np.arange(x.shape[0])
        np.random.shuffle(i)

        return x[i], y[i]

    return map(dataset, zip(*map(train_validate_and_test, map(load_class, sorted(map(attrgetter('path'), os.scandir(data_path)))))))
