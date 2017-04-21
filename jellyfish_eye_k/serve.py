import numpy as np
np.random.seed(1337)

from keras.models import load_model


model = load_model('./jellyfish_eye.h5')


def classify(x):
    return model.predict_classes(np.array((x,)), batch_size=1, verbose=0)[0]


if __name__ == '__main__':
    import matplotlib.pyplot as plot
    import time

    from jellyfish_eye_k.data_set import load_data


    _, _, (x_test, y_test) = load_data()

    y_pred = []

    starting_time = time.time()
    for x in x_test:
        y_pred.append(classify(x))
    finishing_time = time.time()

    print('Elapsed time: {0:.4f} sec'.format((finishing_time - starting_time) / len(y_pred)))
    print('Accuracy = {0:.4f}'.format(sum(1 if y_test == y_pred else 0 for y_test, y_pred in zip(y_test, y_pred)) / len(y_pred)))

    for x, y_true, y_pred in zip(x_test, y_test, y_pred):
        print('{0} : {1}'.format(y_true, y_pred))

        plot.imshow([[[0.0, 0.0, 1.0, 1.0] for x in range(100)] for y in range(100)])
        plot.imshow(x)
        plot.show()
