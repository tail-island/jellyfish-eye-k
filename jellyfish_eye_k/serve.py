import numpy as np
np.random.seed(1337)
    
from keras.models import load_model


model = load_model('./jellyfish_eye.h5')


def classify(x):
    return model.predict_classes(np.array((x,)), batch_size=1, verbose=0)[0]


if __name__ == '__main__':
    import matplotlib.animation as animation
    import matplotlib.pyplot as plot
    import time
    
    from jellyfish_eye_k.data_set import load_data

    
    CLASS_CAPTIONS = ["Calendar", "Drink", "Tissue Paper"]

    _, _, (x_test, y_test) = load_data()

    y_pred = []

    starting_time = time.time()
    for x in x_test:
        y_pred.append(classify(x))
    finishing_time = time.time()

    accuracy = sum(1 if y_test == y_pred else 0 for y_test, y_pred in zip(y_test, y_pred)) / len(y_pred)
    
    print('Elapsed time: {0:.4f} sec'.format((finishing_time - starting_time) / len(y_pred)))
    print('Accuracy = {0:.4f}'.format(accuracy))

    figure = plot.figure()
    images = []

    plot.imshow([[[0.0, 0.0, 0.0, 1.0] for x in range(100)] for y in range(100)])
    
    for x, y_true, y_pred in zip(x_test, y_test, y_pred):
        images.append([plot.imshow(x),
                       plot.text(50, 10, 'Label: {0}'.format(CLASS_CAPTIONS[y_true]), color='w', horizontalalignment='center', size=20),
                       plot.text(50, 15, 'Answer: {0}'.format(CLASS_CAPTIONS[y_pred]), color='w', horizontalalignment='center', size=20),
                       plot.text(50, 25, 'Correct!' if y_true == y_pred else 'Wrong...', color='c' if y_true == y_pred else 'm', horizontalalignment='center', size=20)])

    images.append([plot.text(50, 50, 'Accuracy = {0:.4f}'.format(accuracy), color='w', horizontalalignment='center', size=20)])

    artist_animation = animation.ArtistAnimation(figure, images, interval=500, repeat=False)

    plot.get_current_fig_manager().full_screen_toggle()
    plot.show()
