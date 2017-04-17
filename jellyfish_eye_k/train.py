import numpy as np
np.random.seed(1337)

from jellyfish_eye_k.data_set import load_data
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.models import Sequential, save_model


(x_train, y_train), (x_validation, y_validation), (x_test, y_test) = load_data()

model = Sequential((
    Conv2D(32, 5, activation='relu', input_shape=x_train[0].shape),
    Conv2D(64, 5, activation='relu'),
    MaxPooling2D(),
    Dropout(0.5),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=20, epochs=5, verbose=1, validation_data=(x_validation, y_validation))

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss: {0}'.format(score[0]))
print('Test accuracy: {0}'.format(score[1]))

save_model(model, './jellyfish_eye.h5')
del model
