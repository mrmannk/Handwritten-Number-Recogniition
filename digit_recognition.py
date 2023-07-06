# import the necessary packages and mnist dataset
import numpy as np
import matplotlib.pyplot as plt
# import the MNIST dataset
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
# load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# plot 4 images as gray scale
X_train = X_train.reshape(X_train.shape[0], 28 * 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28 * 28).astype('float32')
# normalize inputs from 0-255 to 0-1
X_train /= 255
X_test /= 255
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
# sequence the model
model = Sequential()
# add layers to the model
model.add(Dense(512, input_shape=(28 * 28,), activation='relu'))
# reduce overfitting
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train , y_train, epochs=10, batch_size=128, validation_data=(X_test, y_test))
# evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss:.4f}')
print(f'Test accuracy: {accuracy:.4f}')
# predict the first images in the test set
predictions = model.predict(X_test)
# plot the images in the test set
plt.plot(predictions[363], label='Predicted', color='red')
plt.xlabel('Predicted Value')
plt.ylabel('Probability')
plt.title('Predicted Probabilities for First Image in Test Set')
# show the plot
plt.show()