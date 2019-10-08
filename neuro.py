PARAMETERS = 8

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
warnings.filterwarnings("ignore")
from keras.models import Sequential
from keras.layers import Dense
import numpy

dataset = numpy.loadtxt("dataset.txt", delimiter=",")

tests = len(dataset)
X = dataset[0 : (tests*3//4), 0 : PARAMETERS]
Y = dataset[0 : (tests*3//4), PARAMETERS]
X_test = dataset[(tests*3//4) : tests, 0 : PARAMETERS]
Y_test = dataset[(tests*3//4) : tests, PARAMETERS]

model = Sequential()

model.add(Dense(8, input_dim=PARAMETERS, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, epochs=180, batch_size=32, verbose=0, shuffle=1)

scores = model.evaluate(X, Y)

print(scores[1])

scores2 = model.evaluate(X_test, Y_test)

print(scores2[1])
