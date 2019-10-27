from keras import Input, Model, callbacks
import numpy as np
from keras.layers import Activation, Conv3D, Dense, Dropout, Flatten, MaxPooling3D
from keras.backend import set_image_data_format
import random as rd

#Function to read and save X and Y data
def open_dat(fil, typ, nb):
	encod = {"h" : [1, 0, 0, 0], "n" : [0, 1, 0, 0], "s" : [0, 0, 1, 0], "c" : [0, 0, 0, 1]}
	f = open(fil, "r")
	f = f.read().splitlines()
	x_data = []
	y_data = []
	dat = []
	for i in f[0:nb]:
		dat = np.load("../data/deepdrug3d_voxel_data/"+i+".npy")
		dat = np.squeeze(dat)
		x_data.append(dat)
		y_data.append(encod[typ])
	return x_data, y_data

(x_heme, y_heme) = open_dat("../data/heme.list", "h", 300) #Max 596
(x_nucle, y_nucle) = open_dat("../data/nucleotid.list", "n", 300) #Max 1553
(x_steroid, y_steroid) = open_dat("../data/steroid.list", "s", 69) #Max 69
(x_control, y_control) = open_dat("../data/control.list", "c", 300) #Max 1946

#Creating one_hot array
X = np.array(x_heme + x_nucle + x_steroid + x_steroid + x_steroid + x_control)
Y = np.array(y_heme + y_nucle + y_steroid + y_steroid + y_steroid + y_control)

X_Y = list(zip(X,Y))
rd.shuffle(X_Y)
X, Y = zip(*X_Y)

oH_x_train = np.array(X)
oH_y_train = np.array(Y)

#Change channel from last to first
set_image_data_format('channels_first')

### Model
inputs = Input(shape=(14, 32, 32, 32))

conv1 = Conv3D(4, (2,2,2), padding="valid", activation="relu")(inputs)
conv2 = Conv3D(16, (2,2,2), padding="valid", activation="relu")(conv1)
drop1 = Dropout(0.2)(conv2)
pool1 = MaxPooling3D(pool_size=(2,2,2))(drop1)
drop2 = Dropout(0.2)(pool1)

flat1 = Flatten()(drop2)
outputs = Dense(4, activation = "softmax")(flat1)
model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

earlyStop = callbacks.EarlyStopping(monitor="val_loss", patience=10, mode="min")

best_model_file = ('best_model.h5')
best_model = callbacks.ModelCheckpoint(best_model_file, monitor='val_loss', save_best_only='TRUE')

my_model = model.fit(oH_x_train, oH_y_train, batch_size=16, epochs=30, validation_split = 0.3, callbacks=[earlyStop,best_model])
