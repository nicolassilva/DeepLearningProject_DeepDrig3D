from keras import *
import numpy as np
from keras.models import load_model
from keras.layers import Activation, Conv3D, Dense, Dropout, Flatten, MaxPooling3D
from keras.backend import set_image_data_format
import random as rd

#Function to read and save X and Y data
def open_dat(fil, typ, nb):
	encod = {"h" : [1., 0., 0., 0.], "n" : [0., 1., 0., 0.], "s" : [0., 0., 1., 0.], "c" : [0., 0., 0., 1.]}
	#encod = {"h" : [1., 0., 0.], "n" : [0., 1., 0.], "c" : [0., 0., 1.]}
	#encod = {"h" : [1., 0.], "n" : [0., 1.]}
	f = open(fil, "r")
	f = f.read().splitlines()
	x_data = []
	y_data = []
	dat = []
	for i in f[-nb:]:
		dat = np.load("../../data/deepdrug3d_voxel_data/"+i+".npy")
		dat = np.squeeze(dat)
		x_data.append(dat)
		y_data.append(encod[typ])
	return x_data, y_data

#Data to test the model
(x_heme, y_heme) = open_dat("../../data/heme.list", "h", 200) #Max 596
(x_nucle, y_nucle) = open_dat("../../data/nucleotid.list", "n", 200) #Max 1553
(x_steroid, y_steroid) = open_dat("../../data/steroid.list", "s", 69) #Max 69
(x_control, y_control) = open_dat("../../data/control.list", "c", 200) #Max 1946

#Creating one_hot array
X = np.array(x_heme + x_nucle + x_steroid + x_control)
Y = np.array(y_heme + y_nucle + y_steroid + y_control)

#Shuffle arrays
X_Y = list(zip(X,Y))
rd.shuffle(X_Y)
X, Y = zip(*X_Y)

oH_x_train = np.array(X)
oH_y_train = np.array(Y)

#Load model
model = load_model('../model_0.93.h5')
#model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
accuracy = model.evaluate(oH_x_train, oH_y_train)
print(model.metrics_names)
print(accuracy)
