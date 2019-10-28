import numpy as np
from keras.models import load_model
import random as rd
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle
import argparse


#Function to read and save X and Y data
def open_dat(fil, typ, nb):
	encod = {"h" : [1, 0, 0, 0], "n" : [0, 1, 0, 0], "s" : [0, 0, 1, 0], "c" : [0, 0, 0, 1]}
	f = open(fil, "r")
	f = f.read().splitlines()
	x_data = []
	y_data = []
	dat = []
	for i in f[-nb:]:
		dat = np.load("../data/deepdrug3d_voxel_data/"+i+".npy")
		dat = np.squeeze(dat)
		x_data.append(dat)
		y_data.append(encod[typ])
	return x_data, y_data


parser = argparse.ArgumentParser(description='Classification of ligand-binding pocket test')
parser.add_argument('-f', metavar='Model', type=str, help='Model file .h5', required=True)
parser.add_argument('-o', metavar='Graph', type=str, help='Output ROC curves')
args = parser.parse_args()
model_file = args.f
save_plot = args.o

#Data to test the model
(x_heme, y_heme) = open_dat("../data/heme.list", "h", 300) #Max 596
(x_nucle, y_nucle) = open_dat("../data/nucleotid.list", "n", 300) #Max 1553
(x_steroid, y_steroid) = open_dat("../data/steroid.list", "s", 69) #Max 69
(x_control, y_control) = open_dat("../data/control.list", "c", 300) #Max 1946

#Creating one_hot array
X = np.array(x_heme + x_nucle + x_steroid + x_control)
Y = np.array(y_heme + y_nucle + y_steroid + y_control)

#Shuffle arrays
X_Y = list(zip(X,Y))
rd.shuffle(X_Y)
X, Y = zip(*X_Y)

oH_x_test = np.array(X)
oH_y_test = np.array(Y)

#Load model
model = load_model(model_file)
model.summary()
accuracy = model.evaluate(oH_x_test, oH_y_test)
print(model.metrics_names)
print(accuracy)

###Courbe ROC par classe
n_classes = 4
y_score = model.predict(oH_x_test)
y_test = oH_y_test
fpr = dict()
tpr = dict()
thresholds_keras = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], thresholds_keras[i] = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves
plt.figure(1)
colors = cycle(['forestgreen', 'red', 'blue', 'black'])
classe = ['Hemes','Nucleotides','Steroïdes','Controls']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='Class {0} (AUC = {1:0.2f})'
             ''.format(classe[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC curves')
plt.legend(loc="lower right")
plt.gca().set_aspect('equal',adjustable='box')
if len(save_plot) > 0:
	plt.savefig(save_plot)
plt.show()
