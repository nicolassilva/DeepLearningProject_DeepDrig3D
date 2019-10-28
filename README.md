## Sujet: Classification of ligand-binding pockets in proteins

Nicolas Silva (silva.nicolas.j@gmail.com)<br/>
Université Paris Diderot - Octobre 2019 - Projet Deep Learning

__Objectif__

Le but de ce projet est d'implémenter un réseau de neurones capable de discriminer des protéines en fonction de ses potentiels d'interaction électrostatiques avec son ligand.<br/><br/>
Ce travail est basé sur l'article suivant :
*Pu L., Govindaraj R.G., Lemoine J.M., Wu H.C. & Brylinski M. (2018). DeepDrug3D: Classification of ligand-binding pockets in proteins with a convolutional neural network. PLoS Comput Biol, 15(2).*

Exécution sous l'environnement python3

#### Répertoires et localisation des fichiers
*********************************************

Le répertoire contient :<br/>
	- un dossier *data* contenant les fichiers de données de potentiels d'intéractions des groupes hèmes, nucléotides, stéroïdes et contröle.<br/>
	- un dossier *doc* contenant le rapport du projet ainsi que l'article sur lequel est basé ce travail.<br/>
	- un dossier *results* contenant les différents modèles implémentés.<br/>
	- un dossier *src* contenant les scripts utilisés.<br/>
	- un fichier *.yml* permettant de recréer l'environnement conda utilisé.<br/>

#### Programme
**************

Il y a deux progammes à exécuter :<br/>
	- le programme *3dDrug.py* qui ne prend pas d'arguments, qui permet tout simplement de faire tourner la création du réseau.<br/>
	- le programme *evaluate3dDrug.py* qui prend deux arguments: le premier est le model à évaluer, le second est le nom du fichier de sortie du graphe des courbes ROC. Ce programme qui permet de tester le réseau implémenté sur un jeu de données.<br/>
	
''' Ex: python3 evaluate3dDrug.py my_model.h5 ROC_model.svg'''

#### Résultats
*************************************

Les modèles et les graphiques des courbes ROC sont dans l dossier *results*.<br/>
La partie des résultats est décrite dans le rapport dans le dossier *doc*.

#### Modules python importés
*****************************

Numpy, Random, Keras (Input, Model, callbacks), Keras.backend (set_image_data_format), Keras.layers (Activation, Conv3D, Dense, Dropout, Flatten, MaxPooling3D)
