# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 00:30:35 2016

@author: Camila Laranjeira
"""
import numpy as np
from sklearn import svm
from sklearn.cross_validation import KFold
from sklearn import preprocessing
import pickle

# Dataset info
project_root = '/home/laranjeira/projects/Pokemom-Recognition/features_labels/'
pikachu = 'pikachu.txt'
bulbasaur = 'bulbasaur.txt'
squirtle = 'squirtle.txt' 
charmander = 'charmander.txt'


def cross_validate(data, labels, classifier):
    """ data: [n_samples, n_features] 
        labels: [n_samples] (value is 0 to n_labels)"""
    kf = KFold(labels.size, n_folds=10)
    scores = []
    for k, (train, test) in enumerate(kf):
        classifier = classifier.fit(data[train], labels[train])
        score = classifier.score(data[test], labels[test])
        scores.append(score)        
        print('{} fold: {:.4f}'.format(k, score))
        
    return np.mean(scores)
    
def store_classifier(classifier):
     # ===== Store classifier ===== #    
    pickle.dump(classifier, open('classifier.p', 'wb'))
        
if __name__ == '__main__':    
    
    data = np.loadtxt(project_root + "features_" + pikachu)
    data = np.append(data, np.loadtxt(project_root + "features_" + bulbasaur), axis=0)
    data = np.append(data, np.loadtxt(project_root + "features_" + squirtle), axis = 0)    
    data = np.append(data, np.loadtxt(project_root + "features_" + charmander), axis = 0)   
        
    labels = np.loadtxt(project_root + "labels_" + pikachu)
    labels = np.append(labels, np.loadtxt(project_root + "labels_" + bulbasaur))
    labels = np.append(labels, np.loadtxt(project_root + "labels_" + squirtle))    
    labels = np.append(labels, np.loadtxt(project_root + "labels_" + charmander))    

    data = preprocessing.scale(data)    
 
    clf = svm.LinearSVC(class_weight='balanced', C=1e-4)  
    acc = cross_validate(data, labels, clf)
    print('Mean accuracy: {:.4f}'.format(acc))    
    
    clf = clf.fit(data, labels)
    store_classifier(clf)

