import numpy as np
import random
import math
from numpy import linalg as LA

X_Cours = np.array([[1, 2, 1], [1, 0, -1], [1, -2, -1], [1, 0, 2]])
t_Cours = np.array([1, 1, -1, -1])

X_ET = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
t_ET = np.array([-1, -1, -1, 1])

X_XOR = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
t_XOR = np.array([-1, 1, 1, 1])

bias = 1
W = np.array([1, 1, 1])
alpha = 0.1

#Vérifie si le vecteur est correctement classé
def IsClassed(tau, y):
    cpt = 0
    for val in tau:
        if val != y[cpt]:
            return False
        cpt = cpt + 1
    return True

#Perceptron version online
def PerceptronIncremental(X, W, t, bias, alpha):
    y = np.ones(len(X))
    cpt = 0

    while not IsClassed(t, y):
        val = random.randint(0, len(X) - 1)
        x_prime = X[val]
        W[0] = bias
        if np.transpose(W).dot(x_prime) > 0:
            y_prime = 1
        else:
            y_prime = -1

        if y_prime != t[val]:
            e = alpha * (t[val] - y_prime)
            delta_w = x_prime.dot(e)
            W = W + delta_w
            bias = bias + e * 1
        y[val] = y_prime
        cpt = cpt + 1

    return y, W, cpt

#Perceptron version batch
def PerceptronBatch(X, W, t, bias, alpha):
    y = np.ones(len(X))
    nb_iter = 0
    while not IsClassed(t, y):
        cpt = 0
        for val in X:
            x_prime = val
            W[0] = bias
            if np.transpose(W).dot(x_prime) > 0:
                y_prime = 1
            else:
                y_prime = -1

            if y_prime != t[cpt]:
                e = alpha * (t[cpt] - y_prime)
                delta_w = x_prime.dot(e)
                W = W + delta_w
                bias = bias + e * 1
            y[cpt] = y_prime
            cpt = cpt + 1
            nb_iter = nb_iter + cpt
        nb_iter = nb_iter + 1
    return y, W, nb_iter

#Génération des données aléatoires et des poids optimaux
def LSAleatoire(P, N):
    X = np.random.rand(P, N)
    X = 2 * X -1
    W = np.random.rand(N + 1, 1)
    W = 2 * W -1
    t = []
    X = np.insert(X, 0, 1, axis = 1)
    for val in X:
       if np.dot(val, W) <= 0:
           t.append(-1)
       else:
           t.append(1)
    W_tmp = []
    for val in W:
        W_tmp.append(val[0])

    return X, W_tmp, t


def PerceptronEleveIncre(P, N):
    pere = LSAleatoire(P, N)
    t_pere = pere[2]
    W_pere = pere[1]
    X_pere = pere[0]
    eleve = PerceptronIncremental(X_pere, W_pere, t_pere, 1, alpha)
    W_fils = eleve[1]
    R = math.cos(np.dot(W_pere, W_fils) / np.dot(LA.norm(W_pere), LA.norm(W_fils)))
    return eleve[2], R

def PerceptronEleveBatch(P, N):
    pere = LSAleatoire(P, N)
    t_pere = pere[2]
    W_pere = pere[1]
    X_pere = pere[0]
    eleve = PerceptronBatch(X_pere, W_pere, t_pere, 1, alpha)
    W_fils = eleve[1]
    R = math.cos(np.dot(W_pere, W_fils) / np.dot(LA.norm(W_pere), LA.norm(W_fils)))
    return eleve[2], R

moy_it = 0
moy_R = 0
nb = 50
for i in range(nb):
    p = PerceptronEleveIncre(500, 1000)
    moy_it = moy_it + p[0]
    moy_R = moy_R + p[1]

print(moy_it/nb)
print(moy_R/nb)

