import numpy as np
import random as rnd


def generar_puntos(N, funcion, rango):
    puntos = np.random.uniform(rango[0], rango[1], (N, 2))  # arreglo de [[x,y],[x2,y2], ... ]
    reales = np.array([])
    deseados = np.array([])
    for i in range(N):
        reales.append(rnd.randint(0, 1))
        valor = 0  # abajo
        if funcion(puntos[i][0]) < puntos[i][1]:
            valor = 1  # arriba
        deseados.append(valor)
    return puntos, reales, deseados



# Main
fx = lambda x : 2*x + 10
points, realOutput = n_points(20, fx)
Ws = np.random.rand(2)
perceptron =
desiredOutput =
