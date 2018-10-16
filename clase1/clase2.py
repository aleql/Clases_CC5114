import numpy as np
import matplotlib.pyplot as plt

from clase3 import SigmoidNeuron, PerceptronNeuron


def generate_points(N, function, range):
    points = np.random.uniform(range[0], range[1], (N, 2))  # arreglo de [[x,y],[x2,y2], ... ]
    desiredOutput = []
    i = 0
    while i < N:
        valor = 0  # abajo
        if (function(points[i][0])) < points[i][1]:
            valor = 1  # arriba
        desiredOutput.append(valor)
        i += 1
    return points, np.array(desiredOutput)



# Main
N = 100
fx = lambda x : 2*x + 10
points, desiredOutput = generate_points(N, fx, [-60, 60])
Ws = np.random.uniform(-2, 2, (2, 1))
bias = np.random.uniform(-2, 2)
perceptron = SigmoidNeuron(Ws, bias)


# Train
x, y = points.T
accuracies = []
for t in range(100):
    realOutput = perceptron.feed(points)
    perceptron.train(desiredOutput, realOutput, 0.1, [x, y])

    # Obtener accuracy
    correct = 0
    for do, ro in zip(desiredOutput, realOutput):
        if do == ro:
            correct += 1
    accuracy = correct/len(desiredOutput)
    accuracies.append(accuracy)

# Plot scatterplot
colors = list(map(lambda c: 'b' if c == 1 else 'r', realOutput))
plt.scatter(x, y, c=colors)
line_x = range(-60, 60)
line_y = list(map(fx, line_x))
plt.plot(line_x, line_y)
plt.show()

# Plot accuracies
plt.plot(accuracies)
plt.show()


# perceptron.train(desiredOutput, realOutput, 0.1, points)







