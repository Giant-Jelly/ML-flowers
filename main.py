import numpy as np
from matplotlib import pyplot as plt


# Data describes the data of the flowers we have with width, height and type (0 - blue, 1 - red)
data = [
	[5,	 	1.5,	1],
	[2,	 	1,		0],
	[4,	 	1.5,	1],
	[3,	 	1,		0],
	[3.5,	.5,		1],
	[2,	 	.5,		0],
	[5.5,	1,		1],
	[1,	 	1,		0],
]

# This flower we don't know the type, the network will find it!
mystery_flower = [4.5, 1]

# Network Structure, 2 inputs 1 output, no hidden layers. Super basic!
#    O 		Flower type
#   / \ 	weight1, weight2, bias
# O    O 	length, width

w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn()


# squish any number to between 0 and 1
def sigmoid(x):
	return 1/(1 + np.exp(-x))


# the derivative (the steepness of the slop) of sigmoid
def d_sigmoid(x):
	return sigmoid(x) * (1 - sigmoid(x))


T = np.linspace(-5, 5, 100)
plt.plot(T, sigmoid(T), c="r")
plt.plot(T, d_sigmoid(T), c="b")
plt.show()

# Scatter of data
plt.axis([0, 6, 0, 6])
plt.grid()
for i in range(len(data)):
	point = data[i]
	color = "r" if point[2] == 1 else 'b'
	plt.scatter(point[0], point[1], c=color)

plt.show()

# Training loop

learning_rate = 0.2

costs = []

for i in range(50000):
	# get a random int and select random data for training
	ri = np.random.randint(len(data))
	point = data[ri]
	target = point[2]

	# calculate the weighted average (prediction according to the weights) with the data
	wa = point[0] * w1 + point[1] * w2 + b
	pred = sigmoid(wa)
	if i % 10000 == 0:
		print(pred, ' - ', target)

	# A value representing how wrong the prediction was
	cost = np.square(pred - target)

	costs.append(cost)

	# Derivatives
	d_cost_pred = 2 * (pred - target)
	d_pred_wa = d_sigmoid(wa)

	d_wa_w1 = point[0]
	d_wa_w2 = point[1]
	d_wa_b = 1

	d_cost_wa = d_cost_pred * d_pred_wa
	d_cost_w1 = d_cost_wa * d_wa_w1
	d_cost_w2 = d_cost_wa * d_wa_w2
	d_cost_b = d_cost_wa * d_wa_b

	w1 = w1 - learning_rate * d_cost_w1
	w2 = w2 - learning_rate * d_cost_w2
	b = b - learning_rate * d_cost_b

plt.plot(costs)
plt.show()


def predict(d):
	return sigmoid(d[0] * w1 + d[1] * w2 + b)


def get_color(x):
	if round(x) == 1:
		return "red"
	else:
		return "blue"


dataCount = len(data)
correct = 0
for i in range(dataCount):
	pred = predict(data[i])
	if round(pred) == data[i][2]:
		correct += 1

print((correct/dataCount)*100, '%')

pred = predict(mystery_flower)
print("The mystery flower is", get_color(pred))

flower = [0, 1]
pred = predict(flower)
print("Your flower is", get_color(pred))
