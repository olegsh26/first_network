import numpy as np

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

training_inputs = np.array([[2-2],
							[2+2],
							[2*2],
							[2/2]])

training_outputs = np.array([[0,4,4,1]]).T

np.random.seed(1)

synaptic_weights = 2 * np.random.random((0,5)) - 1 

print("random initializing scales:")
print(synaptic_weights)

for i in range(200000):
	input_layer = training_inputs
	outputs = sigmoid( np.dot(input_layer, synaptic_weights) )

	err = training_outputs - outputs
	adjustments = np.dot( input_layer.T, err * (outputs * (1 - outputs)) )

	synaptic_weights += adjustments


print("scales after learning:")
print(synaptic_weights)

print("result after learning:")
print(outputs)


#test

new_inputs = np.array([1,1,0])
output = sigmoid( np.dot( new_inputs, synaptic_weights ) )

print("new situation:")
print(output)