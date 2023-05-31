import numpy as np

# this nnet doesn't have an implementation for regularization, or a bias neuron because it's an incredibly small nnet
class XorNNet:
	def __init__(self):
		# predetermined weights (and size of the network)
		self.input_to_hidden_weights_per_neuron = [[-1, -1], [1, 1]]

		self.hidden_to_output_weights_per_neuron = [[1, 1]]

		#training data
		self.truth_table = [[[0, 0], 0], [[1, 1], 0], [[1, 0], 1], [[0, 1], 1]]

	# sigmoid used for activation
	def _sigmoid(self, x):
		return float(1/(1 + float(np.exp(-x))))

	# forward propagation
	def forward(self, input_layer_arg_first, input_layer_arg_second):
		# print(input_layer_arg_first)
		# print(input_layer_arg_second)
		first_hidden_first_weight  = self.input_to_hidden_weights_per_neuron[0][0]
		first_hidden_second_weight = self.input_to_hidden_weights_per_neuron[0][1]
		first_hidden_layer_neuron = self._sigmoid(input_layer_arg_first*first_hidden_first_weight+input_layer_arg_second*input_layer_arg_first)
		# print(first_hidden_layer_neuron)

		second_hidden_first_weight  = self.input_to_hidden_weights_per_neuron[1][0]
		second_hidden_second_weight = self.input_to_hidden_weights_per_neuron[1][1]
		second_hidden_layer_neuron = self._sigmoid(input_layer_arg_second*second_hidden_first_weight+input_layer_arg_second*input_layer_arg_second)
		# print(second_hidden_layer_neuron)

		first_hidden_to_output_weight  = self.hidden_to_output_weights_per_neuron[0][0]
		second_hidden_to_output_weight = self.hidden_to_output_weights_per_neuron[0][1]
		output_layer_neuron = self._sigmoid(first_hidden_layer_neuron*first_hidden_to_output_weight+second_hidden_layer_neuron*second_hidden_to_output_weight)
		# print(output_layer_neuron)

		return output_layer_neuron

	#def backward(input_layer_arg_first, input_layer_arg_second, expected_output):



	def main(self):
		feedforward = self.forward(0, 0)
		if feedforward >= 0.5:
			return 1
		return 0

if __name__ == "__main__":
	xnnt = XorNNet()
	print(xnnt.main())

