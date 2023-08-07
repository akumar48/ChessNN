import random
import math

class Neuron:
    def __init__(self, num_inputs):
        self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
        self.bias = random.uniform(-1, 1)
        
    def activate(self, inputs):
        weighted_sum = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        return self.tanh(weighted_sum)

    def tanh(self, x):
        return math.tanh(x)

class MLP:
    def __init__(self, num_inputs, num_hidden_layers, num_hidden_neurons, num_outputs):
        self.input_layer = [Neuron(num_inputs) for _ in range(num_hidden_neurons)]
        self.hidden_layers = [[Neuron(num_hidden_neurons) for _ in range(num_hidden_neurons)] for _ in range(num_hidden_layers)]
        self.output_layer = [Neuron(num_hidden_neurons) for _ in range(num_outputs)]
        
    def forward_propagation(self, inputs):
        hidden_outputs = [neuron.activate(inputs) for neuron in self.input_layer]
        for hidden_layer in self.hidden_layers:
            hidden_outputs = [neuron.activate(hidden_outputs) for neuron in hidden_layer]
        final_outputs = [neuron.activate(hidden_outputs) for neuron in self.output_layer]
        return final_outputs
    
    def train(self, inputs, targets, learning_rate=0.1):
        # Perform forward propagation to get the outputs
        hidden_outputs = [neuron.activate(inputs) for neuron in self.input_layer]
        for hidden_layer in self.hidden_layers:
            hidden_outputs = [neuron.activate(hidden_outputs) for neuron in hidden_layer]
        final_outputs = [neuron.activate(hidden_outputs) for neuron in self.output_layer]

        # Calculate output layer errors and update weights and biases
        output_errors = [target - output for target, output in zip(targets, final_outputs)]
        for i in range(len(self.output_layer)):
            neuron = self.output_layer[i]
            for j in range(len(neuron.weights)):
                neuron.weights[j] += learning_rate * output_errors[i] * hidden_outputs[j]
            neuron.bias += learning_rate * output_errors[i]

        # Backpropagate hidden layer errors and update weights and biases
        for i in reversed(range(len(self.hidden_layers))):
            hidden_layer = self.hidden_layers[i]
            for j in range(len(hidden_layer)):
                error = 0.0
                for neuron in self.output_layer:
                    error += (neuron.weights[j] * output_errors[i])
                neuron = hidden_layer[j]
                for k in range(len(neuron.weights)):
                    neuron.weights[k] += learning_rate * error * inputs[k]
                neuron.bias += learning_rate * error

# Update the test functionality to calculate the score per game
def test_network(network, test_data):
    cumulative_score = 0.0
    for test_input, expected_output in test_data:
        output = network.forward_propagation(test_input)
        score_per_game = output[0] - expected_output[0]
        cumulative_score += score_per_game
    return cumulative_score / len(test_data)

# Load the training and test data
def read_data(filename):
    with open(filename, "r") as file:
        lines = file.readlines()

    data = []
    for line in lines:
        input_data, output_data = line.strip().split(" ")
        inputs = [float(bit) for bit in input_data.split(",")]
        outputs = [float(bit) for bit in output_data.split(",")]
        data.append((inputs, outputs))

    return data

if __name__ == "__main__":
    num_inputs = 85
    num_hidden_layers = 5
    num_hidden_neurons = 50
    num_outputs = 37

    mlp = MLP(num_inputs, num_hidden_layers, num_hidden_neurons, num_outputs)

    # Load the training data
    training_data = read_data("input_2a.txt")

    epochs = 1000
    for epoch in range(epochs):
        for inputs, targets in training_data:
            mlp.train(inputs, targets)

    # Load the test data
    test_data = read_data("input_7b.txt")  # Update the filename here

    # Test the trained MLP
    cumulative_score = test_network(mlp, test_data)
    print("Cumulative Perception Score:", cumulative_score)
