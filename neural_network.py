import numpy as np
import enum
import math_helper as mh
import json


class Activation(enum.Enum):
    Relu = 1
    LeakyRelu = 2
    Softmax = 3
    Linear = 4

class Layer():
    def __init__(self, shape, type = Activation.LeakyRelu, weights = None, bias = None):
        """Shape format: (# of inputs, # of neurons)"""
        self.shape = shape
        self.type = type
        self.bias = 0.0
        self.output = None
        self.leaky_slope = 0.05
        self.input = None

        self.set_weights(weights)
        self.set_bias(bias)

    def set_weights(self, weights = None):
        if(weights == None):
            self.weights = np.random.standard_normal(self.shape) * np.sqrt(2 / self.shape[0])
        else:
            self.weights = weights

    def set_bias(self, bias = None):
        if(bias != None):
            self.bias = bias

    def forward(self, input):
        self.input = input
        if(self.type == Activation.LeakyRelu):
            return self.leaky_forward(input)
        elif(self.type == Activation.Relu):
            return self.relu_forward(input)
        elif(self.type == Activation.Softmax):
            return self.softmax_forward(input)
        elif(self.type == Activation.Linear):
            return self.linear_forward(input)

    def leaky_forward(self, input):
        self.output = np.dot(input, self.weights) + self.bias
        self.output = np.piecewise(self.output, [self.output > 0, self.output <= 0], [lambda a: a, lambda a: a * self.leaky_slope])
        return self.output

    def relu_forward(self, input):
        self.output = np.dot(input, self.weights) + self.bias
        self.output = mh.relu(self.output)
        return self.output

    def softmax_forward(self, input):
        self.z_output = np.dot(input, self.weights) + self.bias
        self.output, self.softmax_sum = mh.duel_softmax(self.z_output, len(input.shape) - 1)
        return self.output

    def linear_forward(self, input):
        self.output = np.dot(input, self.weights) + self.bias
        return self.output

    def softmax_gradient(self):
        return (self.output / self.softmax_sum) * (np.exp(2 * self.z_output) / self.softmax_sum + 1)

    def leaky_gradient(self):
        return np.piecewise(
            self.output,
            [self.output > 0, self.output <= 0], 
            [lambda a: 1, lambda a: self.leaky_slope])

    def relu_gradient(self):
        return np.piecewise(
            self.output,
            [self.output > 0, self.output <= 0], 
            [lambda a: 1, lambda a: 0])

    def linear_gradient(self):
        return 1

    def gradient(self):
        if(self.type == Activation.Relu):
            return self.relu_gradient()
        if(self.type == Activation.LeakyRelu):
            return self.leaky_gradient()
        if(self.type == Activation.Softmax):
            return self.softmax_gradient()
        if(self.type == Activation.Linear):
            return self.linear_gradient()



class NeuralNetwork():
    def __init__(self, layers = None, heads = None, step = 0.001, weight_clip = 0.5, bias_clip = 0.1):
        self.layers = layers
        self.step = step
        self.heads = heads
        self.output = None
        self.weight_clip = weight_clip
        self.bias_clip = bias_clip
        if(layers != None):
            self.setup()

    def setup(self):
        self.length = len(self.layers)
        self.output_layer = self.layers[self.length - 1]
        self.input_layer = self.layers[0]

    def forward(self, input):
        """
        Input shape: (# of input sets, # of inputs)
        Output shape: (# of input sets, # of outputs)
        """
        self.input = np.array(input)
        if (self.heads == None):
            for i in range(self.length - 1):
                input = self.layers[i].forward(input)
            return self.output_layer.forward(input)
        else:
            output = []
            for i in range(self.length):
                input = self.layers[i].forward(input)
            for head in self.heads:
                output.append(head.forward(input))
            return output

    def loss(self, target, output = None, linear = True):
        loss = 0
        if(output == None):
            if(linear):
                loss = np.sum(np.abs(self.output_layer.output - target))
            else:
                loss = np.sum(np.square(self.output_layer.output - target))
        else:
            if(linear):
                loss = np.sum(np.abs(output - target))
            else:
                loss = np.sum(np.square(output - target))
        return loss

    def loss_heads(self, target, linear = True):
        losses = [self.loss(target[i], self.heads[i].output, linear) for i in range(len(self.heads))]
        return np.sum(losses)

    def backprop_adv_head(self, prob, adv):
        dEdo = [np.log(prob[i]) * adv[i] for i in range(len(self.heads))]
        self.backprop_head(dEdo)

    def backprop_target_head(self, head_targets):
        dEdo = [self.heads[i].output - head_targets[i] for i in range(len(self.heads))]
        self.backprop_head(dEdo)

    def backprop_advantage(self, adv, prob = None):
        if(prob == None):
            self.backprop(np.log10(self.output_layer.output) * adv)
        else:
            self.backprop(np.log10(prob) * adv)

    def backprop_target(self, target):
        """Target shape: (1, # of outputs)"""
        self.backprop(self.output_layer.output - target)

    def backprop_target_batch(self, target):
        self.backprop_batch(self.output_layer.output - target)

    def backprop_head(self, dEdo):
        """Target shape: (# of heads, specific head length)"""
        dEdzout_arry = []
        for i in range(len(self.heads)):
            dEdzout = dEdo[i] * self.heads[i].gradient()
            dEdzout = np.reshape(dEdzout, -1)
            update = np.tile(self.output_layer.output,(self.heads[i].shape[1],1)).transpose() * dEdzout * self.step
            self.heads[i].weights -= np.clip(update, -self.weight_clip, self.weight_clip)
            self.heads[i].bias -= np.clip(np.sum(dEdzout) * self.step, -self.bias_clip, self.bias_clip)
            dEdzout = np.sum(self.heads[i].weights * dEdzout, axis=1) * self.output_layer.gradient()
            dEdzout = np.reshape(dEdzout, -1)
            dEdzout_arry.append(dEdzout)
        update = np.tile(self.layers[self.length - 2].output,(self.output_layer.shape[1],1)).transpose() * np.mean(dEdzout_arry, axis=0) * self.step
        self.output_layer.weights -= np.clip(update, -self.weight_clip, self.weight_clip)
        self.output_layer.bias -= np.clip(np.sum(dEdzout) * self.step, -self.bias_clip, self.bias_clip)
        for i in range(self.length - 2, 0, -1):
            dEdzout = np.sum(self.layers[i+1].weights * dEdzout, axis=1) * self.layers[i].gradient()
            output = np.tile(self.layers[i-1].output,(self.layers[i].shape[1],1)).transpose()
            update = output * dEdzout * self.step
            self.layers[i].weights -= np.clip(update, -self.weight_clip, self.weight_clip)
            self.layers[i].bias -= np.clip(np.sum(dEdzout) * self.step, -self.bias_clip, self.bias_clip)
        dEdzout = np.sum(self.layers[1].weights * dEdzout, axis=1) * self.input_layer.gradient()
        update = np.tile(self.input, (self.input_layer.shape[1],1)).transpose() * dEdzout * self.step
        self.input_layer.weights -= np.clip(update, -self.weight_clip, self.weight_clip)
        self.input_layer.bias -= np.clip(np.sum(dEdzout) * self.step, -self.bias_clip, self.bias_clip)

    def backprop(self, dEdo):
        """Target shape: (1, # of outputs)"""
        #dEdo = self.output_layer.output - target
        dEdzout = dEdo * self.output_layer.gradient()
        dEdzout = np.reshape(dEdzout, -1)
        update = np.tile(self.layers[self.length - 2].output,(self.output_layer.shape[1],1)).transpose() * dEdzout * self.step
        self.output_layer.weights -= np.clip(update, -self.weight_clip, self.weight_clip)
        self.output_layer.bias -= np.clip(np.sum(dEdzout) * self.step, -self.bias_clip, self.bias_clip)
        for i in range(self.length - 2, 0, -1):
            dEdzout = np.sum(self.layers[i+1].weights * dEdzout, axis=1) * self.layers[i].gradient()
            output = np.tile(self.layers[i-1].output,(self.layers[i].shape[1],1)).transpose()
            update = output * dEdzout * self.step
            self.layers[i].weights -= np.clip(update, -self.weight_clip, self.weight_clip)
            self.layers[i].bias -= np.clip(np.sum(dEdzout) * self.step, -self.bias_clip, self.bias_clip)
        dEdzout = np.sum(self.layers[1].weights * dEdzout, axis=1) * self.input_layer.gradient()
        update = np.tile(self.input, (self.input_layer.shape[1],1)).transpose() * dEdzout * self.step
        self.input_layer.weights -= np.clip(update, -self.weight_clip, self.weight_clip)
        self.input_layer.bias -= np.clip(np.sum(dEdzout) * self.step, -self.bias_clip, self.bias_clip)

    def backprop_batch(self, dEdo):
        """Target shape: (# of targets, # of outputs)"""
        #dEdo = self.output_layer.output - target
        dEdzout = dEdo * self.output_layer.gradient()
        output = np.tile(self.layers[self.length - 2].output, (self.output_layer.shape[1], 1)) * np.reshape(dEdzout.T, (-1,1))
        update = np.reshape(output, (self.output_layer.shape[1], self.layers[self.length - 2].output.shape[0], self.layers[self.length - 2].output.shape[1]))
        update = np.mean(update, axis=1).transpose() * self.step
        self.output_layer.weights -= np.clip(update, -self.weight_clip, self.weight_clip)
        self.output_layer.bias -= np.clip(np.sum(dEdzout) * self.step, -self.bias_clip, self.bias_clip)
        for i in range(self.length - 2, 0, -1):
            d = np.reshape(dEdzout, (-1,1)) * np.tile(self.layers[i+1].weights, dEdzout.shape[0]).transpose()
            d = np.reshape(d, (dEdzout.shape[0], self.layers[i+1].shape[1], self.layers[i+1].shape[0]))
            dEdzout = np.sum(d, axis=1) * self.layers[i].gradient()
            output =  np.tile(self.layers[i - 1].output, (self.layers[i].shape[1],1)) * np.reshape(dEdzout.T, (-1,1))
            update = np.reshape(output, (self.layers[i].shape[1], self.layers[i - 1].output.shape[0], self.layers[i - 1].output.shape[1]))
            update = np.mean(update, axis=1).transpose() * self.step
            self.layers[i].weights -= np.clip(update, -self.weight_clip, self.weight_clip)
            self.layers[i].bias -= np.clip(np.sum(dEdzout) * self.step, -self.bias_clip, self.bias_clip)
        d = np.reshape(dEdzout, (-1,1)) * np.tile(self.layers[1].weights, dEdzout.shape[0]).transpose()
        d = np.reshape(d, (dEdzout.shape[0], self.layers[1].shape[1], self.layers[1].shape[0]))
        dEdzout = np.sum(d, axis=1) * self.input_layer.gradient()
        output =  np.tile(self.input, (self.input_layer.shape[1],1)) * np.reshape(dEdzout.T, (-1,1))
        update = np.reshape(output, (self.input_layer.shape[1], self.input.shape[0], self.input.shape[1]))
        update = np.mean(update, axis=1).transpose() * self.step
        self.input_layer.weights -= np.clip(update, -self.weight_clip, self.weight_clip)
        self.input_layer.bias -= np.clip(np.sum(dEdzout) * self.step, -self.bias_clip, self.bias_clip)

    def save(self, name="Network", as_np = True):
        print(f"Saving network as {name}...")
        data = {}
        layers = []
        heads = []
        for layer in self.layers:
            layers.append({"shape":layer.shape, 
                           "type":layer.type, 
                           "weights":layer.weights.tolist(), 
                           "bias":layer.bias,
                           "activation_id":layer.type.value})
        if(self.heads != None):
            for head in self.heads:
                heads.append({"shape":head.shape, 
                               "type":head.type, 
                               "weights":head.weights.tolist(), 
                               "bias":head.bias,
                               "activation_id":head.type.value})
        data["heads"] = heads
        data["layers"] = layers
        data["options"] = {"step":self.step,
                           "weight_clip":self.weight_clip,
                           "bias_clip":self.bias_clip}
        if(as_np):
            np.save(name, data)
        else:
            with open(name+".json", "w") as f:
                json.dump(data, f, indent = 4, sort_keys=True)
        print(f"Network saved")
        return name

    def load(self, location='Network', as_np = True):
        print(f"Loading Network from {location}...")
        data = {}
        layers = []
        heads = []
        try:
            if(as_np):
                data = np.load(location+".npy", allow_pickle=True).item()
            else:
                with open(location+".json","r") as f:
                    data = json.load(f)
                    print(isinstance(data, dict))
            for i in range(len(data["layers"])):
                layers.append(Layer(shape=data["layers"][i]["shape"], 
                                type = Activation(data["layers"][i]["activation_id"]), 
                                weights=data["layers"][i]["weights"], 
                                bias=data["layers"][i]["bias"]))

            for i in range(len(data["heads"])):
                heads.append(Layer(shape=data["heads"][i]["shape"], 
                                type = Activation(data["heads"][i]["activation_id"]), 
                                weights=data["heads"][i]["weights"], 
                                bias=data["heads"][i]["bias"]))

            self.__init__(layers = layers,
                          heads = heads,
                          step = data["options"]["step"], 
                          weight_clip = data["options"]["weight_clip"],
                          bias_clip = data["options"]["bias_clip"])
            print(f"Load success")
        except Exception as e:
            print(f"Load file from {location} failed {e}")


#nn = NeuralNetwork([Layer((21169, 25)), Layer((25,25)), Layer((25,25)), Layer((25,25)), Layer((25,25)), Layer((25,25)), Layer((25, 369), Activation.SoftmaxHead)], step=0.0001)
#nn.save("DR_actor")
#nn = NeuralNetwork([Layer((21169, 25)), Layer((25,25)), Layer((25,25)), Layer((25,25)), Layer((25, 1), Activation.Linear)], step=0.0001)
#nn.save("DR_critic")