import numpy as np
import enum
import math_helper as mh
import json
import utilities as u
import sys

class Activation(enum.Enum):
    Relu = 1
    LeakyRelu = 2
    Linear = 3

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
            self.weights = np.array(weights)

    def set_bias(self, bias = None):
        if(bias != None):
            self.bias = bias

    def forward(self, input):
        self.input = input
        if(self.type == Activation.LeakyRelu):
            return self.leaky_forward(input)
        elif(self.type == Activation.Relu):
            return self.relu_forward(input)
        elif(self.type == Activation.Linear):
            return self.linear_forward(input)

    def leaky_forward(self, input):
        self.z_output = np.dot(input, self.weights) + self.bias
        self.output = np.piecewise(self.z_output, [self.z_output > 0, self.z_output <= 0], [lambda a: a, lambda a: a * self.leaky_slope])
        return self.output

    def relu_forward(self, input):
        self.z_output = np.dot(input, self.weights) + self.bias
        self.output = mh.relu(self.z_output)
        return self.output

    def linear_forward(self, input):
        self.z_output = np.dot(input, self.weights) + self.bias
        self.output = self.z_output
        return self.output

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
        if(self.type == Activation.Linear):
            return self.linear_gradient()



class NeuralNetwork():
    def __init__(self, layers = None, step = 0.001, weight_clip = 0.5, bias_clip = 0.1, reg_coefficient = 0):
        self.layers = layers
        self.step = step
        self.output = None
        self.weight_clip = weight_clip
        self.bias_clip = bias_clip
        self.reg_coefficient = reg_coefficient
        if(layers != None):
            self.setup()

    def setup(self):
        self.length = len(self.layers)
        self.output_layer = self.layers[self.length - 1]
        self.input_layer = self.layers[0]

    def l2norm(self):
        sum = 0.0
        for layer in self.layers:
            sum += np.linalg.norm(layer.weights)
        return sum

    def get_reg(self):
        sum = 0.0
        for layer in self.layers:
            sum += np.sum(layer.weights)
        return sum * 0.008

    def forward(self, input):
        """
        Input shape: (# of input sets, # of inputs)
        Output shape: (# of input sets, # of outputs)
        """
        self.input = np.array(input)
        for i in range(self.length - 1):
            input = self.layers[i].forward(input)
        return self.output_layer.forward(input)

    def loss(self, target, output = None, linear = True):
        loss = 0
        if(output is None):
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

    def backprop_advantage(self, adv, prob = None):
        if(prob is None):
            return self.backprop(np.log10(mh.softmax(self.output_layer.output)) * adv)
        else:
            return self.backprop(np.log10(prob) * adv)

    def backprop_target(self, target):
        """Target shape: (# of outputs, )"""
        return self.backprop(self.output_layer.output - target)

    def backprop_target_batch(self, target):
        return self.backprop_batch(self.output_layer.output - target)

    def backprop(self, dEdo):
        """Target shape: (# of outputs, )"""
        #dEdo = self.output_layer.output - target
        dEdzout = dEdo * self.output_layer.gradient()
        dEdzout = np.reshape(dEdzout, -1)
        update = np.tile(self.layers[self.length - 2].output,(self.output_layer.shape[1],1)).transpose() * dEdzout * self.step
        self.output_layer.weights -= np.clip(update, -self.weight_clip, self.weight_clip) 
        self.output_layer.weights -= self.step * self.reg_coefficient * self.output_layer.weights
        self.output_layer.bias -= np.clip(np.sum(dEdzout) * self.step, -self.bias_clip, self.bias_clip)
        for i in range(self.length - 2, 0, -1):
            dEdzout = np.sum(self.layers[i+1].weights * dEdzout, axis=1) * self.layers[i].gradient()
            output = np.tile(self.layers[i-1].output,(self.layers[i].shape[1],1)).transpose()
            update = output * dEdzout * self.step
            self.layers[i].weights -= np.clip(update, -self.weight_clip, self.weight_clip) + self.reg_coefficient * self.layers[i].weights
            self.layers[i].weights -= self.step * self.reg_coefficient * self.layers[i].weights
            self.layers[i].bias -= np.clip(np.sum(dEdzout) * self.step, -self.bias_clip, self.bias_clip)
        dEdzout = np.sum(self.layers[1].weights * dEdzout, axis=1) * self.input_layer.gradient()
        update = np.tile(self.input, (self.input_layer.shape[1],1)).transpose() * dEdzout * self.step
        self.input_layer.weights -= np.clip(update, -self.weight_clip, self.weight_clip) + self.reg_coefficient * self.input_layer.weights
        self.input_layer.weights -= self.step * self.reg_coefficient * self.input_layer.weights
        self.input_layer.bias -= np.clip(np.sum(dEdzout) * self.step, -self.bias_clip, self.bias_clip)
        dEdzout = np.sum(self.input_layer.weights * dEdzout, axis=1)
        return dEdzout

    def backprop_batch(self, dEdo):
        """Target shape: (# of targets, # of outputs)"""
        dEdzout = dEdo * self.output_layer.gradient()
        output = np.tile(self.layers[self.length - 2].output, (self.output_layer.shape[1], 1)) * np.reshape(dEdzout.T, (-1,1))
        update = np.reshape(output, (self.output_layer.shape[1], self.layers[self.length - 2].output.shape[0], self.layers[self.length - 2].output.shape[1]))
        update = np.mean(update, axis=1).transpose() * self.step
        self.output_layer.weights -= np.clip(update, -self.weight_clip, self.weight_clip)
        self.output_layer.weights -= self.step * self.reg_coefficient * self.output_layer.weights
        self.output_layer.bias -= np.clip(np.sum(dEdzout) * self.step, -self.bias_clip, self.bias_clip)
        for i in range(self.length - 2, 0, -1):
            d = np.reshape(dEdzout, (-1,1)) * np.tile(self.layers[i+1].weights, dEdzout.shape[0]).transpose()
            d = np.reshape(d, (dEdzout.shape[0], self.layers[i+1].shape[1], self.layers[i+1].shape[0]))
            dEdzout = np.sum(d, axis=1) * self.layers[i].gradient()
            output =  np.tile(self.layers[i - 1].output, (self.layers[i].shape[1],1)) * np.reshape(dEdzout.T, (-1,1))
            update = np.reshape(output, (self.layers[i].shape[1], self.layers[i - 1].output.shape[0], self.layers[i - 1].output.shape[1]))
            update = np.mean(update, axis=1).transpose() * self.step
            self.layers[i].weights -= np.clip(update, -self.weight_clip, self.weight_clip)
            self.layers[i].weights -= self.step * self.reg_coefficient * self.layers[i].weights
            self.layers[i].bias -= np.clip(np.sum(dEdzout) * self.step, -self.bias_clip, self.bias_clip)
        d = np.reshape(dEdzout, (-1,1)) * np.tile(self.layers[1].weights, dEdzout.shape[0]).transpose()
        d = np.reshape(d, (dEdzout.shape[0], self.layers[1].shape[1], self.layers[1].shape[0]))
        dEdzout = np.sum(d, axis=1) * self.input_layer.gradient()
        output =  np.tile(self.input, (self.input_layer.shape[1],1)) * np.reshape(dEdzout.T, (-1,1))
        update = np.reshape(output, (self.input_layer.shape[1], self.input.shape[0], self.input.shape[1]))
        update = np.mean(update, axis=1).transpose() * self.step
        self.input_layer.weights -= np.clip(update + self.reg_coefficient * np.sum(self.input_layer.weights), -self.weight_clip, self.weight_clip)
        self.input_layer.weights -= self.step * self.reg_coefficient * self.input_layer.weights
        self.input_layer.bias -= np.clip(np.sum(dEdzout) * self.step, -self.bias_clip, self.bias_clip)
        d = np.reshape(dEdzout, (-1,1)) * np.tile(self.input_layer.weights, dEdzout.shape[0]).transpose()
        d = np.reshape(d, (dEdzout.shape[0], self.input_layer.shape[1], self.input_layer.shape[0]))
        dEdzout = np.sum(d, axis=1)
        return dEdzout

    def save(self, name="Network", as_np = True):
        print(f"Saving network as {name}...")
        data = {}
        layers = []
        for layer in self.layers:
            layers.append({"shape":layer.shape, 
                           "type":layer.type, 
                           "weights":layer.weights.tolist(), 
                           "bias":layer.bias,
                           "activation_id":layer.type.value})
        data["layers"] = layers
        data["options"] = {"step":self.step,
                           "weight_clip":self.weight_clip,
                           "bias_clip":self.bias_clip,
                           "reg_coefficient":self.reg_coefficient}
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
        try:
            if(as_np):
                data = np.load(location+".npy", allow_pickle=True).item()
            else:
                with open(location+".json","r") as f:
                    data = json.load(f)
            for i in range(len(data["layers"])):
                layers.append(Layer(shape=data["layers"][i]["shape"], 
                                type = Activation(data["layers"][i]["activation_id"]), 
                                weights=data["layers"][i]["weights"], 
                                bias=data["layers"][i]["bias"]))
            self.__init__(layers = layers,
                          step = data["options"]["step"], 
                          weight_clip = data["options"]["weight_clip"],
                          bias_clip = data["options"]["bias_clip"],
                          reg_coefficient = data["options"]["reg_coefficient"])
            print(f"Load success")
        except Exception as e:
            print(f"Load file from {location} failed {e}")


#nn = NeuralNetwork([Layer((21188, 100), Activation.Relu), 
#                    Layer((100,100), Activation.Relu), 
#                    Layer((100,100), Activation.Relu), 
#                    Layer((100, 369), Activation.Linear)], step=0.0001)
#nn.save("DR_actor_100")
#nn = NeuralNetwork([Layer((21189, 50), Activation.Relu), 
#                    Layer((50,50), Activation.Relu), 
#                    Layer((50,50), Activation.Relu), 
#                    Layer((50, 1), Activation.Linear)], step=0.0001)
#nn.save("DR_critic_500")


#if __name__=="__main__":
#    nn = NeuralNetwork()
#    nn.load("DR_actor")
#    nn.step = 0.00003
#    i = nn.forward(np.ones(21188))
#    nn.backprop_target(np.ones(369) * 10)
#    f = nn.forward(np.ones(21188))
#    s = f - i
#    print(np.linalg.norm(s))
#    nn2 = NeuralNetwork()
#    nn2.load("DR_actor_100")
#    nn2.step = 0.00004
#    i = nn2.forward(np.ones(21188))
#    nn2.backprop_target(np.ones(369) * 10)
#    f = nn2.forward(np.ones(21188))
#    s = f - i
#    print(np.linalg.norm(s))
    