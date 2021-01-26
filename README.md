# Jonathan's basic Machine Learning API

## About
- Light weight
- Networks saves as numpy files
- Built from scratch with numpy

## How to use?
### Step 1: Specify Neural Network Dimensions.
Start by defining the dimensions of the layers within your network as an list of ```Layer()``` objects with tuples to specify the layer's input/output dimensions. The first tuple element as input of dimension of that layer and second element as output.

Example network with 3 layers:\
Layer one, 5 input, 10 output\
Layer two, 10 input, 5 output\
Layer three, 5 input, 1 output\
```
layers = [Layer((5, 10)), Layer((10, 5)), Layer((5, 1))]
```


### Step 2: Create The Actual Network Object.
Simply instantiate the NeuralNetwork object with specified network dimensions from step 1.

Example:\
```
nn = NeuralNetwork(layers)
```

**Note: Each layer is by default

### Step 3: Train the Neural netwrok.
To train the neural network, it needs to be fed both the input and target output.

To feed the neural network simply use ```nn.forward(input)``` this function will also return the output of the **whole network**.

**Note: nn is an arbitrary variable set in step 2 to reference the created network**

To feed the target output use ```nn.backprop_target(target_output)``` and the network will be trained based on the ```target_output``` pushing the network's output closer to ```target_output``` with respect to ```input```.

**Note: the ```input``` dimension should match the first layer input dimension specificed in step 1. The same applies to ```target_output``` and last layer's specified output dimensions**

### Step4: Save The Neural Network
To save your network simply use ```nn.save("MyNetwork")``` and a file named "MyNetwork.npy" will appear in the same directory. To load a previously saved network simply use ```nn.load("MyNetwork")```.

**Note: ```nn``` has to have ```NeuralNetwork()``` instaniated for ```nn.load()```**

Example:
```
#Saving network from net
net = NeuralNetwork([Layer((1,5)), Layer((5, 5)), Layer((5, 1))])
net.save("MyNetwork")

#Loading MyNetwork into nn
nn = NeuralNetwork()
nn.load("MyNetwork")
```

