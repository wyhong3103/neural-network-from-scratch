# Neural Network From Scratch

I challenged myself to write an artificial neural network from scratch that supports a fully connected layer with ReLU/Sigmoid/Linear activation functions. It's an unoptimized neural network. It runs slowly, but it works fine.

Here is a [note](https://www.wyhong3103.tech/blog/backpropagation) I made about backpropagation. Feel free to check it out if you'd like.

The most interesting part of this project is figuring out the implementation of backpropagation. It's like solving a CP problem, quite fun. I never thought I'd be implementing topological sort in a project. I guess CP wasn't a waste after all! :D

## Instruction

I'm not sure who will use it, but if you're interested, here are the instructions.

1. Import the neural network, the layers and the loss functions
```python
from layers.fc_layer import FC_Layer
from loss.squared_error import squared_error
from loss.binary_crossentropy import binary_crossentropy
from neural_network import NN
from layers.input_layer import Input_Layer
```
2. Define the neural network.

```python
nn = NN([
    Input_Layer(1),
    FC_Layer(5, 'relu'),
    FC_Layer(5, 'relu'),
    FC_Layer(1, 'relu')
], loss_fn=squared_error, learning_rate=0.0001)
```

3. Train it.

```python
X = [[i * 1.0] for i in range(1, 20)]
y = [i*5*1.0 for i in range(1, 20)]

for _ in range(100):
    nn.fit(X, y)
```

4. Predict.

```python
nn.predict(X)
```

## References

- [Neural Network Playlist - 3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) 
  - I basically learned how to do backpropagation here.
- [Backpropagation and Project Advice - Stanford University School of Engineering](https://www.youtube.com/watch?v=isPiE-DBagM)
  - I was looking for a resource that talks about multiple paths chain rule and this helped. 
- [The spelled-out intro to neural networks and backpropagation: building micrograd - Andrej Karpathy](https://www.youtube.com/watch?v=VMj-3S1tku0)
  - I watched the first 30 minutes of the video and got a nice idea on how to implement the computation graph with operators overloading.