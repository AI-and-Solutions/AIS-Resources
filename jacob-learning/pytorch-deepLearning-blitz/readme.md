## PyTorch 60-min blitz Deep Learning w/ MNIST

This is a pytorch tutorial that will help jumpstart my ability to use this ML library. I am listing things I learn in this document.

## tensors

Tensors are similar to arrays / lists. Tensors are used in pytorch to encode data for the model's I/O operations. Tensors are also used to describe the model's parameters. 

The reason why lots of work is done inside this data structure is because the pyTorch tensor is highly optimized for machine learning. Here are some features:

- Tensors can be trained on GPU's, an exponentially faster way to train your model

- Tensors share the same underlying memory as numpy arrays, which means that the data structure doesn't need to copy data from numpy arrays to tensors

- Tensors were built to optimize the differentation calculations done by the autograd feature; these calculations are done during backpropagation, the most resource-intensive process in machine learning. 

