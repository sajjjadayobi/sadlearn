# SadLearn: machine learning Algorithms with Numpy

- A little example of using MLPs algorithm
```python
import FullyConnectedNet
# MLP with 3 hidden layers
clf = FullyConnectedNet(hidden_dims=[100, 100, 100], num_class=10, num_epoch=10)
clf.train((x_train, y_train), (x_test, y_test))
```

## Structure:

    ├── Algorithm   
    |   |
    │   ├── Classification           
    |   |    ├── K Nearest Neighbors
    |   |    ├── Linear classifier
    |   |    ├── LogReg binary
    |   |    |── Naive Bayesian
    |   |
    │   ├── Clustering           
    |   |    ├── K Means
    |   |    ├── DBSCAN
    |   |
    │   ├── Nureal Network    
    |   |   ├── Fully Connected Network
    |   |   |   |
    |   |   |   ├── Optimzation
    |   |   |   ├── Layer
    |   |   ├── convolution Layer
    |   |   
    |   ├── Regression
    |   |   ├── Regression SGD
    |   |   ├── Regression Normal Equation
    |   |   ├── K Neighbors Regression
    |   |
    |   ├── Preprocessing
    |   |   ├── Normalize
    |   |...
    |
    |...
