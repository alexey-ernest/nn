"""
Simple neural network.
"""

import layer
import mlp

import theano
import numpy as np
import matplotlib.pyplot as plt
import theano.tensor as T


def gradient_updates_momentum(cost, params, learning_rate, momentum):
    '''
    Compute updates for gradient descent with momentum

    :parameters:
        - cost : theano.tensor.var.TensorVariable
            Theano cost function to minimize
        - params : list of theano.tensor.var.TensorVariable
            Parameters to compute gradient against
        - learning_rate : float
            Gradient descent learning rate
        - momentum : float
            Momentum parameter, should be at least 0 (standard gradient descent) and less than 1

    :returns:
        updates : list
            List of updates, one for each parameter
    '''
    # Make sure momentum is a sane value
    assert momentum < 1 and momentum >= 0
    # List of update steps for each parameter
    updates = []
    # Just gradient descent on cost
    for param in params:
        # For each parameter, we'll create a previous_step shared variable.
        # This variable will keep track of the parameter's update step across iterations.
        # We initialize it to 0
        previous_step = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
        # Each parameter is updated by taking a step in the direction of the gradient.
        # However, we also "mix in" the previous step according to the given momentum value.
        # Note that we don't need to derive backpropagation to compute updates - just use T.grad!
        step = momentum*previous_step - learning_rate*T.grad(cost, param)
        # Add an update to store the previous step value
        updates.append((previous_step, step))
        # Add an update to apply the gradient descent step to the parameter itself
        updates.append((param, param + step))
    return updates



# Training data - two randomly-generated Gaussian-distributed clouds of points in 2d space
np.random.seed(0)
# Number of points
N = 1000
# Labels for each cluster
y = np.random.random_integers(0, 1, N)
# Mean of each cluster
means = np.array([[-1, 1], [-1, 1]])
# Covariance (in X and Y direction) of each cluster
covariances = np.random.random_sample((2, 2)) + 1
# Dimensions of each point
X = np.vstack([np.random.randn(N)*covariances[0, y] + means[0, y],
               np.random.randn(N)*covariances[1, y] + means[1, y]]).astype(theano.config.floatX)
# Convert to targets, as floatX
y = y.astype(theano.config.floatX)

# Plot the data
# plt.figure(figsize=(8, 8))
# plt.scatter(X[0, :], X[1, :], c=y, lw=.3, s=3, cmap=plt.cm.cool)
# plt.axis([-6, 6, -6, 6])
# plt.show()






# First, set the size of each layer (and the number of layers)
# Input layer size is training data dimensionality (2)
# Output size is just 1-d: class label - 0 or 1
# Finally, let the hidden layers be twice the size of the input.
# If we wanted more layers, we could just add another layer size to this list.
layer_sizes = [X.shape[0], X.shape[0]*2, 1] # [2,4,1]
# Set initial parameter values
W_init = []
b_init = []
activations = []
for n_input, n_output in zip(layer_sizes[:-1], layer_sizes[1:]):
    # Getting the correct initialization matters a lot for non-toy problems.
    # However, here we can just use the following initialization with success:
    # Normally distribute initial weights
    W_init.append(np.random.randn(n_output, n_input))
    # Set initial biases to 1
    b_init.append(np.ones(n_output))
    # We'll use sigmoid activation for all layers
    # Note that this doesn't make a ton of sense when using squared distance
    # because the sigmoid function is bounded on [0, 1].
    activations.append(T.nnet.sigmoid)
# Create an instance of the MLP class
mlp = mlp.MLP(W_init, b_init, activations)

# Create Theano variables for the MLP input
mlp_input = T.matrix('mlp_input')
# ... and the desired output
mlp_target = T.vector('mlp_target')
# Learning rate and momentum hyperparameter values
# Again, for non-toy problems these values can make a big difference
# as to whether the network (quickly) converges on a good local minimum.
learning_rate = 0.01
momentum = 0.9
# Create a function for computing the cost of the network given an input
cost = mlp.squared_error(mlp_input, mlp_target)
# Create a theano function for training the network
train = theano.function([mlp_input, mlp_target], cost,
                        updates=gradient_updates_momentum(cost, mlp.params, learning_rate, momentum))
# Create a theano function for computing the MLP's output given some input
mlp_output = theano.function([mlp_input], mlp.output(mlp_input))




# Keep track of the number of training iterations performed
iteration = 0
# We'll only train the network with 20 iterations.
# A more common technique is to use a hold-out validation set.
# When the validation error starts to increase, the network is overfitting,
# so we stop training the net.  This is called "early stopping", which we won't do here.
max_iteration = 20
while iteration < max_iteration:
    # Train the network using the entire training set.
    # With large datasets, it's much more common to use stochastic or mini-batch gradient descent
    # where only a subset (or a single point) of the training set is used at each iteration.
    # This can also help the network to avoid local minima.
    current_cost = train(X, y) # it also updates parameters by provided 'updates' param
    # Get the current network output for all points in the training set
    current_output = mlp_output(X)
    # We can compute the accuracy by thresholding the output
    # and computing the proportion of points whose class match the ground truth class.
    accuracy = np.mean((current_output > .5) == y)

    # Plot network output after this iteration
    # plt.figure(figsize=(8, 8))
    # plt.scatter(X[0, :], X[1, :], c=current_output,
    #             lw=.3, s=3, cmap=plt.cm.cool, vmin=0, vmax=1)
    # plt.axis([-6, 6, -6, 6])
    # plt.title('Cost: {:.3f}, Accuracy: {:.3f}'.format(float(current_cost), accuracy))
    # plt.show()
    print('Epoch: {:}, Cost: {:.3f}, Accuracy: {:.3f}'.format(iteration, float(current_cost), accuracy))

    iteration += 1


