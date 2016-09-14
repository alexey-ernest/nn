import time

import network3
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer

# load data
training_data, validation_data, test_data = network3.load_data_shared()

# train
start = time.time()
mini_batch_size = 10

# Shallow network with 1 hidden layer of 100 sigmoid neurons, 60 epochs,
# 10 mini batch size and 0.1 learning rate, no regularization,
# softmax final layer with log-likelihood cost function.
# ~97.78%
# net = Network([
#         FullyConnectedLayer(n_in=784, n_out=100),
#         SoftmaxLayer(n_in=100, n_out=10)
#     ],
#     mini_batch_size)

# net.SGD(training_data, 60, mini_batch_size, 0.1,
#         validation_data, test_data)



# One conv layer with 20 feature maps
# ~98.78% (98.49% without fully-connected layer)
# net = Network([
#             ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
#                           filter_shape=(20, 1, 5, 5),
#                           poolsize=(2, 2)),
#             FullyConnectedLayer(n_in=20*12*12, n_out=100),
#             SoftmaxLayer(n_in=100, n_out=10)
#         ],
#         mini_batch_size)
# net.SGD(training_data, 60, mini_batch_size, 0.1,
#         validation_data, test_data)


# Two conv layers with 20 feature maps and 40 feature maps
# ~98.99% epcho 43
net = Network([
            ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                          filter_shape=(20, 1, 5, 5),
                          poolsize=(2, 2)),
            ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                          filter_shape=(40, 20, 5, 5),
                          poolsize=(2, 2)),
            FullyConnectedLayer(n_in=40*4*4, n_out=100),
            SoftmaxLayer(n_in=100, n_out=10)
        ],
        mini_batch_size)
net.SGD(training_data, 60, mini_batch_size, 0.1,
        validation_data, test_data)

print('Learning time {0}'.format(time.time() - start))
