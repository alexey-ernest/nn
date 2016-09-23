import theano
import numpy as np


# You can get the values being used to configure Theano like so:
print(theano.config.device)
print(theano.config.floatX)


# You can also get/set them at runtime:
old_floatX = theano.config.floatX
theano.config.floatX = 'float32'

# Be careful that you're actually using floatX!
# For example, the following will cause var to be a float64 regardless of floatX due to numpy defaults:
var = theano.shared(np.array([1.3, 2.4]))
print(var.type()) #!!!
# So, whenever you use a numpy array, make sure to set its dtype to theano.config.floatX
var = theano.shared(np.array([1.3, 2.4], dtype=theano.config.floatX))
print(var.type())
# Revert to old value
theano.config.floatX = old_floatX
