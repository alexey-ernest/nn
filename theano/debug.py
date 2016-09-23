# Ensure python 3 forward compatibility
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import theano
# By convention, the tensor submodule is loaded as T
import theano.tensor as T


A = T.matrix('A')
B = T.matrix('B')

# And, a symbolic variable which is just A (from above) dotted against B
# At this point, Theano doesn't know the shape of A or B, so there's no way for it to know whether A dot B is valid.
# C = T.dot(A, B)
# Now, let's try to use it
# C.eval({A: np.zeros((3, 4), dtype=theano.config.floatX),
#         B: np.zeros((5, 6), dtype=theano.config.floatX)})


# This tells Theano we're going to use test values, and to warn when there's an error with them.
# The setting 'warn' means "warn me when I haven't supplied a test value"
theano.config.compute_test_value = 'warn'
# Setting the tag.test_value attribute gives the variable its test value
A.tag.test_value = np.random.random((3, 4)).astype(theano.config.floatX)
B.tag.test_value = np.random.random((5, 6)).astype(theano.config.floatX)
# Now, we get an error when we compute C which points us to the correct line!
C = T.dot(A, B)

# We won't be using test values for the rest of the tutorial.
theano.config.compute_test_value = 'off'








