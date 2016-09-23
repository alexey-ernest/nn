# Ensure python 3 forward compatibility
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import theano
# By convention, the tensor submodule is loaded as T
import theano.tensor as T

foo = T.scalar('foo')
bar = foo**2

bar_grad = T.grad(bar, foo) # 2 * foo
print(bar_grad.eval({foo: 10}))


# Recall that y = Ax + b
A = T.matrix('A')
x = T.vector('x')
b = T.vector('b')
y = T.dot(A, x) + b
# We can also compute a Jacobian like so:
y_J = theano.gradient.jacobian(y, x)
linear_mix_J = theano.function([A, x, b], y_J)
# Because it's a linear mix, we expect the output to always be A
print(linear_mix_J(np.array([[9, 8, 7], [4, 5, 6]], dtype=theano.config.floatX), #A
                   np.array([1, 2, 3], dtype=theano.config.floatX), #x
                   np.array([4, 5], dtype=theano.config.floatX))) #b
# We can also compute the Hessian with theano.gradient.hessian (skipping that here)
