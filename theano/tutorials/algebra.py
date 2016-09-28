from theano import *
import theano.tensor as T
from theano import function
from theano import pp

x = T.dscalar('x')
print type(x) # TensorVariable
print x.type  # dscalar theano type

y = T.dscalar('y')

z = x + y
print pp(z)

f = function([x, y], z)

print f(2.,3.)

print numpy.allclose(f(16.3, 12.1), 28.4)
