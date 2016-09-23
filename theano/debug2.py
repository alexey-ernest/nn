# Ensure python 3 forward compatibility
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import theano
# By convention, the tensor submodule is loaded as T
import theano.tensor as T



# A simple division function
num = T.scalar('num')
den = T.scalar('den')
divide = theano.function([num, den], num/den, mode='DebugMode')
print(divide(10, 2))
# This will cause a NaN
print(divide(0, 0))
