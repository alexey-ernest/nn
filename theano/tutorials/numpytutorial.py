import numpy

print numpy.asarray([[1.,2],[3,4],[5,6]])
print numpy.asarray([[1.,2],[3,4],[5,6]]).shape

# broadcasting
a = numpy.asarray([1, 2, 3])
b = 2.0
print a * b # b broadcasted to [[2,0,0],[0,2,0],[0,0,2]] to be compatible with a
