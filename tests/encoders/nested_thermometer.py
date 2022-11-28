import numpy as np
from wisardlib.encoders.thermometer import NestedThermometerEncoder

X = np.array([0, 1, 25, 26, 50, 78, 100])
encoder = NestedThermometerEncoder(4, 2)
encoder.fit(X)
print(encoder.__dict__)
n = encoder.transform(X)
print(encoder.thermometer.__dict__)
print(n)
