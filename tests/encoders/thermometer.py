import numpy as np
from wisardlib.encoders.thermometer import ThermometerEncoder

X = np.array([0, 1, 25, 26, 50, 78, 100])
encoder = ThermometerEncoder(16)
encoder.fit(X)
print(encoder.__dict__)
n = encoder.transform(X)
print(n)
