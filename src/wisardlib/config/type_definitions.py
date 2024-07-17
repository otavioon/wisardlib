import numpy
from typing import Union
import numpy.typing as npt

# The input for WiSARDs is an array of booleans
ByteArray = Union[npt.NDArray[numpy.ubyte], npt.NDArray[numpy.bool]]
Address = ByteArray
Sample = ByteArray
