from timeit import timeit
from numba import jit
import math

# This is the function decorator syntax and is equivalent to `hypot = jit(hypot)`.
# The Numba compiler is just a function you can call whenever you want!
@jit
def hypot(x, y):
    # Implementation from https://en.wikipedia.org/wiki/Hypot
    x = abs(x);
    y = abs(y);
    t = min(x, y);
    x = max(x, y);
    t = t / x;
    return x * math.sqrt(1+t*t)

# print(hypot(3.0, 4.0))    
print(timeit('hypot(3.0, 4.0)', globals=globals(), number=1000000))
# Output should be close to 5.0 
hypot.inspect_types()