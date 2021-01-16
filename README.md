
## Efficient operations on arrays of extremely large or small numbers.  

**Xrange_array** is a [numpy](https://numpy.org/) `nd.array` subclass which allows to represent floating-point numbers in an extended range:  
- `[1.e-646456992, 1.e+646456992]`.   

Float or complex numbers in simple or double precision with extra base-2 exponent stored as and `int32` are implemented.  
It also provides :  
- The main binary operations `(+, -, *, /, <, <=, >, >=)`
- a few selected complex functions `(abs, sqrt, square, conj, log)`
- an `abs2` function: optimal implementation of square of `abs`.

Their use is transparent to the user, as they follow numpy general API.

Basic use:

> `>>> Xa = Xrange_array([["123.456e-1789", "-.3e-7"], ["1.e700", "1.0"]])`  
> `>>> print(Xa**2)`  
> `[[ 1.52413839e-3574  9.00000000e-0016]`  
> ` [ 1.00000000e+1400  1.00000000e+0000]]`  


Performance compared with standard `np.complex128` operations on large (40'000) arrays:  

- `square` overhead ratio: `23.87`
- `add` overhead ratio: `21.9`
- `multiply` overhead ratio: `24.4`
- `abs2` overhead ratio: `7.7`
