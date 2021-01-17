
## Efficient operations on arrays of extremely large or small numbers.  

**Xrange_array** is a [numpy](https://numpy.org/) `nd.array` subclass which allows to represent floating-point numbers in an extended range:  
- `[1.e-646456992, 1.e+646456992]`.   

Float or complex numbers in simple or double precision with extra base-2 exponent stored as and `int32` are implemented.  
It also provides real an complex implementation for:  
- The main binary operations `(+, -, *, /, <, <=, >, >=)`
- a few selected complex functions `(abs, sqrt, square, conj, log)`
- an `abs2` function: optimal implementation of square of `abs` for complex numbers.

Their use is transparent to the user, as they follow numpy general API.

Basic use:

> `>>> Xa = Xrange_array([["123.456e-1789", "-.3e-7"], ["1.e700", "1.0"]])`  
> `>>> print(Xa**2)`  
> `[[ 1.52413839e-3574  9.00000000e-0016]`  
> ` [ 1.00000000e+1400  1.00000000e+0000]]`  

Accurate base-10 conversion for printing:

> `>>> Xb = 1.j * np.pi * Xrange_array(["1.e+646456992","1.e-646456992" ])`
> `>>> with np.printoptions(precision=12, linewidth=100) as _:`
> `        print(Xa)`
> `[ 0.0000000000000e+00➕3.1415926535898e+646456992j  0.0000000000000e+00➕3.1415926535898e-646456992j]`

Performance compared with standard `np.complex128` operations on large (40'000) arrays:  

- `square` overhead ratio: `23.87`
- `add` overhead ratio: `21.9`
- `multiply` overhead ratio: `24.4`
- `abs2` overhead ratio: `7.7`
