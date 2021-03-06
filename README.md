
## Efficient operations on arrays of extremely large or small numbers.  

**Xrange_array** is a [numpy](https://numpy.org/) `nd.array` subclass which allows to represent floating-point numbers in an extended range:  
- `[1.e-646456992, 1.e+646456992]`.   

Float or complex numbers in simple or double precision with an extra base-2 exponent stored as `int32` are implemented.  
It also provides real and complex implementation for:  
- The main binary operations `(+, -, *, /, <, <=, >, >=)`
- a few selected complex functions `(abs, angle, sqrt, square, conj, log)`
- an `abs2` function: optimal implementation of square of `abs` for complex numbers.

Their use is transparent to the user, as they follow numpy general API.

Basic use:

> `>>> Xa = Xrange_array([["123.456e-1789", "-.3e-7"], ["1.e700", "1.0"]])`  
> `>>> print(Xa**2)`  
> `[[ 1.52413839e-3574  9.00000000e-16]`  
> ` [ 1.00000000e+1400  1.00000000e+00]]`  

Accurate base-10 conversion for string-inputs and printing:

> `>>> Xb = np.array([1., -1.j]) * np.pi * Xrange_array(["1.e+646456991","1.e-646456991" ])`  
> `>>> with np.printoptions(precision=13) as _:`  
> `        print(Xb)`  
> `[ 3.1415926535898e+646456991➕0.0000000000000e+00j`  
> `  0.0000000000000e+00➖3.1415926535898e-646456991j]`  

Performance compared with standard `np.complex128` operations on large (40'000) arrays:  

- `square` overhead ratio: `12.6`
- `add` overhead ratio: `10.6`
- `multiply` overhead ratio: `4.3`
- `abs2` overhead ratio: `2.2`

**Xrange_polynomial** represents a polynomial with extended range coefficients. It implements the basic operations +, -, *.

**Xrange_SA** is a subclass of **Xrange_polynomial** representing series approximations of a function. Compared to **Xrange_polynomial** it keeps track of a truncature error (high order terms ignored during multiplication).

