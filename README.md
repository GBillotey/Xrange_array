
## Efficient vectorized operations on extremely large or small numbers.  

**Xrange_array** is a numpy nd.array subclass which allows to represent numbers in the range [`1.e-646456992`, `1.e+646456992`].    
Float or complex numbers in simple or double precision are implemented.  
It also provides :  
- The main binary operations (`+, -, *, /, <, <=, >, >=`)
- a few selected functions (`abs, sqrt, square, conj, log`)  

Their use is transparent to the user, as they follow numpy general API.

Basic usage:

> `>>> Xa = Xrange_array([["123.456e-1789", "-.3e-7"], ["1.e700", "1.0"]])`  
> `>>> print(Xa**2)`

Yields:

> `[[ 1.52413839e-3574  9.00000000e-0016]`  
>` [ 1.00000000e+1400  1.00000000e+0000]]`
