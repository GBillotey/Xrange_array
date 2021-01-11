# -*- coding: utf-8 -*-
import numpy as np
import numbers
import re


class Xrange_array(np.ndarray):
    """
Arrays class for "extended range" floats or complex numbers.
    This class allows to represent floating points numbers in simple or double
    precision in the range [1e-646456992, 1e+646456992].

Parameters:
    mantissa :  Can be either
        - a nd.array of dtype: one of the supported dtype float32, float64,
            complex64, complex128,
        - a string array, each item reprenting a float in standard or
            e-notation e.g. ["123.456e789", "-.3e-7", "1.e-1000", "1.0"]
            Note that list inputs are accepted in both cases, and passed
            through np.asarray() method.
    exp_re :  int32 array of shape sh, if None will default to 0
        ignored if mantissa is provided as string array.
    exp_im :  int32 array of shape sh, if None will default to 0
        ignored if mantissa is provided as string array.
    str_input_dtype : np.int32 of np.int64, only used if mantissa
        provided as a string, to allow specification of dataype.
        (if None, will default to float64)

Return:
    Xrange_array of same shape as parameter 'mantissa' representing
        (real case): mantissa * 2**exp_re  
        (complex case): (mantissa.real * 2**exp_re 
                         + j * mantissa.imag * 2**exp_im)

Usage:
    >>> Xa = Xrange_array([["123.456e-1789", "-.3e-7"], ["1.e700", "1.0"]])
    >>> print(Xa**2)
    [[ 1.52413839e-3574  9.00000000e-0016]
     [ 1.00000000e+1400  1.00000000e+0000]]
    
    >>> b = np.array([1., 1., np.pi, np.pi], dtype=np.float32)
    >>> Xb = Xrange_array(b)
    >>> for exp10 in range(1001):
            Xb = Xb * [-10., 0.1, 10., -0.1]
    >>> Xb
    <class 'arrays.Xrange_array'>
    shape: (4,)
    internal dtype: [('mantissa', '<f8'), ('exp_re', '<i4')]
    [-1.00000000e+1001  1.00000000e-1001  3.14159274e+1001 -3.14159274e-1001]
    >>> print(Xb)
    [-1.00000000e+1001  1.00000000e-1001  3.14159274e+1001 -3.14159274e-1001]

Implementation details:
    Each scalar in the array is stored as a couple: 1 real (resp. complex) and
    1 int32 integer (resp, 1 int32 couple) for an extra base-2 exponent. The
    overall array is stored as a structured array of type :
        - (float32, int32)
        - (float64, int32)
        - (complex64, (int32, int32))
        - (complex128, (int32, int32))
    Hence, the mantissa can be one of 4 supported types :
        float32, float64, complex64, complex128 

    Each class instance exposes the following properties ("views" of the base
    data array):
        real    view of real part, as a real Xrange_array
        imag    view of imaginary part, as a real Xrange_array
        is_complex  Boolean scalar

    The binary operations implemented are :
        +, -, *, /, <, <=, >, >=

    The unary operations implemented are :
        as unfunc : abs, sqrt, square, conj, log
        as instance method : abs2 (square of abs)

    Xrange_array will silently over/underflow, given the implementation of the
    exponent as np.int32 array. If needed, checking for overflow shall be
    implemented in user code.
        >>> np.int32(2**31)
        -2147483648

Reference:
    https://numpy.org/devdocs/user/basics.subclassing.html
    """
    _FLOAT_DTYPES = [np.float32, np.float64]
    _COMPLEX_DTYPES = [np.complex64, np.complex128]

    # types that can be 'viewed' as Xrange_array:
    _HANDLED_TYPES = (np.ndarray, numbers.Number, list)

    def __new__(cls, mantissa, exp_re=None, exp_im=None, str_input_dtype=None):
        """
        Constructor
        """
        mantissa = np.asarray(mantissa)
        if mantissa.dtype.type == np.str_:
            mantissa, exp_re = np.vectorize(cls._convert_from_string)(mantissa)
            if str_input_dtype is not None:
                mantissa = np.asarray(mantissa, dtype=str_input_dtype)

        data = cls._extended_data_array(mantissa, exp_re, exp_im)
        return super().__new__(cls, data.shape, dtype=data.dtype, buffer=data)

    @staticmethod
    def _convert_from_string(input_str):
        """
        Return mantissa and base 2 exponent from float input string
        
        Parameters
        input_str: string (see exp_pattern for accepted patterns)
        
        Return
        m  : mantissa
        exp_re : base 2 exponent
        """
        exp_pattern = ("^([-+]?[0-9]+\.[0-9]*|[-+]?[0-9]*\.[0-9]+)"
                       "([eE]?)([-+]?[0-9]*)$")
        err_msg = ("Unsupported Xrange_array string item: <{}>\n" +
            "(Examples of supported input items: " +
            "<123.456e789>, <-.123e-127>, <+1e77>, <1.0>, ...)")

        match = re.match(exp_pattern, input_str)
        if match:
            m = float(match.group(1))
            exp_re = 0
            if match.group(2) in ["e", "E"]:
                try:
                    exp_re = int(match.group(3))
                except ValueError:
                    raise ValueError(err_msg.format(input_str))
                rr = 3.321928094887362 # (34) mpmath.log(10) / mpmath.log(2)
                exp_re, mod  = np.divmod(exp_re * rr, 1.)
                m = m * 2.**mod # 0.5 <= m10 <  10.0
            return m, exp_re
        else:
            raise ValueError(err_msg.format(input_str))

    @staticmethod
    def _extended_data_array(mantissa, exp_re, exp_im):
        """
        Builds the structured internal array.
        """
        mantissa_dtype = mantissa.dtype
        if mantissa_dtype not in (Xrange_array._COMPLEX_DTYPES +
                                  Xrange_array._FLOAT_DTYPES):
            raise ValueError("Unsupported type{}".format(mantissa_dtype))
        # Builds the exponent array
        is_complex = mantissa_dtype in Xrange_array._COMPLEX_DTYPES
        sh = mantissa.shape

        if is_complex:
            if exp_re is None:
                exp_re = np.zeros(sh, dtype=np.int32)
            if exp_im is None:
                exp_im = np.zeros(sh, dtype=np.int32)
            extended_dtype = np.dtype([('mantissa', mantissa_dtype),
                                       ('exp_re', np.int32),
                                       ('exp_im', np.int32)], align=False)
        else:
            if exp_re is None:
                exp_re = np.zeros(sh, dtype=np.int32)
            extended_dtype = np.dtype([('mantissa', mantissa_dtype),
                                       ('exp_re', np.int32)], align=False)

        data = np.empty(sh, dtype=extended_dtype)
        data['mantissa'] = mantissa
        data['exp_re'] = exp_re
        if is_complex:
            data['exp_im'] = exp_im
        return data

    @property
    def is_complex(self):
        """ boolean scalar, True if Xrange_array is complex"""
        _dtype = self.dtype
        if len(_dtype) > 1:
            _dtype = _dtype[0]
        return _dtype in Xrange_array._COMPLEX_DTYPES

    @property
    def real(self):
        """
        Returns a view to the real part of self, as an Xrange_array.
        """
        if self.is_complex:
            if self.dtype.names is None:
                return np.asarray(self).real.view(Xrange_array)
            assert all([name in np.asarray(self).dtype.names for name in 
                        ['exp_re', 'exp_im']])
            real_bytes = 4
            if self._mantissa.real.dtype == np.float64:
                real_bytes = 8
            data_dtype = np.dtype({'names': ['mantissa', 'exp_re'],
                                   'formats': ["f" + str(real_bytes), "i4"],
                                   'offsets': [0, real_bytes*2],
                                   'itemsize': real_bytes * 2 + 8})
            return np.asarray(self).view(dtype=data_dtype).view(
                Xrange_array)
        else:
            return self

    @property
    def imag(self):
        """
        Returns a view to the imaginary part of self, as an Xrange_array.
        """
        if self.is_complex:
            if self.dtype.names is None:
                return np.asarray(self).imag.view(Xrange_array)
            assert all([name in np.asarray(self).dtype.names for name in 
                        ['exp_re', 'exp_im']])
            real_bytes = 4
            if self._mantissa.real.dtype == np.float64:
                real_bytes = 8
            data_dtype = np.dtype({'names': ['mantissa', 'exp_re'],
                                   'formats': ["f" + str(real_bytes), "i4"],
                                   'offsets': [real_bytes, real_bytes*2 + 4],
                                   'itemsize': real_bytes * 2 + 8})
            return np.asarray(self).view(dtype=data_dtype).view(
                Xrange_array)
        else:
            return 0. * self

    def to_standard(self):
        """ Returns the Xrange_array downcasted to standard np.ndarray ;
        obviously, may overflow. """
        if self.is_complex:
            return (self._mantissa.real * (2.**self._exp_re) +
                   1j * self._mantissa.imag * (2.**self._exp_im))
        else:
            return (self._mantissa * (2.**self._exp_re))

    @property
    def _mantissa(self):
        """ Returns the mantissa of Xrange_array"""
        try:
            return np.asarray(self["mantissa"])
        except IndexError: # Assume we are view casting a np.ndarray
            return np.asarray(self)

    @property
    def _exp_re(self):
        """ Returns the real part of Xrange_array exp array"""
        try:
            return np.asarray(self["exp_re"])
        except IndexError: # We are view casting a np.ndarray
            return np.zeros([], dtype=np.int32)

    @property
    def _exp_im(self):
        """ Returns the imaginary part of Xrange_array exp array"""
        try:
            return np.asarray(self["exp_im"])
        except (IndexError, ValueError): # We are view casting a np.array
            return np.zeros([], dtype=np.int32)

    @property
    def _exp(self):
        """ Returns the Xrange_array exp array: int32 or (int32, int32) """
        if self.is_complex:
            return (self._exp_re, self._exp_im)
        else:
            return self._exp_re

    @staticmethod
    def _normalize_inplace(f, exp):
        """
        Parameters
        f is a float32 or float64 array
        exp is a int32 array

        Return None

        Modifies in place (f, exp) to (nf, nexp) so that
            f * 2**exp == nf * 2**nexp
            .5 <= abs(nf) < 1.
        """
        f[:], exp2 = np.frexp(f)
        exp += exp2

    @staticmethod
    def _normalize(f, exp):
        """
        Parameters
        f :   float32 or float64 np.array
        exp : int32 np.array
        
        Return
        nf : float32 or float64 np.array
        nexp : int32 np.array
            f * 2**exp == nf * 2**nexp
            .5 <= abs(nf) < 1.
        """
        ff, exp2 = np.frexp(f)
        return ff, (exp + exp2)


    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        - *ufunc* is the ufunc object that was called.
        - *method* is a string indicating how the Ufunc was called, either
          ``"__call__"`` to indicate it was called directly, or one of its
          :ref:`methods<ufuncs.methods>`: ``"reduce"``, ``"accumulate"``,
          ``"reduceat"``, ``"outer"``, or ``"at"``.
        - *inputs* is a tuple of the input arguments to the ``ufunc``
        - *kwargs* contains any optional or keyword arguments passed to the
          function. This includes any ``out`` arguments, which are always
          contained in a tuple.
          
        see also:
    https://github.com/numpy/numpy/blob/v1.19.0/numpy/lib/mixins.py#L59-L176
        """
        if "out" in kwargs.keys():
            raise NotImplementedError("ufunc with parameter 'out' not"
                                      "implemented for Xrange_array")
        casted_inputs = ()
        for x in inputs:
            # Only support operations with instances of _HANDLED_TYPES.
            if isinstance(x, Xrange_array):
                casted_inputs += (x,)
            elif isinstance(x, np.ndarray):
                casted_inputs += (x.view(Xrange_array),)
            elif isinstance(x, numbers.Number):
                casted_inputs += (Xrange_array(x),)
            elif isinstance(x, list):
                casted_inputs += (Xrange_array(np.asarray(x)),)
            else:
                raise NotImplementedError("Xrange_array view not handled for "
                    "this type {}, supported types : {}.".format(
                            type(x), Xrange_array._HANDLED_TYPES))

        if method == "__call__":
            if ufunc is np.add:
                return self._add(*casted_inputs)
            elif ufunc is np.negative:
                return self._negative(*casted_inputs)
            elif ufunc is np.subtract:
                return self._subtract(*casted_inputs)
            elif ufunc is np.multiply:
                return self._mul(*casted_inputs)
            elif ufunc is np.true_divide:
                return self._div(*casted_inputs)
            elif ufunc in [np.greater, np.greater_equal, np.less,
                           np.less_equal]:
                return self._compare(*casted_inputs, ufunc=ufunc)
            elif ufunc is np.equal:
                # Unfortunately at parent class implementation level, a call to
                # np.equal with a structured nd.array will result in n_field
                # calls to np.equal, one for each field. 
                # (as of np.__version__ == '1.19.3').
                # Here we need both matissa and exponenent together to decide
                # equality: unicity of the reprensentation is not guaranteed.
                # The below work-around will perform element-wise comparison of
                # of 2 Xrange_arrays, but will not handle mixed cases
                # one Xrange_array with one np.ndarray (an error is 
                # raised)
                cmp0, cmp1 = inputs
                if cmp0.dtype == np.int32: # Skip this one, done with mantissa
                    assert cmp1.dtype == np.int32
                    return np.ones(cmp0.shape, dtype=np.bool)
                try:
                    return self._compare(cmp0.base.view(Xrange_array),
                                         cmp1.base.view(Xrange_array),
                                         ufunc=ufunc)
                except AttributeError:
                    raise TypeError("Can only test equality of 2 "
                                    "Xrange_arrays")
            elif ufunc is np.absolute:
                return self._abs(*casted_inputs)
            elif ufunc is np.sqrt:
                return self._sqrt(*casted_inputs)
            elif ufunc is np.square:
                return self._square(*casted_inputs)
            elif ufunc is np.conj:
                return self._conj(*casted_inputs)
            elif ufunc is np.log:
                return self._log(*casted_inputs)
            else:
                raise NotImplementedError(ufunc)
        else:
            raise NotImplementedError(method)

    def abs2(self):
        """
        Return the square of np.abs(self) (for optimisation purpose).
        """
        if self.is_complex:
            return Xrange_array(*Xrange_array._normalize(
                *Xrange_array._coexp_ufunc(
                    self._mantissa.real**2, 2 * self._exp_re,
                    self._mantissa.imag**2, 2 * self._exp_im, np.add)))
        else:
            return Xrange_array(*Xrange_array._normalize(
                    self._mantissa**2, 2 * self._exp_re))

    @staticmethod
    def _conj(*inputs):
        """ x -> np.conj(x) """
        c0, = inputs
        if c0.is_complex:
            return Xrange_array(np.conj(c0._mantissa), c0._exp_re,
                                       c0._exp_im)
        else:
            return np.copy(c0).view(Xrange_array)

    @staticmethod
    def _square(*inputs):
        """ x -> x**2  """
        sq0, = inputs
        if sq0.is_complex:
            # real part
            m_re, exp_re = Xrange_array._coexp_ufunc(
                    np.square(sq0._mantissa.real), 2 * sq0._exp_re,
                    -np.square(sq0._mantissa.imag), 2 * sq0._exp_im, np.add)
            Xrange_array._normalize_inplace(m_re, exp_re)
            # imaginary part
            m_im, exp_im = Xrange_array._normalize(
                    2 * sq0._mantissa.real * sq0._mantissa.imag,
                    sq0._exp_re + sq0._exp_im)
            dtype = np.complex128 if (m_re.dtype == np.float64
                                      ) else np.complex64
            m = np.empty(m_re.shape, dtype=dtype)
            m.real = m_re
            m.imag = m_im
            return Xrange_array(m, exp_re, exp_im)
        else: # real case
            return Xrange_array(*Xrange_array._normalize(
                    sq0._mantissa**2, 2 * sq0._exp_re))

    @staticmethod
    def _log(*inputs):
        """ x -> np.log(x)  """
        log0, = inputs
        ln2 = 0.6931471805599453
        if log0.is_complex:
            m_re, exp_re = Xrange_array._normalize(
                    log0._mantissa.real, log0._exp_re)
            m_im, exp_im = Xrange_array._normalize(
                    log0._mantissa.imag, log0._exp_im)
            m_re *= 2.
            exp_re -= 1
            m_re, m_im, exp = Xrange_array._coexp_ufunc(
                    m_re, exp_re, m_im, exp_im, None)
            m = m_re + 1.j * m_im
            # Avoid loss of significant digits if e * ln2 close to log(m)
            # ie m close to 2.0
            e_is_m1 = (exp == -1)
            m[e_is_m1] *= 0.5
            exp[e_is_m1] += 1
            return Xrange_array(np.log(m) + m_re.dtype.type(exp * ln2))
        else:
            m, e = Xrange_array._normalize(log0._mantissa, log0._exp_re)
            m *= 2.
            e -= 1
            # Avoid loss of significant digits if e * ln2 close to log(m)
            # ie m close to 2.0
            e_is_m1 = (e == -1)
            m [e_is_m1] *= 0.5
            e [e_is_m1] += 1
            return Xrange_array(np.log(m) + m.dtype.type(e * ln2))

    @staticmethod
    def _sqrt(*inputs):
        """ x -> np.sqrt(x)  """
        sqrt0, = inputs
        
        if sqrt0.is_complex:
            m_re, m_im, exp = Xrange_array._coexp_ufunc(
                    sqrt0._mantissa.real, sqrt0._exp_re,
                    sqrt0._mantissa.imag, sqrt0._exp_im, None)
            m = m_re + 1.j * m_im
            even_exp = ((exp % 2) == 0).astype(np.bool)
            exp = np.where(even_exp, exp // 2, (exp - 1) // 2)
            return Xrange_array(
                np.sqrt(np.where(even_exp, m, m * 2.)), exp, exp)
        else:
            even_exp = ((sqrt0._exp_re % 2) == 0).astype(np.bool)
            return Xrange_array(
                np.sqrt(np.where(even_exp, sqrt0._mantissa,
                                 sqrt0._mantissa * 2.)
                ), np.where(even_exp, sqrt0._exp_re // 2,
                            (sqrt0._exp_re - 1) // 2))

    @staticmethod
    def _abs(*inputs):
        """ x -> np.abs(x) """
        abs0, = inputs
        if abs0.is_complex:
            return np.sqrt((abs0.real * abs0.real +
                            abs0.imag * abs0.imag))
        else:
            return Xrange_array(np.abs(abs0._mantissa), abs0._exp_re)
        raise NotImplementedError

    @staticmethod
    def _compare(*inputs, ufunc):
        """ compare x and y """
        cmp0, cmp1 = inputs
        is_complex = (cmp0.is_complex or cmp1.is_complex)
        if is_complex:
            raise NotImplementedError("Won't handle complex comparison op.")
        else:
            return Xrange_array._coexp_ufunc(cmp0._mantissa, cmp0._exp_re,
                                             cmp1._mantissa, cmp1._exp_re,
                                             ufunc)[0]

    @staticmethod
    def _div(*inputs):
        """ (x, y) -> x / y """
        div0, div1 = inputs
        is_complex = (div0.is_complex or div1.is_complex)
        return Xrange_array._aux_mul(
                div0._mantissa, div0._exp_re, div0._exp_im,
                1. / div1._mantissa, -div1._exp_re, -div1._exp_im, is_complex)

    @staticmethod
    def _mul(*inputs):
        """ (x, y) -> x * y """
        mul0, mul1 = inputs
        is_complex = (mul0.is_complex or mul1.is_complex)
        return Xrange_array._aux_mul(
                mul0._mantissa, mul0._exp_re, mul0._exp_im,
                mul1._mantissa, mul1._exp_re, mul1._exp_im, is_complex)

    @staticmethod
    def _aux_mul(mantissa0, expb0_re, expb0_im,
                 mantissa1, expb1_re, expb1_im, is_complex):
        """ internal auxilliary function for / and * operators """

        if is_complex:
            # real part
            m_re, exp_re = Xrange_array._coexp_ufunc(
                    mantissa0.real * mantissa1.real, expb0_re + expb1_re,
                    -mantissa0.imag * mantissa1.imag, expb0_im + expb1_im,
                    np.add)
            # imaginary part
            m_im, exp_im = Xrange_array._coexp_ufunc(
                    mantissa0.real * mantissa1.imag, expb0_re + expb1_im,
                    mantissa0.imag * mantissa1.real, expb0_im + expb1_re,
                    np.add)
            Xrange_array._normalize_inplace(m_re, exp_re)
            Xrange_array._normalize_inplace(m_im, exp_im)
            dtype = np.complex128 if (m_re.dtype == np.float64
                                      ) else np.complex64
            m = np.empty(m_re.shape, dtype=dtype)
            m.real = m_re
            m.imag = m_im

            return Xrange_array(m, exp_re, exp_im)

        else: # real case
            return Xrange_array(*Xrange_array._normalize(
                    mantissa0 * mantissa1, expb0_re + expb1_re))

    @staticmethod
    def _subtract(*inputs):
        """ (x, y) -> x - y """
        sub0, sub1 = inputs
        is_complex = (sub0.is_complex or sub1.is_complex)
        return Xrange_array._aux_add(
                sub0._mantissa, sub0._exp_re, sub0._exp_im,
                -sub1._mantissa, sub1._exp_re, sub1._exp_im,
                is_complex)

    @staticmethod
    def _negative(*inputs):
        """ x -> -x """
        neg, = inputs
        return Xrange_array(-neg._mantissa, neg._exp_re, neg._exp_im)

    @staticmethod
    def _add(*inputs):
        """ (x, y) -> x + y """
        add0, add1 = inputs
        is_complex = (add0.is_complex or add1.is_complex)
        return Xrange_array._aux_add(
                add0._mantissa, add0._exp_re, add0._exp_im,
                add1._mantissa, add1._exp_re, add1._exp_im,
                is_complex)

    @staticmethod
    def _aux_add(mantissa0, exp0_re, exp0_im, mantissa1, exp1_re, exp1_im,
                 is_complex):
        """ internal auxilliary function for + and - operators """
        if is_complex:
            m_re, exp_re = Xrange_array._coexp_ufunc(
                mantissa0.real, exp0_re, mantissa1.real, exp1_re, np.add)
            m_im, exp_im = Xrange_array._coexp_ufunc(
                mantissa0.imag, exp0_im, mantissa1.imag, exp1_im, np.add)
            
            dtype = np.complex128 if (m_re.dtype == np.float64
                                      ) else np.complex64
            m = np.empty(m_re.shape, dtype=dtype)
            m.real = m_re
            m.imag = m_im
            return Xrange_array(m, exp_re, exp_im)
        else:
            return Xrange_array(*Xrange_array._coexp_ufunc(
                    mantissa0, exp0_re, mantissa1, exp1_re, np.add), None)

    @staticmethod
    def _coexp_ufunc(m0, exp0, m1, exp1, ufunc):
        """ 
        If ufunc is None :
        m0, exp0, m1, exp1, -> co_m0, co_m1, co_exp so that :
        (*)  m0 * 2**exp0 == co_m0 * 2**co_exp
        (*)  m1 * 2**exp1 == co_m1 * 2**co_exp
        (*)  co_exp is the "leading exponent" exp = np.maximum(exp0, exp1)
            except if one of m0, m1 is null.
        If ufunc is provided :
            m0, exp0, m1, exp1, -> ufunc(co_m0, co_m1), co_exp
        """
        co_m0, co_m1 = np.copy(np.broadcast_arrays(m0, m1))
        exp0, exp1 = np.broadcast_arrays(exp0, exp1)
        m0_null = (m0 == 0.)
        m1_null = (m1 == 0.)
        d_exp = exp0 - exp1

        bool0 = ((exp1 > exp0) & ~m1_null)
        co_m0[bool0] = Xrange_array._exp2_shift(co_m0[bool0], d_exp[bool0])
        bool1 = ((exp0 > exp1) & ~m0_null)
        co_m1[bool1] = Xrange_array._exp2_shift(co_m1[bool1], -d_exp[bool1])

        exp = np.maximum(exp0, exp1)
        exp[m0_null] = exp1[m0_null]
        exp[m1_null] = exp0[m1_null]

        if ufunc is not None: 
            return (ufunc(co_m0, co_m1), exp)
        else:
            return (co_m0, co_m1, exp)

    @staticmethod
    def _exp2_shift(m, shift):
        """
        Parameters
            m : float32 or float64 array, mantissa
            exp : int32 array, negative integers array

        Return
            res array of same type as m, shifted by 2**shift :
                res = m * 2**shift

        References:
        https://en.wikipedia.org/wiki/Single-precision_floating-point_format
        s(1)e(8)m(23)
        (bits >> 23) & 0xff  : exponent with bias 127 (0x7f)
        (bits & 0x7fffff) : mantissa, implicit first bit of value 1

        https://en.wikipedia.org/wiki/Double-precision_floating-point_format
        s(1)e(11)m(52)
        (bits >> 52) & 0x7ff : exponent with bias 1023 (0x3ff)
        (bits & 0xfffffffffffff) : mantissa, implicit first bit of value 1
        """
        dtype = m.dtype
        if dtype == np.float32:
            bits = np.abs(m).view(np.int32)
            exp = np.clip(((bits >> 23) & 0xff) + shift, 0, None)
            return np.copysign(((exp << 23)  + (bits & 0x7fffff)).view(
                np.float32), m)
    
        elif dtype == np.float64:
            bits = np.abs(m).view(np.int64)
            exp = np.clip(((bits >> 52) & 0x7ff) + shift, 0, None)
            return np.copysign(((exp << 52)  + (bits & 0xfffffffffffff)).view(
                np.float64) , m)

        else:
            raise ValueError("Unsupported dtype {}".fomat(dtype))


    def __repr__(self):
        """ Detailed string representation of self """
        s = (str(type(self)) + "\nshape: " +str(self.shape) +
             "\ninternal dtype: " + str(self.dtype) + "\n" +
             self.__str__())
        return s

    def __str__(self):
        """
        String representation of self. Takes into account the value of
        np.get_printoptions(precision)

        Usage :
        with np.printoptions(precision=2) as opts:
            print(extended_range_array)
        """
        if self.is_complex:
            s_re = Xrange_array._float_to_char(
                *Xrange_array._normalize(self._mantissa.real,
                                                self._exp_re))
            s_im = Xrange_array._float_to_char(
                *Xrange_array._normalize(self._mantissa.imag,
                                                self._exp_im), im=True)
            s = np.core.defchararray.add(s_re, s_im)
            s = np.core.defchararray.add(s, "j")
        else:
            s = Xrange_array._float_to_char(
                *Xrange_array._normalize(self._mantissa, self._exp_re))

        return np.array2string(s, 
            formatter={'numpystr':lambda x: "{}".format(x)})

    @staticmethod
    def _float_to_char(m2, exp2, im=False, im_p_char = '\u2795',
                       im_m_char = '\u2796'):
        """
        Parameters:
            m2 base 2 real mantissa
            exp2 : base 2 exponent

        Return
            str_arr  string array of representations in base 10.

        Note: precisions according to:
            np.get_printoptions(precision)
        """
        print_decimals = np.get_printoptions()["precision"]
        r = 0.3010299956639812 # mpmath.log(2) / mpmath.log(10)

        m2, exp2 = Xrange_array._normalize(m2, exp2) # 0.5 <= m2 < 1.0
        exp10, mod = np.divmod(exp2 * r, 1.) 
        m10 = m2 * 10.**mod # 0.5 <= m10 <  10.0

        if np.isscalar(m10): # scalar do not support item assignment
            if (np.abs(m10) < 1.0):
                m10 *= 10.
                exp10 -= 1
            exp10 = np.asarray(exp10, np.int32)
            _m10 = np.around(m10, decimals=print_decimals)
            if (np.abs(_m10) >= 10.0):
                m10 *= 0.1
                exp10 += 1
            m10 = np.around(m10, decimals=print_decimals)
            # Special case of 0.
            if (m2 == 0.):
                exp10 = 0
        else:
            m10_up = (np.abs(m10) < 1.0)
            m10[m10_up] *= 10.
            exp10[m10_up] -= 1
            exp10 = np.asarray(exp10, np.int32)
            _m10 = np.around(m10, decimals=print_decimals)
            m10_down= (np.abs(_m10) >= 10.0)
            m10[m10_down] *= 0.1
            exp10[m10_down] += 1
            m10 = np.around(m10, decimals=print_decimals)
            # Special case of 0.
            is_null = (m2 == 0.)
            exp10[is_null] = 0

        if im :
            p_char = im_p_char # bold +
            m_char = im_m_char # bold -
        else:
            p_char = " "
            m_char = "-"
        concat = np.core.defchararray.add
        exp_digits = int(np.log10(max([np.nanmax(np.abs(exp10)), 10.]))) + 1
        str_arr = np.where(m10 < 0., m_char, p_char)
        str_arr = concat(str_arr,
                         np.char.ljust(np.abs(m10).astype("|U" + 
                                       str(print_decimals + 2)),
                                       print_decimals + 2, "0"))
        str_arr = concat(str_arr, "e")
        str_arr = concat(str_arr, np.where(exp10 < 0, "-", "+"))
        str_arr = concat(str_arr,
            np.char.rjust(np.abs(exp10).astype("|U10"), exp_digits, "0"))
        return str_arr