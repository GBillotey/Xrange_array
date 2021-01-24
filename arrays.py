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

    The binary operations implemented are:
        +, -, *, /, <, <=, >, >=
    also the matching 'assignment' operators:
        +=, -=, *=, /=

    The unary operations implemented are :
        as unfunc : abs, sqrt, square, conj, log
        as instance method : abs2 (square of abs)

    Xrange_array will silently over/underflow, due to the implementation of its
    exponent as a np.int32 array. If needed, checks for overflow shall be
    done in user code.
        >>> np.int32(2**31)
        -2147483648

Reference:
    https://numpy.org/devdocs/user/basics.subclassing.html
    """
    _FLOAT_DTYPES = [np.float32, np.float64]
    _COMPLEX_DTYPES = [np.complex64, np.complex128]
    _DTYPES = _FLOAT_DTYPES + _COMPLEX_DTYPES

    # types that can be 'viewed' as Xrange_array:
    _HANDLED_TYPES = (np.ndarray, numbers.Number, list)
    #__array_priority__ = 20
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
        exp_pattern = ("^([-+]?[0-9]+\.[0-9]*|[-+]?[0-9]*\.[0-9]+|[-+]?[0-9]+)"
                       "([eE]?)([-+]?[0-9]*)$")
        err_msg = ("Unsupported Xrange_array string item: <{}>\n" +
            "(Examples of supported input items: " +
            "<123.456e789>, <-.123e-127>, <+1e77>, <1.0>, ...)")

        match = re.match(exp_pattern, input_str)
        if match:
            m = float(match.group(1))
            exp_10 = 0
            if match.group(2) in ["e", "E"]:
                try:
                    exp_10 = int(match.group(3))
                    if abs(exp_10) > 646456992:
                        raise ValueError("Overflow int string input, cannot "
                            "represent exponent with int32, maxint 2**31-1")
                except ValueError:
                    raise ValueError(err_msg.format(input_str))
    # We need 96 bits precision for accurate mantissa in this base-10 to base-2
    # conversion, will use native Python integers, as speed is not critical
    # here.
    # >>> import mpmath
    # >>> mpmath.mp.dps = 30
    # >>> mpmath.log("10.") / mpmath.log("2.") * mpmath.mpf("1.e25")
    # mpmath.log("10.") / mpmath.log("2.") * mpmath.mpf(2**96)
    # mpf('263190258962436467100402834429.2138584375862')
                rr_hex = 263190258962436467100402834429
                exp_10, mod = divmod(exp_10 * rr_hex, 2**96)
                m *= 2.**(mod * 2.**-96)
            return m, exp_10
        else:
            raise ValueError(err_msg.format(input_str))

    @staticmethod
    def _extended_data_array(mantissa, exp_re, exp_im):
        """
        Builds the structured internal array.
        """
        mantissa_dtype = mantissa.dtype
        if mantissa_dtype not in Xrange_array._DTYPES:
            casted = False
            for cast_dtype in Xrange_array._DTYPES:
                if np.can_cast(mantissa_dtype, cast_dtype, "safe"):
                    mantissa = mantissa.astype(cast_dtype)
                    casted = True
                    break
            if not casted:
                raise ValueError("Unsupported type for Xrange_array {}".format(
                    mantissa_dtype))

        # Builds the field-array
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

    @real.setter
    def real(self, value):
        self.real[:] = value

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

    @imag.setter
    def imag(self, value):
        self.imag[:] = value

    @staticmethod
    def empty(shape, dtype, asarray=False):
        """ Return a new Xrange_array of given shape and type, without
        initializing entries.
        
        if asarray is True, return a view as an array, otherwise (default)
        return a Xrange_array
        """
        if dtype in Xrange_array._COMPLEX_DTYPES:
            extended_dtype = np.dtype([('mantissa', dtype),
                                       ('exp_re', np.int32),
                                       ('exp_im', np.int32)], align=False)
        else:
            extended_dtype = np.dtype([('mantissa', dtype),
                                       ('exp_re', np.int32)], align=False)
        if asarray:
            return np.empty(shape, dtype=extended_dtype)
        else:
            return np.empty(shape, dtype=extended_dtype).view(Xrange_array)

    @staticmethod
    def zeros(shape, dtype):
        """ Return a new Xrange_array of given shape and type, with all entries
        initialized with 0."""
        ret = Xrange_array.empty(shape, dtype, asarray=True)
        if type(dtype) is np.dtype:
            dtype = dtype.type
        ret["mantissa"] = dtype(0.)
        ret["exp_re"] = np.int32(0)
        if dtype in Xrange_array._COMPLEX_DTYPES:
            ret["exp_im"] = np.int32(0)
        return ret.view(Xrange_array)

    @staticmethod
    def ones(shape, dtype):
        """ Return a new Xrange_array of given shape and type, with all entries
        initialized with 1."""
        ret = Xrange_array.empty(shape, dtype, asarray=True)
        if type(dtype) is np.dtype:
            dtype = dtype.type
        ret["mantissa"] = dtype(1.)
        ret["exp_re"] = np.int32(0)
        if dtype in Xrange_array._COMPLEX_DTYPES:
            ret["exp_im"] = np.int32(0)
        return ret.view(Xrange_array)

    def fill(self, val):
        """ Fill the array with val.
        Parameter
        ---------
        val : numpy scalar of a Xrange_array of null shape
        """
        fill_dict = {"exp_re": 0, "exp_im": 0}
        if np.isscalar(val):
            fill_dict["mantissa"] = val
        elif isinstance(val, Xrange_array) and (val.shape == ()):
            fill_dict["mantissa"] = val._mantissa
            fill_dict["exp_re"] = val._exp_re
            fill_dict["exp_im"] = 0 if not(val.is_complex) else val._exp_im
        else:
            raise ValueError("Invalid input to Xrange_array.fill, "
                    "expected a numpy scalar of a Xrange_array of null shape")
        keys = ["mantissa", "exp_re"]
        keys += ["exp_im"] if self.is_complex else []
        for key in keys:
            (np.asarray(self)[key]).fill(fill_dict[key])


    def to_standard(self):
        """ Returns the Xrange_array downcasted to standard np.ndarray ;
        obviously, may overflow. """
        if self.is_complex:
            return (self._mantissa.real * (2.**self._exp_re) +
                   1j * self._mantissa.imag * (2.**self._exp_im))
        else:
            return (self._mantissa * (2.**self._exp_re))

    @staticmethod
    def _build_complex(re, im):
        """ Build a complex Xrange_array from 2 similar shaped and typed
        Xrange_array (imag and real parts)"""
        m_re = re._mantissa
        dtype = np.complex128 if (m_re.dtype == np.float64) else np.complex64
        c = np.empty(m_re.shape, dtype=dtype)
        c.real = m_re
        c.imag = im._mantissa
        return Xrange_array(c, re._exp_re, im._exp_re)

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
            return np.int32(0) #np.zeros([], dtype=np.int32)

    @property
    def _exp_im(self):
        """ Returns the imaginary part of Xrange_array exp array"""
        try:
            return np.asarray(self["exp_im"])
        except (IndexError, ValueError): # We are view casting a np.array
            return np.int32(0) #np.zeros([], dtype=np.int32)

    @property
    def _exp(self):
        """ Returns the Xrange_array exp array: int32 or (int32, int32) """
        if self.is_complex:
            return (self._exp_re, self._exp_im)
        else:
            return self._exp_re

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
        f, exp2 = np.frexp(f)
        exp = np.where(f == 0., 0, exp + exp2)
        return f, exp


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
        out = kwargs.pop("out", None)
        if out is not None:
            if ufunc.nout == 1: # Only supported case to date
                out = np.asarray(out[0])

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
                casted_inputs += (Xrange_array(x),)
            else:
                raise NotImplementedError("Xrange_array view not handled for "
                    "this type {}, supported types : {}.".format(
                            type(x), Xrange_array._HANDLED_TYPES))

        if method == "__call__":
            if ufunc in [np.add, np.subtract]:
                return self._add(ufunc, *casted_inputs, out=out)
            elif ufunc is np.negative:
                return self._negative(*casted_inputs, out=out)
            elif ufunc in [np.multiply, np.true_divide]:
                return self._mul(ufunc, *casted_inputs)
            elif ufunc in [np.greater, np.greater_equal, np.less,
                           np.less_equal, np.equal, np.not_equal]:
                return self._compare(ufunc, *casted_inputs, out=out)

            elif ufunc is np.absolute:
                return self._abs(*casted_inputs, out=out)
            elif ufunc is np.sqrt:
                return self._sqrt(*casted_inputs, out=out)
            elif ufunc is np.square:
                return self._square(*casted_inputs, out=out)
            elif ufunc is np.conj:
                return self._conj(*casted_inputs, out=out)
            elif ufunc is np.log:
                return self._log(*casted_inputs, out=out)

        elif method == "reduce" and ufunc is np.add:
            return self._add_reduce(*casted_inputs, out=out, **kwargs)

        else:
            raise NotImplementedError("ufunc {} method {} not implemented for "
                                      "Xrange_array".format(ufunc, method))

    def abs2(self, out=None):
        """
        Return the square of np.abs(self) (for optimisation purpose).
        """
        if out is None:
            out = Xrange_array.empty(self.shape,
                    dtype=self._mantissa.real.dtype, asarray=True)

        if self.is_complex:
            out["mantissa"], out["exp_re"] = Xrange_array._normalize(
                *Xrange_array._coexp_ufunc(
                    self._mantissa.real**2, 2 * self._exp_re,
                    self._mantissa.imag**2, 2 * self._exp_im, np.add))
        else:
            out["mantissa"], out["exp_re"] = Xrange_array._normalize(
                    self._mantissa**2, 2 * self._exp_re)
        return out.view(Xrange_array)

    @staticmethod
    def _conj(*inputs, out=None):
        """ x -> np.conj(x) """
        op0, = inputs
        m0 = op0._mantissa
        if out is None:
            out = Xrange_array.empty(m0.shape, dtype=m0.dtype, asarray=True)

        out["mantissa"] = np.conj(op0._mantissa)
        out["exp_re"] = op0._exp_re
        if op0.is_complex:
            out["exp_im"] = op0._exp_im
        return out.view(Xrange_array)

    @staticmethod
    def _square(*inputs, out=None):
        """ x -> x**2  """
        op0, = inputs
        m0 = op0._mantissa
        exp0_re = op0._exp_re
        exp0_im = op0._exp_im
        if out is None:
            out = Xrange_array.empty(m0.shape, dtype=m0.dtype, asarray=True)

        if op0.is_complex:
            m_re, exp_re = Xrange_array._coexp_ufunc(
                    np.square(m0.real), 2 * exp0_re,
                    -np.square(m0.imag), 2 * exp0_im, np.add)
            out["mantissa"].real, out["exp_re"] = Xrange_array._normalize(
                    m_re, exp_re)
            out["mantissa"].imag, out["exp_im"] = Xrange_array._normalize(
                    2 * m0.real * m0.imag, exp0_re + exp0_im)
        else: # real case
            out["mantissa"], out["exp_re"] = Xrange_array._normalize(
                    np.square(m0), 2 * exp0_re)
        return out.view(Xrange_array)

    @staticmethod
    def _log(*inputs, out=None):
        """ x -> np.log(x)  """
        op0, = inputs
        m0 = op0._mantissa
        if out is None:
            out = Xrange_array.empty(m0.shape, dtype=m0.dtype, asarray=True)

        ln2 = 0.6931471805599453
        if op0.is_complex:
            m_re, exp_re = Xrange_array._normalize(m0.real, op0._exp_re)
            m_im, exp_im = Xrange_array._normalize(
                    m0.imag, op0._exp_im)
            m_re *= 2.
            exp_re -= 1
            m_re, m_im, e = Xrange_array._coexp_ufunc(m_re, exp_re, m_im,
                                                      exp_im, None)
            m = m_re + 1.j * m_im
            out["exp_im"] = 0
        else:
            m, e = Xrange_array._normalize(m0, op0._exp_re)
            m *= 2.
            e -= 1
            m_re = m

        # Avoid loss of significant digits if e * ln2 close to log(m)
        # ie m close to 2.0
        e_is_m1 = (e == -1)
        if np.isscalar(m):
            if e_is_m1:
                m[e_is_m1] *= 0.5
                e[e_is_m1] += 1
        else:
            m[e_is_m1] *= 0.5
            e[e_is_m1] += 1

        out["mantissa"] = np.log(m) + m_re.dtype.type(e * ln2)
        out["exp_re"] = 0
        return out.view(Xrange_array)

    @staticmethod
    def _sqrt(*inputs, out=None):
        """ x -> np.sqrt(x)  """
        sqrt0, = inputs
        m0 = sqrt0._mantissa
        if out is None:
            out = Xrange_array.empty(m0.shape, dtype=m0.dtype, asarray=True)
        
        if sqrt0.is_complex:
            m_re, m_im, exp = Xrange_array._coexp_ufunc(
                    m0.real, sqrt0._exp_re,
                    m0.imag, sqrt0._exp_im, None)
            m = m_re + 1.j * m_im
            even_exp = ((exp % 2) == 0).astype(np.bool)
            exp = np.where(even_exp, exp // 2, (exp - 1) // 2)
            out["mantissa"] = np.sqrt(np.where(even_exp, m, m * 2.))
            out["exp_re"] = exp
            out["exp_im"] = exp
        else:
            even_exp = ((sqrt0._exp_re % 2) == 0).astype(np.bool)
            out["mantissa"] = np.sqrt(np.where(even_exp, sqrt0._mantissa,
                                 sqrt0._mantissa * 2.))
            out["exp_re"] = np.where(even_exp, sqrt0._exp_re // 2,
                    (sqrt0._exp_re - 1) // 2)
        return out.view(Xrange_array)

    @staticmethod
    def _abs(*inputs, out=None):
        """ x -> np.abs(x) """
        op0, = inputs
        m0 = op0._mantissa
        exp0_re = op0._exp_re

        if out is None:
            out = Xrange_array.empty(m0.shape, dtype=m0.real.dtype,
                                     asarray=True)
        if op0.is_complex:
            Xrange_array._sqrt((op0.real * op0.real + op0.imag * op0.imag),
                               out=out)
        else:
            out["mantissa"] = np.abs(m0)
            out["exp_re"] = exp0_re
        return out.view(Xrange_array)

    @staticmethod
    def _compare(ufunc, *inputs, out=None):
        """ compare x and y """
        op0, op1 = inputs
        m0 = op0._mantissa
        m1 = op1._mantissa
        if out is None:
            out = np.empty(np.broadcast(m0, m1).shape, dtype=np.bool)

        if (op0.is_complex or op1.is_complex):
            if ufunc in [np.equal, np.not_equal]:
                re_eq = Xrange_array._coexp_ufunc(
                        m0.real, op0._exp_re, m1.real, op1._exp_re, ufunc)[0]
                im_eq = Xrange_array._coexp_ufunc(
                        m0.imag, op0._exp_im, m1.imag, op1._exp_im, ufunc)[0]
                if ufunc is np.equal:
                    out = re_eq & im_eq
                else:
                    out = re_eq | im_eq
            else:
                raise NotImplementedError(
                    "{} Not supported for complex".format(ufunc))
        else:
            out = Xrange_array._coexp_ufunc(m0, op0._exp_re, m1,
                                            op1._exp_re, ufunc)[0]
        return out

    @staticmethod
    def _mul(ufunc, *inputs, out=None):
        """ internal auxilliary function for * and / operators """
        op0, op1 = inputs
        m0 = op0._mantissa
        exp0_re = op0._exp_re
        exp0_im = op0._exp_im
        if ufunc is np.multiply:
            m1 = op1._mantissa
            exp1_re = op1._exp_re
            exp1_im = op1._exp_im
        elif ufunc is np.true_divide:
            m1 = 1. / op1._mantissa
            exp1_re = -op1._exp_re
            exp1_im = -op1._exp_im

        if out is None:
            out = Xrange_array.empty(np.broadcast(m0, m1).shape,
                                   dtype=np.result_type(m0, m1), asarray=True)

        if (op0.is_complex or op1.is_complex):
            m_re, exp_re = Xrange_array._coexp_ufunc(
                        m0.real * m1.real, exp0_re + exp1_re,
                        -m0.imag * m1.imag, exp0_im + exp1_im, np.add)
            out["mantissa"].real, out["exp_re"] = Xrange_array._normalize(
                    m_re, exp_re)
            m_im, exp_im = Xrange_array._coexp_ufunc(
                        m0.real * m1.imag, exp0_re + exp1_im,
                        m0.imag * m1.real, exp0_im + exp1_re, np.add)
            out["mantissa"].imag, out["exp_im"] = Xrange_array._normalize(
                    m_im, exp_im)
        else:
            out["mantissa"], out["exp_re"] = Xrange_array._normalize(
                    m0* m1, exp0_re + exp1_re)

        return out.view(Xrange_array)

    @staticmethod
    def _negative(*inputs, out=None):
        """ x -> -x """
        op0, = inputs
        m0 = op0._mantissa
        if out is None:
            out = Xrange_array.empty(op0.shape, dtype=m0.dtype, asarray=True)
        out["mantissa"] = -m0
        out["exp_re"] = op0._exp_re
        if op0.is_complex:
            out["exp_im"] = op0._exp_im
        return out.view(Xrange_array) 

    @staticmethod
    def _add(ufunc, *inputs, out=None):
        """ internal auxilliary function for + and - operators """
        op0, op1 = inputs
        m0 = op0._mantissa
        m1 = op1._mantissa
        if out is None:
            out = Xrange_array.empty(np.broadcast(m0, m1).shape,
                                   dtype=np.result_type(m0, m1), asarray=True)

        if (op0.is_complex or op1.is_complex):
            out["mantissa"].real, out["exp_re"] = Xrange_array._coexp_ufunc(
                m0.real, op0._exp_re, m1.real, op1._exp_re, ufunc)
            out["mantissa"].imag, out["exp_im"]  = Xrange_array._coexp_ufunc(
                m0.imag, op0._exp_im, m1.imag, op1._exp_im, ufunc)
        else:
            out["mantissa"], out["exp_re"] = Xrange_array._coexp_ufunc(
                    m0, op0._exp_re, m1, op1._exp_re, ufunc)

        return out.view(Xrange_array) 

    @staticmethod
    def _coexp_ufunc(m0, exp0, m1, exp1, ufunc=None):
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
        exp0 = np.broadcast_to(exp0, co_m0.shape)
        exp1 = np.broadcast_to(exp1, co_m0.shape)

        m0_null = (m0 == 0.)
        m1_null = (m1 == 0.)
        d_exp = exp0 - exp1

        if (co_m0.shape == ()):
            if ((exp1 > exp0) & ~m1_null):
                co_m0 = Xrange_array._exp2_shift(co_m0, d_exp)
            if ((exp0 > exp1) & ~m0_null):
                co_m1 = Xrange_array._exp2_shift(co_m1, -d_exp)
            exp = np.maximum(exp0, exp1)
            if m0_null:
                exp = exp1
            if m1_null:
                exp = exp0
        else:
            bool0 = ((exp1 > exp0) & ~m1_null)
            co_m0[bool0] = Xrange_array._exp2_shift(
                    co_m0[bool0], d_exp[bool0])
            bool1 = ((exp0 > exp1) & ~m0_null)
            co_m1[bool1] = Xrange_array._exp2_shift(
                    co_m1[bool1], -d_exp[bool1])
            exp = np.maximum(exp0, exp1)
            exp[m0_null] = exp1[m0_null]
            exp[m1_null] = exp0[m1_null]

        if ufunc is not None: 
            return (ufunc(co_m0, co_m1), exp)
        else:
            return (co_m0, co_m1, exp)

    @staticmethod
    def _add_reduce(*inputs, out=None, **kwargs):
        """
        """
        if out is not None:
            raise NotImplementedError("`out` keyword not (yet) immplemented "
                "for ufunc {} method {} of Xrange_array".format(
                        np.add, "reduce"))
        op, = inputs
        if op.is_complex:
            re = Xrange_array._add_reduce(op.real, **kwargs)
            im = Xrange_array._add_reduce(op.imag, **kwargs)
            return Xrange_array._build_complex(re, im)
        else:
            axis = kwargs.get("axis", 0)
            co_exp_acc = getattr(np.maximum, "reduce")(op._exp_re, axis=axis)
            brodcast_co_exp_acc = (np.expand_dims(co_exp_acc, axis)
                    if axis is not None else co_exp_acc)

            co_m = Xrange_array._exp2_shift(op._mantissa, 
                                            op._exp_re - brodcast_co_exp_acc)

            return Xrange_array(*Xrange_array._normalize(
                    getattr(np.add, "reduce")(co_m, axis=axis), co_exp_acc))


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
             "\ninternal dtype: " + str(self.dtype) + 
             "\nbase 10 representation:\n" +
             self._to_str_array().__repr__())
        return s

    def __str__(self):
        """
        String representation of self. Takes into account the value of
        np.get_printoptions(precision)

        Usage :
        with np.printoptions(precision=2) as opts:
            print(extended_range_array)
        """
        return np.array2string(self._to_str_array(), 
            formatter={'numpystr':lambda x: "{}".format(x)})

    def _to_str_array(self):
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

        return s
#    np.array2string(s, 
#            formatter={'numpystr':lambda x: "{}".format(x)})

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
        m2, exp2 = Xrange_array._normalize(m2, exp2) # 0.5 <= m2 < 1.0
        m10, exp10 = Xrange_array._rebase_2to10(m2, exp2)

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
            p_char = im_p_char # '\u2795' bold +
            m_char = im_m_char # '\u2796' bold -
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

    @staticmethod
    def _rebase_2to10(m2, exp2):
        """
        Parameters:
        m2 mantissa in base 2
        exp2 int32 exponent in base 2

        Returns:
        m10 mantissa in base 10
        exp10 int32 exponent in base 10

        Note : 
        This is a high-precision version of:
            > r = math.log10(2)
            > exp10, mod = np.divmod(exp2 * r, 1.)
            > return m2 * 10.**mod, exp10

        In order to guarantee an accuracy > 15 digits (in reality, close to 16)
        for `mod` with the 9-digits highest int32 base 2 exponent (2**31 - 1)
        we use an overall precision of 96 bits for this divmod.

        https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html
        """
        # We will divide by hand in base 2**32 (chosen so that exp2 * ri does
        # not overflow an int64 with the largest exp2 == 2**31-1), ri < 2**32.
        # >>> import mpmath
        # >>> mpmath.mp.dps = 35
        # >>> mpmath.log("2.") / mpmath.log("10.") * mpmath.mpf(2**96)
        # mpf('23850053418134191015272426710.02243475524574')
        r_96 = 23850053418134191015272426710
        mm = [None] * 3
        for i in range(3):
            ri = (r_96 >> (32 * (2 - i))) & 0xffffffff
            mm[i] = exp2.astype(np.int64) * ri
            if i == 0: # extract the integer `mod` part
                di, mm[i] = np.divmod(mm[i], 0x100000000)
                d = di.astype(np.int64)
        m = (mm[0] + (mm[1] + mm[2] * 2.**-32) * 2.**-32) * 2**-32
        return  m2 * 10.**m, d.astype(np.int32)


    def __setitem__(self, key, value):
        """ Can be given either a Xrange_array or a complex of float array-like
        (See 'supported types')
        """
        if type(value) is not Xrange_array:
            value = Xrange_array(np.asarray(value))
        if self.is_complex and not(value.is_complex):
            np.ndarray.__setitem__(self.real, key, value)
            np.ndarray.__setitem__(self.imag, key, np.zeros_like(value))
            return
        if not(self.is_complex) and value.is_complex: 
            raise ValueError("Can't assign complex values to a real"
                             " Xrange_array")
        np.ndarray.__setitem__(self, key, value)

    def __getitem__(self, key):
        """ For single item, return array of empty shape rather than a scalar,
        to allow pretty print and maintain assignement behaviour consistent.
        """
        res = np.ndarray.__getitem__(self, key)
        if np.isscalar(res):
            return np.asarray(res).view(Xrange_array)
        return res

    def __eq__(self, other):
        """ Ensure that `!=` is handled by Xrange_array instance. """
        return np.equal(self, other)

    def __ne__(self, other):
        """ Ensure that `==` is handled by Xrange_array instance. """
        return np.not_equal(self, other)


class Xrange_polynomial(np.lib.mixins.NDArrayOperatorsMixin):
    """
    One-dimensionnal SA_Polynomial class which provides:
        - the standard Python numerical methods ‘+’, ‘-‘, ‘*' 
    
    Parameters
    ----------
    coeffs: array_like - can be viewed as a Xrange_array
    Polynomial coefficients in order of increasing degree, i.e.,
    (1, 2, 3) give 1 + 2*x + 3*x**2.

    cutoff : int, maximum degree coefficient. At instanciation but also for
    the subsequent operations, monomes of degree above cutoff will be 
    disregarded.
    
    
    """  
    # Unicode character mappings for "pretty print" of the polynomial
    _superscript_mapping = str.maketrans({
        "0": "⁰",
        "1": "¹",
        "2": "²",
        "3": "³",
        "4": "⁴",
        "5": "⁵",
        "6": "⁶",
        "7": "⁷",
        "8": "⁸",
        "9": "⁹"
    })
    
    def __init__(self, coeffs, cutdeg):
        if isinstance(coeffs, Xrange_array):
            self.coeffs = coeffs[0:cutdeg+1]
        else:
            self.coeffs = (np.asarray(coeffs)[0:cutdeg+1]).view(Xrange_array)
        if self.coeffs.ndim != 1:
            raise ValueError("Only 1-d inputs for Xrange_polynomial")
        self.cutdeg = cutdeg

    def  __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        casted_inputs = ()
        cutdeg = ()

        for x in inputs:
            # Only support operations with instances of 
            # Xrange_array._HANDLED_TYPES.
            if isinstance(x, Xrange_polynomial):
                cutdeg += (x.cutdeg,)
                casted_inputs += (x.coeffs,)
            elif isinstance(x, Xrange_array):
                casted_inputs += (x,)
            elif isinstance(x, np.ndarray):
                casted_inputs += (x.view(Xrange_array),)
            elif isinstance(x, numbers.Number):
                casted_inputs += (Xrange_array([x]),)
            elif isinstance(x, list):
                casted_inputs += (Xrange_array(x),)
            else:
                raise NotImplementedError("SA_Polynomial view not handled for "
                    "this type {}, supported types : {}.".format(type(x),
                    Xrange_array._HANDLED_TYPES + (Xrange_polynomial,)))

        cutdeg = min(cutdeg)
        out = kwargs.pop("out", None)

        if method == "__call__":
            if ufunc in [np.add, np.subtract]:
                return self._add(ufunc, *casted_inputs, cutdeg=cutdeg, out=out)
            elif ufunc is np.negative:
                return self._negative(*casted_inputs, cutdeg=cutdeg, out=out)
            elif ufunc is np.multiply:
                return self._mul(*casted_inputs, cutdeg=cutdeg, out=out)

    @staticmethod
    def _add(ufunc, *inputs, cutdeg, out=None):
        """ Add or Subtract 2 SA_Polynomial """
        op0, op1 = inputs
        res_len = min(max(len(op0), len(op1)), cutdeg + 1)
        op0_len = min(len(op0), res_len)
        op1_len = min(len(op1), res_len)

        dtype=np.result_type(op0._mantissa, op1._mantissa)
        res = Xrange_array(np.zeros([res_len], dtype=dtype))

        res[:op0_len] += op0[:op0_len]
        if ufunc is np.add:
            res[:op1_len] += op1[:op1_len]
        elif ufunc is np.subtract: 
            res[:op1_len] -= op1[:op1_len]
        return Xrange_polynomial(res, cutdeg=cutdeg)

    @staticmethod
    def _negative(*inputs, cutdeg, out=None):
        """ Change sign of a SA_Polynomial """
        op0 = inputs
        return Xrange_polynomial(-op0, cutdeg=cutdeg)

    @staticmethod
    def _mul(*inputs, cutdeg, out=None):
        """ Product of 2 SA_Polynomial """
        op0, op1 = inputs
        # This is a convolution, fix the window with the shortest poly op0,
        # swapping poly if needed. (Note we do not use fft but direct
        # calculation)
        window = min(len(op0), len(op1))
        if len(op0) > window:
            op0, op1 = op1, op0
        l1 = len(op1)
        cutoff_res = min(window + l1- 2, cutdeg) # the degree..
        # The first term is a0 * b0, so sum(a[0:l0] * b[0:-l0])
        # the higest term is sum(a[0:l0] * b[cutoff_res:cutoff_res-l0])
        # So we need to shift b down to -l0 and up to cutoff_res
        #
        # op0                   0  1  2  ...  l0-1
        # op1        ...  2  1  0 -1 -2  ... -l0+1
        #
        # op0                   0  1  2  ...  l0-1
        # op1    >> (cutoff_res)  ...  2  1  0 -1 -2  ... -l0+1
        op1 = np.pad(op1, (window - 1, cutoff_res - l1 + 1),
                     mode='constant').view(Xrange_array)
        shift = np.arange(0, cutoff_res + 1)
        take1 = shift[:, np.newaxis] + (np.arange(window - 1 , -1, -1))
        return Xrange_polynomial(np.sum(op0 * np.take(op1, take1), axis=1),
                cutdeg=cutdeg) # /!\ not cutoff_res

    def __call__(self, arg):
        """ Call self as a function.
        """
        if not isinstance(arg, Xrange_array):
            arg = Xrange_array(np.asarray(arg))

        res_dtype = np.result_type(arg._mantissa, self.coeffs._mantissa)
        res = Xrange_array.empty(arg.shape, dtype=res_dtype)
        res.fill(self.coeffs[-1])

        for i in range(2, len(self.coeffs) + 1):
            res = self.coeffs[-i] + res * arg
        return res

    def __repr__(self):
        return ("SA_Polynomial(cutdeg="+ str(self.cutdeg) +",\n" +
                self.__str__() + ")")

    def __str__(self):
        return self._to_str()

    def _to_str(self):
        """
        Generate the full string representation of the polynomial, using
        `_monome_base_str` to generate each polynomial term.
        """
        str_coeffs = np.abs(self.coeffs)._to_str_array()
        linewidth = np.get_printoptions().get('linewidth', 75)
        if linewidth < 1:
            linewidth = 1
        if self.coeffs[0] >= 0.:
            out = f"{str_coeffs[0][1:]}"
        else:
            out = f"-{str_coeffs[0][1:]}"
        for i, coef in enumerate(str_coeffs[1:]):
            out += " "
            power = str(i + 1)
            # 1st Polynomial coefficient
            if self.coeffs[i + 1] >= 0.:
                next_term = f"+ {coef}"
            else:
                next_term = f"- {coef}"
            # Polynomial term
            next_term += self._monome_base_str(power, "X")
            # Length of the current line with next term added
            line_len = len(out.split('\n')[-1]) + len(next_term)
            # If not the last term in the polynomial, it will be two           
            # characters longer due to the +/- with the next term
            if i < len(self.coeffs[1:]) - 1:
                line_len += 2
            # Handle linebreaking
            if line_len >= linewidth:
                next_term = next_term.replace(" ", "\n", 1)
            next_term = next_term.replace("  ", " ")
            out += next_term
        return out

    @classmethod
    def _monome_base_str(cls, i, var_str):
        return f"·{var_str}{i.translate(cls._superscript_mapping)}"
