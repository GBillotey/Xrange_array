# -*- coding: utf-8 -*-
import numpy as np
from arrays import Xrange_array
import time


def _matching(res, expected, almost=False, dtype=None, cmp_op=False, ktol=1.5):
    if not cmp_op:
        res = res.to_standard()
    if almost:
        np.testing.assert_allclose(res, expected,
                                   rtol= ktol * np.finfo(dtype).eps)
    else:
        np.testing.assert_array_equal(res, expected)



def _test_op1(ufunc, almost=False, cmp_op=False, ktol=1.5):
    print("testing function", ufunc)
    rg = np.random.default_rng(100)

    n_vec = 500
    max_bin_exp = 20
    
    # testing binary operation of reals extended arrays
    for dtype in [np.float64, np.float32]: 
        op1 = rg.random([n_vec], dtype=dtype)
        op1 *= 2.**rg.integers(low=-max_bin_exp, high=max_bin_exp, 
                               size=[n_vec])
        expected = ufunc(op1)
        res = ufunc(Xrange_array(op1))

        _matching(res, expected, almost, dtype, cmp_op, ktol)

        # Checking datatype
        assert res._mantissa.dtype == dtype

        # with non null shift array # culprit
        exp_shift_array = rg.integers(low=-max_bin_exp, high=max_bin_exp, 
                                      size=[n_vec])
        expected = ufunc(op1 * (2.**exp_shift_array).astype(dtype))

        _matching(ufunc(Xrange_array(op1, exp_shift_array)),
                  expected, almost, dtype, cmp_op, ktol)

    # testing binary operation of reals extended arrays
    for dtype in [np.float32, np.float64]:
        op1 = (rg.random([n_vec], dtype=dtype) +
                   1j*rg.random([n_vec], dtype=dtype))
        op1 *= 2.**rg.integers(low=-max_bin_exp, high=max_bin_exp, 
                               size=[n_vec])
        expected = ufunc(op1)
        res = ufunc(Xrange_array(op1))
        _matching(res, expected, almost, dtype, cmp_op, ktol)

        # Checking datatype
        to_complex = {np.float32: np.complex64,
                 np.float64: np.complex128}
        if ufunc in [np.abs]:
            assert res._mantissa.dtype == dtype
        else:
            assert res._mantissa.dtype == to_complex[dtype]

        # with non null shift array
        exp_shift_array = rg.integers(low=-max_bin_exp, high=max_bin_exp, 
                                      size=[n_vec])
        expected = ufunc(op1 * (2.**exp_shift_array))
        _matching(ufunc(Xrange_array(op1, exp_shift_array,
                                            exp_shift_array)),
                  expected, almost, dtype, cmp_op, ktol)


def _test_op2(ufunc, almost=False, cmp_op=False):
    print("testing operation", ufunc)
    rg = np.random.default_rng(100)
#    ea_type = (Xrange_array._FLOAT_DTYPES + 
#               Xrange_array._COMPLEX_DTYPES)
    n_vec = 500
    max_bin_exp = 20
    exp_shift = 2
    
    # testing binary operation of reals extended arrays
    for dtype in [np.float32, np.float64]:
        op1 = rg.random([n_vec], dtype=dtype)
        op2 = rg.random([n_vec], dtype=dtype)
        op1 *= 2.**rg.integers(low=-max_bin_exp, high=max_bin_exp, 
                               size=[n_vec])
        op2 *= 2.**rg.integers(low=-max_bin_exp, high=max_bin_exp,
                               size=[n_vec])

        # testing operation between 2 Xrange_arrays OR between ER_A and 
        # a standard np.array
        expected = ufunc(op1, op2)
        res = ufunc(Xrange_array(op1), Xrange_array(op2))
        _matching(res, expected, almost, dtype, cmp_op)
        
#        # testing operation between 2 Xrange_arrays OR between ER_A and 
#        # a standard np.array xith dim 2
        expected_2d = ufunc(op1.reshape(50, 10),
                         op2.reshape(50, 10))
        res_2d = ufunc(Xrange_array(op1.reshape(50, 10)),
                    Xrange_array(op2.reshape(50, 10)))

        _matching(res_2d, expected_2d, almost, dtype, cmp_op)

        # Checking datatype
        if ufunc in [np.add, np.multiply, np.subtract, np.divide]:
            assert res._mantissa.dtype == dtype

        if ufunc != np.equal:
            _matching(ufunc(op1, Xrange_array(op2)),
                      expected, almost, dtype, cmp_op)
            _matching(ufunc(Xrange_array(op1), op2),
                      expected, almost, dtype, cmp_op)
        # Testing with non-null exponent
        exp_shift_array = rg.integers(low=-exp_shift, high=exp_shift, 
                                      size=[n_vec])
        expected = ufunc(op1 * 2.**exp_shift_array, op2 * 2.**-exp_shift_array)

            

        _matching(ufunc(Xrange_array(op1, exp_shift_array),
                        Xrange_array(op2, -exp_shift_array)),
                  expected, almost, dtype, cmp_op)
        # testing operation of an Xrange_array with a scalar
        if ufunc != np.equal:
            expected = ufunc(op1[0], op2)
            _matching(ufunc(op1[0], Xrange_array(op2)),
                      expected, almost, dtype, cmp_op)
            expected = ufunc(op2, op1[0])
            _matching(ufunc(Xrange_array(op2), op1[0]),
                      expected, almost, dtype, cmp_op)
    if cmp_op:
        return

    # testing binary operation of complex extended arrays
    for dtype in [np.float32, np.float64]:
        n_vec = 20
        max_bin_exp = 20
        rg = np.random.default_rng(1)
        
        op1 = (rg.random([n_vec], dtype=dtype) +
                   1j*rg.random([n_vec], dtype=dtype))
        op2 = (rg.random([n_vec], dtype=dtype) +
                   1j*rg.random([n_vec], dtype=dtype))
        op1 *= 2.**rg.integers(low=-max_bin_exp, high=max_bin_exp, 
                               size=[n_vec])
        op2 *= 2.**rg.integers(low=-max_bin_exp, high=max_bin_exp,
                               size=[n_vec])
        # testing operation between 2 Xrange_arrays OR between ER_A and 
        # a standard np.array
        expected = ufunc(op1, op2)
        res = ufunc(Xrange_array(op1), Xrange_array(op2))
        _matching(res, expected, almost, dtype)
        
        # Checking datatype
        if ufunc in [np.add, np.multiply, np.subtract, np.divide]:
            to_complex = {np.float32: np.complex64,
                 np.float64: np.complex128}
            assert res._mantissa.dtype == to_complex[dtype]

        _matching(ufunc(op1, Xrange_array(op2)),
                  expected, almost, dtype)
        _matching(ufunc(Xrange_array(op1), op2),
                  expected, almost, dtype)
        # Testing with non-null exponent (real and imag)
        expected = ufunc(op1 * 2.**exp_shift, op2 * 2.**-exp_shift)
        exp_shift_array = exp_shift * np.ones([n_vec], dtype=np.int32)
        _matching(ufunc(
                Xrange_array(op1, exp_shift_array, exp_shift_array),
                Xrange_array(op2, -exp_shift_array, -exp_shift_array)),
            expected, almost, dtype)
        # Testing cross product of real with complex
        expected = ufunc(op1 * 2.**exp_shift, (op2 * 2.**-exp_shift).real)
        exp_shift_array = exp_shift * np.ones([n_vec], dtype=np.int32)
        _matching(ufunc(
                Xrange_array(op1, exp_shift_array, exp_shift_array),
                Xrange_array(op2, -exp_shift_array, -exp_shift_array
                                    ).real),
            expected, almost, dtype)
        expected = ufunc((op1 * 2.**exp_shift).imag, op2 * 2.**-exp_shift)
        _matching(ufunc(
                Xrange_array(op1, exp_shift_array, exp_shift_array
                                    ).imag,
                Xrange_array(op2, -exp_shift_array, -exp_shift_array)),
            expected, almost, dtype)
        # testing operation of an Xrange_array with a scalar
        expected = ufunc(op1[0], op2)
        _matching(ufunc(op1[0], Xrange_array(op2)),
                  expected, almost, dtype)
        expected = ufunc(op2, op1[0])
        _matching(ufunc(Xrange_array(op2), op1[0]),
                  expected, almost, dtype)

def test_ops():
    """
    Testing all 4 basic operations +, -, *, /
    Testing all comparisions <=, <, >=, >, ==
    Testing abs and sqrt
    """
    for ufunc in [np.add, np.multiply, np.subtract]:
        _test_op2(ufunc, almost=True)
    _test_op2(np.divide, almost=True)

    for ufunc in [np.greater, np.greater_equal, np.less,
                  np.less_equal, np.equal]:
        _test_op2(ufunc, cmp_op=True)

    for ufunc in [np.abs, np.sqrt, np.square, np.conj, np.log]:
        _test_op1(ufunc, almost=True)
        


def test_edge_cases():
    _dtype = np.complex128
    base = np.linspace(0., 1500., 11, dtype=_dtype)
    base2 = np.linspace(-500., 500., 11, dtype=np.float64)
    # mul
    b = (Xrange_array((2. - 1.j) * base) * 
         Xrange_array((-1. + 1.j) * base2))
    expected = ((2. - 1j) * base) * ((-1. + 1.j) * base2)
    _matching(b, expected)
    # add
    b = (Xrange_array((2. - 1.j) * base) + 
         Xrange_array((-1. + 1.j) * base2))
    expected = ((2. - 1.j) * base) + ((-1. + 1j) * base2)
    _matching(b, expected)
    #  <=
    b = (Xrange_array((2. - 1j) * base).real <= 
         Xrange_array((-1. + 1j) * base2).real)
    expected = ((2. - 1j) * base).real <= ((-1. + 1j) * base2).real
    np.testing.assert_array_equal(b, expected)
    
    base = - np.ones([40], dtype=np.float64)
    base = np.linspace(0., 1., 40, dtype=np.float64)
    base2 = base + np.linspace(-1., 1., 40) * np.finfo(np.float64).eps
    exp = np.ones(40, dtype=np.int32)
    _base = Xrange_array(base, exp)

    _base2 = Xrange_array(base2 * 2., exp - 1)
    np.testing.assert_array_equal(_base == _base2, base == base2)
    np.testing.assert_array_equal(_base <= _base2, base <= base2)
    np.testing.assert_array_equal(_base >= _base2, base >= base2)
    np.testing.assert_array_equal(_base < _base2, base < base2)
    np.testing.assert_array_equal(_base > _base2, base > base2)

    _base2 = Xrange_array(base2 / 2., exp + 1)
    np.testing.assert_array_equal(_base == _base2, base == base2)
    np.testing.assert_array_equal(_base > _base2, base > base2)



def test_template_view():
    """
    Testing basic array capabilities
    Array creation via __new__, template of view
    real and imag are views
    """
    a = np.linspace(0., 5., 12, dtype=np.complex128)
    b = Xrange_array(a)

    # test shape of b and its mantissa / exponenent fields
    assert b.shape == a.shape
    assert b._mantissa.shape == a.shape
    assert b._exp_re.shape == a.shape

    # b is a full copy not a view
    b11_val = b[11]
    assert b[11] == b11_val#(5.0 + 0.j, 0, 0)
    m = b._mantissa
    assert m[11] == 5.
    a[11] = 10.
    assert b[11] == b11_val
    # you have to make a new instance to see the modification
    b = Xrange_array(a)
    assert b[11] != b11_val
    m = b._mantissa
    assert m[11] == 10.

    # Testing Xrange_array from template
    c = b[10:]
    # test shape Xrange_array subarray and its mantissa / exponenent
    assert c.shape == a[10:].shape
    assert c._mantissa.shape == a[10:].shape
    assert c._exp_re.shape == a[10:].shape
    # modifying subarray modifies array
    new_val = (12345., 6, 7)
    c[1] = new_val
    assert b[11] == c[1]
    # modifying array modifies subarray
    new_val = (98765., 4, 3)
    b[10] = new_val
    assert b[10] == c[0]

    # Testing Xrange_array from view
    d = a.view(Xrange_array)
    assert d.shape == a.shape
    assert d._mantissa.shape == a[:].shape

    # modifying array modifies view
    val = a[5]
    assert d._mantissa[5] == val
    val = 8888888.
    a[5] = val
    
    # Check that imag and real are views of the original array 
    e = Xrange_array(a + 2.j * a)
    assert e.to_standard()[4] == (20. + 40.j) / 11.
    e.real[4] = (np.pi, 0)
    e.imag[4] = (-np.pi, 0)

    assert e.to_standard()[4] == (1. - 1.j) * np.pi
    bb = Xrange_array(np.linspace(0., 5., 12, dtype=np.float64))
    
    np.testing.assert_array_equal(bb.real, bb)
    bb.real[0] = (1.875, 6)  # 120...
    assert bb.to_standard()[0] == 120.
    np.testing.assert_array_equal(bb.imag.to_standard(), 0.)




    


def timing_abs2_complex(dtype=np.float64):
    import time
    
    n_vec = 40000
    max_bin_exp = 20
    
    rg = np.random.default_rng(1) 
    
    op = rg.random([n_vec], dtype=dtype) + 1j*rg.random([n_vec], dtype=dtype)
    exp_re = rg.integers(-max_bin_exp, max_bin_exp)
    exp_im = rg.integers(-max_bin_exp, max_bin_exp)
    e_op = Xrange_array(op, exp_re, exp_im)
    op = op.real * 2.**exp_re + 1.j * op.imag * 2.**exp_im
    
    
    t0 = - time.time()
    e_res = e_op.abs2()
    t0 += time.time()
    
    t1 = - time.time()
    expected = op * np.conj(op)
    t1 += time.time()

    np.testing.assert_array_equal(e_res.to_standard(), expected)
    print("timing abs2", t0, t1, t0/t1)


def timing_op1_complex(ufunc, dtype=np.float64):
    import time
    
    n_vec = 40000
    max_bin_exp = 20
    
    rg = np.random.default_rng(1) 
    
    op = rg.random([n_vec], dtype=dtype) + 1j*rg.random([n_vec], dtype=dtype)
    exp_re = rg.integers(-max_bin_exp, max_bin_exp)
    exp_im = rg.integers(-max_bin_exp, max_bin_exp)
    e_op = Xrange_array(op, exp_re, exp_im)
    op = op.real * 2.**exp_re + 1.j * op.imag * 2.**exp_im
    
    
    t0 = - time.time()
    e_res = ufunc(e_op)#.abs2()
    t0 += time.time()
    
    t1 = - time.time()
    expected = ufunc(op)# * np.conj(add1)
    t1 += time.time()

    np.testing.assert_array_equal(e_res.to_standard(), expected)
    print("timing", ufunc, t0, t1, "ratio:", t0/t1)


def timing_op2_complex(ufunc, dtype=np.float64):
    n_vec = 40000
    max_bin_exp = 20
    rg = np.random.default_rng(1) 

    op1 = rg.random([n_vec], dtype=dtype) + 1j*rg.random([n_vec], dtype=dtype)
    exp1_re = rg.integers(-max_bin_exp, max_bin_exp)
    exp1_im = rg.integers(-max_bin_exp, max_bin_exp)
    e_op1 = Xrange_array(op1, exp1_re, exp1_im)
    op1 = op1.real * 2.**exp1_re + 1.j * op1.imag * 2.**exp1_im
    
    op2 = rg.random([n_vec], dtype=dtype) + 1j*rg.random([n_vec], dtype=dtype)
    exp2_re = rg.integers(-max_bin_exp, max_bin_exp)
    exp2_im = rg.integers(-max_bin_exp, max_bin_exp)
    e_op2 = Xrange_array(op2, exp2_re, exp2_im)
    op2 = op2.real * 2.**exp2_re + 1.j * op2.imag * 2.**exp2_im


    t0 = - time.time()
    e_res = ufunc(e_op1, e_op2)
    t0 += time.time()
    
    t1 = - time.time()
    expected = ufunc(op1, op2)
    t1 += time.time()

    np.testing.assert_array_equal(e_res.to_standard(), expected)
    print("timing", ufunc, t0, t1, "ratio:", t0/t1)


        
def test_underflow():
    _dtype = np.float64
    n = 100
    k = np.arange(n)
    a = 0.1 * np.ones([n], dtype=_dtype)
    b = a + 2.**(-k)
    expected = a - b
    
    e_a = Xrange_array(a)
    e_b = Xrange_array(b)
    e_res = e_a - e_b 
    res = e_res._mantissa * 2.**e_res._exp_re
    
    np.testing.assert_array_equal(res, expected)
    


def test_print():
    """
    Testing basic array prints
    """
    a = np.array([1., 1., np.pi, np.pi], dtype=np.float64)
    Xa = Xrange_array(a)
    for exp10 in range(1001):
        Xa = Xa * [-10., 0.1, 10., -0.1]
    str8 = ("[-1.00000000e+1001  1.00000000e-1001"
           "  3.14159265e+1001 -3.14159265e-1001]")
    str2 = ("[-1.00e+1001  1.00e-1001  3.14e+1001 -3.14e-1001]")
    with np.printoptions(precision=2, linewidth=100) as _:
        assert Xa.__str__() == str2
    with np.printoptions(precision=8, linewidth=100) as _:
        assert Xa.__str__() == str8

    a = np.array([0.999999, 1.00000, 0.9999996, 0.9999994], dtype=np.float64)
    str5 =  "[ 9.99999e-01  1.00000e+00  1.00000e+00  9.99999e-01]"
    for k in range(10):
        Xa = Xrange_array(a * 0.5**k, k * np.ones([4], dtype=np.int32))
        with np.printoptions(precision=5) as _:
            assert Xa.__str__() == str5

    a = 1.j * np.array([1., 1., np.pi, np.pi], dtype=np.float64)
    Xa = Xrange_array(a)
    for exp10 in range(1000):
        Xa = [-10., 0.1, 10., -0.1] * Xa
    str2 = ("[ 0.00e+00➕1.00e+1000j  0.00e+00➕1.00e-1000j"
            "  0.00e+00➕3.14e+1000j  0.00e+00➕3.14e-1000j]")
    with np.printoptions(precision=2, linewidth=100) as _:
        assert Xa.__str__() == str2
        
    a = np.array([[0.1, 10.], [np.pi, 1./np.pi]], dtype=np.float64)
    Xa = Xrange_array(a)
    Ya = np.copy(Xa).view(Xrange_array)
    for exp10 in range(21):
        Xa = np.sqrt(Xa * Xa * Xa * Xa)
    for exp10 in range(21):
        Ya = Ya * Ya
    str6 = ("[[ 1.000000e-2097152  1.000000e+2097152]\n"
            " [ 7.076528e+1042598  1.413122e-1042599]]")
    with np.printoptions(precision=6, linewidth=100) as _:
        assert Xa.__str__() == str6
        assert Ya.__str__() == str6

    Xa = Xrange_array([["123.456e-1789", "-.3e-7"], ["1.e700", "1.0"]])
    str6 = ("[[ 1.234560e-1787 -3.000000e-0008]\n"
            " [ 1.000000e+0700  1.000000e+0000]]")
    str6_sq = ("[[ 1.524138e-3574  9.000000e-0016]\n"
               " [ 1.000000e+1400  1.000000e+0000]]")
    Xb = Xa -1.j * Xa**2
    str6b = ("[[ 1.234560e-1787➖1.524138e-3574j "
               "-3.000000e-0008➖9.000000e-0016j]\n"
             " [ 1.000000e+0700➖1.000000e+1400j  "
               "1.000000e+0000➖1.000000e+0000j]]")
    with np.printoptions(precision=6, linewidth=100) as _:
        assert Xa.__str__() == str6
        assert (Xa**2).__str__() == str6_sq
        assert Xb.__str__() == str6b
        
    # Testing accuracy of mantissa for highest exponents    
    Xa = Xrange_array([["1.0e+646456992", "1.23456789012345e+646456992"], 
                       ["1.0e+646456991", "1.23456789012345e+646456991"], 
                       ["1.0e+646456990", "1.23456789012345e+646456990"],
                       ["-1.0e-646456991", "1.23456789012345e-646456991"], 
                       ["1.0e-646456992", "1.23456789012345e-646456992"]])
    str_14 = ("[[ 1.00000000000000e+646456992  1.23456789012345e+646456992]\n"
        " [ 1.00000000000000e+646456991  1.23456789012345e+646456991]\n"
        " [ 1.00000000000000e+646456990  1.23456789012345e+646456990]\n"
        " [-1.00000000000000e-646456991  1.23456789012345e-646456991]\n"
        " [ 1.00000000000000e-646456992  1.23456789012345e-646456992]]")
    with np.printoptions(precision=14, linewidth=100) as _:
        assert Xa.__str__() == str_14



    
if __name__ == "__main__":
    timing_op1_complex(np.square)
    timing_op2_complex(np.add)
    timing_op2_complex(np.multiply)
    timing_abs2_complex(dtype=np.float64)


    test_template_view()
    test_ops()
    test_edge_cases()
    test_underflow()
    
    test_print()
