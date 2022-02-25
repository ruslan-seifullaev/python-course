#!/usr/bin/env python

import simple_math

def test_simple_math():
    assert simple_math.simple_add(1, 2) == 3
    assert simple_math.simple_sub(8, 3) == 5
    assert simple_math.simple_mult(5, 5) == 25
    assert simple_math.simple_div(10, 5) == 2
    assert simple_math.poly_first(2, 2, 5) == 12
    assert simple_math.poly_second(1, 2, 3, 4) == 9
    assert simple_math.pow(2,6) == 64