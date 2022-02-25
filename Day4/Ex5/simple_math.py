"""
A collection of simple math operations
"""

def simple_add(a,b):
    """
    Sum of two numbers
    
    Parameters
    ----------
    
    a: first summand (integer, float)
    b: second summand (integer, float)

    Returns
    -------
    c: sum of a and b, float
    
    Examples
    --------
    >>> simple_math.simple_add(1,2)
    3
    >>> simple_math.simple_add(3,6)
    9
    """
    return a+b

def simple_sub(a,b):
    """
    Substraction of two numbers
    
    Parameters
    ----------
    
    a: minuend (integer, float)
    b: subtrahend (integer, float)

    Returns
    -------
    c: difference of a and b, float
    
    Examples
    --------
    >>> simple_math.simple_sub(10,2)
    8
    >>> simple_math.simple_sub(3,6)
    -3
    """
    return a-b

def simple_mult(a,b):
    """
    Product of two numbers
    
    Parameters
    ----------
    
    a: first multiple (integer, float)
    b: second multiple (integer, float)

    Returns
    -------
    c: product of a and b, float
    
    Examples
    --------
    >>> simple_math.simple_mult(2,4)
    8
    >>> simple_math.simple_mult(3,6)
    18
    """
    return a*b

def simple_div(a,b):
    """
    Ratio between two numbers
    
    Parameters
    ----------
    
    a: dividend (integer, float)
    b: nonzero divisor (integer, float)

    Returns
    -------
    c: quotient, float
    
    Examples
    --------
    >>> simple_math.simple_div(2,4)
    8
    >>> simple_math.simple_div(3,6)
    18
    """
    return a/b

def poly_first(x, a0, a1):
    """
    First order polynomial
    
    Parameters
    ----------
    
    x: argument(integer, float)
    a0: first coefficient (integer, float)
    a1: second coefficient (integer, float)

    Returns
    -------
    c: a0 + a1*x, float
    
    Examples
    --------
    >>> simple_math.poly_first(1,1,1)
    2
    >>> simple_math.poly_first(3,1,6)
    19
    """
    return a0 + a1*x

def poly_second(x, a0, a1, a2):
    """
    Second order polynomial
    
    Parameters
    ----------
    
    x: argument(integer, float)
    a0: first coefficient (integer, float)
    a1: second coefficient (integer, float)
    a2: third coefficient (integer, float)

    Returns
    -------
    c: a0 + a1*x + a2*(x**2), float
    
    Examples
    --------
    >>> simple_math.poly_second(1,0,1,2)
    3
    >>> simple_math.poly_second(2,1,1,4)
    19
    """
    return poly_first(x, a0, a1) + a2*(x**2)

def pow(x, a):
    """
    Power operation
    
    Parameters
    ----------
    
    x: argument(integer, float)
    a: power (integer, float)

    Returns
    -------
    c: x**a, float
    
    Examples
    --------
    >>> simple_math.pow(2,2)
    4
    >>> simple_math.pow(3,2)
    9
    """
    return x**a