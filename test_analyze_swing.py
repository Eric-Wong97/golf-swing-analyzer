import pytest
import numpy as np
from analyze_swing import calculate_angle

def test_calculate_angle_straight_line():
    """Test a 180 degree straight line"""
    a = [0, 1]
    b = [0, 0]
    c = [0, -1]
    angle = calculate_angle(a, b, c)
    assert np.isclose(angle, 180.0)

def test_calculate_angle_right_angle():
    """Test a 90 degree angle"""
    a = [1, 0]
    b = [0, 0]
    c = [0, 1]
    angle = calculate_angle(a, b, c)
    assert np.isclose(angle, 90.0)

def test_calculate_angle_acute_angle():
    """Test a 45 degree angle"""
    a = [1, 1]
    b = [0, 0]
    c = [1, 0]
    angle = calculate_angle(a, b, c)
    assert np.isclose(angle, 45.0)

def test_calculate_angle_reflex_handling():
    """Test that angles > 180 are converted to the inner angle (<= 180)"""
    # The angle from a to b to c in counter-clockwise is 270 degrees.
    # We expect our function to return the inner 90 degree angle.
    a = [1, 0]
    b = [0, 0]
    c = [0, -1]
    angle = calculate_angle(a, b, c)
    assert np.isclose(angle, 90.0)

def test_calculate_angle_zero_angle():
    """Test a 0 degree angle (points folded on top of each other)"""
    a = [1, 0]
    b = [0, 0]
    c = [2, 0]
    angle = calculate_angle(a, b, c)
    assert np.isclose(angle, 0.0)
