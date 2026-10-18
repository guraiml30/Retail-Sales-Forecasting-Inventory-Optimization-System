"""Run with:  python -m pytest -q   (or: python tests/test_inventory.py)"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inventory import classify_status, eoq, reorder_point, safety_stock, z_score


def test_z_score_95():
    assert math.isclose(z_score(0.95), 1.645, abs_tol=0.001)


def test_safety_stock():
    # z(95%)=1.645, sigma=10, lead=4 -> 1.645*10*2 = 32.9 -> 33
    assert safety_stock(10, 4, 0.95) == 33


def test_reorder_point():
    assert reorder_point(20, 5, 30) == 130


def test_eoq():
    # sqrt(2*5000*300/10) = 547.7
    assert math.isclose(eoq(5000, 300, 10), 547.72, abs_tol=0.01)
    assert eoq(5000, 300, 0) == 0.0


def test_status_rules():
    assert classify_status(10, 10, 5, rop=70, max_stock=200) == "CRITICAL"
    assert classify_status(60, 10, 5, rop=70, max_stock=200) == "REORDER"
    assert classify_status(100, 10, 5, rop=70, max_stock=200) == "HEALTHY"
    assert classify_status(250, 10, 5, rop=70, max_stock=200) == "OVERSTOCK"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
