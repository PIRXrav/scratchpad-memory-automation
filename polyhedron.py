"""A minimal abstract interpretation tool
"""

import unittest

class Interval:
    def __init__(self, a, b):
        """Init Interval [a, b]
        """
        if int(a) != a:
            raise Exception(f'{a=} must be as integer')
        if int(b) != b:
            raise Exception(f'{b=} must be as integer')
        self.a = int(a)
        self.b = int(b)

    def __add__(self, interval):
        """
        [0, 5[ + [0, 0[ = [0, 5]
        [0, 5[ + [0, 1[ = [0, 5[
        [0, 5[ + [10, 10[ = [10, 15[
        **The Devil is in the Details**
        """
        i = Interval(self.a + interval.a, self.b + interval.b - (interval.area() != 0 and self.area() != 0))
        return i

    def __sub__(self, interval):
        return Interval(self.a - interval.a, self.b - interval.b)

    def __truediv__(self, interval):
        return Interval(min(self.a / interval.a, self.a / interval.b),
                        max(self.b / interval.a, self.b / interval.b))

    def __mul__(self, interval):
        if interval.a != interval.b:
            raise NotImplementedError("Not implemeneted TODO")
        k = interval.a
        return Interval((self.a - 1) * k + 1 if self.a != 0 else 0,
                        (self.b - 1) * k + 1 if self.b != 0 else 0)

    def __str__(self):
        return str(f"[{self.a}, {self.b}[")

    def __repr__(self):
        return self.__str__()

    def area(self):
        return self.b - self.a

    def __eq__(self, o):
        return self.a == o.a and self.b == o.b


class TestInterval(unittest.TestCase):
    def test_op_add(self):
        assert Interval(0, 5) + Interval(0, 0) == Interval(0, 5)
        assert Interval(0, 5) + Interval(0, 1) == Interval(0, 5)
        assert Interval(0, 5) + Interval(10, 10) == Interval(10, 15)

    def test_op_mul(self):
        # assert Interval(0, 5) * Interval(4, 4) == Interval(0, 20)
        assert Interval(0, 5) * Interval(4, 4) == Interval(0, 17)
        # Explanation:
        # tab[i*4]n, i \in [0, 5[ => i*4 \in [0, 16] = [0, 17[
        assert Interval(0, 256) * Interval(4, 4) == Interval(0, 1021)

    def test_mul_all(self):
        pass


class Polyhedron:
    def __init__(self):
        pass
