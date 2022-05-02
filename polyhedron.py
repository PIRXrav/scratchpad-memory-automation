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

        a = int(a)
        b = int(b)

        if b < a:
            raise Exception(f'{b=} must be greather than {a=}')

        self.a = a
        self.b = b

    def __add__(self, interval):
        """
        [0, 5] + [0, 0[ = [0, 5]
        [0, 5] + [0, 1] = [0, 6]
        [0, 5] + [10, 10] = [10, 15[
        """
        return Interval(self.a + interval.a, self.b + interval.b)

    def __sub__(self, interval):
        return Interval(self.a - interval.a, self.b - interval.b)

    def __truediv__(self, interval):
        raise NotImplementedError("TODO")
        return Interval(min(self.a / interval.a, self.a / interval.b),
                        max(self.b / interval.a, self.b / interval.b))

    def __mul__(self, interval):
        if interval.a != interval.b:
            raise NotImplementedError("Not implemeneted TODO")
        k = interval.a
        if k < 0:
            raise NotImplementedError("TODO")
        return Interval((self.a) * k, (self.b) * k)

    def __str__(self):
        return str(f"[{self.a}:{self.b}]")

    def __repr__(self):
        return self.__str__()

    def area(self):
        return self.b - self.a + 1

    def __eq__(self, o):
        return self.a == o.a and self.b == o.b


class TestInterval(unittest.TestCase):

    def test_eq(self):
        assert Interval(123, 456) == Interval(123, 456)
        assert Interval(123, 456) == Interval(122, 455) + Interval(1, 1)

    def test_op_add(self):
        assert Interval(0, 5) + Interval(0, 0) == Interval(0, 5)
        assert Interval(0, 5) + Interval(0, 1) == Interval(0, 6)
        assert Interval(0, 5) + Interval(10, 10) == Interval(10, 15)

    def test_op_mul(self):
        # assert Interval(0, 5) * Interval(4, 4) == Interval(0, 20)
        assert Interval(0, 5) * Interval(4, 4) == Interval(0, 20)
        # Explanation:
        # tab[i*4]n, i \in [0, 4] => i*4 \in [0, 16]
        assert Interval(0, 256) * Interval(4, 4) == Interval(0, 1024)

    def test_mul_all(self):
        pass

    def test_area(self):
        assert Interval(0, 63).area() == 64
        assert Interval(10, 20).area() == 11


class Polyhedron:
    def __init__(self):
        pass
