"""A minimal abstract interpretation tool
"""

class Interval:
    def __init__(self, a, b):
        """Init Interval [a, b]
        """
        self.a = a
        self.b = b
    
    def __add__(self, interval):
        self.a += interval.a
        self.b += interval.b
        return self

    def __sub__(self, interval):
        self.a -= interval.a
        self.b -= interval.b
        return self

    def __str__(self):
        return str([self.a, self.b])

    def __repr__(self):
        return self.__str__()
    
    def area(self):
        return self.b - self.a
    
class Polyhedron:
    def __init__(self):
        pass
