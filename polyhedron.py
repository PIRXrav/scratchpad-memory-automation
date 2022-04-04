"""A minimal abstract interpretation tool
"""

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
        return Interval(self.a + interval.a, self.b + interval.b)


    def __sub__(self, interval):
        return Interval(self.a - interval.a, self.b - interval.b)

    def __truediv__(self, interval):
        return Interval(min(self.a / interval.a, self.a / interval.b),
                        max(self.b / interval.a, self.b / interval.b))
        
    def __str__(self):
        return str(f"[{self.a}, {self.b}[")

    def __repr__(self):
        return self.__str__()
    
    def area(self):
        return self.b - self.a
    
class Polyhedron:
    def __init__(self):
        pass
