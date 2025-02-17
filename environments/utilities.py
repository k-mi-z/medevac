class IdGenerator():
    def __init__(self):
        self.id = 0

    def next(self):
        self.id += 1
        return self.id

def normalize(x, a, b):
    return (x - a) / (b - a)