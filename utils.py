class Return:
    def __init__(self):
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, value):
        self.sum += value.value
        self.count += 1
        self.avg = self.sum / self.count

    def __call__(self):
        if self.count:
            return self.avg
        else:
            return .0
