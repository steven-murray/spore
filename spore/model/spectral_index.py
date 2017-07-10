import numpy as np

class UniversalDist(object):
    def __init__(self, gamma):
        self.gamma = gamma

    def sample(self,n):
        return self.gamma * np.ones(n)


class NormalDist(object):
    def __init__(self,gamma,sigma):
        self.gamma = gamma
        self.sigma = sigma

    def sample(self,n):
        return np.random.normal(self.gamma,self.sigma,size=n)