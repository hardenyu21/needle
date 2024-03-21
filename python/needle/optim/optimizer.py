"""Optimization module"""
import needle as ndl
import numpy as np
from collections import defaultdict

class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()
    
    def clip_grad_norm(self, max_norm=0.25):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = defaultdict(float)
        self.weight_decay = weight_decay

    def step(self):
        for w in self.params:
            if self.weight_decay > 0:
                grad = self.weight_decay * w.data + w.grad.data 
            else:
                grad = w.grad.data
            self.u[w] = self.momentum * self.u[w] + (1 - self.momentum) * grad
            w.data = w.data - self.lr * self.u[w]
        
        

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        raise NotImplementedError()


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.u = defaultdict(float)
        self.v = defaultdict(float)

    def step(self):
        self.t += 1
        for w in self.params:
            if self.weight_decay > 0:
                grad = w.grad.data + self.weight_decay * w.data
            else:
                grad = w.grad.data
            self.u[w] = self.beta1 * self.u[w] + (1 - self.beta1) * grad
            self.v[w] = self.beta2 * self.v[w] + (1 - self.beta2) * (grad ** 2)

            """bias correction"""
            unbiased_u = self.u[w] / (1 - self.beta1 ** self.t)
            unbiased_v = self.v[w] / (1 - self.beta2 ** self.t)
            w.data = w.data - self.lr * unbiased_u / (unbiased_v**0.5 + self.eps)
            




    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        raise NotImplementedError()