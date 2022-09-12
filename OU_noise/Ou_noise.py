import numpy as np


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=1.0, theta=0.15, dt=1e-1, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        # 因为我使用OU noise时情况特殊，所以每次使用时都会指定x_prev和mu，且只用一次，所以取消了x_prev的继承机制
        # self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)
