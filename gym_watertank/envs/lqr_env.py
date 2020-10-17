import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class LQREnv(gym.Env):
	metadata = {'render.modes': ['human']}

	def __init__(self, A=np.array([[0.9,0.],[0.1,0.9]]),
					   B=np.array([[1],[0.]]),
					   Z1=np.array([[0,0],[0,1]]),
					   Z2=0.1,
					   noise_cov=np.array([[0.01,0],[0,0.01]]),
					   seed=None):
		'''
		A,B are some Matrices here
			if x is mx1 then A is mxm, B is mxn then the u should be nx1
		Z1,Z2 are some positive semi-definite weight matrices mxm or scala
			that determines the trade-off between keeping state small and keeping action small.
		x = Ax + Bu
		state: 1xn nparray x.T
		state = state@A + action@B.T
		'''
		super(LQREnv, self).__init__()
		self.m = A.shape[0]
		assert B.shape[0] == A.shape[1]
		self.n = B.shape[1]
		if not np.isscalar(Z1):
			Z1.shape[0] == A.shape[0]
		if not np.isscalar(Z2):
			Z2.shape == A.shape
		else:
			self.n == 1
		self.A = A
		self.B = B
		self.Z1 = Z1
		self.Z2 = Z2
		self.state = None
		self.noise_mu = np.zeros(self.m)
		self.noise_cov = noise_cov

		if self.n==1:
			self.min_action = -np.inf
			self.max_action = np.inf
			self.action_space = spaces.Box(
									low=self.min_action,
									high=self.max_action,
									shape=(1,),
									dtype=np.float32)
		else:
			self.min_action = -np.inf
			self.max_action = np.inf
			self.low_action = np.ones(n)*self.min_action
			self.high_action = np.ones(n)*self.max_action
			self.action_space = spaces.Box(
									low=self.low_action,
									high=self.high_action,
									dtype=np.float32)
		self.low_state = -np.ones(self.m)*np.inf
		self.high_state = np.ones(self.m)*np.inf
		self.observation_space = spaces.Box(
									low=self.low_state,
									high=self.high_state,
									dtype=np.float32)
		self.seed(seed)
		self.reset()

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def calcu_reward(self, u):
		x = self.state
		if self.n==1:
			cost = x@self.Z1@x.T + u*self.Z2*u
		else:
			cost = x@self.Z1@x.T + u.T@self.Z2@u
		return -float(cost)

	def step(self, action):
		reward = self.calcu_reward(action)
		noise = self.np_random.multivariate_normal(self.noise_mu, self.noise_cov)
		if self.n==1:
			self.state = self.state@self.A.T + action*self.B.T.flatten() + noise
		else:
			self.state = self.state@self.A.T + action@self.B.T + noise
		done = False
		return self.state, reward, done, {}

	def reset(self):
		# self.state: 1xm numpy array
		self.state = self.np_random.uniform(low=-0.5, high=0.5, size=self.m)
		return np.array(self.state)

	def render(self, mode='human'):
		pass

	def close(self):
		self.state = None