import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import scipy.linalg

class WaterTankLQREnv(gym.Env):
	metadata = {'render.modes': ['human']}
	def __init__(self, A=np.array([[0.98, 0], [0.02, 0.98]]),
					   B=np.array([[0.1],[0]]),
					   r = 9.,
					   Z1 = np.array([[0,0],[0,1]]),
					   Z2 = 0.1,
					   x_max = np.array([10,10]),
					   gamma=0.99,
					   noise_cov = np.eye(2)*0.01,
					   seed=None,
					   overflow_cost = -40):
		'''
		Z1 is the because the the second tank is the only goal
			for the first tank, the cost is whether overflow
		'''
		super(WaterTankLQREnv, self).__init__()
		self.m = A.shape[0]
		assert B.shape[0] == A.shape[1]
		self.n = B.shape[1]
		if not np.isscalar(Z2):
			Z2.shape == A.shape
		else:
			self.n == 1
		self.A = A
		self.B = B
		self.r = r
		self.Z1 = Z1
		self.Z2 = Z2
		self.x_max = x_max

		self.noise_mu = np.zeros(self.m)
		self.noise_cov = noise_cov

		if self.n==1:
			self.min_action = -15.
			self.max_action = 15.
			self.action_space = spaces.Box(
									low=self.min_action,
									high=self.max_action,
									shape=(1,),
									dtype=np.float32)
		else:
			self.min_action = -15.
			self.max_action = 15.
			self.low_action = np.ones(self.n)*self.min_action
			self.high_action = np.ones(self.n)*self.max_action
			self.action_space = spaces.Box(
									low=self.low_action,
									high=self.high_action,
									dtype=np.float32)
		self.low_state = np.zeros(self.m)
		self.high_state = x_max
		self.observation_space = spaces.Box(
									low=self.low_state,
									high=self.high_state,
									dtype=np.float32)
		self._max_episode_steps = 1000
		self._episode_steps = 0
		self.overflow_cost = overflow_cost

		self.seed(seed)
		self.linear_KL(gamma)
		


	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]


	def calcu_reward(self,u):
		x = self.state
		cost = (x-self.r)@self.Z1@(x-self.r)
		if self.n==1:
			cost+= u*self.Z2*u
		else:
			cost += u@self.Z2@u.T
		return -float(cost)

	def add_overflow_cost(self, x):
		additional_cost = 0
		if x[0].item() > self.observation_space.high[0]:
			additional_cost +=self.overflow_cost
		if x[1].item() > self.observation_space.high[1]:
			additional_cost +=self.overflow_cost 
		return additional_cost

	def step(self, action_tilde):
		self._episode_steps += 1
		reward = self.calcu_reward(action_tilde)
		action_tilde = np.clip(action_tilde, self.action_space.low, self.action_space.high)
		noise = self.np_random.multivariate_normal(self.noise_mu, self.noise_cov)
		action = action_tilde + self.get_lqr_action(self.state)
		if self.n==1:
			action = np.clip(action, 0, self.action_space.high)
		else:
			action = np.clip(action, np.zeros(self.n), self.action_space.high)
		if self.n==1:
			new_state = self.state@self.A.T + action*self.B.T.flatten() + noise
		else:
			new_state = self.state@self.A.T + action@self.B.T + noise
		reward += self.add_overflow_cost(new_state)
		self.state = np.clip(new_state,self.observation_space.low,self.observation_space.high)
		if self._episode_steps < self._max_episode_steps:
			done = False
		else:
			done = True
		return self.state, reward, done, {}


	def step_eval(self, action_tilde):
		self._episode_steps += 1
		# ----------------------------------------------------
		cost = (self.state[1] - self.r)**2
		# ----------------------------------------------------
		action_tilde = np.clip(action_tilde, self.action_space.low, self.action_space.high)
		noise = self.np_random.multivariate_normal(self.noise_mu, self.noise_cov)
		action = action_tilde + self.get_lqr_action(self.state)
		if self.n==1:
			action = np.clip(action, 0, self.action_space.high)
		else:
			action = np.clip(action, np.zeros(self.n), self.action_space.high)
		if self.n==1:
			new_state = self.state@self.A.T + action*self.B.T.flatten() + noise
		else:
			new_state = self.state@self.A.T + action@self.B.T + noise
		self.state = np.clip(new_state,self.observation_space.low,self.observation_space.high)
		# -----------------------------------------------------------------
		reward = self.add_overflow_cost(new_state) - float(cost)
		# -----------------------------------------------------------------
		if self._episode_steps < self._max_episode_steps:
			done = False
		else:
			done = True
		return self.state, reward, done, {}

	def reset(self):
		self._episode_steps = 0
		random_state = self.np_random.normal(0,0.1, size=self.m)
		self.state = np.clip(random_state, self.observation_space.low, self.observation_space.high)
		return np.array(self.state)

	def render(self, mode='human'):
		pass

	def close(self):
		self.state = None

	def linear_policy_K(self, gamma):
		Z1 = self.Z1
		Z2 = self.Z2
		A = self.A
		B = self.B

		a = np.sqrt(gamma)*A
		b = np.sqrt(gamma)*B
		r = Z2
		q = Z1
		P = scipy.linalg.solve_discrete_are(a,b,q,r)

		K = gamma* (1/(gamma*B.T@P@B+Z2))@B.T@P@A
		return K


	def linear_KL(self, gamma):
		K = self.linear_policy_K(gamma)
		# (I - A + BK)^-1 B L = I
		if self.m==1:
			aans = scipy.linalg.inv(np.eye(self.m) - self.A + self.B@K)@self.B
			L = np.linalg.solve(aans, np.eye(1))
		if self.m==2:
			aans = np.array([0,1])@scipy.linalg.inv(np.eye(self.m) - self.A + self.B@K)@self.B
			L = 1./aans
		self.K = K
		self.L = L
		return K, L

	def get_lqr_action(self, state):
		return  - state@self.K.T + self.L*self.r

# no action punishment
class WaterTankLQREnv1(WaterTankLQREnv):
	metadata = {'render.modes': ['human']}
	def __init__(self, A=np.array([[0.98, 0], [0.02, 0.98]]),
					   B=np.array([[0.1],[0]]),
					   r = 9.,
					   Z1 = np.array([[0,0],[0,1]]),
					   Z2 = 0.1,
					   x_max = np.array([10,10]),
					   gamma=0.99,
					   noise_cov = np.eye(2)*0.01,
					   seed=None,
					   overflow_cost = -40):
		'''
		Z1 is the because the the second tank is the only goal
			for the first tank, the cost is whether overflow
		'''
		super(WaterTankLQREnv1, self).__init__()

	def calcu_reward(self,u):
		# no action punishment
		x = self.state
		cost = (x-self.r)@self.Z1@(x-self.r)
		return -float(cost)

