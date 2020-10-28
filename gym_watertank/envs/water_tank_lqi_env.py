import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import scipy.linalg

class WaterTankLQIEnv(gym.Env):
	"""docstring for WaterTankLQIEnv"""
	metadata = {'render.modes': ['human']}
	def __init__(self, A=np.array([[0.98, 0], 
									[0.02, 0.98]]),
					   B=np.array([[0.1],[0]]),
					   r = 9.,
					   Z1 = np.array([[0,0],[0,0]]),
					   Z2 = 0.1,
					   x_max = np.array([10,10]),
					   gamma=0.99,
					   noise_cov = np.eye(2)*0.01,
					   seed=None,
					   overflow_cost = -40):
		super(WaterTankLQIEnv, self).__init__()
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
		# note: observation sapce is [x,z] dim: (m+1)by 1
		self.low_state = np.zeros(self.m+1)
		self.low_state[-1] = -np.inf
		self.high_state = np.append(x_max, np.inf)
		self.observation_space = spaces.Box(
									low=self.low_state,
									high=self.high_state,
									dtype=np.float32)
		self._max_episode_steps = 1000
		self._episode_steps = 0
		self.overflow_cost = overflow_cost

		self.seed(seed)
		self.linear_policy_K(gamma)

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]


	def calcu_reward(self,u):
		cost = self.interval**2
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
		action_tilde = np.clip(action_tilde, self.action_space.low, self.action_space.high)
		noise = self.np_random.multivariate_normal(self.noise_mu, self.noise_cov)
		# noise = 0.
		action = action_tilde + self.get_lqr_action(self._get_observe())
		# clip action
		if self.n==1:
			action = np.clip(action, 0, self.action_space.high)
		else:
			action = np.clip(action, np.zeros(self.n), self.action_space.high)
		# state transition
		if self.n==1:
			new_state = self.state@self.A.T + action*self.B.T.flatten() + noise
		else:
			new_state = self.state@self.A.T + action@self.B.T + noise
		
		self.state = np.clip(new_state, self.observation_space.low[:self.m], self.observation_space.high[:self.m])
		self.interval  += self.r - self.state[1]

		reward = self.calcu_reward(action_tilde) + self.add_overflow_cost(new_state)

		if self._episode_steps < self._max_episode_steps:
			done = False
		else:
			done = True
		return self._get_observe(), reward, done, {}

	def _get_observe(self):
		return np.append(self.state, self.interval)

	def reset(self):
		self._episode_steps = 0
		self.state = self.np_random.normal(0,1, size=self.m)
		self.interval = 0.
		return self._get_observe()

	def render(self, mode='human'):
		pass

	def close(self):
		self.state = None

	def linear_policy_K(self, gamma):
		Atilde = np.zeros((self.m+1, self.m+1))
		Atilde[0:self.m,0:self.m] = self.A
		Atilde[-1][-1] = 1
		Atilde[self.m, -2] = -1
		# print("Atilde:\n",Atilde)

		Btilde = np.zeros((self.m+1,1))
		Btilde[0:2,:] = self.B
		# print("Btilde:\n", Btilde)

		Z1tilde  = np.zeros((self.m+1, self.m+1))
		Z1tilde[0:self.m,0:self.m] = self.Z1
		Z1tilde[-1,-1] = 1
		# print("Z1tilde:\n", Z1tilde)

		Z2tilde = self.Z2
		# print("Z2tilde:\n", Z2tilde)

		a = np.sqrt(gamma)*Atilde
		b = np.sqrt(gamma)*Btilde
		q = Z1tilde
		r = Z2tilde
		P = scipy.linalg.solve_discrete_are(a,b,q,r)

		K = gamma* (1/(gamma*Btilde.T@P@Btilde+Z2tilde))@Btilde.T@P@Atilde
		self.K = K
		return K


	def get_lqr_action(self, observation):
		return  - observation@self.K.T



		