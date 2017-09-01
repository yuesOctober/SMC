import time
import numpy as np
#from math import sqrt, log
import math
from abc import ABCMeta, abstractmethod
from collections import defaultdict
import random
class MDP:
	def __init__(self,Q,alpha,beta,gamma,rho,m):
		#self.Q=np.zeros((S,A))
		self.Q=Q
		self.alpha=alpha
		self.beta=beta
		self.gamma=gamma
		self.rho=rho
		self.m=m


	@abstractmethod
	def reward_func(self, s ,a ,s_next):
		"""Reward function, return reward r(s,a,s')"""
		pass

	@abstractmethod
	def reward_dist(self, s ,a ,s_next,r):
		"""Reward distribution R(r|s,a,s')"""
		pass



	@abstractmethod
	def transition_dist(self, s ,a ,s_next):
		"""Transition distribution T(s'|a,s)"""
		pass

	@abstractmethod
	def transition(self, s ,a):
		"""Transition function, return next state s'"""
		pass


	def action_prob(self,s,a):
		num=math.exp(self.beta*self.Q[s][a])
		den=0.0
		for i in range(len(self.Q[s])):
			den= den + math.exp(self.beta*self.Q[s][i])
		prob=num/den
		return prob

	def selectAct(self,s):
		prob_thresholds=[]

		prob=0.0
		for i in range(len(self.Q[s])):
			prob=prob+self.action_prob(s,i)
			prob_thresholds.append(prob)
		print prob_thresholds
		rand_prob=random.random()
		print rand_prob
		for i in range(len(prob_thresholds)):
			if rand_prob < prob_thresholds[i]:
				return i

	def updateQ(self,s):
		a=self.selectAct(s)
		s_next=self.transition(s,a)
		r=self.reward_func(s,a,s_next)
		delta= self.rho * r + self.gamma * np.max(self.Q[s])-self.Q[s][a]
		self.Q[s][a]=self.m * Q[s][a]+ self.alpha * delta

		#delta= r+self.gamma*np.max(self.Q[s])-self.Q[s][a]


# test selectAct function
'''
Q=np.array([[2,3,1],[3,2,2]])
alpha=0.1
beta=0.2
gamma=0.3
mdp=MDP(Q,alpha,beta,gamma)
print mdp.selectAct(0)
'''


class SMC:

	def __init__(self,S,R,A):
		'''
		S is s_1 to s_t+1
		R is r_1 to s_t
		A is a_1 to a_t
		'''
		self.S=S
		self.R=R
		self.A=A 


	@abstractmethod
	def genSample(self,distribution="Guassian"):

		'''
		Generate one sample from an initial distribution
		Output: 

			a sample = mdp(Q,alpha,beta,gamma,rho,m)
		'''
		pass


	def calw1(self,sample):
		'''
		Calculate w1 for a single sample
		Args:
			sample: MDP class.
		Output:
			w1 for this sample

		'''
		s1=self.S[0]
		a1=self.A[0]
		s2=self.S[1]
		r1=self.R[0]
		# Q1=sample.Q
		# alpha1=sample.alpha
		# beta1=sample.beta
		# gamma1=sample.gamma
		# rho1=sample.rho
		# m1=sample.m
		Px1Q1= (to be defined)???? # P(x1,Q1)
		q1x1Q1=  (to be defined)??? # q1(x1,Q1)
		R1=sample.reward_dist(s1,a1,s2,r1)
		T1=sample.transition_dist(s1,a1,s2)
		g1=sample.action_prob(s1,a1)
		w1=R1*T1*g1*Px1Q1/q1x1Q1

		return w1

	def systematic_resample(self,weights,samples):
		'''
		Args: 
			weights=[weight1,....weightN]
			samples=[sample1,...sampleN]

		'''
		N=len(weights)
		positions=(np.random.uniform(0,1.0/N)+ np.arange(N)) / N
		indexes = np.zeros(N, 'i')
		cumulative_sum = np.cumsum(weights)
		i, j = 0, 0
		while i < N:
			if positions[i] < cumulative_sum[j]:
				indexes[i] = j
				i += 1
        	else:
				j += 1
		new_samples=np.zeros(N)
		for i  in range(N):
			new_samples[i]=sample(indexes[i])
		return new_samples
		#return indexes



	def calwt(self,wt_previous,t):

		st=self.S[t-1]
		at=self.A[t-1]
		st_next=self.S[t]
		rt=self.R[t-1]
		# Q1=sample.Q
		# alpha1=sample.alpha
		# beta1=sample.beta
		# gamma1=sample.gamma
		# rho1=sample.rho
		# m1=sample.m
		ht= (to be defined) #ht(x_t|x_t-1)
		qt=  (to be defined)#qt(xt|x1:t-1,Q1:t)
		Rt=sample.reward_dist(st,at,st_next,rt)
		Tt=sample.transition_dist(st,at,s_next)
		gt=sample.action_prob(st,at)
		at=Rt*Tt*gt*ht/qt
		wt=at*wt_previous

		return wt
















