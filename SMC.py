import time
import numpy as np
#from math import sqrt, log
import math
from abc import ABCMeta, abstractmethod
from collections import defaultdict
import random
import csv
import xlrd
import re
import copy
from check import readFile
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
		'''
		Calculate the probability of taking action a in state s
		Args:
			s: state
			a: action
		Ouput:
			Prob: probability of taking a in state s
		'''
		num=math.exp(self.beta*self.Q[s][a])
		den=0.0
		for i in range(len(self.Q[s])):
			den= den + math.exp(self.beta*self.Q[s][i])
		prob=num/den
		return prob

	def selectAct(self,s):
		'''
		Select an action in state s
		Args:
			s: state
		Output:
		 	action
		'''
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
		delta= self.rho * r + self.gamma * np.max(self.Q[s_next])-self.Q[s][a]
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




#Given history of state sequence S and action sequence A, calculate the transition probability using MLE
#Here only binary S and A are considered.
def calMLE(S,A):
	tols0a0s0=0.0
	tols0a0s1=0.0
	tols0a1s0=0.0
	tols0a1s1=0.0
	tols1a0s0=0.0
	tols1a0s1=0.0
	tols1a1s0=0.0
	tols1a1s1=0.0
	total=len(S)
	for i in range(len(S)-1):
		if S[i]==0 and a[i]==0 and S[i+1]==0:
			tols0a0s0 += 1
		elif S[i]==0 and a[i]==0 and S[i+1]==1:
			tols0a0s1 += 1
		elif S[i]==0 and a[i]==1 and S[i+1]==0:
			tols0a1s0 += 1
		elif S[i]==0 and a[i]==1 and S[i+1]==1:
			tols0a1s1 += 1
		elif S[i]==1 and a[i]==0 and S[i+1]==0:
			tols1a0s0 += 1
		elif S[i]==1 and a[i]==0 and S[i+1]==1:
			tols1a0s1 += 1
		elif S[i]==1 and a[i]==1 and S[i+1]==0:
			tols1a1s0 += 1
		else:
			tols1a1s1 += 1
	probs0_s0a0=tols0a0s0/(tols0a0s0+tols0a0s1)
	probs1_s0a0=tols0a0s1/(tols0a0s0+tols0a0s1)
	probs0_s0a1=tols0a0s0/(tols0a1s0+tols0a1s1)
	probs1_s0a1=tols0a0s1/(tols0a1s0+tols0a1s1)
	probs0_s1a0=tols1a0s0/(tols1a0s0+tols1a0s1)
	probs1_s1a0=tols1a0s1/(tols1a0s0+tols1a0s1)
	probs0_s1a1=tols1a1s0/(tols1a1s0+tols1a1s1)
	probs1_s1a1=tols1a1s1/(tols1a1s0+tols1a1s1)
	Prob=np.arrary([probs0_s0a0,probs1_s0a0,probs0_s0a1,probs1_s0a1,probs0_s1a0,probs1_s1a0,probs0_s1a1,probs1_s1a1])
	return Prob


'''
s=0, restricted env=0,s=1 restricted env=1
a=0 , no drug, a=1 ,drug
reward func: if s=0,a=0,Pr(r=1)=0.9
if s=1,a=0,Pr(r=1)=0.7
if s=0,a=1, Pr(r=1)=0.5
if s=1,a=1,Pr(r=1)=0.1
'''
class testMDP(MDP):
	def reward_func(self, s ,a ,s_next):
		prob=random.random()
		reward=0
		if s==0 and a==0:
			if prob<=0.9:
				reward=1
		elif s==1 and a==0:
			if prob<=0.7:
				reward=1
		elif s==0 and a==1:
			if prob<=0.5:
				reward=1
		else:
			if prob<=0.1:
				reward=1
	def reward_dist(self, s ,a ,s_next,r):
		prob=0.0
		if s==0 and a==0:
			if r==1:
				prob=0.9
			else:
				prob=0.1
		elif s==1 and a==0:
			if r==1:
				prob=0.7
			else:
				prob=0.3
		elif s==0 and a==1:
			if r==1:
				prob=0.5
			else:
				prob=0.5
		else:
			if r==1:
				prob=0.1
			else:
				prob=0.9
		return prob

	def transition_dist(self, s ,a ,s_next):
		S,A=readFile("Exp_data_1.xlsx")
		Prob=calMLE(S,A)
		prob=0.0
		if s==0 and a==0:
			if s_next==0:
				prob=Prob[0]
			else:
				prob=Prob[1]
		elif s==0 and a==1:
			if s_next==0:
				prob=Prob[2]
			else:
				prob=Prob[3]
		elif s==1 and a==1:
			if s_next==0:
				prob=Prob[4]
			else:
				prob=Prob[5]
		elif s==1 and a==1:
			if s_next==0:
				prob=Prob[6]
			else:
				prob=Prob[7]
		return prob						

	def transition(self, s ,a):
		S,A=readFile("Exp_data_1.xlsx")
		Prob=calMLE(S,A)
		prob=random.random()
		if s==0 and a==0:
			if prob<=Prob[0]:
				s_next=0
			else:
				s_next=1
		elif s==0 and a==1:
			if prob<=Prob[2]:
				s_next=0
			else:
				s_next=1
		if s==1 and a==0:
			if prob<=Prob[4]:
				s_next=0
			else:
				s_next=1
		if s==1 and a==1:
			if prob<=Prob[6]:
				s_next=0
			else:
				s_next=1
		return s_next





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
		positions=(random.random() + np.arange(N)) / N
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
			new_samples[i]=samples[indexes[i]]
		return new_samples
		#return indexes



	def caleff(self,weights):
		N=len(weights):
		total=0.0
		for weight in weights:
			total += weight*weight
		Neff=1.0/total
		return Neff



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
















