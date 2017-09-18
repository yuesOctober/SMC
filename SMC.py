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
from check import readFile,readFilen,calMLEn,calMLE
import matplotlib.pyplot as plt
class MDP:
	def __init__(self,Q,alpha,beta,gamma,rho,m):
		#self.Q=np.zeros((S,A))
		self.Q=Q
		self.alpha=alpha
		self.beta=beta
		self.gamma=gamma
		self.rho=rho
		self.m=m
	def __repr__(self):
		return '<Q:%s,alpha:%s,beta:%s,gamma:%s,rho:%s,m:%s>' %(self.Q,self.alpha,self.beta,self.gamma,self.rho,self.m)


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
		# print "s,a",s,a
		# print self.Q


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

	def updateQ(self,s,a,s_next):
		#a=self.selectAct(s)
		#s_next=self.transition(s,a)
		r=self.reward_func(s,a,s_next)
		delta= self.rho * r + self.gamma * np.max(self.Q[s_next])-self.Q[s][a]
		self.Q[s][a]=self.m * self.Q[s][a]+ self.alpha * delta

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

		#probabilistic

		# prob=random.random()
		# reward=0
		# if s==0 and a==0:
		# 	if prob<=0.9:
		# 		reward=1
		# elif s==1 and a==0:
		# 	if prob<=0.7:
		# 		reward=1
		# elif s==0 and a==1:
		# 	if prob<=0.5:
		# 		reward=1
		# else:
		# 	if prob<=0.1:
		# 		reward=1
		# return reward



		#deterministic
		reward=0.0
		if a==1:
			reward=1.0
		return reward



	def reward_dist(self, s ,a ,s_next,r):

		#probabilistic

		# prob=0.0
		# if s==0 and a==0:
		# 	if r==1:
		# 		prob=0.9
		# 	else:
		# 		prob=0.1
		# elif s==1 and a==0:
		# 	if r==1:
		# 		prob=0.7
		# 	else:
		# 		prob=0.3
		# elif s==0 and a==1:
		# 	if r==1:
		# 		prob=0.5
		# 	else:
		# 		prob=0.5
		# else:
		# 	if r==1:
		# 		prob=0.1
		# 	else:
		# 		prob=0.9


		#deterministic
		prob=0.0
		if a==1 and r==1:
			prob=1.0
		elif a==0 and r==0:
			prob=1.0
		return prob


	def transition_dist(self, s ,a ,s_next):
		S,A=readFile("Exp_data_1.xlsx")
		Prob=calMLE(S,A)
		prob=0.0
		#print "Prob is:",Prob
		#print s,a,s_next
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
		elif s==1 and a==0:
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





# class SMC:

# 	def __init__(self,S,R,A):
# 		'''
# 		S is s_1 to s_t+1
# 		R is r_1 to s_t
# 		A is a_1 to a_t
# 		'''
# 		self.S=S
# 		self.R=R
# 		self.A=A 


# 	@abstractmethod
# 	def genSample(self,mean,variance):

# 		'''
# 		Generate one sample from an initial distribution
# 		Output: 

# 			a sample = mdp(Q,alpha,beta,gamma,rho,m)
# 		'''
# 		pass


# 	def calw1(self,sample):
# 		'''
# 		Calculate w1 for a single sample
# 		Args:
# 			sample: MDP class.
# 		Output:
# 			w1 for this sample

# 		'''
# 		s1=self.S[0]
# 		a1=self.A[0]
# 		s2=self.S[1]
# 		r1=self.R[0]
# 		# Q1=sample.Q
# 		# alpha1=sample.alpha
# 		# beta1=sample.beta
# 		# gamma1=sample.gamma
# 		# rho1=sample.rho
# 		# m1=sample.m
# 		Px1Q1= (to be defined)???? # P(x1,Q1)
# 		q1x1Q1=  (to be defined)??? # q1(x1,Q1)
# 		R1=sample.reward_dist(s1,a1,s2,r1)
# 		T1=sample.transition_dist(s1,a1,s2)
# 		g1=sample.action_prob(s1,a1)
# 		w1=R1*T1*g1*Px1Q1/q1x1Q1

# 		return w1

# 	def systematic_resample(self,weights,samples):
# 		'''
# 		Args: 
# 			weights=[weight1,....weightN]
# 			samples=[sample1,...sampleN]

# 		'''
# 		N=len(weights)
# 		positions=(random.random() + np.arange(N)) / N
# 		indexes = np.zeros(N, 'i')
# 		cumulative_sum = np.cumsum(weights)
# 		i, j = 0, 0
# 		while i < N:
# 			if positions[i] < cumulative_sum[j]:
# 				indexes[i] = j
# 				i += 1
#         	else:
# 				j += 1
# 		new_samples=np.zeros(N)
# 		for i  in range(N):
# 			new_samples[i]=samples[indexes[i]]
# 		return new_samples
# 		#return indexes



# 	def caleff(self,weights):
# 		N=len(weights):
# 		total=0.0
# 		for weight in weights:
# 			total += weight*weight
# 		Neff=1.0/total
# 		return Neff



# 	def calwt(self,wt_previous,t):

# 		st=self.S[t-1]
# 		at=self.A[t-1]
# 		st_next=self.S[t]
# 		rt=self.R[t-1]
# 		# Q1=sample.Q
# 		# alpha1=sample.alpha
# 		# beta1=sample.beta
# 		# gamma1=sample.gamma
# 		# rho1=sample.rho
# 		# m1=sample.m
# 		ht= (to be defined) #ht(x_t|x_t-1)
# 		qt=  (to be defined)#qt(xt|x1:t-1,Q1:t)
# 		Rt=sample.reward_dist(st,at,st_next,rt)
# 		Tt=sample.transition_dist(st,at,s_next)
# 		gt=sample.action_prob(st,at)
# 		at=Rt*Tt*gt*ht/qt
# 		wt=at*wt_previous

# 		return wt







def genSample(mean=[ [[0,0],[0,0]] ,0.3,0.2,0.6,1,1],std=[ [[1,1],[1,1]],0.1,0.1,0.1,0.1,0.1] ):

	'''
	Generate one sample from an initial distribution
	Output: 

		a sample = mdp(Q,alpha,beta,gamma,rho,m)
	'''
	assert len(mean)==len(std)
	Q_states=len(mean[0])
	Q_actions=len(mean[0][0])
	Qsamples=np.zeros([Q_states,Q_actions])
	sample=[]
	for i in range(len(mean[0])):
		for j in range(len(mean[0][0])):
			eachmean=mean[0][i][j]
			eachstd=std[0][i][j]
			Qsamples[i][j]=np.random.normal(eachmean, eachstd, 1)[0]
	sample.append(Qsamples)
	for i in range(1,len(mean)):
		logvarmean=math.log(mean[i])
		varstd=std[i]
		logvarsample=np.random.normal(logvarmean, varstd ,1)

		varsample=math.pow(math.e,logvarsample[0])
		sample.append(varsample)
	#print sample
	mdpsample=testMDP(sample[0],sample[1],sample[2],sample[3],sample[4],sample[5])
	return mdpsample

	#pass

def calw1(sample,S,A):
	'''
	Calculate w1 for a single sample
	Args:
		sample: MDP class.
	Output:
		w1 for this sample

	'''
	s1=S[0]
	a1=A[0]
	s2=S[1]
	r1=R[0]
	# Q1=sample.Q
	# alpha1=sample.alpha
	# beta1=sample.beta
	# gamma1=sample.gamma
	# rho1=sample.rho
	# m1=sample.m

	# Px1Q1= (to be defined)???? # P(x1,Q1)
	# q1x1Q1=  (to be defined)??? # q1(x1,Q1)
	Px1Q1= 1 # P(x1,Q1)
	q1x1Q1=  1 # q1(x1,Q1)
	R1=sample.reward_dist(s1,a1,s2,r1)
	#print "R1 is", R1
	T1=sample.transition_dist(s1,a1,s2)
	#print "T1 is", T1
	g1=sample.action_prob(s1,a1)
	#print "g1 is", g1
	w1=R1*T1*g1*Px1Q1/q1x1Q1
	#print "w1 is", w1
	return w1

def systematic_resample(weights,samples):
	'''
	Args: 
		weights=[weight1,....weightN]
		samples=[sample1,...sampleN]

	'''
	#print "weights:", weights
	N=len(weights)
	#print "N:",N
	positions=(random.random() + np.arange(N)) / N
	#print "positions:",positions
	indexes = np.zeros(N, 'i')
	cumulative_sum = np.cumsum(weights)
	#print "cumulative_sum:",cumulative_sum
	i, j = 0, 0
	while i < N:
		#print i,j,positions[i],cumulative_sum[j]
		if positions[i] < cumulative_sum[j]:
			indexes[i] = j
			i += 1
		else:
			j += 1
	#print "here"
	new_samples=[]
	for i  in range(N):
		new_samples.append(samples[indexes[i]])
	return new_samples
	#return indexes


#simplified
def samplex(sigma_x=[0.05,0.005,0.005,0.005,0.005],sample=None):
	'''
		self.Q=Q
		self.alpha=alpha
		self.beta=beta
		self.gamma=gamma
		self.rho=rho
		self.m=m
	'''
	samplex=[]

	prev_x=[sample.alpha,sample.beta,sample.gamma,sample.rho,sample.m]
	N= len(prev_x)
	for i in range(N):
		logvarpre=math.log(prev_x[i])
		varstd=sigma_x[i]
		logsamplex=np.random.normal(logvarpre, varstd ,1)
		samplex.append(math.pow(math.e,logsamplex[0]))
	sample.alpha=samplex[0]
	sample.beta=samplex[1]
	sample.gamma=samplex[2]
	sample.rho=samplex[3]
	sample.m=samplex[4]
	#return sample




def caleff(weights):
	N=len(weights)
	total=0.0
	for weight in weights:
		total += weight*weight

	Neff=1.0/total
	return Neff



def calwt(wt_previous,sample,t,S,A,R):

	st=S[t-1]
	at=A[t-1]
	st_next=S[t]
	rt=R[t-1]
	# Q1=sample.Q
	# alpha1=sample.alpha
	# beta1=sample.beta
	# gamma1=sample.gamma
	# rho1=sample.rho
	# m1=sample.m


	# ht= (to be defined) #ht(x_t|x_t-1)
	# qt=  (to be defined)#qt(xt|x1:t-1,Q1:t)
	ht= 1 #ht(x_t|x_t-1)
	qt=  1#qt(xt|x1:t-1,Q1:t)
	Rt=sample.reward_dist(st,at,st_next,rt)
	Tt=sample.transition_dist(st,at,st_next)
	gt=sample.action_prob(st,at)
	at=Rt*Tt*gt*ht/qt
	wt=at*wt_previous

	return wt




S,A=readFile("Exp_data_1.xlsx")
R=copy.deepcopy(A)

def SMC(Nsamples=10,c=5,T=10,S=None,A=None,R=None):
	'''
	c is the threshold cN for Neff
	Nsample is the sample numnbers 
	'''

	#initialization
	samples=[]
	weights=[]



	Qs0T=[]
	alphas0T=[]
	betas0T=[]
	gammas0T=[]
	rhos0T=[]
	ms0T=[]


	for i in range(Nsamples):
		sample=genSample()
		samples.append(sample)
		weight=calw1(sample,S,A)
		weights.append(weight)
	weights=np.array(weights)
	#print weights
	weights=weights/np.sum(weights)

	#print weights
	#resample
	Neff=caleff(weights)
	#print "samples are:",samples
	#print "Neff is:", Neff
	if Neff < c*Nsamples:
		#print "here"
		resamples=systematic_resample(weights,samples)
		#print "resamples are:",resamples


	#time 0:
	Qs=[]
	alphas=[]
	betas=[]
	gammas=[]
	rhos=[]
	ms=[]
	for i in range(Nsamples):
		Qs.append(resamples[i].Q)
		alphas.append(resamples[i].alpha)
		betas.append(resamples[i].beta)
		gammas.append(resamples[i].gamma)
		rhos.append(resamples[i].rho)
		ms.append(resamples[i].m)
	Qs0T.append(Qs)
	alphas0T.append(alphas)
	betas0T.append(betas)
	gammas0T.append(gammas)
	rhos0T.append(rhos)
	ms0T.append(ms)

	#for time 1 to T.
	for t in range(1,T):
		print "time t:" ,t
		#obtain updatedQ and xt
		for i in range(Nsamples):
			resamples[i].updateQ(S[i],A[i],S[i+1])
			samplex(sample=resamples[i])
			wt_previous=1.0/Nsamples

			# for future use
			Qs0t=[row[i] for row in Qs0T]
			alphas0t=[row[i] for row in alphas0T]
			betas0t=[row[i] for row in betas0T]
			gammas0t=[row[i] for row in gammas0T]
			rhos0t=[row[i] for row in rhos0T]
			ms0t=[row[i] for row in ms0T]

			Qs0t.append(resamples[i].Q)
			alphas0t.append(resamples[i].alpha)
			betas0t.append(resamples[i].beta)
			gammas0t.append(resamples[i].gamma)
			rhos0t.append(resamples[i].rho)
			ms0t.append(resamples[i].m)

			weights[i]=calwt(wt_previous,resamples[i],t,S,A,R)
		#normalize	
		weights=np.array(weights)
		weights=weights/np.sum(weights)

		#resample
		Neff=caleff(weights)
		if Neff < c*Nsamples:
			resamples=systematic_resample(weights,samples)
		Qs=[]
		alphas=[]
		betas=[]
		gammas=[]
		rhos=[]
		ms=[]
		for i in range(Nsamples):
			Qs.append(resamples[i].Q)
			alphas.append(resamples[i].alpha)
			betas.append(resamples[i].beta)
			gammas.append(resamples[i].gamma)
			rhos.append(resamples[i].rho)
			ms.append(resamples[i].m)
		Qs0T.append(Qs)
		alphas0T.append(alphas)
		betas0T.append(betas)
		gammas0T.append(gammas)
		rhos0T.append(rhos)
		ms0T.append(ms)

	# alphas=np.array(alphas)
	# betas=np.array(betas)
	# gammas=np.array(gammas)
	# rhos=np.array(rhos)
	# ms=np.array(ms)
	#plt.plot(alphas)
	#plt.show()

SMC(S=S,A=A,R=R)








