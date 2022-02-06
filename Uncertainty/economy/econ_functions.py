import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import cauchy

def k_model(k, i, tstep=1, horizon=None, dk=0.9):
	k = [k[0]]
	if horizon is None:
		horizon = round((2014 - 1960 - x) / tstep)
	for t in range(1,horizon):
		k.append(k[(t-1)]*(dk**(tstep)) + i[x + (t-1)*tstep] * tstep)
	return k

def residual_k_model(dk, k, i, tstep=1, horizon=None):
	residual = []
	k_dice = k_model(k, i, tstep=tstep, horizon=horizon, dk=dk)
	residual = [(el1 - el2)**2 for el1,el2 in zip(k[0::tstep],k_dice)]
	return sum(residual)

def k_model_endo(k, l, A, tstep=1, horizon=None, dk=0.9):
	k = [k[0]]
	y = []
	i = []
	optlrsav = 0.2582781457
	if horizon is None:
		horizon = round((2014 - 1960) / tstep)
	for t in range(1, horizon):
		y.append(k[t-1]**0.3 * (l[t-1]/1000)**0.7 * A[t-1])
		i.append(optlrsav * y[-1])
		k.append(k[t-1]*(dk**tstep) + tstep*i[-1])
	return k, i

def residual_k_model_endo(dk, k, l, A, tstep=1, horizon=None):
	residual = []
	k_dice, _ = k_model_endo(k, l, A, 
		tstep=tstep, horizon=horizon, dk=dk)
	residual = [(el1 - el2)**2 for el1,el2 in zip(k[0::tstep],k_dice)]
	return sum(residual)

def pop_model(l0, popasym=11500, popadj=0.134, horizon=57, tstep=1):
	l_model = [l0]
	for el in range(horizon - 1):
		l_model.append( l_model[-1] * \
			((popasym/l_model[-1])**(popadj/5*tstep)))
	return l_model

def residual_pop_model(params, l):
	popasym = params[0]
	popadj = params[1]
	l_model = pop_model(l[0], popasym=popasym, 
		popadj=popadj, horizon=len(l))
	residuals = np.asarray(l) - np.asarray(l_model)
	return sum([el**2 for el in residuals])

def tfp_model(A0, deltaa=0.005, ga0=0.079, horizon=53, tstep=1):
	tfp_model = [A0]
	ga = ga0
	count = 0
	for el in range(horizon - 1):
		# ga = ga / (1 + deltaa)**(tstep/5)
		# tfp_model.append( tfp_model[-1] * (1+ga)**(tstep/5))
		ga = ga0 * np.exp( - deltaa * count * tstep)
		tfp_model.append( tfp_model[-1] / ((1-ga)**(tstep/5)))
		count += 1
	return tfp_model

def residual_tfp_model(params, A):
	deltaa = params[0]
	ga0 = params[1]
	tfp = tfp_model(A[0], deltaa, ga0, horizon=len(A))
	residuals = np.asarray(A) - np.asarray(tfp)
	return sum([el**2 for el in residuals])

def gdp_model(k, l, A):
	gdp = []
	for el in zip(k,l,A):
		gdp.append((el[0]**0.3) * ((el[1]/1000)**0.7) * el[2])
	return gdp

# stochastic models
def tfp_model_s(A0, deltaa=0.005, tstep=1, ga0=0.079, 
	mu=0, sigma=0, horizon=53):
	tfp_model = [A0]
	ga = ga0
	count = 0
	for el in range(horizon - 1):
		# ga = ga / (1 + deltaa)**(tstep/5)
		# tfp_model.append( tfp_model[-1] * (1+ga)**(tstep/5) + \
		# 	np.random.normal(mu,sigma))
		ga = ga0 * np.exp( - deltaa * count * tstep)
		tfp_model.append( tfp_model[-1] / ((1-ga)**(tstep/5)) + \
			np.random.normal(mu,sigma))
		count += 1
	return tfp_model

def tfp_model_s_c(A0, deltaa=0.005, tstep=1, 
	ga0=0.079, loc=0, scale=0, horizon=53):
	tfp_model = [A0]
	ga = ga0
	count = 0
	for el in range(horizon - 1):
		# ga = ga / (1 + deltaa)**(tstep/5)
		# tfp_model.append( tfp_model[-1] * (1+ga)**(tstep/5) + \
		#  min( +0.14*tfp_model[-1], \
		#  	max(-0.14*tfp_model[-1], cauchy.rvs(*params))))					
		ga = ga0 * np.exp( - deltaa * count * tstep)
		tfp_model.append( tfp_model[-1] / ((1-ga)**(tstep/5)) + \
		 min( +0.15*tfp_model[-1], \
		 	max(-0.15*tfp_model[-1], cauchy.rvs(loc, scale))))		
		count += 1
	return tfp_model

def tfp_model_smul(A0, deltaa=0.005, tstep=1, 
	ga0=0.079, mu=1, sigma=0, horizon=53):
	tfp_model = [A0]
	ga = ga0
	count = 0
	for el in range(horizon - 1):
		# ga = ga / (1 + deltaa)**(tstep/5)
		# tfp_model.append( tfp_model[-1] * (1+ga)**(tstep/5) * \
		# 	np.random.normal(mu,sigma))
		ga = ga0 * np.exp( - deltaa * count * tstep)
		tfp_model.append( tfp_model[-1] / ((1-ga)**(tstep/5)) * \
			np.random.normal(mu,sigma))
		count += 1
	return tfp_model

def tfp_model_smul_c(A0, deltaa=0.006, tstep=1, 
	ga0=0.079, loc=1, scale=0, horizon=53):
	tfp_model = [A0]
	ga = ga0
	count = 0
	for el in range(horizon - 1):
		# ga = ga / (1 + deltaa)**(tstep/5)
		# tfp_model.append( tfp_model[-1] * (1+ga)**(tstep/5) + \
		#  min( +0.14*tfp_model[-1], \
		#  	max(-0.14*tfp_model[-1], cauchy.rvs(*params))))					
		ga = ga0 * np.exp( - deltaa * count * tstep)
		tfp_model.append( tfp_model[-1] / ((1-ga)**(tstep/5)) * \
			np.clip(cauchy.rvs(loc, scale), 0.86, 1.14))
		count += 1
	return tfp_model