import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy.special import erf
import scipy
import pandas as pd
import seaborn as sns
from functools import reduce
import os, matplotlib, subprocess, time, chart_studio, math
import matplotlib.font_manager as font_manager
import numpy as np

sns.set_style('ticks')
sns.set_context('paper')
import matplotlib.font_manager as font_manager
fontpath = '/Users/angelocarlino/Library/Fonts/OpenSans-Regular.ttf'
fontpath = '/System/Library/Fonts/Helvetica.ttc'
prop = font_manager.FontProperties(fname=fontpath, size='large')
prop.set_size(12)
matplotlib.rcParams['font.family'] = prop.get_name()
matplotlib.rcParams['font.size'] = prop.get_size()
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = prop.get_name()
matplotlib.rcParams['mathtext.sf'] = prop.get_name()
matplotlib.rcParams['mathtext.it'] = prop.get_name()
matplotlib.rcParams['mathtext.bf'] = prop.get_name()
matplotlib.rcParams['mathtext.tt'] = prop.get_name()
matplotlib.rcParams['mathtext.cal'] = prop.get_name()

# ECS 3, 2.5-4, 2-5
# TCR 1.8, 1.4-2.2, 1.2-2.4

def eqns(vars, *data ):
	median, bottom_likely, upper_likely, bottom_vlikely, upper_vlikely = data
	# mu = np.log(median)
	mu, sigma = vars
	# sigma = max(1e-6, sigma)
	# mu, sigma = vars
	# sigma = vars
	eqs = []
	# eqs.append(np.log(median) - mu)
	# mu = np.log(data[0])
	eqs.append( 0.17 - 1/2 * (1 + erf((np.log(bottom_likely) - mu) / (sigma*(2**0.5)) ) ) )
	eqs.append( 0.83 - 1/2 * (1 + erf((np.log(upper_likely) - mu) / (sigma*(2**0.5)) ) ) )
	eqs.append(0.05 - 1/2 * (1 + erf((np.log(bottom_vlikely) - mu) / (sigma*(2**0.5)) ) ) )
	eqs.append(0.95 - 1/2 * (1 + erf((np.log(upper_vlikely) - mu) / (sigma*(2**0.5)) ) ) )
	# return [eq1,eq2,eq3,eq4]
	return np.sum(np.asarray(eqs)**2)

fig, ax = plt.subplots(2,1, sharex=True, sharey=True)

## ECS
data = (3.0, 2.5, 4.0, 2.0, 5.0)
guess = (1, 0.3)
res = scipy.optimize.minimize(eqns, x0=guess, args=data)
print(res.x)

mu, sigma = np.log(data[0]), res.x[1]
mu_ecs, Sigma_ecs = np.log(data[0]), res.x[1]
mu, sigma = res.x
mu_ecs, Sigma_ecs = res.x
# res = root(eqns, guess, args = data, method='lm', options={'xtol':1e-12})
# # mu = np.log(data[0])
# mu, sigma = res.x
# # sigma = res.x
# print(mu, sigma)

s = np.ndarray([])
sl = np.ndarray([])
for n in range(10000):
	s = np.append(s, np.random.normal(loc=mu, scale=sigma))


x = np.linspace(0.1, 8, 100)
pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi)))

ax[0].hist(np.exp(s), density=True)
ax[0].plot(x, pdf)
ax[0].plot([2.5,4],[max(pdf)+0.2,max(pdf)+0.2],'r')
ax[0].plot([np.percentile(np.exp(s),17),np.percentile(np.exp(s),83)],[max(pdf)+0.3,max(pdf)+0.3],'r--')
ax[0].plot([2,5],[max(pdf)+0.4,max(pdf)+0.4],'k')
ax[0].plot(np.exp(mu),max(pdf)+0.6,'o')
ax[0].plot([np.percentile(np.exp(s),5),np.percentile(np.exp(s),95)],[max(pdf)+0.5,max(pdf)+0.5],'k--')
# ax[0].set_xlabel('[°C]')
ax[0].set_ylabel('ECS Probability density')

## TCR
data = (1.8, 1.4, 2.2, 1.2, 2.4)
guess = (0.5, 0.2)
res = scipy.optimize.minimize(eqns, x0 = guess, args = data)
print(res.x)
# # mu = np.log(data[0])
# mu, sigma = res.x
# # sigma = res.x
# print(mu, sigma)
mu, sigma = np.log(data[0]), res.x[1]
mu_tcr, Sigma_tcr = np.log(data[0]), res.x[1]
mu, sigma = res.x
mu_tcr, Sigma_tcr = res.x

s = np.ndarray([])
sl = np.ndarray([])
for n in range(10000):
	s = np.append(s, np.random.normal(loc=mu, scale=sigma))


x = np.linspace(0.1, 8, 50)
pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi)))

ax[1].hist(np.exp(s), density=True)
ax[1].plot(x, pdf)
ax[1].plot([1.4,2.2],[max(pdf)+0.2,max(pdf)+0.2],'r')
ax[1].plot([np.percentile(np.exp(s),17),np.percentile(np.exp(s),83)],[max(pdf)+0.3,max(pdf)+0.3],'r--')
ax[1].plot([1.2,2.4],[max(pdf)+0.4,max(pdf)+0.4],'k')
ax[1].plot(np.exp(mu),max(pdf)+0.6,'o')
ax[1].plot([np.percentile(np.exp(s),5),np.percentile(np.exp(s),95)],[max(pdf)+0.5,max(pdf)+0.5],'k--')
ax[1].set_xlabel('[°C]')
ax[1].set_ylabel('TCR Probability density')
ax[1].set_xlim((0,8))

mu = np.array([mu_tcr, mu_ecs])
Sigma = np.array([Sigma_tcr, Sigma_ecs])
mu = np.exp(mu + (Sigma**2)/2)
Sigma = (np.exp(Sigma**2) - 1) * np.exp(2*mu + Sigma**2)
corr = np.array([[1., 0.81739462], [0.81739462, 1.]])

alpha = [mu_tcr, mu_ecs]
# print("alpha = ", alpha)
beta = np.diag([Sigma_tcr**2, Sigma_ecs**2])
# print("beta = ", beta)
delta = reduce(np.matmul, [np.sqrt(beta), corr, np.sqrt(beta)])
# print(delta)
delta_eigenvalues, delta_eigenvectors = np.linalg.eig(delta)
# print("delta eigval & eigvec = ", delta_eigenvalues, delta_eigenvectors)
root_delta = reduce(np.matmul, 
	[delta_eigenvectors, np.diag(np.sqrt(delta_eigenvalues)), delta_eigenvectors.T])
print("alpha = ", alpha)
print("root_delta = ", root_delta)

s_size = 10000
tcrs = np.empty(s_size)
ecss = np.empty(s_size)
categ = []
for n in range(s_size):
	rand0 = max(-3,min(3,np.random.randn()))
	rand1 = max(-3,min(3,np.random.randn()))
	while np.exp(0.57913745 + 0.17632248*rand0 + 0.10108759*rand1) > np.exp(1.14131177 + 0.10108759*rand0 + 0.21679573*rand1):
		rand0 = max(-3,min(3,np.random.randn()))
		rand1 = max(-3,min(3,np.random.randn()))
	tcrs[n] = np.exp(0.57913745 + 0.17632248*rand0 + 0.10108759*rand1)
	ecss[n] = np.exp(1.14131177 + 0.10108759*rand0 + 0.21679573*rand1)
	categ.append("FAIR")
data = pd.DataFrame(list(zip(tcrs, ecss, categ)), columns=['TCR','ECS','TYPE'])

# sns.displot(data, x='TCR',y='ECS')
# plt.title('FAIR standard TCR ECS sampling')
# print('here')
tcrs = np.empty(s_size)
ecss = np.empty(s_size)
categ = []
for n in range(s_size):
	rand0 = max(-3,min(3,np.random.randn()))
	rand1 = max(-3,min(3,np.random.randn()))
	while (np.exp(alpha[0] + root_delta[0][0]*rand0 + root_delta[0][1]*rand1) > 
		np.exp(alpha[1] + root_delta[1][0]*rand0 + root_delta[1][1]*rand1)):
		rand0 = max(-3,min(3,np.random.randn()))
		rand1 = max(-3,min(3,np.random.randn()))
	tcrs[n] = np.exp(alpha[0] + root_delta[0][0]*rand0 + root_delta[0][1]*rand1)
	ecss[n] = np.exp(alpha[1] + root_delta[1][0]*rand0 + root_delta[1][1]*rand1)
	categ.append("AR6")
data_ = pd.DataFrame(list(zip(tcrs, ecss, categ)), columns=['TCR [°C]','ECS [°C]','TYPE'])
data = data.append(data_, ignore_index=True)
# print(data)
data.to_csv('./TCRECS.csv')
data = data.loc[data['TYPE']=='AR6']
sns.jointplot(data=data, x='TCR [°C]', y='ECS [°C]', kind='hex')
# plt.title('AR6 based TCR ECS sampling')
# fig, ax = plt.subplots(2,1, sharex=True, sharey=True)

# ## ECS
# mu, sigma = 1.14131177, 0.21679573
# s = np.ndarray([])
# sl = np.ndarray([])
# for n in range(100000):
# 	s = np.append(s, np.random.normal(loc=mu, scale=sigma))


# x = np.linspace(0.1, 10, 100)
# pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi)))

# ax[0].hist(np.exp(s), 100, density=True)
# ax[0].plot(x, pdf)
# ax[0].plot([2.5,4],[max(pdf)+0.2,max(pdf)+0.2],'r')
# ax[0].plot([np.percentile(np.exp(s),17),np.percentile(np.exp(s),83)],[max(pdf)+0.3,max(pdf)+0.3],'r--')
# ax[0].plot([2,5],[max(pdf)+0.4,max(pdf)+0.4],'k')
# ax[0].plot(np.exp(mu),max(pdf)+0.6,'o')
# print(np.exp(mu))
# ax[0].plot([np.percentile(np.exp(s),5),np.percentile(np.exp(s),95)],[max(pdf)+0.5,max(pdf)+0.5],'k--')
# ax[0].set_xlabel('ECS [°C]')

# ## TCR
# mu , sigma = 0.57913745, 0.17632248
# # sigma = res.x
# print(mu, sigma)

# s = np.ndarray([])
# sl = np.ndarray([])
# for n in range(100000):
# 	s = np.append(s, np.random.normal(loc=mu, scale=sigma))


# x = np.linspace(0.1, 5, 50)
# pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi)))

# ax[1].hist(np.exp(s), 100, density=True)
# ax[1].plot(x, pdf)
# ax[1].plot([1.4,2.2],[max(pdf)+0.2,max(pdf)+0.2],'r')
# ax[1].plot([np.percentile(np.exp(s),17),np.percentile(np.exp(s),83)],[max(pdf)+0.3,max(pdf)+0.3],'r--')
# ax[1].plot([1.2,2.4],[max(pdf)+0.4,max(pdf)+0.4],'k')
# ax[1].plot(np.exp(mu),max(pdf)+0.6,'o')
# print(np.exp(mu))
# ax[1].plot([np.percentile(np.exp(s),5),np.percentile(np.exp(s),95)],[max(pdf)+0.5,max(pdf)+0.5],'k--')
# ax[1].set_xlabel('TCR [°C]')

plt.show()