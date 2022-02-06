import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
import pandas as pd
import scipy
from scipy.stats import cauchy, norm, kstest
from statsmodels.graphics import tsaplots
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.distributions.empirical_distribution import ECDF
## SETTINGS FOR PLOTS
sns.set_style('whitegrid')

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

import econ_functions as econ_func

##### READ DATASETS #####

year_obs = [x for x in range(1960,2014)]
## Reading ICSD IMF Dataset - converting in trillion USD
file = pd.read_excel('./ICSD_IMF_17.xlsx', sheet_name='Data')
file['kgov_rppp'] = pd.to_numeric(file['kgov_rppp'], errors='coerce')
file['kpriv_rppp'] = pd.to_numeric(file['kpriv_rppp'], errors='coerce')
file['kppp_rppp'] = pd.to_numeric(file['kppp_rppp'], errors='coerce')
file['k'] = file.fillna(0)['kgov_rppp'] + \
	file.fillna(0)['kpriv_rppp'] + file.fillna(0)['kppp_rppp']
file['igov_rppp'] = pd.to_numeric(file['igov_rppp'], errors='coerce')
file['ipriv_rppp'] = pd.to_numeric(file['ipriv_rppp'], errors='coerce')
file['ippp_rppp'] = pd.to_numeric(file['ippp_rppp'], errors='coerce')
file['i'] = file.fillna(0)['igov_rppp'] + \
	file.fillna(0)['ipriv_rppp'] + file.fillna(0)['ippp_rppp']
file['GDP_rppp'] = pd.to_numeric(file['GDP_rppp'], errors='coerce')
k = np.asarray([file.loc[file['year']==x]['k'].sum(skipna=True)
	for x in year_obs]) / 1000
i = np.asarray([file.loc[file['year']==x]['i'].sum(skipna=True) 
	for x in year_obs]) / 1000
GDP = np.asarray([file.loc[
	file['year']==x]['GDP_rppp'].sum(skipna=True)
	 for x in year_obs]) / 1000
# load population dataset - World Bank - converting in million people
file = pd.read_excel('./API_SP.POP.TOTL_DS2_en_excel_v2_1622070.xls', 
	sheet_name='Data', header=3)
POP = np.asarray([file.loc[file['Country Name']=='World'][str(x)] 
	for x in year_obs]).flatten() / 10**6
# computing observed TFP
gamma = 0.3
A = GDP / ( k**gamma * (POP/1000)**(1 - gamma) )
ga = (A[1:] - A[:-1]) / A[1:]
alpha = A[1:]/A[:-1]
ga = -((1 / alpha)**(5) - 1)

plt.plot(A)
plt.plot(ga)
ga2015 = 0.076
dela = 0.005
ga_model = [ga2015 * np.exp(-dela*x) for x in range(1960-2014,-1)]
ga_model2 = [ga2015 * np.exp(-dela*x) for x in range(500)]
ga_model3 = [ga2015 * np.exp(-dela*5*x) for x in range(100)]
# A_model = [A[0]]
# for el in range(len(ga)):
# 	A_model.append(A_model[el]/(1-ga_model[el])**(1/5))
# plt.plot(A_model)
# plt.plot(ga_model)
# plt.figure()
# plt.plot([x for x in range(1960-2014,-1)], ga)
# plt.plot([x for x in range(1960-2014,-1)], ga_model)
# plt.plot([x for x in range(500)], ga_model2)
# plt.plot([x for x in range(0,500,5)], ga_model3)

error = []
error_ga = []
for el in range(len(ga_model)-1):
	ga_m = ga2015 * np.exp(-dela *(1960-2014+el) )
	error_ga.append(ga_m - ga[el])
	A_m = A[el] / (1 - ga_m)**(1/5)
	error.append(A[el+1] / A_m - 1)

# removing outlier when more countries are added to the database
error.pop(29)
error_ga.pop(29)

for el in range(1):
	ar_model = AutoReg(error, lags=el).fit()
	print(el, ar_model.test_normality())
	ar_model.plot_diagnostics()
	print(ar_model.summary())
	print(ar_model.params)
	print(np.std(ar_model.resid))
	stdev = np.std(ar_model.resid)
	plt.suptitle(el)

for el in range(1):
	ar_model = AutoReg(error_ga, lags=el).fit()
	print(el, ar_model.test_normality() )
	ar_model.plot_diagnostics()
	print(ar_model.summary())
	print(ar_model.params)
	print(np.std(ar_model.resid))
	plt.suptitle(el)

A_model = [A[0]]
for el in range(550-1):
	A_model.append(A_model[el]/(1-ga2015*np.exp(-dela*(1960-2014+el) ) )**(1/5))

plt.figure()
y_his = [y for y in range(1960,2014)]
y_fut = [y for y in range(1960,2510)]
series = []
data = []
for x in range(30):
	A_model2 = [A[0]]
	y = [1960]
	data.append([A[0], y[0]])
	for el in range(550-1):
		if el > 54:
			A_model2.append(A_model2[-1]/(1-ga2015*np.exp(-dela*(1960-2014+el) ) )**(1/5) * (1+np.random.randn()*stdev) )
		else:
			A_model2.append(A_model2[-1]/(1-ga2015*np.exp(-dela*(1960-2014+el) ) )**(1/5) )
		y.append(y[-1]+1)
		data.append([A_model2[el+1], y[el+1],x])
	plt.plot(y_fut, A_model2,'--', linewidth=0.5)
	series.append(A_model2)
data_sim = pd.DataFrame(data, columns = ['Total Factor Productivity','Year','Sim'])
plt.plot(y_fut, A_model, 'r-.', label='original model')
plt.plot(y_fut, np.mean(np.asarray(series), axis=0), label='mean of stochastic models')
plt.plot(y_his, A, 'k', label='observations')
plt.ylabel('Total Factor Productivity [-]')
plt.xlabel('Year')
plt.xlim((1960,2500))

plt.figure()
y_his = [y for y in range(1960,2014)]
y_fut = [y for y in range(1960,2510,5)]
series = []
data = []
for x in range(50):
	A_model2 = [A[0]]
	y = [1960]
	data.append([A[0], y[0]])
	el = 0
	for ts in range(0,550-5,5):
		if el > 54/5:
			A_model2.append(A_model2[-1]/(1-ga2015*np.exp(-dela*(1960-2014+el*5) ) )**(5/5) * 
				(1+max(-3, min(3,np.random.randn()))*stdev*(5**0.5)) )
		else:
			A_model2.append(A_model2[-1]/(1-ga2015*np.exp(-dela*(1960-2014+el*5) ) )**(5/5) )
		y.append(y[-1]+1)
		data.append([A_model2[el+1], y[el+1],x])
		el += 1
	plt.plot(y_fut, A_model2,'--', linewidth=0.5)
	series.append(A_model2)
data_sim = pd.DataFrame(data, columns = ['Total Factor Productivity','Year','Sim'])
plt.plot(y_fut, A_model[::5], 'r-.', label='original model')
plt.plot(y_fut, np.mean(np.asarray(series), axis=0), label='mean of stochastic models')
plt.plot(y_his, A, 'k', label='observations')
plt.ylabel('Total Factor Productivity [-]')
plt.xlabel('Year')
plt.xlim((1960,2500))

plt.show()
