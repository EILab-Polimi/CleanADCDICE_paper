import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
import scipy, math
import scipy.signal
from statsmodels.graphics import tsaplots
from statsmodels.tsa.arima_model import ARMA
from statsmodels.graphics.gofplots import qqplot
from statsmodels.distributions.empirical_distribution import ECDF
from statsmodels.tsa.ar_model import AutoReg
import pandas as pd
from scipy.stats import cauchy, probplot, norm, levy_stable, kstest
## PLOT SETTINGS

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
fontsize = 11

import carbon_intensity_functions as ci_func

## read IMF data
file = pd.read_excel('./ICSD_IMF_17.xlsx', sheet_name='Data')
file['GDP_rppp'] = pd.to_numeric(file['GDP_rppp'], errors='coerce')
GDP_IMF = np.asarray([file.loc[
	file['year']==x]['GDP_rppp'].sum(skipna=True)
	 for x in range(1960,2016)]) / 1000
# READING EMISSIONS DATASET - WORLD BANK - CDIAC oak ridge lab
file = pd.read_excel('./API_EN.ATM.CO2E.KT_DS2_en_excel_v2_1562807.xls',
	sheet_name='Data', header=3)
## File is in kilotonnes CO2 
## but World entry is in MtCO2 -> convert into GtCO2
emissions = np.asarray([file.loc[
	file['Country Name']=='World'][str(x)] 
	for x in range(1960,2016)]).flatten() / 1000

# define year array
year = [x for x in range(1960,2016)]
# computing carbon intensity
sigma = np.asarray(emissions)/np.asarray(GDP_IMF) #(kgCO2/2005 constant USD)


gsigma2015 = -0.0152
dsig = -0.001
gsig = [gsigma2015 / (1+dsig)**(len(sigma)) * (1+dsig)**x for x in range(len(sigma))]
sigma_model = [sigma[0]]
sigma_model2 = [sigma[0]]
gsig_model = [gsigma2015 / (1+dsig)**(len(sigma))]
gsig_model2 = [gsigma2015 / (1+dsig)**(len(sigma))]
error = []
for el in range(len(sigma)-1):
	gsig_model.append(gsig_model[-1] * (1 + dsig))
	sigma_model.append(sigma_model[-1] * np.exp(gsig_model[-2]))
	# sigma_model2.append(sigma[el] * np.exp(gsig_model[-2]))
	error.append(sigma[el+1]/sigma_model[-1] - 1)

# error.pop(29)

fig, ax = plt.subplots(2,1)
ax[0].plot(gsig)
ax[0].plot(gsig_model)
ax[1].plot(sigma)
ax[1].plot(sigma_model)
# ax[1].plot(sigma_model2)
fig, ax = plt.subplots()
# ax[0].plot(gsig)
# ax[0].plot(gsig_model)
ax.plot(error)

for el in range(2):
	ar_model = AutoReg(error, lags=el).fit()
	print(ar_model.aic, ar_model.bic)
	ar_model.plot_diagnostics()
	print(ar_model.params)
	print(ar_model.summary())
	plt.suptitle(el)
residuals2 = ar_model.resid
stdev = np.std(residuals2)
print(stdev)

fig, ax = plt.subplots()
horizon=550
sigma_model = [0.0 for x in range(horizon)]
sigma_model[0] = sigma[0]
gsigma_model = [gsigma2015 / (1+dsig)**len(sigma) for el in range(horizon+1)]
for el in range(horizon-1):
	gsigma_model[el+1] = gsigma_model[el] * (1+dsig)
	sigma_model[el+1] = sigma_model[el] * np.exp(gsigma_model[el])

# ar_model.params[0] = 1.0
y_his = [y for y in range(1960,2016)]
y_fut = [y for y in range(1960,2510)]
series = []
for el in range(30):
	sigma_model2 = [0.0 for el in range(horizon)]
	sigma_model2[0] = sigma[0]
	gsigma_model = [gsigma2015 / (1+dsig)**len(sigma) for el in range(horizon+1)]
	noise = [np.random.randn()*stdev for x in range(horizon+1)]
	for el in range(horizon-1):
		gsigma_model[el+1] = gsigma_model[el] * (1+dsig)
		sigma_model2[el+1] = sigma_model2[el] * np.exp(gsigma_model[el]) *( 1 + \
			np.sum([ar_model.params[x]*noise[el-x] for x in range(len(ar_model.params)) if (el-x)>=0]))
	ax.plot(y_fut, sigma_model2, '--', linewidth=0.5)
	series.append(sigma_model2)
ax.plot(y_fut, sigma_model, 'r')
ax.plot(y_fut, np.mean(np.asarray(series), axis=0), 'green')
ax.plot(y_his, sigma, 'k')
ax.set_xlabel('Year')
ax.set_ylabel('Carbon intensity [kgCO2/USD(2005)]')
plt.xlim((1960, 2500))

for el in range(1):
	ar_model = AutoReg(error, lags=el).fit()
	print(ar_model.aic, ar_model.bic)
	ar_model.plot_diagnostics()
	print(ar_model.params)
	print(ar_model.summary())
	plt.suptitle(el)
residuals2 = ar_model.resid
stdev = np.std(residuals2)
print(stdev)

fig, ax = plt.subplots()
horizon=550
sigma_model = [0.0 for x in range(horizon)]
sigma_model[0] = sigma[0]
gsigma_model = [gsigma2015 / (1+dsig)**len(sigma) for el in range(horizon+1)]
for el in range(horizon-1):
	gsigma_model[el+1] = gsigma_model[el] * (1+dsig)
	sigma_model[el+1] = sigma_model[el] * np.exp(gsigma_model[el])

# stdev=0.021809064958843667
y_his = [y for y in range(1960,2016)]
y_fut = [y for y in range(1960,2510)]
series = []
for niter in range(30):
	sigma_model2 = [0.0 for el in range(horizon)]
	sigma_model2[0] = sigma[0]
	gsigma_model = [gsigma2015 / (1+dsig)**len(sigma) for el in range(horizon+1)]
	noise = [np.random.randn()*stdev for x in range(horizon+1)]
	for el in range(horizon-1):
		gsigma_model[el+1] = gsigma_model[el] * (1+dsig)
		if el > 55:
			sigma_model2[el+1] = sigma_model2[el] * np.exp(gsigma_model[el]) *( 1 + \
				noise[el])
		else:
			sigma_model2[el+1] = sigma_model2[el] * np.exp(gsigma_model[el])
	ax.plot(y_fut, sigma_model2, '--', linewidth=0.5)
	series.append(sigma_model2)
ax.plot(y_fut, sigma_model, 'r')
ax.plot(y_fut, np.mean(np.asarray(series), axis=0), 'green')
ax.plot(y_his, sigma, 'k')
ax.set_xlabel('Year')
ax.set_ylabel('Carbon intensity [kgCO2/USD(2005)]')
plt.xlim((1960, 2500))
# plt.show()

fig, ax = plt.subplots()
y_his = [y for y in range(1960,2016)]
y_fut = [y for y in range(1960,2510,5)]
series = []
for niter in range(50):
	sigma_model2 = [0.0 for el in range(0,horizon,5)]
	sigma_model2[0] = sigma[0]
	gsigma_model = [gsigma2015 / (1+dsig)**len(sigma) for el in range(horizon+1)]
	el = 0
	for ts in range(0,horizon-5,5):
		gsigma_model[el+1] = gsigma_model[el] * (1+dsig)**5
		if el > 55/5:
			sigma_model2[el+1] = sigma_model2[el] * (np.exp(gsigma_model[el]*5) *( 1 + \
				max(-3, min(3,np.random.randn()))*stdev*(5**0.5)))
		else:
			sigma_model2[el+1] = sigma_model2[el] * np.exp(gsigma_model[el]*5)
		el += 1
	ax.plot(y_fut, sigma_model2, '--', linewidth=0.5)
	series.append(sigma_model2)
ax.plot(y_fut, sigma_model[::5], 'r')
ax.plot(y_fut, np.mean(np.asarray(series), axis=0), 'green')
ax.plot(y_his, sigma, 'k')
ax.set_xlabel('Year')
ax.set_ylabel('Carbon intensity [kgCO2/USD(2005)]')
plt.xlim((1960, 2500))
plt.show()