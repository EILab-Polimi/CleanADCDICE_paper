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
import pandas as pd
from scipy.stats import cauchy, probplot, norm, levy_stable, kstest
## PLOT SETTINGS

# sns.set_style('whitegrid')
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
sig = np.asarray(emissions)/np.asarray(GDP_IMF) #(kgCO2/2005 constant USD)

gsig0=(-0.0152)/((1-0.001)**55)
sigma_model, gsig = ci_func.sigma_model(sig[0], gsig0=gsig0, horizon=56)
## plot model in the past
# plt.figure()
# plt.plot(year, sig, label='obs')
# plt.plot(year, sigma_model, label='model')
# plt.xlabel('Years')
# plt.ylabel('Carbon intensity of global economy [kgCO2/USD(2005)]')

# ## Consider additive noise
# print('\n\n\t\t#### ADDITIVE NOISE CASE ####')
# print('\t\t#### residuals ####')
# residuals = (sig - sigma_model)
# title = 'Additive disturbance case'

# # ## normality test on residuals
# # ## residuals are not normal
# f = qqplot(np.asarray(residuals), line='s')
# f.suptitle(title)
# result = scipy.stats.anderson(np.asarray(residuals))
# print('Anderon-Darling test - Statistic: %.3f' % result.statistic)
# for i in range(len(result.critical_values)):
# 	sl, cv = result.significance_level[i], result.critical_values[i]
# 	if result.statistic < result.critical_values[i]:
# 		print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
# 	else:
# 		print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))
# stat, p = scipy.stats.normaltest(np.asarray(residuals))
# print("D'agostino test: s = %.3f, p-value = %.3f" % (stat, p))
# stat, p = scipy.stats.shapiro(np.asarray(residuals))
# print("Shapiro-Wilk test: s = %.3f, p-value = %.3f" % (stat, p))

# ## residuals are autocorrelated
# ## PLOT 1 : residuals
# fig, ax = plt.subplots(3, 1)
# ax[0].plot(year, residuals,'*')
# ax[0].set_xlabel('Year', fontsize=fontsize)
# b1 = tsaplots.plot_acf(residuals, ax=ax[1], lags=10, label='autocorr')
# ax[1].set_title('')
# ax[1].set_ylabel('Autocorrelation', fontsize=fontsize)
# ax[1].set_xlabel('Lags', fontsize=fontsize)
# normal = [np.random.normal(np.mean(residuals), np.std(residuals),100000)]
# sns.distplot(np.asarray(residuals), rug=1, 
# 	kde_kws={"lw": 1, "label": "KDE"}, 	ax=ax[2])
# sns.distplot(np.asarray(normal), rug=1, 
# 	kde_kws={"lw": 1, "label": "normal"}, ax=ax[2])
# ax[2].legend(loc='upper left', fontsize=fontsize)
# ax[2].set_ylabel('Probability density', fontsize=fontsize)
# fig.suptitle(title)
# fig.tight_layout()
# ## remove autoregressive component of residuals
# print('\n\n\t\t#### autoregressed residuals ####')
# residuals2 = (residuals[1:] - residuals[:-1] * np.exp(np.asarray(gsig[:-1])))

# ## normality test on residuals
# ## residuals are not normal yet
# f = qqplot(np.asarray(residuals2), line='s')
# f.suptitle(title)
# result = scipy.stats.anderson(np.asarray(residuals2))
# print('Anderon-Darling test - Statistic: %.3f' % result.statistic)
# for i in range(len(result.critical_values)):
# 	sl, cv = result.significance_level[i], result.critical_values[i]
# 	if result.statistic < result.critical_values[i]:
# 		print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
# 	else:
# 		print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))
# stat, p = scipy.stats.normaltest(np.asarray(residuals2))
# print("D'agostino test: s = %.3f, p-value = %.3f" % (stat, p))
# stat, p = scipy.stats.shapiro(np.asarray(residuals2))
# print("Shapiro-Wilk test: s = %.3f, p-value = %.3f" % (stat, p))

# # residuals are not autocorrelated
# # PLOT 2 : residuals
# fig, ax = plt.subplots(3, 1)
# ax[0].plot(year[:-1], residuals2,'*')
# ax[0].set_xlabel('Year', fontsize=fontsize)
# b1 = tsaplots.plot_acf(residuals2, ax=ax[1], lags=10, label='autocorr')
# ax[1].set_title('')
# ax[1].set_ylabel('Autocorrelation', fontsize=fontsize)
# ax[1].set_xlabel('Lags', fontsize=fontsize)
# normparams = norm.fit(residuals2)
# cauchyparams = cauchy.fit(residuals2)
# sns.distplot(np.asarray(residuals2), rug=1, 
# 	kde_kws={"lw": 1, "label": "KDE"}, 	ax=ax[2])
# sns.distplot(norm.rvs(*normparams, 10000), rug=1,
# 	kde_kws={"lw": 1, "label": "normal"}, ax=ax[2])
# sns.distplot(np.clip(cauchy.rvs(*cauchyparams, 10000), -0.2, 0.2), rug=1,
# 	kde_kws={"lw": 1, "label": "cauchy"}, ax=ax[2])
# ax[2].legend(loc='upper left', fontsize=fontsize)
# ax[2].set_ylabel('Probability density', fontsize=fontsize)
# fig.suptitle(title)
# fig.tight_layout()
# ## kolmogorov smirnov test
# print('\t\tKolmogorov Smirnov test for Normal distribution:')
# print('\t\t\t\t'+str(kstest(residuals2, 'norm', normparams)))
# print('\t\tKolmogorov Smirnov test for Cauchy distribution:')
# print('\t\t\t\t'+str(kstest(residuals2, 'cauchy', cauchyparams)))
# ## plot ECDF
# ecdf = ECDF(residuals2)
# plt.figure()
# vec = [x for x in np.arange(-0.05, 0.05, 0.001)]
# plt.plot(vec, ecdf(vec), label='ECDF')
# plt.plot(vec, cauchy.cdf(vec, *cauchyparams), label='cauchy')
# plt.plot(vec, norm.cdf(vec, *normparams), label='normal')
# plt.legend()
# plt.title(title)

## consider multiplicative noise
print('\n\n\t\t#### MULTIPLICATIVE NOISE CASE ####')
print('\t\t#### residuals ####')
residuals = sigma_model/sig
title = 'Multiplicative disturbance case'

## normality test on residuals
## residuals are not normal
f = qqplot(np.asarray(residuals), line='s')
f.suptitle(title)
result = scipy.stats.anderson(np.asarray(residuals))
print('Anderon-Darling test - Statistic: %.3f' % result.statistic)
for i in range(len(result.critical_values)):
	sl, cv = result.significance_level[i], result.critical_values[i]
	if result.statistic < result.critical_values[i]:
		print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
	else:
		print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))
stat, p = scipy.stats.normaltest(np.asarray(residuals))
print("D'agostino test: s = %.3f, p-value = %.3f" % (stat, p))
stat, p = scipy.stats.shapiro(np.asarray(residuals))
print("Shapiro-Wilk test: s = %.3f, p-value = %.3f" % (stat, p))

## residuals are autocorrelated
## PLOT 1 : residuals
fig, ax = plt.subplots(3, 1)
ax[0].plot(year, residuals,'*')
ax[0].set_xlabel('Year', fontsize=fontsize)
b1 = tsaplots.plot_acf(residuals, ax=ax[1], lags=10, label='autocorr')
ax[1].set_title('')
ax[1].set_ylabel('Autocorrelation', fontsize=fontsize)
ax[1].set_xlabel('Lags', fontsize=fontsize)
normparams = norm.fit(residuals)
sns.distplot(np.asarray(residuals), rug=1, 
	kde_kws={"lw": 1, "label": "KDE"}, 	ax=ax[2])
sns.distplot(norm.rvs(*normparams, 1000), rug=1, 
	kde_kws={"lw": 1, "label": "normal"}, ax=ax[2])
ax[2].legend(loc='upper left', fontsize=fontsize)
ax[2].set_ylabel('Probability density', fontsize=fontsize)
fig.suptitle(title)
fig.tight_layout()

## remove autoregressive component of residuals
print('\n\n\t\t#### autoregressed residuals ####')
residuals2 = (residuals[1:]/(residuals[:-1]))

## normality test on residuals
## residuals are not normal yet
f = qqplot(np.asarray(residuals2), line='s')
f.suptitle(title)
result = scipy.stats.anderson(np.asarray(residuals2))
print('Anderon-Darling test - Statistic: %.3f' % result.statistic)
for i in range(len(result.critical_values)):
	sl, cv = result.significance_level[i], result.critical_values[i]
	if result.statistic < result.critical_values[i]:
		print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
	else:
		print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))
stat, p = scipy.stats.normaltest(np.asarray(residuals2))
print("D'agostino test: s = %.3f, p-value = %.3f" % (stat, p))
stat, p = scipy.stats.shapiro(np.asarray(residuals2))
print("Shapiro-Wilk test: s = %.3f, p-value = %.3f" % (stat, p))

## 
fig, ax = plt.subplots(3, 1)
ax[0].plot(year[:-1], residuals2,'*')
ax[0].set_xlabel('Year', fontsize=fontsize)
b1 = tsaplots.plot_acf(residuals2, ax=ax[1], lags=10, label='autocorr')
ax[1].set_title('')
ax[1].set_ylabel('Autocorrelation', fontsize=fontsize)
ax[1].set_xlabel('Lags', fontsize=fontsize)
normparams = norm.fit(residuals2)
cauchyparams = cauchy.fit(residuals2)
sns.distplot(np.asarray(residuals2), rug=1, 
	kde_kws={"lw": 1, "label": "KDE"}, 	ax=ax[2])
sns.distplot(norm.rvs(*normparams, 10000), rug=1,
	kde_kws={"lw": 1, "label": "normal"}, ax=ax[2])
sns.distplot(np.clip(cauchy.rvs(*cauchyparams, 10000), 0.84, 1.16), rug=1, 
	kde_kws={"lw": 1, "label": "cauchy"}, ax=ax[2])
ax[2].legend(loc='upper left', fontsize=fontsize)
ax[2].set_ylabel('Probability density', fontsize=fontsize)
fig.suptitle(title)
fig.tight_layout()
## kolmogorov smirnov test
print('\t\tKolmogorov Smirnov test for Normal distribution:')
print('\t\t\t\t'+str(kstest(residuals2, 'norm', normparams)))
print('\t\tKolmogorov Smirnov test for Cauchy distribution:')
print('\t\t\t\t'+str(kstest(residuals2, 'cauchy', cauchyparams)))
## plot ECDF
ecdf = ECDF(residuals2)
plt.figure()
vec = [x for x in np.arange(0.84, 1.16, 0.001)]
plt.plot(vec, ecdf(vec), label='ECDF')
plt.plot(vec, cauchy.cdf(vec, *cauchyparams), label='cauchy')
plt.plot(vec, norm.cdf(vec, *normparams), label='normal')
plt.legend()
plt.title(title)

### IN THE LONG RUN - MULTIPLICATIVE NOISE

plt.figure()
sigma_model, _ = ci_func.sigma_model(sig[0], gsig0=gsig0, horizon=540)
plt.legend()
plt.xlabel("Year")
plt.ylabel("[kgCO2/constant 2005 USD]")
plt.title("Carbon Intensity")
vec = []
for x in range(0,100):
	sigma_model_s = ci_func.sigma_model_smul(sig[-1], gsig0=gsig0,
	 stdev=normparams[1], horizon=485)
	vec.append(sigma_model_s)
	plt.plot([y for y in range(2015, 2500)], sigma_model_s, 
		linewidth=0.25, linestyle='dashed')
plt.plot([y for y in range(2015, 2500)], np.mean(vec, axis=0), 
	color='green', linestyle='dotted')
plt.plot([y for y in range(1960, 2500)], sigma_model, label='model')
plt.plot([y for y in range(1960, 2016)], sig, label='obs', color='red')
plt.legend()
plt.title('Normal mult. disturbance - tstep = 1y')

plt.figure()
sigma_model, _ = ci_func.sigma_model(sig[0], gsig0=gsig0, horizon=540)
plt.legend()
plt.xlabel("Year")
plt.ylabel("[kgCO2/constant 2005 USD]")
plt.title("Carbon Intensity")
vec = []
params = cauchy.fit(residuals2)
# print(params)
for x in range(0,100):
	sigma_model_s = ci_func.sigma_model_smul_c(sig[-1], gsig0=gsig0,
	 params=[1.00,cauchyparams[1]], horizon=485)
	vec.append(sigma_model_s)
	plt.plot([y for y in range(2015, 2500)], sigma_model_s, 
		linewidth=0.25, linestyle='dashed')
plt.plot([y for y in range(2015, 2500)], np.mean(vec, axis=0), 
	color='green', linestyle='dotted')
plt.plot([y for y in range(1960, 2500)], sigma_model, label='model')
plt.plot([y for y in range(1960, 2016)], sig, label='obs', color='red')
plt.legend()
plt.title('Cauchy mult. disturbance - tstep = 1y')

tstep = 5
plt.figure()
vec = []
for x in range(0,100):
	sigma_model_s = ci_func.sigma_model_smul(sig[-1], gsig0=gsig0, 
		tstep=tstep, stdev=(tstep**0.5)*normparams[1], 
		horizon=round(485/tstep))
	vec.append(sigma_model_s)
	plt.plot([y for y in range(2015, 2498, tstep)], sigma_model_s, 
		linewidth=0.3, linestyle='dashed')
plt.plot([y for y in range(2015, 2498, tstep)], np.mean(vec, axis=0), 
	color='green', linestyle='dotted')
plt.plot([y for y in range(1960, 2500)], sigma_model, label='model')
plt.plot([y for y in range(1960, 2016)], sig, label='obs', color='red')
plt.xlabel("Year")
plt.ylabel("[kgCO2/constant 2005 USD]")
plt.legend()
plt.title('Normal mult. disturbance - tstep = 5y')

plt.figure()
sigma_model, _ = ci_func.sigma_model(sig[0], gsig0=gsig0, horizon=540)
vec = []
for x in range(0,100):
	sigma_model_s = ci_func.sigma_model_smul_c(sig[-1], tstep=tstep, 
		gsig0=gsig0, params=[1.00,(tstep**0.5)*cauchyparams[1]], 
		horizon=round(485/tstep))
	vec.append(sigma_model_s)
	plt.plot([y for y in range(2015, 2498, tstep)], sigma_model_s, 
		linewidth=0.25, linestyle='dashed')
plt.plot([y for y in range(2015, 2498, tstep)], np.mean(vec, axis=0), 
	color='green', linestyle='dotted')
plt.plot([y for y in range(1960, 2500)], sigma_model, label='model')
plt.plot([y for y in range(1960, 2016)], sig, label='obs', color='red')
plt.xlabel("Year")
plt.ylabel("[kgCO2/constant 2005 USD]")
plt.legend()
plt.title('Cauchy mult. disturbance - tstep = 5y')

print('Scale parameter of Cauchy distribution for 5-y time steps is', 
	(5**0.5)*params[1])
print('Scale parameter of Normal distribution for 5-y time steps is', 
	(5**0.5)*np.std(residuals2))

plt.show()