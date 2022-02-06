import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
import pandas as pd
import scipy
from scipy.stats import cauchy, norm, kstest
from statsmodels.graphics import tsaplots
from statsmodels.distributions.empirical_distribution import ECDF
## SETTINGS FOR PLOTS
sns.set_style('whitegrid')

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

##### MODEL HINDCAST AND REOPTIMIZATION #####

##### population model
popasym = 11500  ## default param in DICE2016
popadj = 0.134  ## default param in DICE2016
# standard model
l_model = econ_func.pop_model(POP[0], popasym=popasym, 
	popadj=popadj, horizon=len(POP))
# reoptimize params
popasym_opt, popadj_opt = scipy.optimize.minimize( 
	econ_func.residual_pop_model,
	[popasym, popadj], args=(POP), method='L-BFGS-B', 
	bounds=[(11499,11501), (0,1)]).x
l_model_opt = econ_func.pop_model(POP[0], popasym=popasym_opt, 
	popadj=popadj_opt, horizon=len(POP))

##### TFP model
delaa = 0.005 ## default param in DICE2016
ga0 = 0.079 ## default param in DICE2016
# standard model
tfp_model = econ_func.tfp_model(A[0], deltaa = 0.005, 
	ga0 = ga0/np.exp(-delaa*(2015-year_obs[0])), horizon=len(A))
# reoptimize params
delaa_opt, ga0_opt = scipy.optimize.minimize( 
	econ_func.residual_tfp_model, 
	[0.005, 0.076], args=(A), bounds=[(0.0, 0.99), (0.0,0.99)], 
	method='L-BFGS-B').x
tfp_model_opt = econ_func.tfp_model(A[0], horizon=len(A), 
	deltaa=delaa_opt, ga0=ga0_opt)

###### capital stock model
# standard model based on model output
k_model, i_model = econ_func.k_model_endo(k, 
	l_model, tfp_model, tstep=1)
# standard model based on reoptimized model output
k_model_endo, i_model_endo = econ_func.k_model_endo(k, 
	l_model_opt, tfp_model_opt, tstep=1)
# reoptimize params
dk_model = 0.9 ## default param in DICE2016
dk_model_opt = scipy.optimize.minimize_scalar( 
	econ_func.residual_k_model_endo, 
	args=(k, l_model_opt, tfp_model_opt), bounds=( (0.0, 1.0)), 
	method='bounded').x
k_model_endo_opt, i_model_endo_opt = econ_func.k_model_endo(k, 
	l_model_opt, tfp_model_opt, tstep=1, dk=dk_model_opt)

##### GDP model
# standard model
gdp_model = econ_func.gdp_model(k_model, l_model, tfp_model)
# standard model , reopt pop & tfp
gdp_model_endo = econ_func.gdp_model(k_model_endo, 
	l_model_opt, tfp_model_opt)
# reopt model
gdp_model_endo_opt = econ_func.gdp_model(k_model_endo_opt, 
	l_model_opt, tfp_model_opt)

### FIGURE 1 - obs vs models
fig, ax = plt.subplots(2,2)
## Population
ax[0,0].set_title("Population")
ax[0,0].plot(year_obs, POP, 'k--', label='Observed (World Bank)')
ax[0,0].plot(year_obs, l_model, label='Model')
ax[0,0].plot(year_obs, l_model_opt, label='Model-opt')
ax[0,0].legend()
ax[0,0].set_ylabel("Million people")
ax[0,0].set_xlabel("Year")
## TFP
ax[0,1].set_title("Total Factor Productivity")
ax[0,1].set_xlabel("Year")
ax[0,1].plot(year_obs, A, 'k--', label = 'Based on observations (IMF)')
ax[0,1].plot(year_obs, tfp_model, label = 'Model')
ax[0,1].plot(year_obs, tfp_model_opt, label = 'Model-opt')
ax[0,1].set_ylabel("-")
ax[0,1].legend()
## Capital stock
ax[1,0].set_title("Capital stock")
ax[1,0].plot(year_obs, k, 'k--', label = 'Observed (IMF)')
ax[1,0].plot(year_obs, k_model, label = 'Model')
# ax[1,0].plot(year_obs, k_model_endo, label = 'Model-endo')
ax[1,0].plot(year_obs, k_model_endo_opt, label = 'Model-opt')
ax[1,0].set_ylabel("Trillion USD")
ax[1,0].legend()
ax[1,0].set_xlabel("Year")
## GDP
ax[1,1].set_title("GDP")
ax[1,1].plot(year_obs, GDP, 'k--', label = 'Observed (IMF)')
ax[1,1].plot(year_obs, gdp_model, label = 'Model')
# ax[1,1].plot(year_obs, gdp_model_endo, label = 'Model-endo')
ax[1,1].plot(year_obs, gdp_model_endo_opt, label = 'Model-opt')
ax[1,1].set_ylabel("Trillion USD")
ax[1,1].legend()
ax[1,1].set_xlabel("Year")
fig.suptitle('Observed data, models and recalibrated models')
fig.tight_layout()

print("\n\n\t\t\tADDITIVE NOISE\n\n\n")

#### ADDITIVE NOISE CASE #####
# analyze errors
### FIGURE 2 - errors of the models
fig,ax = plt.subplots(2,2, sharex=True)
error_pop = (np.asarray(POP) - np.asarray(l_model))
ax[0,0].plot(year_obs, error_pop)
ax[0,0].set_ylabel("Million people")
ax[0,0].set_title("Population")

error_A = (np.asarray(A) - np.asarray(tfp_model))#/np.asarray(A)
ax[0,1].plot(year_obs, error_A)
ax[0,1].set_ylabel("-")
ax[0,1].set_title("TFP")

error_k = (np.asarray(k) - np.asarray(k_model))#/np.asarray(k)
ax[1,0].plot(year_obs, error_k)
ax[1,0].set_ylabel("Trillion USD")
ax[1,0].set_xlabel("Year")
ax[1,0].set_title("Capital Stock")

error_gdp = (np.asarray(GDP) - np.asarray(gdp_model))
ax[1,1].plot(year_obs, error_gdp)
ax[1,1].set_ylabel("Trillion USD")
ax[1,1].set_xlabel("Year")
ax[1,1].set_title("GDP")
fig.suptitle('Residuals of the models')
fig.tight_layout()

### FIGURE 3 - errors of the reoptimized models
fig,ax = plt.subplots(2,2)
error_pop = (np.asarray(POP) - np.asarray(l_model_opt))
ax[0,0].plot(year_obs, error_pop)
ax[0,0].set_ylabel("Million people")
ax[0,0].set_title("Population")

error_A = (np.asarray(A) - np.asarray(tfp_model_opt))#/np.asarray(A)
ax[0,1].plot(year_obs, error_A)
ax[0,1].set_ylabel("-")
ax[0,1].set_title("TFP")

error_k = (np.asarray(k) - np.asarray(k_model_endo_opt))#/np.asarray(k)
ax[1,0].plot(year_obs, error_k)
ax[1,0].set_ylabel("Trillion USD")
ax[1,0].set_xlabel("Year")
ax[1,0].set_title("Capital Stock")

error_gdp = (np.asarray(GDP) - np.asarray(gdp_model_endo_opt))
ax[1,1].plot(year_obs, error_gdp)
ax[1,1].set_ylabel("Trillion USD")
ax[1,1].set_xlabel("Year")
ax[1,1].set_title("GDP")
fig.suptitle('Residuals of the reoptimized models')
fig.tight_layout()

# ### FIGURE 4 - residuals distribution
# fig = plt.figure()
# ax1 = plt.subplot2grid((4,4),(0,0),rowspan=2)
# ax2 = plt.subplot2grid((4,4),(0,1))
# ax3 = plt.subplot2grid((4,4),(1,1))
# ax4 = plt.subplot2grid((4,4),(0,2),rowspan=2)
# ax5 = plt.subplot2grid((4,4),(0,3))
# ax6 = plt.subplot2grid((4,4),(1,3))
# ax7 = plt.subplot2grid((4,4),(2,0),rowspan=2)
# ax8 = plt.subplot2grid((4,4),(2,1))
# ax9 = plt.subplot2grid((4,4),(3,1))
# ax10 = plt.subplot2grid((4,4),(2,2),rowspan=2)
# ax11 = plt.subplot2grid((4,4),(2,3))
# ax12 = plt.subplot2grid((4,4),(3,3))

# error_pop = np.asarray(POP) - np.asarray(l_model)
# sns.distplot(np.asarray(error_pop), rug=1, 
# 	ax=ax1, kde_kws={"label": "KDE"})
# sns.distplot(norm.rvs(*norm.fit(error_pop), size=10000), 
# 	rug=1, ax=ax1, kde_kws={"label": "Normal dist. fit"})
# ax1.set_title("Population")
# ax1.legend()
# tsaplots.plot_acf(error_pop, ax=ax2, lags=10)
# tsaplots.plot_pacf(error_pop, ax=ax3,lags=10, method='ols')

# error_A = np.asarray(A) - np.asarray(tfp_model)
# sns.distplot(np.asarray(error_A), rug=1, 
# 	ax=ax4, kde_kws={"label": "KDE"})
# sns.distplot(norm.rvs(*norm.fit(error_A), size=10000), 
# 	rug=1, ax=ax4, kde_kws={"label": "Normal dist. fit"})
# ax4.set_title("TFP")
# ax4.legend()
# tsaplots.plot_acf(error_A, ax=ax5, lags=10)
# tsaplots.plot_pacf(error_A, ax=ax6,lags=10)

# error_k = np.asarray(k) - np.asarray(k_model)
# sns.distplot(np.asarray(error_k), rug=1, 
# 	ax=ax7, kde_kws={"label": "KDE"})
# sns.distplot(norm.rvs(*norm.fit(error_k), size=10000), 
# 	rug=1, ax=ax7, kde_kws={"label": "Normal dist. fit"})
# ax7.set_title("Capital Stock")
# ax7.legend()
# tsaplots.plot_acf(error_k, ax=ax8, lags=10)
# tsaplots.plot_pacf(error_k, ax=ax9,lags=10)

# error_gdp = np.asarray(GDP) - np.asarray(gdp_model)
# sns.distplot(np.asarray(error_gdp), rug=1, 
# 	ax=ax10, kde_kws={"label": "KDE"})
# sns.distplot(norm.rvs(*norm.fit(error_gdp), size=10000), 
# 	rug=1, ax=ax10, kde_kws={"label": "Normal dist. fit"})
# ax10.set_title("GDP")
# ax10.legend()
# tsaplots.plot_acf(error_gdp, ax=ax11, lags=10)
# tsaplots.plot_pacf(error_gdp, ax=ax12,lags=10)
# fig.suptitle('Models residuals distributions')
# plt.tight_layout()

# ### FIGURE 5 - residuals distribution
# fig = plt.figure()
# ax1 = plt.subplot2grid((4,4),(0,0),rowspan=2)
# ax2 = plt.subplot2grid((4,4),(0,1))
# ax3 = plt.subplot2grid((4,4),(1,1))
# ax4 = plt.subplot2grid((4,4),(0,2),rowspan=2)
# ax5 = plt.subplot2grid((4,4),(0,3))
# ax6 = plt.subplot2grid((4,4),(1,3))
# ax7 = plt.subplot2grid((4,4),(2,0),rowspan=2)
# ax8 = plt.subplot2grid((4,4),(2,1))
# ax9 = plt.subplot2grid((4,4),(3,1))
# ax10 = plt.subplot2grid((4,4),(2,2),rowspan=2)
# ax11 = plt.subplot2grid((4,4),(2,3))
# ax12 = plt.subplot2grid((4,4),(3,3))

# error_pop = np.asarray(POP) - np.asarray(l_model_opt)
# sns.distplot(np.asarray(error_pop), rug=1, 
# 	ax=ax1, kde_kws={"label": "KDE"})
# sns.distplot(norm.rvs(*norm.fit(error_pop), size=10000), 
# 	rug=1, ax=ax1, kde_kws={"label": "Normal dist. fit"})
# ax1.set_title("Population")
# ax1.legend()
# tsaplots.plot_acf(error_pop, ax=ax2, lags=10)
# tsaplots.plot_pacf(error_pop, ax=ax3,lags=10, method='ols')

# error_A = np.asarray(A) - np.asarray(tfp_model_opt)
# sns.distplot(np.asarray(error_A), rug=1, 
# 	ax=ax4, kde_kws={"label": "KDE"})
# sns.distplot(norm.rvs(*norm.fit(error_A), size=10000), 
# 	rug=1, ax=ax4, kde_kws={"label": "Normal dist. fit"})
# ax4.set_title("TFP")
# ax4.legend()
# tsaplots.plot_acf(error_A, ax=ax5, lags=10)
# tsaplots.plot_pacf(error_A, ax=ax6,lags=10)

# error_k = np.asarray(k) - np.asarray(k_model_endo_opt)
# sns.distplot(np.asarray(error_k), rug=1, 
# 	ax=ax7, kde_kws={"label": "KDE"})
# sns.distplot(norm.rvs(*norm.fit(error_k), size=10000), 
# 	rug=1, ax=ax7, kde_kws={"label": "Normal dist. fit"})
# ax7.set_title("Capital Stock")
# ax7.legend()
# tsaplots.plot_acf(error_k, ax=ax8, lags=10)
# tsaplots.plot_pacf(error_k, ax=ax9,lags=10)

# error_gdp = np.asarray(GDP) - np.asarray(gdp_model_endo_opt)
# sns.distplot(np.asarray(error_gdp), rug=1, 
# 	ax=ax10, kde_kws={"label": "KDE"})
# sns.distplot(norm.rvs(*norm.fit(error_gdp), size=10000), 
# 	rug=1, ax=ax10, kde_kws={"label": "Normal dist. fit"})
# ax10.set_title("GDP")
# ax10.legend()
# tsaplots.plot_acf(error_gdp, ax=ax11, lags=10)
# tsaplots.plot_pacf(error_gdp, ax=ax12,lags=10)
# fig.suptitle('Reoptimized models residuals distributions')
# plt.tight_layout()

## following these plots we proceed to model stochastic disturbance
## only in tfp dynamic equation 

## computing and plotting residuals after autoregressing
error_A = np.asarray(A) - np.asarray(tfp_model) 
regr_error_A = []
for x in range(len(error_A[:-1])):
	regr_error_A.append(error_A[x] / \
		(1 - ga0/(np.exp(-delaa*(2015-year_obs[0])))*np.exp(-delaa*(x)*1)))
residuals2 = error_A[1:] - np.asarray(regr_error_A)
## does not change significantly if tfp_model_opt is used
# error_A = np.asarray(A) - np.asarray(tfp_model_opt) 
# regr_error_A = []
# for x in range(len(error_A[:-1])):
# 	regr_error_A.append(error_A[x] / \
#		(1 - ga0_opt*np.exp(-delaa_opt*(x)*1)))
# residuals2 = error_A[1:] - np.asarray(regr_error_A)

### FIGURE 6 - autoregressed residuals distribution
fig, ax = plt.subplots(3,1)
ax[0].plot(year_obs[:-1], residuals2, '*')
ax[0].set_xlabel('Year')
ax[0].set_ylabel('-')
ax[0].set_title('TFP residuals after autoregression')
tsaplots.plot_acf(residuals2, ax=ax[1], lags=10)
ax[1].set_title('Autocorrelation')
ax[1].set_ylabel('-')
ax[1].set_xlabel('Lags')
sns.distplot(np.asarray(residuals2), rug=1, 
	ax = ax[2], kde_kws={"label": "KDE"})
sns.distplot(norm.rvs(*norm.fit(residuals2), size=10000), 
	rug=1, ax = ax[2], kde_kws={"label": "Normal dist. fit"})
sns.distplot(np.clip( 
	cauchy.rvs(*cauchy.fit(residuals2), size=10000), -1, 1.0),  
	rug=1, ax = ax[2], kde_kws={"label": "Cauchy dist. fit"})
ax[2].legend()
ax[2].set_title('Distribution')
ax[2].set_ylabel('Probability density')
fig.tight_layout()
## FIGURE - plot ECDF
ecdf = ECDF(residuals2)
plt.figure()
vec = [x for x in np.arange(-0.5, 0.5, 0.001)]
plt.plot(vec, ecdf(vec), label='ECDF')
plt.plot(vec, norm.cdf(vec, *norm.fit(residuals2)), label='normal')
plt.plot(vec, cauchy.cdf(vec, *cauchy.fit(residuals2)), label='cauchy')
plt.legend()
## kolmogorov smirnov test
normparams = norm.fit(residuals2)
cauchyparams = cauchy.fit(residuals2)
print('\t\tKolmogorov Smirnov test for Normal distribution:')
print('\t\t\t\t'+str(kstest(residuals2, 'norm', normparams)))
print('\t\tKolmogorov Smirnov test for Cauchy distribution:')
print('\t\t\t\t'+str(kstest(residuals2, 'cauchy', cauchyparams)))
print(normparams)
print(cauchyparams)

#### what happens in the long run?
default = True ##if true simulates DICE, false uses reoptimized models
if default is False:
	params = [popasym_opt, popadj_opt, delaa_opt, ga0_opt, dk_model_opt]
	init = [POP[0], A[0], [k[0]]]
	start_horizon = year_obs[0]
else:
	params = [popasym, popadj, delaa, ga0, dk_model]
	start_horizon = 2015
	init = [7403, 5.115, [223]]
end_horizon=2500
tstep = 5
year_sim = [x for x in range(start_horizon, end_horizon, tstep)]
#### FIGURE - simulation
fig, ax = plt.subplots(2,2, sharex=True)
l_model_lt = econ_func.pop_model(init[0], popasym=params[0], 
	popadj=params[1], horizon=len(year_sim), tstep=tstep)
vec, vec2, vec3 = [], [], []
for el in range(0,100):
	## uncomment the following to check normal distribution
	# tfp_model_s = econ_func.tfp_model_s(A[0], deltaa=delaa_opt, 
	# 	ga0=ga0_opt, mu=0, sigma=normparams[1]*(tstep**0.5), horizon=len(year_sim))
	## comment the following to hide cauchy distribution
	tfp_model_s = econ_func.tfp_model_s_c(init[1], deltaa=params[2], 
		ga0=params[3], loc=0, scale=cauchyparams[1]*(tstep**0.5), 
		tstep=tstep, horizon=len(year_sim))
	vec.append(tfp_model_s)
	ax[0][1].plot(year_sim, tfp_model_s, linewidth=0.25, linestyle='dashed')
	k_model_s, _ = econ_func.k_model_endo(init[2], l_model_lt, tfp_model_s,
		tstep=tstep, dk=params[4], horizon=len(year_sim))
	vec2.append(k_model_s)
	ax[1][0].plot(year_sim, k_model_s, linewidth=0.25, linestyle='dashed')
	gdp_model_s = econ_func.gdp_model(k_model_s, l_model_lt, tfp_model_s)
	vec3.append(gdp_model_s)
	ax[1][1].plot(year_sim, gdp_model_s, linewidth=0.25, linestyle='dashed')
ax[0][1].plot(year_sim,	np.mean(vec, axis=0), label='mean', color='green')
ax[1][0].plot(year_sim, np.mean(vec2, axis=0), label='mean', color='green')
ax[1][1].plot(year_sim, np.mean(vec3, axis=0), label='mean', color='green')
ax[0][0].set_ylabel('Million people')
ax[0][0].set_title('Population')
ax[0][1].set_ylabel('-')
ax[0][1].set_title('TFP')
ax[1][0].set_ylabel('Trillion USD')
ax[1][0].set_title('Capital Stock')
ax[1][0].set_xlabel('Year')
ax[1][1].set_ylabel('Trillion USD')
ax[1][1].set_title('GDP')
ax[1][1].set_xlabel('Year')
ax[0][0].plot(year_sim, l_model_lt, 'g', label='long-term')
ax[0][0].plot(year_obs, l_model_opt, label='model-opt')
ax[0][1].plot(year_obs, tfp_model_opt, label='model-opt')
ax[1][0].plot(year_obs,	k_model_endo_opt, label='endo-opt')
ax[1][1].plot(year_obs, gdp_model_endo_opt, label='endo-opt')
ax[0][0].plot(year_obs, POP, 'k--', label='obs')
ax[0][1].plot(year_obs, A, 'k--', label='obs')
ax[1][0].plot(year_obs, k, 'k--', label='obs')
ax[1][1].plot(year_obs, GDP, 'k--', label='obs')
for el in ax:
	for el2 in el:
		el2.legend()
fig.suptitle('Simulation from '+str(start_horizon)+' to '+str(end_horizon)+', tstep = '+str(tstep))
plt.tight_layout()

print(cauchyparams[1]*(5**0.5))
print("\n\n\t\t\tMULTIPLICATIVE NOISE\n\n\n")

# # ## MULTIPLICATIVE NOISE

# analyze errors
### FIGURE 2 - errors of the models
fig,ax = plt.subplots(2,2, sharex=True)
error_pop = (np.asarray(POP) / np.asarray(l_model))
ax[0,0].plot(year_obs, error_pop)
ax[0,0].set_ylabel("Million people")
ax[0,0].set_title("Population")

error_A = (np.asarray(A) / np.asarray(tfp_model))#/np.asarray(A)
ax[0,1].plot(year_obs, error_A)
ax[0,1].set_ylabel("-")
ax[0,1].set_title("TFP")

error_k = (np.asarray(k) / np.asarray(k_model))#/np.asarray(k)
ax[1,0].plot(year_obs, error_k)
ax[1,0].set_ylabel("Trillion USD")
ax[1,0].set_xlabel("Year")
ax[1,0].set_title("Capital Stock")

error_gdp = (np.asarray(GDP) / np.asarray(gdp_model))
ax[1,1].plot(year_obs, error_gdp)
ax[1,1].set_ylabel("Trillion USD")
ax[1,1].set_xlabel("Year")
ax[1,1].set_title("GDP")
fig.suptitle('Residuals of the models')
fig.tight_layout()

### FIGURE 3 - errors of the reoptimized models
fig,ax = plt.subplots(2,2)
error_pop = (np.asarray(POP) / np.asarray(l_model_opt))
ax[0,0].plot(year_obs, error_pop)
ax[0,0].set_ylabel("Million people")
ax[0,0].set_title("Population")

error_A = (np.asarray(A) / np.asarray(tfp_model_opt))#/np.asarray(A)
ax[0,1].plot(year_obs, error_A)
ax[0,1].set_ylabel("-")
ax[0,1].set_title("TFP")

error_k = (np.asarray(k) / np.asarray(k_model_endo_opt))#/np.asarray(k)
ax[1,0].plot(year_obs, error_k)
ax[1,0].set_ylabel("Trillion USD")
ax[1,0].set_xlabel("Year")
ax[1,0].set_title("Capital Stock")

error_gdp = (np.asarray(GDP) / np.asarray(gdp_model_endo_opt))
ax[1,1].plot(year_obs, error_gdp)
ax[1,1].set_ylabel("Trillion USD")
ax[1,1].set_xlabel("Year")
ax[1,1].set_title("GDP")
fig.suptitle('Residuals of the reoptimized models')
fig.tight_layout()

# ### FIGURE 4 - residuals distribution
# fig = plt.figure()
# ax1 = plt.subplot2grid((4,4),(0,0),rowspan=2)
# ax2 = plt.subplot2grid((4,4),(0,1))
# ax3 = plt.subplot2grid((4,4),(1,1))
# ax4 = plt.subplot2grid((4,4),(0,2),rowspan=2)
# ax5 = plt.subplot2grid((4,4),(0,3))
# ax6 = plt.subplot2grid((4,4),(1,3))
# ax7 = plt.subplot2grid((4,4),(2,0),rowspan=2)
# ax8 = plt.subplot2grid((4,4),(2,1))
# ax9 = plt.subplot2grid((4,4),(3,1))
# ax10 = plt.subplot2grid((4,4),(2,2),rowspan=2)
# ax11 = plt.subplot2grid((4,4),(2,3))
# ax12 = plt.subplot2grid((4,4),(3,3))

# error_pop = np.asarray(POP) / np.asarray(l_model)
# sns.distplot(np.asarray(error_pop), rug=1, 
# 	ax=ax1, kde_kws={"label": "KDE"})
# sns.distplot(norm.rvs(*norm.fit(error_pop), size=10000), 
# 	rug=1, ax=ax1, kde_kws={"label": "Normal dist. fit"})
# ax1.set_title("Population")
# ax1.legend()
# tsaplots.plot_acf(error_pop, ax=ax2, lags=10)
# tsaplots.plot_pacf(error_pop, ax=ax3,lags=10, method='ols')

# error_A = np.asarray(A) / np.asarray(tfp_model)
# sns.distplot(np.asarray(error_A), rug=1, 
# 	ax=ax4, kde_kws={"label": "KDE"})
# sns.distplot(norm.rvs(*norm.fit(error_A), size=10000), 
# 	rug=1, ax=ax4, kde_kws={"label": "Normal dist. fit"})
# ax4.set_title("TFP")
# ax4.legend()
# tsaplots.plot_acf(error_A, ax=ax5, lags=10)
# tsaplots.plot_pacf(error_A, ax=ax6,lags=10)

# error_k = np.asarray(k) / np.asarray(k_model)
# sns.distplot(np.asarray(error_k), rug=1, 
# 	ax=ax7, kde_kws={"label": "KDE"})
# sns.distplot(norm.rvs(*norm.fit(error_k), size=10000), 
# 	rug=1, ax=ax7, kde_kws={"label": "Normal dist. fit"})
# ax7.set_title("Capital Stock")
# ax7.legend()
# tsaplots.plot_acf(error_k, ax=ax8, lags=10)
# tsaplots.plot_pacf(error_k, ax=ax9,lags=10)

# error_gdp = np.asarray(GDP) / np.asarray(gdp_model)
# sns.distplot(np.asarray(error_gdp), rug=1, 
# 	ax=ax10, kde_kws={"label": "KDE"})
# sns.distplot(norm.rvs(*norm.fit(error_gdp), size=10000), 
# 	rug=1, ax=ax10, kde_kws={"label": "Normal dist. fit"})
# ax10.set_title("GDP")
# ax10.legend()
# tsaplots.plot_acf(error_gdp, ax=ax11, lags=10)
# tsaplots.plot_pacf(error_gdp, ax=ax12,lags=10)
# fig.suptitle('Models residuals distributions')
# plt.tight_layout()

# ### FIGURE 5 - residuals distribution
# fig = plt.figure()
# ax1 = plt.subplot2grid((4,4),(0,0),rowspan=2)
# ax2 = plt.subplot2grid((4,4),(0,1))
# ax3 = plt.subplot2grid((4,4),(1,1))
# ax4 = plt.subplot2grid((4,4),(0,2),rowspan=2)
# ax5 = plt.subplot2grid((4,4),(0,3))
# ax6 = plt.subplot2grid((4,4),(1,3))
# ax7 = plt.subplot2grid((4,4),(2,0),rowspan=2)
# ax8 = plt.subplot2grid((4,4),(2,1))
# ax9 = plt.subplot2grid((4,4),(3,1))
# ax10 = plt.subplot2grid((4,4),(2,2),rowspan=2)
# ax11 = plt.subplot2grid((4,4),(2,3))
# ax12 = plt.subplot2grid((4,4),(3,3))

# error_pop = np.asarray(POP) / np.asarray(l_model_opt)
# sns.distplot(np.asarray(error_pop), rug=1, 
# 	ax=ax1, kde_kws={"label": "KDE"})
# sns.distplot(norm.rvs(*norm.fit(error_pop), size=10000), 
# 	rug=1, ax=ax1, kde_kws={"label": "Normal dist. fit"})
# ax1.set_title("Population")
# ax1.legend()
# tsaplots.plot_acf(error_pop, ax=ax2, lags=10)
# tsaplots.plot_pacf(error_pop, ax=ax3,lags=10, method='ols')

# error_A = np.asarray(A) / np.asarray(tfp_model_opt)
# sns.distplot(np.asarray(error_A), rug=1, 
# 	ax=ax4, kde_kws={"label": "KDE"})
# sns.distplot(norm.rvs(*norm.fit(error_A), size=10000), 
# 	rug=1, ax=ax4, kde_kws={"label": "Normal dist. fit"})
# ax4.set_title("TFP")
# ax4.legend()
# tsaplots.plot_acf(error_A, ax=ax5, lags=10)
# tsaplots.plot_pacf(error_A, ax=ax6,lags=10)

# error_k = np.asarray(k) / np.asarray(k_model_endo_opt)
# sns.distplot(np.asarray(error_k), rug=1, 
# 	ax=ax7, kde_kws={"label": "KDE"})
# sns.distplot(norm.rvs(*norm.fit(error_k), size=10000), 
# 	rug=1, ax=ax7, kde_kws={"label": "Normal dist. fit"})
# ax7.set_title("Capital Stock")
# ax7.legend()
# tsaplots.plot_acf(error_k, ax=ax8, lags=10)
# tsaplots.plot_pacf(error_k, ax=ax9,lags=10)

# error_gdp = np.asarray(GDP) / np.asarray(gdp_model_endo_opt)
# sns.distplot(np.asarray(error_gdp), rug=1, 
# 	ax=ax10, kde_kws={"label": "KDE"})
# sns.distplot(norm.rvs(*norm.fit(error_gdp), size=10000), 
# 	rug=1, ax=ax10, kde_kws={"label": "Normal dist. fit"})
# ax10.set_title("GDP")
# ax10.legend()
# tsaplots.plot_acf(error_gdp, ax=ax11, lags=10)
# tsaplots.plot_pacf(error_gdp, ax=ax12,lags=10)
# fig.suptitle('Reoptimized models residuals distributions')
# plt.tight_layout()

## following these plots we proceed to model stochastic disturbance
## only in tfp dynamic equation 

## computing and plotting residuals after autoregressing
error_A = np.asarray(A) / np.asarray(tfp_model) 
residuals2 = error_A[1:] / error_A[:-1]
## does not change significantly if tfp_model_opt is used
# error_A = np.asarray(A) - np.asarray(tfp_model_opt) 
# regr_error_A = []
# for x in range(len(error_A[:-1])):
# 	regr_error_A.append(error_A[x] / \
#		(1 - ga0_opt*np.exp(-delaa_opt*(x)*1)))
# residuals2 = error_A[1:] - np.asarray(regr_error_A)

### FIGURE 6 - autoregressed residuals distribution
fig, ax = plt.subplots(3,1)
ax[0].plot(year_obs[:-1], residuals2, '*')
ax[0].set_xlabel('Year')
ax[0].set_ylabel('-')
ax[0].set_title('TFP residuals after autoregression')
tsaplots.plot_acf(residuals2, ax=ax[1], lags=10)
ax[1].set_title('Autocorrelation')
ax[1].set_ylabel('-')
ax[1].set_xlabel('Lags')
sns.distplot(np.asarray(residuals2), rug=1, 
	ax = ax[2], kde_kws={"label": "KDE"})
sns.distplot(norm.rvs(*norm.fit(residuals2), size=10000), 
	rug=1, ax = ax[2], kde_kws={"label": "Normal dist. fit"})
sns.distplot(np.clip( 
	cauchy.rvs(*cauchy.fit(residuals2), size=10000), 0.5, 1.5),  
	rug=1, ax = ax[2], kde_kws={"label": "Cauchy dist. fit"})
ax[2].legend()
ax[2].set_title('Distribution')
ax[2].set_ylabel('Probability density')
fig.tight_layout()
## FIGURE - plot ECDF
ecdf = ECDF(residuals2)
plt.figure()
vec = [x for x in np.arange(0.5, 1.5, 0.001)]
plt.plot(vec, ecdf(vec), label='ECDF')
plt.plot(vec, norm.cdf(vec, *norm.fit(residuals2)), label='normal')
plt.plot(vec, cauchy.cdf(vec, *cauchy.fit(residuals2)), label='cauchy')
plt.legend()
## kolmogorov smirnov test
normparams = norm.fit(residuals2)
cauchyparams = cauchy.fit(residuals2)
print('\t\tKolmogorov Smirnov test for Normal distribution:')
print('\t\t\t\t'+str(kstest(residuals2, 'norm', normparams)))
print('\t\tKolmogorov Smirnov test for Cauchy distribution:')
print('\t\t\t\t'+str(kstest(residuals2, 'cauchy', cauchyparams)))
print(normparams)
print(cauchyparams)
print(cauchyparams[1]*(5**0.5))

#### what happens in the long run?
default = True ##if true simulates DICE, false uses reoptimized models
if default is False:
	params = [popasym_opt, popadj_opt, delaa_opt, ga0_opt, dk_model_opt]
	init = [POP[0], A[0], [k[0]]]
	start_horizon = year_obs[0]
else:
	params = [popasym, popadj, delaa, ga0, dk_model]
	start_horizon = 2015
	init = [7403, 5.115, [223]]
end_horizon=2500
tstep = 5
year_sim = [x for x in range(start_horizon, end_horizon, tstep)]
#### FIGURE - simulation
fig, ax = plt.subplots(2,2, sharex=True)
l_model_lt = econ_func.pop_model(init[0], popasym=params[0], 
	popadj=params[1], horizon=len(year_sim), tstep=tstep)
vec, vec2, vec3 = [], [], []
for el in range(0,100):
	## uncomment the following to check normal distribution
	# tfp_model_s = econ_func.tfp_model_smul(init[1], deltaa=delaa_opt, 
	# 	ga0=ga0_opt, mu=1, sigma=normparams[1]*(tstep**0.5), 
	# 	tstep=tstep, horizon=len(year_sim))
	## comment the following to hide cauchy distribution
	tfp_model_s = econ_func.tfp_model_smul_c(init[1], deltaa=params[2], 
		ga0=params[3], loc=1, scale=cauchyparams[1]*(tstep**0.5), 
		tstep=tstep, horizon=len(year_sim))
	vec.append(tfp_model_s)
	ax[0][1].plot(year_sim, tfp_model_s, linewidth=0.25, linestyle='dashed')
	k_model_s, _ = econ_func.k_model_endo(init[2], l_model_lt, tfp_model_s,
		tstep=tstep, dk=params[4], horizon=len(year_sim))
	vec2.append(k_model_s)
	ax[1][0].plot(year_sim, k_model_s, linewidth=0.25, linestyle='dashed')
	gdp_model_s = econ_func.gdp_model(k_model_s, l_model_lt, tfp_model_s)
	vec3.append(gdp_model_s)
	ax[1][1].plot(year_sim, gdp_model_s, linewidth=0.25, linestyle='dashed')
ax[0][1].plot(year_sim,	np.mean(vec, axis=0), label='mean', color='green')
ax[1][0].plot(year_sim, np.mean(vec2, axis=0), label='mean', color='green')
ax[1][1].plot(year_sim, np.mean(vec3, axis=0), label='mean', color='green')
ax[0][0].set_ylabel('Million people')
ax[0][0].set_title('Population')
ax[0][1].set_ylabel('-')
ax[0][1].set_title('TFP')
ax[1][0].set_ylabel('Trillion USD')
ax[1][0].set_title('Capital Stock')
ax[1][0].set_xlabel('Year')
ax[1][1].set_ylabel('Trillion USD')
ax[1][1].set_title('GDP')
ax[1][1].set_xlabel('Year')
ax[0][0].plot(year_sim, l_model_lt, 'g', label='long-term')
ax[0][0].plot(year_obs, l_model_opt, label='model-opt')
ax[0][1].plot(year_obs, tfp_model_opt, label='model-opt')
ax[1][0].plot(year_obs,	k_model_endo_opt, label='endo-opt')
ax[1][1].plot(year_obs, gdp_model_endo_opt, label='endo-opt')
ax[0][0].plot(year_obs, POP, 'k--', label='obs')
ax[0][1].plot(year_obs, A, 'k--', label='obs')
ax[1][0].plot(year_obs, k, 'k--', label='obs')
ax[1][1].plot(year_obs, GDP, 'k--', label='obs')
for el in ax:
	for el2 in el:
		el2.legend()
fig.suptitle('Simulation from '+str(start_horizon)+' to '+str(end_horizon)+', tstep = '+str(tstep))
plt.tight_layout()

plt.show()