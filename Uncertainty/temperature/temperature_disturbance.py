import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import seaborn as sns
from statsmodels.graphics import tsaplots
from statsmodels.stats import diagnostic
from statsmodels.tsa.arima_model import ARMA
from statsmodels.graphics.gofplots import qqplot
import math
import pandas as pd
import temp_functions as tmp_func 

## SETTINGS FOR PLOTS
sns.set('paper')
sns.set_style('whitegrid')
fontsize=matplotlib.rcParams['font.size']
fontsize=11
## READ TEMPERATURE DATA (HadCRUT)
cru_year = []
cru_temp = []
cru_coverage = []
with open('./HadCRUT4-gl.dat.txt') as f:
	file = f.read()
for el in file.split("\n"):
	line = el.split()
	try:
		float(line[0])
		if float(line[0]) not in cru_year:
			cru_year.append(float(line[0]))
			cru_temp.append(float(line[13]))
		elif float(line[0]) in cru_year:
			cru_coverage.append(float(line[1]))
	except:
		continue
## since average temperature is computed using 1986-2005 
## -0.63 comes from IPCC SR15C (definition of preindustrial temperature)
## to obtain temperature levels wrt 1850-1900 average (preindustrial)
tempmean1900 = np.mean(np.asarray(cru_temp[30+106:30+125])) - 0.63 
cru_temp = [el-tempmean1900 for el in cru_temp]
# print(cru_year[-10:-4], cru_temp[-10:-4], np.mean(cru_temp[-10:-4]))
# print(cru_year[131:161], cru_temp[131:161], np.mean(cru_temp[131:161]))
# print(cru_year[161:170], np.mean(cru_temp[161:170]))

## READ TEMPERATURE DATA (NASA)
nasa_year = []
nasa_temp = []
with open('./NASA_GISS.txt') as f:
	file = f.read()
for el in file.split("\n"):
	line = el.split()
	# print(line)
	try:
		int(line[0])
		nasa_year.append(float(line[0]))
		nasa_temp.append(0.01*float(line[13]))
	except:
		continue
nasa_year = nasa_year[:-1] #remove 2019 as not finished yet
tempmean1900 = np.mean(np.asarray(nasa_temp[106:125])) - 0.63
nasa_temp = [el-tempmean1900 for el in nasa_temp]

## READ TEMPERATURE DATA (NOAA)
with open("./Global_temp_NOAA.txt") as f:
	file = f.read()
noaa_year = []
noaa_temp = []
for line in file.split("\n")[5:-1]:
	noaa_year.append(float(line.split(",")[0]))
	noaa_temp.append(float(line.split(",")[1]))
tempmean1900 = np.mean(np.asarray(noaa_temp[106:125])) - 0.63
noaa_temp = [el-tempmean1900 for el in noaa_temp]
## compare  datasets (they are very similar, use HadCRUT)
# plt.figure()
# plt.plot(noaa_year, noaa_temp, label = 'NOAA' )
# plt.plot(nasa_year, nasa_temp, label = 'NASA')
# plt.plot(cru_year, cru_temp, label = 'HadCRUT')
# plt.figure()
# plt.plot(cru_year, cru_coverage)
# year = nasa_year
# temp = nasa_temp
# year = noaa_year
# temp = noaa_temp
year = cru_year
temp = cru_temp

## READ FORCING DATA (PIK)
file = pd.read_excel("./RCP3PD_MIDYEAR_RADFORCING.xls", 
	sheet_name="RCP3PD_MIDYEAR_RADFORCING")
timef = file['Unnamed: 0'][59:-1].reset_index()['Unnamed: 0'].to_list()
forcpd = file['Unnamed: 1'][59:-1].reset_index()['Unnamed: 1'].to_list()
file = pd.read_excel("./RCP45_MIDYEAR_RADFORCING.xls", 
	sheet_name="RCP45_MIDYEAR_RADFORCING")
forc45 = file['Unnamed: 1'][59:-1].reset_index()['Unnamed: 1'].to_list()
file = pd.read_excel("./RCP6_MIDYEAR_RADFORCING.xls", 
	sheet_name="RCP6_MIDYEAR_RADFORCING")
forc6 = file['Unnamed: 1'][59:-1].reset_index()['Unnamed: 1'].to_list()
file = pd.read_excel("./RCP85_MIDYEAR_RADFORCING.xls", 
	sheet_name="RCP85_MIDYEAR_RADFORCING")
forc85 = file['Unnamed: 1'][59:-1].reset_index()['Unnamed: 1'].to_list()
## check forcing - plot
# plt.figure()
# plt.plot([x for x in range(1765,2500)],forcpd)
# plt.plot([x for x in range(1765,2500)],forc45)
# plt.plot([x for x in range(1765,2500)],forc6)
# plt.plot([x for x in range(1765,2500)],forc85)
forcs = [forcpd, forc45, forc6, forc85]

## SELECT FORCING, START YEAR AND CREATE DICT WITH ALL NEEDED DATA
# select forcing to be used
forc = forc6
# define starting year
x=0
#create data dict
data = {'timef': timef, 'forc': forc, 'year': year, 'temp': temp}

tstep=1
## TRY THE MODELS
t_a, t_o, time = tmp_func.temp_model(data, x=x, tstep=tstep)

# compute DICE temperature model residuals
residuals = []
for el in enumerate(t_a):
	for el1,el2 in zip(year[x:],temp[x:]):
		if int(el1)==time[el[0]]:
			residuals.append(el2-t_a[el[0]]) 

# ## normality test
# ##qqplot
# qqplot(np.asarray(residuals), line='s')
# ## anderson test
# result = scipy.stats.anderson(np.asarray(residuals))
# print('Anderon-Darling test - Statistic: %.3f' % result.statistic)
# for i in range(len(result.critical_values)):
# 	sl, cv = result.significance_level[i], result.critical_values[i]
# 	if result.statistic < result.critical_values[i]:
# 		print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
# 	else:
# 		print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))
# ## d'agostino test
# stat, p = scipy.stats.normaltest(np.asarray(residuals))
# print("D'agostino test: s = %.3f, p-value = %.3f" % (stat, p))
# ## shapiro-wilk test
# stat, p = scipy.stats.shapiro(np.asarray(residuals))
# print("Shapiro-Wilk test: s = %.3f, p-value = %.3f" % (stat, p))
# mu = np.mean(residuals)
# sigma = np.var(residuals)**0.5
# print('variance of residuals = '+ str(sigma**2)+ ' & st.dev. = ', sigma)
# ## PLOT: residuals, their autocorrelation and distribution 
# fig, ax = plt.subplots(3, 1)
# ax[0].plot(year[x::tstep], residuals)
# ax[0].set_ylabel('Temperature [°C]', fontsize=fontsize)
# ax[0].set_xlabel('Year', fontsize=fontsize)
# b1 = tsaplots.plot_acf(residuals, ax=ax[1], lags=10, label='autocorr')
# ax[1].set_title('')
# ax[1].set_ylabel('Autocorrelation', fontsize=fontsize)
# ax[1].set_xlabel('Lags', fontsize=fontsize)
# normal = [(1/(2 * math.pi * sigma**2)**0.5) * \
# 	np.exp(-((x - mu)**2)/(2 * sigma**2)) \
# 	for x in np.arange(mu - 3*sigma, mu + 3*sigma,0.01)]
# sns.distplot(np.asarray(residuals), rug=1, 
# 	kde_kws={"lw": 3, "label": "KDE"}, hist_kws={"label": 'Histogram'}, 
# 	bins=int(np.log(len(residuals))/np.log(2)), ax=ax[2])
# ax[2].plot([x for x in np.arange(mu - 3*sigma, mu + 3*sigma, 0.01)], 
# 	normal, label='N(' + str(round(mu,4)) + ',' + str(round(sigma,4)) + ')')
# ax[2].legend(loc='upper left', fontsize=fontsize)
# ax[2].set_ylabel('Probability density', fontsize=fontsize)
# ax[2].set_xlabel('Temperature [°C]', fontsize=fontsize)
# fig.tight_layout()

## compute residuals after differencing residuals time series 
## using the auto regressive coefficient of the temperautree model
# ## dice model coefficients
# c1 = 0.1005
# c3 = 0.088
# c4 = 0.025
## use coefficients from Geoffroy model
c1 = 1 / 7.3
c3 = 0.73
c4 = 0.73/106
fco22x = 3.6813
t2xco2 = 3.1
lam = fco22x / t2xco2
residuals2 = np.asarray(residuals[1:]) - \
	((1 - c1*(lam + c3))**tstep)*np.asarray(residuals[:-1])
mu = np.mean(residuals2)
sigma = np.var(residuals2)**0.5
print('variance of residuals2 = '+ str(sigma**2)+ ' & st.dev. = ', sigma)
## PLOT 1: residuals, their autocorrelation and 
## distribution after autoregressive correction
fig, ax = plt.subplots(3, 1)
ax[0].plot(year[x:-1 - tstep + 1 :tstep], residuals2)
ax[0].set_ylabel('Temperature [°C]', fontsize=fontsize)
ax[0].set_xlabel('Year', fontsize=fontsize)
ax[0].tick_params( axis='both', labelsize=fontsize)
b1 = tsaplots.plot_acf(residuals2, ax=ax[1], label='autocorr')
ax[1].set_title('')
ax[1].set_ylabel('Autocorrelation', fontsize=fontsize)
ax[1].set_xlabel('Lags', fontsize=fontsize)
ax[1].tick_params( axis='both', labelsize=fontsize)
normal = [(1/(2 * math.pi * sigma**2)**0.5) * \
	np.exp(-((x - mu)**2)/(2 * sigma**2)) \
	for x in np.arange(mu - 3*sigma, mu + 3*sigma,0.01)]
sns.distplot(np.asarray(residuals2), rug=1, 
	kde_kws={"lw": 3, "label": "KDE"}, hist_kws={"label": 'Histogram'}, 
	bins=int(np.log(len(residuals2))/np.log(2)), ax=ax[2])
ax[2].plot([x for x in np.arange(mu - 3*sigma, mu + 3*sigma, 0.01)], 
	normal, label='N(' + str(round(mu,4)) + ',' + str(round(sigma,4)) + ')')
ax[2].legend(loc='upper left', fontsize=fontsize)
ax[2].set_ylabel('Probability density', fontsize=fontsize)
ax[2].set_xlabel('Temperature [°C]', fontsize=fontsize)
ax[2].tick_params( axis='both', labelsize=fontsize)
fig.tight_layout()
# fig.legend()

# ## Normality tests and check
# ## qqplot
# qqplot(np.asarray(residuals2), line='s')
# result = scipy.stats.anderson(np.asarray(residuals2))
# ## anderson darling test
# print('Anderon-Darling test - Statistic: %.3f' % result.statistic)
# for i in range(len(result.critical_values)):
# 	sl, cv = result.significance_level[i], result.critical_values[i]
# 	if result.statistic < result.critical_values[i]:
# 		print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
# 	else:
# 		print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))
# ## d'agostino test
# stat, p = scipy.stats.normaltest(np.asarray(residuals2))
# print("D'agostino test: s = %.3f, p-value = %.3f" % (stat, p))
# ## shapiro-wilk test
# stat, p = scipy.stats.shapiro(np.asarray(residuals2))
# print("Shapiro test: s = %.3f, p-value = %.3f" % (stat, p))

# #CHECK AUTOCORRELATION OF RESIDUALS
# print(diagnostic.acorr_ljungbox(np.asarray(residuals), lags=5))
# print(diagnostic.acorr_ljungbox(np.asarray(residuals2), lags=5))
# # RESIDUALS ARE AUTOCORRELATED
# # AR(1) REMOVES AUTOCORRELATION

# ## CHECK HETEROSCEDASTICITY OF RESIDUALS
# print(diagnostic.het_breuschpagan(np.asarray(residuals), 
#	 np.asarray([np.asarray(temp[:len(residuals)]), 
#	 np.asarray(forc[:len(residuals)])]).transpose()))
# print(diagnostic.het_breuschpagan(np.asarray(residuals2), 
#	 np.asarray([np.asarray(temp[:len(residuals2)]), 
#	 np.asarray(forc[:len(residuals2)])]).transpose()))
# # NO HOMOSCEDASTICITY -> BOXCOX?

## PLOT 2 : Model vs obs
t = time[:len(residuals)]
plt.figure()
# plt.plot(year, temp, 'k*-', linewidth=1, label='HadCRUT4 obs.')
plt.plot(year[x::tstep], temp[x::tstep], 
	'k*-', linewidth=1, label='HadCRUT4 obs.')
# plt.plot(nasa_year, nasa_temp, 'g*-', linewidth=1, label='NASA GISTEMP obs.')
# plt.plot(noaa_year, noaa_temp, 'b*-', linewidth=1, label='NOAA obs.')
plt.plot(time[:round((len(temp)-x)/tstep)],
	t_a[:round((len(temp)-x)/tstep)], 'r', linewidth=2, label='DICE model')
plt.xlabel('Year', fontsize=fontsize)
plt.ylabel("Temperature [°C]", fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.legend(fontsize=fontsize)
plt.tight_layout()

## PLOT  : RCP temperature with stochastic disturbance, tstep=5
tstep = 5
print('st. dev. with Dt= '+str(tstep)+' is '+str( ((sigma**2)*tstep)**0.5))
print('variance. with Dt= '+str(tstep)+' is '+str( ((sigma**2)*tstep)))
# plt.figure()
# color = ['limegreen','blue','darkorange','red']
# models = ['RCP2.6', 'RCP4.5', 'RCP6.0','RCP8.5']
# counter = 0
# for el in forcs:
# 	bound_max = []
# 	bound_min = []
# 	data = {'timef': timef, 'forc': el, 'year': year, 'temp': temp}
# 	# define starting year
# 	x=0
# 	plt.title('Stochastic temperature model under \
# 		historical and future RCPs forcing', fontsize=fontsize)
# 	vec = []
# 	t_a, t_o, time = tmp_func.temp_model(data, x=x, tstep=tstep)
# 	for el in range(0,100):
# 		t_as, t_os, times = tmp_func.temp_model_s(data, x=x, 
# 			tstep=tstep, sigma=sigma*(tstep**0.5), mul=False)
# 		vec.append(t_as)
# 		plt.plot(times,t_as, alpha=0.15, linewidth=0.3, color=color[counter])
# 	vec = np.asarray(vec).transpose()
# 	med = []
# 	lb = []
# 	ub = []
# 	ub90 = []
# 	lb90 = []
# 	for el in vec:
# 		med.append(np.mean(el))
# 		lb.append(min(el))
# 		ub.append(max(el))
# 		ub90.append(np.percentile(el,95))
# 		lb90.append(np.percentile(el,5))
# 	df = [med,lb,ub,ub90,lb90]
# 	plt.plot(times, med, label='mean '+models[counter], color=color[counter])
# 	plt.fill_between(times, lb90, ub90, edgecolor='black', 
# 		label='90% C.I. '+models[counter], alpha=0.2, color=color[counter])
# 	plt.plot(time, t_a, '--', label='deterministic '+\
# 		models[counter], color=color[counter])
# 	plt.xlabel("Year", fontsize=fontsize)
# 	plt.ylabel("Temperature [°C]", fontsize=fontsize)
# 	counter += 1
# plt.plot(year[x:], temp[x:], 'black', linewidth=1, label='HadCRUT obs.')
# plt.xticks(fontsize=fontsize)
# plt.yticks(fontsize=fontsize)
# plt.ylim(-2,13)
# plt.legend(loc='best', fontsize=fontsize-1)
# plt.tight_layout()

## PLOT 3 : RCP temperature with stochastic disturbance, tstep=1
plt.figure()
color = ['limegreen','blue','darkorange','red']
models = ['RCP2.6', 'RCP4.5', 'RCP6.0','RCP8.5']
counter = 0
tstep = 1
# sigma = (sigma/(5**0.5))
print('st. dev. with Dt= '+str(tstep)+' is '+str( sigma))
print('variance. with Dt= '+str(tstep)+' is '+str( ((sigma**2))))
for el in forcs:
	bound_max = []
	bound_min = []
	data = {'timef': timef, 'forc': el, 'year': year, 'temp': temp}
	# define starting year
	x=0
	# plt.title('Stochastic temperature model \
	#	under historical and future RCPs forcing', fontsize=fontsize)
	vec = []
	t_a, t_o, time = tmp_func.temp_model(data, x=x, tstep=tstep)
	## 100 scenarios simulated for each RCP
	for el in range(0,100):
		t_as, t_os, times = tmp_func.temp_model_s(data, x=x,
		 tstep=tstep, sigma=sigma)
		vec.append(t_as)
		plt.plot(times,t_as, alpha=0.15, linewidth=0.3,
		 color=color[counter])
	## get median and bounds for 90% c.i.
	vec = np.asarray(vec).transpose()
	med = []
	lb = []
	ub = []
	ub90 = []
	lb90 = []
	for el in vec:
		med.append(np.mean(el))
		lb.append(min(el))
		ub.append(max(el))
		ub90.append(np.percentile(el,95))
		lb90.append(np.percentile(el,5))
	df = [med,lb,ub,ub90,lb90]
	plt.plot(times, med, label='mean '+models[counter],
	 color=color[counter])
	plt.fill_between(times, lb90, ub90, edgecolor='black',
	 label='90% C.I. '+models[counter], alpha=0.2, color=color[counter])
	plt.plot(time, t_a, '--', label='deterministic '+\
		models[counter], color=color[counter])
	plt.xlabel("Year", fontsize=fontsize)
	plt.ylabel("Temperature [°C]", fontsize=fontsize)
	counter += 1
plt.plot(year[x:], temp[x:], 'black', linewidth=1, label='HadCRUT obs.')
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.ylim(-2,13)
# plt.plot(year[x:], noaa_temp[x:], 'pink', linewidth=2, label='NOAA obs.')
plt.legend(loc='best', fontsize=fontsize-1)
plt.tight_layout()

plt.show()