import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import time, sklearn, sys, math, scipy, matplotlib
import seaborn as sns
from sklearn.neural_network import MLPRegressor
import pandas as pd
sys.path.append('/Users/angelocarlino/models/FAIR')
from fair.forward import fair_scm
from fair.RCPs import rcp26, rcp45, rcp60, rcp85
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.font_manager as font_manager
fontpath = '/System/Library/Fonts/Helvetica.ttc'
prop = font_manager.FontProperties(fname=fontpath, size='large')
matplotlib.rcParams['font.family'] = prop.get_name()
sns.set_context('paper')
from statsmodels.graphics import tsaplots
from statsmodels.tsa.ar_model import AutoReg
# from statsmodels.stats import diagnostic
# from statsmodels.tsa.arima_model import ARMA
# from statsmodels.graphics.gofplots import qqplot


## compute state-dependent absorption of carbon reservoirs
def alphaeq(alpha, iirf, fraction, t_scale):
	return min(97.0 , iirf) - np.sum([alpha*fraction[box]*t_scale[box] * \
		(1 - np.exp(-100/(alpha * t_scale[box]))) for box in range(4)])

## carbon model params (FAIRsettings.txt)
class FAIRParams():
	r0       = 35.0
	rc       = 0.019
	rt       = 4.165
	iirf_h   = 100.0
	iirf_max = 97.0
	ppm_to_gtc = 2.124
	a = np.asarray([0.2173, 0.2240, 0.2824, 0.2763])
	tau = np.asarray([1000000, 394.4, 36.54, 4.304])
	TCR = 1.6
	ECS = 2.75
	TCR = 1.8 # central value AR6
	ECS = 3.0 # central value AR6
	# TCR = 1.2 # lower bound very likely AR6
	# ECS = 2.0 # lower bound very likely AR6
	# TCR = 2.4 # upper bound very likely AR6
	# ECS = 5.0 # upper bound very likely AR6
	f2x = 3.71
	ds = 239.0
	df = 4.1
	# ds = 333.0 	# AR6 fair parameters
	# df = 4.6 	# AR6 fair parameters
	tcr_dbl = np.log(2) / np.log(1.01)
	ks = 1.0 - (ds/tcr_dbl)*(1.0 - np.exp(-tcr_dbl/ds))
	kf = 1.0 - (df/tcr_dbl)*(1.0 - np.exp(-tcr_dbl/df))
	qs = (1.0/f2x) * (1.0/(ks - kf)) * (TCR - ECS * kf)
	qf = (1.0/f2x) * (1.0/(ks - kf)) * (ECS * ks - TCR)
	# with open('./RFoth_26.txt', 'r') as f:
	# 	forc_oth = [float(x) for x in f.read().split('\n')[:-1]]

class FAIRVariables():
	def __init__(self, horizon):
		self.iirf = np.asarray([0.0 for x in range(horizon+1)])
		self.alpha = np.asarray([0.0 for x in range(horizon+1)])
		self.carbon_boxes = np.asarray([[0.0 for x in range(4)] for t in range(horizon+1)])
		self.carbon_boxes[0] = [58.71337292, 43.28685286, 18.44893718, 3.81581747]
		self.carbon_boxes[0] = [0.,0.,0.,0.]
		self.c_pi = 278.0
		self.c = np.asarray([0.0 for x in range(horizon+1)])
		self.c[0] = self.c_pi + np.sum(self.carbon_boxes[0])
		self.c_acc = np.asarray([0.0 for x in range(horizon+1)])
		self.c_acc[0] = 305.1673751542558
		self.c_acc[0] = 0.0
		self.forc = np.asarray([0.0 for x in range(horizon+1)])
		self.ts = np.asarray([0.0 for x in range(horizon+1)])
		self.ts[0] = 0.11759814
		self.ts[0] = 0.0
		self.tf = np.asarray([0.0 for x in range(horizon+1)])
		self.tf[0] = 1.02697844
		self.tf[0] = 0.0
		self.gmst = np.asarray([0.0 for x in range(horizon+1)])
		self.gmst[0] = self.tf[0] + self.ts[0]
		self.noise = np.asarray([0.0 for x in range(horizon+1)])
		self.noise[0] = 0.0

## carbon class
class FAIR():
	def __init__(self, horizon=100, mode='orig', ann=None, 
		annprms = None, ranges=None, rf_oth=None, stdev=None, ar=[1.0]):
		self.mode = mode
		self.stdev = stdev
		self.ar = ar
		self.params = FAIRParams()
		self.vars = FAIRVariables(horizon)
		if rf_oth is not None:
			self.params.forc_oth = rf_oth
		while len(self.params.forc_oth) <= horizon:
			self.params.forc_oth.append(self.params.forc_oth[-1])
		self.nn = ann
		self.nnprms = annprms
		self.ranges = ranges
		self.t = 0
		self.y = np.asarray([2015+x for x in range(horizon+1)])
		self.y = np.asarray([1765+x for x in range(horizon+1)])
	# simulate one step ahead
	def next_step(self, emiss):
		self.vars.iirf[self.t] = self.params.r0 + \
			self.params.rc * self.vars.c_acc[self.t] + \
			self.params.rt * self.vars.gmst[self.t]
		if self.mode=='orig':
			try:
				self.vars.alpha[self.t] = opt.root_scalar(alphaeq, \
				bracket = [1e-16, 1e10], xtol=1e-16, rtol=1e-10, \
				args=(self.vars.iirf[self.t], \
				self.params.a, self.params.tau)).root
				# self.alpha[self.t] = min(self.alpha[self.t], 1000)
			except ValueError:
				self.vars.alpha[self.t] = 1e-8

		elif self.mode=='ann-old':
			inputval = [0.0 for x in range(2)]
			inputval[0] = max(0.0, (self.vars.c_acc[self.t] +\
				 - 30.966999999999985) / (620.967 - 30.966999999999985) )
			inputval[1] = max(0.0, (self.vars.gmst[self.t] - 1.0)/(3.95-1.0))
			self.vars.alpha[self.t] = 0.4224476755742342 + \
			    (-2.24079509) * \
			    	(-1.0 + 2.0 / \
			        	( 1.0 + np.exp( -2.0 * \
			          	(-1.1679476786651246 + \
			            	(-0.5497803029711411) * inputval[0] + \
			            	(-0.6082563253131715) * inputval[1] )))) +\
			    (2.10715655) * \
			    	(-1.0 + 2.0 / \
			        	( 1.0 + np.exp( -2.0 * \
			        	(-1.9221811095464068 + \
			            	(0.8797355517352923) * inputval[0] + \
			            	(0.9631872008727567) * inputval[1] ))))

		elif self.mode=='ann':
			x = [self.vars.c_acc[self.t], self.vars.gmst[self.t]]
			count = 0
			for idx in range(len(x)):
				x[count] = (x[count] - self.ranges[count*2]) / \
					(self.ranges[count*2+1] - self.ranges[count*2])
				count += 1
			self.vars.alpha[self.t] = self.nnprms[0] + \
			    (self.nnprms[1]) * \
			    	(-1.0 + 2.0 / \
			        	( 1.0 + np.exp( -2.0 * \
			          	(self.nnprms[2] + \
			            	(self.nnprms[3]) * x[0] + \
			            	(self.nnprms[4]) * x[1] )))) +\
			    (self.nnprms[5]) * \
			    	(-1.0 + 2.0 / \
			        	( 1.0 + np.exp( -2.0 * \
			        	(self.nnprms[6] + \
			            	(self.nnprms[7]) * x[0] + \
			            	(self.nnprms[8]) * x[1] ))))
			self.vars.alpha[self.t] = max(1e-10, self.vars.alpha[self.t])

		# elif self.mode=='ann':
		# 	x = [self.vars.c_acc[self.t], self.vars.gmst[self.t]]
		# 	count = 0
		# 	for idx in range(len(x)):
		# 		x[count] = (x[count] - self.ranges[count*2]) / \
		# 			(self.ranges[count*2+1] - self.ranges[count*2])
		# 		count += 1
		# 	self.vars.alpha[self.t] = max(1e-6, np.exp(self.nn.predict([x])[0]) ) 
		else:
			print('Please use either "orig" or "ann" mode')

		if self.t == 0:
			self.vars.carbon_boxes[self.t] = + \
				self.params.a * emiss[self.t] / self.params.ppm_to_gtc
			self.vars.c[self.t] = self.vars.c_pi + np.sum(self.vars.carbon_boxes[self.t])
			# self.vars.c_acc[self.t] = self.vars.c_acc[self.t] + emiss[self.t] + emiss[self.t])/2 + \
			# 	- (self.vars.c[self.t+1] - self.vars.c[self.t]) * self.params.ppm_to_gtc

			self.vars.forc[self.t] = self.params.f2x / np.log(2) * \
				np.log(self.vars.c[self.t] / self.vars.c_pi)
			self.vars.ts[self.t] = self.vars.ts[self.t] * np.exp(-1.0/self.params.ds) + \
				self.params.qs*(1.0-np.exp(-1.0/self.params.ds)) * (self.vars.forc[self.t] + self.params.forc_oth[self.t])
			self.vars.tf[self.t] = self.vars.tf[self.t] * np.exp(-1.0/self.params.df) + \
				self.params.qf*(1.0-np.exp(-1.0/self.params.df)) * (self.vars.forc[self.t] + self.params.forc_oth[self.t])
			self.vars.gmst[self.t] = self.vars.tf[self.t] + self.vars.ts[self.t]


		self.vars.carbon_boxes[self.t+1] = \
			self.vars.carbon_boxes[self.t] * \
			np.exp(-1.0 / (self.params.tau * self.vars.alpha[self.t]) ) + \
			self.params.a * (emiss[self.t+1] + emiss[self.t])/2 / self.params.ppm_to_gtc
		self.vars.c[self.t+1] = self.vars.c_pi + np.sum(self.vars.carbon_boxes[self.t+1])
		self.vars.c_acc[self.t+1] = self.vars.c_acc[self.t] + (emiss[self.t+1] + emiss[self.t])/2 + \
			- (self.vars.c[self.t+1] - self.vars.c[self.t]) * self.params.ppm_to_gtc

		self.vars.forc[self.t+1] = self.params.f2x / np.log(2) * \
			np.log(self.vars.c[self.t+1] / self.vars.c_pi)
		self.vars.ts[self.t+1] = self.vars.ts[self.t] * np.exp(-1.0/self.params.ds) + \
			self.params.qs*(1.0-np.exp(-1.0/self.params.ds)) * (self.vars.forc[self.t+1] + self.params.forc_oth[self.t+1])
		self.vars.tf[self.t+1] = self.vars.tf[self.t] * np.exp(-1.0/self.params.df) + \
			self.params.qf*(1.0-np.exp(-1.0/self.params.df)) * (self.vars.forc[self.t+1] + self.params.forc_oth[self.t+1])
		if self.stdev != None:
			self.vars.noise[self.t] = np.random.randn()*self.stdev
			self.vars.tf[self.t+1] = self.vars.tf[self.t] * np.exp(-1.0/self.params.df) + \
				self.params.qf*(1.0-np.exp(-1.0/self.params.df)) * \
				(self.vars.forc[self.t+1] + self.params.forc_oth[self.t+1])	+ \
				np.sum([self.ar[x] * self.vars.noise[self.t-x] for x in range(len(self.ar)) if (self.t-x)>=0])

		self.vars.gmst[self.t+1] = self.vars.tf[self.t+1] + self.vars.ts[self.t+1]
		self.t += 1

# ## simulate combined climate and carbon models
def simulateFAIR(emiss, mode='orig', horizon=100, 
	ann=None, annprms=None, ranges=None, rf_oth=None, stdev=None, ar=[1.0]):
	fair = FAIR(mode=mode, horizon=horizon, ann=ann,
		annprms=annprms, ranges=ranges, rf_oth=rf_oth, stdev=stdev, ar=ar)
	for t in range(horizon):
		fair.next_step(emiss)
	return fair.vars.c, fair.vars.gmst, fair.y, fair.vars.alpha, fair

class oneStepSim():
	TCR = 1.6
	ECS = 2.75

	f2x = 3.71
	ds = 239.0
	df = 4.1
	tcr_dbl = np.log(2) / np.log(1.01)
	ks = 1.0 - (ds/tcr_dbl)*(1.0 - np.exp(-tcr_dbl/ds))
	kf = 1.0 - (df/tcr_dbl)*(1.0 - np.exp(-tcr_dbl/df))
	qs = (1.0/f2x) * (1.0/(ks - kf)) * (TCR - ECS * kf)
	qf = (1.0/f2x) * (1.0/(ks - kf)) * (ECS * ks - TCR)
	def setParams(self, params):
		self.TCR = params[0]
		self.ECS = params[1]
		self.qs = params[2]
		self.qf = params[3]
		self.ds = params[4]
		self.df = params[5]
	def step(self, ts, tf, forctot):
		ts = ts * np.exp(-1.0/self.ds) + \
			self.qs*(1.0-np.exp(-1.0/self.ds)) * forctot
		tf = tf * np.exp(-1.0/self.df) + \
			self.qf*(1.0-np.exp(-1.0/self.df)) * forctot
		# gmst = tf + ts
		return ts, tf

## READ TEMPERATURE DATA (HadCRUT)
hadcrut5 = pd.read_csv('./HadCRUT_5_0_1_0_analysis_summary_series_global_annual.csv')
cru_year = hadcrut5.loc[hadcrut5['Time']<2021]['Time'].tolist()
cru_temp = hadcrut5.loc[hadcrut5['Time']<2021]['Anomaly (deg C)'].tolist()

tempmean18501900 = np.mean(np.asarray(cru_temp[0:50]))  
cru_temp = [el-tempmean18501900 for el in cru_temp]

year = cru_year
temp = cru_temp

y = [y for y in range(1765,2501)]

emissions = []
C = []
F = []
T = []
emiss = []
rf_oth = []
other_rf = []
ranges = [0.0, 2000.0, -1.0, 10.0]
fig, ax = plt.subplots(2,1)
emissions.append(rcp26.Emissions.emissions)
[c, f, t] = fair_scm(emissions=emissions[-1])
C.append(c)
F.append(f)
T.append(t)
rf_oth.append(list(np.sum(F[-1][:,1:], axis=1) ) )
emiss.append( np.sum(emissions[-1][:,1:3],axis=1).flatten()) 
other_rf.append( np.array(np.sum(F[-1][:,1:], axis=1))) 
annprms = [-6.66006035e+02, 2.09443154e+02, -4.83968920e+00, 2.31243377e+00,
	2.75031497e+00, 8.89902682e+02, 2.40146799e+00, 6.83316702e-02, 
	2.89753011e-02] ## obtained minimizing error w.r.t co2 only carbon concentration
[mat_ann, tatm_ann, y_mod, alpha_ann, fair] = simulateFAIR(emiss[-1], 
	horizon=2500-1765, mode='ann', annprms = annprms, ranges=ranges, rf_oth = rf_oth[-1])

ts = fair.vars.ts
tf = fair.vars.tf
forctot = fair.vars.forc + fair.params.forc_oth
params = [fair.params.TCR,fair.params.ECS, fair.params.qs,fair.params.qf, fair.params.ds, fair.params.df]
simulator = oneStepSim()
simulator.setParams(params)
error = []
offset = year[0]-y_mod[0]
for y in range(len(year[:-15])):
	tfs = temp[y] - ts[y+offset]
	tss, tfs = simulator.step(ts[y+offset], tfs, forctot[y+offset])
	error.append(temp[y+1] - (tfs+tss) )
ax[0].plot(y_mod, tatm_ann)
ax[0].plot(year, temp)
ax[1].plot(year[:-15], error)
print(np.mean(error),np.std(error))

error = error - np.mean(error)

fontsize=12
fig, ax = plt.subplots(4, 1)
ax[0].plot(year[:-15], error)
ax[0].set_ylabel('Temperature [°C]')
ax[0].set_xlabel('Year')
ax[0].tick_params( axis='both')
b1 = tsaplots.plot_acf(np.asarray(error), ax=ax[1], label='autocorr')
ax[1].set_title('')
ax[1].set_ylabel('Autocorrelation')
ax[1].set_xlabel('Lags')
ax[1].tick_params( axis='both')
b2 = tsaplots.plot_pacf(np.asarray(error), ax=ax[2], label='autocorr')
ax[2].set_title('')
ax[2].set_ylabel('Partial Autocorrelation')
ax[2].set_xlabel('Lags')
ax[2].tick_params( axis='both')
mu, sigma = np.mean(error), np.var(error)**0.5
normal = [(1/(2 * math.pi * sigma**2)**0.5) * \
	np.exp(-((x - mu)**2)/(2 * sigma**2)) \
	for x in np.arange(mu - 3*sigma, mu + 3*sigma,0.01)]
sns.distplot(np.asarray(error), rug=1, 
	kde_kws={"lw": 3, "label": "KDE"}, hist_kws={"label": 'Histogram'}, 
	bins=int(np.log(len(error))/np.log(2)), ax=ax[3])
ax[3].plot([x for x in np.arange(mu - 3*sigma, mu + 3*sigma, 0.01)], 
	normal, label='N(' + str(round(mu,4)) + ',' + str(round(sigma,4)) + ')')
ax[3].legend(loc='upper left')
ax[3].set_ylabel('Probability density')
ax[3].set_xlabel('Temperature [°C]')
ax[3].tick_params( axis='both')
fig.tight_layout()

for el in range(5):
	ar_model = AutoReg(error, lags=el).fit()
	print(el, ar_model.aic, ar_model.bic )
	print(np.std(ar_model.resid), ar_model.params)
	ar_model.plot_diagnostics()
	plt.suptitle(el)

	ar = [1.0]
	for x in range(1,el+1):
		ar.append(ar_model.params[x])
	stdev = np.std(ar_model.resid)
	fig, ax = plt.subplots(2,1)
	for el in range(30):
		[mat_ann_, tatm_ann_, y_mod_, alpha_ann_, fair_] = simulateFAIR(emiss[-1], 
		horizon=2500-1765, mode='ann', annprms = annprms, ranges=ranges, rf_oth = rf_oth[-1], stdev=stdev, ar=ar)
		ts = fair_.vars.ts
		tf = fair_.vars.tf
		forctot = fair_.vars.forc + fair_.params.forc_oth
		params = [fair_.params.TCR, fair_.params.ECS, fair_.params.qs, fair_.params.qf, fair_.params.ds, fair_.params.df]
		simulator = oneStepSim()
		simulator.setParams(params)
		error_ = []
		offset = year[0]-y_mod[0]
		for y in range(len(year[:-15])):
			tfs = temp[y] - ts[y+offset]
			tss, tfs = simulator.step(ts[y+offset], tfs, forctot[y+offset])
			error_.append(temp[y+1] - (tfs + min(3, max(-3,np.random.randn()))*sigma + tss) )
		ax[0].plot(y_mod_, tatm_ann_, '--', alpha=0.5)#, linewidth=0.5)
		ax[0].set_ylabel('°C')
		ax[0].set_xlim((1850,2150))
		ax[1].plot(year[:-15], error_)

	ax[0].plot(y_mod, tatm_ann,'r-')
	ax[0].plot(year, temp,'k-')
	ax[1].plot(year[:-15], error, 'k-')


for el in range(1):
	ar_model = AutoReg(error, lags=el).fit()
	print(el, ar_model.aic, ar_model.bic )
	print(np.std(ar_model.resid), ar_model.params)
	ar_model.plot_diagnostics()
	plt.suptitle(el)

	ar = [1.0]
	for x in range(1,el+1):
		ar.append(ar_model.params[x])
	stdev = np.std(ar_model.resid)
	fig, ax = plt.subplots(1,1)
	for el in range(30):
		[mat_ann_, tatm_ann_, y_mod_, alpha_ann_, fair_] = simulateFAIR(emiss[-1], 
		horizon=2500-1765, mode='ann', annprms = annprms, ranges=ranges, rf_oth = rf_oth[-1], stdev=stdev, ar=ar)
		ts = fair_.vars.ts
		tf = fair_.vars.tf
		forctot = fair_.vars.forc + fair_.params.forc_oth
		params = [fair_.params.TCR, fair_.params.ECS, fair_.params.qs, fair_.params.qf, fair_.params.ds, fair_.params.df]
		simulator = oneStepSim()
		simulator.setParams(params)
		error_ = []
		offset = year[0]-y_mod[0]
		for y in range(len(year[:-15])):
			tfs = temp[y] - ts[y+offset]
			tss, tfs = simulator.step(ts[y+offset], tfs, forctot[y+offset])
			error_.append(temp[y+1] - (tfs + np.random.randn()*sigma + tss) )
		ax.plot(y_mod_, tatm_ann_, '--', alpha=0.5)#, linewidth=0.5)
		ax.set_ylabel('°C')
		ax.set_xlim((1850,2150))
# 		# ax[1].plot(year[:-15], error_)

	ax.plot(y_mod, tatm_ann,'r-')
	ax.plot(year, temp,'k-')
	ax.set_ylabel('GMST [°C]')
	ax.set_xlabel('Year')
ax.set_ylim((-0.7,2.4))
# 	# ax[1].plot(year[:-15], error, 'k-')
plt.show()