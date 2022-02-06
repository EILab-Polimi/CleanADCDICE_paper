import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import time
import seaborn as sns
from sklearn.neural_network import MLPRegressor
import sklearn
import sys
sys.path.append('/Users/angelocarlino/models/FAIR')
from fair.forward import fair_scm
from fair.RCPs import rcp26, rcp45, rcp60, rcp85
import scipy, matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42

import matplotlib.font_manager as font_manager
fontpath = '/System/Library/Fonts/Helvetica.ttc'
prop = font_manager.FontProperties(fname=fontpath, size='large')
matplotlib.rcParams['font.family'] = prop.get_name()
sns.set_context('paper')

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
	# TCR = 1.1 # central value AR6
	# ECS = 1.6 # central value AR6
	f2x = 3.71
	ds = 239.0
	df = 4.1
	tcr_dbl = np.log(2) / np.log(1.01)
	eff = 1.0
	ks = 1.0 - (ds/tcr_dbl)*(1.0 - np.exp(-tcr_dbl/ds))
	kf = 1.0 - (df/tcr_dbl)*(1.0 - np.exp(-tcr_dbl/df))
	qs = (1.0/f2x) * (1.0/(ks - kf)) * (TCR - ECS * kf)
	qf = (1.0/f2x) * (1.0/(ks - kf)) * (ECS * ks - TCR)
	qs = 0.33
	qf = 0.41
	with open('./RFoth_26.txt', 'r') as f:
		forc_oth = [float(x) for x in f.read().split('\n')[:-1]]

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


## carbon class
class FAIR():
	def __init__(self, horizon=100, mode='orig', ann=None, annprms = None, ranges=None, rf_oth=None):
		self.mode = mode
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
			if x[0] < 0 or x[1] < 0:
				print(x, [self.vars.c_acc[self.t], self.vars.gmst[self.t]])
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
		self.vars.gmst[self.t+1] = self.vars.tf[self.t+1] + self.vars.ts[self.t+1]
		self.t += 1

# ## simulate combined climate and carbon models
def simulateFAIR(emiss, mode='orig', horizon=100, ann=None, annprms=None, ranges=None, rf_oth=None):
	fair = FAIR(mode=mode, horizon=horizon, ann=ann, annprms=annprms, ranges=ranges, rf_oth=rf_oth)
	for t in range(horizon):
		fair.next_step(emiss)
	return fair.vars.c, fair.vars.gmst, fair.y, fair.vars.alpha, fair.vars.c_acc, fair.vars.forc, fair.params.forc_oth


rcps = [rcp26, rcp45, rcp60, rcp85]
rcps = [rcp26, rcp45, rcp60, rcp85]

y = [y for y in range(1765,2501)]
# fig, ax = plt.subplots(4,1, sharex=True)
# for el in rcps:
# 	emissions = el.Emissions.emissions
# 	[C, F, T] = fair_scm(emissions=emissions)
# 	ax[0].plot(y, T, label='orig')
# 	ax[1].plot(y, np.sum(F, axis=1), label='orig')
# 	ax[2].plot(y, C[:,0] , label='orig')
# 	emiss = list(np.sum(emissions[:,1:3],axis=1).flatten())
# 	rf_oth = list(np.sum(F[:,1:], axis=1) )
# 	other_rf = np.array(np.sum(F[:,1:], axis=1))
# 	[C_co2, F_co2, T_co2] = fair_scm(emissions=np.sum(emissions[:,1:3], axis=1).flatten(), useMultigas=False, other_rf=other_rf)
# 	[mat_mod, tatm_mod, y_mod, alpha_mod, c_acc_mod, forc_mod, forc_oth_mod] = simulateFAIR(emiss, 
# 		horizon=2500-1765, mode='orig', rf_oth = rf_oth)
# 	[mat_ann, tatm_ann, y_mod, alpha_ann, c_acc_ann, forc_ann, forc_oth_ann] = simulateFAIR(emiss, 
# 		horizon=2500-1765, mode='ann-old', rf_oth = rf_oth)
# 	ax[0].plot(y, T_co2, linestyle='dashdot', label='origco2')
# 	ax[0].plot(y_mod, tatm_mod,'--', label='mod')
# 	ax[0].plot(y_mod, tatm_ann, linestyle='dotted')
# 	ax[1].plot(y, F_co2, linestyle='dashdot', label='origco2')
# 	ax[1].plot(y_mod, forc_mod + rf_oth,'--', label='mod')
# 	ax[1].plot(y_mod, forc_ann + rf_oth, linestyle='dotted')
# 	ax[2].plot(y, C_co2, linestyle='dashdot', label='origco2')
# 	ax[2].plot(y_mod, mat_mod , '--', label='mod')
# 	ax[2].plot(y_mod, mat_ann , linestyle='dotted')
# 	ax[3].plot(y, emiss)


def compute_error_ann_FAIR(annparams, rcps, ranges, C, T, emiss, rf_oth):
	error = 0.0
	for el in range(len(rcps)):
		# emissions = el.Emissions.emissions
		# # run original model
		# [C, F, T] = fair_scm(emissions=emissions)
		# # use emissions and other forcing from original model
		# emiss = list(np.sum(emissions[:,1:3],axis=1).flatten())
		# rf_oth = list(np.sum(F[:,1:], axis=1) )
		# other_rf = np.array(np.sum(F[:,1:], axis=1))
		# run only co2 original model
		# [C_co2, F_co2, T_co2] = fair_scm(emissions=np.sum(emissions[:,1:3], axis=1).flatten(), useMultigas=False, other_rf=other_rf)
		# run new ANN model
		[mat_mod, tatm_mod, y_mod, alpha_mod, c_acc_mod, forc_mod, forc_oth_mod] = simulateFAIR(emiss[el], 
			horizon=2500-1765, mode='ann', annprms = annparams, ranges=ranges, rf_oth = rf_oth[el])
		error += np.sum((C[el] - mat_mod)**2)
		# error += np.sum((T[el] - tatm_mod)**2)
	return error

## calibrate new ANN
ranges = [0.0, 2000.0, -1.0, 10.0]
## need to write this function to train ANN directly using bgfs or evolutionary methods

emissions = []
emiss = []
rf_oth = []
other_rf = []
C = []
F = []
T = []
C_co2 = []
F_co2 = []
T_co2 = []
for el in rcps:
	emissions.append(el.Emissions.emissions)
	# run original model
	[c, f, temp] = fair_scm(emissions=emissions[-1])
	# print(np.mean(temp[1980-1765:2010-1765]))
	C.append(c)
	F.append(f)
	T.append(temp)
	rf_oth.append(list(np.sum(F[-1][:,1:], axis=1) ) )
	emiss.append( np.sum(emissions[-1][:,1:3],axis=1).flatten()) 
	other_rf.append( np.array(np.sum(F[-1][:,1:], axis=1))) 
	[c_co2, f_co2, temp_co2] = fair_scm(emissions=emiss[-1], useMultigas=False, other_rf=other_rf[-1])
	C_co2.append(c_co2)
	F_co2.append(f_co2)
	T_co2.append(temp_co2)
	# use emissions and other forcing from original model


bnds = ((-1000,1000),(-1000,1000),(-6,6),(-6,6),(-6,6),(-1000,1000),(-6,6),(-6,6),(-6,6))

## to obtain the params below, uncomment the lines below the params to run the calibration

annprms = [-6.73031424e+02, 2.12637160e+02, -4.21588222e+00, -6.81077625e-01,
 	6.00000000e+00, 8.89025716e+02, 3.22741020e+00, 3.98104388e-01,
 	-9.67139675e-01] ## obtained minimizing error w.r.t original tempererature
annprms = [-6.66006035e+02, 2.09443154e+02, -4.83968920e+00, 2.31243377e+00,
	2.75031497e+00, 8.89902682e+02, 2.40146799e+00, 6.83316702e-02, 
	2.89753011e-02] ## obtained minimizing error w.r.t co2 only carbon concentration


# init = np.ones((15,9))
# init[0] = np.array(annprms)
# for el in range(1,15):
# 	init[el] = np.random.rand(9)
# res = scipy.optimize.differential_evolution(compute_error_ann_FAIR, bnds, 
# 	args=(rcps, ranges, C_co2, T, emiss, rf_oth), maxiter=1000, 
# 	popsize=15, tol=0.01, mutation=(0.5, 1), recombination=0.7, 
# 	seed=None, callback=None, disp=True, polish=True, 
# 	init=init, atol=0, updating='immediate', 
# 	workers=1)
# print(res.x)
# annprms = res.x

colors = ['navy','lightblue','orange','firebrick']
colors = ['#1F4C8A','#42BCF4','#F26419','#B80C09']
final_error = [0.0 for x in range(4)]
fig, ax = plt.subplots(4,1, sharex=True, figsize=(5,6))
count = 0
for el in rcps:
	emissions = el.Emissions.emissions
	[C, F, T] = fair_scm(emissions=emissions)
	ax[0].plot(y, T, label='orig', color=colors[count], linestyle='dotted')
	ax[1].plot(y, np.sum(F, axis=1), color=colors[count], linestyle='dotted')
	ax[2].plot(y, C[:,0], color=colors[count] , linestyle='dotted')
	emiss = list(np.sum(emissions[:,1:3],axis=1).flatten())
	rf_oth = list(np.sum(F[:,1:], axis=1) )
	other_rf = np.array(np.sum(F[:,1:], axis=1))
	[C, F, T] = fair_scm(emissions=np.sum(emissions[:,1:3], axis=1).flatten(), useMultigas=False, other_rf=other_rf)
	[mat_mod, tatm_mod, y_mod, alpha_mod, c_acc_mod, forc_mod, forc_oth_mod] = simulateFAIR(emiss, 
		horizon=2500-1765, mode='orig', rf_oth = rf_oth)
	[mat_ann, tatm_ann, y_mod, alpha_ann, c_acc_ann, forc_ann, forc_oth_ann] = simulateFAIR(emiss, 
		horizon=2500-1765, mode='ann', annprms = annprms, ranges=ranges, rf_oth = rf_oth)
	ax[0].plot(y, T, linestyle='dashdot', label='orig-co2_only', color=colors[count])
	ax[0].plot(y_mod, tatm_mod,'--', label='mod', color=colors[count])
	ax[0].plot(y_mod, tatm_ann, label='mod-ann', color=colors[count])
	ax[1].plot(y, F, linestyle='dashdot', color=colors[count])
	ax[1].plot(y_mod, forc_mod + rf_oth,'--', color=colors[count])
	ax[1].plot(y_mod, forc_ann + rf_oth, color=colors[count])
	ax[2].plot(y, C, linestyle='dashdot', color=colors[count])
	ax[2].plot(y_mod, mat_mod , '--', color=colors[count])
	ax[2].plot(y_mod, mat_ann , color=colors[count])
	ax[3].plot(y, emiss, color=colors[count], linestyle='dotted')
	ax[0].set_ylabel('GMST [°C]')
	ax[1].set_ylabel('Forcing [W/m^2]')
	ax[2].set_ylabel('CO2 conc. [ppmv]')
	ax[3].set_ylabel('CO2 emissions [GtC]')
	ax[3].set_xlabel('Year')
	## compute approximation error
	error = np.mean([abs(a - b) for a, b in zip(C, mat_ann)])
	final_error[0] = max(error, final_error[0])
	error = np.mean([abs(a - b) for a, b in zip(T, tatm_ann)])
	final_error[1] = max(error, final_error[1])
	error = max([abs(a - b) for a, b in zip(C, mat_ann)])
	final_error[2] = max(error, final_error[2])
	error = max([abs(a - b) for a, b in zip(T, tatm_ann)])
	final_error[3] = max(error, final_error[3])

	count += 1
handles, labels = ax[0].get_legend_handles_labels()
handles = [x for x in handles[:4]]
labels = [x for x in labels[:4]]
for x in range(len(handles)):
	handles[x].set_color('k')
rcpss = ['RCP2.6','RCP4.5','RCP6.0','RCP8.5']
for el in range(4):
	handles.append(matplotlib.patches.Patch(color=colors[el]))
	labels.append(rcpss[el])
print('Mean absolute error on carbon concentration & tempetature; max absolute error on carbon concentration & tempetature:')
print(final_error)
fig.legend(handles, labels, ncol=2, loc='lower center')
plt.tight_layout()
fig.subplots_adjust(bottom=0.225, hspace=0.3)
plt.show()
