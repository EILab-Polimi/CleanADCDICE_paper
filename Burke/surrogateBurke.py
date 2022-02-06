import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils
import scipy
import sklearn
from sklearn.neural_network import MLPRegressor
import time
from mpl_toolkits.mplot3d import axes3d
import os, matplotlib, subprocess, time, chart_studio, math
import seaborn as sns
import matplotlib.font_manager as font_manager

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

## load dataset from Burke model simulation
dataset = pd.read_csv('./Burke_dataset.csv')
gtemp = np.reshape(dataset['gtemp'].values, 
	(len(dataset['gtemp'].values)//((2100-2010)//5 + 1), -1))
damages = np.reshape(dataset['damages'].values, 
	(len(dataset['damages'].values)//((2100-2010)//5 + 1), -1))

fig, ax = plt.subplots()
obs = ax.scatter(np.transpose(gtemp)[:-1],np.transpose(damages)[:-1],
	 c=np.transpose(damages)[1:], vmin=0, vmax=0.4, s=4)
ax.set_xlabel('Temperature [°C]')
ax.set_ylabel('Damages at time t [%GWP]')
cbar = fig.colorbar(obs)
cbar.set_label('Damages at time t+1 [%GWP]')
fig, ax = plt.subplots()
sns.scatterplot(x=np.transpose(gtemp)[:-1].flatten(), y=np.transpose(damages)[:-1].flatten(),
	 hue=np.transpose(damages)[1:].flatten() - np.transpose(damages)[:-1].flatten(), palette='plasma_r')

## Create training dataset for ANN
x, y = [], []
gtemp_norm = gtemp
gtemp_norm = (gtemp - 14)/(25-14)
for el in zip(gtemp_norm, damages):
	for el2 in zip(el[0][:-1], el[1][:-1], el[1][1:]):
		x.append(el2[:2])
		y.append(el2[-1])
## fit using ANN
nn = sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(2,),
 activation='tanh', solver='adam', learning_rate='adaptive', 
 random_state=3, n_iter_no_change=1000, tol=1e-18, max_iter=100000) 
start = time.time()
nn = nn.fit(x, y)
elapsed = time.time() - start
# print(nn.coefs_, nn.intercepts_)
# print("Elapsed time: " + str(elapsed) + ' seconds')

## plot result (dataset on right, result on the left)
fig, ax = plt.subplots(1,2, sharey=True)
obs = ax[0].scatter(14 + np.transpose(x)[0]*(25-14), np.transpose(x)[1],
 c=y, vmin=0, vmax=0.5, s=4)
model = ax[1].scatter(14 + np.transpose(x)[0]*(25-14), np.transpose(x)[1],
 c = nn.predict(x), vmin=0, vmax=0.5, s=4)
fig.subplots_adjust(bottom=0.25)
cax = fig.add_axes([0.15, 0.12, 0.7, 0.03])
cbar = fig.colorbar(obs, cax=cax, orientation='horizontal')
[el.set_xlabel('Temperature [°C]') for el in ax]
ax[0].set_ylabel('Damages at time t [%GWP]')
cbar.set_label('Damages at time t+1 [%GWP]')
## plot percentage error
fig, ax = plt.subplots(1,1)
err = ax.scatter(np.transpose(x)[0], np.transpose(x)[1],
 c = 100 * np.abs( (y - nn.predict(x))/np.asarray(y) ), s=4,
 vmin=0,vmax=max(
 100 * np.abs(
 	(y - nn.predict(x))/np.asarray(y) )))

cbar = fig.colorbar(err)
ax.set_xlabel('Temperature [°C]')
ax.set_ylabel('Damages at time t [%GWP]')
cbar.set_label('Absolute percenatage error [%]')


## FUNCTIONS NEEDED FOR POLYNOMIAL
def reg_func(x, params):
	a2 = params[0]
	a2c = params[1]
	a1 = params[2]
	a1c = params[3]
	b2 = params[4]
	b2c = params[5]
	b1 = params[6]
	b1c = params[7]
	# a3 = params[8]
	# a3c = params[9]
	# b3 = params[10]
	# b3c = params[11]
	x = np.transpose(x)
	y = a1*(x[0]-a1c) + b1*(x[1]-b1c) + \
		+ a2*(x[0]-a2c)**2 + b2*(x[1]-b2c)**2# + \
		# + a3*(x[0]-a3c)**3 + b3*(x[1]-b3c)**3
	return y
## TRAIN POLY IN SIMULATION
def res_sim(params, x, y):
	# simulate over different gtemp
	output = []
	for el in x:
		if el[1]==0:
			output.append( 
				max(0.0, el[1] + reg_func(el,params) ) )
		else:
			output.append(
				max(0.0, output[-1] + reg_func([el[0],output[-1]],params)))
	# collect residual
	residual = np.sum( ( (np.asarray(y) - np.asarray(output) ))**2)
	return residual
## CREATE TRAINING DATASET FOR POLY
x, y = [], []
for el in zip(gtemp, damages):
	for el2 in zip(el[0][:-1], el[1][:-1], el[1][1:]):
		x.append(el2[:2])
		y.append(el2[-1])
## FIT USING 2nd DEGREE POLY
start = time.time()
# norm = +float('inf')
norm = 2
params = scipy.optimize.minimize(res_sim, [0.0 for el in range(8)], 
	args=(x,y), method='BFGS', options={'disp':True}).x
	#'maxiter':1000000, 'gtol':1e-10, 'disp':True, 'norm':norm, 'eps':1e-7}).x
elapsed = time.time() - start
print('Parameters of the polynomial function to be used to model empirical climate damages based on Burke et al. (2015):')
print(params)
print("Elapsed time: " + str(elapsed) + ' seconds')
## plot result (dataset on right, result on the left)
fig, ax = plt.subplots(1,2, sharey=True)
obs = ax[0].scatter(np.transpose(x)[0], np.transpose(x)[1],
 c=y, vmin=0, vmax=0.5, s=4)
model = ax[1].scatter(np.transpose(x)[0], np.transpose(x)[1],
 c = np.transpose(x)[1] + reg_func(params=params, x=x) , 
 vmin=0, vmax=0.5, s=4)
fig.subplots_adjust(bottom=0.25)
cax = fig.add_axes([0.15, 0.12, 0.7, 0.03])
cbar = fig.colorbar(obs, cax=cax, orientation='horizontal')
[el.set_xlabel('Temperature [°C]') for el in ax]
ax[0].set_ylabel('Damages at time t [%GWP]')
cbar.set_label('Damages at time t+1 [%GWP]')
## plot percentage error
fig, ax = plt.subplots(1,1)
err = ax.scatter(np.transpose(x)[0], np.transpose(x)[1], s=2,
	c = 100 * np.abs(
		(y-np.transpose(x)[1]-reg_func(x=x,params=params))/np.asarray(y)),
		vmin=0,vmax=max(
			100 * np.abs(
		(y-np.transpose(x)[1]-reg_func(x=x,params=params))/np.asarray(y))))
cbar = fig.colorbar(err)
ax.set_xlabel('Temperature [°C]')
ax.set_ylabel('Damages at time t [%GWP]')
cbar.set_label('Absolute percenatage error [%]')

## ANN & POLY - check
# plt.figure()
idx = [x for x in range(0,len(gtemp), len(gtemp)//8)]
fig, ax = plt.subplots(3,3, sharey=True, sharex=True)
fig2, ax2 = plt.subplots(3,3, sharey=True, sharex=True)
count = 0
for el2 in idx:
	# simulate ANN
	el = [gtemp[el2], damages[el2]]
	y_ann = np.asarray([0.0])
	el = np.transpose(el)
	for el2 in el[:-1]:
		y_ann = np.append(y_ann, 
			max(0.0, nn.predict([[(el2[0] - 14)/(25 - 14), y_ann[-1]]])))
		# y_ann.append(nn.predict([el2))
	# simulate POLY
	y_poly = np.asarray([0.0])
	for el2 in el[:-1]:
		y_poly = np.append(y_poly, 
			max(0.0, y_poly[-1] + \
			reg_func(x = [el2[0], y_poly[-1]], params=params)))
	el = np.transpose(el)
	ax[count//3][count%3].plot(el[0], el[1], 
		label='orig.', color='k', marker='*')
	ax[count//3][count%3].plot(el[0], y_ann, 
		label='ANN', color='b', linestyle='--')
	ax[count//3][count%3].plot(el[0], y_poly, 
		label='POLY', color='r', linestyle='--')
	ax[count//3][count%3].set_title('Temperature increase: + '+\
		str(round(el[0][-1]-el[0][0]+0.6, 2))+' °C')
	count += 1
	
[el.set_xlabel('Temperature [°C]') for el in ax[2]]
[el[0].set_ylabel('Damages [%GDP]') for el in ax]
fig.legend(['Original Burke Model', 'ANN' , '2nd deg. POLY'], 
	loc='lower center', ncol=3)

# ## create 500 year scenarios to check function in the long term
# for el in range(40):
# 	temp_sim = [gtemp[0][0]]
# 	for t in range(0, 485, 5):
# 		if round(t/5) < el + 8:
# 			temp_sim.append(temp_sim[-1] + 0.072)
# 		else:
# 			temp_sim.append(temp_sim[-1])
# 	# simulate ANN
# 	y_ann = np.asarray([0.0])
# 	for el2 in temp_sim:
# 		y_ann = np.append(
# 			y_ann, max(0.0, nn.predict([[(el2 - 14)/(25 - 14), y_ann[-1]]])))

# 	# simulate POLY
# 	y_poly = np.asarray([0.0])
# 	for el2 in temp_sim:
# 		y_poly = np.append(y_poly, 
# 			max(0.0, y_poly[-1]+reg_func(x=[el2,y_poly[-1]],params=params)))
# 	fig, ax = plt.subplots(2,1)
# 	ax[0].plot([x for x in range(2010,2500,5)],temp_sim)
# 	ax[1].plot([x for x in range(2010,2505,5)],y_ann,label='ANN',color='b')
# 	ax[1].plot([x for x in range(2010,2505,5)],y_poly,label='POLY',color='r')
# 	fig.legend()


plt.show()