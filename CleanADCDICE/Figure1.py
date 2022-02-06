import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import os, matplotlib, subprocess, time, chart_studio, math
import seaborn as sns
import matplotlib.font_manager as font_manager
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from chart_studio import plotly as py
import pareto
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

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

def plot_prctiles(df, col, prc=[0.5, 0.17, 0.73], color='tab:blue', ax=None, linestyle='-'):
	if ax==None:
		fig, ax = plt.subplots()
	lines = []
	for p in prc:
		line = []
		for y in df['YEAR'].unique():
			line.append(df.loc[df['YEAR']==y][col].quantile(p))
		lines.append(line)
	ax.plot(df['YEAR'].unique(), lines[0], color=color, linestyle=linestyle)
	ax.fill_between(df['YEAR'].unique(), lines[1], lines[2], facecolor=color, alpha=0.1, zorder = -1)
	ax.set_ylabel(col)
	return ax

def simulate(params, niter=10, adaptive=1, adaptation=0, unc=0, nobjs=1, seed=1):
	with open('./sol.txt','w') as f:
		for el in params:
			f.write(str(el)+" ")
	with open('./src/config/config.txt') as f:
		file = f.read().split("\n")
	newfile = file
	for el in range(len(file))[:-1]:
		if file[el].split()[0] == "niter":
			newfile[el] = 'niter '+str(niter)
		if file[el].split()[0] == "nobjs":
			newfile[el] = 'nobjs '+str(nobjs)
		if file[el].split()[0] == 'adaptive':
			newfile[el] = 'adaptive '+str(adaptive)
		if file[el].split()[0] == "adaptation":
			newfile[el] = 'adaptation '+str(adaptation)
		if file[el].split()[0] == 'writefile':
			newfile[el] = 'writefile 1'
		if any(x in file[el].split()[0] for x in ['unc','stoch']):
			newfile[el] = newfile[el].split()[0]+"\t"+str(unc)
	with open('./src/config/config.txt','w') as f:
		for el in range(len(newfile)):
			f.write(newfile[el])
			if el < len(newfile)-1:
				f.write("\n")
	output = subprocess.check_output('./ADCDICE2016 '+str(seed)+' < sol.txt', shell=True).decode("utf-8")
	output = [float(x) for x in output.split('\n')[0].split()]
	print(output)
	return output

nseeds = 5
nobjs = 1

all_val_objs = []
all_cal_objs = []

fig1, ax1 = plt.subplots(2,2, sharex=True, sharey=True)
fig2, ax2 = plt.subplots(2,2, sharex=True, sharey=True)

folders = ['SO','SO_AD','SO_UNC','SO_AD_UNC']
folders = ['BorgOutput_' + x for x in folders]
colors = ['#397ea1','#c59824','#c94933','#ab1818']
colors = ['orange','orange','red','red']
count = 0
for folder in folders:
	print(folder)
	with open('./'+folder+'/optADCDICE2016.reference') as f:
		file = f.read()
	ref = [x.split(' ') for x in file.split("\n")[:-1]]
	for idx in range(len(ref)):
		ref[idx] = [float(x) for x in ref[idx]]
	sols = []
	for seed in range(nseeds):
		with open('./'+folder+'/optADCDICE2016_'+str(seed+1)+'.out') as f:
			file = f.read()
		outfile = [x.split(' ') for x in file.split("\n")[12:-3]]
		for idx in range(len(outfile)):
			outfile[idx] = [float(x) for x in outfile[idx]]
		for el in outfile:
			if len(el) > nobjs and el[-nobjs:] in ref:
				print(el[-nobjs:])
				sols.append([float(x) for x in el])
	print(len(sols))
	adaptive = 0
	adaptation = 0
	unc = 1
	niter = 1000
	seed = 2
	if 'AD' in folder:
		adaptation = 1
	if 'UNC' in folder:
		unc = 1
	val_objs = []
	cal_objs = []
	for sol in sols:
		val_obj = simulate(sol[:-nobjs], adaptive=adaptive, niter=niter, 
			nobjs=6, adaptation=adaptation, unc=unc, seed=seed)
		val_objs.append(val_obj)
		cal_objs.append(sol[-nobjs:])
	for el in range(len(val_objs)):
		val_objs[el].append(folder[10:])
		cal_objs[el].append(folder[10:])
		all_cal_objs.append(cal_objs[el])
		all_val_objs.append(val_objs[el])
	output = pd.read_csv('./SimulationsOutput.txt', sep='\t')
	output = output.loc[output['YEAR'] <= 2150]
	output['MIU'] = output['MIU']*100
	output['S'] = output['S']*100
	output['IA'] = output['IA']*100
	output['FAD'] = output['FAD']*100

	output['ADAPTEFF_d'] = pd.cut(output['ADAPTEFF'], bins=[-0.001,0.0001,0.7,1.3,2.0], labels=['zero','low','nominal','high'])
	output['DAMTYPE_d'] = pd.cut(output['DAMTYPE'], bins=[0.9,1.1,2.1], labels=['Level','Growth'])
	output['Non-CO2 RF'] = pd.cut(output['RFOTH'], bins=[-0.1,0.1,1.1,2.1,3.1], labels=['RCP2.6','RCP4.5','RCP6.0','RCP8.5'])
	output['TCR'] = pd.cut(output['TCR'], bins=[-0.1,1.4,2.2,10.0], labels=['Lower than likely','Likely','Higher than likely'])
	output.columns = ['Damages type' if x=='DAMTYPE_d' else x for x in output.columns] 
	output.columns = ['Adapt. eff.' if x=='ADAPTEFF_d' else x for x in output.columns] 
	output.columns = ['GMST [°C]' if x=='TATM' else x for x in output.columns] 
	output.columns = ['Emission control [%]' if x=='MIU' else x for x in output.columns] 
	output.columns = ['Adapt. invest. [%GDP]' if x=='IA' else x for x in output.columns] 
	output.columns = ['Flow adapt. [%GDP]' if x=='FAD' else x for x in output.columns] 
	output.columns = ['CO2 Ind. Emissions [GtCO2]' if x=='EIND' else x for x in output.columns] 
	
	linestyle = '-'
	if 'AD' in folder:
		linestyle=':'
	plot_prctiles(output, 'GMST [°C]', ax=ax1[math.floor(count/2)][count%2], color=colors[count], linestyle=linestyle)
	plot_prctiles(output, 'CO2 Ind. Emissions [GtCO2]', ax=ax2[math.floor(count/2)][count%2], color=colors[count], linestyle=linestyle)
	ax1[math.floor(count/2)][count%2].set_xlim((2020,2150))
	ax2[math.floor(count/2)][count%2].set_xlim((2020,2150))
	ax1[math.floor(count/2)][count%2].set_ylim((1.0, 4.0))
	ax2[math.floor(count/2)][count%2].set_ylim((-25.0, 50.0))
	ax1[math.floor(count/2)][count%2].plot([x for x in range(2020,2150)],[2 for x in range(2020,2150)],'k--', alpha=0.5, zorder=-1)
	ax1[math.floor(count/2)][count%2].plot([x for x in range(2020,2150)],[1.5 for x in range(2020,2150)],'k--', alpha=0.5, zorder=-1)
	ax2[math.floor(count/2)][count%2].plot([x for x in range(2020,2150)],[0.0 for x in range(2020,2150)],'k--', alpha=0.5, zorder=-1)
	

	count += 1
ax1[0][1].set_ylabel('')
ax1[1][1].set_ylabel('')
ax2[0][1].set_ylabel('')
ax2[1][1].set_ylabel('')
fig1.subplots_adjust(top=0.95, bottom=0.1, hspace=0.3)
fig2.subplots_adjust(top=0.95, bottom=0.1, hspace=0.3)

plt.figure()

patchd = mpatches.Patch(color='orange', label='Deterministic') 
patchu = mpatches.Patch(color='red', label='Uncertain') 
line = Line2D([0], [0], label='Implicit adaptation', color='k', linestyle='-')
linead = Line2D([0], [0], label='Explicit adaptation', color='k', linestyle=':')
handles = [patchd, patchu, line, linead]
plt.legend(handles=handles)
plt.show()