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

def patch_violinplot():
     from matplotlib.collections import PolyCollection
     ax = plt.gca()
     for art in ax.get_children():
          if isinstance(art, PolyCollection):
              art.set_edgecolor((0, 0, 0, 0))

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

folders = ['SO','SO_AD','SO_UNC','SO_AD_UNC']
folders = ['BorgOutput_' + x for x in folders]
colors = ['#397ea1','#c59824','#c94933','#ab1818']
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

so_sols_cal = pd.DataFrame(all_cal_objs)
so_sols_val = pd.DataFrame(all_val_objs)
so_sols_cal.columns = ['Welfare_cal','Type']
so_sols_val.columns = ['Welfare', 'P(GMST > 2°C)', 'Warming above 2°C [°C]',
       'NPV Damages [10^12 USD]', 'NPV Abat. costs [10^12 USD]', 
       'NPV Adapt. costs [10^12 USD]','Type']
so_sols = pd.merge(so_sols_cal,so_sols_val, how='outer', on='Type')
so_sols[['DegY1.5°C_cal', 'DegY2°C_cal', 'NPV Damages_cal', 'NPV Abatement_cal',
       'NPV Adaptation_cal']] = 0.0
so_sols = so_sols[['Welfare_cal', 'DegY1.5°C_cal', 'DegY2°C_cal', 'NPV Damages_cal', 'NPV Abatement_cal',
       'NPV Adaptation_cal', 'Type', 'Welfare', 'P(GMST > 2°C)', 'Warming above 2°C [°C]',
       'NPV Damages [10^12 USD]', 'NPV Abat. costs [10^12 USD]', 'NPV Adapt. costs [10^12 USD]']]

allsols = pd.read_csv('./SimulationValFull.csv')
allsols.columns = ['Welfare_cal', 'DegY1.5°C_cal', 'DegY2°C_cal', 'NPV Damages_cal', 'NPV Abatement_cal',
       'NPV Adaptation_cal', 'Type', 'Welfare', 'P(GMST > 2°C)', 'Warming above 2°C [°C]',
       'NPV Damages [10^12 USD]', 'NPV Abat. costs [10^12 USD]', 'NPV Adapt. costs [10^12 USD]']#,
allsols = allsols.loc[allsols['Welfare']<0]

dps = allsols.loc[allsols['Type']=='DPS']


epss = [25,1]
nondom = pareto.eps_sort([list(dps.itertuples(False))], 
	objectives=[7,9], epsilons=epss, kwargs={'maximize':[0]})
dps = pd.DataFrame.from_records(nondom, columns=list(dps.columns.values))
so = allsols.loc[allsols['Type']=='SO']

nondom = pareto.eps_sort([list(so.itertuples(False))], 
	objectives=[7,9], epsilons=epss, kwargs={'maximize':[0]})
so = pd.DataFrame.from_records(nondom, columns=list(so.columns.values))
allsols = pd.concat([so,dps], ignore_index=True)


allsols = allsols.append(so_sols)
allsols['TypeID'] = allsols['Type'].astype('category').cat.codes
allsols['\u0394 CBGE [%]'] = 100 * \
	((allsols.loc[allsols['Type']=='_SO_AD_UNC']['Welfare'].min() / allsols['Welfare'])**(1/(1-1.45)) - 1)

allsols['NPV Total costs [10^12 USD]'] = allsols['NPV Damages [10^12 USD]'] + \
	allsols['NPV Adapt. costs [10^12 USD]'] + allsols['NPV Abat. costs [10^12 USD]']

print(allsols['Type'].unique())
allsols = allsols.loc[(allsols['\u0394 CBGE [%]'] > allsols.loc[allsols['Type']=='SO']['\u0394 CBGE [%]'].min())
	| (allsols['Type'].str.contains('_SO'))]
allsols = allsols.loc[(allsols['\u0394 CBGE [%]'] > -5)]
allsols['Method'] = ['Static intertemporal (multi-objective, uncertain)' if x =='SO' else
			'Static intertemporal (single-objective, deterministic, no adaptation)' if x =='_SO' else
			'Static intertemporal (single-objective, uncertain, no adaptation)' if x =='_SO_UNC' else
			'Static intertemporal (single-objective, deterministic)' if x =='_SO_AD' else
			'Static intertemporal (single-objective, uncertain)' if x =='_SO_AD_UNC'
			 else 'Self-adaptive (multi-objective, uncertain)' for x in allsols['Type'] ]
allsols['order'] = [1 if x =='SO' else
					5 if x =='_SO' else
					3 if x =='_SO_UNC' else
					4 if x =='_SO_AD' else
					2 if x =='_SO_AD_UNC'
					 else 0 for x in allsols['Type'] ]

allsols = allsols.sort_values(by='order', ascending=True)

plt.figure()
colors = ['forestgreen','darkorchid','red','red','darkorange','darkorange']
markers=['o','o','^','s','^','s']
for el in range(len(allsols['Type'].unique())):
	seltype = allsols['Type'].unique()[el]
	sel = allsols.loc[allsols['Type']==seltype]
	plt.scatter(sel['Warming above 2°C [°C]'].values.tolist(),
		sel['\u0394 CBGE [%]'].values.tolist(), color=colors[el], marker=markers[el], alpha=0.8, label=sel['Method'].values[0])
plt.xlabel('Warming above 2°C [°C]')
plt.ylabel('\u0394 CBGE [%]')
plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), ncol=1)
plt.gcf().set_size_inches(9, 4)
plt.tight_layout()

allsols = allsols.drop(['Method'], axis=1)

plt.show()
