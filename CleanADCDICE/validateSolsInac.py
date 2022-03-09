import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import os, matplotlib, subprocess, time, chart_studio
import seaborn as sns
import matplotlib.font_manager as font_manager
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from chart_studio import plotly as py
import pareto

sns.set_style('whitegrid')
# sns.set_context('paper')
import matplotlib.font_manager as font_manager
fontpath = '/Users/angelocarlino/Library/Fonts/OpenSans-Regular.ttf'
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

def simulate(params, niter=10, adaptive=1, seed=1):
	with open('./sol.txt','w') as f:
		for el in params:
			f.write(str(el)+" ")
	with open('./src/config/config.txt') as f:
		file = f.read().split("\n")
	newfile = file
	for el in range(len(file))[:-1]:
		if file[el].split()[0] == "niter":
			newfile[el] = 'niter '+str(niter)
		if file[el].split()[0] == 'adaptive':
			newfile[el] = 'adaptive '+str(adaptive)
		if file[el].split()[0] == 'writefile':
			newfile[el] = 'writefile 0'
		if file[el].split()[0] == 'scc':
			newfile[el] = 'scc 0'
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
nobjs = 6
path = './'
RNTS = {}
srand = np.random.randint(0,1e6)
class Rnt:
	def __init__(self):
		self.NFE = []
		self.SBX = []
		self.DE = []
		self.PCX = []
		self.SPX = []
		self.UNDX = []
		self.UM = []
		self.IMP = []
		self.RES = []
		self.ARCSIZE = []
		self.OBJS = []
		self.PARAMS = []

solsneeded = pd.read_csv('./SimulationValFull.csv')
epss = [25,0.05,5,1,1,1]
solsneeded.columns = ['Welfare_cal', 'DegY1.5°C_cal', 'DegY2°C_cal', 'NPV Damages_cal', 'NPV Abatement_cal',
       'NPV Adaptation_cal', 'Type', 'Welfare', 'P(GMST > 2°C)', 'Warming above 2°C [°C yr]',
       'NPV Damages [10^12 USD]', 'NPV Abat. costs [10^12 USD]', 'NPV Adapt. costs [10^12 USD]']
solsneeded = solsneeded.loc[solsneeded['Welfare'] < 0]
dps = solsneeded.loc[solsneeded['Type']=='DPS']
nondom = pareto.eps_sort([list(dps.itertuples(False))], 
       objectives=[7,8,9,10,11,12], epsilons=epss, kwargs={'maximize':[0]})
dps = pd.DataFrame.from_records(nondom, columns=list(dps.columns.values))
so = solsneeded.loc[solsneeded['Type']=='SO']
nondom = pareto.eps_sort([list(so.itertuples(False))], 
       objectives=[7,8,9,10,11,12], epsilons=epss, kwargs={'maximize':[0]})
so = pd.DataFrame.from_records(nondom, columns=list(so.columns.values))
solsneeded = pd.concat([so,dps], ignore_index=True)
columns_to_use = ['Welfare', 'P(GMST > 2°C)', 'Warming above 2°C [°C yr]',
       'NPV Damages [10^12 USD]', 'NPV Abat. costs [10^12 USD]', 'NPV Adapt. costs [10^12 USD]']
nondom = pareto.eps_sort([list(solsneeded.itertuples(False))], 
       objectives=[7,8,9,10,11,12], epsilons=epss, kwargs={'maximize':[0]})
solsneeded = pd.DataFrame.from_records(nondom, columns=list(solsneeded.columns.values))
solsneeded = solsneeded[solsneeded.columns[0:6]].values.tolist()
# for idx in range(len(solsneeded)):
# 	solsneeded[idx] = [round(x,4) for x in solsneeded[idx]]

allsols = []
folders = ['BorgOutput_SO_AD_UNC','BorgOutput_SO_AD_UNC_6OBJS','BorgOutput_DPS_AD_UNC_6OBJS']

all_val_objs = []
all_cal_objs = []
for folder in folders:
	print(folder)
	if '6OBJS' in folder:
		nobjs = 6
	else:
		nobjs = 1
	with open('./'+folder+'/optADCDICE2016.reference') as f:
		file = f.read()
	ref = [x.split(' ') for x in file.split("\n")[:-1]]
	for idx in range(len(ref)):
		ref[idx] = [float(x) for x in ref[idx]]
		# ref[idx] = [round(float(x),4) for x in ref[idx]]
	print(len(ref))
	if '6OBJS' in folder:
		newsols = []
		for el in ref:
			if el in solsneeded:
				newsols.append(el)
		ref = newsols
	print(len(ref))
	sols = []
	for seed in range(nseeds):
		with open('./'+folder+'/optADCDICE2016_'+str(seed+1)+'.out') as f:
			file = f.read()
		outfile = [x.split(' ') for x in file.split("\n")[12:-3]]
		for idx in range(len(outfile)):
			# outfile[idx] = [float(x) for x in outfile[idx]]
			outfile[idx] = [float(x) for x in outfile[idx]]
		for el in outfile:
			if len(el) > nobjs and el[-nobjs:] in ref:
				sols.append([float(x) for x in el])
	print(len(sols))
	adaptive = 0
	dec = 'SO'
	niter = 1000
	seed = 2
	if 'DPS' in folder:
		adaptive=1
		dec = 'DPS'
	val_objs = []
	cal_objs = []
	for sol in sols:
		if sol[-nobjs] < -500000: ## welfare for loss of 5% GCBE w.r.t welfare maximizing solution
			val_obj = simulate(sol[:-nobjs], adaptive=adaptive, niter=niter, seed=seed)
			val_objs.append(val_obj)
			cal_objs.append(sol[-nobjs:])
	for el in range(len(val_objs)):
		val_objs[el].append(dec)
		cal_objs[el].append(dec)
		all_cal_objs.append(cal_objs[el])
		all_val_objs.append(val_objs[el])

allsols = all_val_objs
allsols = pd.DataFrame(allsols, columns=['Welfare','DegY1.5°C','DegY2°C','NPV Damages','NPV Abatement','NPV Adaptation','Type'])
for el in allsols.columns[:-1]:
	allsols[el] = pd.to_numeric(allsols[el])

allsols['TypeID'] = allsols['Type'].astype('category').cat.codes

allsols['\u0394 CBGE'] = 100 * ((allsols.loc[allsols['Type']=='SO']['Welfare'].min() / allsols['Welfare'])**(1/(1-1.45)) - 1)

cols = ['Welfare','DegY1.5°C','DegY2°C','NPV Damages','NPV Abatement','NPV Adaptation','Type']
sols_pd = pd.DataFrame(all_cal_objs, columns=cols)
cols = ['Welfare','DegY1.5°C','DegY2°C','NPV Damages','NPV Abatement','NPV Adaptation','Type']
cols = [x+'_val' for x in cols]
sols_val_pd = pd.DataFrame(all_val_objs, columns=cols)
val_file = pd.concat([sols_pd, sols_val_pd], axis=1)
val_file = val_file[[x for x in val_file.columns[:-1]]]
val_file.to_csv('SimulationValDPSInac10y.csv', index=False)


# allsols = allsols.loc[allsols['\u0394 CBGE'] > allsols.loc[allsols['Type']=='SO']['\u0394 CBGE'].min()]
allsols = allsols.loc[allsols['\u0394 CBGE'] > -5]

for col in allsols.columns[1:3]:
	allsols = allsols.loc[allsols[col] <= allsols.loc[allsols['Type']=='SO'][col].max()]

allsols.columns = ['Welfare','DegYrs 1.5°C [°C]','DegYrs 2°C [°C]','NPV Damages [10^12 USD]','NPV Abat. costs [10^12 USD]','NPV Adapt. costs [10^12 USD]','Type','TypeID','\u0394 CBGE [%]']
allsols = allsols[['\u0394 CBGE [%]','DegYrs 1.5°C [°C]','DegYrs 2°C [°C]','NPV Damages [10^12 USD]','NPV Abat. costs [10^12 USD]','NPV Adapt. costs [10^12 USD]','Type','TypeID','Welfare']]
allsols = allsols.loc[ allsols['\u0394 CBGE [%]'] >= allsols.loc[allsols['Type']=='SO']['\u0394 CBGE [%]'].min() ]

for col in allsols.columns[1:6]:
	allsols = allsols.loc[ allsols[col] <= allsols.loc[allsols['Type']=='SO'][col].max() ]

vec = allsols
fig = go.Figure(data = 
	go.Parcoords(
		line = dict(color = vec['TypeID'],
			colorscale = [[0,'forestgreen'],[1,'darkorchid']],
			showscale = False),
		dimensions = list([
			# dict(range = [vec['\u0394 CBGE'].min(), vec['\u0394 CBGE'].max()],
			# 	label = '\u0394 CBGE', values = vec['\u0394 CBGE']),
			dict(range = [vec['\u0394 CBGE [%]'].max(), vec['\u0394 CBGE [%]'].min()],
				label = '\u0394 CBGE [%]', values = vec['\u0394 CBGE [%]']),
			# dict(range = [vec['Welfare'].min(), vec['Welfare'].max()],
			# 	label = 'Welfare', values = vec['Welfare']),
			dict(range = [vec['DegYrs 1.5°C [°C]'].min(), vec['DegYrs 1.5°C [°C]'].max()],
				label = 'DegYrs 1.5°C [°C]', values = vec['DegYrs 1.5°C [°C]']),
			dict(range = [vec['DegYrs 2°C [°C]'].min(), vec['DegYrs 2°C [°C]'].max()],
				label = 'DegYrs 2°C [°C]', values = vec['DegYrs 2°C [°C]']),
			dict(range = [vec['NPV Abat. costs [10^12 USD]'].min(), vec['NPV Abat. costs [10^12 USD]'].max()],
				label = 'NPV Abat. costs [10^12 USD]', values = vec['NPV Abat. costs [10^12 USD]']),
			dict(range = [vec['NPV Adapt. costs [10^12 USD]'].min(), vec['NPV Adapt. costs [10^12 USD]'].max()],
				label = 'NPV Adapt. costs [10^12 USD]', values = vec['NPV Adapt. costs [10^12 USD]']),
			dict(range = [vec['NPV Damages [10^12 USD]'].min(), vec['NPV Damages [10^12 USD]'].max()],
				label = 'NPV Damages [10^12 USD]', values = vec['NPV Damages [10^12 USD]']),
			])
		)
	# , layout = go.Layout(height=500, width=1000)
	)
fig.update_layout(title_text='Self-adaptive (green) and intertemporal optimization (purple) solutions in the objectives<span>&#39;</span> space')
fig.show()

nondominated = pareto.eps_sort([list(allsols.itertuples(False))], objectives=[x for x in range(6)], epsilons=[1,1,1,1,1,1], kwargs={'maximize':[0]})

allsols = pd.DataFrame.from_records(nondominated, columns=list(allsols.columns.values))
# allsols = allsols.loc[allsols['Welfare']<-500000]
vec = allsols
fig = go.Figure(data = 
	go.Parcoords(
		line = dict(color = vec['TypeID'],
			colorscale = [[0,'forestgreen'],[1,'darkorchid']],
			showscale = False),
		dimensions = list([
			# dict(range = [vec['\u0394 CBGE'].min(), vec['\u0394 CBGE'].max()],
			# 	label = '\u0394 CBGE', values = vec['\u0394 CBGE']),
			dict(range = [vec['\u0394 CBGE [%]'].max(), vec['\u0394 CBGE [%]'].min()],
				label = '\u0394 CBGE [%]', values = vec['\u0394 CBGE [%]']),
			# dict(range = [vec['Welfare'].min(), vec['Welfare'].max()],
			# 	label = 'Welfare', values = vec['Welfare']),
			dict(range = [vec['DegYrs 1.5°C [°C]'].min(), vec['DegYrs 1.5°C [°C]'].max()],
				label = 'DegYrs 1.5°C [°C]', values = vec['DegYrs 1.5°C [°C]']),
			dict(range = [vec['DegYrs 2°C [°C]'].min(), vec['DegYrs 2°C [°C]'].max()],
				label = 'DegYrs 2°C [°C]', values = vec['DegYrs 2°C [°C]']),
			dict(range = [vec['NPV Abat. costs [10^12 USD]'].min(), vec['NPV Abat. costs [10^12 USD]'].max()],
				label = 'NPV Abat. costs [10^12 USD]', values = vec['NPV Abat. costs [10^12 USD]']),
			dict(range = [vec['NPV Adapt. costs [10^12 USD]'].min(), vec['NPV Adapt. costs [10^12 USD]'].max()],
				label = 'NPV Adapt. costs [10^12 USD]', values = vec['NPV Adapt. costs [10^12 USD]']),
			dict(range = [vec['NPV Damages [10^12 USD]'].min(), vec['NPV Damages [10^12 USD]'].max()],
				label = 'NPV Damages [10^12 USD]', values = vec['NPV Damages [10^12 USD]']),
			])
		)
	# , layout = go.Layout(height=500, width=1000)
	)
fig.update_layout(title_text='Self-adaptive (green) and intertemporal optimization (purple) solutions in the objectives<span>&#39;</span> space')
fig.show()

# plt.figure()
# sns.scatterplot(data=vec, x='DegYrs 2°C [°C]', y='\u0394 CBGE [%]', size='NPV Damages [10^12 USD]', hue='Type', palette=['darkorchid','forestgreen'])
# plt.figure()
# sns.scatterplot(data=vec, x='DegYrs 2°C [°C]', y='NPV Abat. costs [10^12 USD]', size='NPV Adapt. costs [10^12 USD]', hue='Type', palette=['darkorchid','forestgreen'])
# plt.figure()
# sns.scatterplot(data=vec, x='DegYrs 2°C [°C]', y='NPV Damages [10^12 USD]', hue='Type', palette=['darkorchid','forestgreen'])
plt.figure()
sns.scatterplot(data=vec, x='DegYrs 2°C [°C]', size='NPV Abat. costs [10^12 USD]', y='\u0394 CBGE [%]', hue='Type', palette=['darkorchid','forestgreen'])
plt.figure()
sns.scatterplot(data=vec, x='DegYrs 2°C [°C]', y='\u0394 CBGE [%]', size='NPV Adapt. costs [10^12 USD]', hue='Type', palette=['darkorchid','forestgreen'])
plt.figure()
sns.scatterplot(data=vec, x='DegYrs 2°C [°C]', y='\u0394 CBGE [%]', size='NPV Damages [10^12 USD]', hue='Type', palette=['darkorchid','forestgreen'])
fig, ax = plt.subplots(1,3, sharey=True)
sns.scatterplot(ax = ax[0], data=vec, size='DegYrs 2°C [°C]', 
	y='\u0394 CBGE [%]', x='NPV Abat. costs [10^12 USD]', 
	hue='Type', palette=['darkorchid','forestgreen'])
sns.scatterplot(ax = ax[1], data=vec, size='DegYrs 2°C [°C]', 
	y='\u0394 CBGE [%]', x='NPV Adapt. costs [10^12 USD]', 
	hue='Type', palette=['darkorchid','forestgreen'], legend=False)
sns.scatterplot(ax = ax[2], data=vec, size='DegYrs 2°C [°C]', 
	y='\u0394 CBGE [%]', x='NPV Damages [10^12 USD]', 
	hue='Type', palette=['darkorchid','forestgreen'], legend=False)
handles , labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels, loc ='lower center', ncol=9)
ax[0].legend([], [], frameon=False)

fig, ax = plt.subplots(1,3, sharey=True)
sns.scatterplot(ax = ax[0], data=vec, x='DegYrs 2°C [°C]', 
	size='\u0394 CBGE [%]', y='NPV Abat. costs [10^12 USD]', alpha = 0.25, edgecolor=None,
	hue='Type', palette=['darkorchid','forestgreen'])
sns.scatterplot(ax = ax[1], data=vec, x='DegYrs 2°C [°C]', 
	size='\u0394 CBGE [%]', y='NPV Adapt. costs [10^12 USD]', alpha = 0.25, edgecolor=None,
	hue='Type', palette=['darkorchid','forestgreen'], legend=False)
sns.scatterplot(ax = ax[2], data=vec, x='DegYrs 2°C [°C]',  
	size='\u0394 CBGE [%]', y='NPV Damages [10^12 USD]', alpha = 0.25, edgecolor=None,
	hue='Type', palette=['darkorchid','forestgreen'], legend=False)
handles , labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels, loc ='lower center', ncol=9)
ax[0].legend([], [], frameon=False)

plt.show()
# username = 'angelocarlino'
# api_key = 'yz46AoJhyNvvOBRwXXkK'
# chart_studio.tools.set_credentials_file(username=username, api_key=api_key)
# fig.write_html("InteractiveParallel.html")
# py.plot(fig, filename='InteractiveParallel', auto_open=False)
# print(chart_studio.tools.get_embed('https://plotly.com/~angelocarlino/1'))

