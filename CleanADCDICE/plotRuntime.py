import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import os, matplotlib, subprocess, time
import seaborn as sns
import matplotlib.font_manager as font_manager
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

sns.set_style('whitegrid')
# sns.set_context('paper')
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

nseeds = 5
nobjs = 6
epss = [25.0,25.0,10.0,25.0,10.0,20.0]
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

allsols = []
folders = ['BorgOutput_SO','BorgOutput_SO_AD','BorgOutput_SO_UNC','BorgOutput_SO_AD_UNC','BorgOutput_SO_AD_UNC_6OBJS','BorgOutput_DPS_AD_UNC_6OBJS',]
folders = ['BorgOutput_DPS_AD_UNC_6OBJS']
for folder in folders:

	if '6OBJS' not in folder:
		nobjs = 1
	else:
		nobjs = 6
	color = ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9','C10']
	rnts = [Rnt() for el in range(nseeds)]
	vec = []
	vecp = []
	fig4 = plt.figure( figsize=(9,6))
	gs = fig4.add_gridspec(4,1)
	f4_ax1 = fig4.add_subplot(gs[0,0])
	f4_ax2 = fig4.add_subplot(gs[1,0], xticklabels = [], sharex=f4_ax1)
	f4_ax3 = fig4.add_subplot(gs[2,0], xticklabels = [], sharex=f4_ax1)
	f4_ax4 = fig4.add_subplot(gs[3,0])
	fig5, ax5 = plt.subplots(nseeds,1, sharex=True)
	# f4_ax5 = fig4.add_subplot(gs[:,1])
	count = 0
	sols = []
	maxvalIMP = 0
	maxvalRES = 0
	maxvalARCSIZE = 0
	maxvaldIMP = 0
	for seed in range(1,nseeds+1):
		print(seed)
		try:
			with open('./'+folder+'/rntdynamics_'+str(seed)+'.txt') as f:
				file = f.read()
			if file !='':
				for line in file.split("\n"):
					if "NFE" in line:
						rnts[seed-1].NFE.append(int(line.split('=')[-1]))
					if "SBX" in line:
						rnts[seed-1].SBX.append(float(line.split('=')[-1]))
					if "DE" in line:
						rnts[seed-1].DE.append(float(line.split('=')[-1]))
					if "PCX" in line:
						rnts[seed-1].PCX.append(float(line.split('=')[-1]))
					if "SPX" in line:
						rnts[seed-1].SPX.append(float(line.split('=')[-1]))
					if "UNDX" in line:
						rnts[seed-1].UNDX.append(float(line.split('=')[-1]))
					if "UM" in line:
						rnts[seed-1].UM.append(float(line.split('=')[-1]))
					if "Improvements" in line:
						rnts[seed-1].IMP.append(int(line.split('=')[-1]))
					if "Restarts" in line:
						rnts[seed-1].RES.append(int(line.split('=')[-1]))
					if "ArchiveSize" in line:
						rnts[seed-1].ARCSIZE.append(int(line.split('=')[-1]))			
					if "#" in line:
						rnts[seed-1].OBJS.append(vec)
						vec = []
						rnts[seed-1].PARAMS.append(vecp)
						vecp = []
					if "//" not in line and "#" not in line and line not in ['', '\n']:
						vec.append([float(x) for x in line.split(' ')[-nobjs:]])
						vecp.append([float(x) for x in line.split(' ')[:-nobjs]])

				f4_ax1.plot(rnts[seed-1].NFE, rnts[seed-1].IMP, label='seed '+str(seed), color=color[seed-1])
				f4_ax1.set_ylabel('Improvements')
				maxvalIMP = max(maxvalIMP,max(rnts[seed-1].IMP))
				f4_ax1.set_ylim((-10.0,1.2*maxvalIMP))
				f4_ax2.plot(rnts[seed-1].NFE, rnts[seed-1].ARCSIZE, color=color[seed-1])
				f4_ax2.set_ylabel('Archive Size')
				maxvalARCSIZE = max(maxvalARCSIZE,max(rnts[seed-1].ARCSIZE))
				f4_ax2.set_ylim((-10.0,1.2*maxvalARCSIZE))
				f4_ax3.plot(rnts[seed-1].NFE, rnts[seed-1].RES, color=color[seed-1])
				f4_ax3.set_ylabel('Restarts')
				maxvalRES = max(maxvalRES,max(rnts[seed-1].RES))
				f4_ax3.set_ylim((-10.0,1.2*maxvalRES))
				f4_ax4.plot(rnts[seed-1].NFE, [rnts[seed-1].IMP[0], *np.diff(rnts[seed-1].IMP)], color=color[seed-1])
				f4_ax4.set_xlabel('NFE')
				maxvaldIMP = max(maxvaldIMP,max([rnts[seed-1].IMP[0], *np.diff(rnts[seed-1].IMP)]))
				f4_ax4.set_ylim((-10.0,1.2*maxvaldIMP))
				f4_ax4.set_ylabel('New Improvements')
				fig4.suptitle(folder)
				plt.tight_layout()
				ax5[seed-1].stackplot(rnts[seed-1].NFE, rnts[seed-1].SBX, rnts[seed-1].DE,
					rnts[seed-1].PCX, rnts[seed-1].SPX, rnts[seed-1].UNDX,
					rnts[seed-1].UM, labels=['SBX','DE','PCX','SPX','UNDX','UM'], linewidth=0.1)
				ax5[seed-1].set_ylabel('[%]')
				if seed == nseeds:
					ax5[seed-1].set_xlabel('NFE')
					fig5.legend(ax5[0].get_legend_handles_labels()[0][:6],ax5[0].get_legend_handles_labels()[1][:6])
				fig5.suptitle(folder)
				plt.tight_layout()
				[sols.append(x) for x in rnts[seed-1].OBJS[-1]]
		except FileNotFoundError:
			print('Runtime file is not present for seed ' + str(seed))
		count += 1

	# newsols = []
	# for i in range(len(sols)):
	# 	j = 0
	# 	breaker=True
	# 	while (breaker and j < len(sols)):
	# 		flag = 0
	# 		for nobj in range(nobjs):
	# 			if sols[i][nobj] > sols[j][nobj] :
	# 				flag += 1
	# 		if flag == nobjs:
	# 			breaker = False
	# 		j = j+1
	# 	if breaker:
	# 		newsols.append(sols[i])
	# print(len(sols), len(newsols))
	# sols = newsols

	# for x in range(len(sols)):
	# 	sols[x].append(folder.split('_')[1])
	# 	allsols.append(sols[x])
	# sols = pd.DataFrame(sols, columns=['Welfare','DegY1.5°C','DegY2°C','NPV Damages','NPV Abatement','NPV Adaptation','Type'])

	# sols = sols.loc[(sols['Welfare']<0) & (sols['NPV Abatement']>0)]
	# vec = sols
	# fig = go.Figure(data = 
	# 	go.Parcoords(
	# 		line = dict(color = vec['NPV Adaptation'],
	# 			colorscale = 'Tealrose',
	# 			showscale = True),
	# 		dimensions = list([
	# 			dict(range = [vec['Welfare'].min(), vec['Welfare'].max()],
	# 				label = 'Welfare', values = vec['Welfare']),
	# 			dict(range = [vec['DegY1.5°C'].min(), vec['DegY1.5°C'].max()],
	# 				label = 'DegY1.5°C', values = vec['DegY1.5°C']),
	# 			dict(range = [vec['DegY2°C'].min(), vec['DegY2°C'].max()],
	# 				label = 'DegY2°C', values = vec['DegY2°C']),
	# 			dict(range = [vec['NPV Damages'].min(), vec['NPV Damages'].max()],
	# 				label = 'NPV Damages', values = vec['NPV Damages']),
	# 			dict(range = [vec['NPV Abatement'].min(), vec['NPV Abatement'].max()],
	# 				label = 'NPV Abatement', values = vec['NPV Abatement']),
	# 			dict(range = [vec['NPV Adaptation'].min(), vec['NPV Adaptation'].max()],
	# 				label = 'NPV Adaptation', values = vec['NPV Adaptation']),
	# 			])
	# 		))
	# # Show the plot
	# fig.show()

# allsols = pd.DataFrame(allsols, columns=['Welfare','DegY1.5°C','DegY2°C','NPV Damages','NPV Abatement','NPV Adaptation','Type'])
# allsols = allsols.loc[allsols['Welfare']<=10]
# for el in allsols.columns:
# 	allsols = allsols.loc[allsols[el]<=allsols.loc[allsols['Type']=='SO'][el].max()]
# # allsols = allsols.loc[allsols['Welfare']<0]
# allsols['TypeID'] = allsols['Type'].astype('category').cat.codes

# vec = allsols
# print(vec)
# fig = go.Figure(data = 
# 	go.Parcoords(
# 		# line = dict(color = vec['Welfare'],
# 		# 	colorscale = 'Tealrose',
# 		# 	showscale = True),
# 		line = dict(color = vec['TypeID'],
# 			colorscale = [[0,'green'],[1,'red']],
# 			showscale = True),
# 		dimensions = list([
# 			dict(range = [vec['Welfare'].min(), vec['Welfare'].max()],
# 				label = 'Welfare', values = vec['Welfare']),
# 			dict(range = [vec['DegY1.5°C'].min(), vec['DegY1.5°C'].max()],
# 				label = 'DegY1.5°C', values = vec['DegY1.5°C']),
# 			dict(range = [vec['DegY2°C'].min(), vec['DegY2°C'].max()],
# 				label = 'DegY2°C', values = vec['DegY2°C']),
# 			dict(range = [vec['NPV Damages'].min(), vec['NPV Damages'].max()],
# 				label = 'NPV Damages', values = vec['NPV Damages']),
# 			dict(range = [vec['NPV Abatement'].min(), vec['NPV Abatement'].max()],
# 				label = 'NPV Abatement', values = vec['NPV Abatement']),
# 			dict(range = [vec['NPV Adaptation'].min(), vec['NPV Adaptation'].max()],
# 				label = 'NPV Adaptation', values = vec['NPV Adaptation']),
# 			])
# 		))
# # Show the plot
# fig.show()
# vec['Welfare'] = -vec['Welfare']
# plt.figure()
# sns.scatterplot(data=vec, x='DegY1.5°C', y='Welfare', hue='Type', size='DegY2°C', edgecolor=None)
# fig, ax = plt.subplots(1,2, sharex=True, sharey=True)
# vec = allsols.loc[allsols['Type']=='SO']
# sns.scatterplot(data=vec, x='DegY1.5°C', y='Welfare', hue='DegY2°C', cmap = 'plasma_r', edgecolor=None, ax = ax[0])
# vec = allsols.loc[allsols['Type']=='DPS']
# sns.scatterplot(data=vec, x='DegY1.5°C', y='Welfare', hue='DegY2°C', cmap = 'plasma_r', edgecolor=None, ax = ax[1])

# vec=allsols
# vec['DegY1.5°C'] = pd.cut(vec['DegY1.5°C'], bins=[x for x in np.linspace(vec['DegY1.5°C'].min(),vec['DegY1.5°C'].max(),6)])
# plt.figure()
# sns.violinplot(data=vec, x='DegY1.5°C', y='Welfare', palette=['forestgreen','darkorchid'], hue='Type', split=True)

# plt.figure()
# sns.boxplot(data=vec, x='DegY1.5°C', y='Welfare', palette=['forestgreen','darkorchid'], hue='Type')
# sns.stripplot(data=vec, x='DegY1.5°C', y='Welfare', palette=['forestgreen','darkorchid'], hue='Type', edgecolor='grey', size=3, dodge=True, alpha=0.)


plt.show()