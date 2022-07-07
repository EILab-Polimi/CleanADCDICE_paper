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
from sklearn import preprocessing

sns.set_style('whitegrid')
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

allsols = pd.read_csv('./SimulationValFull.csv')
epss = [25,0.05,5,1,1,1]

allsols.columns = ['Welfare_cal', 'DegY1.5°C_cal', 'DegY2°C_cal', 'NPV Damages_cal', 'NPV Abatement_cal',
       'NPV Adaptation_cal', 'Type', 'Welfare', 'P(GMST > 2°C)', 'Warming above 2°C [°C]',
       'NPV Damages [10^12 USD]', 'NPV Abat. costs [10^12 USD]', 'NPV Adapt. costs [10^12 USD]']#,
       # 'P > 1.5°C', 'P > 2°C']

allsols['TypeID'] = allsols['Type'].astype('category').cat.codes
allsols['\u0394 CBGE [%]'] = 100 * ((allsols.loc[allsols['Type']=='SO_1obj']['Welfare'].min() / allsols['Welfare'])**(1/(1-1.45)) - 1)
# allsols['\u0394 CBGE [%]'] = 100 * ((-508595.92753430415 / allsols['Welfare'])**(1/(1-1.45)) - 1)
allsols['NPV Total costs [10^12 USD]'] = allsols['NPV Damages [10^12 USD]'] + allsols['NPV Adapt. costs [10^12 USD]'] + allsols['NPV Abat. costs [10^12 USD]']

allsols = allsols.loc[allsols['Welfare']<0]


for col in allsols.columns[7:13]:
	allsols = allsols.loc[ allsols[col] <= allsols.loc[allsols['Type']=='SO'][col].max() ]


print(allsols.columns)
newallsols = allsols.copy()
new_so = newallsols.loc[newallsols['Type'].str.contains('SO')]
nondominated = pareto.eps_sort([list(new_so.itertuples(False))], objectives=[7,8,9,10,11,12], epsilons=epss)#, kwargs={'maximize':[0]})
new_so = pd.DataFrame.from_records(nondominated, columns=list(allsols.columns.values))

new_dps = newallsols.loc[newallsols['Type']=='DPS']
nondominated = pareto.eps_sort([list(new_dps.itertuples(False))], objectives=[7,8,9,10,11,12], epsilons=epss)#, kwargs={'maximize':[0]})
new_dps = pd.DataFrame.from_records(nondominated, columns=list(allsols.columns.values))

newallsols = pd.concat([new_so, new_dps], ignore_index=True)


allsols = newallsols.copy()

vec = allsols

vecdps = allsols.loc[allsols['Type']=='DPS']
fig = go.Figure(data = 
	go.Parcoords(
		# line = dict(color = vec['TypeID'],
		# 	colorscale = [[0,'forestgreen'],[0.5,'darkorchid'],[1.0,'royalblue']],
		# 	showscale = False),
		line = dict(color = vecdps['\u0394 CBGE [%]'],
			colorscale = 'viridis_r',
			showscale = True,
			cmin=allsols['\u0394 CBGE [%]'].min(), cmax=allsols['\u0394 CBGE [%]'].max(),
			cmid=0,
			colorbar = dict( title = dict(text='\u0394 CBGE [%]', font=dict(family='Helvetica', size=20) ) )
			),

		dimensions = list([
			# dict(range = [vec['\u0394 CBGE'].min(), vec['\u0394 CBGE'].max()],
			# 	label = '\u0394 CBGE', values = vec['\u0394 CBGE']),
			# dict(range = [vec['Welfare'].min(), vec['Welfare'].max()],
			# 	label = 'Welfare', values = vec['Welfare']),
			dict(range = [vec['P(GMST > 2°C)'].min(), vec['P(GMST > 2°C)'].max()],
				label = 'P(GMST > 2°C)', values = vecdps['P(GMST > 2°C)']),
			dict(range = [vec['\u0394 CBGE [%]'].max(), vec['\u0394 CBGE [%]'].min()],
				label = '\u0394 CBGE [%]', values = vecdps['\u0394 CBGE [%]']),
			dict(range = [vec['Warming above 2°C [°C]'].min(), vec['Warming above 2°C [°C]'].max()],
				label = 'Warming above 2°C [°C]', values = vecdps['Warming above 2°C [°C]']),
			# dict(range = [vec['P > 1.5°C'].min(), vec['P > 1.5°C'].max()],
			# 	label = 'P > 1.5°C', values = vec['P > 1.5°C']),
			# dict(range = [vec['P > 2°C'].min(), vec['P > 2°C'].max()],
			# 	label = 'P > 2°C', values = vec['P > 2°C']),
			dict(range = [vec['NPV Abat. costs [10^12 USD]'].min(), vec['NPV Abat. costs [10^12 USD]'].max()],
				label = 'NPV Abat. costs [10^12 USD]', values = vecdps['NPV Abat. costs [10^12 USD]']),
			dict(range = [vec['NPV Damages [10^12 USD]'].min(), vec['NPV Damages [10^12 USD]'].max()],
				label = 'NPV Damages [10^12 USD]', values = vecdps['NPV Damages [10^12 USD]']),
			dict(range = [vec['NPV Adapt. costs [10^12 USD]'].min(), vec['NPV Adapt. costs [10^12 USD]'].max()],
				label = 'NPV Adapt. costs [10^12 USD]', values = vecdps['NPV Adapt. costs [10^12 USD]']),
			# dict(range = [vec['NPV Total costs [10^12 USD]'].min(), vec['NPV Total costs [10^12 USD]'].max()],
			# 	label = 'NPV Total costs [10^12 USD]', values = vec['NPV Total costs [10^12 USD]']),
			])
		)
	, layout = go.Layout(height=800, width=1600)
	)

fig.update_layout(font=dict(
        family="Helvetica",
        size=20),
	margin=dict(l=100, r=200)
)
fig.write_image("Figures/ParallelDPS.pdf")
fig.show()

vecso = allsols.loc[allsols['Type'].str.contains('SO')]
fig = go.Figure(data = 
	go.Parcoords(
		# line = dict(color = vec['TypeID'],
		# 	colorscale = [[0,'forestgreen'],[0.5,'darkorchid'],[1.0,'royalblue']],
		# 	showscale = False),
		line = dict(color = vecso['\u0394 CBGE [%]'],
			colorscale = 'viridis_r',
			showscale = True,
			cmin=allsols['\u0394 CBGE [%]'].min(), cmax=allsols['\u0394 CBGE [%]'].max(),
			cmid=0,
			colorbar = dict( title = dict(text='\u0394 CBGE [%]', font=dict(family='Helvetica', size=20) ) )
			),


		dimensions = list([
			# dict(range = [vec['\u0394 CBGE'].min(), vec['\u0394 CBGE'].max()],
			# 	label = '\u0394 CBGE', values = vec['\u0394 CBGE']),
			# dict(range = [vec['Welfare'].min(), vec['Welfare'].max()],
			# 	label = 'Welfare', values = vec['Welfare']),
			dict(range = [vec['P(GMST > 2°C)'].min(), vec['P(GMST > 2°C)'].max()],
				label = 'P(GMST > 2°C)', values = vecso['P(GMST > 2°C)']),
			dict(range = [vec['\u0394 CBGE [%]'].max(), vec['\u0394 CBGE [%]'].min()],
				label = '\u0394 CBGE [%]', values = vecso['\u0394 CBGE [%]']),
			dict(range = [vec['Warming above 2°C [°C]'].min(), vec['Warming above 2°C [°C]'].max()],
				label = 'Warming above 2°C [°C]', values = vecso['Warming above 2°C [°C]']),
			# dict(range = [vec['P > 1.5°C'].min(), vec['P > 1.5°C'].max()],
			# 	label = 'P > 1.5°C', values = vec['P > 1.5°C']),
			# dict(range = [vec['P > 2°C'].min(), vec['P > 2°C'].max()],
			# 	label = 'P > 2°C', values = vec['P > 2°C']),
			dict(range = [vec['NPV Abat. costs [10^12 USD]'].min(), vec['NPV Abat. costs [10^12 USD]'].max()],
				label = 'NPV Abat. costs [10^12 USD]', values = vecso['NPV Abat. costs [10^12 USD]']),
			dict(range = [vec['NPV Damages [10^12 USD]'].min(), vec['NPV Damages [10^12 USD]'].max()],
				label = 'NPV Damages [10^12 USD]', values = vecso['NPV Damages [10^12 USD]']),
			dict(range = [vec['NPV Adapt. costs [10^12 USD]'].min(), vec['NPV Adapt. costs [10^12 USD]'].max()],
				label = 'NPV Adapt. costs [10^12 USD]', values = vecso['NPV Adapt. costs [10^12 USD]']),
			# dict(range = [vec['NPV Total costs [10^12 USD]'].min(), vec['NPV Total costs [10^12 USD]'].max()],
			# 	label = 'NPV Total costs [10^12 USD]', values = vec['NPV Total costs [10^12 USD]']),
			])
		)
	, layout = go.Layout(height=800, width=1600)
	)

fig.update_layout(font=dict(
        family="Helvetica",
        size=20),
	margin=dict(l=100, r=200)
)
fig.write_image("Figures/ParallelSO.pdf")
fig.show()
