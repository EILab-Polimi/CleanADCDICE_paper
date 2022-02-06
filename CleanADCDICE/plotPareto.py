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
       'NPV Adaptation_cal', 'Type', 'Welfare', 'P(GMST > 2°C)', 'Warming above 2°C [°C yr]',
       'NPV Damages [10^12 USD]', 'NPV Abat. costs [10^12 USD]', 'NPV Adapt. costs [10^12 USD]']#,
       # 'P > 1.5°C', 'P > 2°C']

allsols['TypeID'] = allsols['Type'].astype('category').cat.codes
allsols['\u0394 CBGE [%]'] = 100 * ((allsols.loc[allsols['Type']=='SO_1obj']['Welfare'].min() / allsols['Welfare'])**(1/(1-1.45)) - 1)
# allsols['\u0394 CBGE [%]'] = 100 * ((-508595.92753430415 / allsols['Welfare'])**(1/(1-1.45)) - 1)
allsols['NPV Total costs [10^12 USD]'] = allsols['NPV Damages [10^12 USD]'] + allsols['NPV Adapt. costs [10^12 USD]'] + allsols['NPV Abat. costs [10^12 USD]']

allsols = allsols.loc[allsols['Welfare']<0]
# allsols = allsols.loc[allsols['\u0394 CBGE [%]'] > allsols.loc[allsols['Type']=='SO']['\u0394 CBGE [%]'].min()]
# allsols = allsols.loc[allsols['\u0394 CBGE [%]'] > -5]

# allsols.columns = ['Welfare','DegYrs 1.5°C [°C]','DegYrs 2°C [°C]','NPV Damages [10^12 USD]','NPV Abat. costs [10^12 USD]','NPV Adapt. costs [10^12 USD]','Years above 1.5°C','Years above 2°C','Type','TypeID','\u0394 CBGE [%]']
# allsols = allsols[['\u0394 CBGE [%]','DegYrs 1.5°C [°C]','DegYrs 2°C [°C]','NPV Damages [10^12 USD]','NPV Abat. costs [10^12 USD]','NPV Adapt. costs [10^12 USD]','Years above 1.5°C','Years above 2°C','Type','TypeID','Welfare']]
# allsols = allsols.loc[ allsols['\u0394 CBGE [%]'] >= allsols.loc[allsols['Type']=='SO']['\u0394 CBGE [%]'].min() ]

# allsols['Method'] = ['Static intertemporal' if x =='SO' else 'Self-adaptive' for x in allsols['Type'] ]
# plt.figure()
# allsols = allsols.sort_values(by='Type', ascending=True)
# sns.scatterplot(data=allsols, x='Warming above 2°C [°C yr]', y='\u0394 CBGE [%]', hue='Method', palette=['forestgreen','darkorchid'], alpha=0.3, edgecolor=None)
# # plt.figure()
# # sns.scatterplot(data=allsols, x='P(GMST > 2°C)', y='\u0394 CBGE [%]', hue='Method', palette=['darkorchid','forestgreen'], alpha=0.5, edgecolor=None)
# allsols = allsols.drop(['Method'], axis=1)


# allsols_scaled = allsols.copy()
# columns_to_scale = ['Welfare',
#        'P(GMST > 2°C)', 'Warming above 2°C [°C yr]', 'NPV Damages [10^12 USD]',
#        'NPV Abat. costs [10^12 USD]', 'NPV Adapt. costs [10^12 USD]',
#        'Δ CBGE [%]', 'NPV Total costs [10^12 USD]']
# allsols_scaled[columns_to_scale] = preprocessing.MinMaxScaler().fit_transform(allsols[columns_to_scale].values)

# fig, ax = plt.subplots()
# color = ['forestgreen','darkorchid']
# # allsols_scaled = allsols_scaled.sort_values(by='TypeID')
# pd.plotting.parallel_coordinates(frame=allsols_scaled, class_column='Type', 
# 	cols = ['\u0394 CBGE [%]','P(GMST > 2°C)','Warming above 2°C [°C yr]',
#        'NPV Abat. costs [10^12 USD]', 'NPV Damages [10^12 USD]','NPV Adapt. costs [10^12 USD]'], 
# 	color=color, ax = ax, alpha=0.75, linewidth=0.2, axvlines=True, sort_labels=True)
# # plt.show()


# axnew.axes.yaxis.set_visible(False)
# axnew.set_xticks(np.arange(6))
# # xticklabels = [str(round(np.min(df_[columns[el+1]].values)))+'\n'+columns[el+1] for el in range(6)]
# # xticklabels[2] = str(round( - np.min(df_[columns[3]].values)))+'\n'+columns[3]
# # xticklabels[3] = str(round( np.min(df_[columns[4]].values),1))+'\n'+columns[4]
# # axnew.set_xticklabels(xticklabels)
# # axnew2 = axnew.twiny()
# # axnew2.set_xticks(np.arange(6))
# # xticklabels = [str(round(np.max(df_[columns[el+1]].values))) for el in range(6)]
# # xticklabels[2] = str(round( - np.max(df_[columns[3]].values)))
# # xticklabels[3] = str(round( np.max(df_[columns[4]].values)))
# # axnew2.set_xticklabels(xticklabels)
# # axnew2.get_yaxis().set_visible([])
# handles, labels = axnew.get_legend_handles_labels()
# axnew.get_legend().remove()
# plt.text(-.05, 0.5, ' $\\longleftarrow$ Direction of Preference ',
#          horizontalalignment='left',
#          verticalalignment='center',
#          rotation=90,
#          clip_on=False,
#          transform=plt.gca().transAxes)
# # axnew.axes.legend(handles=handles[:3], labels=label, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.17))
# fignew.set_size_inches(10,7)
# fignew.subplots_adjust(top=0.87, bottom=0.11, left=0.08, right=0.92)
# plt.tight_layout()

# print(asd)

# fig, ax = plt.subplots(1,6)
# allsols_vp = allsols.copy()
# allsols_vp['all'] = ""
# allsols_vp = allsols_vp.loc[allsols_vp['Type']!='SO_1obj']
# selectcols = ['\u0394 CBGE [%]','P(GMST > 2°C)','Warming above 2°C [°C yr]',
# 	'NPV Abat. costs [10^12 USD]','NPV Damages [10^12 USD]','NPV Adapt. costs [10^12 USD]']
# for el in range(6): 
# 	sns.violinplot(data = allsols_vp, x="all", y=selectcols[el], hue='Type', ax=ax[el], 
# 		split=True, palette=['darkorchid','forestgreen'], inner='quartile', cut=0)
# 	ax[el].set_xlabel("")
# ax[0].invert_yaxis()


# vec = allsols
# vec = vec.sort_values(by='Type', ascending='True')
# fig = go.Figure(data = 
# 	go.Parcoords(
# 		line = dict(color = vec['TypeID'],
# 			colorscale = [[0,'forestgreen'],[0.5,'darkorchid'], [1,'royalblue']],
# 			showscale = False),
# 		# line = dict(color = vec['\u0394 CBGE [%]'],
# 		# 	colorscale = 'viridis_r',
# 		# 	showscale = True),
# 		uirevision=dict(colorbartitle='\u0394 CBGE [%]') ,

# 		dimensions = list([
# 			# dict(range = [vec['\u0394 CBGE'].min(), vec['\u0394 CBGE'].max()],
# 			# 	label = '\u0394 CBGE', values = vec['\u0394 CBGE']),
# 			dict(range = [vec['\u0394 CBGE [%]'].max(), vec['\u0394 CBGE [%]'].min()],
# 				label = '\u0394 CBGE [%]', values = vec['\u0394 CBGE [%]']),
# 			# dict(range = [vec['Welfare'].min(), vec['Welfare'].max()],
# 			# 	label = 'Welfare', values = vec['Welfare']),
# 			dict(range = [vec['P(GMST > 2°C)'].min(), vec['P(GMST > 2°C)'].max()],
# 				label = 'P(GMST > 2°C)', values = vec['P(GMST > 2°C)']),
# 			dict(range = [vec['Warming above 2°C [°C yr]'].min(), vec['Warming above 2°C [°C yr]'].max()],
# 				label = 'Warming above 2°C [°C yr]', values = vec['Warming above 2°C [°C yr]']),
# 			# dict(range = [vec['P > 1.5°C'].min(), vec['P > 1.5°C'].max()],
# 			# 	label = 'P > 1.5°C', values = vec['P > 1.5°C']),
# 			# dict(range = [vec['P > 2°C'].min(), vec['P > 2°C'].max()],
# 			# 	label = 'P > 2°C', values = vec['P > 2°C']),
# 			dict(range = [vec['NPV Abat. costs [10^12 USD]'].min(), vec['NPV Abat. costs [10^12 USD]'].max()],
# 				label = 'NPV Abat. costs [10^12 USD]', values = vec['NPV Abat. costs [10^12 USD]']),
# 			dict(range = [vec['NPV Damages [10^12 USD]'].min(), vec['NPV Damages [10^12 USD]'].max()],
# 				label = 'NPV Damages [10^12 USD]', values = vec['NPV Damages [10^12 USD]']),
# 			dict(range = [vec['NPV Adapt. costs [10^12 USD]'].min(), vec['NPV Adapt. costs [10^12 USD]'].max()],
# 				label = 'NPV Adapt. costs [10^12 USD]', values = vec['NPV Adapt. costs [10^12 USD]']),
# 			# dict(range = [vec['NPV Total costs [10^12 USD]'].min(), vec['NPV Total costs [10^12 USD]'].max()],
# 			# 	label = 'NPV Total costs [10^12 USD]', values = vec['NPV Total costs [10^12 USD]']),
# 			])
# 		)
# 	# , layout = go.Layout(height=800, width=1600)
# 	)
# # fig.update_layout(title_text='Self-adaptive (green) and intertemporal optimization (purple) solutions in the objectives<span>&#39;</span> space')
# fig.update_layout(font=dict(
#         family="Helvetica",
#         size=20),
# 	margin=dict(l=150, r=150)
# )
# # fig.write_image("Figures/Parallel.pdf")
# fig.show()

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

# newallsols = pd.concat([new_so, new_dps, allsols.loc[allsols['Type']=='SO_1obj']], ignore_index=True)

allsols = newallsols.copy()

# allsols['Method'] = ['Static intertemporal' if x =='SO' else 'Self-adaptive' for x in allsols['Type'] ]
# plt.figure()
# allsols = allsols.sort_values(by='Type', ascending=True)

# sns.scatterplot(data=allsols, x='Warming above 2°C [°C yr]', y='\u0394 CBGE [%]', hue='Method', palette=['forestgreen','darkorchid'], alpha=0.3, edgecolor=None)
# # plt.figure()
# # sns.scatterplot(data=allsols, x='P(GMST > 2°C)', y='\u0394 CBGE [%]', hue='Method', palette=['darkorchid','forestgreen'], alpha=0.5, edgecolor=None)
# allsols = allsols.drop(['Method'], axis=1)


# df = allsols.to_list()
# df_ = pd.DataFrame(allsols[6:13], allsols.columns[6:13])
# x = np.transpose(np.transpose(df_)[1:]) 
# df_[df_.columns[1:]] = df_[df_.columns[1:]].astype(float)
# df_[df_.columns[0]] = df_[df_.columns[0]].astype(str)
# min_max_scaler = preprocessing.MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(x)

# df2 = np.transpose([np.transpose(df[6:13])[0],*np.transpose(x_scaled)])
# df2 = pd.DataFrame(df2,columns=df_.columns)
# df2[df_.columns[1:]] = df2[df_.columns[1:]].astype(float)
# fignew, axnew = plt.subplots()
# color = ['forestgreen','darkorchid']
# pd.plotting.parallel_coordinates(df2,'Type', color=color, ax = axnew, alpha=0.65, linewidth=1)

# axnew.axes.yaxis.set_visible(False)
# axnew.set_xticks(np.arange(6))
# # xticklabels = [str(round(np.min(df_[columns[el+1]].values)))+'\n'+columns[el+1] for el in range(6)]
# # xticklabels[2] = str(round( - np.min(df_[columns[3]].values)))+'\n'+columns[3]
# # xticklabels[3] = str(round( np.min(df_[columns[4]].values),1))+'\n'+columns[4]
# # axnew.set_xticklabels(xticklabels)
# # axnew2 = axnew.twiny()
# # axnew2.set_xticks(np.arange(6))
# # xticklabels = [str(round(np.max(df_[columns[el+1]].values))) for el in range(6)]
# # xticklabels[2] = str(round( - np.max(df_[columns[3]].values)))
# # xticklabels[3] = str(round( np.max(df_[columns[4]].values)))
# # axnew2.set_xticklabels(xticklabels)
# # axnew2.get_yaxis().set_visible([])
# handles, labels = axnew.get_legend_handles_labels()
# axnew.get_legend().remove()
# plt.text(-.05, 0.5, ' $\\longleftarrow$ Direction of Preference ',
#          horizontalalignment='left',
#          verticalalignment='center',
#          rotation=90,
#          clip_on=False,
#          transform=plt.gca().transAxes)
# # axnew.axes.legend(handles=handles[:3], labels=label, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.17))
# fignew.set_size_inches(10,7)
# fignew.subplots_adjust(top=0.87, bottom=0.11, left=0.08, right=0.92)
# plt.tight_layout()

# fig, ax = plt.subplots(1,6)
# allsols['all'] = ""
# selectcols = ['\u0394 CBGE [%]','P(GMST > 2°C)','Warming above 2°C [°C yr]',
# 	'NPV Abat. costs [10^12 USD]','NPV Damages [10^12 USD]','NPV Adapt. costs [10^12 USD]']
# for el in range(6): 
# 	sns.violinplot(data = allsols, x="all", y=selectcols[el], hue='Type', ax=ax[el], 
# 		split=True, palette=['darkorchid','forestgreen'], inner='quartile', cut=0)
# 	ax[el].set_xlabel("")
# ax[0].invert_yaxis()

# allsols = allsols.drop(['all'], axis=1)

# for el in allsols.columns[7:13]:
# 	allsols = allsols.loc[allsols[el] <= allsols.loc[allsols['Type']=='SO'][el].max()]

vec = allsols
# fig = go.Figure(data = 
# 	go.Parcoords(
# 		line = dict(color = vec['TypeID'],
# 			colorscale = [[0,'forestgreen'],[0.5,'darkorchid'],[1.0,'royalblue']],
# 			showscale = False),
# 		# line = dict(color = vec['\u0394 CBGE [%]'],
# 		# 	colorscale = 'viridis_r',
# 		# 	showscale = True),
# 		uirevision=dict(colorbartitle='\u0394 CBGE [%]') ,

# 		dimensions = list([
# 			# dict(range = [vec['\u0394 CBGE'].min(), vec['\u0394 CBGE'].max()],
# 			# 	label = '\u0394 CBGE', values = vec['\u0394 CBGE']),
# 			dict(range = [vec['\u0394 CBGE [%]'].max(), vec['\u0394 CBGE [%]'].min()],
# 				label = '\u0394 CBGE [%]', values = vec['\u0394 CBGE [%]']),
# 			# dict(range = [vec['Welfare'].min(), vec['Welfare'].max()],
# 			# 	label = 'Welfare', values = vec['Welfare']),
# 			dict(range = [vec['P(GMST > 2°C)'].min(), vec['P(GMST > 2°C)'].max()],
# 				label = 'P(GMST > 2°C)', values = vec['P(GMST > 2°C)']),
# 			dict(range = [vec['Warming above 2°C [°C yr]'].min(), vec['Warming above 2°C [°C yr]'].max()],
# 				label = 'Warming above 2°C [°C yr]', values = vec['Warming above 2°C [°C yr]']),
# 			# dict(range = [vec['P > 1.5°C'].min(), vec['P > 1.5°C'].max()],
# 			# 	label = 'P > 1.5°C', values = vec['P > 1.5°C']),
# 			# dict(range = [vec['P > 2°C'].min(), vec['P > 2°C'].max()],
# 			# 	label = 'P > 2°C', values = vec['P > 2°C']),
# 			dict(range = [vec['NPV Abat. costs [10^12 USD]'].min(), vec['NPV Abat. costs [10^12 USD]'].max()],
# 				label = 'NPV Abat. costs [10^12 USD]', values = vec['NPV Abat. costs [10^12 USD]']),
# 			dict(range = [vec['NPV Damages [10^12 USD]'].min(), vec['NPV Damages [10^12 USD]'].max()],
# 				label = 'NPV Damages [10^12 USD]', values = vec['NPV Damages [10^12 USD]']),
# 			dict(range = [vec['NPV Adapt. costs [10^12 USD]'].min(), vec['NPV Adapt. costs [10^12 USD]'].max()],
# 				label = 'NPV Adapt. costs [10^12 USD]', values = vec['NPV Adapt. costs [10^12 USD]']),
# 			# dict(range = [vec['NPV Total costs [10^12 USD]'].min(), vec['NPV Total costs [10^12 USD]'].max()],
# 			# 	label = 'NPV Total costs [10^12 USD]', values = vec['NPV Total costs [10^12 USD]']),
# 			])
# 		)
# 	, layout = go.Layout(height=800, width=1600)
# 	)
# # fig.update_layout(title_text='Self-adaptive (green) and intertemporal optimization (purple) solutions in the objectives<span>&#39;</span> space')
# fig.update_layout(font=dict(
#         family="Helvetica",
#         size=20),
# 	margin=dict(l=150, r=150)
# )
# # fig.write_image("Figures/Parallel.pdf")
# fig.show()

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
			dict(range = [vec['Warming above 2°C [°C yr]'].min(), vec['Warming above 2°C [°C yr]'].max()],
				label = 'Warming above 2°C [°C yr]', values = vecdps['Warming above 2°C [°C yr]']),
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
# fig.update_coloraxes(cmin = allsols['\u0394 CBGE [%]'].min() , cmax = allsols['\u0394 CBGE [%]'].max())
# fig.update_layout(title_text='Self-adaptive (green) and intertemporal optimization (purple) solutions in the objectives<span>&#39;</span> space')
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
			dict(range = [vec['Warming above 2°C [°C yr]'].min(), vec['Warming above 2°C [°C yr]'].max()],
				label = 'Warming above 2°C [°C yr]', values = vecso['Warming above 2°C [°C yr]']),
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
# fig.update_coloraxes()
# fig.data[0].line.colorbar.tickvals=[allsols['\u0394 CBGE [%]'].min(),allsols['\u0394 CBGE [%]'].max()]
# fig.update_layout(title_text='Self-adaptive (green) and intertemporal optimization (purple) solutions in the objectives<span>&#39;</span> space')
fig.update_layout(font=dict(
        family="Helvetica",
        size=20),
	margin=dict(l=100, r=200)
)
fig.write_image("Figures/ParallelSO.pdf")
fig.show()

# nondominated = pareto.eps_sort([list(allsols.itertuples(False))], objectives=[7,8,9,10,11,12], epsilons=epss, kwargs={'maximize':[0]})

# allsols_nd = pd.DataFrame.from_records(nondominated, columns=list(allsols.columns.values))
# # allsols = allsols.loc[allsols['Welfare']<-500000]
# vec = allsols_nd
# fig = go.Figure(data = 
# 	go.Parcoords(
# 		line = dict(color = vec['TypeID'],
# 			colorscale = [[0,'forestgreen'],[0.5,'darkorchid'],[1,'royalblue']],
# 			showscale = False),
# 		# line = dict(color = vec['\u0394 CBGE [%]'],
# 		# 	colorscale = 'viridis_r',
# 		# 	showscale = True),
# 		dimensions = list([
# 			# dict(range = [vec['\u0394 CBGE'].min(), vec['\u0394 CBGE'].max()],
# 			# 	label = '\u0394 CBGE', values = vec['\u0394 CBGE']),
# 			dict(range = [vec['\u0394 CBGE [%]'].max(), vec['\u0394 CBGE [%]'].min()],
# 				label = '\u0394 CBGE [%]', values = vec['\u0394 CBGE [%]']),
# 			# dict(range = [vec['Welfare'].min(), vec['Welfare'].max()],
# 			# 	label = 'Welfare', values = vec['Welfare']),
# 			dict(range = [vec['P(GMST > 2°C)'].min(), vec['P(GMST > 2°C)'].max()],
# 				label = 'P(GMST > 2°C)', values = vec['P(GMST > 2°C)']),
# 			dict(range = [vec['Warming above 2°C [°C yr]'].min(), vec['Warming above 2°C [°C yr]'].max()],
# 				label = 'Warming above 2°C [°C yr]', values = vec['Warming above 2°C [°C yr]']),
# 			# dict(range = [vec['P > 1.5°C'].min(), vec['P > 1.5°C'].max()],
# 			# 	label = 'P > 1.5°C', values = vec['P > 1.5°C']),
# 			# dict(range = [vec['P > 2°C'].min(), vec['P > 2°C'].max()],
# 			# 	label = 'P > 2°C', values = vec['P > 2°C']),
# 			dict(range = [vec['NPV Abat. costs [10^12 USD]'].min(), vec['NPV Abat. costs [10^12 USD]'].max()],
# 				label = 'NPV Abat. costs [10^12 USD]', values = vec['NPV Abat. costs [10^12 USD]']),
# 			dict(range = [vec['NPV Damages [10^12 USD]'].min(), vec['NPV Damages [10^12 USD]'].max()],
# 				label = 'NPV Damages [10^12 USD]', values = vec['NPV Damages [10^12 USD]']),
# 			dict(range = [vec['NPV Adapt. costs [10^12 USD]'].min(), vec['NPV Adapt. costs [10^12 USD]'].max()],
# 				label = 'NPV Adapt. costs [10^12 USD]', values = vec['NPV Adapt. costs [10^12 USD]']),
# 			# dict(range = [vec['NPV Total costs [10^12 USD]'].min(), vec['NPV Total costs [10^12 USD]'].max()],
# 			# 	label = 'NPV Total costs [10^12 USD]', values = vec['NPV Total costs [10^12 USD]']),
# 			])
# 		)
# 	# , layout = go.Layout(height=500, width=1000)
# 	)
# # fig.update_layout(title_text='Self-adaptive (green) and intertemporal optimization (purple) solutions in the objectives<span>&#39;</span> space')
# fig.update_layout(font=dict(
#         family="Helvetica",
#         size=20),
# 	margin=dict(l=150, r=150)
# )
# fig.show()


# allsols_scaled = allsols_nd.copy()
# columns_to_scale = ['Welfare',
#        'P(GMST > 2°C)', 'Warming above 2°C [°C yr]', 'NPV Damages [10^12 USD]',
#        'NPV Abat. costs [10^12 USD]', 'NPV Adapt. costs [10^12 USD]',
#        'Δ CBGE [%]', 'NPV Total costs [10^12 USD]']
# allsols_scaled[columns_to_scale] = preprocessing.MinMaxScaler().fit_transform(allsols_nd[columns_to_scale].values)

# fig, ax = plt.subplots()
# color = ['forestgreen','darkorchid']
# # allsols_scaled = allsols_scaled.sort_values(by='TypeID')
# pd.plotting.parallel_coordinates(frame=allsols_scaled, class_column='Type', 
# 	cols = ['\u0394 CBGE [%]','P(GMST > 2°C)','Warming above 2°C [°C yr]',
#        'NPV Abat. costs [10^12 USD]', 'NPV Damages [10^12 USD]','NPV Adapt. costs [10^12 USD]'], 
# 	color=color, ax = ax, alpha=0.1, linewidth=0.2, axvlines=True, sort_labels=True)
# # plt.show()

# allsols_nd['Method'] = ['Static intertemporal' if x =='SO' else 'Self-adaptive' for x in allsols_nd['Type'] ]
# plt.figure()
# sns.scatterplot(data=allsols_nd, x='Warming above 2°C [°C yr]', y='\u0394 CBGE [%]', hue='Method', palette=['darkorchid','forestgreen'], alpha=0.5)
# plt.figure()
# sns.scatterplot(data=allsols_nd, x='P(GMST > 2°C)', y='\u0394 CBGE [%]', hue='Method', palette=['darkorchid','forestgreen'], alpha=0.5)
# allsols_nd = allsols_nd.drop(['Method'], axis=1)



# plt.figure()
# perc_nd = [[el, 100.0*allsols_nd.loc[allsols_nd['Type']==el].count()[0] / allsols.loc[allsols['Type']==el].count()[0]] for el in allsols['Type'].unique()]
# perc_nd = pd.DataFrame(perc_nd, columns=['Type','Nondominated solutions [%]'])
# perc_nd['Method'] = ['Static intertemporal' if x =='SO' else 'Self-adaptive' for x in perc_nd['Type'] ]
# perc_nd['Nondominated solutions [%]'] = pd.to_numeric(perc_nd['Nondominated solutions [%]'])
# print(perc_nd)
# sns.barplot(data=perc_nd, x='Method', y='Nondominated solutions [%]', palette=['darkorchid', 'forestgreen'])
# plt.figure()
# sns.scatterplot(data=vec, x='DegYrs 2°C [°C]', y='NPV Abat. costs [10^12 USD]', size='NPV Adapt. costs [10^12 USD]', hue='Type', palette=['darkorchid','forestgreen'])
# plt.figure()
# sns.scatterplot(data=vec, x='DegYrs 2°C [°C]', y='NPV Damages [10^12 USD]', hue='Type', palette=['darkorchid','forestgreen'])
# plt.figure()
# sns.scatterplot(data=vec, x='DegYrs 2°C [°C]', size='NPV Abat. costs [10^12 USD]', y='\u0394 CBGE [%]', hue='Type', palette=['darkorchid','forestgreen'])
# plt.figure()
# sns.scatterplot(data=vec, x='DegYrs 2°C [°C]', y='\u0394 CBGE [%]', size='NPV Adapt. costs [10^12 USD]', hue='Type', palette=['darkorchid','forestgreen'])
# plt.figure()
# sns.scatterplot(data=vec, x='DegYrs 2°C [°C]', y='\u0394 CBGE [%]', size='NPV Damages [10^12 USD]', hue='Type', palette=['darkorchid','forestgreen'])
# fig, ax = plt.subplots(1,3, sharey=True)
# sns.scatterplot(ax = ax[0], data=vec, size='DegYrs 2°C [°C]', 
# 	y='\u0394 CBGE [%]', x='NPV Abat. costs [10^12 USD]', 
# 	hue='Type', palette=['darkorchid','forestgreen'])
# sns.scatterplot(ax = ax[1], data=vec, size='DegYrs 2°C [°C]', 
# 	y='\u0394 CBGE [%]', x='NPV Adapt. costs [10^12 USD]', 
# 	hue='Type', palette=['darkorchid','forestgreen'], legend=False)
# sns.scatterplot(ax = ax[2], data=vec, size='DegYrs 2°C [°C]', 
# 	y='\u0394 CBGE [%]', x='NPV Damages [10^12 USD]', 
# 	hue='Type', palette=['darkorchid','forestgreen'], legend=False)
# handles , labels = ax[0].get_legend_handles_labels()
# fig.legend(handles, labels, loc ='lower center', ncol=9)
# ax[0].legend([], [], frameon=False)

# fig, ax = plt.subplots(1,3, sharey=True)
# sns.scatterplot(ax = ax[0], data=vec, x='DegYrs 2°C [°C]', 
# 	size='\u0394 CBGE [%]', y='NPV Abat. costs [10^12 USD]', alpha = 0.25, edgecolor=None,
# 	hue='Type', palette=['darkorchid','forestgreen'])
# sns.scatterplot(ax = ax[1], data=vec, x='DegYrs 2°C [°C]', 
# 	size='\u0394 CBGE [%]', y='NPV Adapt. costs [10^12 USD]', alpha = 0.25, edgecolor=None,
# 	hue='Type', palette=['darkorchid','forestgreen'], legend=False)
# sns.scatterplot(ax = ax[2], data=vec, x='DegYrs 2°C [°C]',  
# 	size='\u0394 CBGE [%]', y='NPV Damages [10^12 USD]', alpha = 0.25, edgecolor=None,
# 	hue='Type', palette=['darkorchid','forestgreen'], legend=False)
# handles , labels = ax[0].get_legend_handles_labels()
# fig.legend(handles, labels, loc ='lower center', ncol=9)
# ax[0].legend([], [], frameon=False)

plt.show()
# username = 'angelocarlino'
# api_key = 'yz46AoJhyNvvOBRwXXkK'
# chart_studio.tools.set_credentials_file(username=username, api_key=api_key)
# fig.write_html("InteractiveParallel.html")
# py.plot(fig, filename='InteractiveParallel', auto_open=False)
# print(chart_studio.tools.get_embed('https://plotly.com/~angelocarlino/1'))