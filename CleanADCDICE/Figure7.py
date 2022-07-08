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
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

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

def plot_prctiles(df, col, prc=[0.5, 0.17, 0.73], color='tab:blue', 
       ax=None, linestyle='-', plotmean=False):
       if ax==None:
              fig, ax = plt.subplots()
       lines = []
       meanline = []
       if plotmean==True:
              for y in df['YEAR'].unique():
                     meanline.append(df.loc[df['YEAR']==y][col].mean())
       for p in prc:
              line = []
              for y in df['YEAR'].unique():
                     line.append(df.loc[df['YEAR']==y][col].quantile(p))
              lines.append(line)
       if plotmean==True:
              ax.plot(df['YEAR'].unique(), meanline, color=color, linestyle='--')
       ax.plot(df['YEAR'].unique(), lines[0], color=color, linestyle=linestyle)
       # print(meanline[0], lines[0][0])
       ax.fill_between(df['YEAR'].unique(), lines[1], lines[2], facecolor=color, alpha=0.1, zorder=-1)
       ax.set_ylabel(col)
       return ax

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
                     newfile[el] = 'writefile 1'
              if file[el].split()[0] == 'scc':
                     newfile[el] = 'scc 1'
       with open('./src/config/config.txt','w') as f:
              for el in range(len(newfile)):
                     f.write(newfile[el])
                     if el < len(newfile)-1:
                            f.write("\n")
       output = subprocess.check_output('./ADCDICE2016 '+str(seed)+' < sol.txt', shell=True).decode("utf-8")
       output = [float(x) for x in output.split('\n')[0].split()]
       print(output)
       return output

# allsols = pd.read_csv('./SimulationValFull.csv')
# epss = [25,0.05,5,1,1,1]

# allsols.columns = ['Welfare_cal', 'DegY1.5°C_cal', 'DegY2°C_cal', 'NPV Damages_cal', 'NPV Abatement_cal',
#        'NPV Adaptation_cal', 'Type', 'Welfare', 'P(GMST > 2°C)', 'Warming above 2°C [°C]',
#        'NPV Damages [10^12 USD]', 'NPV Abat. costs [10^12 USD]', 'NPV Adapt. costs [10^12 USD]']#,
#        # 'P > 1.5°C', 'P > 2°C']

# allsols['TypeID'] = allsols['Type'].astype('category').cat.codes
# allsols['\u0394 CBGE [%]'] = 100 * ((allsols.loc[allsols['Type']=='SO_1obj']['Welfare'].min() / allsols['Welfare'])**(1/(1-1.45)) - 1)
# allsols['NPV Total costs [10^12 USD]'] = allsols['NPV Damages [10^12 USD]'] + allsols['NPV Adapt. costs [10^12 USD]'] + allsols['NPV Abat. costs [10^12 USD]']

# allsols = allsols.loc[allsols['Welfare']<0]

# epss = [25,1]
# dps = allsols.loc[allsols['Type']=='DPS']
# nondom = pareto.eps_sort([list(dps.itertuples(False))], objectives=[7,9], epsilons=epss, kwargs={'maximize':[0]})
# dps = pd.DataFrame.from_records(nondom, columns=list(dps.columns.values))
# so = allsols.loc[allsols['Type'].str.contains('SO')]
# nondom = pareto.eps_sort([list(so.itertuples(False))], objectives=[7,9], epsilons=epss, kwargs={'maximize':[0]})
# so = pd.DataFrame.from_records(nondom, columns=list(so.columns.values))
# allsols = pd.concat([so,dps], ignore_index=True)
# allsols = allsols.loc[allsols['\u0394 CBGE [%]']>=-1.2794494499268727] ## selects same solution as in figure 2, removing solutions worse than static deterministic w.r.t. welfare
# selects = allsols
# selects = selects[[x for x in selects.columns[:6]]].values

# nseeds = 5
# nobjs = 6
# path = './'
# RNTS = {}
# srand = np.random.randint(0,1e6)
# class Rnt:
#        def __init__(self):
#               self.NFE = []
#               self.SBX = []
#               self.DE = []
#               self.PCX = []
#               self.SPX = []
#               self.UNDX = []
#               self.UM = []
#               self.IMP = []
#               self.RES = []
#               self.ARCSIZE = []
#               self.OBJS = []
#               self.PARAMS = []

# allsols = []
# folders = ['BorgOutput_SO_AD_UNC','BorgOutput_SO_AD_UNC_6OBJS','BorgOutput_DPS_AD_UNC_6OBJS']

# all_val_objs = []
# all_cal_objs = []
# sccs = []
# mcabates = []
# tbp = []
# for select in selects:
#        flag = 0
#        for folder in folders:
#               if flag == 1:
#                      break
#               print(folder)
#               nobjs = 1
#               ref = np.asarray([round(x,4) for x in select])
#               nobjs = 6
#               if sum(ref[1:-1]) == 0.0:
#                      print("here")
#                      with open('./'+folder+'/optADCDICE2016.reference') as f:
#                             file = f.read()
#                      ref = [round(float(x),4) for x in file.split("\n")[:-1][0].split(" ")]
#                      nobjs = 1
#               sols = []
#               for seed in range(nseeds):
#                      with open('./'+folder+'/optADCDICE2016_'+str(seed+1)+'.out') as f:
#                             file = f.read()
#                      outfile = [x.split(' ') for x in file.split("\n")[12:-3]]
#                      for idx in range(len(outfile)):
#                             outfile[idx] = [float(x) for x in outfile[idx]]
#                      for el in outfile:
#                             if len(el) > nobjs and \
#                             all(np.asarray([round(x,4) for x in el[-nobjs:]]) == ref):
#                                    sols.append([float(x) for x in el])
#               niter = 1000
#               seed = 2
#               adaptive = 0
#               dec = 'SO'
#               print(ref)
#               if 'DPS' in folder:
#                      adaptive=1
#                      dec = 'DPS'
#               for sol in sols:
#                      print('simulating')
#                      val_obj = simulate(sol[:-nobjs], adaptive=adaptive, niter=niter, seed=seed)
#                      mcabate = pd.read_csv('./SimulationsOutput.txt', sep='\t')[['YEAR','MCABATE']]
#                      scc = pd.read_csv('./SCC.txt', sep='\t', names=['ITER','YEAR','SCC [USD/tCO2]'])

#                      tbp.append([val_obj[0], val_obj[2], adaptive, 
#                             scc.loc[scc['YEAR']==2020]['SCC [USD/tCO2]'].median(),
#                             scc.loc[scc['YEAR']==2025]['SCC [USD/tCO2]'].median(),
#                             scc.loc[scc['YEAR']==2030]['SCC [USD/tCO2]'].median()])
#                      if dec=='DPS':
#                             scc['Method'] = 'Self-Adaptive'
#                             mcabate['Method'] = 'Self-Adaptive'
#                      else:
#                             scc['Method'] = 'Static intertemporal'
#                             mcabate['Method'] = 'Static intertemporal'
#                      sccs.append(scc)
#                      mcabates.append(mcabate)
#                      flag = 1


# print(tbp)
# tbp = pd.DataFrame(tbp)
# tbp.columns = ['Welfare','Warming above 2°C [°C]','Type','SCC(2020) [USD/tCO2]','SCC(2025) [USD/tCO2]','SCC(2030) [USD/tCO2]']
# print(tbp.loc[tbp['Type']==0]['Welfare'].min())
# tbp['\u0394 CBGE [%]'] = 100 * ((tbp.loc[tbp['Type']==0]['Welfare'].min() / tbp['Welfare'])**(1/(1-1.45)) - 1)
# tbp['Method'] = ['Self-adaptive' if x==1 else 'Static intertemporal' for x in tbp['Type'] ]
# tbp.to_csv('./SCC_output.txt')

### uncomment above to run simulations and compute SCC


tbp = pd.read_csv('./SCC_output.txt')
allsols = tbp

fig, ax = plt.subplots(1,2, sharey=True)
markers=['^','o']
for el in range(len(allsols['Type'].unique())):
       seltype = allsols['Type'].unique()[el]
       sel = allsols.loc[allsols['Type']==seltype]
       cm = ax[0].scatter(sel['Warming above 2°C [°C]'].values.tolist(),
              sel['SCC(2020) [USD/tCO2]'].values.tolist(), c=sel['\u0394 CBGE [%]'].values.tolist(), 
              cmap='viridis_r', vmin=tbp['\u0394 CBGE [%]'].min(),  vmax = tbp['\u0394 CBGE [%]'].max(), 
              marker=markers[el], alpha=0.8, label=sel['Method'].values[0], edgecolor='k', s=50, linewidth=.5)
       ax[1].scatter(sel['Warming above 2°C [°C]'].values.tolist(),
              sel['SCC(2030) [USD/tCO2]'].values.tolist(), c=sel['\u0394 CBGE [%]'].values.tolist(), 
              cmap='viridis_r', vmin=tbp['\u0394 CBGE [%]'].min(),  vmax = tbp['\u0394 CBGE [%]'].max(), 
              marker=markers[el], alpha=0.8, label=sel['Method'].values[0], edgecolor='k', s=50, linewidth=.5)

ax[0].set_xlabel('Warming above 2°C [°C]')
ax[1].set_xlabel('Warming above 2°C [°C]')
ax[0].set_ylabel('Median SCC [USD $\mathregular{{tCO}_{2}^{-1}}$]')
plt.tight_layout()
plt.subplots_adjust(bottom=0.05)
cbar = plt.colorbar(cm, ax=ax[0:2], shrink=0.6, location='bottom')
cbar.set_label('\u0394 CBGE [%]')
plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), ncol=1)
plt.gcf().set_size_inches(9/1.25, 5/1.25)
plt.figure()
legend_elements = [Line2D([0], [0], color='w', markeredgecolor='k', marker='^', lw=0, label='Static intertemporal',
                     markersize=7.5),
                   Line2D([0], [0], marker='o', color='w', markeredgecolor='k', lw=0, label='Self-adaptive',
                     markersize=7.5)]
plt.legend(handles=legend_elements)

plt.show()

