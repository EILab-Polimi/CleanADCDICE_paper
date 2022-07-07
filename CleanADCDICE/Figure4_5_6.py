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
from matplotlib.lines import Line2D

sns.set_style('whitegrid')
sns.set_style('ticks')
sns.set_context('paper')

figsize=(4,3.25)

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
       ax=None, linestyle='-', plotmean=False, label=None):
       if ax==None:
              fig, ax = plt.subplots(figsize=figsize)
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
              ax.plot(df['YEAR'].unique(), meanline, color=color, linestyle=None, marker='*')
       ax.plot(df['YEAR'].unique(), lines[0], color=color, linestyle=linestyle, label=label)
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

allsols = pd.read_csv('./SimulationValFull.csv')
epss = [25,0.05,5,1,1,1]

allsols.columns = ['Welfare_cal', 'DegY1.5°C_cal', 'DegY2°C_cal', 'NPV Damages_cal', 'NPV Abatement_cal',
       'NPV Adaptation_cal', 'Type', 'Welfare', 'P(GMST > 2°C)', 'Warming above 2°C [°C yr]',
       'NPV Damages [10^12 USD]', 'NPV Abat. costs [10^12 USD]', 'NPV Adapt. costs [10^12 USD]']#,

allsols['TypeID'] = allsols['Type'].astype('category').cat.codes
allsols['\u0394 CBGE [%]'] = 100 * ((allsols.loc[allsols['Type']=='SO_1obj']['Welfare'].min() / allsols['Welfare'])**(1/(1-1.45)) - 1)
allsols['NPV Total costs [10^12 USD]'] = allsols['NPV Damages [10^12 USD]'] + allsols['NPV Adapt. costs [10^12 USD]'] + allsols['NPV Abat. costs [10^12 USD]']

allsols = allsols.loc[allsols['Welfare']<0]

for col in allsols.columns[7:13]:
       allsols = allsols.loc[ allsols[col] <= allsols.loc[allsols['Type']=='SO'][col].max() ]

new_dps = allsols.loc[allsols['Type']=='DPS']
nondominated = pareto.eps_sort([list(new_dps.itertuples(False))], objectives=[7,8,9,10,11,12], epsilons=epss)#, kwargs={'maximize':[0]})
new_dps = pd.DataFrame.from_records(nondominated, columns=list(allsols.columns.values))

new_dps = new_dps.loc[(new_dps[new_dps.columns[14]] >= -0.5)]

for el in new_dps.columns[7:11]:
       new_dps[el+'_rank'] = new_dps[el].rank(ascending=True)
new_dps['rank'] = new_dps[[x for x in new_dps.columns if '_rank' in x]].max(axis=1)
select = new_dps.loc[new_dps['rank']==new_dps['rank'].min()]
print(select.values)
select = select.iloc[0].values[:6]


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

allsols = []
folders = ['BorgOutput_DPS_AD_UNC_6OBJS']

all_val_objs = []
all_cal_objs = []
for folder in folders:
       ref = np.asarray([round(x,4) for x in select])
       sols = []
       for seed in range(nseeds):
              with open('./'+folder+'/optADCDICE2016_'+str(seed+1)+'.out') as f:
                     file = f.read()
              outfile = [x.split(' ') for x in file.split("\n")[12:-3]]
              for idx in range(len(outfile)):
                     outfile[idx] = [float(x) for x in outfile[idx]]
              for el in outfile:
                     if len(el) > nobjs and \
                     all(np.asarray([round(x,4) for x in el[-nobjs:]]) == ref):
                            sols.append([float(x) for x in el])
       niter = 1000
       seed = 2
       adaptive=1
       dec = 'DPS'
       for sol in sols:
              val_obj = simulate(sol[:-nobjs], adaptive=adaptive, niter=niter, seed=seed)
print(val_obj)

output = pd.read_csv('./SimulationsOutput.txt', sep='\t')

output['MIU'] = output['MIU']*100
output['S'] = output['S']*100
output['IA'] = output['IA']*100
output['FAD'] = output['FAD']*100

output['ADAPTEFF_d'] = pd.cut(output['ADAPTEFF'], bins=[-0.001,0.1,0.7,1.3,2.0], labels=['Zero','Low','Nominal','High'])
output['DAMTYPE_d'] = pd.cut(output['DAMTYPE'], bins=[0.9,1.1,2.1], labels=['Level','Growth'])
output['Non-CO2 RF'] = pd.cut(output['RFOTH'], bins=[-0.1,0.1,1.1,2.1,3.1], labels=['RCP2.6','RCP4.5','RCP6.0','RCP8.5'])
output['TCR'] = pd.cut(output['TCR'], bins=[-0.1,1.4,2.2,10.0], labels=['Lower than likely','Likely','Higher than likely'])
output['Abatement type'] = pd.cut(output['GRUBB'], bins=[-0.1,0.8,1.2], labels=['Standard','Pliable'])
output.columns = ['Damages type' if x=='DAMTYPE_d' else x for x in output.columns] 
output.columns = ['Adapt. eff.' if x=='ADAPTEFF_d' else x for x in output.columns] 
output.columns = ['GMST [°C]' if x=='TATM' else x for x in output.columns] 
output.columns = ['Emission control [%]' if x=='MIU' else x for x in output.columns] 
output.columns = ['Adapt. invest. [%GDP]' if x=='IA' else x for x in output.columns] 
output.columns = ['Flow adapt. [%GDP]' if x=='FAD' else x for x in output.columns] 
output.columns = ['$\mathregular{CO_{2}}$ Ind. Emissions [Gt$\mathregular{CO_{2}}$]' if x=='EIND' else x for x in output.columns] 
output['YGROSS'] = output['GDP'] + output['DAMAGES'] + output['ABATECOST']
output['CUMCEM'] = output['CEMUTOTPERT'] * 5

m_cost = []
d_cost = []
a_cost = []
t_cost = []
cons = []
welfare = []
for x in output['NITER'].unique():
       sel = output.loc[output['NITER']==x]
       temp = sel['ABATECOST'] * sel['RI']
       m_cost.append(temp.sum())
       temp = sel['DAMAGES'] * sel['RI']
       d_cost.append(temp.sum())
       temp = sel['ADAPTCOST'] * sel['RI']
       a_cost.append(temp.sum())
       temp = sel['GDP'] \
              * (1 - sel['S']) * sel['RI']
       cons.append(temp.sum())
       welfare.append(sel['CUMCEM'].sum())
       t_cost.append(m_cost[-1] + d_cost[-1] + a_cost[-1])
dps_costs = [m_cost, d_cost, a_cost, t_cost, cons, welfare]

ax1 = plot_prctiles(output, 'GMST [°C]', color='forestgreen')
ax1.set_xlim((2020,2150))
ax1.set_ylim((1.0, 4.0))
ax1.plot([x for x in range(2020,2150)],[2 for x in range(2020,2150)],'k--', alpha=0.5, zorder=-1)
ax1.plot([x for x in range(2020,2150)],[1.5 for x in range(2020,2150)],'k--', alpha=0.5, zorder=-1)
plt.tight_layout()
ax2 = plot_prctiles(output, '$\mathregular{CO_{2}}$ Ind. Emissions [Gt$\mathregular{CO_{2}}$]', color='forestgreen')
ax2.set_xlim((2020,2150))
ax2.set_ylim((-25.0, 50.0))
ax2.plot([x for x in range(2020,2150)],[0.0 for x in range(2020,2150)],'k--', alpha=0.5, zorder=-1)
plt.tight_layout()

custom_dict_ad = {'High': 0, 'Nominal': 1, 'Low': 2, 'Zero': 3} 
custom_dict_tcr = {'Higher than likely': 0, 'Likely': 1, 'Lower than likely': 2} 
custom_dict_dm = {'Level': 0, 'Growth': 1} 
custom_dict_ab = {'Standard': 0, 'Pliable': 1} 
custom_dict = {'Adapt. eff.': custom_dict_ad,'Damages type': custom_dict_dm,'TCR': custom_dict_tcr, 'Abatement type': custom_dict_ab}


filters = ['Adapt. eff.','Damages type','TCR']
for filt in filters:
       colors = ['#c59824','#397ea1','#c94933','#ab1818']
       if filt=='Damages type':
              colors = ['#397ea1','#ab1818']
       fig, ax = plt.subplots(figsize=figsize)
       count = 0
       cats = output[filt].unique().tolist()
       cats.sort( key=lambda x: custom_dict[filt][x])
       for cat in cats:
              print(cat)
              sel = output.loc[output[filt]==cat]
              ax = plot_prctiles(sel, '$\mathregular{CO_{2}}$ Ind. Emissions [Gt$\mathregular{CO_{2}}$]', ax=ax, color=colors[count], label=cat)
              count += 1       
       ax.legend()
       ax.set_xlim((2020,2150))
       ax.plot([x for x in range(2020,2150)],[0.0 for x in range(2020,2150)],'k--', alpha=0.5, zorder=-1)
       plt.tight_layout()


output['Adaptation costs [%GDP]'] = output['Flow adapt. [%GDP]'] + output['Adapt. invest. [%GDP]']
filters = ['Adapt. eff.','Damages type','TCR']
for filt in filters:
       colors = ['#c59824','#397ea1','#c94933','#ab1818']
       if filt=='Damages type':
              colors = ['#397ea1','#ab1818']
       fig, ax = plt.subplots(figsize=figsize)
       count = 0
       cats = output[filt].unique().tolist()
       cats.sort( key=lambda x: custom_dict[filt][x])
       for cat in cats:
              print(cat)
              sel = output.loc[output[filt]==cat]
              ax = plot_prctiles(sel, 'Adaptation costs [%GDP]', ax=ax, color=colors[count], label=cat)
              count += 1
       ax.legend()
       ax.set_xlim((2020,2150))
       ax.set_ylim((0.0,2.5))
       plt.tight_layout()

folders = ['BorgOutput_SO_AD_UNC']

all_val_objs = []
all_cal_objs = []
sols = []
nobjs = 1
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
       niter = 1000
       seed = 2
       adaptive=0
       dec = 'DPS'
       for sol in sols:
              val_obj = simulate(sol[:-nobjs], adaptive=adaptive, niter=niter, seed=seed)
       print(val_obj)

output = pd.read_csv('./SimulationsOutput.txt', sep='\t')

output['MIU'] = output['MIU']*100
output['S'] = output['S']*100
output['IA'] = output['IA']*100
output['FAD'] = output['FAD']*100

output['ADAPTEFF_d'] = pd.cut(output['ADAPTEFF'], bins=[-0.001,0.0001,0.7,1.3,2.0], labels=['Zero','Low','Nominal','High'])
output['DAMTYPE_d'] = pd.cut(output['DAMTYPE'], bins=[0.9,1.1,2.1], labels=['Level','Growth'])
output['Non-CO2 RF'] = pd.cut(output['RFOTH'], bins=[-0.1,0.1,1.1,2.1,3.1], labels=['RCP2.6','RCP4.5','RCP6.0','RCP8.5'])
output['TCR'] = pd.cut(output['TCR'], bins=[-0.1,1.4,2.2,10.0], labels=['Lower than likely','Likely','Higher than likely'])
output['Abatement type'] = pd.cut(output['GRUBB'], bins=[-0.1,0.8,1.2], labels=['Standard','Pliable'])
output.columns = ['Damages type' if x=='DAMTYPE_d' else x for x in output.columns] 
output.columns = ['Adapt. eff.' if x=='ADAPTEFF_d' else x for x in output.columns] 
output.columns = ['GMST [°C]' if x=='TATM' else x for x in output.columns] 
output.columns = ['Emission control [%]' if x=='MIU' else x for x in output.columns] 
output.columns = ['Adapt. invest. [%GDP]' if x=='IA' else x for x in output.columns] 
output.columns = ['Flow adapt. [%GDP]' if x=='FAD' else x for x in output.columns] 
output.columns = ['$\mathregular{CO_{2}}$ Ind. Emissions [Gt$\mathregular{CO_{2}}$]' if x=='EIND' else x for x in output.columns] 
output['YGROSS'] = output['GDP'] + output['DAMAGES']
output['CUMCEM'] = output['CEMUTOTPERT'] * 5

m_cost = []
d_cost = []
a_cost = []
t_cost = []
cons = []
welfare = []
for x in output['NITER'].unique():
       sel = output.loc[output['NITER']==x]
       temp = sel['ABATECOST'] * sel['RI']
       m_cost.append(temp.sum())
       temp = sel['DAMAGES'] * sel['RI']
       d_cost.append(temp.sum())
       temp = sel['ADAPTCOST'] * sel['RI']
       a_cost.append(temp.sum())
       temp = sel['GDP'] \
              * (1 - sel['S']) * sel['RI']
       cons.append(temp.sum())
       welfare.append(sel['CUMCEM'].sum())
       cons.append(temp.sum())
       t_cost.append(m_cost[-1] + d_cost[-1] + a_cost[-1])
so_costs = [m_cost, d_cost, a_cost, t_cost, cons, welfare]


ax1 = plot_prctiles(output, 'GMST [°C]', ax=ax1, color='red')
plt.tight_layout()
ax2 = plot_prctiles(output, '$\mathregular{CO_{2}}$ Ind. Emissions [Gt$\mathregular{CO_{2}}$]', ax=ax2, color='red')
plt.tight_layout()  

custom_dict_ad = {'High': 0, 'Nominal': 1, 'Low': 2, 'Zero': 3} 
custom_dict_tcr = {'Higher than likely': 0, 'Likely': 1, 'Lower than likely': 2} 
custom_dict_dm = {'Level': 0, 'Growth': 1} 
custom_dict_ab = {'Standard': 0, 'Pliable': 1}
custom_dict = {'Adapt. eff.': custom_dict_ad,'Damages type': custom_dict_dm,'TCR': custom_dict_tcr, 'Abatement type': custom_dict_ab}
output['Ratio mitigation / adaptation costs'] = output['ABATECOST'] / output['ADAPTCOST']

filters = ['Adapt. eff.','Damages type','TCR']
for filt in filters:
       colors = ['#c59824','#397ea1','#c94933','#ab1818']
       if filt=='Damages type':
              colors = ['#397ea1','#ab1818']
       fig, ax = plt.subplots(figsize=figsize)
       count = 0
       cats = output[filt].unique().tolist()
       cats.sort( key=lambda x: custom_dict[filt][x])
       for cat in cats:
              print(cat)
              sel = output.loc[output[filt]==cat]
              ax = plot_prctiles(sel, '$\mathregular{CO_{2}}$ Ind. Emissions [Gt$\mathregular{CO_{2}}$]', ax=ax, color=colors[count], label=cat)
              count += 1       
       ax.legend()
       ax.set_xlim((2020,2150))
       ax.plot([x for x in range(2020,2150)],[0.0 for x in range(2020,2150)],'k--', alpha=0.5, zorder=-1)
       plt.tight_layout()

output['Adaptation costs [%GDP]'] = output['Flow adapt. [%GDP]'] + output['Adapt. invest. [%GDP]']
filters = ['Adapt. eff.','Damages type','TCR']
for filt in filters:
       colors = ['#c59824','#397ea1','#c94933','#ab1818']
       if filt=='Damages type':
              colors = ['#397ea1','#ab1818']
       fig, ax = plt.subplots(figsize=figsize)
       count = 0
       cats = output[filt].unique().tolist()
       cats.sort( key=lambda x: custom_dict[filt][x])
       for cat in cats:
              print(cat)
              sel = output.loc[output[filt]==cat]
              ax = plot_prctiles(sel, 'Adaptation costs [%GDP]', ax=ax, color=colors[count], label=cat)
              count += 1
       ax.legend()
       ax.set_xlim((2020,2150))
       ax.set_ylim((0.0,2.5))
       plt.tight_layout()

plt.figure()
legend_elements = [Line2D([0], [0], color='red', lw=1, label='Static intertemporal'),
                     Line2D([0], [0], color='forestgreen', lw=1, label='Self-adaptive'),]

plt.legend(handles = legend_elements)
plt.show()


