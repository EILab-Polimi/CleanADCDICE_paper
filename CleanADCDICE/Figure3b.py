import pandas as pd
import pareto, matplotlib, subprocess
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns
import matplotlib.font_manager as font_manager

sns.set_style('ticks')
sns.set_context('paper')
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


epss = [25,0.05,5,1,1,1]
# epss = [25,0.05,5,25,25,25]

allsols = pd.read_csv('./SimulationValFull.csv')
allsols.columns = ['Welfare_cal', 'DegY1.5°C_cal', 'DegY2°C_cal', 'NPV Damages_cal', 'NPV Abatement_cal',
       'NPV Adaptation_cal', 'Type', 'Welfare', 'P(GMST > 2°C)', 'Warming above 2°C [°C]',
       'NPV Damages [10^12 USD]', 'NPV Abat. costs [10^12 USD]', 'NPV Adapt. costs [10^12 USD]']
allsols = allsols.loc[allsols['Welfare'] < 0]
dps = allsols.loc[allsols['Type']=='DPS']
nondom = pareto.eps_sort([list(dps.itertuples(False))], 
       objectives=[7,8,9,10,11,12], epsilons=epss, kwargs={'maximize':[0]})
dps = pd.DataFrame.from_records(nondom, columns=list(dps.columns.values))
so = allsols.loc[allsols['Type']=='SO']
nondom = pareto.eps_sort([list(so.itertuples(False))], 
       objectives=[7,8,9,10,11,12], epsilons=epss, kwargs={'maximize':[0]})
so = pd.DataFrame.from_records(nondom, columns=list(so.columns.values))
allsols = pd.concat([so,dps], ignore_index=True)
columns_to_use = ['Welfare', 'P(GMST > 2°C)', 'Warming above 2°C [°C]',
       'NPV Damages [10^12 USD]', 'NPV Abat. costs [10^12 USD]', 'NPV Adapt. costs [10^12 USD]']
nondom = pareto.eps_sort([list(allsols.itertuples(False))], 
       objectives=[7,8,9,10,11,12], epsilons=epss, kwargs={'maximize':[0]})
allsols = pd.DataFrame.from_records(nondom, columns=list(allsols.columns.values))
fracs = [['SO', allsols.loc[allsols['Type']=='SO'].count()[0] / allsols.count()[0] * 100],
       ['DPS',allsols.loc[allsols['Type']=='DPS'].count()[0]/allsols.count()[0] * 100]]
fracs = pd.DataFrame(fracs, columns=['Method','Percentage'])
allsols = allsols[columns_to_use]
dps = dps[columns_to_use]
so = so[columns_to_use]

for el in columns_to_use:
       minv = allsols[el].min()
       maxv = allsols[el].max()
       allsols[el] = 1 - (maxv - allsols[el]) / (maxv-minv)
       dps[el] = 1 - (maxv - dps[el]) / (maxv-minv)
       so[el] = 1 - (maxv - so[el]) / (maxv-minv)

allsols = allsols.values.tolist()
dps = dps.values.tolist()
so = so.values.tolist()
with open('FRONTFILE.txt','w') as f:
       f.write("#\n")
       for front in [so,dps,allsols]:
              for line in front:
                     [f.write(str(x)+" ") for x in line]
                     f.write("\n")
              f.write("#\n")

refv = 1.0
ref = str(refv)
for el in range(5):
       ref+=" "+str(refv)
print('Computing Hypervolume ...')
output = subprocess.check_output('./WFG-hypervolume/wfg3 FRONTFILE.txt '+ref, shell=True).decode("utf-8")
output = [[x for x in line.split()] for line in output.split('\n')]
hvso, hvdps, hvref = output[0][2], output[2][2], output[4][2]
hvso = float(hvso)/float(hvref)
hvdps = float(hvdps)/float(hvref)

hvs = [['SO', hvso],['DPS',hvdps]]
hvs = pd.DataFrame(hvs, columns=['Method','Hypervolume'])

hvs['Method'] = ['Static' if x=='SO' else 'Self-adaptive' for x in hvs['Method']]
fracs['Method'] = ['Static' if x=='SO' else 'Self-adaptive' for x in fracs['Method']]

palette = ['darkorchid','forestgreen']

fig, ax = plt.subplots(1,2, figsize=(7/1.5,6/1.5))
sns.barplot(data=fracs, y='Percentage', x='Method', palette=palette, ax = ax[0])
ax[0].set_ylabel('Percentage of solutions in the final reference set')
sns.barplot(data=hvs, y='Hypervolume', x='Method',palette=palette, ax=ax[1])
plt.subplots_adjust(wspace=0.4, top=0.97, left=0.125, hspace=0.4, right=0.975)

plt.show()
