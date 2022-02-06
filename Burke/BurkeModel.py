import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils
import os

# import seaborn as sns
# sns.set_style('whitegrid')
# import matplotlib
# import matplotlib.font_manager as font_manager
# fontpath = '/Users/angelocarlino/Library/Fonts/OpenSans-Regular.ttf'
# prop = font_manager.FontProperties(fname=fontpath, size='large')
# prop.set_size(12)
# matplotlib.rcParams['font.family'] = prop.get_name()
# matplotlib.rcParams['font.size'] = prop.get_size()
# matplotlib.rcParams['mathtext.fontset'] = 'custom'
# matplotlib.rcParams['mathtext.rm'] = prop.get_name()
# matplotlib.rcParams['mathtext.sf'] = prop.get_name()
# matplotlib.rcParams['mathtext.it'] = prop.get_name()
# matplotlib.rcParams['mathtext.bf'] = prop.get_name()
# matplotlib.rcParams['mathtext.tt'] = prop.get_name()
# matplotlib.rcParams['mathtext.cal'] = prop.get_name()

### setting path folder
path = os.getcwd()+'/'
## read temperature, gdp growth from SSP,
## gdp data and SSP population projections
## from input dataset of Burke et al. 2015
burke_path='BurkeHsiangMiguel2015_Replication/data/input/'
tempdata2 = pd.read_csv(path + burke_path + 
	'CCprojections/CountryTempChange_RCP85_ISO.csv')
gdpgwthdata = pd.read_excel(path+ burke_path + 
	'SSP/SSP_GrowthProjections.xlsx', skipfooter=1)
data8010 = pd.read_csv(path + burke_path + 
	'GrowthClimateDataset.csv')
popdata = pd.read_excel(path+ burke_path + 
	'SSP/SSP_PopulationProjections.xlsx', skipfooter=1)
countries = sorted(gdpgwthdata['Region'].unique())

## initialize dataset for surrogate Burke model training
dataset = {}
dataset['gtemp'] = []
dataset['damages'] = []

## simulate Burke model under different temperature increase
## and collect data in the dataset dictionary
endhor=2100
fig, ax = plt.subplots(1,1)
tempfactorvec = [el for el in np.arange(0.05,2.05,0.02)]
# tempfactorvec = [el for el in np.arange(1,2.05,0.13)]
for tempfactor in tempfactorvec:
	[GWP, GWPnoCC, BURKERES] = utils.run_model(countries,
	 data8010, gdpgwthdata, popdata, tempdata2,
	  endhor = endhor, tempfactor=tempfactor, cap=True)

	## collect global temperature from 1980 to 2100
	## averaging across monthly measurements (in [K])
	rcp85 = []
	with open(path + 'global_tas_Amon_modmean_rcp85_000.txt') as f:
		for el in f:
			if el[0]!="#" and int(el.split()[0]) in range(1980,2100):
				rcp85.append(
					np.mean([float(x) - 273.15 for x in el.split()[1:]]))

	## from 2010 to 2100 compute global temp 
	## interpolating linearly as in Burke et al. 2015
	## using tempfactor to explore different
	## temperature increase scenarios
	gtemp = [(rcp85[-1] - rcp85[30]) * tempfactor * el/90 + 
		np.mean(rcp85[:30]) for el in range(0, endhor - 2010 + 1)]

	## collect damages from Burke simulation output 
	damages = []
	for idx in range(len(GWP) ):
		damages.append( 1 - (GWP[idx])/GWPnoCC[idx])
	## plot simulation output in GWP
	ax.plot(GWP)
	ax.plot(GWPnoCC, 'k--')

	## save global temp and damages
	## using a 5-year time step
	## to be consistent with DICE
	[dataset['gtemp'].append(x) for x in gtemp[::5]]
	[dataset['damages'].append(x) for x in damages[::5]]

data = pd.DataFrame.from_dict(dataset)
data.to_csv('./Burke_dataset.csv')

## scatter plot to visualize relationship among variables
gtemp = np.reshape(data['gtemp'].values, 
	(len(data['gtemp'].values)//((2100-2010)//5 + 1), -1))
damages = np.reshape(data['damages'].values, 
	(len(data['damages'].values)//((2100-2010)//5 + 1), -1))
fig, ax = plt.subplots()
obs = ax.scatter(np.transpose(gtemp)[:-1],np.transpose(damages)[:-1],
	 c=np.transpose(damages)[1:], vmin=0, vmax=0.4)
ax.set_xlabel('Temperature [°C]')
ax.set_ylabel('Damages at time t [%GWP]')
cbar = fig.colorbar(obs)
cbar.set_label('Damages at time t+1 [%GWP]')
plt.show()