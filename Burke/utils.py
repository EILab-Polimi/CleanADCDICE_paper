import numpy as np
import math

## 			BURKE_model 
# computes gdp per capita projections for a single region based on:
# - gdp per capita in 2010 (used for initialization)
# - baseline gdp per capita growth in percentage (here supposed in 5-y timesteps)
# - a list with base temperature and its increase at the end of horizon
# 	(- or a temperature trajectory) 
# endhor is the final year on the horizon starting in 2010
# cap enforces a limit on gdp per capita maximum growth (2 times the baseline)

def BURKE_model(gdppc2010, gdppcgwth, tempvec, endhor=2100, cap=True):
	## parameters from Burke et al. (2015) benchmark model
	beta1 = 0.0127 
	beta2 = -0.0005 
	## prepare lists
	gdppc = [gdppc2010]
	gdppcnocc = [gdppc2010]
	## interpolate gdp growth data from 5-y to annual time step
	temp = gdppcgwth
	gdppcgwth = []
	for idx in range(len(temp) - 1):
		for el in range(5):
			gdppcgwth.append(temp[idx]  + el/5 * ( temp[idx+1] - temp[idx]))
	
	if endhor > 2100:
		[gdppcgwth.append(gdppcgwth[-1]) for x in range(endhor - 2100)]
	## extract base temperature
	## case where base temperature and temperature increase are given
	if len(tempvec) == 2:
		tempbase = tempvec[0]
	else:
		tempbase = np.mean(tempvec[:30])
		tempvec = tempvec[30:]

	for t in range(2010,endhor):
		
		# get temperature at time step
		if len(tempvec)==2:
			# interpolate temperature
			temp = tempbase + (tempvec[1]) * (t - 2010)/ (endhor - 2010)
		else:
			temp = tempvec[t - 2010]

		# compute gdp growth rate correction 
		delta = beta1 * min(temp, 30.0) + \
		 + beta2 * min(temp, 30.0)**2 - \
		 ( beta1 * tempbase + beta2 * tempbase**2)
		
		# compute gdp per capita at next step
		# the cap bounds gdp per capita growth rate to be below doubling of base gdppc growth 
		if cap==True:
			gdppc.append( gdppc[-1] *( 1 + 
				min(gdppcgwth[t-2010]* 0.01 + delta, 
				gdppcgwth[t - 2010]*0.01*2)))
		else:
			gdppc.append( gdppc[-1] *( 1 
				+ gdppcgwth[t-2010]* 0.01 + delta))

		# compute gdp per capita without climate change effects
		gdppcnocc.append( gdppcnocc[-1] * ( 1 + gdppcgwth[t-2010]*0.01 ))

	return gdppc, gdppcnocc


##			run_model 
# runs BURKE_model for each country 
# and computes global GWP with and without CC
# this function is made to reproduce Burke et al. (2015)
# methodology and results.
def run_model(countries, data8010, 
	lgdppcgwth, popdata, tempdata2, 
	tempfactor=1, cap=True, endhor=2100):

	GWP = [0.0 for el in range(2010,endhor+1)]
	GWPnoCC = [0.0 for el in range(2010,endhor+1)]
	gdptraj = []
	count = 0
	
	## iterate over countries list
	for el in countries:

		# get gdppc in 2010
		if el =='COD':
			el=	'ZAR'
		if el=='ROU':
			el='ROM'
		gdppc2010 = np.nanmean(
			np.asarray(data8010.loc[(data8010['iso']==el) & 
			(data8010['year'].between(1980,2010, inclusive=True))]['TotGDP'].values)
			/ 
			np.asarray(data8010.loc[(data8010['iso']==el) & 
			(data8010['year'].between(1980,2010, inclusive=True))]['Pop'].values)
			)	
		
		# get gdppc growth (SSP data in 5 year time step!)
		if el=='ZAR':
			el='COD'
		if el=='ROM':
			el='ROU'
		gdppcgwth = lgdppcgwth.loc[ 
			(lgdppcgwth['Region']==el) & 
			(lgdppcgwth['Scenario'].str.contains('SSP5')) & 
			(lgdppcgwth['Model'].str.contains('OECD')) 
			][[x for x in range(2010,2100,5)]].values.tolist()[0]
		gdppcgwth.append(gdppcgwth[-1])
		
		# get base temperature (1980-2010 mean)
		if el=='COD':
			el='ZAR'
		if el=='ROU':
			el='ROM'
		tempbase = np.nanmean(
			data8010.loc[ 
			(data8010['iso']==el) & 
			(data8010['year'].between(1980,2010, inclusive=True))
			]['UDel_temp_popweight'].values)
		if math.isnan(tempbase):
			print(el + ' skipped as no temp available')
			continue
		
		# set base and modify target 2100 temperature
		if el=='ZAR':
			el='COD'
		if el=='ROM':
			el='ROU'
		tempvec = [tempbase, 
		tempdata2.loc[ 
		(tempdata2['layercountry2_ISOCODE']==el)
		]['Tchg'].values[0]*tempfactor]


		# run BURKE_model for the selected region
		gdptraj.append([el, 
			[BURKE_model(gdppc2010, gdppcgwth, tempvec, 
			cap=cap, endhor=endhor)]])
		
		# use population data to compute GWP and GWPnoCC
		# (SSP data in 5 year time step!)
		# compute population of region considered
		temp = popdata.loc[ 
			(popdata['Region'] == el) & 
			(popdata['Scenario'].str.contains('SSP5')) 
			][[x for x in range(2010,2101,5)]].values.tolist()[0]
		# compute ratio of region to world population
		# temp = np.asarray(temp) / popdata.loc[ 
		# 	(popdata['Scenario'].str.contains('SSP5')) 
		# 	][[x for x in range(2010,2101,5)]].sum().values
		#interpolate ratio from 5 year to annual time step
		pop = []
		for idx in range(len(temp) - 1):
			for elx in range(5):
				pop.append(temp[idx] + elx/5 * (temp[idx+1] - temp[idx]))
		pop.append(temp[-1])
		if endhor > 2100:
			[pop.append(pop[-1]) for x in range(endhor - 2100)]
			# [pop.append(pop[-1] + (pop[-2] + pop[-3])) for x in range(endhor - 2100)]
		# get GWP by weighting gdppc by popularion ratio
		for elb in enumerate(gdptraj[-1][1][0][0]):
			if math.isnan(elb[1]):
				print(el)
			GWP[elb[0]] += elb[1] * pop[elb[0]]
		for elb in enumerate(gdptraj[-1][1][0][1]):
			if math.isnan(elb[1]):
				print(el)
			GWPnoCC[elb[0]] += elb[1] * pop[elb[0]]
		count += 1

	print(str(count) + ' countries simulated')

	return GWP, GWPnoCC, gdptraj

