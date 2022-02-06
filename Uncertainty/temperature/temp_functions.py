import numpy as np
import matplotlib.pyplot as plt

# tocean0  Initial lower stratum temp change (C from 1900)  /.0068 /
# tatm0    Initial atmospheric temp change (C from 1900)    /0.85  /
c1 = 0.1005
c3 = 0.088
c4 = 0.025
## Geoffroy parametrization
c1 = 1 / 7.3
c3 = 0.73
c4 = 0.73/106
fco22x = 3.6813
t2xco2 = 3.1
lam = fco22x / t2xco2

def unpack_data(timef, forc, year, temp):
	#start datasets from same data
	if year[0] > timef[0]:
		idx = year[0]-timef[0]
		timef = timef[int(idx)-1:]
		forc = forc[int(idx)-1:]
	else:
		idx = timef[0]-year[0]
		year = year[int(idx):]
		temp = temp[int(idx):]
	return timef, forc, year, temp

def temp_model(data, x=0, tstep=1, horizon=None):
	# initialization
	_, forc, year, temp = unpack_data(**data)
	t = [year[0] + x]
	t_a = [float(np.sum([temp[x+el] for el in range(-3,4) if x+el>=0 ])/7)]
	t_a = [temp[x]]
	t_o = [0.0]
	if horizon is None:
		horizon = len(forc[x:-1])
	# iterate over tsteps
	for el in range(1, round(horizon/tstep)):
		t.append(t[-1] + tstep)
		t_a.append(t_a[-1] + c1/5*tstep * (forc[x + el * tstep] -\
		 lam * t_a[-1]  - c3 * (t_a[-1] - t_o[-1])))
		t_o.append(t_o[-1] + c4/5*tstep * (t_a[-2] - t_o[-1]))
	return t_a, t_o, t

def temp_model_s(data, x=0, tstep=1, mu=0, sigma=0.1, 
	mul=False, horizon=None):
	# initialization
	_, forc, year, temp = unpack_data(**data)
	t = [year[0] + x]
	t_a = [float(np.sum([temp[x+el] for el in range(-3,4) if x+el>=0 ])/7)]
	t_a = [temp[x]]
	t_o = [0.0]
	if horizon is None:
		horizon = len(forc[x:-1])
	# iterate over tsteps
	for el in range(1,round(horizon/tstep)):
		t.append(t[-1]+tstep)
		if mul==True:
			t_a.append( (t_a[-1] + c1 / 5 * tstep * ((forc[x + el * tstep] -\
			 	lam * t_a[-1] ) - c3 * (t_a[-1] - t_o[-1])) ) + \
				(np.random.normal(0.0, sigma) ))
		else:
			t_a.append(t_a[-1] + c1 / 5 * tstep * ((forc[x + el * tstep+1] -\
				lam * t_a[-1] ) - c3 * (t_a[-1] - t_o[-1])) + \
				max(-4, min(4, np.random.normal(0,1)))*sigma)
				# max(-4*sigma, min(4*sigma, np.random.normal(0,1)))*sigma)
		t_o.append(t_o[-1] + c4 / 5 * tstep * (t_a[-2] - t_o[-1]))
	return t_a, t_o, t
