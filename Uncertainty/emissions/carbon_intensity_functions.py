import numpy as np
from scipy.stats import levy_stable, cauchy

def sigma_model(sigma0, gsig0=-0.0152, 
	dsig=-0.001, tstep=1, horizon=54):
	sigma = [sigma0]
	gsig = [gsig0]
	for t in range(1, horizon):
		gsig.append( gsig[t-1]*((1+dsig)**tstep) )
		sigma.append( sigma[t-1]*np.exp(gsig[t-1]*tstep) )
	return sigma, gsig

def residual_sigma_model(params, sigma, tstep=1):
	gsig0 = params[0]
	dsig = params[1]
	residual = []
	sigma_dice, _ = sigma_model(sigma[0], gsig0=gsig0, 
		dsig=dsig, tstep=tstep)
	residual = [(el1 - el2)**2 for el1,el2 
		in zip(sigma[0::tstep], sigma_dice)]
	return sum(residual)

def sigma_model_s(sigma0, gsig0=-0.0152, 
	dsig=-0.001, tstep=1, horizon=54, mu=0, stdev=0):
	sigma = [sigma0]
	gsig = [gsig0]
	for t in range(1, horizon):
		gsig.append( gsig[t-1]*((1+dsig)**tstep) )
		value = sigma[t-1]*np.exp(gsig[t-1]*tstep) +\
		 np.random.normal(mu,stdev) 
		if value > 0:
			sigma.append( value )
		else:
			sigma.append(0)
	return sigma

def sigma_model_smul(sigma0, gsig0=-0.0152, 
	dsig=-0.001, tstep=1, horizon=54, mu=1, stdev=0):
	sigma = [sigma0]
	gsig = [gsig0]
	for t in range(1, horizon):
		gsig.append( gsig[t-1]*((1+dsig)**tstep) )
		value = sigma[t-1]*np.exp(gsig[t-1]*tstep) *\
		 np.random.normal(mu,stdev)
		if value > 0:
			sigma.append( value )
		else:
			sigma.append(0)
	return sigma

def sigma_model_smul_ls(sigma0, params, 
	gsig0=-0.0152, dsig=-0.001, tstep=1, horizon=54, mu=1):
	sigma = [sigma0]
	gsig = [gsig0]
	for t in range(1, horizon):
		gsig.append( gsig[t-1]*((1+dsig)**tstep) )
		value = sigma[t-1]*np.exp(gsig[t-1]*tstep) * \
		min(1.5,max(levy_stable.rvs(*params),0.5)) 
		if value > 0:
			sigma.append( value )
		else:
			sigma.append(0)
	return sigma

def sigma_model_smul_c(sigma0, params, 
	gsig0=-0.0152, dsig=-0.001, tstep=1, horizon=54, mu=1):
	sigma = [sigma0]
	gsig = [gsig0]
	for t in range(1, horizon):
		gsig.append( gsig[t-1]*((1+dsig)**tstep) )
		value = sigma[t-1]*np.exp(gsig[t-1]*tstep) *\
		 min(1.16,max(cauchy.rvs(*params),0.84)) 
		if value > 0:
			sigma.append( value )
		else:
			sigma.append(0)
	return sigma