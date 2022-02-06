#ifndef __clim_h
#define __clim_h

struct Clim{
	// Climate model parameters ---------------------
	double t2xco2;	// Equilibrium temperature impact [dC per doubling CO2]
	double tocean0;	// Initial lower stratum temperature change [dC from 1900]
	double tatm0;		// Initial atmospheric temperature change [dC from 1900]
	double c1;		// Climate equation coefficient for upper level
	double c3;		// Transfer coefficient upper to lower stratum
	double c4;		// Transfer coefficient for lower level
	double lam;		// Climate model parameter


	// Endogenous variables ==================================================
	double *tatm;		// Increase temperature of atmosphere [dC from 1900]
	double *tocean;	// Increase temperature of lower oceans [dC from 1900]
	double *maxtemp_vec;	// Vector that contains the maximum temperature increase from stochastic runs
};

#endif