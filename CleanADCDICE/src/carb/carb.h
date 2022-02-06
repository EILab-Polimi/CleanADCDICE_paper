#ifndef __carb_h
#define __carb_h
#include "../config/config.h"

class Carb{
public:
	Carb();
	~Carb();
	// Carbon cycle ---------------------------------
	// Initial conditions	
	double mat0;		// Initial concentration in atmosphere 2010 [GtC]
	double mu0;		// Initial concentration in upper strata [GtC]
	double ml0;		// Initial concentration in lower strata [GtC]
	double mateq;		// Equilibrium concentration in atmosphere [GtC]
	double mueq;		// Equilibrium concentration in upper strata [GtC]
	double mleq;		// Equilibrium concentration in lower strata [GtC]

	// Flow parameters (Carbon cycle transition matricies)
	double b12;
	double b23;
	double b11;
	double b21;
	double b22;
	double b32;
	double b33;

	// Forcing parameters
	double fco22x;	// Forcings of equilibrium CO2 doubling [Wm-2]
	double fex0;		// 2010 forcings of non-CO2 greenhouse gases (GHG) [Wm-2]
	double fex1;		// 2100 forcings of non-CO2 GHG [Wm-2]

	// Availability of fossil fuels ------------------
	double fosslim;	// Maximum cummulative extraction fossil fuels [GtC]

	// Exogenous timeseries variables
	double *forcoth;	// Exogenous forcing for other GHG

	// Endogenous variables
	double *forc;		// Increase in radiative forcing [Wm-2 from 1900]
	double *mat;		// Carbon concentration increase in atmosphere [GtC from 1750]
	double *mu;		// Carbon concentration increase in shallow oceans [GtC from 1750]
	double *ml;		// Carbon concentration increase in lower oceans [GtC from 1750]

	double *atfrac;		// Atmospheric fraction since 1850
	double *atfrac2010;	// Atmospheric fraction since 2010
	double *ppm;		// Atmospheric concentration (actually, mixing ratio)

	void allocate();
	void nextStep();
	void free();
	Config *config;
};

#endif