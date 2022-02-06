#ifndef __econ_h
#define __econ_h
#include "../config/config.h"
#include "../dvars/dvars.h"

class Econ{
public:
	Econ();
	~Econ();

	Config *config;
	Dvars *dvars;

	int horizon;
	int t;
	int tstep;
	// Preferences ---------------------------------
	double elasmu;	// Elasticity of marginal utility of consumption
	double prstp;		// Initial rate of social time preference (per year)


	// Population and Technology -------------------
	double gama;		// Capital elasticity in production function
	double popadj;	// Growth rate to calibrate to 2050 population projection
	double popasym;	// Asymptotic world population [Millions]
	double dk;		// Depreciation rate on capital (per year)
	double q0;		// Initial world gross output [Trillions 2005 US $]
	double dela;		// Decline rate of TFP (per 5 years)
	double optlrsav;

	// Emissions control and decarbonization
	double dsig;		// Decline rate of decarbonization (per period)
	double miu0; // Inital emissions control rate

	// Climate damage parameters ---------------------
	double a2_hs;
	double a0_agr;
	double a1_agr;
	double a2_agr;

	// NEEDED FOR AD-DICE
	double b1;		// coefficient CES adaptation function
	double b2;		// coefficient CES adaptation function
	double b3;		// coefficient CES adaptation function
	double rho;		// coefficient CES adaptation function
	double *ia;		// Investment in Stock adaptation 
	double *fad;		// Flow adaptation 
	double *sad;		// Stock adaptation
	double *adaptcost;		// Adaptation costs
	double *adapt;	// Adaptation (sum of flow adaptation and stock adaptation)
	double *rd;		// Residual Damages

	// Abatement cost --------------------------------
	double expcost2;	// Exponent of control cost function
	double pback;		// Cost of backstop [2005$ per tCO2 2010]
	double gback;		// Initial cost decline backstop [cost per period]
	double pli;   // to implement abatement costs from Grubb et al. (2021)
	double plitime;   // to implement abatement costs from Grubb et al. (2021)

	// Emissions parameters ------------------------
	double deland;	// Decline rate of land emissions (per period)

	// Scaling and inessential parameters ------------
	// "Note that these are unnecessary for the calculations but are for convenience"
	// Quoted directly from comments in original GAMS code
	double scale1;	// Multiplicitive scaling coefficient
	double scale2;        // Additive scaling coefficient

	// Exogenous timeseries variables
	double *pop;		// Level of population and labor (millions)
	double *pop5;		// Level of population and labor (millions) every 5 years
	double *tfp;		// Level of total factor productivity
	double *noise_tfp;		// Level of total factor productivity
	double stdev_tfp;
	double *cost1;	// Adjusted cost for backstop
	double *pbacktime;	// Backstop price
	double *ga;		// Growth rate of productivity from (sic)
	double *gsig;		// Change in sigma (cumulative improvement of energy efficiency)
	double *sigma;	// CO2-equivalent-emissions output ratio 
	double *noise_sigma;	// CO2-equivalent-emissions output ratio 
	double ar_model_sigma;
	double stdev_sigma;
	double *eland;	// Emissions from deforestation

	// Endogenous variables
	double *c;		// Consumption [Trillions 2005 US$ per year]
	double *k;		// Capital stock [Trillions 2005 US$]
	double *cpc;		// Per capita consumption [Thousands 2005 US$ per year]
	double *i;		// Investment [trillions 2005 US$ per year]
	double *ri;		// Real interest rate (per annum)
	double *y;		// Gross world product net of abatement and damages [Trillions 2005 US$ per year]
	double *ygross;	// Gross world product GROSS of abatement and damages [Trillions 2005 US$ per year]
	double *ynet;		// Output net of damages equation [Trillions of 2005 US$ per year]
	double *damages;	// Damages [Trillions 2005 US$ per year]
	double *damfrac;	// Damages as fraction of gross output
	double *abatecost;	// Cost of emissions reductions [Trillions 2005 US$ per year]
	double *mabatecost;	// Cost of emissions reductions [Trillions 2005 US$ per year]
	double *periodu;	// One period utility function
	double *cemutotper;	// Period utility
	double utility;	// Welfare function (Sum of discounted utility of per capita consumption)
	double *scc;		// Social cost of carbon
	double *rr;		// Average utility social discount rate
	double *ri_disc;	// Real interest rate (discounted) for present value calculations
	double *pv_damages;	// Present value of damages
	double *pv_abatecost;	// Present value of abatement costs
	double *totalcost;	// Present value of total costs (abatement + damages)
	double *pv_totalcost;	// Present value of total costs (abatement + damages + adaptation)
	double *pv_consumption;	// Present value consumption 
	double *pv_adaptcost;	// Present value of adaptation costs 
	double *e;		// Total CO2 emissions [GtCO2 per year]
	double *eind;		// Industrial emissions [GtCO2 per year]
	double *miueff; 	// Effective emissions control rate (used to constrain negative emissions on SSP)

	double owaactc;
	double owarada;
	double owaprada;
	double owagcap;
	double owascap;
	double owaact;
	double owacap;
	double rhoact;
	double rhocap;
	double rhotfp;
	double rhoada;
	double k0rd;
	double k0edu;
	double deg1;
	double deg2;
	double deg3;
	double deg4;

	double *iprada;
	double *irada;
	double *iscap;
	double *qact;
	double *kprada;
	double *qgcap;
	double *qcap;
	double *kscap;
	double dkscap;
	double dkprada;
	// double *k_static;   // Level of total factor productivity
	// double *sad_static;   // Level of total factor productivity
	// double *al_static;   // Level of total factor productivity
	// double *l_static;   // Level of total factor productivity
	// double *sigma_static;   // Level of total factor productivity
	// double *pbacktime_static;   // Level of total factor productivity

	void allocate(Config *configPtr, Dvars *dvarsPtr);
	void nextStep(double temp);
	void reset();
	void sampleUnc();
};

#endif