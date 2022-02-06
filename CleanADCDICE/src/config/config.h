#ifndef __config_h
#define __config_h

class Config{
public:
	Config();
	~Config();
	int horizon;
	int econ_tstep;
	int decs_tstep;
	int *year;
	int nobjs;          // number of objectives to be considered (indicator of the quality of the policy)
	int adaptive;          // number of variables needed to implement the policy
	int adaptation;        // use model with adaptation based on de bruin (2014) and agrawala (2010)
	int damages;          // cost function : burke or dice
	double adapteff;        // adaptation costs multiplicative factor
	int niter;					// number of iterations: useful when stochastic disturbances are present
	int tatm_stoch;               // additive atmospheric temperature stochastic noise
	int cl_sen_unc;               // climate sensitivity as uncertain parameter.
	int tfp_stoch;               // additive TFP stochastic noise
	int pop_unc;               // population uncertin parameterization
	int sigma_stoch;               // additive sigma stochastic noise
	int tfp_unc;
	int sigma_unc;
	int abate_unc;
	int abatetype_unc;
	int adapt_unc;
	int damages_unc;
	int rfoth_unc;
	int limmiu_s;                   // first period miu>1 is uncertain (range: 2050-2150)
	int cost_effective;           // removes temperature damages on the GDP
	int time_pol;           // time is an input to the policy if 1
	int annmo;           // multi-output ANN if 1
	int exog;                 // number of exogeonus input in policy
	int writefile;           //writes output file
	int scc;             // computeSCC 
	int scc_;            // use fixed temperature stochastic disturbance for SCC computation	
	int grubb;
	int rfoth;
};

#endif