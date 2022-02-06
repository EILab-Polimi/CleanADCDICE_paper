#ifndef __dvars_h
#define __dvars_h
#include "../config/config.h"
#include "../emodps/param_function.h"
#include "../emodps/rbf.h"
#include "../emodps/ann.h"
#include "../emodps/ann_mo.h"
#include "../emodps/pwLinear.h"
#include "../emodps/ncRBF.h"
#include <fstream>
#include <vector>

class Dvars{
public:
	Dvars();
	~Dvars();

	Config *config;

	int horizon;
	int tstep;

	double limmiu;	// Upper limit on control rate after 2150
	double miu0;		// Initial emissions control rate for base case 2010

	double *miu;	// Emission control rate GHGs
	double *s;	// Gross savings rate as function of gross world production

	// NEEDED FOR AD-DICE
	double *p_ia;		// percentage of GDP used for Investment in Stock Adaptation
	double *p_fad;	// percentage of GDP used for Flow Adaptation
	double *p_iprada;
	double *p_irada;
	double *p_scap;
	
	std::pFunction_param p_param;
	std::param_function* policy;
	std::vector<double> controls;
	void allocate(Config *configPtr);
	void allocatePolicy();
	void nextAction(std::vector<double> states, int t);
	void setVariables(double *vars);
	void readPolicySettings();
};

#endif