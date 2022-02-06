#ifndef __cdice_h
#define __cdice_h
#include <fstream>

// #include "./carb/carb.h"
// #include "./clim/clim.h"
#include "./config/config.h"
#include "./FAIR/FAIR.h"
#include "./econ/econ.h"
#include "./dvars/dvars.h"
#include <vector>

class CDICE{
public:
	CDICE();
	~CDICE();

	int t;

	// Carb carb;
	// Clim clim;
	Config config;
	FAIR fair;
	Econ econ;
	Dvars dvars;
	std::vector<double> states;

	void simulate();
	void nextStep();
	void postProcess(int tidxs = 0);
	int getNVars();
	int getNObjs();
	void setVariables(double *vars);
	void writeOutput(std::ofstream &output, int niter = 0);
	void writeOutputHeader(std::ofstream &output);
	void computeSCC(std::ofstream &output, int niter = 0);
	void computeSCCDamages(std::ofstream &output, int niter = 0);
};

#endif