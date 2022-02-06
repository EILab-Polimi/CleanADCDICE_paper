#ifndef __fair_h
#define __fair_h

#include "../config/config.h"

class FAIR{
public:
	FAIR();
	~FAIR();

	Config *config;

	// compute irf and compute alpha
	int t;
	int horizon;
	double iirf_max;
	double iirf_h;
	double time_scale_sf;
	double r0;
	double rc;
	double rt;
	double iirf;
	double* alpha;
	double ppm_to_gtc;

	// carbon cycle
	double a[4];
	double tau[4];
	double** carbon_boxes;
	double taunew;
	double* c;
	double c_pi;
	double* c_acc;
	double* forc;

	// nedeed for temperature
	double TCR;
	double ECS;
	double f2x;
	double ds;
	double df;
	double tcr_dbl;
	double ks;
	double kf;
	double qs;
	double qf;
	double eff;
	double* ts;
	double* tf;
	double* gmst;
	double* noise;
	// double alpha_tcrecs[2] =  {0.57913745,1.14131177};
	// double root_delta[2][2] =  { {0.17632248,0.10108759} , {0.10108759,0.21679573} }; 
	double alpha_tcrecs[2]; // =  {0.5543371849882144, 1.1512945858642294};
	double root_delta[2][2]; // =  { {0.19695275, 0.11014884}, {0.11014884, 0.22981541} };
	double stdev; // = 0.10729947684829523;
	double ranges[4]; // = {0.0, 2000.0, -1.0, 10.0};
	double nnprms[9]; // = {-6.66006035e+02, 2.09443154e+02, 
		// -4.83968920e+00, 2.31243377e+00, 2.75031497e+00, 
		// 8.89902682e+02, 2.40146799e+00, 
		// 6.83316702e-02, 2.89753011e-02};

	double rfoth[4][486];
	double rfothidx[4]; // = {26,45,60,85};

	double twoDegYrs, oneFiveDegYrs, aboveTwo, aboveOneFive;

	void allocate(Config *config);
	void nextStep(double e);
	void computeAlpha();
	void calculate_q();
	void sampleTCRECS();
	void sampleUnc();
	void reset();

};

#endif