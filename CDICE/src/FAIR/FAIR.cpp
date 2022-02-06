#include "./FAIR.h"
#include <fstream>
#include <math.h>
#include <iostream>

FAIR::FAIR(){

}

FAIR::~FAIR(){
	for (int subt = 0; subt < horizon+1; subt++){
		delete[] carbon_boxes[subt];
	}
	delete[] carbon_boxes;
	delete[] gmst;
	delete[] ts;
	delete[] tf;
	delete[] alpha;
	delete[] c_acc;
	delete[] forc;
	delete[] c;
	delete[] noise;
}

void FAIR::allocate(Config *configPtr){
	config = configPtr;
	horizon = config->horizon;
	r0       = 35.0;
	rc       = 0.019;
	rt       = 4.165;
	iirf_h   = 100.0;
	iirf_max = 97.0;
	iirf = 0.0;
	alpha = new double[horizon+1];
	ppm_to_gtc = 2.124;
	a[0] = 0.2173;
	a[1] = 0.2240;
	a[2] = 0.2824;
	a[3] = 0.2763;
	tau[0] = 1000000;
	tau[1] = 394.4;
	tau[2] = 36.54;
	tau[3] = 4.304;
	carbon_boxes = new double*[horizon+1];
	for (int subt = 0; subt < horizon+1; subt++){
		carbon_boxes[subt] = new double[4];
	}
	carbon_boxes[0][0] = 58.71337292; // * ppm_to_gtc; //127.159
	carbon_boxes[0][1] = 43.28685286; // * ppm_to_gtc; //93.313;
	carbon_boxes[0][2] = 18.44893718; // * ppm_to_gtc; //37.840;
	carbon_boxes[0][3] = 3.81581747; // * ppm_to_gtc; //7.721
	taunew = 0.0;
	c_pi = 278.0; // * 2.124;
	c = new double[horizon+1];
	c[0] = c_pi;
	for (int idx = 0; idx < 4; idx++){
		c[0] += carbon_boxes[0][idx];
	}
	c_acc = new double[horizon+1];
	c_acc[0] = 305.1673751542558; // * ppm_to_gtc; //597.0; //(400+197)
	forc = new double[horizon+1];
	forc[0] = 0.0;

	alpha_tcrecs[0] = 0.5543371849882144;
	alpha_tcrecs[1] =  1.1512945858642294;
	root_delta[0][0] = 0.19695275;
	root_delta[0][1] = 0.11014884;
	root_delta[1][0] = 0.11014884;
	root_delta[1][1] = 0.22981541;
	stdev = 0.10729947684829523;
	ranges[0] = 0.0;
	ranges[1] =2000.0;
	ranges[2] = -1.0;
	ranges[3] = 10.0;
	nnprms[0] = -6.66006035e+02;
	nnprms[1] = 2.09443154e+02;
	nnprms[2] = -4.83968920e+00;
	nnprms[3] = 2.31243377e+00;
	nnprms[4] = 2.75031497e+00;
	nnprms[5] = 8.89902682e+02;
	nnprms[6] = 2.40146799e+00;
	nnprms[7] = 6.83316702e-02;
	nnprms[8] = 2.89753011e-02;
	rfothidx[0] = 26;;
	rfothidx[1] = 45;
	rfothidx[2] = 60;
	rfothidx[3] = 85;


	TCR = 1.6;
	ECS = 2.75;
	TCR = 1.8; // central value in AR6
	ECS = 3.0; // central value in AR6
	f2x = 3.71;
	ds = 239.0;
	df = 4.1;
	tcr_dbl = log(2) / log(1.01);
	ks = 0.0;
	kf = 0.0;
	qs = 0.0;
	qf = 0.0;
	ts = new double[horizon+1];
	tf = new double[horizon+1];
	gmst = new double[horizon+1];
	noise = new double[horizon+1];
	ts[0] = 0.11759814;
	tf[0] = 1.02697844;
	eff = 1.0;
	gmst[0] = ts[0]+tf[0];

	calculate_q();

	std::ifstream forc_file;
	for (int rfidx = 0; rfidx < 4; rfidx++){
		forc_file.open("./src/RFoth_"+std::to_string((unsigned long long int) rfothidx[rfidx])+".txt", std::ios_base::in);
		if(!forc_file){
			std::cout << "The other forcing file specified could not be found!" << std::endl;
			exit(1);
		}
		for (int idx=0; idx < 486 ; idx++){
			forc_file >> rfoth[rfidx][idx];
		}
		forc_file.close();		
	}
	t = 0;
	return;
}


void FAIR::nextStep(double e){
	//carbon
	e = e*12/44;
	iirf = std::min(r0 + rc*c_acc[t] + rt*gmst[t], iirf_max);
	computeAlpha();
	for (int box = 0; box < 4; box++){
		taunew = tau[box] * alpha[t];
		carbon_boxes[t+1][box] = 
			carbon_boxes[t][box] * exp(-1.0/taunew) + 
			a[box] * e / ppm_to_gtc;
	}
	c[t+1] = c_pi;
	for (int box = 0; box < 4; box++){
		c[t+1] += carbon_boxes[t+1][box];
	}
	c_acc[t+1] = c_acc[t] + e - (c[t+1]-c[t]) * ppm_to_gtc;
	if (t < 486){
		forc[t+1] = f2x/log(2.0) * log(c[t+1]/c_pi) + rfoth[config->rfoth][t];
	}
	else{
		forc[t+1] = f2x/log(2.0) * log(c[t+1]/c_pi) + rfoth[config->rfoth][485];
	}
	//temperature
	ts[t+1] = ts[t] * exp(-1.0/ds) + qs * (1.0 - exp(-1.0/ds)) * forc[t+1]*eff;
	tf[t+1] = tf[t] * exp(-1.0/df) + qf * (1.0 - exp(-1.0/df)) * forc[t+1]*eff;
	if (config->tatm_stoch == 1){
		// sample from standard normal distribution using box-cox method 
		if (config->scc_ == 0){
			noise[t+1] = std::max(-3.0, std::min(3.0, 
				sqrt(-2 * log(rand() * (1.0 / RAND_MAX)))
					 * sin (2 * M_PI * rand() * (1.0 /RAND_MAX)))) ; 
		}
		tf[t+1] = tf[t+1] + noise[t+1] * stdev;
	}
	gmst[t+1] = ts[t+1] + tf[t+1];
	if (gmst[t+1] > 1.5){
		if (gmst[t+1] > 2.0){
			twoDegYrs += gmst[t+1] - 2.0;
			aboveTwo = 1.0;
		}
		oneFiveDegYrs += gmst[t+1] - 1.5;
		aboveOneFive = 1.0;
	}
	t++;
	return;
}


void FAIR::calculate_q(){
	ks = 1.0 - (ds/tcr_dbl)*(1.0 - exp(-tcr_dbl/ds));
	kf = 1.0 - (df/tcr_dbl)*(1.0 - exp(-tcr_dbl/df));
	qs = (1.0/f2x) * (1.0/(ks - kf)) * (TCR - ECS * kf);
	qf = (1.0/f2x) * (1.0/(ks - kf)) * (ECS * ks - TCR);
	return;
}

void FAIR::sampleTCRECS(){
	// sample normal;
    double norm0 = std::max( -3.0,std::min( 3.0, 
		sqrt(-2 * log(rand() * (1.0 / RAND_MAX))) * 
		sin (2 * M_PI * rand() * (1.0 /RAND_MAX)))) ; // sample from standard normal distribution using box-cox method 
    double norm1 = std::max( -3.0, std::min( 3.0, 
		sqrt(-2 * log(rand() * (1.0 / RAND_MAX))) * 
		sin (2 * M_PI * rand() * (1.0 /RAND_MAX)))) ; // sample from standard normal distribution using box-cox method 

	// obtain tcr as :
	TCR = exp(alpha_tcrecs[0] + root_delta[0][0] * norm0 + root_delta[0][1] * norm1);
	// obtain ecs as :
	ECS = exp(alpha_tcrecs[1] + root_delta[1][0] * norm0 + root_delta[1][1] * norm1);

	calculate_q();
	return;
}

// to correct as FAIR has been corrected
void FAIR::computeAlpha(){
	double input[2];
	//normalize inputs
	input[0] = (c_acc[t] - ranges[0]) /	(ranges[1] - ranges[0]);
	input[1] = (gmst[t] - ranges[2]) / (ranges[3] - ranges[2]);

	//compute alpha via ANN 
	alpha[t] = nnprms[0] + \
	    (nnprms[1]) * \
	    	(-1.0 + 2.0 / \
	        	( 1.0 + exp( -2.0 * \
	          	(nnprms[2] + \
	            	(nnprms[3]) * input[0] + \
	            	(nnprms[4]) * input[1] )))) +\
	    (nnprms[5]) * \
	    	(-1.0 + 2.0 / \
	        	( 1.0 + exp( -2.0 * \
	        	(nnprms[6] + \
	            	(nnprms[7]) * input[0] + \
	            	(nnprms[8]) * input[1] ))));
	if (alpha[t] < 1e-3){
		alpha[t] = alpha[t-1];
	}
	return;
}

void FAIR::sampleUnc(){
	if (config->cl_sen_unc==1){
		sampleTCRECS();
	}
	if (config->rfoth_unc==1){
		config->rfoth = floor( rand() * (1.0/RAND_MAX) * 4 );
	}
	return;
}

void FAIR::reset(){
	t = 0;
	oneFiveDegYrs = 0.0;
	twoDegYrs = 0.0;
	aboveTwo = 0.0;
	aboveOneFive = 0.0;
	// c[0] = c_pi;
	// for (int idx = 0; idx < 4; idx++){
	// 	c[0] += carbon_boxes[idx];
	// }
	// c_acc[0] = 305.1673751542558; // * ppm_to_gtc; //597.0; //(400+197)
	// forc[0] = 0.0;
	// ts[0] = 0.11759814;
	// tf[0] = 1.02697844;
	// gmst[0] = ts[0]+tf[0];
	return;
}
