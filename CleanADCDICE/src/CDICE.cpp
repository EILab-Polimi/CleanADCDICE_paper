#include <iostream>
#include <fstream>
#include "CDICE.h"

CDICE::CDICE(){
	Config *configPtr = &config;
	fair.allocate(configPtr);
	dvars.allocate(configPtr);
	Dvars *dvarsPtr = &dvars;
	econ.allocate(configPtr, dvarsPtr);
	// std::cout << "building" << std::endl;
}

CDICE::~CDICE(){
	// std::cout << "destructing" << std::endl;
}

void CDICE::simulate(){
	econ.reset();
	fair.reset();
	t = 0;
	// std::cout << "simulating" << std::endl;
	while (t < config.horizon/config.econ_tstep){
		nextStep();
	}
	postProcess(0);
	return;
}

void CDICE::nextStep(){
	if ((t * config.econ_tstep) % config.decs_tstep == 0){
		states.clear();
		states.push_back(econ.k[t]/(econ.pop[t] * econ.tfp[t]));
		states.push_back(fair.gmst[t*config.econ_tstep]);
		states.push_back(fair.c[t*config.econ_tstep]);
		if (t > 0){
			// states.push_back(1000.0 * econ.damages[t-1]/(econ.pop[t-1] * econ.tfp[t-1]));
			states.push_back(100.0 * econ.damages[t-1]/econ.y[t-1]);
		}
		else{
			states.push_back(0.0);
		}
		states.push_back(t*config.econ_tstep/5.0);
		if (config.adaptation == 1){
			states.push_back(econ.sad[t]);
		}
		if (config.adaptation == 2){
			states.push_back(econ.kprada[t] / econ.k[t]);
			states.push_back(econ.kscap[t] / econ.k[t]);
		}
		if (config.scc_>=0){
			dvars.nextAction(states, (int) t*config.econ_tstep/config.decs_tstep);
		}
	}
	econ.nextStep(fair.gmst[fair.t]);
	for (int subt=0; subt < config.econ_tstep; subt++){
		fair.nextStep(econ.e[econ.t-1]);
	}
	t++;
	return;
}

void CDICE::postProcess(int tidxs){
	// Calculate additional model outputs ------------------------------------------
	for (int tidx=0; tidx < (config.horizon - 1) / config.econ_tstep; tidx++){
	    // One period utility function
	    econ.periodu[tidx] = ( 
			pow(econ.c[tidx] * 1000.0 / econ.pop[tidx] , 1.0 - econ.elasmu) - 1.0) /
			(1.0 - econ.elasmu) - 1.0;
	    // Period utility
	    econ.cemutotper[tidx] = econ.periodu[tidx] * econ.pop[tidx] * econ.rr[tidx];
	    // Total cost of abatement and damages
	    econ.totalcost[tidx] = econ.damages[tidx] + econ.abatecost[tidx] + econ.adaptcost[tidx];
	}
	// Discounted utility of consumption (objective function) -----------------------
	double tempsum;
	tempsum = 0.0;
	for (int tidx=0; tidx < (config.horizon-1) / config.econ_tstep; tidx++){
		tempsum += econ.cemutotper[tidx];
    } 
	// econ->utility = (config->tstep*econ->scale1*temfripsum)+econ->scale2;
	econ.utility = (config.econ_tstep * tempsum);
	// Present Value Calculations for damages and abatement costs --------------------
	econ.pv_damages[0] = econ.damages[0] * config.econ_tstep;
	econ.pv_abatecost[0] = econ.abatecost[0] * config.econ_tstep;
	econ.pv_adaptcost[0] = econ.adaptcost[0] * config.econ_tstep;
	econ.pv_consumption[0] = econ.c[0] * config.econ_tstep;
	econ.ri_disc[0] = 1.0;
	for (int tidx=1; tidx < (config.horizon) / config.econ_tstep; tidx++){
		if(econ.ri[tidx] <= 0.0) {
			econ.ri_disc[tidx] = 0.0;
		}
		else{
    		econ.ri_disc[tidx] = 1.0 / 
				pow(1.0 + econ.ri[tidx], config.econ_tstep * tidx );
		}
		econ.pv_damages[tidx] = econ.pv_damages[tidx-1] + 
			econ.ri_disc[tidx] * econ.damages[tidx] * econ.tstep;
		econ.pv_abatecost[tidx] = econ.pv_abatecost[tidx-1] + 
			econ.ri_disc[tidx] * econ.abatecost[tidx] * econ.tstep;
		econ.pv_adaptcost[tidx] = econ.pv_adaptcost[tidx-1] + 
			econ.ri_disc[tidx] * econ.adaptcost[tidx] * econ.tstep;
		econ.pv_totalcost[tidx] = econ.pv_totalcost[tidx-1] + 
			econ.ri_disc[tidx] * econ.totalcost[tidx] * econ.tstep;
		econ.pv_consumption[tidx] = econ.pv_consumption[tidx-1] + 
			econ.ri_disc[tidx] * econ.c[tidx] * econ.tstep;
    }  
 //   	econ.pv_damages[tidxs] = econ.damages[tidxs];
	// econ.pv_abatecost[tidxs] = econ.abatecost[tidxs];
	// econ.pv_adaptcost[tidxs] = econ.adaptcost[tidxs];
	// econ.ri_disc[tidxs] = 1.0;
	// for (int tidx=tidxs+1; tidx < (config.horizon+1) / config.econ_tstep; tidx++){
	// 	if(econ.ri[tidx] <= 0.0) {
	// 		econ.ri_disc[tidx] = 0.0;
	// 	}
	// 	else{
 //    		econ.ri_disc[tidx] = 1.0 / 
	// 			pow(1.0 + econ.ri[tidx], config.econ_tstep * (tidx - tidxs) );
	// 	}
	// 	econ.pv_damages[tidx] = econ.pv_damages[tidx-1] + 
	// 		econ.ri_disc[tidx] * econ.damages[tidx] * econ.tstep;
	// 	econ.pv_abatecost[tidx] = econ.pv_abatecost[tidx-1] + 
	// 		econ.ri_disc[tidx] * econ.abatecost[tidx] * econ.tstep;
	// 	econ.pv_adaptcost[tidx] = econ.pv_adaptcost[tidx-1] + 
	// 		econ.ri_disc[tidx] * econ.adaptcost[tidx] * econ.tstep;
	// 	econ.pv_totalcost[tidx] = econ.pv_totalcost[tidx-1] + 
	// 		econ.ri_disc[tidx] * econ.totalcost[tidx] * econ.tstep;
 //    }  
	return;
}

void CDICE::computeSCC(std::ofstream &output, int niter){
	simulate();
	double scc, dutil_e, dutil_c;
	double ref_w = econ.utility;
	for (int tidx = 1; tidx < econ.horizon; tidx++){
		// compute welfare delta from pulse in consumption
		econ.c[tidx] += 1.0;
		postProcess();
		dutil_c = econ.utility - ref_w;
		econ.c[tidx] -= 1.0;
		postProcess();
		// compute welfare delta from pulse in emissions
		econ.e[tidx] += 1.0;
		t = tidx;
		econ.t = t;
		fair.t = t*econ.tstep;
		config.scc_ = 1;
		for (int subt=0; subt < config.econ_tstep; subt++){
			fair.nextStep(econ.e[t]);
		}
		t++;
		econ.t++;
		while (t < config.horizon/config.econ_tstep){
			nextStep();
		}
		postProcess();
		dutil_e = econ.utility - ref_w;
		// go back to initial simulation
		econ.e[tidx] -= 1.0;
		t = tidx;
		econ.t = t;
		fair.t = t*econ.tstep;
		for (int subt=0; subt < config.econ_tstep; subt++){
			fair.nextStep(econ.e[t]);
		}
		t++;
		econ.t++;
		while (t < config.horizon/config.econ_tstep){
			nextStep();
		}
		postProcess();
		scc = -1000.0 * (dutil_e / (0.0001 + dutil_c));
		config.scc_ = 0;
		output << niter << "\t" << config.year[tidx] 
			<< "\t" << scc << std::endl;
		// std::cout << scc << std::endl;
	}
	// for (int tidx = 5/econ.tstep; tidx <= 500/econ.tstep; tidx = tidx+10/econ.tstep){
	// 	// compute welfare delta from pulse in consumption
	// 	std::vector<double> scc, dutil_e, dutil_c, ref_w;
	// 	for (int iter=0; iter < niter; iter++){
	// 		fair.sampleUnc();
	// 		econ.sampleUnc();
	// 		simulate();
	// 		postProcess();
	// 		ref_w.push_back(econ.utility);
	// 		// add cons
	// 		econ.c[tidx] += 10.0;
	// 		t = tidx;
	// 		econ.t = t;
	// 		fair.t = t*econ.tstep;
	// 		config.scc_ = 1;
	// 		for (int subt=0; subt < config.econ_tstep; subt++){
	// 			fair.nextStep(econ.e[t]);
	// 		}
	// 		t++;
	// 		econ.t++;
	// 		while (t < config.horizon/config.econ_tstep){
	// 			nextStep();
	// 		}
	// 		postProcess();
	// 		dutil_c.push_back(econ.utility - ref_w[iter]);
	// 		econ.c[tidx] -= 10.0;
	// 		// add emissions
	// 		econ.e[tidx] += 10.0;
	// 		t = tidx;
	// 		econ.t = t;
	// 		fair.t = t*econ.tstep;
	// 		config.scc_ = 1;
	// 		for (int subt=0; subt < config.econ_tstep; subt++){
	// 			fair.nextStep(econ.e[t]);
	// 		}
	// 		t++;
	// 		econ.t++;
	// 		while (t < config.horizon/config.econ_tstep){
	// 			nextStep();
	// 		}
	// 		postProcess();
	// 		dutil_e.push_back(econ.utility - ref_w[iter]);
	// 		econ.e[tidx] -= 10.0;
	// 		// postProcess();
	// 		config.scc_ = 0;

	// 	}
	// 	scc.push_back( -1000.0 * (std::utils::computeMean(dutil_e)) / (0.0001 + std::utils::computeMean(dutil_c) ));			
	// 	output << config.year[tidx] << 
	// 		"\t" << scc[0] << std::endl;				
	// 	config.scc_ = 0;
	// 	// std::cout << scc << std::endl;
	// }
	return;
}

void CDICE::computeSCCDamages(std::ofstream &output, int niter){
	simulate();
	double scc, dutil_d, dutil_c, ref_w, ref_d, disc_rate;
	// ref_w = econ.utility;
	for (int tidx = 1; tidx < econ.horizon; tidx++){
		postProcess(0);
		ref_d = econ.c[tidx] * econ.tstep;
		for (int subt = tidx+1; subt < econ.horizon; subt++) {
			disc_rate = pow((1.0 + econ.prstp + econ.elasmu * (pow(econ.c[econ.horizon-1] / econ.c[0], 1.0/(econ.tstep*(econ.horizon-1))) - 1.0)), (subt - tidx) * econ.tstep);
			// std::cout << econ.prstp + econ.elasmu * (pow(econ.cpc[subt] / econ.cpc[subt-1], 1.0/(econ.tstep)) - 1.0) << std::endl;
			// disc_rate = pow((1.0 + 0.07 ), (subt - tidx) * econ.tstep);
			ref_d = ref_d + econ.tstep * econ.c[subt]/disc_rate;
			// ref_d = ref_d + econ.tstep * econ.c[subt]/pow((1.0 + econ.prstp), (subt - tidx) * econ.tstep);
		}
		// ref_d = econ.pv_consumption[econ.horizon-1];
		// ref_d = econ.pv_damages[econ.horizon-1];
		// ref_w = econ.periodu[tidx];
		// compute welfare delta from pulse in consumption
		// econ.c[tidx] += 1.0;
		// postProcess();
		// dutil_c = econ.periodu[tidx] - ref_w;
		// econ.c[tidx] -= 1.0;
		// postProcess();
		// compute welfare delta from pulse in emissions
		econ.e[tidx] += 1.0;
		t = tidx;
		econ.t = t;
		fair.t = t*econ.tstep;
		config.scc_ = 1;
		for (int subt=0; subt < config.econ_tstep; subt++){
			fair.nextStep(econ.e[t]);
		}
		t++;
		econ.t++;
		while (t < config.horizon/config.econ_tstep){
			nextStep();
		}
		postProcess(0);
		dutil_d = econ.c[tidx] * econ.tstep;
		for (int subt = tidx+1; subt < econ.horizon; subt++) {
			disc_rate = pow((1.0 + econ.prstp + econ.elasmu * (pow(econ.c[econ.horizon-1] / econ.c[0], 1.0/(econ.tstep*(econ.horizon-1))) - 1.0)), (subt - tidx) * econ.tstep);
			// disc_rate = pow((1.0 + 0.07 ), (subt - tidx) * econ.tstep);
			dutil_d = dutil_d + econ.tstep * econ.c[subt]/disc_rate;
		}

		// dutil_d = econ.pv_consumption[econ.horizon-1] - ref_d;
		dutil_d = ref_d - dutil_d;
		// go back to initial simulation
		econ.e[tidx] -= 1.0;
		t = tidx;
		econ.t = t;
		fair.t = t*econ.tstep;
		for (int subt=0; subt < config.econ_tstep; subt++){
			fair.nextStep(econ.e[t]);
		}
		t++;
		econ.t++;
		while (t < config.horizon/config.econ_tstep){
			nextStep();
		}
		postProcess(0);
		scc = pow(10,3) * dutil_d / 5.0;
		config.scc_ = 0;
		output << niter << "\t" << config.year[tidx] 
			<< "\t" << scc << std::endl;
		// std::cout << scc << std::endl;
	}
	return;
}

int CDICE::getNVars(){
	int nvars = 0;
	if (config.adaptive == 0){
		nvars = config.horizon/config.decs_tstep * (2 + 2 * config.adaptation);
	}
	else{ //only works for RBF
		nvars = dvars.p_param.policyStr * \
			(2*dvars.p_param.policyInput + dvars.p_param.policyOutput) + 
			dvars.p_param.policyOutput;
	}
	return nvars;
}

int CDICE::getNObjs(){
	return config.nobjs;
}

void CDICE::setVariables(double *vars){
	dvars.setVariables(vars);
	return;
}


void CDICE::writeOutputHeader(std::ofstream &output){
	output <<  "NITER\tYEAR\tMIU\tS\tIA\tFAD\tSAD\t"
		// "IPRADA\tIRADA\tSCAP\tKPRADA\tKSCAP\t"
		"K\tTATM\tC"
		"\tPOP\tTFP\tSIGMA"
		"\tECS\tTCR\tADAPTEFF\tDAMTYPE\tRFOTH\tGRUBB"
		"\tEIND\tGDP\tABATECOST\tMCABATE\tADAPTCOST"
		"\tDAMFRAC\tRD\tDAMAGES\tADAPT\tCEMUTOTPERT\tRI" << std::endl ;	
	return;
}

void CDICE::writeOutput(std::ofstream &output, int niter){
	t = 0;	
	while (t < config.horizon/config.econ_tstep){
		output << niter << "\t" << config.year[t] << "\t" <<
			dvars.miu[(int) t*config.econ_tstep/config.decs_tstep] << "\t" <<
			dvars.s[(int) t*config.econ_tstep/config.decs_tstep] << "\t" <<
			dvars.p_ia[(int) t*config.econ_tstep/config.decs_tstep] << "\t" <<
			dvars.p_fad[(int) t*config.econ_tstep/config.decs_tstep] << "\t" <<
			econ.sad[t] << "\t" <<
			// dvars.p_iprada[t] << "\t" <<
			// dvars.p_irada[t] << "\t" <<
			// dvars.p_scap[t] << "\t" <<
			// econ.kprada[t] << "\t" <<
			// econ.kscap[t] << "\t" <<
			econ.k[t] << "\t" <<
			fair.gmst[t*config.econ_tstep] << "\t" <<
			fair.c[t*config.econ_tstep] << "\t" <<
			econ.pop[t] << "\t" <<
			econ.tfp[t] << "\t" <<
			econ.sigma[t] << "\t" <<
			fair.ECS << "\t" <<
			fair.TCR << "\t" <<
			config.adapteff << "\t" <<
			config.damages << "\t" <<
			config.rfoth << "\t" <<
			config.grubb << "\t" <<
			econ.eind[t] << "\t" <<
			econ.y[t] << "\t" <<
			econ.abatecost[t] << "\t" <<
			econ.mabatecost[t] << "\t" <<
			econ.adaptcost[t] << "\t" <<
			econ.damfrac[t] << "\t" <<
			econ.rd[t] << "\t" <<
			econ.damages[t] << "\t" <<
			econ.adapt[t] << "\t" <<
			// econ.damfrac[t] << "\t" <<
			// econ.a0_agr * fair.gmst[t*econ.tstep] +
			// 	 econ.a1_agr * pow(fair.gmst[t*econ.tstep], econ.a2_agr) << "\t" <<
			// econ.adapt[t] << "\t" <<
			// econ.adapt[t] / (1.0+econ.adapt[t]) << "\t" <<
			// econ.rd[t] << "\t" <<
			econ.cemutotper[t] << "\t" <<
			// 1.0/pow(1.0 + econ.prstp + econ.elasmu * 
			// (pow(econ.c[t] / econ.c[0], 1.0/(t*config.econ_tstep)) - 1.0), 
			// t*config.econ_tstep) << "\t" <<
			econ.ri_disc[t] << 
			std::endl;
		t++;
	}	
	return;
}
