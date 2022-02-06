#include "./econ.h"
#include <math.h>
#include <iostream>

Econ::Econ(){
	
}

Econ::~Econ(){
	delete[] ia;
	delete[] fad;
	delete[] sad;
	delete[] adaptcost;
	delete[] adapt;
	delete[] rd;
	delete[] pop;
	delete[] pop5;
	delete[] tfp;
	delete[] cost1;
	delete[] pbacktime;
	delete[] ga;
	delete[] gsig;
	delete[] sigma;
	delete[] eland;
	delete[] c;
	delete[] k;
	delete[] cpc;
	delete[] i;
	delete[] ri;
	delete[] y;
	delete[] ygross;
	delete[] ynet;
	delete[] damages;
	delete[] damfrac;
	delete[] abatecost;
	delete[] mabatecost;
	delete[] periodu;
	delete[] cemutotper;
	delete[] scc;
	delete[] rr;
	delete[] ri_disc;
	delete[] pv_damages;
	delete[] pv_abatecost;
	delete[] totalcost;
	delete[] pv_totalcost;
	delete[] pv_consumption;
	delete[] pv_adaptcost;
	delete[] e;
	delete[] eind;
	delete[] miueff;

	delete[] iprada;
	delete[] irada;
	delete[] iscap;
	delete[] qact;
	delete[] kprada;
	delete[] qgcap;
	delete[] qcap;
	delete[] kscap;

}

void Econ::allocate(Config *configPtr, Dvars *dvarsPtr){
	config = configPtr;
	dvars = dvarsPtr;
	horizon = config->horizon/config->econ_tstep;
	tstep = config->econ_tstep;
	t = 0;
	// economial params
	elasmu = 1.45;
	prstp = 0.015;
	gama = 0.3;
	dk = 0.1;
	q0 = 105.5;
	k = new double[horizon+1];
	k[0] = 223.0;
	tfp = new double[horizon+1];
	noise_tfp = new double[horizon+1];
	noise_tfp[0] = 0.0;
	stdev_tfp = 0.015942083088376052;
	tfp[0] = 5.115;
	ga = new double[horizon+1];
	ga[0] = 0.076;
	dela = 0.005;
	optlrsav = (dk + 0.004) / (dk + 0.004 * elasmu + prstp) * gama;
	// population
	pop = new double[horizon+1];
	pop[0] = 7403;
	pop5 = new double[horizon+1];
	pop5[0] = 7403;
	popadj = 0.134;
	popasym = 11500;
	// carbon intensity
	sigma = new double[horizon+1];
	noise_sigma = new double[horizon+1];
	noise_sigma[0] = 0.0;
	ar_model_sigma = 0.88387757;
	stdev_sigma = 0.04679427678753168;
	gsig = new double[horizon+1];
	eind = new double[horizon+1];
	miueff = new double[horizon+1];
	eland = new double[horizon+1];
	e = new double[horizon+1];
	gsig[0] = -0.0152;
	dsig = -0.001;
	eland[0] = 2.6;
	deland = 0.115;
	eind[0] = 35.85;
	e[0] = eind[0] + eland[0];
	miu0 = 0.03;
	sigma[0] = eind[0]/(q0 * (1-miu0));

	// Climate damage parameters ---------------------
	a2_hs = 0.007438;
	a0_agr = 0.003;
	a1_agr = 0.0007;
	a2_agr = 3.62;

	// NEEDED FOR AD-DICE
	b1 = 90;
	b2 = 0.49;
	b3 = 0.8;
	rho = 0.5;
	ia = new double[horizon+1];
	fad = new double[horizon+1];
	sad = new double[horizon+1];
	adaptcost = new double[horizon+1];
	adapt = new double[horizon+1];
	rd = new double[horizon+1];


	iprada = new double[horizon+1];
	irada = new double[horizon+1];
	iscap = new double[horizon+1];
	qact = new double[horizon+1];
	kprada = new double[horizon+1];
	qgcap = new double[horizon+1];
	qcap = new double[horizon+1];
	kscap = new double[horizon+1];
	dkscap = 0.03;
	dkprada = 0.1;

	// Abatement cost --------------------------------
	expcost2 = 2.6;	
	pback = 550;		// Cost of backstop [2005$ per tCO2 2010]
	gback = 0.025;		// Initial cost decline backstop [cost per period]
	// cprice0 = 2;	// Initial base carbon price [2005$ per tCO2]
	// gcprice = 0.02;	// Growth rate of base carbon price (per year)
	pli = 0.5;
	plitime = 40;

	scale1 = 0.0302455265681763;	// Multiplicitive scaling coefficient
	scale2 = -10993.704;     // Additive scaling coefficient

	// Exogenous timeseries variables
	cost1 = new double[horizon+1];
	pbacktime = new double[horizon+1];
	pbacktime[0] = pback ;
	cost1[0] = pbacktime[0] * sigma[0]/expcost2/1000.0;
	scc = new double[horizon+1];
	rr = new double[horizon+1];
	rr[0] = 1.0;

	// Endogenous variables
	c = new double[horizon+1];
	cpc = new double[horizon+1];
	i = new double[horizon+1];
	ri = new double[horizon+1];
	ri[0] = 1.0;
	y = new double[horizon+1];
	ygross = new double[horizon+1];
	ynet = new double[horizon+1];
	damages = new double[horizon+1];
	damfrac = new double[horizon+1];
	abatecost = new double[horizon+1];
	mabatecost = new double[horizon+1];
	periodu = new double[horizon+1];
	cemutotper = new double[horizon+1];
	utility = 0.0;
	ri_disc = new double[horizon+1];
	ri_disc[0] = 1.0;
	pv_damages = new double[horizon+1];
	pv_abatecost = new double[horizon+1];
	pv_adaptcost = new double[horizon+1];
	totalcost = new double[horizon+1];
	pv_totalcost = new double[horizon+1];
	pv_consumption = new double[horizon+1];

	owaactc = 6.0;
	owarada = 0.8;
	owaprada = 0.999;
	owagcap = 0.99;
	owascap = 0.9;
	owaact = 30.;
	owacap = 0.5;
	rhoact = 0.17;
	rhocap = -4.0;
	rhotfp = 3.9;
	rhoada = -0.11111;
	k0rd = 0.124282;
	k0edu = 45.6883865;
	deg1 = 0.004753846;
	deg2 = 0.003;
	deg3 = 2.0;
	deg4 = 0.001692308;

	owaactc = 1.965384615;
	owarada = 0.486230769;
	owaprada = 0.513769231;
	owagcap = 0.553076923;
	owascap = 0.446923077;
	owaact = 5.028461538;
	owacap = 0.213846154;
	rhoact = 0.17;
	rhocap = -4.0;
	rhotfp = 1.310769231;
	rhoada = -0.11111;
	k0rd = 0.124282;
	k0edu = 45.6883865;
	deg1 = 0.004753846;
	deg2 = 0.003;
	deg3 = 2.0;
	deg4 = 0.001692308;
	// double *k_static;   // Level of total factor productivity
	// double *sad_static;   // Level of total factor productivity
	// double *al_static;   // Level of total factor productivity
	// double *l_static;   // Level of total factor productivity
	// double *sigma_static;   // Level of total factor productivity
	// double *pbacktime_static;   // Level of total factor productivity

	return;
}

void Econ::nextStep(double temp){
	// compute exogenous trajectories
	if (t%(5/tstep) == 0){
		pop5[t/(5/tstep)+1] = pop5[t/(5/tstep)] * 
			pow(popasym/pop5[t/(5/tstep)], popadj);
	}
	pop[t + 1] = pop5[t/(5/tstep)] + (t%(5/tstep) + 1.0) / 
		 (5.0/tstep) * (pop5[t/(5/tstep)+1] - pop5[t/(5/tstep)]);
	ga[t + 1] = ga[0] * exp(-dela*tstep*double(t + 1));
	tfp[t + 1] = tfp[t] / pow(1-ga[t], tstep/5.0);
	if (config->tfp_stoch == 1){
		if (config->scc_ == 0){
			noise_tfp[t+1] = std::max(-3.0, std::min(3.0, 
				sqrt(-2 * log(rand() * (1.0 / RAND_MAX)))
				 * sin (2 * M_PI * rand() * (1.0 /RAND_MAX)))) ; 			
		}
		tfp[t+1] = tfp[t+1] * (1 + noise_tfp[t+1] * stdev_tfp * pow(tstep,0.5));
	}
	gsig[t + 1] = gsig[t]*(pow(1+dsig, tstep));
	sigma[t + 1] = sigma[t] * exp(gsig[t]*tstep);
	if (config->sigma_stoch == 1){
		if (config->scc_ == 0){
			noise_sigma[t+1] = std::max(-3.0, std::min(3.0, 
				sqrt(-2 * log(rand() * (1.0 / RAND_MAX)))
				 * sin (2 * M_PI * rand() * (1.0 /RAND_MAX)))) ; 
		}
		// correlated model, ok only for annual time step
		// sigma[t+1] = sigma[t+1] * 
		// 	(1 + stdev_sigma * (noise_sigma[t+1] + ar_model_sigma * noise_sigma[t]) );
		sigma[t+1] = sigma[t+1] * 
			(1 + stdev_sigma * (noise_sigma[t+1]) * pow(tstep,0.5));
	}
	pbacktime[t + 1] = pback * pow(1-gback, double(t + 1) * tstep / 5);
	cost1[t + 1] = pbacktime[t + 1] * sigma[t + 1]/expcost2/1000.0;
	eland[t + 1] = eland[0]*pow(1-deland, double(t + 1) * tstep / 5);
	rr[t + 1] = 1/pow(1+prstp, double(t + 1) * tstep);

	// compute endogenous variables
	ygross[t] = tfp[t] * pow((pop[t] / 1000.0) , (1.0 - gama)) * 
	    pow(k[t], gama);
	// check constraints on miu
	if (2015 + t * tstep == 2020){
		dvars->miu[(int) t*tstep/config->decs_tstep] = std::max(0.0,
			1.0 - (eind[0] * pow(1.01,5 + t*tstep%tstep)) / 
			(sigma[t] * ygross[t]));
	}
	// uncomment below for inaction analysis
	// if (2015 + t * tstep <= 2025){
	// 	dvars->miu[(int) t*tstep/config->decs_tstep] = std::max(0.0,
	// 		1.0 - (eind[0] * pow(1.01,5 + 1*tstep%tstep)) / 
	// 		(sigma[t] * ygross[t]));		
	// }
	// if (2015 + t*tstep > 2024 && 2015 + t*tstep < 2050){
	// 	dvars->miu[(int) t*tstep/config->decs_tstep] = std::max(0.0,
	// 		std::min(1.0, std::min(dvars->miu[(int) t*tstep/config->decs_tstep],
	// 		1.0 - (eind[t-1] - 2 * tstep) / 
	// 		(sigma[t] * ygross[t])) ) ); 
	// }
	// if (2015 + t*tstep >= 2050){
	// 	// NET limit is equal to 
	// 	// the maximum NET observed 
	// 	// in each time step in the CMIP6 scenarios from SSP database
	double miuMaxNET; 
	if (2015 + t*tstep <= 2050){
		miuMaxNET = 1.0;
	}
	else if (2015 + t*tstep <= 2060){
		miuMaxNET = 1.0 - ( - 1525.978 * pow(10.0,-3) / (sigma[t] * ygross[t]) );
	}
	else if (2015 + t*tstep <= 2070){
		miuMaxNET = 1.0 - ( - 5768.062 * pow(10.0,-3) / (sigma[t] * ygross[t]) );			
	}
	else if (2015 + t*tstep <= 2080){
		miuMaxNET = 1.0 - ( - 14855.333 * pow(10.0,-3) / (sigma[t] * ygross[t]) );			
	}
	else if (2015 + t*tstep <= 2090){
		miuMaxNET = 1.0 - ( - 18556.245 * pow(10.0,-3) / (sigma[t] * ygross[t]) );			
	}
	else {
		miuMaxNET = 1.0 - ( - 20311.382 * pow(10.0,-3) / (sigma[t] * ygross[t]) );			
	}
	miueff[t] = std::max(0.0,
		std::min(miuMaxNET, dvars->miu[(int) t*tstep/config->decs_tstep]) );
		// dvars->miu[(int) t*tstep/config->decs_tstep] = std::max(0.0,
		// 	std::min(miuMaxNET, std::min(dvars->miu[(int) t*tstep/config->decs_tstep],
		// 		dvars->miu[(int) (t-1)*tstep/config->decs_tstep] 
		// 		* pow(1.1,0.2*config->decs_tstep)) ) );
	// }
	// abatement costs
	if (config->grubb == 0){
		abatecost[t] = ygross[t] * cost1[t] * pow(miueff[t] , expcost2); // standard abatement costs
		mabatecost[t] = pbacktime[t] * pow(miueff[t] , expcost2); // standard abatement costs
	}
	// Abatement costs (Grubb, 2021)
	else{
		if (t > 0){
			abatecost[t] = ygross[t] * cost1[t] * 
				( (1 - pli) * pow(miueff[t], expcost2) + 
				pli * pow(plitime, expcost2) / (expcost2 + 1) * 
				pow(1.0 / config->decs_tstep * std::max(0.0,miueff[t] - miueff[t-1]), expcost2) );
			mabatecost[t] = pbacktime[t] * 
				( (1 - pli) * pow(miueff[t], expcost2) + 
				pli * pow(plitime, expcost2) / (expcost2 + 1) * 
				pow(1.0 / config->decs_tstep * std::max(0.0,miueff[t] - miueff[t-1]), expcost2) );
		}
		else{
			abatecost[t] = ygross[t] * cost1[t] * 
				( (1 - pli) * pow(miueff[t], expcost2) + 
				pli * pow(plitime, expcost2) / 
				(expcost2 + 1) * pow(1.0 / config->decs_tstep * std::max(0.0, miueff[t]), expcost2) );  
			mabatecost[t] = pbacktime[t] * 
				( (1 - pli) * pow(miueff[t], expcost2) + 
				pli * pow(plitime, expcost2) / 
				(expcost2 + 1) * pow(1.0 / config->decs_tstep * std::max(0.0, miueff[t]), expcost2) );  
		}		
	}
	//damages
	if (config->damages==1){
		damfrac[t] =  a2_hs * pow(temp, 2);  // Howard Sterner
	}
	else if(config->damages==2){ //BURKE using polynomial damage update (see surrogateBurke.py)
		double temp_abs = temp + 13.48;
		damfrac[t+1] = std::max(damfrac[t], damfrac[t] + (1.0/5.0*tstep) *(
			0.00099337 * pow(temp_abs - (- 0.62685192), 2) + 
			(-0.01453289) * (temp_abs - (- 0.74157515) ) + 
			(-0.69189433) * pow (damfrac[t]  - (+ 0.00825518), 2) + 
			(0.05745062) * (damfrac[t]  - (+ 0.01692067) ) ) ) ;
	}
	// Adaptation Agrawala
	if (config->adaptation == 1){
		ia[t] = dvars->p_ia[(int) t*tstep/config->decs_tstep] ;
		fad[t] = dvars->p_fad[(int) t*tstep/config->decs_tstep] ;
		adaptcost[t] = (ia[t] + fad[t]) * ygross[t];
		adapt[t] = config->adapteff * b1 * pow((b2 * pow(fad[t],rho) + 
			(1 - b2) * pow(sad[t], rho) ), b3/rho);
		rd[t] = std::max(0.0, damfrac[t] / (1 + adapt[t]));// - 
			// std::min(damfrac[t],(a0_agr * temp + a1_agr * pow(temp, a2_agr))) * 
			// adapt[t] / (1.0 + adapt[t]) );
		// rd[t] = (a0_agr * temp + a1_agr * pow(temp, a2_agr)) / (1.0 + adapt[t]) ;
		sad[t+1] = ia[t] * tstep / 5 + pow(1 - dk, tstep) * sad[t];	
	}
	//Adaptation AD-WITCH
	else if (config->adaptation == 2){
		iprada[t] = dvars->p_iprada[(int) t*tstep/config->decs_tstep] * ygross[t];
		irada[t] = dvars->p_irada[(int) t*tstep/config->decs_tstep] * ygross[t];
		iscap[t] = dvars->p_scap[(int) t*tstep/config->decs_tstep] * ygross[t];
		adaptcost[t] = iprada[t] + irada[t] +iscap[t];

		qact[t] = config->adapteff * owaactc * pow(owarada * pow(irada[t] ,rhoact) + 
			owaprada * pow(kprada[t], rhoact) , 1.0/rhoact );
		qgcap[t] = (k0rd + k0edu)/1000.0/2.0 * tfp[t];
		qcap[t] = pow(owagcap * pow(qgcap[t], rhocap) + 
			owascap * pow(kscap[t], rhocap), 1.0/rhocap);
		adapt[t] = rhotfp * pow(owaact * pow(qact[t] , rhoada) + 
			owacap * pow(qcap[t], rhoada), 1.0/rhoada );
		rd[t] = std::max(0.0, damfrac[t] - 
			std::min(damfrac[t],(deg1 * temp + deg2 * pow(temp, deg3) + deg4)) * 
			adapt[t] / (1.0 + adapt[t]) );

		kprada[t+1] = iprada[t] * tstep + pow(1 - dkprada, tstep) * kprada[t];	
		kscap[t+1] = iscap[t] * tstep + pow(1 - dkscap, tstep) * kscap[t];	
	}
	else{
		rd[t] = damfrac[t];
	}
	//
    if (config->cost_effective==1){
		ynet[t] = ygross[t];
	}
	else{
		ynet[t] = ygross[t] * (1.0 - rd[t]);
	}
	// Damages
	damages[t] = ygross[t]*rd[t]; //rd[t] in place of damfrac[t] -> AD-DICE    
	// Gross world product (net of abatement and damages) //AD-DICE and of adaptation costs
	y[t] = std::max(pow(10.0,-6), ynet[t] - abatecost[t] - adaptcost[t]);
	// Industrial emissions
	eind[t] = sigma[t] * ygross[t] * (1.0-miueff[t]);
	// Total emissions  
	e[t] = eind[t]+eland[t];
	// Investment
	i[t] = std::min(0.99, std::max(0.01,dvars->s[(int) t*tstep/config->decs_tstep])) *y[t];
	// Consumption
	c[t] = y[t]-i[t];
	// Per capita consumption
	cpc[t] = 1000.0 * c[t] / pop[t];
	// Real interest rate
	if (t > 0){
		// ri[t-1] = (1.0 + prstp) * pow(cpc[t]/cpc[t-1] , elasmu/tstep) - 1.0;
		ri[t] = prstp + elasmu * (pow(cpc[t] / cpc[0], 1.0/(t*tstep)) - 1.0);
		// ri[t] = prstp + elasmu * (c[t] / c[t-1] - 1.0);
	}
	// Capital stock
	k[t + 1] = std::max(pow(10.0, -6), tstep * i[t] + pow(1 - dk , tstep) * k[t]);
	if (y[t] == pow(10.0,-6)){
		k[t + 1] = 0.0;
		c[t] = 0.0;
		cpc[t] = 0.0;
		ri[t] = prstp;
	}

	t++;
	return;
}

void Econ::sampleUnc(){
	if (config->pop_unc == 1){
		popasym = rand() * (1.0/RAND_MAX) * (13800 - 9200) + 9200; //parameters coming from lamontagne et al. (2019)
		popadj = rand() * (1.0/RAND_MAX) * (0.201 - 0.067) + 0.067; //parameters coming from lamontagne et al. (2019)
	}
	if (config->tfp_unc==1){
		ga[0] = rand() * (1.0/RAND_MAX) * (0.0912 - 0.076) + 0.076; //parameters coming from lamontagne et al. (2019)
		dela = rand() * (1.0/RAND_MAX) * (0.006 - 0.004) + 0.004; //parameters coming from lamontagne et al. (2019)
	}
	if (config->sigma_unc==1){
		gsig[0] = rand() * (1.0/RAND_MAX) *(-0.014 - (-0.01824)) + (-0.01824); //parameters coming from lamontagne et al. (2019)
		dsig = rand() * (1.0/RAND_MAX) *(-0.0005  - (-0.0015)) + (-0.0015); //parameters coming from lamontagne et al. (2019)
	}
	if (config->abatetype_unc == 1){
		config->grubb = (rand() * (1.0/RAND_MAX) > 0.5);
	}
	if (config->abate_unc==1){
		expcost2 = rand() * (1.0/RAND_MAX) *(2.786 - 2.414) + 2.414; //parameters coming from lamontagne et al. (2019)
		pback = rand() * (1.0/RAND_MAX) *(660 - 440) + 440; //parameters coming from lamontagne et al. (2019)
		gback = rand() * (1.0/RAND_MAX) *(0.0275 - 0.0225) + 0.0225; //parameters coming from lamontagne et al. (2019)
		if (config->grubb==1){
			pli = rand() * (1.0/RAND_MAX);  // uncertainties from grubb et al. (2021)
			plitime = rand() * (1.0/RAND_MAX) * (40-20) + 20; // uncertainties from grubb et al. (2021)			
		}
	}
	if (config->adapt_unc==1){
		config->adapteff = ((rand() * (1.0/RAND_MAX)) > 0.5) * rand() * (1.0/RAND_MAX) * 2.0; //parameters coming from lamontagne et al. (2019)
	}
	if (config->damages_unc==1){
		config->damages = 1 + (rand() * (1.0/RAND_MAX) > 0.5);
	}
	return;
}

void Econ::reset(){
	t = 0;
}