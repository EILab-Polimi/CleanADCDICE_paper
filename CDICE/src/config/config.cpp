#include "./config.h"
#include <iostream>
#include <fstream>
#include <string>

Config::Config(){
	std::ifstream in;
	in.open("./src/config/config.txt");
	std::string sJunk = "";
	while (sJunk!="horizon"){
		in >> sJunk;
	}
	in >> horizon;
	while (sJunk!="econ_tstep"){
		in >> sJunk;
	}
	in >> econ_tstep;
	while (sJunk!="decs_tstep"){
		in >> sJunk;
	}
	in >> decs_tstep;
	while (sJunk!="nobjs"){
		in >> sJunk;
	}
	in >> nobjs;
	while (sJunk!="adaptive"){
		in >> sJunk;
	}
	in >> adaptive;
	while (sJunk!="time_pol"){
		in >> sJunk;
	}
	in >> time_pol;
	while (sJunk!="annmo"){
		in >> sJunk;
	}
	in >> annmo;
	while (sJunk!="exog"){
		in >> sJunk;
	}
	in >> exog;
	while (sJunk!="adaptation"){
		in >> sJunk;
	}
	in >> adaptation;
	while (sJunk!="adapteff"){
		in >> sJunk;
	}
	in >> adapteff;
	while (sJunk!="damages"){
		in >> sJunk;
	}
	in >> damages;
	while (sJunk!="niter"){
		in >> sJunk;
	}
	in >> niter;
	while (sJunk!="tatm_stoch"){
		in >> sJunk;
	}
	in >> tatm_stoch;
	while (sJunk!="tfp_stoch"){
		in >> sJunk;
	}
	in >> tfp_stoch;
	while (sJunk!="sigma_stoch"){
		in >> sJunk;
	}
	in >> sigma_stoch;
	while (sJunk!="cl_sen_unc"){
		in >> sJunk;
	}
	in >> cl_sen_unc;
	while (sJunk!="pop_unc"){
		in >> sJunk;
	}
	in >> pop_unc;
	while (sJunk!="tfp_unc"){
		in >> sJunk;
	}
	in >> tfp_unc;
	while (sJunk!="sigma_unc"){
		in >> sJunk;
	}
	in >> sigma_unc;
	while (sJunk!="abate_unc"){
		in >> sJunk;
	}
	in >> abate_unc;
	while (sJunk!="abatetype_unc"){
		in >> sJunk;
	}
	in >> abatetype_unc;
	while (sJunk!="adapt_unc"){
		in >> sJunk;
	}
	in >> adapt_unc;
	while (sJunk!="damages_unc"){
		in >> sJunk;
	}
	in >> damages_unc;
	while (sJunk!="rfoth_unc"){
		in >> sJunk;
	}
	in >> rfoth_unc;
	while (sJunk!="limmiu_s"){
		in >> sJunk;
	}
	in >> limmiu_s;
	while (sJunk!="cost_effective"){
		in >> sJunk;
	}
	in >> cost_effective;
	while (sJunk!="writefile"){
		in >> sJunk;
	}
	in >> writefile;
	while (sJunk!="scc"){
		in >> sJunk;
	}
	in >> scc;
	scc_ = 0;
	while (sJunk!="grubb"){
		in >> sJunk;
	}
	in >> grubb;
	while (sJunk!="rfoth"){
		in >> sJunk;
	}
	in >> rfoth;

	year = new int[horizon/econ_tstep];
	for (int idx=0; idx < horizon/econ_tstep; idx++){
		year[idx] = 2015 + idx * econ_tstep;
	}
	in.close();
	return;
}

Config::~Config(){
	delete[] year;
}