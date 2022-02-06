#include "./dvars.h"
#include "../econ/econ.h"

Dvars::Dvars(){
	
}

Dvars::~Dvars(){
	delete[] miu;
	delete[] s;
	delete[] p_ia;
	delete[] p_fad;
	delete[] p_iprada;
	delete[] p_irada;
	delete[] p_scap;
}

void Dvars::allocate(Config *configPtr){
	config = configPtr;
	horizon = config->horizon / config->decs_tstep;
	tstep = config->decs_tstep;

	miu = new double[horizon+1];
	s = new double[horizon+1];
	p_ia = new double[horizon+1];
	p_fad = new double[horizon+1];
	p_iprada = new double[horizon+1];
	p_irada = new double[horizon+1];
	p_scap = new double[horizon+1];

	if (config->adaptive==1){
		allocatePolicy();
	}

	return;
}

void Dvars::setVariables(double *vars){
	if (config->adaptive == 0){
		for (int tidx=0; tidx < horizon; tidx++){
			miu[tidx] = vars[0];
			s[tidx] = vars[1];
			vars = vars + 2;
			if (config->adaptation == 1){
				p_ia[tidx] = vars[0];
				p_fad[tidx] = vars[1];
				vars = vars + 2;
			}
			if (config->adaptation == 2){
				p_iprada[tidx] = vars[0];
				p_irada[tidx] = vars[1];
				p_scap[tidx] = vars[2];
				vars = vars + 3;
			}
		}
	}
	else{
		//set policy parameters
		policy->clearParameters();
		policy->setParameters(vars);
	}
	return;
}

void Dvars::readPolicySettings(){

    std::ifstream in;
    std::string sJunk = "";
    in.open("./src/settingsCDICEPolicy.txt", std::ios_base::in);
    if(!in)
    {
		std::cout << "The Policy settings file specified could not be found!" << std::endl;
		exit(1);
    }

    //Look for the <POLICY_CLASS> key
    while (sJunk != "<POLICY_CLASS>")
    {
		in >> sJunk;
    }
    in >> p_param.tPolicy;
    //Return to the beginning of the file
    in.seekg(0, std::ios::beg);

    //Look for the <NUM_INPUT> key
    double i1, i2;
    while (sJunk != "<NUM_INPUT>")
    {
		in >> sJunk;
    }
    in >> p_param.policyInput;
	in.ignore(1000,'\n');
    //Loop through all of the input data and read in this order:
    for (int i=0; i< p_param.policyInput; i++)
    {
		in >> i1 >> i2;
		p_param.mIn.push_back(i1);
		p_param.MIn.push_back(i2);
		in.ignore(1000,'\n');
    }
    if (config->adaptation==1){
    	p_param.policyInput += 1;
    }
    if (config->adaptation==2){
    	p_param.policyInput += 2;
    }
    if (config->adaptation==1){
    	p_param.mIn.push_back(0.0);
		p_param.MIn.push_back(0.1);
    }
    if (config->adaptation==2){
    	p_param.mIn.push_back(0.0);
		p_param.MIn.push_back(0.1);
    	p_param.mIn.push_back(0.0);
		p_param.MIn.push_back(0.1);
    }
    //Return to the beginning of the file
    in.seekg(0, std::ios::beg);

    //Look for the <NUM_OUTPUT> key
    double o1, o2;
    while (sJunk != "<NUM_OUTPUT>")
    {
        in >> sJunk;
    }
    in >> p_param.policyOutput;
    in.ignore(1000,'\n');
    //Loop through all of the input data and read in this order:
    for (int i=0; i< p_param.policyOutput; i++)
    {
		in >> o1 >> o2;
		p_param.mOut.push_back(o1);
		p_param.MOut.push_back(o2);
		in.ignore(1000,'\n');
    }
    if (config->adaptation==1){
    	p_param.policyOutput += 2;
    }
    if (config->adaptation==2){
    	p_param.policyOutput += 3;
    }
    if (config->adaptation==1){
		p_param.mOut.push_back(0.0);
		p_param.MOut.push_back(0.1);
		p_param.mOut.push_back(0.0);
		p_param.MOut.push_back(0.1);
    }
    if (config->adaptation==2){
		p_param.mOut.push_back(0.0);
		p_param.MOut.push_back(0.1);
		p_param.mOut.push_back(0.0);
		p_param.MOut.push_back(0.1);
		p_param.mOut.push_back(0.0);
		p_param.MOut.push_back(0.1);
    }   
    //Return to the beginning of the file
    in.seekg(0, std::ios::beg);

    //Look for the <POLICY_STRUCTURE> key
    while (sJunk != "<POLICY_STRUCTURE>")
    {
		in >> sJunk;
    }
    in >> p_param.policyStr;
    //Return to the beginning of the file
    in.seekg(0, std::ios::beg);

    //Close the input file
    in.close();

    return;
}

void Dvars::allocatePolicy(){
	readPolicySettings();
	switch (p_param.tPolicy) {
		case 1: { // RBF policy
			policy = new std::rbf(p_param.policyInput,p_param.policyOutput,p_param.policyStr);
			break;
		}
		case 2:{ // ANN
			policy = new std::ann(p_param.policyInput,p_param.policyOutput,p_param.policyStr);
			break;
		}
		case 3:{ // piecewise linear policy
			policy = new std::pwLinear(p_param.policyInput,p_param.policyOutput,p_param.policyStr);
			break;
		}
		case 4:{
			policy = new std::ncRBF(p_param.policyInput,p_param.policyOutput,p_param.policyStr);
			break;
		}
		case 5:{
			policy = new std::annmo(p_param.policyInput,p_param.policyOutput,p_param.policyStr);
			break;
		}
		default:
			break;
	}
	policy->setMaxInput(p_param.MIn); policy->setMaxOutput(p_param.MOut);
	policy->setMinInput(p_param.mIn); policy->setMinOutput(p_param.mOut);
	return;
}

void Dvars::nextAction(std::vector<double> states, int t){
	if (config->adaptive == 1) {
		controls.clear();
		controls = policy->get_NormOutput(states);
		miu[t] = controls[0];
		s[t] = controls[1];
		if (config->adaptation == 1){
			p_ia[t] = controls[2];
			p_fad[t] = controls[3];
		}
		if (config->adaptation == 2){
			p_iprada[t] = controls[2];
			p_irada[t] = controls[3];
			p_scap[t] = controls[4];
		}
	}
	miu[t] = std::max(0.0, std::min(1.2, miu[t]));
	s[t] = std::max(0.1, std::min(0.9, s[t]));
	if (config->adaptation == 1){
		p_ia[t] = std::max(0.0, std::min(0.2, p_ia[t]));
		p_fad[t] = std::max(0.0, std::min(0.2, p_fad[t]));
	}
	if (config->adaptation == 2){
		p_iprada[t] = std::max(0.0, std::min(0.2, p_iprada[2]));
		p_irada[t] = std::max(0.0, std::min(0.2, p_irada[3]));
		p_scap[t] = std::max(0.0, std::min(0.2, p_scap[4]));
	}
	if (2015 + t * tstep < 2020){
		miu[t] = 0.03;
		s[t] = 0.27;
		p_ia[t] = 0.0;
		p_fad[t] = 0.0;
		p_iprada[t] = 0.0;
		p_irada[t] = 0.0;
		p_scap[t] = 0.0;
	}
	// std::cout << t << std::endl;
	// for (int ns=0; ns < states.size() ; ns++){
	// 	std::cout << states[ns] << " ";
	// }
	// std::cout << miu[t] << std::endl;
	return;
}

