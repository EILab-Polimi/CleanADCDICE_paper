#include <iostream>
#include <fstream>
#include "CDICE.h"
#include "./emodps/utils.h"
#include "./moeaframework/moeaframework.h"

int main(int argc, char *argv[]){
	CDICE dice;
	int seed;
	if (argc > 1){
		seed = atoi(argv[1]);
		srand(seed);
	}
	else{
		srand(time(NULL));
	}
	double vars[dice.getNVars()];
	double objs[dice.getNObjs()];
	// std::cout << dice.getNVars() << std::endl;
	// Run model
	std::ofstream outputFile;
	clock_t start, end;
	start = clock();
	MOEA_Init(dice.getNObjs(), 0);
	while (MOEA_Next_solution() == MOEA_SUCCESS) {
		MOEA_Read_doubles(dice.getNVars(), vars);
		dice.setVariables(vars);
		srand(seed);
		std::vector<std::vector<double>> objsvec;
		for (int nobj = 0; nobj < 8 ; nobj++){
			objsvec.push_back(std::vector<double>());
		}
		if (dice.config.writefile==1){
			outputFile.open("./SimulationsOutput.txt");
			dice.writeOutputHeader(outputFile);
		}
		for (int nidx=0; nidx < dice.config.niter ; nidx++){
			dice.fair.sampleUnc();
			dice.econ.sampleUnc();
			dice.simulate();
			if (dice.config.writefile==1){
				dice.writeOutput(outputFile, nidx);
			}
			// std::cout << -dice.econ.utility << std::endl;
			// allsols['\u0394 CBGE [%]'] = 100 * ((allsols.loc[allsols['Type']=='SO']['Welfare'].min() / allsols['Welfare'])**(1/(1-1.45)) - 1)

			if (dice.econ.utility < 0){
				dice.econ.utility = 1e-3;
			}
			objsvec[0].push_back(-dice.econ.utility);
			objsvec[7].push_back(dice.fair.oneFiveDegYrs);
			objsvec[2].push_back(dice.fair.twoDegYrs);
			objsvec[3].push_back(dice.econ.pv_damages[dice.econ.horizon-1]);
			objsvec[4].push_back(dice.econ.pv_abatecost[dice.econ.horizon-1]);
			objsvec[5].push_back(dice.econ.pv_adaptcost[dice.econ.horizon-1]);
			objsvec[6].push_back(dice.fair.aboveOneFive);
			objsvec[1].push_back(dice.fair.aboveTwo);
		}
		for (int nobj = 0; nobj < 8 ; nobj++){
			if (nobj < dice.getNObjs()){
				objs[nobj] = std::utils::computeMean(objsvec[nobj]);
				// if (nobj == 0){
				// 	if (objs[nobj] < 0){
				// 		objs[nobj] = 1e-3;
				// 	}
				// 	objs[nobj] = -100.0 * (pow( 509461.95256387384 / objs[nobj], 1.0/(1.0-1.45) ) - 1.0);				
				// }
			}
			objsvec[nobj].clear();
		}
		objsvec.clear();
		MOEA_Write(objs, NULL);
	}
	end = clock();
	if (dice.config.writefile==1){
		outputFile.close();
	}
	if (dice.config.scc==1){
		std::ofstream SCCfile;
		SCCfile.open("./SCC.txt");
		srand(seed);
		// dice.computeSCC(SCCfile, 1000);
		for (int nidx=0; nidx < dice.config.niter ; nidx++){
			dice.fair.sampleUnc();
			dice.econ.sampleUnc();
			dice.computeSCCDamages(SCCfile, nidx);
		}
		SCCfile.close();
	}
	std::cout << "time elapsed: " << ((end - start)/double(CLOCKS_PER_SEC)) << " seconds" << std::endl;

	return 0;
}