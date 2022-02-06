/*
 * param_policy.cpp
 *
 *  Created on: 31/oct/2013
 *      Author: MatteoG
 */


#include "param_policy.h"

using namespace std;

param_policy::param_policy()
{
}

param_policy::~param_policy() {
}
/*
param_policy::param_policy(unsigned int pM, unsigned int pK, unsigned int pN){
    M = pM;
    K = pK;
    N = pN;
}
*/
unsigned int param_policy::getInputNumber(){
    return M;
}

unsigned int param_policy::getOutputNumber(){
    return K;
}

void param_policy::setMaxInput(vector<double> pV){

    for(unsigned int i=0; i<pV.size(); i++){
        input_max.push_back( pV[i] );
    }
}

void param_policy::setMaxOutput(vector<double> pV){

    for(unsigned int i=0; i<pV.size(); i++){
        output_max.push_back( pV[i] );
    }
}

void param_policy::setMinInput(vector<double> pV){

    for(unsigned int i=0; i<pV.size(); i++){
        input_min.push_back( pV[i] );
    }
}

void param_policy::setMinOutput(vector<double> pV){

    for(unsigned int i=0; i<pV.size(); i++){
        output_min.push_back( pV[i] );
    }
}

vector<double> param_policy::getMaxInput(){
    return input_max;
}

vector<double> param_policy::getMaxOutput(){
    return output_max;
}

vector<double> param_policy::getMinInput(){
    return input_min;
}

vector<double> param_policy::getMinOutput(){
    return output_min;
}





