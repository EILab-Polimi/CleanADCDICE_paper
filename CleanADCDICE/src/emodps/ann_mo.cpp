/*
 * ann.cpp
 *
 *  Created on: 07/feb/2013
 *      Author: EmanueleM
 */

#include "ann_mo.h"

using namespace std;

annmo::annmo()
{
}

annmo::~annmo(){
}


annmo::annmo(unsigned int pM, unsigned int pK, unsigned int pN){
    M = pM;
    K = pK;
    N = pN;

    param.a.resize(K);
    param.b.resize(N);
    for (unsigned int n=0; n<N; n++){
        param.b[n].resize(K);
    }
    param.c.resize(M);
    for(unsigned int m=0; m<M; m++) {
        param.c[m].resize(N);
    }
    param.d.resize(N);
}

void annmo::setParameters(double* pTheta){
    unsigned int idx_theta = 0;
  
        
    for(unsigned int n = 0; n < N; n++) {
        param.d[n] = pTheta[idx_theta];
        idx_theta++;
    }

    for(unsigned int n = 0; n < N; n++) {
        for(unsigned int m = 0; m < M; m++) {
            param.c[m][n] = pTheta[idx_theta];
            idx_theta++;
        }
    }

    for (unsigned int k = 0; k < K; k++) {
        param.a[k] = pTheta[idx_theta];
        idx_theta++;
    }
    
    for (unsigned int k = 0; k < K; k++) {
        for(unsigned int n = 0; n < N; n++) {
            param.b[n][k] = pTheta[idx_theta];
            idx_theta++;
        }
    }

}

void annmo::clearParameters(){

}


vector<double> annmo::get_output(vector<double> input){

    // ANN
    vector<double> neurons;
    double value, o;
    // output
    vector<double> y;
    
    for(unsigned int n = 0; n < N; n++){
        value = param.d[n];
        for(unsigned int m = 0; m < M; m++){
            value = value + (param.c[m][n] * input[m]);
        }      
        value = 2 / ( 1 + exp(-2 * value) ) - 1;
        neurons.push_back( value );
    }
    for(unsigned int k = 0; k < K; k++){
        o = param.a[k];
        for(unsigned int n = 0; n < N; n++){
            o = o + param.b[n][k] * neurons[n] ;
        }
        y.push_back(o);   
    }
    neurons.clear();

    return y;
}
