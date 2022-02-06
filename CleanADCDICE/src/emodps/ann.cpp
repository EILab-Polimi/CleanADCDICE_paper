/*
 * ann.cpp
 *
 *  Created on: 07/feb/2013
 *      Author: EmanueleM
 */

#include "ann.h"

using namespace std;

ann::ann()
{
}

ann::~ann(){
}


ann::ann(unsigned int pM, unsigned int pK, unsigned int pN){
    M = pM;
    K = pK;
    N = pN;

    param.resize(K);
    for(unsigned int k=0; k<K; k++) {
         param[k].d.resize(N);
         param[k].b.resize(N);
         param[k].c.resize(M);
         for(unsigned int m=0; m<M; m++) {
             param[k].c[m].resize(N);
         }
    }
}

void ann::setParameters(double* pTheta){
    unsigned int idx_theta = 0;
  
    for (unsigned int k = 0; k < K; k++) {
        
        for(unsigned int n = 0; n < N; n++) {
            param[k].d[n] = pTheta[idx_theta];
            idx_theta++;
        }
        
        param[k].a = pTheta[idx_theta];
        idx_theta++;
        
        for(unsigned int n = 0; n < N; n++) {
            param[k].b[n] = pTheta[idx_theta];
            idx_theta++;
        }
        
        for(unsigned int m = 0; m < M; m++) {
            for(unsigned int n = 0; n < N; n++) {
                param[k].c[m][n] = pTheta[idx_theta];
                idx_theta++;
            }
        }
        
    }

}

void ann::clearParameters(){

}


vector<double> ann::get_output(vector<double> input){

    // ANN
    vector<double> neurons;
    double value, o;
    // output
    vector<double> y;
    
    for(unsigned int k = 0; k < K; k++){
        
        for(unsigned int n = 0; n < N; n++){
            
            value = param[k].d[n];
            for(unsigned int m = 0; m < M; m++){
                value = value + (param[k].c[m][n] * input[m]);
            }
            
            value = 2 / ( 1 + exp(-2 * value) ) - 1;
            neurons.push_back( value );
        }
        o = param[k].a;
        for(unsigned int n = 0; n < N; n++){
            o = o + param[k].b[n] * neurons[n] ;
        }
        y.push_back(o);
        
        neurons.clear();
    }

    return y;
}
