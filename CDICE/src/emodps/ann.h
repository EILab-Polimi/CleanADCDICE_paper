/*
 * ann.h
 *
 *  Created on: 7/02/2014
 *      Author: EmanueleM
 */


#ifndef ANN_H
#define ANN_H

#include <math.h>
#include <vector>
#include <string>
#include <iostream>
#include "param_function.h"

namespace std{

struct ANNparams {
    double a;  // scalar
    vector<double> b;  // vector[number_neurons]
    //for each input, vector[number_neurons]
    vector<vector<double> > c;
    vector<double> d;  // vector[number_neurons]
};

/**
 * Generic MIMO single layer ANN function with its parameters.
 * a + sum b * tansig(c*input + d);
 */
class ann : public param_function
{
public:
    ann();
    virtual ~ann();

    /**
      * constructor with parameters:
      *     pM = number of input
      *     pK = number of output
      *     pN = number of neurons
      **/
    ann(unsigned int pM, unsigned int pK, unsigned int pN);

    /**
      * Clear policy parameters
      */
    void clearParameters();

    /**
      * ANN function (input and output are normalized/standardized)
      **/
    vector<double> get_output(vector<double> input);

    /**
     * ANNsetParameters(double* pTheta)
     *      pTheta = array of parameters (c,b,w)
     */
    void setParameters(double* pTheta);

protected:
    unsigned int N; // number of neurons
    unsigned int M; // number of input
    unsigned int K; // number of output

    vector<ANNparams> param;

};
}

#endif // ANN_H
