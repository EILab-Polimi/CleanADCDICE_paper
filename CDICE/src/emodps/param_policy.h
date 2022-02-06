/*
 * param_policy.h
 *
 *  Created on: 31/oct/2013
 *      Author: MatteoG
 */

#ifndef PARAMPOLICY_H
#define PARAMPOLICY_H

#include <math.h>
#include <vector>
#include <string>
#include <iostream>
#include "utils.h"

namespace std{

struct policy_param{
    int tPolicy;            // class of policy { 1=RBF; 2=ANN; 3=pwLinear }
    int policyInput;        // number of policy input
    int policyOutput;        // number of policy output
    int policyStr;        // policy architecture (e.g., number of RBF, number of linear segments)
    vector<double> mIn,mOut,MIn,MOut; // min-max policy input
};

class param_policy
{
public:
    param_policy();
    virtual ~param_policy();


    /**
      * constructor with parameters:
      *     pM = number of input
      *     pK = number of output
      *     pN = policy architecture (e.g., number of RBF, number of linear segments)
      **/
    //param_policy(unsigned int pM, unsigned int pK, unsigned int pN);

    /**
     * Get policy input/output
     */
    unsigned int getInputNumber();
    unsigned int getOutputNumber();

    /**
      * Set policy parameters
      **/
    virtual void setParameters(double* pTheta) = 0;

    /**
      * Clear policy parameters
      */
    virtual void clearParameters() = 0;

    /**
      * generic parametrized policy (input and output are normalized)
      **/
    virtual vector<double> get_decision(vector<double> input) = 0;

    /**
     * Set/Get min/max policy input/output for normalization
     */
    void setMaxInput(vector<double> pV);
    void setMaxOutput(vector<double> pV);
    void setMinInput(vector<double> pV);
    void setMinOutput(vector<double> pV);

    vector<double> getMaxInput();
    vector<double> getMaxOutput();
    vector<double> getMinInput();
    vector<double> getMinOutput();


protected:
    unsigned int M; // number of input
    unsigned int K; // number of output
    unsigned int N; // policy architecture

    // policy input/output normalization
    vector<double> input_max;
    vector<double> output_max;
    vector<double> input_min;
    vector<double> output_min;

};
}

#endif
