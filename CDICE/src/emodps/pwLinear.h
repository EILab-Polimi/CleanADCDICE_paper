/*
 * pwLinear.h
 *
 *  Created on: 5/nov/2013
 *      Author: MatteoG
 */


#ifndef pwLinear_H
#define pwLinear_H

#include <math.h>
#include <vector>
#include <string>
#include <iostream>
#include "param_function.h"

namespace std{

struct pwLinearParams{
    //      a + tan(b1)*(x-c1)      if 0<x<=c1
    // u =  a                       if c1<x<=c2
    //      a + tan(b2)*(x-c2)      if c2<x
    double a;
    vector<double> b;
    vector<double> c;
} ;

class pwLinear : public param_function
{
public:
    pwLinear();
    virtual ~pwLinear();

    /**
      * constructor with parameters:
      *     pM = number of input
      *     pK = number of output
      *     pN = number of linear segments
      **/
    pwLinear(unsigned int pM, unsigned int pK, unsigned int pN);

    void setPolicySetting( vector<int> pN );

    /**
      * pwLinear function (input and output are normalized/standardized)
      **/
    vector<double> get_output(vector<double> input);

    /**
     * pwLinearsetParameters(double* pTheta)
     *      pTheta = array of parameters (c,b,w)
     */
    void setParameters(double* pTheta);

    /**
      * Clear policy parameters
      */
    void clearParameters() ;

protected:
    unsigned int N;     // number of linear segments (3)
    unsigned int M;     // number of input
    unsigned int K;     // number of output


    pwLinearParams param;

};
}

#endif // pwLinear_H
