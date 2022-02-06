/*
 * pwLinear.cpp
 *
 *  Created on: 31/oct/2013
 *      Author: MatteoG
 */

#include "pwLinear.h"
#include <math.h>       /* tan */

#define PI 3.14159265

using namespace std;


pwLinear::pwLinear()
{
}

pwLinear::~pwLinear(){
}


pwLinear::pwLinear(unsigned int pM, unsigned int pK, unsigned int pN){
    M = pM;
    K = pK;
    N = pN;
}


void pwLinear::setParameters(double* pTheta){

    // piecewise linear parameters (b1,c1,b2,c2,a)
    unsigned int count = 0;
    for(unsigned int i=0; i<N-1; i++){
        param.b.push_back( pTheta[count] );
        count = count+1;
        param.c.push_back( pTheta[count] );
        count = count+1;
    }
    param.a =  pTheta[count] ;
    
}


void pwLinear::clearParameters(){
    param.a = 0.0;
    param.b.clear();
    param.c.clear();
}

vector<double> pwLinear::get_output(vector<double> input){

    double y0 = param.a + tan( param.b[0]*PI/180.0 )*(input[0]-param.c[0]) ;
    double y1 = param.a + tan( param.b[1]*PI/180.0 )*(input[0]-param.c[1]) ;
    double y = param.a;
    if( y0 < param.a ){
        y = y0 ;
    }
    if( y < y1 ){
        y = y1 ;
    }
    
    vector<double> yy;
    yy.push_back( y );

    return yy;
}
