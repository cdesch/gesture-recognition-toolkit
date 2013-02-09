/*
 GRT MIT License
 Copyright (c) <2012> <Nicholas Gillian, Media Lab, MIT>
 
 Permission is hereby granted, free of charge, to any person obtaining a copy of this software 
 and associated documentation files (the "Software"), to deal in the Software without restriction, 
 including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
 and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, 
 subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in all copies or substantial 
 portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT 
 LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
 IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
 WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
 SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 
 This code is based on the GMM code from Numerical Recipes (3rd Edition)
 
 */
#pragma once

#include "../../Util/GRTCommon.h"
#include "../../DataStructures/Matrix.h"
#include "../../Util/LUdcmp.h"
#include "../../Util/Cholesky.h"
#include "../../DataStructures/UnlabelledClassificationData.h"

namespace GRT {

class GaussianMixtureModels
{
public:
	GaussianMixtureModels(void);
	~GaussianMixtureModels(void);

	bool train(UnlabelledClassificationData &trainingData,UINT K);
    bool getModelTrained(){ return modelTrained; }
    
    UINT getK(){ return K; }
    Matrix< double > getMu(){ if( modelTrained ){ return mu; } return Matrix<double>(); }
    vector< Matrix< double > > getSigma(){ if( modelTrained ){ return sigma; } return vector< Matrix<double> >(); }
    Matrix< double > getSigma(UINT k){
        if( k < K && modelTrained ){
            return sigma[k];
        }
        return Matrix<double>();
    }
    
    bool setMinChange(double minChange){
        if( minChange > 0 ){
            this->minChange = minChange;
            return true;
        }
        return false;
    }
    bool setMaxIter(UINT maxIter){
        if( maxIter > 0 ){
            this->maxIter = maxIter;
            return true;
        }
        return false;
    }
	
private:
    double estep();
	void mstep();
	bool computeInvAndDet();
	inline void SWAP(UINT &a,UINT &b);
	inline double SQR(double v){ return v*v; }
    
	UINT N;                                     //The number of dimensions in the training data
	UINT M;                                     //The number of samples in the training data
	UINT K;                                     //The number of Gaussian Models we want to fit to the data
	UINT maxIter;                               //The maximum number of iterations allowed during the training routine
	double loglike;                             //The current loglikelihood value of the models given the data
	double minChange;                           //The minimum change value that signals if the training routine has converged
	Matrix<double> data;                        //A matrix holding the training data
	Matrix<double> mu;                          //A matrix holding the estimated mean values of each Gaussian
	Matrix<double> resp;                        //The responsibility matrix
	vector<double> frac;                        //A vector holding the P(k)'s 
	vector<double> lndets;                      //A vector holding the log detminants of SIGMA'k
	vector<double> det;                         
	vector< Matrix<double > > sigma;
	vector< Matrix<double> > invSigma;
	bool modelTrained;
    bool failed;
    
    DebugLog debugLog;
    ErrorLog errorLog;
    WarningLog warningLog;
	
};
    
}//End of namespace GRT
