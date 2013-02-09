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
 */

#pragma once

#include "../../DataStructures/Matrix.h"
#include "../../DataStructures/LabelledClassificationData.h"

namespace GRT{

class KMeans{

public:
	//Constructor,destructor
	KMeans();
    ~KMeans();
    
    bool train(UINT K, LabelledClassificationData &trainingData);
	bool train(UINT K, Matrix<double> data);
	bool train(Matrix<double> data,Matrix<double> clusters);
	bool saveKMeansModelToFile(string fileName);
	bool loadKMeansModelFromFile(string fileName);

    //Getters
	UINT getNumClusters(){ return K; }
	UINT getNumDimensions(){ return N; }
    UINT getNumTrainingIterations(){ return numTrainingIterations; }
	double getTheta(){ return finalTheta; }
    inline double SQR(const double a) {return a*a;};
    vector< double > getTrainingThetaLog(){ return thetaTracker; }
    Matrix< double > getClusters(){ return clusters; }
    vector< UINT > getClassLabelsVector(){ return assign; }
    vector< UINT > getClassCountVector(){ return count; }
    bool isModelTrained(){ return trained; }
    
    //Setters
    bool setComputeTheta(bool computeTheta);
    bool setMinChange(double minChange);
    bool setMinNumEpochs(UINT minNumEpochs);
    bool setMaxNumEpochs(UINT maxNumEpochs);

private:
	bool train(Matrix< double > &data);
    UINT estep(Matrix< double > &data);
	void mstep(Matrix< double > &data);
	double calculateTheta(Matrix< double > &data);

	UINT M;                             //Number of training examples
	UINT N;                             //Number of dimensions
	UINT K;                             //Number of clusters
	UINT nchg;                          //Number of values changes
    UINT minNumEpochs;      
    UINT maxNumEpochs;
    UINT numTrainingIterations;
	double finalTheta;
    double minChange;
    Matrix<double> clusters;
	vector< UINT > assign, count;
    vector< double > thetaTracker;
	bool computeTheta;
    bool trained;
		
};
    
}//End of namespace GRT
