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

#include "KMeans.h"

namespace GRT{

//Constructor,destructor
KMeans::KMeans(){
    M = N = K = nchg = 0;
    minNumEpochs = 10;
    maxNumEpochs = 1000;
    minChange = 1.0e-5;
    finalTheta = 0;
    numTrainingIterations = 0;
    computeTheta = true;
    trained = false;
}

KMeans::~KMeans(){

}

bool KMeans::train(UINT K, LabelledClassificationData &trainingData){

    //Convert the labelled training data to a training matrix
    UINT N = trainingData.getNumDimensions();
    UINT M = trainingData.getNumSamples();

    Matrix<double> data(M,N);

    for(UINT i=0; i<M; i++){
        for(UINT j=0; j<N; j++){
            data[i][j] = trainingData[i][j];
        }
    }

    return train(K, data);
}

bool KMeans::train(UINT K, Matrix<double> data){
    this->K = K;
	M = data.getNumRows();
	N = data.getNumCols();

	clusters.resize(K,N);
	assign.resize(M);
	count.resize(K);

	//Randomly pick k data points as the starting clusters
	Random random;
	vector< UINT > randPoints(K);
	for(UINT k=0; k<K; k++){
        randPoints[k] = random.getRandomNumberInt(0,M);
	}

	for(UINT k=0; k<K; k++){
		for(UINT j=0; j<N; j++){
            clusters[k][j] = data[ randPoints[k] ][j];
		}
	}

	return train( data );
}
bool KMeans::train(Matrix<double> data,Matrix<double> clusters){
	this->clusters = clusters;
	M = data.getNumRows();
	N = data.getNumCols();
	K = clusters.getNumRows();
	assign.resize( M );
	count.resize( K );

	return train( data );
}

bool KMeans::train(Matrix< double > &data){

	UINT currentIter = 0;
    UINT numChanged = 0;
	bool keepTraining = true;
    bool converged = false;
    double theta = 0;
    thetaTracker.clear();
    finalTheta = 0;
    numTrainingIterations = 0;
    trained = false;

    //Init the assign and count vectors
    //Assign is set to K+1 so that the nChanged values in the eStep at the first iteration will be updated correctly
    for(UINT m=0; m<M; m++) assign[m] = K+1;
	for(UINT k=0; k<K; k++) count[k] = 0;

    //Run the training loop
	while( keepTraining ){

		//Compute the E step
		numChanged = estep(data);

        //Compute the M step
        mstep(data);

        //Update the iteration counter
        currentIter++;

		//Check convergance
		if( computeTheta ) theta = calculateTheta(data);
		if( numChanged == 0 && currentIter > minNumEpochs ){ converged = true; keepTraining = false; }
		if( currentIter >= maxNumEpochs ){ keepTraining = false; }
		if( fabs( finalTheta - theta ) < minChange && computeTheta && currentIter > minNumEpochs ){ converged = true; keepTraining = false; }
        if( computeTheta )  thetaTracker.push_back( theta );
	}

    finalTheta = theta;
    numTrainingIterations = currentIter;
    trained = true;
	return converged;
}

UINT KMeans::estep(Matrix< double > &data) {
		UINT k,m,n,kmin;
		double dmin,d;
		nchg = 0;
		kmin = 0;

		//Reset Count
		for (k=0;k<K;k++) count[k] = 0;

		//Look for the closest center and reasign if needed
		for (m=0; m<M; m++) {
			dmin = 9.99e+99;
			for (k=0; k<K; k++) {
				d = 0.0;
				for (n=0; n<N; n++)
					d += SQR( data[m][n]-clusters[k][n] );
				if (d <= dmin){ dmin = d; kmin = k; }
			}
			if ( kmin != assign[m] ){
                nchg++;
                assign[m] = kmin;
            }
			count[kmin]++;
		}
		return nchg;
}

void KMeans::mstep(Matrix< double > &data) {
    UINT n,k,m;

    //Reset means to zero
    for (k=0;k<K;k++)
        for (n=0;n<N;n++)
            clusters[k][n] = 0.;

    //Get new mean by adding assigned data points and dividing by the number of values in each cluster
    for(m=0; m<M; m++)
        for(n=0; n<N; n++)
            clusters[ assign[m] ][n] += data[m][n];

    for (k=0; k<K; k++) {
        if (count[k] > 0){
            for (n=0; n<N; n++){
                clusters[k][n] /= double(count[k]);
            }
        }
    }
}

double KMeans::calculateTheta(Matrix< double > &data){

	double theta = 0;

	for(UINT m=0; m<M; m++){
		UINT k = assign[m];
		double sum = 0;
		for(UINT n=0;n<N;n++){
				sum += SQR(clusters[k][n] - data[m][n]);
		}
		theta += sum;
	}

	return theta;

}


bool KMeans::saveKMeansModelToFile(string fileName){

    if( !trained ){
        return false;
    }

   std::fstream file;
   file.open(fileName.c_str(), std::ios::out);

   if( !file.is_open() ){
       return false;
   }

   file << "GRT_KMEANS_MODEL_FILE_V1.0\n";
   file << "K: " << K << endl;
   file << "N: " << N <<endl;
   file << "Clusters:\n";

   for(UINT k=0; k<K; k++){
       for(UINT n=0; n<N; n++){
            file << clusters[k][n] << "\t";
       }file << endl;
   }

   file.close();

   return true;

}


bool KMeans::loadKMeansModelFromFile(string fileName){

	clusters.clear();
	K = 0;
	N = 0;
	M = 0;

   std::fstream file;
   string word;
   file.open(fileName.c_str(), std::ios::in);

   if(!file.is_open()){
	   return false;
   }

   file >> word;
   if( word != "GRT_KMEANS_MODEL_FILE_V1.0" ){
	   file.close();
	   return false;
   }

   file >> word;
   if( word != "K:" ){
	   file.close();
	   return false;
   }
   file >> K;

   file >> word;
   if( word != "N:" ){
	   file.close();
	   return false;
   }
   file >> N;

   file >> word;
   if( word != "Clusters:" ){
	   file.close();
	   return false;
   }

   //Resize the buffers
   clusters.resize(K,N);

   //Load the data
   for(UINT k=0; k<K; k++){
	   for(UINT n=0; n<N; n++){
	      file >> clusters[k][n];
	   }
   }

   file.close();

    trained = true;

   return true;
}

bool KMeans::setComputeTheta(bool computeTheta){
    this->computeTheta = computeTheta;
    return true;
}

bool KMeans::setMinChange(double minChange){
    if( minChange > 0 ){
        this->minChange = minChange;
        return true;
    }
    return false;
}

bool KMeans::setMinNumEpochs(UINT minNumEpochs){
    if( minNumEpochs > 0 ){
        this->minNumEpochs = minNumEpochs;
        return true;
    }
    return false;
}

bool KMeans::setMaxNumEpochs(UINT maxNumEpochs){
    if( maxNumEpochs > 0 ){
        this->maxNumEpochs = maxNumEpochs;
        return true;
    }
    return false;
}

}//End of namespace GRT
