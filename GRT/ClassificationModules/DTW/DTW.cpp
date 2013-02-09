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

#include "DTW.h"

namespace GRT{
    
//Register the DTW module with the Classifier base class
RegisterClassifierModule< DTW > DTW::registerModule("DTW");

DTW::DTW(bool useScaling,bool useNullRejection,double nullRejectionCoeff,UINT rejectionMode,bool useSmoothing,UINT smoothingFactor)
{
    this->useScaling=useScaling;
    this->useNullRejection = useNullRejection;
    this->nullRejectionCoeff = nullRejectionCoeff;
    this->rejectionMode = rejectionMode;
    this->useSmoothing = useSmoothing;
    this->smoothingFactor = smoothingFactor;

	trained=false;
	useZNormalisation=false;
	constrainZNorm=false;
	dtwConstrain=true;
    trimTrainingData = false;

	crossValidationAccuracy=0.0;
	zNormConstrainThreshold=0.2;

	numTemplates=0;
	distanceMethod=EUCLIDEAN_DIST;

	averageTemplateLength =0;

    classifierType = "DTW";
    classifierMode = TIMESERIES_CLASSIFIER_MODE;
    debugLog.setProceedingText("[DEBUG NDDTW]");
    errorLog.setProceedingText("[ERROR NDDTW]");
    trainingLog.setProceedingText("[TRAINING NDDTW]");
    warningLog.setProceedingText("[WARNING NDDTW]");
}

DTW::~DTW(void)
{
}

////////////////////////// TRAINING FUNCTIONS //////////////////////////
bool DTW::train(LabelledTimeSeriesClassificationData &trainingData){
    _train(trainingData);
    return trained;
}

bool DTW::_train(LabelledTimeSeriesClassificationData &labelledTrainingData){

	UINT bestIndex = 0;
	UINT worstIndex = 0;

	//Cleanup Memory
	templatesBuffer.clear();
    classLabels.clear();
	trained = false;
    continuousInputDataBuffer.clear();

    if( labelledTrainingData.getNumSamples() == 0 ){
        errorLog << "_train(LabelledTimeSeriesClassificationData &labelledTrainingData) - Can't train model as there are no samples in training data!" << endl;
        return false;
    }

    if( trimTrainingData ){
        LabelledTimeSeriesClassificationSampleTrimmer timeSeriesTrimmer(0.1,50);
        LabelledTimeSeriesClassificationSample sample;
        for(UINT i=0; i<labelledTrainingData.getNumSamples(); i++){
            timeSeriesTrimmer.trimTimeSeries( labelledTrainingData[i] );
        }
    }

	//Assign
    numClasses = labelledTrainingData.getNumClasses();
	numTemplates = labelledTrainingData.getNumClasses();
    numFeatures = labelledTrainingData.getNumDimensions();
	templatesBuffer.resize(numTemplates);
    classLabels.resize(numClasses);
	averageTemplateLength = 0;

	//Need to copy the labelled training data incase we need to scale it or znorm it
	LabelledTimeSeriesClassificationData trainingData( labelledTrainingData );

	//Perform any scaling or normalisation
    rangesBuffer = trainingData.getRanges();
	if( useScaling ) scaleData( trainingData );
	if( useZNormalisation ) znormData( trainingData );

	//For each class, run a one-to-one DTW and find the template the best describes the data
	for(UINT k=0; k<numTemplates; k++){
        //Get the class label for the cth class
        UINT classLabel = trainingData.getClassTracker()[k].classLabel;
        LabelledTimeSeriesClassificationData classData = trainingData.getClassData( classLabel );
		UINT numExamples = classData.getNumSamples();
		bestIndex = 0;
	    worstIndex = 0;

        //Set the class label of this template
        templatesBuffer[k].classLabel = classLabel;

        //Set the kth class label
        classLabels[k] = classLabel;

		//Check to make sure we actually have some training examples
		if(numExamples<1){
            errorLog << "_train(LabelledTimeSeriesClassificationData &labelledTrainingData) - Can not train model: Num of Example is < 1! Class: " << classLabel << endl;
			return false;
		}

		if(numExamples==1){//If we have just one training example then we have to use it as the template
            bestIndex = 0;
            worstIndex = 0;

            templatesBuffer[k].threshold = 0.0;//TODO-We might need a better way of calculating this!
            warningLog << "_train(LabelledTimeSeriesClassificationData &labelledTrainingData) - Can't compute reject thresholds for class " << classLabel << " as there is only 1 training example" << endl;
		}else{//Search for the best training example for this class
			if( !_train_NDDTW(classData,templatesBuffer[k],bestIndex) ){
                errorLog << "_train(LabelledTimeSeriesClassificationData &labelledTrainingData) - Failed to train template for class with label: " << classLabel << endl;
                return false;
            }
		}

		//Add the template with the best index to the buffer
		int trainingMethod = 0;
		if(useSmoothing) trainingMethod = 1;

		switch (trainingMethod) {
			case(0)://Standard Training
				templatesBuffer[k].timeSeries = classData[bestIndex].getData();
				break;
			case(1)://Training using Smoothing
				//Smooth the data, reducing its size by a factor set by smoothFactor
				smoothData(classData[ bestIndex ].getData(),smoothingFactor,templatesBuffer[k].timeSeries);
				break;
			default:
				cout<<"Can not train model: Unknown training method \n";
				return false;
				break;
		}

		//Add the average length of the training examples for this template to the overall averageTemplateLength
		averageTemplateLength += templatesBuffer[k].averageTemplateLength;

	}

    //Flag that the models have been trained
	trained = true;
	averageTemplateLength = (UINT) averageTemplateLength/numTemplates;

    //Recompute the null rejection thresholds
    recomputeNullRejectionThresholds();

    //Resize the prediction results to make sure it is setup for realtime prediction
    continuousInputDataBuffer.clear();
    continuousInputDataBuffer.resize(averageTemplateLength,vector<double>(numFeatures,0));
    classLikelihoods.resize(numTemplates,DEFAULT_NULL_LIKELIHOOD_VALUE);
    classDistances.resize(numTemplates,0);
    predictedClassLabel = 0;
    maxLikelihood = DEFAULT_NULL_LIKELIHOOD_VALUE;

	//Training complete
	return trained;
}

bool DTW::_train_NDDTW(LabelledTimeSeriesClassificationData &trainingData,DTWTemplate &dtwTemplate,UINT &bestIndex){

   UINT numExamples = trainingData.getNumSamples();
   vector<double> results(numExamples,0.0);
   Matrix<double> distanceResults(numExamples,numExamples);
   dtwTemplate.averageTemplateLength = 0;
   for(UINT m=0; m<numExamples; m++){
	   Matrix<double> templateA; //The m'th template
	   Matrix<double> templateB; //The n'th template
	   dtwTemplate.averageTemplateLength += trainingData[m].getLength();

	   //Smooth the data if required
	   if( useSmoothing ) smoothData(trainingData[m].getData(),smoothingFactor,templateA);
	   else templateA = trainingData[m].getData();

	   for(UINT n=0; n<numExamples; n++){
		if(m!=n){
		    //Smooth the data if required
		    if( useSmoothing ) smoothData(trainingData[n].getData(),smoothingFactor,templateB);
		    else templateB = trainingData[n].getData();

			//Compute the distance between the two time series
			double dist = 0.0;
			dist = computeDistance(templateA,templateB);

			//Update the results values
			distanceResults[m][n] = dist;
			results[m] += dist;
		}
	   }
   }

	for(UINT m=0; m<numExamples; m++) results[m]/=(numExamples-1);
	//Find the best average result
	bestIndex = 0;
	double bestAverage = results[0];
	for(UINT m=1; m<numExamples; m++){
		if(results[m]<bestAverage){
			bestAverage = results[m];
			bestIndex = m;
		}
	}

    if( numExamples > 2 ){

        //Work out the threshold value for the best template
        dtwTemplate.trainingMu = results[bestIndex];
        dtwTemplate.trainingSigma = 0.0;

        for(UINT n=0; n<numExamples; n++){
            if(n!=bestIndex){
                dtwTemplate.trainingSigma += SQR(distanceResults[bestIndex][n]-results[bestIndex]);
            }
        }

        dtwTemplate.trainingSigma = sqrt( dtwTemplate.trainingSigma / (numExamples-2) );

        //The threshold is set as the mean distance plus gamma standard deviations
        dtwTemplate.threshold = dtwTemplate.trainingMu + (dtwTemplate.trainingSigma * nullRejectionCoeff);

    }else{
        warningLog << "_train_NDDTW(LabelledTimeSeriesClassificationData &trainingData,DTWTemplate &dtwTemplate,UINT &bestIndex - There are not enough examples to compute the trainingMu and trainingSigma for the template for class " << dtwTemplate.classLabel << endl;
        dtwTemplate.trainingMu = 0.0;
        dtwTemplate.trainingSigma = 0.0;
    }

	//Set the average length of the training examples
	dtwTemplate.averageTemplateLength = (UINT) (dtwTemplate.averageTemplateLength/double(numExamples));

    //Flag that the training was successfull
	return true;
}


bool DTW::predict(Matrix<double> &inputTimeSeries){

    if( !trained ){
        errorLog << "predict(Matrix<double> &inputTimeSeries) - The DTW templates have not been trained!" << endl;
        return false;
    }
    
    bool debug = false;

    if( classLikelihoods.size() != numTemplates ) classLikelihoods.resize(numTemplates);
    if( classDistances.size() != numTemplates ) classDistances.resize(numTemplates);

    predictedClassLabel = 0;
    maxLikelihood = DEFAULT_NULL_LIKELIHOOD_VALUE;
    for(UINT k=0; k<classLikelihoods.size(); k++){
        classLikelihoods[k] = 0;
        classDistances[k] = DEFAULT_NULL_LIKELIHOOD_VALUE;
    }

	if( numFeatures != inputTimeSeries.getNumCols() ){
        errorLog << "predict(Matrix<double> &inputTimeSeries) - The number of features in the model (" << numFeatures << ") do not match that of the input time series (" << inputTimeSeries.getNumCols() << ")" << endl;
        return false;
    }

	//Perform any preprocessing if requried
    Matrix< double > *timeSeriesPtr = &inputTimeSeries;
    Matrix< double > processedTimeSeries;
    Matrix<double> tempMatrix;
	if(useScaling){
        scaleData(*timeSeriesPtr,processedTimeSeries);
        timeSeriesPtr = &processedTimeSeries;
    }
	if(useZNormalisation){
        znormData(*timeSeriesPtr,processedTimeSeries);
        timeSeriesPtr = &processedTimeSeries;
    }

	//Smooth the data if required
	if(useSmoothing){
		smoothData(*timeSeriesPtr,smoothingFactor,tempMatrix);
		timeSeriesPtr = &tempMatrix;
	}

	//Make the prediction by finding the closest template
    double sum = 0;
	//Test the timeSeries against all the templates in the timeSeries buffer
	for(UINT k=0; k<numTemplates; k++){
		//Perform DTW
		classDistances[k] = computeDistance(templatesBuffer[k].timeSeries,*timeSeriesPtr);

        classLikelihoods[k] = exp( 1.0 - (classDistances[k]) );

        sum += classLikelihoods[k];
	}

    if( debug ){
        cout << "SUM: " << sum << endl;
        cout << "Distances: ";
        for(UINT k=0; k<numTemplates; k++){
            cout << classDistances[k] << "\t";
        }cout << endl;
        cout << "Thresholds: ";
        for(UINT k=0; k<numTemplates; k++){
            cout << templatesBuffer[k].threshold << "\t";
        }cout << endl;
        cout << "Likelihoods: ";
        for(UINT k=0; k<numTemplates; k++){
            cout << classLikelihoods[k] << "\t";
        }cout << endl;
    }

    //If all the class distances are very far away then the sum could be zero, so catch this
    bool sumIsZero = false;
    if( sum == 0 ) sumIsZero = true;

	//See which gave the min distance
	UINT closestTemplateIndex = 0;
	bestDistance = classDistances[0];
	for(UINT k=1; k<numTemplates; k++){
		if( classDistances[k] < bestDistance ){
			bestDistance = classDistances[k];
			closestTemplateIndex = k;
		}
	}

    //Normalize the class likelihoods and check which class has the maximum likelihood
    maxLikelihood = 0;
    UINT maxLikelihoodIndex = 0;
    if( !sumIsZero ){
        classLikelihoods[0] /= sum;
        maxLikelihood = classLikelihoods[0];
        for(UINT k=1; k<numTemplates; k++){
            classLikelihoods[k] /= sum;
            if( classLikelihoods[k] > maxLikelihood ){
                maxLikelihood = classLikelihoods[k];
                maxLikelihoodIndex = k;
            }
        }
    }

    if( debug ){
        cout << "NormedLikelihoods: ";
        for(UINT k=0; k<numTemplates; k++){
            cout << classLikelihoods[k] << "\t";
        }cout << endl;
    }

    if( useNullRejection ){

        switch( rejectionMode ){
            case TEMPLATE_THRESHOLDS:
                if( bestDistance <= templatesBuffer[ closestTemplateIndex ].threshold ) predictedClassLabel = templatesBuffer[ closestTemplateIndex ].classLabel;
                else predictedClassLabel = 0;
                break;
            case CLASS_LIKELIHOODS:
                if( maxLikelihood >= 0.99 )  predictedClassLabel = templatesBuffer[ maxLikelihoodIndex ].classLabel;
                else predictedClassLabel = 0;
                break;
            case THRESHOLDS_AND_LIKELIHOODS:
                if( bestDistance <= templatesBuffer[ closestTemplateIndex ].threshold && maxLikelihood >= 0.99 )
                    predictedClassLabel = templatesBuffer[ closestTemplateIndex ].classLabel;
                else predictedClassLabel = 0;
                break;
            default:
                errorLog << "predict(Matrix<double> &timeSeries) - Unknown RejectionMode!" << endl;
                return false;
                break;
        }

	}else predictedClassLabel = templatesBuffer[ closestTemplateIndex ].classLabel;

    return true;
}

bool DTW::predict(vector<double> inputVector){

    if( !trained ){
        errorLog << "predict(vector<double> inputVector) - The model has not been trained!" << endl;
        return false;
    }
    predictedClassLabel = 0;
    maxLikelihood = DEFAULT_NULL_LIKELIHOOD_VALUE;
    for(UINT c=0; c<classLikelihoods.size(); c++){
        classLikelihoods[c] = DEFAULT_NULL_LIKELIHOOD_VALUE;
    }

	if( numFeatures != inputVector.size() ){
        errorLog << "predict(vector<double> inputVector) - The number of features in the model " << numFeatures << " does not match that of the input vector " << inputVector.size() << endl;
        return false;
    }

    //Add the new input to the circular buffer
    continuousInputDataBuffer.push_back( inputVector );

    if( continuousInputDataBuffer.getNumValuesInBuffer() < averageTemplateLength ){
        //We haven't got enough samples yet so can't do the prediction
        return true;
    }

    //Copy the data into a temporary matrix
    Matrix< double > predictionTimeSeries(averageTemplateLength,numFeatures);
    for(UINT i=0; i<predictionTimeSeries.getNumRows(); i++){
        for(UINT j=0; j<predictionTimeSeries.getNumCols(); j++){
            predictionTimeSeries[i][j] = continuousInputDataBuffer[i][j];
        }
    }

    //Run the prediction
    return predict(predictionTimeSeries);

}

bool DTW::reset(){
    continuousInputDataBuffer.clear();
    if( trained ){
        continuousInputDataBuffer.resize(averageTemplateLength,vector<double>(numFeatures,0));
        recomputeNullRejectionThresholds();
    }
    return true;
}

bool DTW::recomputeNullRejectionThresholds(){
	if(!trained) return false;

    //Copy the null rejection thresholds into one buffer so they can easily be accessed from the base class
    nullRejectionThresholds.resize(numTemplates);

	for(UINT k=0; k<numTemplates; k++){
		//The threshold is set as the mean distance plus gamma standard deviations
		templatesBuffer[k].threshold = templatesBuffer[k].trainingMu + (templatesBuffer[k].trainingSigma * nullRejectionCoeff);
        nullRejectionThresholds[k] = templatesBuffer[k].threshold;
	}

	return true;

}

////////////////////////// computeDistance ///////////////////////////////////////////

double DTW::computeDistance(Matrix<double> &timeSeriesA,Matrix<double> &timeSeriesB){

	double** distMatrix = NULL;
	vector< IndexDist > WarpPath;
	IndexDist tempW;
	const UINT M = timeSeriesA.getNumRows();
	const UINT N = timeSeriesB.getNumRows();
	const UINT C = timeSeriesA.getNumCols();
	UINT i,j,k,index = 0;
	double totalDist,v,normFactor = 0.;

    radius = ceil( min(M,N)/2.0 );

	//Construct the Distance Matrix
	distMatrix = new double*[M];
	for(i=0; i<M; i++) distMatrix[i] = new double[N];

	switch (distanceMethod) {
		case (ABSOLUTE_DIST):
			for(i=0; i<M; i++){
				for(j=0; j<N; j++){
					distMatrix[i][j] = 0.0;
					for(k=0; k< C; k++){
					   distMatrix[i][j] += fabs(timeSeriesA[i][k]-timeSeriesB[j][k]);
					}
				}
			}
			break;
		case (EUCLIDEAN_DIST):
			//Calculate Euclidean Distance for all possible values
			for(i=0; i<M; i++){
				for(j=0; j<N; j++){
					distMatrix[i][j] = 0.0;
					for(k=0; k< C; k++){
						distMatrix[i][j] += SQR(timeSeriesA[i][k]-timeSeriesB[j][k]);
					}
					distMatrix[i][j] = sqrt(distMatrix[i][j]);
				}
			}
			break;
		case (NORM_ABSOLUTE_DIST):
			for(i=0; i<M; i++){
				for(j=0; j<N; j++){
					distMatrix[i][j] = 0.0;
					for(k=0; k< C; k++){
					   distMatrix[i][j] += fabs(timeSeriesA[i][k]-timeSeriesB[j][k]);
					}
					distMatrix[i][j]/=N;
				}
			}
			break;
		default:
			errorLog<<"ERROR: Unknown distance method: "<<distanceMethod<<endl;
			//Cleanup Memory
			for(i=0; i<M; i++){
				delete[] distMatrix[i];
				distMatrix[i] = NULL;
			}
			delete[] distMatrix;
			distMatrix = NULL;
			return -1;
			break;
	}

    double distance = sqrt( d(M-1,N-1,distMatrix,M,N) );

    if( isinf(distance) ){
        warningLog << "DTW computeDistance(...) - Distance Matrix Values are INF!" << endl;
        return INFINITY;
    }

    //The distMatrix values are negative so make them positive
    for(i=0; i<M; i++){
        for(j=0; j<N; j++){
            distMatrix[i][j] = fabs( distMatrix[i][j] );
        }
    }

	//Now Create the Warp Path through the cost matrix, starting at the end
    i=M-1;
	j=N-1;
	tempW.x = i;
	tempW.y = j;
    tempW.dist = distMatrix[tempW.x][tempW.y];
	totalDist = distMatrix[tempW.x][tempW.y];
    WarpPath.push_back(tempW);
    
	//Use dynamic programming to navigate through the cost matrix until [0][0] has been reached
    normFactor = 1;
	while( i != 0 && j != 0 ) {
		if(i==0) j--;
		else if(j==0) i--;
		else{
            //Find the minimum cell to move to
			v = 99e+99;
			index = 0;
			if( distMatrix[i-1][j] < v ){ v = distMatrix[i-1][j]; index = 1; }
			if( distMatrix[i][j-1] < v ){ v = distMatrix[i][j-1]; index = 2; }
			if( distMatrix[i-1][j-1] < v ){ v = distMatrix[i-1][j-1]; index = 3; }
			switch(index){
				case(1):
					i--;
					break;
				case(2):
					j--;
					break;
				case(3):
					i--;
					j--;
					break;
				default:
                    warningLog << "DTW computeDistance(...) - Could not compute a warping path for the input matrix!" << endl;
					return INFINITY;
					break;
			}
		}
        normFactor++;
		tempW.x = i;
		tempW.y = j;
        tempW.dist = distMatrix[tempW.x][tempW.y];
		totalDist += distMatrix[tempW.x][tempW.y];
		WarpPath.push_back(tempW);
	}

	//Cleanup Memory
	for(i=0; i<M; i++){
		delete[] distMatrix[i];
		distMatrix[i] = NULL;
	}
	delete[] distMatrix;
	distMatrix = NULL;

	return totalDist/normFactor;

}

double DTW::d(int m,int n,double **distMatrix,const int M,const int N){
    double dist = 0;
    //The following is based on Matlab the DTW code by Eamonn Keogh and Michael Pazzani

    if( dtwConstrain ){
        //Test to see if the current cell is outside of the warping window
        if( m > 0 ){
            if( fabs( double( n-(N/(M/m)) ) ) > radius ){

                //Test to see if the current cell is above the warping window
                if( n-(N/(M/m)) > 0 ){
                    //Set all the values above and to the right to NAN
                    for(int i=0; i<m; i++)
                        for(int j=n; j<N; j++)
                            distMatrix[i][j] = NAN;
                }else{
                    //Set all the values below and to the left to NAN
                    for(int i=m; i<M; i++)
                        for(int j=0; j<n; j++)
                            distMatrix[i][j] = NAN;
                }

            }
        }
    }

    //If this cell contains a negative value then it has already been searched
    //The cost is therefore the absolute value of the negative value so return it
    if( distMatrix[m][n] < 0 || isnan( distMatrix[m][n] ) ){
        dist = fabs( distMatrix[m][n] );
        return dist;
    }

    //Case 1: A warping path has reached the end
    //Return the contribution of distance
    //Negate the value, to record the fact that this cell has been visited
    //End of recursion
    if( m == 0 && n == 0 ){
        dist = distMatrix[0][0];
        distMatrix[0][0] = -distMatrix[0][0];
        return dist;
    }

    //Case 2: we are somewhere in the top row of the matrix
    //Only need to consider moving left
    if( m == 0 ){
        double contribDist = d(m,n-1,distMatrix,M,N);

        dist = distMatrix[m][n] + contribDist;

        distMatrix[m][n] = -dist;
        return dist;
    }else{
        //Case 3: we are somewhere in the left column of the matrix
        //Only need to consider moving down
        if ( n == 0) {
            double contribDist = d(m-1,n,distMatrix,M,N);

            dist = distMatrix[m][n] + contribDist;

            distMatrix[m][n] = -dist;
            return dist;
        }else{
            //Case 4: We are somewhere away from the edges so consider moving in the three main directions
            double contribDist1 = d(m-1,n-1,distMatrix,M,N);
            double contribDist2 = d(m-1,n,distMatrix,M,N);
            double contribDist3 = d(m,n-1,distMatrix,M,N);
            double minValue = 99e+99;
            int index = 0;
            if( contribDist1 < minValue ){ minValue = contribDist1; index = 1; }
			if( contribDist2 < minValue ){ minValue = contribDist2; index = 2; }
			if( contribDist3 < minValue ){ minValue = contribDist3; index = 3; }

            switch ( index ) {
                case 1:
                    dist = distMatrix[m][n] + minValue;
                    break;
                case 2:
                    dist = distMatrix[m][n] + minValue;
                    break;
                case 3:
                    dist = distMatrix[m][n] + minValue;
                    break;

                default:
                    break;
            }

            distMatrix[m][n] = -dist; //Negate the value to record that it has been visited
            return dist;
        }
    }

    //This should not happen!
    return dist;
}

inline double DTW::MIN_(double a,double b, double c){
	double v = a;
	if(b<v) v = b;
	if(c<v) v = c;
	return v;
}


////////////////////////// SCALING AND NORMALISATION FUNCTIONS //////////////////////////

void DTW::scaleData(LabelledTimeSeriesClassificationData &trainingData){

	//Scale the data using the min and max values
    for(UINT i=0; i<trainingData.getNumSamples(); i++){
        scaleData( trainingData[i].getData(), trainingData[i].getData() );
    }

}

void DTW::scaleData(Matrix<double> &data,Matrix<double> &scaledData){

	const int R = data.getNumRows();
	const int C = data.getNumCols();

    if( scaledData.getNumRows() != R || scaledData.getNumCols() != C ){
        scaledData.resize(R, C);
    }

	//Scale the data using the min and max values
	for(int i=0; i<R; i++)
		for(int j=0; j<C; j++)
			scaledData[i][j] = scale(data[i][j],rangesBuffer[j].minValue,rangesBuffer[j].maxValue,0.0,1.0);

}

void DTW::znormData(LabelledTimeSeriesClassificationData &trainingData){

    for(UINT i=0; i<trainingData.getNumSamples(); i++){
        znormData( trainingData[i].getData(), trainingData[i].getData() );
    }

}

void DTW::znormData(Matrix<double> &data, Matrix<double> &normData){

	const UINT R = data.getNumRows();
	const UINT C = data.getNumCols();

    if( normData.getNumRows() != R || normData.getNumCols() != C ){
        normData.resize(R,C);
    }

	for(UINT j=0; j<C; j++){
		double mean = 0.0;
		double stdDev = 0.0;

		//Calculate Mean
		for(UINT i=0; i<R; i++) mean += data[i][j];
		mean /= double(R);

		//Calculate Std Dev
		for(UINT i=0; i<R; i++)
			stdDev += SQR(data[i][j]-mean);
		stdDev = sqrt( stdDev / (R - 1.0) );

		if(constrainZNorm && stdDev < 0.01){
			//Normalize the data to 0 mean
		    for(UINT i=0; i<R; i++)
			data[i][j] = (data[i][j] - mean);
		}else{
			//Normalize the data to 0 mean and standard deviation of 1
		    for(UINT i=0; i<R; i++)
			data[i][j] = (data[i][j] - mean) / stdDev;
		}

	}
}

void DTW::smoothData(vector<double> &data,UINT smoothFactor,vector<double> &resultsData){

	const UINT M = data.size();
	const UINT N = (UINT) floor(double(M)/double(smoothFactor));
	resultsData.resize(N,0);
	for(UINT i=0; i<N; i++) resultsData[i]=0.0;

	if(smoothFactor==1 || M<smoothFactor){
		resultsData = data;
		return;
	}

	for(UINT i=0; i<N; i++){
	    double mean = 0.0;
		UINT index = i*smoothFactor;
		for(UINT x=0; x<smoothFactor; x++){
			mean += data[index+x];
		}
		resultsData[i] = mean/smoothFactor;
	}
	//Add on the data that does not fit into the window
	if(M%smoothFactor!=0.0){
		double mean = 0.0;
			for(UINT i=N*smoothFactor; i<M; i++) mean += data[i];
        mean/=M-(N*smoothFactor);
		//Add one to the end of the vector
		vector<double> tempVector(N+1);
		for(UINT i=0; i<N; i++) tempVector[i] = resultsData[i];
		tempVector[N] = mean;
		resultsData = tempVector;
	}

}

void DTW::smoothData(Matrix<double> &data,UINT smoothFactor,Matrix<double> &resultsData){

	const UINT M = data.getNumRows();
	const UINT C = data.getNumCols();
	const UINT N = (UINT) floor(double(M)/double(smoothFactor));
	resultsData.resize(N,C);

	if(smoothFactor==1 || M<smoothFactor){
		resultsData = data;
		return;
	}

	for(UINT i=0; i<N; i++){
		for(UINT j=0; j<C; j++){
	     double mean = 0.0;
		 int index = i*smoothFactor;
		 for(UINT x=0; x<smoothFactor; x++){
			mean += data[index+x][j];
		 }
		 resultsData[i][j] = mean/smoothFactor;
		}
	}

	//Add on the data that does not fit into the window
	if(M%smoothFactor!=0.0){
		vector <double> mean(C,0.0);
		for(UINT j=0; j<C; j++){
		 for(UINT i=N*smoothFactor; i<M; i++) mean[j] += data[i][j];
		 mean[j]/=M-(N*smoothFactor);
		}

		//Add one row to the end of the Matrix
		Matrix<double> tempMatrix(N+1,C);

		for(UINT i=0; i<N; i++)
			for(UINT j=0; j<C; j++)
				tempMatrix[i][j] = resultsData[i][j];

        for(UINT j=0; j<C; j++) tempMatrix[N][j] = mean[j];
		resultsData = tempMatrix;
	}

}

////////////////////////////// SAVE & LOAD FUNCTIONS ////////////////////////////////

bool DTW::saveModelToFile( string fileName ){

    std::fstream file;

    if(!trained){
       errorLog << "saveDTWModelToFile( string fileName ) - Model not trained yet, can not save to file" << endl;
     return false;
    }

    file.open(fileName.c_str(), std::ios::out);

    if( !saveModelToFile( file ) ){
        return false;
    }

    file.close();
    return true;
}
    
bool DTW::saveModelToFile( fstream &file ){
    
    if(!file.is_open()){
        errorLog << "saveDTWModelToFile( string fileName ) - Could not open file to save data" << endl;
        return false;
    }
    
    file << "GRT_DTW_Model_File_V1.0" <<endl;
    file << "NumberOfDimensions: " << numFeatures <<endl;
    file << "NumberOfClasses: " << numClasses << endl;
    file << "NumberOfTemplates: " << numTemplates <<endl;
    file << "DistanceMethod: ";
    switch(distanceMethod){
        case(ABSOLUTE_DIST):
            file <<ABSOLUTE_DIST<<endl;
            break;
        case(EUCLIDEAN_DIST):
            file <<EUCLIDEAN_DIST<<endl;
            break;
        default:
            file <<ABSOLUTE_DIST<<endl;
            break;
    }
    file << "UseNullRejection: "<<useNullRejection<<endl;
    file << "UseSmoothing: "<<useSmoothing<<endl;
    file << "SmoothingFactor: "<<smoothingFactor<<endl;
    file << "UseScaling: "<<useScaling<<endl;
    file << "UseZNormalisation: "<<useZNormalisation<<endl;
    file << "RejectionMode: " << rejectionMode<< endl;
    file << "NullRejectionCoeff: "<<nullRejectionCoeff<<endl;
    file << "OverallAverageTemplateLength: "<<averageTemplateLength<<endl;
    //Save each template
    for(UINT i=0; i<numTemplates; i++){
        file<<"Template: "<<i+1<<endl;
        file<<"ClassLabel: "<<templatesBuffer[i].classLabel<<endl;
        file<<"TimeSeriesLength: "<<templatesBuffer[i].timeSeries.getNumRows()<<endl;
        file<<"TemplateThreshold: "<<templatesBuffer[i].threshold<<endl;
        file<<"TrainingMu: "<<templatesBuffer[i].trainingMu<<endl;
        file<<"TrainingSigma: "<<templatesBuffer[i].trainingSigma<<endl;
        file<<"AverageTemplateLength: "<<templatesBuffer[i].averageTemplateLength<<endl;
        file<<"TimeSeries: \n";
        for(UINT k=0; k<templatesBuffer[i].timeSeries.getNumRows(); k++){
            for(UINT j=0; j<templatesBuffer[i].timeSeries.getNumCols(); j++){
                file << templatesBuffer[i].timeSeries[k][j] << "\t";
            }file << endl;
        }
        file<<"***************************"<<endl;
        file<<endl;
    }
    
    return true;
}


bool DTW::loadModelFromFile( string fileName ){

   std::fstream file;
   file.open(fileName.c_str(), std::ios::in);

    if( !loadModelFromFile( file ) ){
        return false;
    }
    
	file.close();
	

    return trained;
}

bool DTW::loadModelFromFile( fstream &file ){
    
    std::string word;
    UINT timeSeriesLength;
    UINT ts;
    
    if(!file.is_open())
    {
        errorLog << "loadDTWModelFromFile( string fileName ) - Failed to open file!" << endl;
        return false;
    }
    
    //Check to make sure this is a file with the DTW File Format
    file >> word;
    if(word != "GRT_DTW_Model_File_V1.0"){
        errorLog << "loadDTWModelFromFile( string fileName ) - Unknown file header!" << endl;
        return false;
    }
    
    //Check and load the Number of Dimensions
    file >> word;
    if(word != "NumberOfDimensions:"){
        errorLog << "loadDTWModelFromFile( string fileName ) - Failed to find NumberOfDimensions!" << endl;
        return false;
    }
    file >> numFeatures;
    
    //Check and load the Number of Classes
    file >> word;
    if(word != "NumberOfClasses:"){
        errorLog << "loadDTWModelFromFile( string fileName ) - Failed to find NumberOfClasses!" << endl;
        return false;
    }
    file >> numClasses;
    
    //Check and load the Number of Templates
    file >> word;
    if(word != "NumberOfTemplates:"){
        errorLog << "loadDTWModelFromFile( string fileName ) - Failed to find NumberOfTemplates!" << endl;
        return false;
    }
    file >> numTemplates;
    
    //Check and load the Distance Method
    file >> word;
    if(word != "DistanceMethod:"){
        errorLog << "loadDTWModelFromFile( string fileName ) - Failed to find DistanceMethod!" << endl;
        return false;
    }
    file >> distanceMethod;
    
    //Check and load if UseNullRejection is used
    file >> word;
    if(word != "UseNullRejection:"){
        errorLog << "loadDTWModelFromFile( string fileName ) - Failed to find UseNullRejection!" << endl;
        return false;
    }
    file >> useNullRejection;
    
    //Check and load if Smoothing is used
    file >> word;
    if(word != "UseSmoothing:"){
        errorLog << "loadDTWModelFromFile( string fileName ) - Failed to find UseSmoothing!" << endl;
        return false;
    }
    file >> useSmoothing;
    
    //Check and load what the smoothing factor is
    file >> word;
    if(word != "SmoothingFactor:"){
        errorLog << "loadDTWModelFromFile( string fileName ) - Failed to find SmoothingFactor!" << endl;
        return false;
    }
    file >> smoothingFactor;
    
    //Check and load if Scaling is used
    file >> word;
    if(word != "UseScaling:"){
        errorLog << "loadDTWModelFromFile( string fileName ) - Failed to find UseScaling!" << endl;
        return false;
    }
    file >> useScaling;
    
    //Check and load if Scaling is used
    file >> word;
    if(word != "UseZNormalisation:"){
        errorLog << "loadDTWModelFromFile( string fileName ) - Failed to find UseZNormalisation!" << endl;
        return false;
    }
    file >> useZNormalisation;
    
    //Check and load if Scaling is used
    file >> word;
    if(word != "RejectionMode:"){
        errorLog << "loadDTWModelFromFile( string fileName ) - Failed to find RejectionMode!" << endl;
        return false;
    }
    file >> rejectionMode;
    
    //Check and load gamma
    file >> word;
    if(word != "NullRejectionCoeff:"){
        errorLog << "loadDTWModelFromFile( string fileName ) - Failed to find NullRejectionCoeff!" << endl;
        return false;
    }
    file >> nullRejectionCoeff;
    
    //Check and load the overall average template length
    file >> word;
    if(word != "OverallAverageTemplateLength:"){
        errorLog << "loadDTWModelFromFile( string fileName ) - Failed to find OverallAverageTemplateLength!" << endl;
        return false;
    }
    file >> averageTemplateLength;
    
    //Clean and reset the memory
    templatesBuffer.resize(numTemplates);
    classLabels.resize(numTemplates);
    
    //Load each template
    for(UINT i=0; i<numTemplates; i++){
        //Check we have the correct template
        file >> word;
        while(word != "Template:"){
            file >> word;
        }
        file >> ts;
        
        //Check the template number
        if(ts!=i+1){
            numTemplates=0;
            trained = false;
            errorLog << "loadDTWModelFromFile( string fileName ) - Failed to find Invalid Template Number!" << endl;
            return false;
        }
        
        //Get the class label of this template
        file >> word;
        if(word != "ClassLabel:"){
            numTemplates=0;
            trained = false;
            errorLog << "loadDTWModelFromFile( string fileName ) - Failed to find ClassLabel!" << endl;
            return false;
        }
        file >> templatesBuffer[i].classLabel;
        classLabels[i] = templatesBuffer[i].classLabel;
        
        //Get the time series length
        file >> word;
        if(word != "TimeSeriesLength:"){
            numTemplates=0;
            trained = false;
            errorLog << "loadDTWModelFromFile( string fileName ) - Failed to find TimeSeriesLength!" << endl;
            return false;
        }
        file >> timeSeriesLength;
        
        //Resize the buffers
        templatesBuffer[i].timeSeries.resize(timeSeriesLength,numFeatures);
        
        //Get the template threshold
        file >> word;
        if(word != "TemplateThreshold:"){
            numTemplates=0;
            trained = false;
            errorLog << "loadDTWModelFromFile( string fileName ) - Failed to find TemplateThreshold!" << endl;
            return false;
        }
        file >> templatesBuffer[i].threshold;
        
        //Get the mu values
        file >> word;
        if(word != "TrainingMu:"){
            numTemplates=0;
            trained = false;
            errorLog << "loadDTWModelFromFile( string fileName ) - Failed to find TrainingMu!" << endl;
            return false;
        }
        file >> templatesBuffer[i].trainingMu;
        
        //Get the sigma values
        file >> word;
        if(word != "TrainingSigma:"){
            numTemplates=0;
            trained = false;
            errorLog << "loadDTWModelFromFile( string fileName ) - Failed to find TrainingSigma!" << endl;
            return false;
        }
        file >> templatesBuffer[i].trainingSigma;
        
        //Get the AverageTemplateLength value
        file >> word;
        if(word != "AverageTemplateLength:"){
            numTemplates=0;
            trained = false;
            errorLog << "loadDTWModelFromFile( string fileName ) - Failed to find AverageTemplateLength!" << endl;
            return false;
        }
        file >> templatesBuffer[i].averageTemplateLength;
        
        //Get the data
        file >> word;
        if(word != "TimeSeries:"){
            numTemplates=0;
            trained = false;
            errorLog << "loadDTWModelFromFile( string fileName ) - Failed to find template timeseries!" << endl;
            return false;
        }
        for(UINT k=0; k<timeSeriesLength; k++)
            for(UINT j=0; j<numFeatures; j++)
                file >> templatesBuffer[i].timeSeries[k][j];
        
        //Check for the footer
        file >> word;
        if(word != "***************************"){
            numTemplates=0;
            numClasses = 0;
            numFeatures=0;
            trained = false;
            errorLog << "loadDTWModelFromFile( string fileName ) - Failed to find template footer!" << endl;
            return false;
        }
    }
    
    //Recompute the null rejection thresholds
    recomputeNullRejectionThresholds( );
    
    //Resize the prediction results to make sure it is setup for realtime prediction
    continuousInputDataBuffer.clear();
    continuousInputDataBuffer.resize(averageTemplateLength,vector<double>(numFeatures,0));
    maxLikelihood = DEFAULT_NULL_LIKELIHOOD_VALUE;
    bestDistance = DEFAULT_NULL_DISTANCE_VALUE;
    classLikelihoods.resize(numClasses,DEFAULT_NULL_LIKELIHOOD_VALUE);
    classDistances.resize(numClasses,DEFAULT_NULL_DISTANCE_VALUE);
    
    trained = true;
    
    return true;
}
bool DTW::setRejectionMode(UINT rejectionMode){
    if( rejectionMode == TEMPLATE_THRESHOLDS || rejectionMode == CLASS_LIKELIHOODS || rejectionMode == THRESHOLDS_AND_LIKELIHOODS ){
        this->rejectionMode = rejectionMode;
        return true;
    }
    return false;
}

} //End of namespace GRT

