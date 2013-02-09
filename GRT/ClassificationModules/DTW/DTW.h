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

#include "../../GestureRecognitionPipeline/Classifier.h"
#include "../../Util/LabelledTimeSeriesClassificationSampleTrimmer.h"

namespace GRT{
    
class IndexDist{
public:
	IndexDist():x(0),y(0),dist(0){};
	~IndexDist(){};
	IndexDist& operator=(const IndexDist &rhs){
		if(this!=&rhs){
            this->x = rhs.x;
            this->y = rhs.y;
            this->dist = rhs.dist;
		}
		return (*this);
	}

	int x;
	int y;
    double dist;
};

///////////////// DTW Template /////////////////
class DTWTemplate{
public:
	DTWTemplate(){
        classLabel = 0;
		trainingMu = 0.0;
		trainingSigma = 0.0;
		threshold=0.0;
		averageTemplateLength=0;
	}
	~DTWTemplate(){};

    UINT classLabel;                    //The class that this template belongs to
	Matrix<double> timeSeries;          //The raw time series
	double trainingMu;                  //The mean distance value of the training data with the trained template 
	double trainingSigma;               //The sigma of the distance value of the training data with the trained template 
	double threshold;                   //The classification threshold
	UINT averageTemplateLength;          //The average length of the examples used to train this template
};

class DTW : public Classifier
{
public:
	DTW(bool useScaling=false,bool useNullRejection=false,double nullRejectionCoeff=10.0,UINT rejectionMode = DTW::TEMPLATE_THRESHOLDS,bool useSmoothing = false,UINT smoothingFactor = 5);
	virtual ~DTW(void);
    
    DTW(const DTW &rhs){
        this->templatesBuffer = rhs.templatesBuffer;
        this->rangesBuffer = rhs.rangesBuffer;
        this->continuousInputDataBuffer = rhs.continuousInputDataBuffer;
        this->numTemplates = rhs.numTemplates;
        this->trained = rhs.trained;
        this->useSmoothing = rhs.useSmoothing;
        this->useScaling = rhs.useScaling;
        this->useZNormalisation = rhs.useZNormalisation;
        this->constrainZNorm = rhs.constrainZNorm;
        this->dtwConstrain = rhs.dtwConstrain;
        this->maxLikelihood = rhs.maxLikelihood;
        this->crossValidationAccuracy = rhs.crossValidationAccuracy;
        this->zNormConstrainThreshold = rhs.zNormConstrainThreshold;
        this->radius = rhs.radius;
        this->smoothingFactor = rhs.smoothingFactor;
        this->distanceMethod = rhs.distanceMethod;
        this->rejectionMode = rhs.rejectionMode;
        this->averageTemplateLength = rhs.averageTemplateLength;
        
        //Copy the classifier variables
        copyBaseVariables(this, (Classifier*)&rhs);
    }
    
    //Override the base class methods
    virtual bool clone(const Classifier *classifier){
        if( classifier == NULL ) return false;
        
        if( this->getClassifierType() == classifier->getClassifierType() ){
            
            DTW *ptr = (DTW*)classifier;
            //Clone the NDDTW values 
            this->templatesBuffer = ptr->templatesBuffer;
            this->rangesBuffer = ptr->rangesBuffer;
            this->continuousInputDataBuffer = ptr->continuousInputDataBuffer;
            this->numTemplates = ptr->numTemplates;
            this->trained = ptr->trained;
            this->useSmoothing = ptr->useSmoothing;
            this->useZNormalisation = ptr->useZNormalisation;
            this->constrainZNorm = ptr->constrainZNorm;
            this->dtwConstrain = ptr->dtwConstrain;
            this->maxLikelihood = ptr->maxLikelihood;
            this->crossValidationAccuracy = ptr->crossValidationAccuracy;
            this->zNormConstrainThreshold = ptr->zNormConstrainThreshold;
            this->radius = ptr->radius;
            this->smoothingFactor = ptr->smoothingFactor;
            this->distanceMethod = ptr->distanceMethod;
            this->rejectionMode = ptr->rejectionMode;
            this->averageTemplateLength = ptr->averageTemplateLength;
            
            //Clone the classifier variables
            return copyBaseVariables(this, ptr);
        }
        return false;
    }
    virtual bool train(LabelledTimeSeriesClassificationData &trainingData);
    virtual bool predict(vector< double > inputVector);
    virtual bool reset();
    virtual bool recomputeNullRejectionThresholds();
    virtual bool saveModelToFile(string filename);
    virtual bool saveModelToFile( fstream &file );
    virtual bool loadModelFromFile(string filename);
    virtual bool loadModelFromFile( fstream &file );
    UINT getNumTemplates(){ return numTemplates; }

	//NDDTW Public Methods
    bool predict(Matrix<double> &timeSeries);
    
    bool setRejectionMode(UINT rejectionMode);
    UINT getRejectionMode(){ return rejectionMode; }
    
    bool enableZNormalization(bool useZNormalisation){ this->useZNormalisation = useZNormalisation; return true; }

private:
	//Public training and prediction methods
    bool _train(LabelledTimeSeriesClassificationData &trainingData);
	bool _train_NDDTW(LabelledTimeSeriesClassificationData &trainingData,DTWTemplate &dtwTemplate,UINT &bestIndex);

	//The actual DTW function
	double computeDistance(Matrix<double> &timeSeriesA,Matrix<double> &timeSeriesB);
    double d(int m,int n,double **AccMatrix,const int M,const int N);
	double inline MIN_(double a,double b, double c);

	//Private Scaling and Utility Functions
	void scaleData(LabelledTimeSeriesClassificationData &trainingData);
	void scaleData(Matrix<double> &data,Matrix<double> &scaledData);
	void znormData(LabelledTimeSeriesClassificationData &trainingData);
	void znormData(Matrix<double> &data,Matrix<double> &normData);
	void smoothData(vector<double> &data,UINT smoothFactor,vector<double> &resultsData);
	void smoothData(Matrix<double> &data,UINT smoothFactor,Matrix<double> &resultsData);
    
    static RegisterClassifierModule< DTW > registerModule;

public:
	vector< DTWTemplate > templatesBuffer;		//A buffer to store the templates for each time series
	vector< MinMax >	rangesBuffer;			//A buffer to store the min-max ranges for scaling each channel
    CircularBuffer< vector< double > > continuousInputDataBuffer;
	UINT				numTemplates;			//The number of templates in our buffer
    UINT                rejectionMode;          //The rejection mode used to reject null gestures during the prediction phase

	//Flags
	bool				useSmoothing;			//A flag to check if we need to smooth the data
	bool				useZNormalisation;		//A flag to check if we need to znorm the training and prediction data
	bool				constrainZNorm;			//A flag to check if we need to constrain zNorm (only zNorm if stdDev > zNormConstrainThreshold)
	bool				dtwConstrain;			//A flag to check if we need to constrain the dtw cost matrix and search
    bool                trimTrainingData;       //A flag to check if we need to trim the training data first before training

	double				crossValidationAccuracy;//The cross validation result
	double				zNormConstrainThreshold;//The threshold value to be used if constrainZNorm is turned on
	double				radius;					//The radius value to use if dtwConstrain is turned on
	
	UINT				smoothingFactor;		//The smoothing factor if smoothing is used
	UINT				distanceMethod;			//The distance method to be used (should be of enum DISTANCE_METHOD)
	UINT				averageTemplateLength;	//The overall average template length (over all the templates)
	
	enum DistanceMethods{ABSOLUTE_DIST=0,EUCLIDEAN_DIST,NORM_ABSOLUTE_DIST};
    enum RejectionModes{TEMPLATE_THRESHOLDS=0,CLASS_LIKELIHOODS,THRESHOLDS_AND_LIKELIHOODS};

};
    
}//End of namespace GRT
