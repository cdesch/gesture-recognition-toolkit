/**
 @file
 @author  Nicholas Gillian <ngillian@media.mit.edu>
 @version 1.0
 
 @section LICENSE
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
 
 @section DESCRIPTION
 This class implements the Gaussian Mixture Model Classifier algorithm. 
 
 The Gaussian Mixture Model Classifier (GMM) is basic but useful classification algorithm that can be used to
 classify an N-dimensional signal.
 */

#pragma once

#include "../../GestureRecognitionPipeline/Classifier.h"
#include "../../ClusteringModules/GaussianMixtureModels/GaussianMixtureModels.h"
#include "MixtureModel.h"

#define GMM_MIN_SCALE_VALUE 0.0001
#define GMM_MAX_SCALE_VALUE 1

namespace GRT {

class GMM : public Classifier
{
public:
	GMM(UINT numMixtureModels = 2,bool useScaling=false,bool useNullRejection=false,double nullRejectionCoeff=1.0,UINT maxIter=100,double minChange=1.0e-5);
    
	virtual ~GMM(void);
    
    /**
     Defines how the data from the rhs GMM should be copied to this GMM
     
     @param const GMM &rhs: another instance of a GMM
     @return returns a pointer to this instance of the GMM
     */
	GMM &operator=(const GMM &rhs){
		if( this != &rhs ){
            
            this->numMixtureModels = rhs.numMixtureModels;
            this->maxIter = rhs.maxIter;
            this->minChange = rhs.minChange;
            this->models = rhs.models;
            
            this->debugLog = rhs.debugLog;
            this->errorLog = rhs.errorLog;
            this->warningLog = rhs.warningLog;

            //Classifier variables
            copyBaseVariables(this, (Classifier*)&rhs);
		}
		return *this;
	}
    
    //Override the base class methods
    /**
     This is required for the Gesture Recognition Pipeline for when the pipeline.setClassifier method is called.  
     It clones the data from the Base Class Classifier pointer (which should be pointing to an GMM instance) into this instance
     
     @param Classifier *classifier: a pointer to the Classifier Base Class, this should be pointing to another GMM instance
     @return returns true if the clone was successfull, false otherwise
     */
    virtual bool clone(const Classifier *classifier){
        if( classifier == NULL ) return false;
        
        if( this->getClassifierType() == classifier->getClassifierType() ){
            
            GMM *ptr = (GMM*)classifier;
            //Clone the GMM values 
            this->numMixtureModels = ptr->numMixtureModels;
            this->maxIter = ptr->maxIter;
            this->minChange = ptr->minChange;
            this->models = ptr->models;
            
            this->debugLog = ptr->debugLog;
            this->errorLog = ptr->errorLog;
            this->warningLog = ptr->warningLog;
            
            //Clone the classifier variables
            return copyBaseVariables(this, classifier);
        }
        return false;
    }
    
    /**
     This trains the GMM model, using the labelled classification data.
     This overrides the train function in the Classifier base class.
     The GMM is an unsupervised learning algorithm, it will therefore NOT use any class labels provided
     
     @param LabelledClassificationData &trainingData: a reference to the training data
     @return returns true if the GMM model was trained, false otherwise
     */
    virtual bool train(LabelledClassificationData &trainingData);
    
    /**
     This predicts the class of the inputVector.
     This overrides the predict function in the Classifier base class.
     
     @param vector< double > inputVector: the input vector to classify
     @return returns true if the prediction was performed, false otherwise
     */
    virtual bool predict(vector< double > inputVector);
    
    /**
     This saves the trained GMM model to a file.
     This overrides the saveModelToFile function in the Classifier base class.
     
     @param string filename: the name of the file to save the GMM model to
     @return returns true if the model was saved successfully, false otherwise
     */
    virtual bool saveModelToFile(string filename);
    
    /**
     This saves the trained GMM model to a file.
     This overrides the saveModelToFile function in the Classifier base class.
     
     @param fstream &file: a reference to the file the GMM model will be saved to
     @return returns true if the model was saved successfully, false otherwise
     */
    virtual bool saveModelToFile(fstream &file);
    
    /**
     This loads a trained GMM model from a file.
     This overrides the loadModelFromFile function in the Classifier base class.
     
     @param string filename: the name of the file to load the GMM model from
     @return returns true if the model was loaded successfully, false otherwise
     */
    virtual bool loadModelFromFile(string filename);
    
    /**
     This loads a trained GMM model from a file.
     This overrides the loadModelFromFile function in the Classifier base class.
     
     @param fstream &file: a reference to the file the GMM model will be loaded from
     @return returns true if the model was loaded successfully, false otherwise
     */
    virtual bool loadModelFromFile(fstream &file);
    
    bool recomputeNullRejectionThresholds();
    
	UINT getNumMixtureModels(){ return numMixtureModels; }
    
    vector< MixtureModel > getModels(){ if( trained ){ return models; } return vector< MixtureModel >(); }
    
    bool setNumMixtureModels(UINT K){
        if( K > 0 ){
            numMixtureModels = K;
            return true;
        }
        return false;
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
    
protected:
    double computeMixtureLikelihood(vector<double> &x,UINT k);
    
    UINT numMixtureModels;
    UINT maxIter;
    double minChange;
    vector< MixtureModel > models;
    
    DebugLog debugLog;
    ErrorLog errorLog;
    WarningLog warningLog;
    
    static RegisterClassifierModule< GMM > registerModule;
	
};
    
}//End of namespace GRT
