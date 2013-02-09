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

#include "Neuron.h"
#include "../../../DataStructures/LabelledRegressionData.h"
#include "../../../GestureRecognitionPipeline/Regressifier.h"

namespace GRT{

class MLP : public Regressifier{
public:
    MLP();
    ~MLP();
    MLP &operator=(const MLP &rhs){
		if( this != &rhs ){
            //MLP variables
            this->numInputNeurons = rhs.numInputNeurons;
            this->numHiddenNeurons = rhs.numHiddenNeurons;
            this->numOutputNeurons = rhs.numOutputNeurons;
            this->inputLayerActivationFunction = rhs.inputLayerActivationFunction;
            this->hiddenLayerActivationFunction = rhs.hiddenLayerActivationFunction;
            this->outputLayerActivationFunction = rhs.outputLayerActivationFunction;
            this->minNumEpochs = rhs.minNumEpochs;
            this->maxNumEpochs = rhs.maxNumEpochs;
            this->numRandomTrainingIterations = rhs.numRandomTrainingIterations;
            this->validationSetSize = rhs.validationSetSize;
            this->minChange = rhs.minChange;
            this->trainingRate = rhs.trainingRate;
            this->momentum = rhs.momentum;
            this->gamma = rhs.gamma;
            this->trainingError = rhs.trainingError;
            this->useValidationSet = rhs.useValidationSet;
            this->randomiseTrainingOrder = rhs.randomiseTrainingOrder;
            this->useMultiThreadingTraining = rhs.useMultiThreadingTraining;
            this->initialized = rhs.initialized;
            this->inputLayer = rhs.inputLayer;
            this->hiddenLayer = rhs.hiddenLayer;
            this->outputLayer = rhs.outputLayer;
            this->inputVectorRanges = rhs.inputVectorRanges;
            this->targetVectorRanges = rhs.targetVectorRanges;
            this->trainingErrorLog = rhs.trainingErrorLog;
            
            this->classificationModeActive = rhs.classificationModeActive;
            this->useNullRejection = rhs.useNullRejection;
            this->predictedClassLabel = rhs.predictedClassLabel;
            this->nullRejectionCoeff = rhs.nullRejectionCoeff;
            this->nullRejectionThreshold = rhs.nullRejectionThreshold;
            this->maxLikelihood = rhs.maxLikelihood;
            this->classLikelihoods = rhs.classLikelihoods;
            
            //Copy the base variables
            copyBaseVariables(this,(Regressifier*)&rhs);
		}
		return *this;
	}
    
    //Override the base class methods
    virtual bool clone(const Regressifier *regressifierPtr){
        
        if( regressifierPtr == NULL ){
            errorLog << "clone(Regressifier *regressifierPtr) - regressifierPtr is NULL!\n" << endl;
            return false;
        }
        
        if( this->getRegressifierType() != regressifierPtr->getRegressifierType() ){
            errorLog << "clone(Regressifier *regressifierPtr) - regressifierPtr is not the correct type!\n" << endl;
            return false;
        }
        
        MLP *ptr = (MLP*)regressifierPtr;
        
        this->numInputNeurons = ptr->numInputNeurons;
        this->numHiddenNeurons = ptr->numHiddenNeurons;
        this->numOutputNeurons = ptr->numOutputNeurons;
        this->inputLayerActivationFunction = ptr->inputLayerActivationFunction;
        this->hiddenLayerActivationFunction = ptr->hiddenLayerActivationFunction;
        this->outputLayerActivationFunction = ptr->outputLayerActivationFunction;
        this->minNumEpochs = ptr->minNumEpochs;
        this->maxNumEpochs = ptr->maxNumEpochs;
        this->numRandomTrainingIterations = ptr->numRandomTrainingIterations;
        this->validationSetSize =  ptr->validationSetSize;
        this->minChange = ptr->minChange;
        this->trainingRate = ptr->trainingRate;
        this->momentum = ptr->momentum;
        this->gamma = ptr->gamma;
        this->trainingError = ptr->trainingError;
        this->useValidationSet = ptr->useValidationSet;
        this->randomiseTrainingOrder = ptr->randomiseTrainingOrder;
        this->useMultiThreadingTraining = ptr->useMultiThreadingTraining;
        this->initialized = ptr->initialized;
        this->classificationModeActive = ptr->classificationModeActive;
        this->useNullRejection = ptr->useNullRejection;
        this->inputLayer = ptr->inputLayer;
        this->hiddenLayer = ptr->hiddenLayer;
        this->outputLayer = ptr->outputLayer;
        this->inputVectorRanges = ptr->inputVectorRanges;
        this->targetVectorRanges =  ptr->targetVectorRanges;
        this->trainingErrorLog = ptr->trainingErrorLog;
        
        this->classificationModeActive = ptr->classificationModeActive;
        this->useNullRejection = ptr->useNullRejection;
        this->predictedClassLabel = ptr->predictedClassLabel;
        this->nullRejectionCoeff = ptr->nullRejectionCoeff;
        this->nullRejectionThreshold = ptr->nullRejectionThreshold;
        this->maxLikelihood = ptr->maxLikelihood;
        this->classLikelihoods = ptr->classLikelihoods;
        
        //Copy the base variables
        if( !Regressifier::copyBaseVariables(this,regressifierPtr) ) return false;
        
        return true;
    }
    bool train(LabelledClassificationData &trainingData);
    bool train(LabelledRegressionData &trainingData);
    bool predict(vector< double > inputVector);
    bool saveModelToFile(string filename){ return saveMLPToFile(filename); }
    bool loadModelFromFile(string filename){ return loadMLPFromFile(filename); }
    UINT getNumClasses(){ if( classificationModeActive ){ return numOutputNeurons; } else return 0; }
    
    bool init(UINT numInputNeurons,UINT numHiddenNeurons,UINT numOutputNeurons,UINT inputLayerActivationFunction = Neuron::LINEAR,
              UINT hiddenLayerActivationFunction = Neuron::LINEAR,UINT outputLayerActivationFunction = Neuron::LINEAR);
    void clear();
    
    double back_prop(vector< double > &trainingExample,vector< double > &targetVector,double alpha,double beta);
    vector< double > feedforward(vector< double > trainingExample);
    void feedforward(vector< double > &trainingExample,vector< double > &inputNeuronsOuput,
                     vector< double > &hiddenNeuronsOutput,vector< double > &outputNeuronsOutput);
    void printNetwork();
    bool checkForNAN();
    bool inline isNAN(double v);
	bool saveMLPToFile(string fileName);
	bool loadMLPFromFile(string fileName);
	string activationFunctionToString(UINT activationFunction);
	UINT activationFunctionFromString(string activationName);
	bool validateActivationFunction(UINT actvationFunction);

	//Getters 
	UINT getNumInputNeurons(){ return numInputNeurons; }
	UINT getNumHiddenNeurons(){ return numHiddenNeurons; }
	UINT getNumOutputNeurons(){ return numOutputNeurons; }
	UINT getInputLayerActivationFunction(){ return inputLayerActivationFunction; }
	UINT getHiddenLayerActivationFunction(){ return hiddenLayerActivationFunction; }
	UINT getOutputLayerActivationFunction(){ return outputLayerActivationFunction; }
	UINT getMinNumEpochs(){ return minNumEpochs; }
	UINT getMaxNumEpochs(){ return maxNumEpochs; }
	UINT getNumRandomTrainingIterations(){ return numRandomTrainingIterations; }
	UINT getValidationSetSize(){ return validationSetSize; }
	double getMinChange(){ return minChange; }
	double getTrainingRate(){ return trainingRate; }
	double getMomentum(){ return momentum; }
	double getGamma(){ return gamma; }
	double getTrainingError(){ return trainingError; }
	bool getUseValidationSet(){ return useValidationSet; }
	bool getRandomiseTrainingOrder(){ return randomiseTrainingOrder; }
    bool getUseNullRejection(){ return useNullRejection; }
	bool getTrainingStatus(){ return trained; }
    bool getClassificationModeActive(){ return classificationModeActive; }
    bool getRegressionModeActive(){ return !classificationModeActive; }
	vector< Neuron > getInputLayer(){ return inputLayer; }
	vector< Neuron > getHiddenLayer(){ return hiddenLayer; }
	vector< Neuron > getOutputLayer(){ return outputLayer; }
	vector< MinMax > getInputRanges(){ return inputVectorRanges; }
	vector< MinMax > getOutputRanges(){ return targetVectorRanges; }
	vector< vector< double > > getTrainingLog(){ return trainingErrorLog; }
    
    //Classifier Getters
    bool getNullRejectionEnabled(){ return useNullRejection; }
    
    /**
     Returns the current gamma value.
     The gamma parameter is a multipler controlling the null rejection threshold for each class.
     
     @return returns the current gamma value
     */
    double getNullRejectionCoeff(){ return nullRejectionCoeff; }
    double getNullRejectionThreshold(){ return nullRejectionThreshold; }
    double getMaximumLikelihood(){ if( trained ){ return maxLikelihood; } return DEFAULT_NULL_LIKELIHOOD_VALUE; }
    
    vector< double > getClassLikelihoods(){ if( trained && classificationModeActive ){ return classLikelihoods; } return vector< double>(); }
    
    /**
     Gets the predicted class label from the last prediction.
     
     @return returns the label of the last predicted class, a value of 0 will be returned if the model has not been trained
     */
    UINT getPredictedClassLabel(){ if( trained && classificationModeActive ){ return predictedClassLabel; } return 0; }

	//Setters
	bool setMinChange(double minChange);
	bool setTrainingRate(double trainingRate);
	bool setMomentum(double momentum);
	bool setGamma(double gamma);
	bool setUseValidationSet(bool useValidationSet);
	bool setRandomiseTrainingOrder(bool randomiseTrainingOrder);
    bool setUseMultiThreadingTraining(bool useMultiThreadingTraining);
    bool setMinNumEpochs(UINT minNumEpochs);
    bool setMaxNumEpochs(UINT maxNumEpochs);
    bool setNumRandomTrainingIterations(UINT numRandomTrainingIterations);
    bool setValidationSetSize(UINT validationSetSize);
    
    //Classifier Setters
    bool setNullRejection(bool useNullRejection){ this->useNullRejection = useNullRejection; return true; }
    bool setNullRejectionCoeff(double nullRejectionCoeff){ if( nullRejectionCoeff > 0 ){ this->nullRejectionCoeff = nullRejectionCoeff; return true; } return false; }
    
private:
    UINT numInputNeurons;
    UINT numHiddenNeurons;
    UINT numOutputNeurons;
    UINT inputLayerActivationFunction;
    UINT hiddenLayerActivationFunction;
    UINT outputLayerActivationFunction;
	UINT minNumEpochs;
    UINT maxNumEpochs;
    UINT numRandomTrainingIterations;
	UINT validationSetSize;
    double minChange;
	double trainingRate;
	double momentum;
	double gamma;
	double trainingError;
    bool initialized;
	bool useValidationSet;
	bool randomiseTrainingOrder;
    bool useMultiThreadingTraining;
    Random random;
    
    vector< Neuron > inputLayer;
    vector< Neuron > hiddenLayer;
    vector< Neuron > outputLayer;
	vector< MinMax > inputVectorRanges;
	vector< MinMax > targetVectorRanges;
	vector< vector< double > > trainingErrorLog;
    
    //Classifier Variables
    bool classificationModeActive;
    bool useNullRejection;
    UINT predictedClassLabel;
    double nullRejectionThreshold;
    double nullRejectionCoeff;
    double maxLikelihood;
    vector< double > classLikelihoods;
    
    static RegisterRegressifierModule< MLP > registerModule;
    
};

} //End of namespace GRT