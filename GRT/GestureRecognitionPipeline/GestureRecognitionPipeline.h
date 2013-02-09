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

#include "PreProcessing.h"
#include "FeatureExtraction.h"
#include "Classifier.h"
#include "Regressifier.h"
#include "PostProcessing.h"
#include "Context.h"
#include "../DataStructures/LabelledContinuousTimeSeriesClassificationData.h"

namespace GRT{
    
#define INSERT_AT_END_INDEX 99999

class GestureRecognitionPipeline
{
public:
	GestureRecognitionPipeline(void);
	~GestureRecognitionPipeline(void);
    
    //Main training functions
    bool train(LabelledClassificationData &trainingData);
    bool train(LabelledClassificationData &trainingData, UINT kFoldValue, bool useStratifiedSampling = false );
    bool train(LabelledTimeSeriesClassificationData &trainingData);
    bool train(LabelledRegressionData &trainingData);
    
    //Main testing functions
    bool test(LabelledClassificationData &testData);
    bool test(LabelledTimeSeriesClassificationData &testData);
    bool test(LabelledContinuousTimeSeriesClassificationData &testData);
    bool test(LabelledRegressionData &testData);
    
    //Main prediction functions
    bool predict(vector< double > inputVector);
    bool map(vector< double > inputVector);
    
    //The main util functions
    bool reset();
    bool savePipelineToFile(string filename);
    bool loadPipelineFromFile(string filename);
    
    //Some useful util functions for training and testing the pre-processing and feature extraction modules
    bool preProcessData(vector< double > inputVector,bool computeFeatures = true);
    
    //Getters
    bool getIsInitialized();
    bool getTrained();
    bool getIsPreProcessingSet();
    bool getIsFeatureExtractionSet();
    bool getIsClassifierSet();
    bool getIsRegressifierSet();
    bool getIsPostProcessingSet();
    bool getIsContextSet();
    bool getIsPipelineModeSet();
    bool getIsPipelineInClassificationMode();
    bool getIsPipelineInRegressionMode();
    
    UINT getInputVectorDimensionsSize();
    UINT getOutputVectorDimensionsSize();
    UINT getNumClassesInModel();
    UINT getNumPreProcessingModules();
    UINT getNumFeatureExtractionModules();
    UINT getNumPostProcessingModules();
    UINT getPredictionModuleIndexPosition();
    UINT getPredictedClassLabel();
    UINT getUnProcessedPredictedClassLabel();
    
    double getMaximumLikelihood();
    double getCrossValidationAccuracy();
    double getTestAccuracy();
    double getTestRMSError();
    double getTestFMeasure(UINT classLabel);
    double getTestPrecision(UINT classLabel);
    double getTestRecall(UINT classLabel);
    double getTestRejectionPrecision();
    double getTestRejectionRecall();
    Matrix<double> getTestConfusionMatrix();
    vector< double > getClassLikelihoods();
    vector< double > getClassDistances();
    vector< double > getNullRejectionThresholds();
    vector< double > getRegressionData();
    vector< double > getUnProcessedRegressionData();
    vector< double > getPreProcessedData();
    vector< double > getPreProcessedData(UINT moduleIndex);
    vector< double > getFeatureExtractionData();
    vector< double > getFeatureExtractionData(UINT moduleIndex);
    vector< UINT > getClassLabels();
    
    template <class T> T* getPreProcessingModule(UINT moduleIndex){ 
        if( moduleIndex < preProcessingModules.size() ){
            return (T*)preProcessingModules[ moduleIndex ];
        }
        return NULL;
    }
    
    template <class T> T* getFeatureExtractionModule(UINT moduleIndex){ 
        if( moduleIndex < featureExtractionModules.size() ){
            return (T*)featureExtractionModules[ moduleIndex ];
        }
        return NULL;
    }
    template <class T> T* getClassifier(){ return (T*)classifier; }
    
    template <class T> T* getPostProcessingModule(UINT moduleIndex){ 
        if( moduleIndex < postProcessingModules.size() ){
            return (T*)postProcessingModules[ moduleIndex ];
        }
        return NULL;
    }
    
    template <class T> T* getContextModule(UINT contextLevel,UINT moduleIndex){ 
        if( contextLevel < contextModules.size() ){
            if( moduleIndex < contextModules[ contextLevel ].size() ){
                return (T*)contextModules[ contextLevel ][ moduleIndex ];
            }
        }
        return NULL;
    }
    
    PreProcessing* getPreProcessingModule(UINT moduleIndex);
    FeatureExtraction* getFeatureExtractionModule(UINT moduleIndex);
    Classifier* getClassifier();
    Regressifier* getRegressifier();
    PostProcessing* getPostProcessingModule(UINT moduleIndex);
    Context* getContextModule(UINT contextLevel,UINT moduleIndex);
    
    string getPipelineModeAsString();
    
    //Setters
    bool addPreProcessingModule(const PreProcessing &preProcessingModule,UINT insertIndex = INSERT_AT_END_INDEX);
    bool addFeatureExtractionModule(const FeatureExtraction &featureExtractionModule,UINT insertIndex = INSERT_AT_END_INDEX);
    bool setClassifier(const Classifier &classifier);
    bool setRegressifier(const Regressifier &regressifier);
    bool addPostProcessingModule(const PostProcessing &postProcessingModule,UINT insertIndex = INSERT_AT_END_INDEX);
    
    template <class T> bool addContextModule(T contextModule,UINT contextLevel,UINT insertIndex = INSERT_AT_END_INDEX){
        
        if( contextLevel >= contextModules.size() ){
            errorLog << "addContextModule(...) - Context Level is out of bounds!" << endl;
            return false;
        }
        
        //Validate the insertIndex is valid
        if( insertIndex != INSERT_AT_END_INDEX && insertIndex >= contextModules[contextLevel].size() ){
            errorLog << "addContextModule(...) - Invalid insertIndex value!" << endl;
            return false;
        }
        
        //Create a new instance of the context module and then clone the values across from the reference contextModule
        Context *newInstance = new T;
        
        //Verify that the clone was successful
        if( !newInstance->clone( &contextModule ) ){
            delete newInstance;
            newInstance = NULL;
            errorLog << "addContextModule(...) - Context Module Not Set!" << endl;
            return false;
        }
        
        //Add the new instance to the contextModules
        vector< Context* >::iterator iter = contextModules[ contextLevel ].begin();
        
        if( insertIndex == INSERT_AT_END_INDEX ) iter = contextModules[ contextLevel ].end(); 
        else iter = contextModules[ contextLevel ].begin() + insertIndex;
        
        contextModules[ contextLevel ].insert(iter, newInstance);
        
        return true;
    }
    
    bool updateContextModule(bool value,UINT contextLevel = 0,UINT moduleIndex = 0){
        
        //Validate the contextLevel is valid
        if( contextLevel >= contextModules.size() ){
            errorLog << "updateContextModule(...) - Context Level is out of bounds!" << endl;
            return false;
        }
        
        //Validate the moduleIndex is valid
        if( moduleIndex >= contextModules[contextLevel].size() ){
            errorLog << "updateContextModule(...) - Invalid contextLevel value!" << endl;
            return false;
        }
        
        cout << "*******************UPDATING CONTEXT MODULE*******************!\n";
        
        return contextModules[contextLevel][moduleIndex]->updateContext( value );
    }

    bool removeAllPreProcessingModules();
    bool removePreProcessingModule(UINT moduleIndex);
    bool removeAllFeatureExtractionModules();
    bool removeFeatureExtractionModule(UINT moduleIndex);
    bool removeClassifier(){ deleteClassifier(); return true; }
    bool removeAllPostProcessingModules();
    bool removePostProcessingModule(UINT moduleIndex);
    bool removeContextModule(UINT contextLevel,UINT moduleIndex);
    bool removeAllContextModules();

protected:
    void deleteAllPreProcessingModules();
    void deleteAllFeatureExtractionModules();
    void deleteClassifier();
    void deleteRegressifier();
    void deleteAllPostProcessingModules();
    void deleteAllContextModules();
    bool updateTestMetrics(const UINT classLabel,const UINT predictedClassLabel,vector< double > &precisionCounter,vector< double > &recallCounter,double &rejectionPrecisionCounter,double &rejectionRecallCounter,vector< double > &confusionMatrixCounter);
    bool computeTestMetrics(vector< double > &precisionCounter,vector< double > &recallCounter,double &rejectionPrecisionCounter,double &rejectionRecallCounter,vector< double > &confusionMatrixCounter,const UINT numTestSamples);
    inline double SQR(double x){ return x*x; }
    
    bool initialized;
    bool trained;
    UINT inputVectorDimensions;
    UINT outputVectorDimensions;
    UINT predictedClassLabel;
    UINT pipelineMode;
    UINT predictionModuleIndex;
    double testAccuracy;
    double testRMSError;
    vector< double > testFMeasure;
    vector< double > testPrecision;
    vector< double > testRecall;
    vector< double > regressionData;
    double testRejectionPrecision;
    double testRejectionRecall;
    Matrix< double > testConfusionMatrix;
    
    vector< PreProcessing* > preProcessingModules;
    vector< FeatureExtraction* > featureExtractionModules;
    Classifier *classifier;
    Regressifier *regressifier;
    vector< PostProcessing* > postProcessingModules;
    vector< vector< Context* > > contextModules;
    
    DebugLog debugLog;
    ErrorLog errorLog;
    WarningLog warningLog;
    
    enum PipelineModes{PIPELINE_MODE_NOT_SET=0,CLASSIFICATION_MODE,REGRESSION_MODE};
    
public:
    enum ContextLevels{START_OF_PIPELINE=0,AFTER_PREPROCESSING,AFTER_FEATURE_EXTRACTION,AFTER_CLASSIFIER,END_OF_PIPELINE,NUM_CONTEXT_LEVELS};
    
};

} //End of namespace GRT

