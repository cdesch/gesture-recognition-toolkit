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

#include "GestureRecognitionPipeline.h"

namespace GRT{

GestureRecognitionPipeline::GestureRecognitionPipeline(void)
{
    initialized = false;
    trained = false;
    pipelineMode = PIPELINE_MODE_NOT_SET;
    inputVectorDimensions = 0;
    outputVectorDimensions = 0;
    predictedClassLabel = 0;
    predictionModuleIndex = 0;
    testAccuracy = 0;
    testRMSError = 0;
    testRejectionPrecision = 0;
    testRejectionRecall = 0;
    classifier = NULL;
    regressifier = NULL;
    contextModules.resize( NUM_CONTEXT_LEVELS );
    
    debugLog.setProceedingText("[DEBUG GRP]");
    errorLog.setProceedingText("[ERROR GRP]");
    warningLog.setProceedingText("[WARNING GRP]");
}

GestureRecognitionPipeline::~GestureRecognitionPipeline(void)
{
    //Clean up the memory
    deleteAllPreProcessingModules();
    deleteAllFeatureExtractionModules();
    deleteClassifier();
    deleteRegressifier();
    deleteAllPostProcessingModules();
    deleteAllContextModules();
} 
    
bool GestureRecognitionPipeline::train(LabelledClassificationData &trainingData){
    
    trained = false;
    testAccuracy = 0;
    testRMSError = 0;
    testFMeasure.clear();
    testPrecision.clear();
    testRecall.clear();
    testRejectionPrecision = 0;
    testRejectionRecall = 0;
    
    if( !getIsClassifierSet() ){
        errorLog << "train(LabelledClassificationData &trainingData) - Failed To Train Classifier, the classifier has not been set!" << endl;
        return false;
    }
    
    if( trainingData.getNumSamples() == 0 ){
        errorLog << "train(LabelledClassificationData &trainingData) - Failed To Train Classifier, there is no training data!" << endl;
        return false;
    }
    
    //Set the input vector dimension size
    inputVectorDimensions = trainingData.getNumDimensions();
    
    //Pass the training data through any pre-processing or feature extraction units
    LabelledClassificationData processedTrainingData( trainingData.getNumDimensions() );
    
    for(UINT i=0; i<trainingData.getNumSamples(); i++){
        bool okToAddProcessedData = true;
        UINT classLabel = trainingData[i].getClassLabel();
        vector< double > trainingSample = trainingData[i].getSample();
        
        //Perform any preprocessing
        if( getIsPreProcessingSet() ){
            for(UINT moduleIndex=0; moduleIndex<preProcessingModules.size(); moduleIndex++){
                if( !preProcessingModules[moduleIndex]->process( trainingSample ) ){
                    errorLog << "train(LabelledClassificationData &trainingData) - Failed to PreProcess Training Data. PreProcessingModuleIndex: " << moduleIndex << endl;
                    return false;
                }
                trainingSample = preProcessingModules[moduleIndex]->getProcessedData();
            }
        }
        
        //Compute any features
        if( getIsFeatureExtractionSet() ){
            for(UINT moduleIndex=0; moduleIndex<featureExtractionModules.size(); moduleIndex++){
                if( !featureExtractionModules[moduleIndex]->computeFeatures( trainingSample ) ){
                    errorLog << "train(LabelledClassificationData &trainingData) - Failed to Compute Features from Training Data. FeatureExtractionModuleIndex: " << moduleIndex << endl;
                    return false;
                }
                if( featureExtractionModules[moduleIndex]->getFeatureDataReady() ){
                    trainingSample = featureExtractionModules[moduleIndex]->getFeatureVector();
                }else{
                   okToAddProcessedData = false;
                   break;
                }
            }
        }

        if( okToAddProcessedData ){
            //Add the training sample to the processed training data
            processedTrainingData.addSample(classLabel, trainingSample);
        }
        
    }
    
    if( processedTrainingData.getNumSamples() != trainingData.getNumSamples() ){
        warningLog << "train(LabelledClassificationData &trainingData) - Lost " << trainingData.getNumSamples()-processedTrainingData.getNumSamples() << " of " << trainingData.getNumSamples() << " training samples due to the processing stage!" << endl;
    }
    
    //Train the classifier
    trained = classifier->train( processedTrainingData );
    if( !trained ){
        errorLog << "train(LabelledClassificationData &trainingData) - Failed To Train Classifier" << endl;
        return false;
    }
    
    return true;
}
    
bool GestureRecognitionPipeline::train(LabelledClassificationData &trainingData, UINT kFoldValue, bool useStratifiedSampling){
    
    trained = false;
    testAccuracy = 0;
    testRMSError = 0;
    testFMeasure.clear();
    testPrecision.clear();
    testRecall.clear();
    testRejectionPrecision = 0;
    testRejectionRecall = 0;
    
    if( !getIsClassifierSet() ){
        errorLog << "train(LabelledClassificationData &trainingData,UINT kFoldValue, bool useStratifiedSampling) - Failed To Train Classifier, the classifier has not been set!" << endl;
        return false;
    }
    
    if( trainingData.getNumSamples() == 0 ){
        errorLog << "train(LabelledClassificationData &trainingData,UINT kFoldValue, bool useStratifiedSampling) - Failed To Train Classifier, there is no training data!" << endl;
        return false;
    }

    //Spilt the data into K folds
    bool spiltResult = trainingData.spiltDataIntoKFolds(kFoldValue, useStratifiedSampling);
    
    if( !spiltResult ){
        return false;
    }
    
    //Run the k-fold training and testing
    double crossValidationAccuracy = 0;
    bool trainingResult = false;
    bool testingResult = false;
    LabelledClassificationData foldTrainingData;
    LabelledClassificationData foldTestData;
    for(UINT k=0; k<kFoldValue; k++){
        ///Train the classification system
        foldTrainingData = trainingData.getTrainingFoldData(k);
        trainingResult = train( foldTrainingData );
        
        if( !trainingResult ){
            return false;
        }
        
        //Test the classification system
        foldTestData = trainingData.getTestFoldData(k);
        testingResult = test( foldTestData );
        
        if( !testingResult ){
            return false;
        }
        
        crossValidationAccuracy += getTestAccuracy();
    }
    
    //Set the accuracy of the classification system averaged over the kfolds
    testAccuracy = crossValidationAccuracy / double(kFoldValue);
    
    return true;
}

bool GestureRecognitionPipeline::train(LabelledTimeSeriesClassificationData &trainingData){
    
    trained = false;
    testAccuracy = 0;
    testRMSError = 0;
    testFMeasure.clear();
    testPrecision.clear();
    testRecall.clear();
    testRejectionPrecision = 0;
    testRejectionRecall = 0;
    
    if( !getIsClassifierSet() ){
        errorLog << "train(LabelledTimeSeriesClassificationData &trainingData) - Failed To Train Classifier, the classifier has not been set!" << endl;
        return false;
    }
    
    if( trainingData.getNumSamples() == 0 ){
        errorLog << "train(LabelledTimeSeriesClassificationData &trainingData) - Failed To Train Classifier, there is no training data!" << endl;
        return false;
    }
    
    //Set the input vector dimension size of the pipeline
    inputVectorDimensions = trainingData.getNumDimensions();
    
    LabelledTimeSeriesClassificationData processedTrainingData( trainingData.getNumDimensions() );
    LabelledTimeSeriesClassificationData labelledTimeseriesClassificationData;
    LabelledClassificationData labelledClassificationData;
    
    if( classifier->getTimeseriesCompatible() ){
        UINT trainingDataInputDimensionSize = trainingData.getNumDimensions();
        if( getIsFeatureExtractionSet() ){
            trainingDataInputDimensionSize = featureExtractionModules[ featureExtractionModules.size()-1 ]->getNumOutputDimensions();
        }
        labelledTimeseriesClassificationData.setNumDimensions( trainingDataInputDimensionSize );
    }else{
        UINT trainingDataInputDimensionSize = trainingData.getNumDimensions();
        if( getIsFeatureExtractionSet() ){
            trainingDataInputDimensionSize = featureExtractionModules[ featureExtractionModules.size()-1 ]->getNumOutputDimensions();
        }
        labelledClassificationData.setNumDimensions( trainingDataInputDimensionSize );
    }
    
    //Pass the timeseries data through any pre-processing modules and add it to the processedTrainingData structure
    for(UINT i=0; i<trainingData.getNumSamples(); i++){
        UINT classLabel = trainingData[i].getClassLabel();
        Matrix< double > trainingSample = trainingData[i].getData();
        
        if( getIsPreProcessingSet() ){
            
            //Try to process the matrix data row-by-row
            bool resetPreprocessingModule = true;
            for(UINT r=0; r<trainingSample.getNumRows(); r++){
                vector< double > sample = trainingSample.getRowVector( r );
                
                for(UINT moduleIndex=0; moduleIndex<preProcessingModules.size(); moduleIndex++){
                    
                    if( resetPreprocessingModule ){
                        preProcessingModules[moduleIndex]->reset();
                    }
                    
                    //Validate the input and output dimensions match!
                    if( preProcessingModules[moduleIndex]->getNumInputDimensions() != preProcessingModules[moduleIndex]->getNumOutputDimensions() ){
                        errorLog << "train(LabelledTimeSeriesClassificationData &trainingData) - Failed To PreProcess Training Data. The number of inputDimensions (" << preProcessingModules[moduleIndex]->getNumInputDimensions() << ") in  PreProcessingModule " << moduleIndex << " do not match the number of outputDimensions (" << preProcessingModules[moduleIndex]->getNumOutputDimensions() << ")" << endl;
                        return false;
                    }
                    
                    if( !preProcessingModules[moduleIndex]->process( sample ) ){
                        errorLog << "train(LabelledTimeSeriesClassificationData &trainingData) - Failed To PreProcess Training Data. PreProcessingModuleIndex: " << moduleIndex << endl;
                        return false;
                    }
                    sample = preProcessingModules[moduleIndex]->getProcessedData();
                }
                
                //The preprocessing modules should only be reset when r==0
                resetPreprocessingModule = false;
                
                //Overwrite the original training sample with the preProcessed sample
                for(UINT c=0; c<sample.size(); c++){
                    trainingSample[r][c] = sample[c];
                }
            }
            
        }
        
        //Add the training sample to the processed training data
        processedTrainingData.addSample(classLabel,trainingSample);
    }
    
    //Loop over the processed training data, perfrom any feature extraction if needed
    //Add the data to either the timeseries or classification data structures
    for(UINT i=0; i<processedTrainingData.getNumSamples(); i++){
        UINT classLabel = processedTrainingData[i].getClassLabel();
        Matrix< double > trainingSample = processedTrainingData[i].getData();
        bool featureDataReady = false;
        bool resetFeatureExtractionModules = true;
        
        Matrix< double > featureData;
        //Try to process the matrix data row-by-row
        for(UINT r=0; r<trainingSample.getNumRows(); r++){
            vector< double > inputVector = trainingSample.getRowVector( r );
            featureDataReady = true;
            
             //Pass the processed training data through the feature extraction
            if( getIsFeatureExtractionSet() ){
            
                for(UINT moduleIndex=0; moduleIndex<featureExtractionModules.size(); moduleIndex++){
                    
                    if( resetFeatureExtractionModules ){
                        featureExtractionModules[moduleIndex]->reset();
                    }
                    
                    if( !featureExtractionModules[moduleIndex]->computeFeatures( inputVector ) ){
                        errorLog << "train(LabelledTimeSeriesClassificationData &trainingData) - Failed To Compute Features For Training Data. FeatureExtractionModuleIndex: " << moduleIndex << endl;
                        return false;
                    }
                    
                    //Overwrite the input vector with the features so this can either be input to the next feature module 
                    //or converted to the LabelledClassificationData format
                    inputVector = featureExtractionModules[moduleIndex]->getFeatureVector();
                    featureDataReady = featureExtractionModules[moduleIndex]->getFeatureDataReady();
                }
                
                //The feature extraction modules should only be reset on r == 0
                resetFeatureExtractionModules = false;
                
                if( featureDataReady ){
                    
                    if( classifier->getTimeseriesCompatible() ){
                        if( !featureData.push_back( inputVector ) ){
                            errorLog << "train(LabelledTimeSeriesClassificationData &trainingData) - Failed To add feature vector to feature data matrix! FeatureExtractionModuleIndex: " << endl;
                            return false;
                        }
                    }else labelledClassificationData.addSample(classLabel, inputVector);
                }
                
            }else{
                if( classifier->getTimeseriesCompatible() ){
                    if( !featureData.push_back( inputVector ) ){
                        errorLog << "train(LabelledTimeSeriesClassificationData &trainingData) - Failed To add feature vector to feature data matrix! FeatureExtractionModuleIndex: " << endl;
                        return false;
                    }
                }
                else labelledClassificationData.addSample(classLabel, inputVector);
            }
        }
        
        if( classifier->getTimeseriesCompatible() ) labelledTimeseriesClassificationData.addSample(classLabel, featureData);
        
    }
        
    //Train the classification system
    if( classifier->getTimeseriesCompatible() ) trained =  classifier->train( labelledTimeseriesClassificationData );
    else trained =  classifier->train( labelledClassificationData );

    if( !trained ){
        errorLog << "train(LabelledTimeSeriesClassificationData &trainingData) - Failed To Train Classifier" << endl;
        return false;
    }
    return true;
}
    
bool GestureRecognitionPipeline::train(LabelledRegressionData &trainingData){
    
    trained = false;
    testAccuracy = 0;
    testRMSError = 0;
    testFMeasure.clear();
    testPrecision.clear();
    testRecall.clear();
    testRejectionPrecision = 0;
    testRejectionRecall = 0;
    
    //Set the input vector dimension size
    inputVectorDimensions = trainingData.getNumInputDimensions();
    
    //Pass the training data through any pre-processing or feature extraction units
    LabelledRegressionData processedTrainingData;
    
    //Set the dimensionality of the data
    UINT numInputs = 0;
    UINT numTargets = trainingData.getNumTargetDimensions();
    if( !getIsPreProcessingSet() && !getIsFeatureExtractionSet() ){
        numInputs = trainingData.getNumInputDimensions();
    }else{
        
        if( getIsPreProcessingSet() && !getIsFeatureExtractionSet() ){
            numInputs = preProcessingModules[ preProcessingModules.size()-1 ]->getNumOutputDimensions();
        }
        
        if( !getIsPreProcessingSet() && getIsFeatureExtractionSet() ){
            numInputs = featureExtractionModules[ featureExtractionModules.size()-1 ]->getNumOutputDimensions();
        }
        
        if( getIsPreProcessingSet() && getIsFeatureExtractionSet() ){
            numInputs = featureExtractionModules[ featureExtractionModules.size()-1 ]->getNumOutputDimensions();
        }
    }
    
    processedTrainingData.setInputAndTargetDimensions(numInputs, numTargets);
    
    for(UINT i=0; i<trainingData.getNumSamples(); i++){
        vector< double > inputVector = trainingData[i].getInputVector();
        vector< double > targetVector = trainingData[i].getTargetVector();
        
        if( getIsPreProcessingSet() ){
            for(UINT moduleIndex=0; moduleIndex<preProcessingModules.size(); moduleIndex++){
                if( !preProcessingModules[ moduleIndex ]->process( inputVector ) ){
                    errorLog << "train(LabelledRegressionData &trainingData) - Failed To Compute Features For Training Data. PreProcessingModuleIndex: " << moduleIndex << endl;
                    return false;
                }
                
                inputVector = preProcessingModules[ moduleIndex ]->getProcessedData();
            }
        }
        
        if( getIsFeatureExtractionSet() ){
            for(UINT moduleIndex=0; moduleIndex<featureExtractionModules.size(); moduleIndex++){
                if( !featureExtractionModules[ moduleIndex ]->computeFeatures( inputVector ) ){
                    errorLog << "train(LabelledRegressionData &trainingData) - Failed To Compute Features For Training Data. FeatureExtractionModuleIndex: " << moduleIndex << endl;
                    return false;
                }
                
                inputVector = featureExtractionModules[ moduleIndex ]->getFeatureVector();
            }
        }
        
        //Add the training sample to the processed training data
        if( !processedTrainingData.addSample(inputVector,targetVector) ){
            errorLog << "train(LabelledRegressionData &trainingData) - Failed to add processed training sample to training data" << endl;
            return false;
        }
    }
    
    //Train the classification system
    if( getIsRegressifierSet() ){
        trained =  regressifier->train( processedTrainingData );
        if( !trained ){
            errorLog << "train(LabelledRegressionData &trainingData) - Failed To Train Classifier" << endl;
            return false;
        }
    }else{
        errorLog << "train(LabelledRegressionData &trainingData) - Classifier is not set" << endl;
        return false;
    }
    
    trained = true;
    return true;
}
    
bool GestureRecognitionPipeline::test(LabelledClassificationData &testData){
    
    testAccuracy = 0;
    testRMSError = 0;
    testFMeasure.clear();
    testPrecision.clear();
    testRecall.clear();
    testRejectionPrecision = 0;
    testRejectionRecall = 0;
    testConfusionMatrix.clear();
    
    //Make sure the classification model has been trained
    if( !trained ){
        errorLog << "test(LabelledClassificationData &testData) - Classifier is not trained" << endl;
        return false;
    }
    
    //Make sure the dimensionality of the test data matches the input vector's dimensions
    if( testData.getNumDimensions() != inputVectorDimensions ){
        errorLog << "test(LabelledClassificationData &testData) - The dimensionality of the test data (" << testData.getNumDimensions() << ") does not match that of the input vector dimensions of the pipeline (" << inputVectorDimensions << ")" << endl;
        return false;
    }
    
    if( !getIsClassifierSet() ){
        errorLog << "test(LabelledClassificationData &testData) - The classifier has not been set" << endl;
        return false;
    }

	//Validate that the class labels in the test data match the class labels in the model
	bool classLabelValidationPassed = true;
	for(UINT i=0; i<testData.getNumClasses(); i++){
		bool labelFound = false;
		for(UINT k=0; k<classifier->getNumClasses(); k++){
			if( testData.getClassTracker()[i].classLabel == classifier->getClassLabels()[k] ){
				labelFound = true;
				break;
			}
		}

		if( !labelFound ){
			classLabelValidationPassed = false;
			errorLog << "test(LabelledClassificationData &testData) - The test dataset contains a class label (" << testData.getClassTracker()[i].classLabel << ") that is not in the model!" << endl;
		}
	}

	if( !classLabelValidationPassed ){
        errorLog << "test(LabelledClassificationData &testData) -  Model Class Labels: ";
        for(UINT k=0; k<classifier->getNumClasses(); k++){
			errorLog << classifier->getClassLabels()[k] << "\t";
		}
        errorLog << endl;
        return false;
    }

    double rejectionPrecisionCounter = 0;
    double rejectionRecallCounter = 0;
    unsigned int confusionMatrixSize = classifier->getNullRejectionEnabled() ? classifier->getNumClasses()+1 : classifier->getNumClasses();
    vector< double > precisionCounter(classifier->getNumClasses(), 0);
    vector< double > recallCounter(classifier->getNumClasses(), 0);
    vector< double > confusionMatrixCounter(confusionMatrixSize,0);
    
    //Resize the test matrix
    testConfusionMatrix.resize(confusionMatrixSize, confusionMatrixSize);
    testConfusionMatrix.setAllValues(0);
    
    //Resize the precision and recall vectors
    testPrecision.clear();
    testRecall.clear();
    testFMeasure.clear();
    testPrecision.resize(getNumClassesInModel(), 0);
    testRecall.resize(getNumClassesInModel(), 0);
    testFMeasure.resize(getNumClassesInModel(), 0);

    //Run the test
    for(UINT i=0; i<testData.getNumSamples(); i++){
        UINT classLabel = testData[i].getClassLabel();
        vector< double > testSample = testData[i].getSample();
        
        //Pass the test sample through the pipeline
        if( !predict( testSample ) ){
            errorLog << "test(LabelledClassificationData &testData) - Prediction failed for test sample at index: " << i << endl;
            return false;
        }
        
        //Update the test metrics
        UINT predictedClassLabel = getPredictedClassLabel();
        
        if( !updateTestMetrics(classLabel,predictedClassLabel,precisionCounter,recallCounter,rejectionPrecisionCounter,rejectionRecallCounter, confusionMatrixCounter) ){
            errorLog << "test(LabelledClassificationData &testData) - Failed to update test metrics at test sample index: " << i << endl;
            return false;
        }
    }
    
    if( !computeTestMetrics(precisionCounter,recallCounter,rejectionPrecisionCounter,rejectionRecallCounter, confusionMatrixCounter, testData.getNumSamples()) ){
        errorLog << "test(LabelledClassificationData &testData) - Failed to compute test metrics !" << endl;
        return false;
    }
    
    return true;
}
    
bool GestureRecognitionPipeline::test(LabelledTimeSeriesClassificationData &testData){
    
    testAccuracy = 0;
    testFMeasure.clear();
    testPrecision.clear();
    testRecall.clear();
    testRejectionPrecision = 0;
    testRejectionRecall = 0;
    testConfusionMatrix.clear();
    
    //Make sure the classification model has been trained
    if( !trained ){
        errorLog << "test(LabelledTimeSeriesClassificationData &testData) - The classifier has not been trained" << endl;
        return false;
    }
    
    //Make sure the dimensionality of the test data matches the input vector's dimensions
    if( testData.getNumDimensions() != inputVectorDimensions ){
        errorLog << "test(LabelledTimeSeriesClassificationData &testData) - The dimensionality of the test data (" << testData.getNumDimensions() << ") does not match that of the input vector dimensions of the pipeline (" << inputVectorDimensions << ")" << endl;
        return false;
    }
    
    if( !getIsClassifierSet() ){
        errorLog << "test(LabelledTimeSeriesClassificationData &testData) - The classifier has not been set" << endl;
        return false;
    }
    
    double rejectionPrecisionCounter = 0;
    double rejectionRecallCounter = 0;
    unsigned int confusionMatrixSize = classifier->getNullRejectionEnabled() ? classifier->getNumClasses()+1 : classifier->getNumClasses();
    vector< double > precisionCounter(classifier->getNumClasses(), 0);
    vector< double > recallCounter(classifier->getNumClasses(), 0);
    vector< double > confusionMatrixCounter(confusionMatrixSize,0);
    
    //Resize the test matrix
    testConfusionMatrix.resize(confusionMatrixSize,confusionMatrixSize);
    testConfusionMatrix.setAllValues(0);
    
    //Resize the precision and recall vectors
    testPrecision.resize(getNumClassesInModel(), 0);
    testRecall.resize(getNumClassesInModel(), 0);
    testFMeasure.resize(getNumClassesInModel(), 0);
    
    //Run the test
    for(UINT i=0; i<testData.getNumSamples(); i++){
        UINT classLabel = testData[i].getClassLabel();
        Matrix< double > testMatrix = testData[i].getData();
        
        for(UINT x=0; x<testMatrix.getNumRows(); x++){
            vector< double > testSample = testMatrix.getRowVector( x );
            
            //Pass the test sample through the pipeline
            if( !predict( testSample ) ){
                return false;
            }
            
            //Update the test metrics
            UINT predictedClassLabel = getPredictedClassLabel();
            
            if( !updateTestMetrics(classLabel,predictedClassLabel,precisionCounter,recallCounter,rejectionPrecisionCounter,rejectionRecallCounter, confusionMatrixCounter) ){
                errorLog << "test(LabelledTimeSeriesClassificationData &testData) - Failed to update test metrics at test sample index: " << i << endl;
                return false;
            }
        }
    }
        
    if( !computeTestMetrics(precisionCounter,recallCounter,rejectionPrecisionCounter,rejectionRecallCounter, confusionMatrixCounter, testData.getNumSamples()) ){
        errorLog << "test(LabelledTimeSeriesClassificationData &testData) - Failed to compute test metrics !" << endl;
        return false;
    }
    
    return true;
}
    
bool GestureRecognitionPipeline::test(LabelledContinuousTimeSeriesClassificationData &testData){
    
    testAccuracy = 0;
    testRMSError = 0;
    testFMeasure.clear();
    testPrecision.clear();
    testRecall.clear();
    testRejectionPrecision = 0;
    testRejectionRecall = 0;
    testConfusionMatrix.clear();
    
    //Make sure the classification model has been trained
    if( !trained ){
        errorLog << "test(LabelledContinuousTimeSeriesClassificationData &testData) - The classifier has not been trained" << endl;
        return false;
    }
    
    //Make sure the dimensionality of the test data matches the input vector's dimensions
    if( testData.getNumDimensions() != inputVectorDimensions ){
        errorLog << "test(LabelledContinuousTimeSeriesClassificationData &testData) - The dimensionality of the test data (" << testData.getNumDimensions() << ") does not match that of the input vector dimensions of the pipeline (" << inputVectorDimensions << ")" << endl;
        return false;
    }
    
    if( !getIsClassifierSet() ){
        errorLog << "test(LabelledContinuousTimeSeriesClassificationData &testData) - The classifier has not been set" << endl;
        return false;
    }
    
    double rejectionPrecisionCounter = 0;
    double rejectionRecallCounter = 0;
    unsigned int confusionMatrixSize = classifier->getNullRejectionEnabled() ? classifier->getNumClasses()+1 : classifier->getNumClasses();
    vector< double > precisionCounter(getNumClassesInModel(), 0);
    vector< double > recallCounter(getNumClassesInModel(), 0);
    vector< double > confusionMatrixCounter(confusionMatrixSize,0);
    
    //Resize the test matrix
    testConfusionMatrix.resize(confusionMatrixSize,confusionMatrixSize);
    testConfusionMatrix.setAllValues(0);
    
    //Resize the precision and recall vectors
    testPrecision.resize(getNumClassesInModel(), 0);
    testRecall.resize(getNumClassesInModel(), 0);
    testFMeasure.resize(getNumClassesInModel()), 0;
    
    //Run the test
    testData.resetPlaybackIndex(0); //Make sure that the test data start at 0
    for(UINT i=0; i<testData.getNumSamples(); i++){
        LabelledClassificationSample sample = testData.getNextSample();
        UINT classLabel = sample.getClassLabel();
        vector< double > testSample = sample.getSample();
            
        //Pass the test sample through the pipeline
        if( !predict( testSample ) ){
            errorLog << "test(LabelledContinuousTimeSeriesClassificationData &testData) - Prediction Failed!" << endl;
            return false;
        }
        
        //Update the test metrics
        UINT predictedClassLabel = getPredictedClassLabel();
        
        
        if( !updateTestMetrics(classLabel,predictedClassLabel,precisionCounter,recallCounter,rejectionPrecisionCounter,rejectionRecallCounter, confusionMatrixCounter) ){
            errorLog << "test(LabelledContinuousTimeSeriesClassificationData &testData) - Failed to update test metrics at test sample index: " << i << endl;
            return false;
        }
    }
    
    if( !computeTestMetrics(precisionCounter,recallCounter,rejectionPrecisionCounter,rejectionRecallCounter, confusionMatrixCounter, testData.getNumSamples()) ){
        errorLog << "test(LabelledContinuousTimeSeriesClassificationData &testData) - Failed to compute test metrics !" << endl;
        return false;
    }
    
    return true;
}
    
bool GestureRecognitionPipeline::test(LabelledRegressionData &testData){
    
    testAccuracy = 0;
    testRMSError = 0;
    testFMeasure.clear();
    testPrecision.clear();
    testRecall.clear();
    testRejectionPrecision = 0;
    testRejectionRecall = 0;
    testConfusionMatrix.clear();
    
    //Make sure the classification model has been trained
    if( !trained ){
        errorLog << "test(LabelledRegressionData &testData) - Regressifier is not trained" << endl;
        return false;
    }
    
    //Make sure the dimensionality of the test data matches the input vector's dimensions
    if( testData.getNumInputDimensions() != inputVectorDimensions ){
        errorLog << "test(LabelledRegressionData &testData) - The dimensionality of the test data (" << testData.getNumInputDimensions() << ") does not match that of the input vector dimensions of the pipeline (" << inputVectorDimensions << ")" << endl;
        return false;
    }
    
    if( !getIsRegressifierSet() ){
        errorLog << "test(LabelledRegressionData &testData) - The regressifier has not been set" << endl;
        return false;
    }
    
    
    if( regressifier->getNumOutputDimensions() != testData.getNumTargetDimensions() ){
        errorLog << "test(LabelledRegressionData &testData) - The size of the output of the regressifier (" << regressifier->getNumOutputDimensions() << ") does not match that of the size of the number of target dimensions (" << testData.getNumTargetDimensions() << ")" << endl;
        return false;
    }
    
    //Run the test
    for(UINT i=0; i<testData.getNumSamples(); i++){
        vector< double > inputVector = testData[i].getInputVector();
        vector< double > targetVector = testData[i].getTargetVector();
        
        //Pass the test sample through the pipeline
        if( !map( inputVector ) ){
            errorLog << "test(LabelledRegressionData &testData) - Failed to map input vector!" << endl;
            return false;
        }
        
        //Update the RMS error
        double sum = 0;
        vector< double > regressionData = regressifier->getRegressionData();
        for(UINT j=0; j<targetVector.size(); j++){
            sum += SQR( regressionData[j]-targetVector[j] );
        }
        testRMSError += sum;
    }
    
    //Compute the test metrics
    testRMSError = sqrt( testRMSError / double( testData.getNumSamples() ) );
    
    return true;
}

bool GestureRecognitionPipeline::predict(vector< double > inputVector){
    
    predictedClassLabel = 0;
    
    //Make sure the classification model has been trained
    if( !trained ){
        errorLog << "predict(vector< double > inputVector) - The classifier has not been trained" << endl;
        return false;
    }
    
    //Make sure the dimensionality of the input vector matches the inputVectorDimensions
    if( inputVector.size() != inputVectorDimensions ){
        errorLog << "predict(vector< double > inputVector) - The dimensionality of the input vector (" << inputVector.size() << ") does not match that of the input vector dimensions of the pipeline (" << inputVectorDimensions << ")" << endl;
        return false;
    }
    
    if( !getIsClassifierSet() ){
        errorLog << "predict(vector< double > inputVector) - Classifier is not set" << endl;
        return false;
    }
    
    //Update the context module
    predictionModuleIndex = START_OF_PIPELINE;
    if( contextModules[ START_OF_PIPELINE ].size() ){
        for(UINT moduleIndex=0; moduleIndex<contextModules[ START_OF_PIPELINE ].size(); moduleIndex++){
            if( !contextModules[ START_OF_PIPELINE ][moduleIndex]->process( inputVector ) ){
                errorLog << "predict(vector< double > inputVector) - Context Module Failed at START_OF_PIPELINE. ModuleIndex: " << moduleIndex << endl;
                return false;
            }
            if( !contextModules[ START_OF_PIPELINE ][moduleIndex]->getOK() ){
                return true;
            }
            inputVector = contextModules[ START_OF_PIPELINE ][moduleIndex]->getProcessedData();
        }
    }
    
    //Perform any pre-processing
    if( getIsPreProcessingSet() ){
        for(UINT moduleIndex=0; moduleIndex<preProcessingModules.size(); moduleIndex++){
            if( !preProcessingModules[moduleIndex]->process( inputVector ) ){
                errorLog << "predict(vector< double > inputVector) - Failed to PreProcess Input Vector. PreProcessingModuleIndex: " << moduleIndex << endl;
                return false;
            }
            inputVector = preProcessingModules[moduleIndex]->getProcessedData();
        }
    }
    
    //Update the context module
    predictionModuleIndex = AFTER_PREPROCESSING;
    if( contextModules[ AFTER_PREPROCESSING ].size() ){
        for(UINT moduleIndex=0; moduleIndex<contextModules[ AFTER_PREPROCESSING ].size(); moduleIndex++){
            if( !contextModules[ AFTER_PREPROCESSING ][moduleIndex]->process( inputVector ) ){
                errorLog << "predict(vector< double > inputVector) - Context Module Failed at AFTER_PREPROCESSING. ModuleIndex: " << moduleIndex << endl;
                return false;
            }
            if( !contextModules[ AFTER_PREPROCESSING ][moduleIndex]->getOK() ){
                predictionModuleIndex = AFTER_PREPROCESSING;
                return false;
            }
            inputVector = contextModules[ AFTER_PREPROCESSING ][moduleIndex]->getProcessedData();
        }
    }
    
    //Perform any feature extraction
    if( getIsFeatureExtractionSet() ){
        for(UINT moduleIndex=0; moduleIndex<featureExtractionModules.size(); moduleIndex++){
            if( !featureExtractionModules[moduleIndex]->computeFeatures( inputVector ) ){
                errorLog << "predict(vector< double > inputVector) - Failed to compute features from data. FeatureExtractionModuleIndex: " << moduleIndex << endl;
                return false;
            }
            inputVector = featureExtractionModules[moduleIndex]->getFeatureVector();
        }
    }
    
    //Update the context module
    predictionModuleIndex = AFTER_FEATURE_EXTRACTION;
    if( contextModules[ AFTER_FEATURE_EXTRACTION ].size() ){
        for(UINT moduleIndex=0; moduleIndex<contextModules[ AFTER_FEATURE_EXTRACTION ].size(); moduleIndex++){
            if( !contextModules[ AFTER_FEATURE_EXTRACTION ][moduleIndex]->process( inputVector ) ){
                errorLog << "predict(vector< double > inputVector) - Context Module Failed at AFTER_FEATURE_EXTRACTION. ModuleIndex: " << moduleIndex << endl;
                return false;
            }
            if( !contextModules[ AFTER_FEATURE_EXTRACTION ][moduleIndex]->getOK() ){
                predictionModuleIndex = AFTER_FEATURE_EXTRACTION;
                return false;
            }
            inputVector = contextModules[ AFTER_FEATURE_EXTRACTION ][moduleIndex]->getProcessedData();
        }
    }
    
    //Perform the classification
    if( !classifier->predict(inputVector) ){
        errorLog << "predict(vector< double > inputVector) - Prediction Failed" << endl;
        return false;
    }
    predictedClassLabel = classifier->getPredictedClassLabel();
    
    //Update the context module
    if( contextModules[ AFTER_CLASSIFIER ].size() ){
        for(UINT moduleIndex=0; moduleIndex<contextModules[ AFTER_CLASSIFIER ].size(); moduleIndex++){
            if( !contextModules[ AFTER_CLASSIFIER ][moduleIndex]->process( vector<double>(1,predictedClassLabel) ) ){
                errorLog << "predict(vector< double > inputVector) - Context Module Failed at AFTER_CLASSIFIER. ModuleIndex: " << moduleIndex << endl;
                return false;
            }
            if( !contextModules[ AFTER_CLASSIFIER ][moduleIndex]->getOK() ){
                predictionModuleIndex = AFTER_CLASSIFIER;
                return false;
            }
            predictedClassLabel = (UINT)contextModules[ AFTER_CLASSIFIER ][moduleIndex]->getProcessedData()[0];
        }
    }
    
    //Perform any post processing
    predictionModuleIndex = AFTER_CLASSIFIER;
    if( getIsPostProcessingSet() ){
        
        if( pipelineMode != CLASSIFICATION_MODE){
            errorLog << "predict(vector< double > inputVector) - Pipeline Mode Is Not in CLASSIFICATION_MODE!" << endl;
            return false;
        }
        
        vector< double > data;
        for(UINT moduleIndex=0; moduleIndex<postProcessingModules.size(); moduleIndex++){
            
            //Select which input we should give the postprocessing module
            if( postProcessingModules[moduleIndex]->getIsPostProcessingInputModePredictedClassLabel() ){
                //Set the input
                data.resize(1);
                data[0] = predictedClassLabel;
                
                //Verify that the input size is OK
                if( data.size() != postProcessingModules[moduleIndex]->getNumInputDimensions() ){
                    errorLog << "predict(vector< double > inputVector) - The size of the data vector (" << data.size() << ") does not match that of the postProcessingModule (" << postProcessingModules[moduleIndex]->getNumInputDimensions() << ") at the moduleIndex: " << moduleIndex <<endl;
                    return false;
                }
                
                //Postprocess the data
                if( !postProcessingModules[moduleIndex]->process( data ) ){
                    errorLog << "predict(vector< double > inputVector) - Failed to post process data. PostProcessing moduleIndex: " << moduleIndex <<endl;
                    return false;
                }
                
                //Select which output we should update
                data = postProcessingModules[moduleIndex]->getProcessedData();  
            }
            
            //Select which output we should update
            if( postProcessingModules[moduleIndex]->getIsPostProcessingOutputModePredictedClassLabel() ){
                //Get the processed predicted class label
                data = postProcessingModules[moduleIndex]->getProcessedData(); 
                
                //Verify that the output size is OK
                if( data.size() != 1 ){
                    errorLog << "predict(vector< double > inputVector) - The size of the processed data vector (" << data.size() << ") from postProcessingModule at the moduleIndex: " << moduleIndex << " is not equal to 1 even though it is in OutputModePredictedClassLabel!" << endl;
                    return false;
                }
                
                //Update the predicted class label
                predictedClassLabel = (UINT)data[0];
            }
                  
        }
    } 
    
    //Update the context module
    predictionModuleIndex = END_OF_PIPELINE;
    if( contextModules[ END_OF_PIPELINE ].size() ){
        for(UINT moduleIndex=0; moduleIndex<contextModules[ END_OF_PIPELINE ].size(); moduleIndex++){
            if( !contextModules[ END_OF_PIPELINE ][moduleIndex]->process( vector<double>(1,predictedClassLabel) ) ){
                errorLog << "predict(vector< double > inputVector) - Context Module Failed at END_OF_PIPELINE. ModuleIndex: " << moduleIndex << endl;
                return false;
            }
            if( !contextModules[ END_OF_PIPELINE ][moduleIndex]->getOK() ){
                predictionModuleIndex = END_OF_PIPELINE;
                return false;
            }
            predictedClassLabel = (UINT)contextModules[ END_OF_PIPELINE ][moduleIndex]->getProcessedData()[0];
        }
    }
    
    return true;
}
    
bool GestureRecognitionPipeline::map(vector< double > inputVector){
    
    predictedClassLabel = 0;
    
    //Make sure the regression model has been trained
    if( !trained ){
        errorLog << "map(vector< double > inputVector) - The regressifier has not been trained" << endl;
        return false;
    }
    
    //Make sure the dimensionality of the input vector matches the inputVectorDimensions
    if( inputVector.size() != inputVectorDimensions ){
        errorLog << "map(vector< double > inputVector) - The dimensionality of the input vector (" << inputVector.size() << ") does not match that of the input vector dimensions of the pipeline (" << inputVectorDimensions << ")" << endl;
        return false;
    }
    
    if( !getIsRegressifierSet() ){
        errorLog << "map(vector< double > inputVector) - Regressifier is not set" << endl;
        return false;
    }
    
    //Update the context module
    predictionModuleIndex = START_OF_PIPELINE;
    if( contextModules[ START_OF_PIPELINE ].size() ){
        for(UINT moduleIndex=0; moduleIndex<contextModules[ START_OF_PIPELINE ].size(); moduleIndex++){
            if( !contextModules[ START_OF_PIPELINE ][moduleIndex]->process( inputVector ) ){
                errorLog << "map(vector< double > inputVector) - Context Module Failed at START_OF_PIPELINE. ModuleIndex: " << moduleIndex << endl;
                return false;
            }
            if( !contextModules[ START_OF_PIPELINE ][moduleIndex]->getOK() ){
                return true;
            }
            inputVector = contextModules[ START_OF_PIPELINE ][moduleIndex]->getProcessedData();
        }
    }
    
    //Perform any pre-processing
    if( getIsPreProcessingSet() ){
        for(UINT moduleIndex=0; moduleIndex<preProcessingModules.size(); moduleIndex++){
            if( !preProcessingModules[moduleIndex]->process( inputVector ) ){
                errorLog << "map(vector< double > inputVector) - Failed to PreProcess Input Vector. PreProcessingModuleIndex: " << moduleIndex << endl;
                return false;
            }
            inputVector = preProcessingModules[moduleIndex]->getProcessedData();
        }
    }
    
    //Update the context module
    predictionModuleIndex = AFTER_PREPROCESSING;
    if( contextModules[ AFTER_PREPROCESSING ].size() ){
        for(UINT moduleIndex=0; moduleIndex<contextModules[ AFTER_PREPROCESSING ].size(); moduleIndex++){
            if( !contextModules[ AFTER_PREPROCESSING ][moduleIndex]->process( inputVector ) ){
                errorLog << "map(vector< double > inputVector) - Context Module Failed at AFTER_PREPROCESSING. ModuleIndex: " << moduleIndex << endl;
                return false;
            }
            if( !contextModules[ AFTER_PREPROCESSING ][moduleIndex]->getOK() ){
                predictionModuleIndex = AFTER_PREPROCESSING;
                return false;
            }
            inputVector = contextModules[ AFTER_PREPROCESSING ][moduleIndex]->getProcessedData();
        }
    }
    
    //Perform any feature extraction
    if( getIsFeatureExtractionSet() ){
        for(UINT moduleIndex=0; moduleIndex<featureExtractionModules.size(); moduleIndex++){
            if( !featureExtractionModules[moduleIndex]->computeFeatures( inputVector ) ){
                errorLog << "map(vector< double > inputVector) - Failed to compute features from data. FeatureExtractionModuleIndex: " << moduleIndex << endl;
                return false;
            }
            inputVector = featureExtractionModules[moduleIndex]->getFeatureVector();
        }
    }
    
    //Update the context module
    predictionModuleIndex = AFTER_FEATURE_EXTRACTION;
    if( contextModules[ AFTER_FEATURE_EXTRACTION ].size() ){
        for(UINT moduleIndex=0; moduleIndex<contextModules[ AFTER_FEATURE_EXTRACTION ].size(); moduleIndex++){
            if( !contextModules[ AFTER_FEATURE_EXTRACTION ][moduleIndex]->process( inputVector ) ){
                errorLog << "map(vector< double > inputVector) - Context Module Failed at AFTER_FEATURE_EXTRACTION. ModuleIndex: " << moduleIndex << endl;
                return false;
            }
            if( !contextModules[ AFTER_FEATURE_EXTRACTION ][moduleIndex]->getOK() ){
                predictionModuleIndex = AFTER_FEATURE_EXTRACTION;
                return false;
            }
            inputVector = contextModules[ AFTER_FEATURE_EXTRACTION ][moduleIndex]->getProcessedData();
        }
    }
    
    //Perform the regression
    if( !regressifier->predict(inputVector) ){
            errorLog << "map(vector< double > inputVector) - Prediction Failed" << endl;
            return false;
    }
    regressionData = regressifier->getRegressionData();
    
    //Update the context module
    if( contextModules[ AFTER_CLASSIFIER ].size() ){
        for(UINT moduleIndex=0; moduleIndex<contextModules[ AFTER_CLASSIFIER ].size(); moduleIndex++){
            if( !contextModules[ AFTER_CLASSIFIER ][moduleIndex]->process( regressionData ) ){
                errorLog << "map(vector< double > inputVector) - Context Module Failed at AFTER_CLASSIFIER. ModuleIndex: " << moduleIndex << endl;
                return false;
            }
            if( !contextModules[ AFTER_CLASSIFIER ][moduleIndex]->getOK() ){
                predictionModuleIndex = AFTER_CLASSIFIER;
                return false;
            }
            regressionData = contextModules[ AFTER_CLASSIFIER ][moduleIndex]->getProcessedData();
        }
    }
    
    //Perform any post processing
    predictionModuleIndex = AFTER_CLASSIFIER;
    if( getIsPostProcessingSet() ){
        
        if( pipelineMode != REGRESSION_MODE ){
            errorLog << "map(vector< double > inputVector) - Pipeline Mode Is Not In RegressionMode!" << endl;
            return false;
        }
          
        for(UINT moduleIndex=0; moduleIndex<postProcessingModules.size(); moduleIndex++){
            if( regressionData.size() != postProcessingModules[moduleIndex]->getNumInputDimensions() ){
                errorLog << "map(vector< double > inputVector) - The size of the regression vector (" << regressionData.size() << ") does not match that of the postProcessingModule (" << postProcessingModules[moduleIndex]->getNumInputDimensions() << ") at the moduleIndex: " << moduleIndex <<endl;
                return false;
            }
            
            if( !postProcessingModules[moduleIndex]->process( regressionData ) ){
                errorLog << "map(vector< double > inputVector) - Failed to post process data. PostProcessing moduleIndex: " << moduleIndex <<endl;
                return false;
            }
            regressionData = postProcessingModules[moduleIndex]->getProcessedData();        
        }
        
    } 
    
    //Update the context module
    predictionModuleIndex = END_OF_PIPELINE;
    if( contextModules[ END_OF_PIPELINE ].size() ){
        for(UINT moduleIndex=0; moduleIndex<contextModules[ END_OF_PIPELINE ].size(); moduleIndex++){
            if( !contextModules[ END_OF_PIPELINE ][moduleIndex]->process( inputVector ) ){
                errorLog << "map(vector< double > inputVector) - Context Module Failed at END_OF_PIPELINE. ModuleIndex: " << moduleIndex << endl;
                return false;
            }
            if( !contextModules[ END_OF_PIPELINE ][moduleIndex]->getOK() ){
                predictionModuleIndex = END_OF_PIPELINE;
                return false;
            }
            regressionData = contextModules[ END_OF_PIPELINE ][moduleIndex]->getProcessedData();
        }
    }
    
    return true;
}
    
bool GestureRecognitionPipeline::reset(){
    
    if( getIsPreProcessingSet() ){
        for(UINT moduleIndex=0; moduleIndex<preProcessingModules.size(); moduleIndex++){
            if( !preProcessingModules[ moduleIndex ]->reset() ){
                errorLog << "Failed To Reset PreProcessingModule " << moduleIndex << endl;
                return false;
            }
        }
    }
    
    //Perform any feature extraction
    if( getIsFeatureExtractionSet() ){
        for(UINT moduleIndex=0; moduleIndex<featureExtractionModules.size(); moduleIndex++){
            if( !featureExtractionModules[ moduleIndex ]->reset() ){
                errorLog << "Failed To Reset FeatureExtractionModule " << moduleIndex << endl;
                return false;
            }
        }
    }
    
    //Perform the classification
    if( getIsClassifierSet() ){
        if( !classifier->reset() ){
            errorLog << "Failed To Reset Classifier" << endl;
            return false;
        }
    }
    
    if( getIsPostProcessingSet() ){
        for(UINT moduleIndex=0; moduleIndex<postProcessingModules.size(); moduleIndex++){
            if( !postProcessingModules[ moduleIndex ]->reset() ){
                errorLog << "Failed To Reset PostProcessingModule " << moduleIndex << endl;
                return false;
            }
        }
    } 
    
    return true;
    
}
    
bool GestureRecognitionPipeline::savePipelineToFile(string filename){
    
    if( !initialized ){
        errorLog << "Failed to write pipeline to file as the pipeline has not been initialized yet!" << endl;
        return false;
    }
    
    fstream file;
    
    file.open(filename.c_str(), iostream::out );
    
    if( !file.is_open() ){
        errorLog << "Failed to open file with filename: " << filename << endl;
        return false;
    }
    
    //Write the pipeline header info
    file << "GRT_PIPELINE_FILE_V1.0\n";
    file << "PipelineMode: " << getPipelineModeAsString() << endl;
    file << "NumPreprocessingModules: " << getNumPreProcessingModules() << endl;
    file << "NumFeatureExtractionModules: " << getNumFeatureExtractionModules() << endl;
    file << "NumPostprocessingModules: " << getNumPostProcessingModules() << endl;
    file << "Trained: " << getTrained() << endl;
    
    //Write the module datatype names
    file << "PreProcessingModuleDatatypes:";
    for(UINT i=0; i<getNumPreProcessingModules(); i++){
        file << "\t" << preProcessingModules[i]->getPreProcessingType();
    }
    file << endl;
    
    file << "FeatureExtractionModuleDatatypes:";
    for(UINT i=0; i<getNumFeatureExtractionModules(); i++){
        file << "\t" << featureExtractionModules[i]->getFeatureExtractionType();
    }
    file << endl;
    
    switch( pipelineMode ){
        case PIPELINE_MODE_NOT_SET:
            break;
        case CLASSIFICATION_MODE:
            if( getIsClassifierSet() ) file << "ClassificationModuleDatatype:\t" << classifier->getClassifierType() << endl;
            else file << "ClassificationModuleDatatype:\tCLASSIFIER_NOT_SET" << endl;
            break;
        case REGRESSION_MODE:
            if( getIsRegressifierSet() ) file << "RegressionnModuleDatatype:\t" << regressifier->getRegressifierType() << endl;
            else file << "RegressionnModuleDatatype:\tREGRESSIFIER_NOT_SET" << endl;
            break;
        default:
            break;
    }
    
    file << "PostProcessingModuleDatatypes:";
    for(UINT i=0; i<getNumPostProcessingModules(); i++){
        file << "\t" << postProcessingModules[i]->getPostProcessingType();
    }
    file << endl;
    
    //Write the preprocessing module data to the file
    for(UINT i=0; i<getNumPreProcessingModules(); i++){
        file << "PreProcessingModule_" << Util::intToString(i+1) << endl;
        if( !preProcessingModules[i]->saveSettingsToFile( file ) ){
            errorLog << "Failed to write preprocessing module " <<  i << " settings to file!" << endl;
            file.close();
            return false;
        }
    }
    
    //Write the feature extraction module data to the file
    for(UINT i=0; i<getNumFeatureExtractionModules(); i++){
        file << "FeatureExtractionModule_" << Util::intToString(i+1) << endl;
        if( !featureExtractionModules[i]->saveSettingsToFile( file ) ){
            errorLog << "Failed to write feature extraction module " <<  i << " settings to file!" << endl;
            file.close();
            return false;
        }
    }
    
    switch( pipelineMode ){
        case PIPELINE_MODE_NOT_SET:
            break;
        case CLASSIFICATION_MODE:
            if( getIsClassifierSet() ){
                if( !classifier->saveModelToFile( file ) ){
                    errorLog << "Failed to write classifier model to file!" << endl;
                    file.close();
                    return false;
                }
            }
            break;
        case REGRESSION_MODE:
            if( getIsRegressifierSet() ){
                if( !regressifier->saveModelToFile( file ) ){
                    errorLog << "Failed to write regressifier model to file!" << endl;
                    file.close();
                    return false;
                }
            }
            break;
        default:
            break;
    }
    
    //Write the post processing module data to the file
    for(UINT i=0; i<getNumPostProcessingModules(); i++){
        file << "PostProcessingModule_" << Util::intToString(i+1) << endl;
        if( !postProcessingModules[i]->saveSettingsToFile( file ) ){
            errorLog << "Failed to write post processing module " <<  i << " settings to file!" << endl;
            file.close();
            return false;
        }
    }
    
    //Close the file
    file.close();
    
    return true;
}

bool GestureRecognitionPipeline::loadPipelineFromFile(string filename){
    
    //TODO
    
    return false;
}
    
bool GestureRecognitionPipeline::preProcessData(vector< double > inputVector,bool computeFeatures){
    
    if( getIsPreProcessingSet() ){
        for(UINT moduleIndex=0; moduleIndex<preProcessingModules.size(); moduleIndex++){
            
            if( inputVector.size() != preProcessingModules[ moduleIndex ]->getNumInputDimensions() ){
                errorLog << "preProcessData(vector< double > inputVector) - The size of the input vector (" << preProcessingModules[ moduleIndex ]->getNumInputDimensions() << ") does not match that of the PreProcessing Module at moduleIndex: " << moduleIndex << endl;
                return false;
            }
            
            if( !preProcessingModules[ moduleIndex ]->process( inputVector ) ){
                errorLog << "preProcessData(vector< double > inputVector) - Failed To PreProcess Input Vector. PreProcessing moduleIndex: " << moduleIndex << endl;
                return false;
            }
            inputVector = preProcessingModules[ moduleIndex ]->getProcessedData();
        }
    }
    
    //Perform any feature extraction
    if( getIsFeatureExtractionSet() && computeFeatures ){
        for(UINT moduleIndex=0; moduleIndex<featureExtractionModules.size(); moduleIndex++){
            if( inputVector.size() != featureExtractionModules[ moduleIndex ]->getNumInputDimensions() ){
                errorLog << "preProcessData(vector< double > inputVector) - The size of the input vector (" << featureExtractionModules[ moduleIndex ]->getNumInputDimensions() << ") does not match that of the FeatureExtraction Module at moduleIndex: " << moduleIndex << endl;
                return false;
            }
            
            if( !featureExtractionModules[ moduleIndex ]->computeFeatures( inputVector ) ){
                errorLog << "preProcessData(vector< double > inputVector) - Failed To Compute Features from Input Vector. FeatureExtraction moduleIndex: " << moduleIndex << endl;
                return false;
            }
            inputVector = featureExtractionModules[ moduleIndex ]->getFeatureVector();
        }
    }
    
    return true;
}
 
/////////////////////////////// GETTERS ///////////////////////////////
bool GestureRecognitionPipeline::getIsInitialized(){ 
    return initialized; 
}
    
bool GestureRecognitionPipeline::getTrained(){ 
    return trained; 
}
    
bool GestureRecognitionPipeline::getIsPreProcessingSet(){ 
    return preProcessingModules.size() > 0; 
} 
    
bool GestureRecognitionPipeline::getIsFeatureExtractionSet(){ 
    return featureExtractionModules.size() > 0; 
}
    
bool GestureRecognitionPipeline::getIsClassifierSet(){ 
    return (classifier!=NULL); 
}
    
bool GestureRecognitionPipeline::getIsRegressifierSet(){ 
    return (regressifier!=NULL); 
}
    
bool GestureRecognitionPipeline::getIsPostProcessingSet(){ 
    return postProcessingModules.size() > 0; 
}
    
bool GestureRecognitionPipeline::getIsContextSet(){ 
    for(UINT i=0; i<NUM_CONTEXT_LEVELS; i++){
        if( contextModules[i].size() > 0 ) return true;
    }
    return false;
}
    
bool GestureRecognitionPipeline::getIsPipelineModeSet(){ 
    return pipelineMode!=PIPELINE_MODE_NOT_SET; 
}
    
bool GestureRecognitionPipeline::getIsPipelineInClassificationMode(){ 
    return pipelineMode==CLASSIFICATION_MODE; 
}
    
bool GestureRecognitionPipeline::getIsPipelineInRegressionMode(){ 
    return pipelineMode==REGRESSION_MODE; 
}

UINT GestureRecognitionPipeline::getInputVectorDimensionsSize(){ 
    
    if( getIsPreProcessingSet() ){
        return preProcessingModules[0]->getNumInputDimensions();
    }
    
    if( getIsFeatureExtractionSet() ){
        return featureExtractionModules[0]->getNumInputDimensions();
    }
    
    if( getIsPipelineInClassificationMode() && getIsClassifierSet() ){
        return classifier->getNumInputFeatures();
    }
    if( getIsPipelineInRegressionMode() && getIsRegressifierSet() ){
        return regressifier->getNumInputFeatures(); 
    }
    return 0; 
}
    
UINT GestureRecognitionPipeline::getOutputVectorDimensionsSize(){ 
    if( getIsClassifierSet() ) return 1;    //The output of the pipeline for classification will always be 1
    if( getIsRegressifierSet() ){
        return regressifier->getNumOutputDimensions();  
    }
    return 0;
}
    
UINT GestureRecognitionPipeline::getNumClassesInModel(){ 
    return (getIsClassifierSet() ? classifier->getNumClasses() : 0); 
}
    
UINT GestureRecognitionPipeline::getNumPreProcessingModules(){ 
    return (UINT)preProcessingModules.size(); 
}
    
UINT GestureRecognitionPipeline::getNumFeatureExtractionModules(){ 
    return (UINT)featureExtractionModules.size(); 
}
    
UINT GestureRecognitionPipeline::getNumPostProcessingModules(){ 
    return (UINT)postProcessingModules.size(); 
}
    
UINT GestureRecognitionPipeline::getPredictionModuleIndexPosition(){ 
    return predictionModuleIndex; 
}
    
UINT GestureRecognitionPipeline::getPredictedClassLabel(){ 
    return (getIsClassifierSet() ? predictedClassLabel : 0); 
}
    
UINT GestureRecognitionPipeline::getUnProcessedPredictedClassLabel(){ 
    return (getIsClassifierSet() ? classifier->getPredictedClassLabel() : 0); 
}

double GestureRecognitionPipeline::getMaximumLikelihood(){ 
    return (getIsClassifierSet() ? classifier->getMaximumLikelihood() : 0); 
}
    
double GestureRecognitionPipeline::getCrossValidationAccuracy(){ 
    return (getIsClassifierSet() ? testAccuracy : 0); 
}
    
double GestureRecognitionPipeline::getTestAccuracy(){ 
    return testAccuracy; 
}
    
double GestureRecognitionPipeline::getTestRMSError(){ 
    return testRMSError; 
}
    
double GestureRecognitionPipeline::getTestFMeasure(UINT classLabel){
    
    if( !getIsClassifierSet() ) return -1;
    if( getClassLabels().size() != testFMeasure.size() ) return -1;
    
    for(UINT i=0; i<testFMeasure.size(); i++){
        if( getClassLabels()[i] == classLabel ){
            return testFMeasure[i];
        }
    }
    return -1;
}

double GestureRecognitionPipeline::getTestPrecision(UINT classLabel){
    
    if( !getIsClassifierSet() ) return -1;
    if( getClassLabels().size() != testFMeasure.size() ) return -1;
    
    for(UINT i=0; i<testPrecision.size(); i++){
        if( getClassLabels()[i] == classLabel ){
            return testPrecision[i];
        }
    }
    return -1;
}

double GestureRecognitionPipeline::getTestRecall(UINT classLabel){
    
    if( !getIsClassifierSet() ) return -1;
    if( getClassLabels().size() != testFMeasure.size() ) return -1;
    
    for(UINT i=0; i<testRecall.size(); i++){
        if( getClassLabels()[i] == classLabel ){
            return testRecall[i];
        }
    }
    return -1;
}

double GestureRecognitionPipeline::getTestRejectionPrecision(){ 
    return testRejectionPrecision; 
}
    
double GestureRecognitionPipeline::getTestRejectionRecall(){ 
    return testRejectionRecall; 
}
    
Matrix<double> GestureRecognitionPipeline::getTestConfusionMatrix(){ 
    return testConfusionMatrix; 
}

vector< double > GestureRecognitionPipeline::getClassLikelihoods(){ 
    if( getIsClassifierSet() ){ return classifier->getClassLikelihoods(); }
    else{ return vector< double >(); } 
}

vector< double > GestureRecognitionPipeline::getClassDistances(){ 
    if( getIsClassifierSet() ){ return classifier->getClassDistances(); }
    else{ return vector< double >(); } 
}

vector< double > GestureRecognitionPipeline::getNullRejectionThresholds(){
    if( getIsClassifierSet() ){ return classifier->getNullRejectionThresholds(); }
    else{ return vector< double >(); } 
}

vector< double > GestureRecognitionPipeline::getRegressionData(){ 
    if( getIsRegressifierSet() ){
        if( getIsPostProcessingSet() ){ 
            return postProcessingModules[ postProcessingModules.size()-1 ]->getProcessedData(); 
        }
        return regressifier->getRegressionData();
    }
    return vector< double >();
}

vector< double > GestureRecognitionPipeline::getUnProcessedRegressionData(){ 
    if( getIsRegressifierSet() ) {
        return regressifier->getRegressionData();
    }
    return vector< double >();
}
    
vector< double > GestureRecognitionPipeline::getPreProcessedData(){
    if( getIsPreProcessingSet() ){ 
        return preProcessingModules[ preProcessingModules.size()-1 ]->getProcessedData(); 
    }
    return vector< double >();
}

vector< double > GestureRecognitionPipeline::getPreProcessedData(UINT moduleIndex){
    if( getIsPreProcessingSet() ){ 
        if( moduleIndex < preProcessingModules.size() ){
            return preProcessingModules[ moduleIndex ]->getProcessedData(); 
        }
    }
    return vector< double >();
}

vector< double > GestureRecognitionPipeline::getFeatureExtractionData(){
    if( getIsFeatureExtractionSet() ){ 
        return featureExtractionModules[ featureExtractionModules.size()-1 ]->getFeatureVector(); 
    }
    return vector< double >();
}
    
vector< double > GestureRecognitionPipeline::getFeatureExtractionData(UINT moduleIndex){
    if( getIsFeatureExtractionSet() ){ 
        if( moduleIndex < featureExtractionModules.size() ){
            return featureExtractionModules[ moduleIndex ]->getFeatureVector(); 
        }
    }
    return vector< double >();
}
    
vector< UINT > GestureRecognitionPipeline::getClassLabels(){ 
    if( trained && getIsClassifierSet() )
        return classifier->getClassLabels(); 
    return vector< UINT>(); 
}
    
PreProcessing* GestureRecognitionPipeline::getPreProcessingModule(UINT moduleIndex){ 
    if( moduleIndex < preProcessingModules.size() ){
        return preProcessingModules[ moduleIndex ];
    }
    return NULL;
}
    
FeatureExtraction* GestureRecognitionPipeline::getFeatureExtractionModule(UINT moduleIndex){ 
    if( moduleIndex < featureExtractionModules.size() ){
        return featureExtractionModules[ moduleIndex ];
    }
    return NULL;
}
    
Classifier* GestureRecognitionPipeline::getClassifier(){
    return classifier; 
}
    
Regressifier* GestureRecognitionPipeline::getRegressifier(){ 
    return regressifier; 
}
    
PostProcessing* GestureRecognitionPipeline::getPostProcessingModule(UINT moduleIndex){ 
    if( moduleIndex < postProcessingModules.size() ){
        return postProcessingModules[ moduleIndex ];
    }
    return NULL;
}
    
Context* GestureRecognitionPipeline::getContextModule(UINT contextLevel,UINT moduleIndex){ 
    if( contextLevel < contextModules.size() ){
        if( moduleIndex < contextModules[ contextLevel ].size() ){
            return contextModules[ contextLevel ][ moduleIndex ];
        }
    }
    return NULL;
}
    
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////  
///////////////////////////////////////////                SETTERS                    ///////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
    
bool GestureRecognitionPipeline::addPreProcessingModule(const PreProcessing &preProcessingModule,UINT insertIndex){
    
    //Validate the insertIndex is valid
    if( insertIndex != INSERT_AT_END_INDEX && insertIndex >= preProcessingModules.size() ){
        errorLog << "addPreProcessingModule(const PreProcessing &preProcessingModule) - Invalid insertIndex value!" << endl;
        return false;
    }
    
    //Create a new instance of the preProcessing and then clone the values across from the reference preProcessing
    PreProcessing *newInstance = preProcessingModule.createNewInstance();
    
    //Verify that the clone was successful
    if( !newInstance->clone( &preProcessingModule ) ){
        delete newInstance;
        newInstance = NULL;
        errorLog << "addPreProcessingModule(const PreProcessing &preProcessingModule) - PreProcessing Module Not Set!" << endl;
        return false;
    }
    
    //Add the new instance to the preProcessingModules
    vector< PreProcessing* >::iterator iter = preProcessingModules.begin();
    
    if( insertIndex == INSERT_AT_END_INDEX ) iter = preProcessingModules.end(); 
    else iter = preProcessingModules.begin() + insertIndex;
    
    preProcessingModules.insert(iter, newInstance);
    
    return true;
}

bool GestureRecognitionPipeline::addFeatureExtractionModule(const FeatureExtraction &featureExtractionModule,UINT insertIndex){
    
    //Validate the insertIndex is valid
    if( insertIndex != INSERT_AT_END_INDEX && insertIndex >= featureExtractionModules.size() ){
        errorLog << "addFeatureExtractionModule(const FeatureExtraction &featureExtractionModule) - Invalid insertIndex value!" << endl;
        return false;
    }
    
    //Create a new instance of the preProcessing and then clone the values across from the reference preProcessing
    FeatureExtraction *newInstance = featureExtractionModule.createNewInstance();
    
    //Verify that the clone was successful
    if( !newInstance->clone( &featureExtractionModule ) ){
        delete newInstance;
        newInstance = NULL;
        errorLog << "addFeatureExtractionModule(const FeatureExtraction &featureExtractionModule - FeatureExtraction Module Not Set!" << endl;
        return false;
    }
    
    //Add the new instance to the preProcessingModules
    vector< FeatureExtraction* >::iterator iter = featureExtractionModules.begin();
    
    if( insertIndex == INSERT_AT_END_INDEX ) iter = featureExtractionModules.end(); 
    else iter = featureExtractionModules.begin() + insertIndex;
    
    featureExtractionModules.insert(iter, newInstance);
    
    return true;
}

bool GestureRecognitionPipeline::setClassifier(const Classifier &classifier){
    //Delete any previous classifier and regressifier
    deleteClassifier();
    deleteRegressifier();
    
    //Create a new instance of the classifier and then clone the values across from the reference classifier
    this->classifier = classifier.createNewInstance();
    
    //Validate that the classifier was cloned correctly
    if( !this->classifier->clone( &classifier ) ){
        deleteClassifier();
        pipelineMode = PIPELINE_MODE_NOT_SET;
        errorLog << "setClassifier(const Classifier classifier) - Classifier Module Not Set!" << endl;
        return false;
    }
    
    //Set the mode of the pipeline to classification mode
    pipelineMode = CLASSIFICATION_MODE;
    
    //Flag that the key part of the pipeline has now been initialized
    initialized = true;
    
    return true;
}

bool GestureRecognitionPipeline::setRegressifier(const Regressifier &regressifier){
    //Delete any previous classifier and regressifier
    deleteClassifier();
    deleteRegressifier();
    
    //Set the mode of the pipeline to regression mode
    pipelineMode = REGRESSION_MODE;
    
    //Create a new instance of the regressifier and then clone the values across from the reference regressifier
    this->regressifier = regressifier.createNewInstance();
    
    //Validate that the regressifier was cloned correctly
    if( !this->regressifier->clone( &regressifier ) ){
        deleteRegressifier();
        pipelineMode = PIPELINE_MODE_NOT_SET;
        errorLog << "setRegressifier(const Regressifier &regressifier) - Regressifier Module Not Set!" << endl;
        return false;
    }
    
    //Flag that the key part of the pipeline has now been initialized
    initialized = true;
    
    return true;
}

bool GestureRecognitionPipeline::addPostProcessingModule(const PostProcessing &postProcessingModule,UINT insertIndex){
    
    //Validate the insertIndex is valid
    if( insertIndex != INSERT_AT_END_INDEX && insertIndex >= postProcessingModules.size() ){
        errorLog << "addPostProcessingModule((const PostProcessing &postProcessingModule) - Invalid insertIndex value!" << endl;
        return false;
    }
    
    //Create a new instance of the preProcessing and then clone the values across from the reference preProcessing
    PostProcessing *newInstance = postProcessingModule.createNewInstance();
    
    //Verify that the clone was successful
    if( !newInstance->clone( &postProcessingModule ) ){
        delete newInstance;
        newInstance = NULL;
        errorLog << "addPostProcessingModule(const PostProcessing &postProcessingModule) - PostProcessing Module Not Set!" << endl;
        return false;
    }
    
    //Add the new instance to the postProcessingModules
    vector< PostProcessing* >::iterator iter = postProcessingModules.begin();
    
    if( insertIndex == INSERT_AT_END_INDEX ) iter = postProcessingModules.end(); 
    else iter = postProcessingModules.begin() + insertIndex;
    
    postProcessingModules.insert(iter, newInstance);
    
    return true;
}
    
bool GestureRecognitionPipeline::removeAllPreProcessingModules(){
    deleteAllPreProcessingModules();
    return true;
}
    
bool GestureRecognitionPipeline::removePreProcessingModule(UINT moduleIndex){
    if( moduleIndex >= preProcessingModules.size() ){
        errorLog << "removePreProcessingModule(UINT moduleIndex) - Invalid moduleIndex " << moduleIndex << ". The size of the preProcessingModules vector is " << preProcessingModules.size() << endl;
        return false;
    }
    
    //Delete the module
    delete preProcessingModules[ moduleIndex ];
    preProcessingModules[ moduleIndex ] = NULL;
    preProcessingModules.erase( preProcessingModules.begin() + moduleIndex );
    
    return true;
}
   
bool GestureRecognitionPipeline::removeAllFeatureExtractionModules(){
    deleteAllFeatureExtractionModules();
    return true;
}
        
bool GestureRecognitionPipeline::removeFeatureExtractionModule(UINT moduleIndex){
    if( moduleIndex >= featureExtractionModules.size() ){
        errorLog << "removeFeatureExtractionModule(UINT moduleIndex) - Invalid moduleIndex " << moduleIndex << ". The size of the featureExtractionModules vector is " << featureExtractionModules.size() << endl;
        return false;
    }
    
    //Delete the module
    delete featureExtractionModules[ moduleIndex ];
    featureExtractionModules[ moduleIndex ] = NULL;
    featureExtractionModules.erase( featureExtractionModules.begin() + moduleIndex );
    
    return true;
}

bool GestureRecognitionPipeline::removeAllPostProcessingModules(){
    deleteAllPostProcessingModules();
    return true;
}

bool GestureRecognitionPipeline::removePostProcessingModule(UINT moduleIndex){
    if( moduleIndex >= postProcessingModules.size() ){
        errorLog << "removePostProcessingModule(UINT moduleIndex) - Invalid moduleIndex " << moduleIndex << ". The size of the postProcessingModules vector is " << postProcessingModules.size() << endl;
        return false;
    }
    
    //Delete the module
    delete postProcessingModules[ moduleIndex ];
    postProcessingModules[ moduleIndex ] = NULL;
    postProcessingModules.erase( postProcessingModules.begin() + moduleIndex );
    
    return true;
}
    
bool GestureRecognitionPipeline::removeContextModule(UINT contextLevel,UINT moduleIndex){
    if( contextLevel >= NUM_CONTEXT_LEVELS ){
        errorLog << "removeContextModule(UINT contextLevel,UINT moduleIndex) - Invalid moduleIndex " << moduleIndex << " is out of bounds!" << endl;
        return false;
    }
    
    if( moduleIndex >= contextModules[contextLevel].size() ){
        errorLog << "removePostProcessingModule(UINT moduleIndex) - Invalid moduleIndex " << moduleIndex << ". The size of the contextModules vector at context level " << contextLevel << " is " << contextModules[contextLevel].size() << endl;
        return false;
    }
    
    //Delete the module
    delete contextModules[contextLevel][moduleIndex];
    contextModules[contextLevel][moduleIndex] = NULL;
    contextModules[contextLevel].erase( contextModules[contextLevel].begin() + moduleIndex );
    return true;
}
    
bool GestureRecognitionPipeline::removeAllContextModules(){
    deleteAllContextModules();
    return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////  
///////////////////////////////////////////          PROTECTED FUNCTIONS              ///////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
    
void GestureRecognitionPipeline::deleteAllPreProcessingModules(){
    if( preProcessingModules.size() != 0 ){
        for(UINT i=0; i<preProcessingModules.size(); i++){
            delete preProcessingModules[i];
            preProcessingModules[i] = NULL;
        }
        preProcessingModules.clear();
        trained = false;
    }
}
    
void GestureRecognitionPipeline::deleteAllFeatureExtractionModules(){
    if( featureExtractionModules.size() != 0 ){
        for(UINT i=0; i<featureExtractionModules.size(); i++){
            delete featureExtractionModules[i];
            featureExtractionModules[i] = NULL;
        }
        featureExtractionModules.clear();
        trained = false;
    }
}
    
void GestureRecognitionPipeline::deleteClassifier(){
    if( classifier != NULL ){
        delete classifier;
        classifier = NULL;
    }
    trained = false;
    initialized = false;
}
    
void GestureRecognitionPipeline::deleteRegressifier(){
    if( regressifier != NULL ){
        delete regressifier;
        regressifier = NULL;
    }
    trained = false;
    initialized = false;
}
    
void GestureRecognitionPipeline::deleteAllPostProcessingModules(){
    if( postProcessingModules.size() != 0 ){
        for(UINT i=0; i<postProcessingModules.size(); i++){
            delete postProcessingModules[i];
            postProcessingModules[i] = NULL;
        }
        postProcessingModules.clear();
        trained = false;
    }
}
    
void GestureRecognitionPipeline::deleteAllContextModules(){
    for(UINT i=0; i<contextModules.size(); i++){
        for(UINT j=0; j<contextModules[i].size(); j++){
            delete contextModules[i][j];
            contextModules[i][j] = NULL;
        }
        contextModules[i].clear();
    }
}
    
bool GestureRecognitionPipeline::updateTestMetrics(const UINT classLabel,const UINT predictedClassLabel,vector< double > &precisionCounter,vector< double > &recallCounter,double &rejectionPrecisionCounter,double &rejectionRecallCounter,vector< double > &confusionMatrixCounter){

    //Find the index of the classLabel
    UINT predictedClassLabelIndex =0;
    bool predictedClassLabelIndexFound = false;
    if( predictedClassLabel != 0 ){
        for(UINT k=0; k<getNumClassesInModel(); k++){
            if( predictedClassLabel == classifier->getClassLabels()[k] ){
                predictedClassLabelIndex = k;
                predictedClassLabelIndexFound = true;
                break;
            }
        }
        
        if( !predictedClassLabelIndexFound ){
            errorLog << "Failed to find class label index for label: " << predictedClassLabel << endl;
            return false;
        }
    }

    //Find the index of the class label
    UINT actualClassLabelIndex = 0;
    if( classLabel != 0 ){
        for(UINT k=0; k<getNumClassesInModel(); k++){
            if( classLabel == classifier->getClassLabels()[k] ){
                actualClassLabelIndex = k;
                break;
            }
        }
    }

    //Update the classification accuracy
    if( classLabel == predictedClassLabel ){
        testAccuracy++;
    }

    //Update the precision
    if( predictedClassLabel != 0 ){
        if( classLabel == predictedClassLabel ){
            //Update the precision value
            testPrecision[ predictedClassLabelIndex ]++;
        }
        //Update the precision counter
        precisionCounter[ predictedClassLabelIndex ]++;
    }

    //Update the recall
    if( classLabel != 0 ){
        if( classLabel == predictedClassLabel ){
            //Update the recall value
            testRecall[ predictedClassLabelIndex ]++;
        }
        //Update the recall counter
        recallCounter[ actualClassLabelIndex ]++;
    }

    //Update the rejection precision
    if( predictedClassLabel == 0 ){
        if( classLabel == 0 ) testRejectionPrecision++;
        rejectionPrecisionCounter++;
    }

    //Update the rejection recall
    if( classLabel == 0 ){
        if( predictedClassLabel == 0 ) testRejectionRecall++;
        rejectionRecallCounter++;
    }

    //Update the confusion matrix
    if( classifier->getNullRejectionEnabled() ){
        if( classLabel == 0 ) actualClassLabelIndex = 0;
        else actualClassLabelIndex++;
        if( predictedClassLabel == 0 ) predictedClassLabelIndex = 0;
        else predictedClassLabelIndex++;
    }
    testConfusionMatrix[ actualClassLabelIndex  ][ predictedClassLabelIndex ]++;
    confusionMatrixCounter[ actualClassLabelIndex ]++;
    
    return true;
}

bool GestureRecognitionPipeline::computeTestMetrics(vector< double > &precisionCounter,vector< double > &recallCounter,double &rejectionPrecisionCounter,double &rejectionRecallCounter,vector< double > &confusionMatrixCounter,const UINT numTestSamples){
        
    //Compute the test metrics
    testAccuracy = testAccuracy/double(numTestSamples) * 100.0;
    
    for(UINT k=0; k<getNumClassesInModel(); k++){
        if( precisionCounter[k] > 0 ) testPrecision[k] /= precisionCounter[k];
        else testPrecision[k] = 0;
        if( recallCounter[k] > 0 ) testRecall[k] /= recallCounter[k];
        else testRecall[k] = 0;
        
        if( precisionCounter[k] + recallCounter[k] > 0 )
            testFMeasure[k] = 2 * ((testPrecision[k]*testRecall[k])/(testPrecision[k]+testRecall[k]));
        else testFMeasure[k] = 0;
    }
    if( rejectionPrecisionCounter > 0 ) testRejectionPrecision /= rejectionPrecisionCounter;
    if( rejectionRecallCounter > 0 ) testRejectionRecall /= rejectionRecallCounter;
    
    
    for(UINT r=0; r<confusionMatrixCounter.size(); r++){
        if( confusionMatrixCounter[r] > 0 ){
            for(UINT c=0; c<testConfusionMatrix.getNumCols(); c++){
                testConfusionMatrix[r][c] /= confusionMatrixCounter[r];
            }
        }
    }
    
    return true;
}
    
string GestureRecognitionPipeline::getPipelineModeAsString(){
    switch( pipelineMode ){
        case PIPELINE_MODE_NOT_SET:
            return "PIPELINE_MODE_NOT_SET";
            break;
        case CLASSIFICATION_MODE:
            return "CLASSIFICATION_MODE";
            break;
        case REGRESSION_MODE:
            return "REGRESSION_MODE";
            break;
        default:
            return "ERROR_UNKNWON_PIPELINE_MODE";
            break;
    }
    
    return "ERROR_UNKNWON_PIPELINE_MODE";
}

} //End of namespace GRT

