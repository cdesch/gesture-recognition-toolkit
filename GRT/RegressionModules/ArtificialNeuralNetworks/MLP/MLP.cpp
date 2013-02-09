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

#include "MLP.h"

namespace GRT{
    
//Register the MLP module with the Regressifier base class
RegisterRegressifierModule< MLP > MLP::registerModule("MLP");

MLP::MLP(){
    inputLayerActivationFunction = Neuron::LINEAR;
    hiddenLayerActivationFunction = Neuron::LINEAR;
    outputLayerActivationFunction = Neuron::LINEAR;
	minNumEpochs = 10;
    maxNumEpochs = 100;
    numRandomTrainingIterations = 10;
    validationSetSize = 20;	//20% of the training data will be set aside for the validation set
    minChange = 1.0e-10;
	trainingRate = 0.1;
	momentum = 0.5;
	gamma = 2.0;
	trainingError = 0;
    nullRejectionCoeff = 0.9;
    nullRejectionThreshold = 0;
	useValidationSet = true;
	randomiseTrainingOrder = false;
    useMultiThreadingTraining = false;
	useScaling = true;
	trained = false;
    initialized = false;
    classificationModeActive = false;
    useNullRejection = true;
    clear();
    regressifierType = "MLP";
    debugLog.setProceedingText("[DEBUG MLP]");
    errorLog.setProceedingText("[ERROR MLP]");
    trainingLog.setProceedingText("[TRAINING MLP]");
    warningLog.setProceedingText("[WARNING MLP]");
}

MLP::~MLP(){
    clear();
}
    
//Classifier interface
bool MLP::train(LabelledClassificationData &trainingData){
    
    if( !initialized ){
        errorLog << "train(LabelledClassificationData &trainingData) - The MLP has not been initialized!" << endl;
        return false;
    }
    
    if( trainingData.getNumDimensions() != numInputNeurons ){
        errorLog << "train(LabelledRegressionData trainingData) - The number of input dimensions in the training data (" << trainingData.getNumDimensions() << ") does not match that of the MLP (" << numInputNeurons << ")" << endl;
        return false;
    }
    if( trainingData.getNumClasses() != numOutputNeurons ){
        errorLog << "train(LabelledRegressionData trainingData) - The number of classes in the training data (" << trainingData.getNumClasses() << ") does not match that of the MLP (" << numOutputNeurons << ")" << endl;
        return false;
    }
    
    //Reformat the LabelledClassificationData as LabelledRegressionData
    LabelledRegressionData regressionData = trainingData.reformatAsLabelledRegressionData();
    
    //Flag that the MLP is being used for classification, not regression
    classificationModeActive = true;
    
    return train(regressionData);
}
    
//Classifier interface
bool MLP::predict(vector< double > inputVector){
    
    if( !trained ){
        errorLog << "predict(vector< double > inputVector) - Model not trained!" << endl;
        return false;
    }
    
    if( inputVector.size() != numInputNeurons ){
        errorLog << "predict(vector< double > inputVector) - The sie of the input vector (" << inputVector.size() << ") does not match that of the number of input dimensions (" << numInputNeurons << ") " << endl;
        return false;
    }
    
    //Set the mapped data as the classLikelihoods
    regressionData = feedforward(inputVector);
    
    if( classificationModeActive ){
        double bestValue = classLikelihoods[0];
        UINT bestIndex = 0;
        for(UINT i=1; i<classLikelihoods.size(); i++){
            if( classLikelihoods[i] > bestValue ){
                bestValue = classLikelihoods[i];
                bestIndex = i;
            }
        }
        
        //Set the maximum likelihood and predicted class label
        maxLikelihood = bestValue;
        predictedClassLabel = bestIndex+1;
        
        if( useNullRejection ){
            if( maxLikelihood < nullRejectionCoeff ){
                predictedClassLabel = 0;
            }
        }
    }
    
    return true;
}

bool MLP::init(UINT numInputNeurons,UINT numHiddenNeurons,UINT numOutputNeurons,UINT inputLayerActivationFunction,
                   UINT hiddenLayerActivationFunction,UINT outputLayerActivationFunction){
    
    //Clear any previous models
    clear();

	//Initialize the random seed
	random.setSeed( (UINT)time(NULL) );
    
    if( numInputNeurons == 0 || numHiddenNeurons == 0 || numOutputNeurons == 0 ){
        if( numInputNeurons == 0 )  errorLog << "init(...) - The number of input neurons is zero!" << endl;
        if( numHiddenNeurons == 0 )  errorLog << "init(...) - The number of hidden neurons is zero!" << endl;
        if( numOutputNeurons == 0 )  errorLog << "init(...) - The number of output neurons is zero!" << endl;
        return false;
    }
    
    //Validate the activation functions
    if( !validateActivationFunction(inputLayerActivationFunction) || !validateActivationFunction(hiddenLayerActivationFunction) || !validateActivationFunction(outputLayerActivationFunction) ){
        errorLog << "init(...) - One Of The Activation Functions Failed The Validation Check" << endl;
        return false;
    }

    //Set the size of the MLP
    this->numInputNeurons = numInputNeurons;
    this->numHiddenNeurons = numHiddenNeurons;
    this->numOutputNeurons = numOutputNeurons;
       
    //Set the validation layers
    this->inputLayerActivationFunction = inputLayerActivationFunction;
    this->hiddenLayerActivationFunction = hiddenLayerActivationFunction;
    this->outputLayerActivationFunction = outputLayerActivationFunction;
    
    //Setup the neurons for each of the layers
    inputLayer.resize(numInputNeurons);
    hiddenLayer.resize(numHiddenNeurons);
    outputLayer.resize(numOutputNeurons);
    
    //Init the neuron memory for each of the layers
    for(UINT i=0; i<numInputNeurons; i++){
        inputLayer[i].init(1,inputLayerActivationFunction);
        inputLayer[i].weights[0] = 1.0; //The weights for the input layer should always be 1
		inputLayer[i].bias = 0.0; //The bias for the input layer should always be 0
		inputLayer[i].gamma = gamma;
    }
    
    for(UINT i=0; i<numHiddenNeurons; i++){
        hiddenLayer[i].init(numInputNeurons,hiddenLayerActivationFunction);
		hiddenLayer[i].gamma = gamma;
    }
    
    for(UINT i=0; i<numOutputNeurons; i++){
        outputLayer[i].init(numHiddenNeurons,outputLayerActivationFunction);
		outputLayer[i].gamma = gamma;
    }
    
    initialized = true;
    
    return true;

}

void MLP::clear(){
    numInputNeurons = 0;
    numHiddenNeurons = 0;
    numOutputNeurons = 0;
    inputLayer.clear();
    hiddenLayer.clear();
    outputLayer.clear();
	trained = false;
    initialized = false;
}

bool MLP::train(LabelledRegressionData &trainingData_){

    trained = false;
    
    if( !initialized ){
        errorLog << "train(LabelledRegressionData trainingData) - The MLP has not be initialized!" << endl;
        return false;
    }
    
    
    if( trainingData_.getNumSamples() == 0 ){
        errorLog << "train(LabelledRegressionData trainingData) - The training data is empty!" << endl;
        return false;
    }
    
    //Copy the training data
    LabelledRegressionData trainingData = trainingData_;
    
    //Create a validation dataset, if needed
	LabelledRegressionData validationData;
	if( useValidationSet ){
		validationData = trainingData.partition( 100 - validationSetSize );
	}

    //Clear the ranges of the input vector and target vectors
	inputVectorRanges.clear();
    targetVectorRanges.clear();

    const UINT M = trainingData.getNumSamples();
    const UINT N = trainingData.getNumInputDimensions();
    const UINT T = trainingData.getNumTargetDimensions();
	const UINT numTestingExamples = useValidationSet ? validationData.getNumSamples() : M;

    if( N != numInputNeurons ){
        errorLog << "train(LabelledRegressionData trainingData) - The number of input dimensions in the training data (" << N << ") does not match that of the MLP (" << numInputNeurons << ")" << endl;
        return false;
    }
    if( T != numOutputNeurons ){
        errorLog << "train(LabelledRegressionData trainingData) - The number of target dimensions in the training data (" << T << ") does not match that of the MLP (" << numOutputNeurons << ")" << endl;
        return false;
    }
    
    //Set the Regressifier input and output dimensions
    numFeatures = numInputNeurons;
    numOutputDimensions = numOutputNeurons;

    //Scale the training and validation data, if needed
	if( useScaling ){
		//Find the ranges for the input data
        inputVectorRanges = trainingData.getInputRanges();
        
        //Find the ranges for the target data
		targetVectorRanges = trainingData.getTargetRanges();

		//Now scale the training data and the validation data if required
		trainingData.scale(inputVectorRanges,targetVectorRanges,0.0,1.0);
        
		if( useValidationSet ){
			validationData.scale(inputVectorRanges,targetVectorRanges,0.0,1.0);
		}
	}
    
    //Setup the training loop
    bool keepTraining = true;
    UINT epoch = 0;
    double lastErr = 0;
    double alpha = trainingRate;
	double beta = momentum;
    UINT bestIter = 0;
    MLP bestNetwork;
    double bestErr = 99e+99;
	vector< UINT > indexList(M);
	vector< vector< double > > tempTrainingErrorLog;
	trainingErrorLog.clear();
	trainingError = 0;

    //Reset the indexList, this is used to randomize the order of the training examples, if needed
	for(UINT i=0; i<M; i++) indexList[i] = i;
	
    for(UINT iter=0; iter<numRandomTrainingIterations; iter++){

        lastErr = 0;
        epoch = 0;
        keepTraining = true;
		tempTrainingErrorLog.clear();
        
		//Randomise the start values of the neurons
        init(numInputNeurons,numHiddenNeurons,numOutputNeurons);
        
        while( keepTraining ){

			if( randomiseTrainingOrder ){
				for(UINT i=0; i<M*100; i++){
					UINT indexA = random.getRandomNumberInt(0, M*2)%M;
					UINT indexB = random.getRandomNumberInt(0, M*2)%M;
					UINT temp = indexList[ indexA ];
					indexList[ indexA ] = indexList[ indexB ];
					indexList[ indexB ] = temp;
				}
			}
            
            //Perform one training epoch
            double error = 0;
            for(UINT i=0; i<M; i++){
                vector< double > trainingExample(N);
                vector< double > targetVector(T);
                for(UINT j=0; j<N; j++) trainingExample[j] = trainingData[ indexList[i] ].getInputVectorValue(j);
                for(UINT j=0; j<T; j++) targetVector[j] = trainingData[ indexList[i] ].getTargetVectorValue(j);
                
                double backPropError = back_prop(trainingExample,targetVector,alpha,beta);

				if( classificationModeActive ){
                    vector< double > y = feedforward(trainingExample);
                    
                    //Get the class label
                    double bestValue = targetVector[0];
                    UINT bestIndex = 0;
                    for(UINT i=1; i<targetVector.size(); i++){
                        if( targetVector[i] > bestValue ){
                            bestValue = targetVector[i];
                            bestIndex = i;
                        }
                    }
                    UINT classLabel = bestIndex + 1;
                    
                    //Get the predicted class label
                    bestValue = y[0];
                    bestIndex = 0;
                    for(UINT i=1; i<y.size(); i++){
                        if( y[i] > bestValue ){
                            bestValue = y[i];
                            bestIndex = i;
                        }
                    }
                    predictedClassLabel = bestIndex+1;
                    
                    if( classLabel != predictedClassLabel ){
                        error++;
                    }
                    
                }else{
                    error += backPropError;
                }

				if( checkForNAN() ){
					keepTraining = false;
                    errorLog << "train(LabelledRegressionData trainingData) - NaN found!" << endl;
					break;
				}
            }

            double trainingSetClassificationError = 0;
			double trainingSetRmsError = 0;
            
            if( classificationModeActive ) trainingSetClassificationError = (M-error)/double(M);
            else trainingSetRmsError = sqrt( error / double(M) );
            
			if( useValidationSet ){
				error = 0;
				//We don't need to scale the validation data as it is already scaled, so make sure scaling is set to off
				bool tempScalingState = useScaling;
				useScaling = false;
				for(UINT i=0; i<validationData.getNumSamples(); i++){
					vector< double > trainingExample(N);
					vector< double > targetVector(T);
					for(UINT j=0; j<N; j++) trainingExample[j] = validationData[i].getInputVectorValue(j);
					for(UINT j=0; j<T; j++) targetVector[j] = validationData[i].getTargetVectorValue(j);
                    
                    vector< double > y = feedforward(trainingExample);
                    
                    if( classificationModeActive ){
                        //Get the class label
                        double bestValue = targetVector[0];
                        UINT bestIndex = 0;
                        for(UINT i=1; i<targetVector.size(); i++){
                            if( targetVector[i] > bestValue ){
                                bestValue = targetVector[i];
                                bestIndex = i;
                            }
                        }
                        UINT classLabel = bestIndex + 1;
                        
                        //Get the predicted class label
                        bestValue = y[0];
                        bestIndex = 0;
                        for(UINT i=1; i<y.size(); i++){
                            if( y[i] > bestValue ){
                                bestValue = y[i];
                                bestIndex = i;
                            }
                        }
                        predictedClassLabel = bestIndex+1;
                        
                        if( classLabel != predictedClassLabel ){
                            error++;
                        }
                        
                    }else{
                        //Update the rms error
                        for(UINT j=0; j<T; j++){
                            error += (targetVector[j]-y[j])*(targetVector[j]-y[j]);
                        }
                    }
				}
                //Reset the scaling flag
				useScaling = tempScalingState;
			}
            
            //Compute the error of all the training data
            if( classificationModeActive ){
                //Compute the classification error
                double classificationErr = (numTestingExamples-error)/double(numTestingExamples);
                vector< double > temp(2);
                temp[0] = trainingSetClassificationError;
                temp[1] = classificationErr;
                tempTrainingErrorLog.push_back( temp );
                
                if( ++epoch >= maxNumEpochs ){
                    keepTraining = false;
                }else{
                    if( fabs( classificationErr - lastErr ) <= minChange && epoch >= minNumEpochs ){
                        keepTraining = false;
                    }
                }
                
                //Update the last error
                lastErr = classificationErr;
            }else{
                //We are in regression mode, so compute the RMS Error
                double rmsErr = sqrt( error / double(numTestingExamples) );
                vector< double > temp(2);
                temp[0] = trainingSetRmsError;
                temp[1] = rmsErr;
                tempTrainingErrorLog.push_back( temp );
                
                if( ++epoch >= maxNumEpochs ){
                    keepTraining = false;
                }else{
                    if( fabs( rmsErr - lastErr ) <= minChange && epoch >= minNumEpochs ){
                        keepTraining = false;
                    }
                }
                
                //Update the last error
                lastErr = rmsErr;
            }
            
            trainingLog << "Random Training Iteration: " << iter+1 << " Epoch: " << epoch-1 << " Error: " << lastErr << endl;
            
        }//End of While( keepTraining )
        
        if( lastErr < bestErr ){
            bestIter = iter;
			trainingError = bestErr;
            bestErr = lastErr;
            bestNetwork = *this;
			trainingErrorLog = tempTrainingErrorLog;
        }

    }//End of For( numRandomTrainingIterations )
    
    trainingLog << "BestError: " << bestErr << " in Random Training Iteration: " << bestIter+1 << endl;

	//Check to make sure the best network has not got any NaNs in it
	if( checkForNAN() ){
        errorLog << "train(LabelledRegressionData trainingData) - NAN Found!" << endl;
		return false;
	}
    
    //Set the MLP model to the model that best during training
    *this = bestNetwork;
    
    //Compute the rejection threshold
    if( classificationModeActive ){
        double averageValue = 0;
        vector< double > classificationPredictions;
        
        for(UINT i=0; i<numTestingExamples; i++){
            vector< double > inputVector = useValidationSet ? validationData[i].getInputVector() : trainingData[i].getInputVector();
            vector< double > targetVector = useValidationSet ? validationData[i].getTargetVector() : trainingData[i].getTargetVector();
            
            //Make the prediction
            vector< double > y = feedforward(inputVector);
            
            //Get the class label
            double bestValue = targetVector[0];
            UINT bestIndex = 0;
            for(UINT i=1; i<targetVector.size(); i++){
                if( targetVector[i] > bestValue ){
                    bestValue = targetVector[i];
                    bestIndex = i;
                }
            }
            UINT classLabel = bestIndex + 1;
            
            //Get the predicted class label
            bestValue = y[0];
            bestIndex = 0;
            for(UINT i=1; i<y.size(); i++){
                if( y[i] > bestValue ){
                    bestValue = y[i];
                    bestIndex = i;
                }
            }
            predictedClassLabel = bestIndex+1;
            
            //Only add the max value if the prediction is correct
            if( classLabel == predictedClassLabel ){
                classificationPredictions.push_back( bestValue );
                averageValue += bestValue;
            }
        }
        
        averageValue /= double(classificationPredictions.size());
        double stdDev = 0;
        for(UINT i=0; i<classificationPredictions.size(); i++){
            stdDev += SQR(classificationPredictions[i]-averageValue);
        }
        stdDev = sqrt( stdDev / double(classificationPredictions.size()-1) );
        
        nullRejectionThreshold = averageValue-(stdDev*nullRejectionCoeff);
    }
    
    //Flag that the model has been successfully trained
	trained = true;

    return true;
}

double MLP::back_prop(vector< double > &trainingExample,vector< double > &targetVector,double alpha,double beta){
    
    double update = 0;
    vector< double > inputNeuronsOuput;
    vector< double > hiddenNeuronsOutput;
    vector< double > outputNeuronsOutput;
    vector< double > deltaO(numOutputNeurons);
    vector< double > deltaH(numHiddenNeurons);
    
    //Forward propagation
    feedforward(trainingExample,inputNeuronsOuput,hiddenNeuronsOutput,outputNeuronsOutput);
    
    //Compute the error of the output layer: the derivative of the function times the error of the output
    for(UINT i=0; i<numOutputNeurons; i++){
		deltaO[i] = outputLayer[i].der(outputNeuronsOutput[i]) * (targetVector[i]-outputNeuronsOutput[i]);
    }
    
    //Compute the error of the hidden layer
    for(UINT i=0; i<numHiddenNeurons; i++){
        double sum = 0;
        for(UINT j=0; j<numOutputNeurons; j++){
            sum += outputLayer[j].weights[i] * deltaO[j];
        }
		deltaH[i] = hiddenLayer[i].der(hiddenNeuronsOutput[i]) * sum;
    }
    
    //Update the hidden weights: old hidden weights + (learningRate * inputToTheHiddenNeuron * deltaHidden )
    for(UINT i=0; i<numHiddenNeurons; i++){
        for(UINT j=0; j<numInputNeurons; j++){
			//Compute the update
            update = alpha * (beta * hiddenLayer[i].previousUpdate[j] + (1.0 - beta) * inputNeuronsOuput[j] * deltaH[i]);

			//Update the weights
			hiddenLayer[i].weights[j] += update;

			//Store the previous update
			hiddenLayer[i].previousUpdate[j] = update; 
        }
    }
    
    //Update the output weights
    for(UINT i=0; i<numOutputNeurons; i++){
        for(UINT j=0; j<numHiddenNeurons; j++){
			//Compute the update
            update = alpha * (beta * outputLayer[i].previousUpdate[j] + (1.0 - beta) * hiddenNeuronsOutput[j] * deltaO[i]);

			//Update the weights
			outputLayer[i].weights[j] += update;

			//Store the update
			outputLayer[i].previousUpdate[j] = update;

        }
    }
    
    //Update the hidden bias
    for(UINT i=0; i<numHiddenNeurons; i++){
		//Compute the update
		update = alpha * (beta * hiddenLayer[i].previousBiasUpdate + (1.0 - beta) * deltaH[i]);

		//Update the bias
        hiddenLayer[i].bias += update;

		//Store the update
		hiddenLayer[i].previousBiasUpdate = update;
    }
    
    //Update the output bias
    for(UINT i=0; i<numOutputNeurons; i++){
		//Compute the update
		update = alpha * (beta * outputLayer[i].previousBiasUpdate + (1.0 - beta) * deltaO[i]);

		//Update the bias
        outputLayer[i].bias += update;
		
		//Stire the update
		outputLayer[i].previousBiasUpdate = update;
    }
    
    //Compute the error 
    double error = 0;
    for(UINT i=0; i<numOutputNeurons; i++){
        error += (targetVector[i]-outputNeuronsOutput[i]) * (targetVector[i]-outputNeuronsOutput[i]);
    }

    return error;
}

vector< double > MLP::feedforward(vector< double > trainingExample){
    
    vector< double > inputNeuronsOuput(numInputNeurons,0);
    vector< double > hiddenNeuronsOutput(numHiddenNeurons,0);
    vector< double > outputNeuronsOutput(numOutputNeurons,0);

	//Scale the input vector if required
	if( useScaling ){
		for(UINT i=0; i<numInputNeurons; i++){
			trainingExample[i] = scale(trainingExample[i],inputVectorRanges[i].minValue,inputVectorRanges[i].maxValue,0.0,1.0);
		}
	}
    
    //Input layer
	vector< double > input(1);
    for(UINT i=0; i<numInputNeurons; i++){
        input[0] = trainingExample[i];
        inputNeuronsOuput[i] = inputLayer[i].fire(input);
    }
    
    //Hidden Layer
    for(UINT i=0; i<numHiddenNeurons; i++){
        hiddenNeuronsOutput[i] = hiddenLayer[i].fire(inputNeuronsOuput);
    }
    
    //Output Layer
    for(UINT i=0; i<numOutputNeurons; i++){
        outputNeuronsOutput[i] = outputLayer[i].fire(hiddenNeuronsOutput);
    }

	//Scale the output vector if required
	if( useScaling ){
		for(unsigned int i=0; i<numOutputNeurons; i++){
			outputNeuronsOutput[i] = scale(outputNeuronsOutput[i],0.0,1.0,targetVectorRanges[i].minValue,targetVectorRanges[i].maxValue);
		}
	}
    
    return outputNeuronsOutput;
    
}

void MLP::feedforward(vector< double > &trainingExample,vector< double > &inputNeuronsOuput,
                      vector< double > &hiddenNeuronsOutput,vector< double > &outputNeuronsOutput){
    
    inputNeuronsOuput.resize(numInputNeurons,0);
    hiddenNeuronsOutput.resize(numHiddenNeurons,0);
    outputNeuronsOutput.resize(numOutputNeurons,0);
    
    //Input layer
	vector< double > input(1);
    for(UINT i=0; i<numInputNeurons; i++){
        input[0] = trainingExample[i];
        inputNeuronsOuput[i] = inputLayer[i].fire(input);
    }
    
    //Hidden Layer
    for(UINT i=0; i<numHiddenNeurons; i++){
        hiddenNeuronsOutput[i] = hiddenLayer[i].fire(inputNeuronsOuput);
    }
    
    //Output Layer
    for(UINT i=0; i<numOutputNeurons; i++){
        outputNeuronsOutput[i] = outputLayer[i].fire(hiddenNeuronsOutput);
    }
    
}

void MLP::printNetwork(){
    cout<<"***************** MLP *****************\n";
    cout<<"NumInputNeurons: "<<numInputNeurons<<endl;
    cout<<"NumHiddenNeurons: "<<numHiddenNeurons<<endl;
    cout<<"NumOutputNeurons: "<<numOutputNeurons<<endl;
    
    cout<<"InputWeights:\n";
    for(UINT i=0; i<numInputNeurons; i++){
        cout<<"Neuron: "<<i<<" Bias: " << inputLayer[i].bias << " Weights: ";
        for(UINT j=0; j<inputLayer[i].weights.size(); j++){
            cout<<inputLayer[i].weights[j]<<"\t";
        }cout<<endl;
    }
    
    cout<<"HiddenWeights:\n";
    for(UINT i=0; i<numHiddenNeurons; i++){
        cout<<"Neuron: "<<i<<" Bias: " << hiddenLayer[i].bias << " Weights: ";
        for(UINT j=0; j<hiddenLayer[i].weights.size(); j++){
            cout<<hiddenLayer[i].weights[j]<<"\t";
        }cout<<endl;
    }
    
    cout<<"OutputWeights:\n";
    for(UINT i=0; i<numOutputNeurons; i++){
        cout<<"Neuron: "<<i<<" Bias: " << outputLayer[i].bias << " Weights: ";
        for(UINT j=0; j<outputLayer[i].weights.size(); j++){
            cout<<outputLayer[i].weights[j]<<"\t";
        }cout<<endl;
    }
    
}

bool MLP::checkForNAN(){
    
    for(UINT i=0; i<numInputNeurons; i++){
        if( isNAN(inputLayer[i].bias) ) return true;
        for(UINT j=0; j<inputLayer[i].weights.size(); j++){
            if( isNAN(inputLayer[i].weights[j]) ) return true;
        }
    }
    
    for(UINT i=0; i<numHiddenNeurons; i++){
        if( isNAN(hiddenLayer[i].bias) ) return true;
        for(UINT j=0; j<hiddenLayer[i].weights.size(); j++){
            if( isNAN(hiddenLayer[i].weights[j]) ) return true;
        }
    }
    
    for(UINT i=0; i<numOutputNeurons; i++){
        if( isNAN(outputLayer[i].bias) ) return true;
        for(UINT j=0; j<outputLayer[i].weights.size(); j++){
            if( isNAN(outputLayer[i].weights[j]) ) return true;
        }
    }
    
    return false;
}

bool inline MLP::isNAN(double v){
    if( v != v ) return true;
    return false;
}

bool MLP::saveMLPToFile(string fileName){

	std::fstream file; 

	if( !trained ){
		cout << "ERROR: The network has not been trained so it can not be saved!\n";
		return false;
	}

	file.open(fileName.c_str(), std::ios::out);

	if( !file.is_open() ){
		return false;
	}

	file << "GRT_MLP_FILE_V1.0\n";
	file << "NumInputNeurons: "<<numInputNeurons<<endl;
	file << "NumHiddenNeurons: "<<numHiddenNeurons<<endl;
	file << "NumOutputNeurons: "<<numOutputNeurons<<endl;
	file << "InputLayerActivationFunction: " <<activationFunctionToString(inputLayerActivationFunction)<< endl;
	file << "HiddenLayerActivationFunction: " <<activationFunctionToString(hiddenLayerActivationFunction)<< endl;
	file << "OutputLayerActivationFunction: " <<activationFunctionToString(outputLayerActivationFunction)<< endl;
	file << "MinNumEpochs: " << minNumEpochs << endl;
	file << "MaxNumEpochs: " << maxNumEpochs << endl;
	file << "NumRandomTrainingIterations: " << numRandomTrainingIterations << endl;
	file << "ValidationSetSize: " << validationSetSize << endl;
	file << "MinChange: " << minChange << endl;
	file << "TrainingRate: " << trainingRate << endl;
	file << "Momentum: " << momentum << endl;
	file << "Gamma: " << gamma << endl;
	file << "UseValidationSet: " << useValidationSet << endl;
	file << "RandomiseTrainingOrder: " << randomiseTrainingOrder << endl;
	file << "UseScaling: " << useScaling << endl;
    file << "ClassificationMode: " << classificationModeActive << endl;
    file << "UseNullRejection: " << useNullRejection << endl;
    file << "RejectionThreshold: " << nullRejectionThreshold << endl;
	
	file << "InputLayer: \n";
	for(UINT i=0; i<numInputNeurons; i++){
		file << "InputNeuron: " << i+1 << endl;
		file << "NumInputs: " << inputLayer[i].numInputs << endl;
		file << "Bias: " << inputLayer[i].bias << endl;
		file << "Gamma: " << inputLayer[i].gamma << endl;
		file << "Weights: " << endl;
		for(UINT j=0; j<inputLayer[i].numInputs; j++){
			file << inputLayer[i].weights[j] << "\t";
		}
		file << endl;
	}
	file << "\n";

	file << "HiddenLayer: \n";
	for(UINT i=0; i<numHiddenNeurons; i++){
		file << "HiddenNeuron: " << i+1 << endl;
		file << "NumInputs: " << hiddenLayer[i].numInputs << endl;
		file << "Bias: " << hiddenLayer[i].bias << endl;
		file << "Gamma: " << hiddenLayer[i].gamma << endl;
		file << "Weights: " << endl;
		for(UINT j=0; j<hiddenLayer[i].numInputs; j++){
			file << hiddenLayer[i].weights[j] << "\t";
		}
		file << endl;
	}
	file << "\n";

	file << "OutputLayer: \n";
	for(UINT i=0; i<numOutputNeurons; i++){
		file << "OutputNeuron: " << i+1 << endl;
		file << "NumInputs: " << outputLayer[i].numInputs << endl;
		file << "Bias: " << outputLayer[i].bias << endl;
		file << "Gamma: " << outputLayer[i].gamma << endl;
		file << "Weights: " << endl;
		for(UINT j=0; j<outputLayer[i].numInputs; j++){
			file << outputLayer[i].weights[j] << "\t";
		}
		file << endl;
	}

	if( useScaling ){
		file << "InputVectorRanges: \n";
		for(UINT j=0; j<numInputNeurons; j++){
			file << inputVectorRanges[j].minValue << "\t" << inputVectorRanges[j].maxValue << endl;
		}
		file << endl;

		file << "OutputVectorRanges: \n";
		for(UINT j=0; j<numOutputNeurons; j++){
			file << targetVectorRanges[j].minValue << "\t" << targetVectorRanges[j].maxValue << endl;
		}
		file << endl;
	}

	file.close();
	return true;
}

bool MLP::loadMLPFromFile(string fileName){

	std::fstream file; 
	string activationFunction;

	//Clear any previous models
	clear();

	file.open(fileName.c_str(), std::ios::in);

	trained = false;

	if( !file.is_open() ){
		return false;
	}

	string word;

	//Check to make sure this is a file with the MLP File Format
	file >> word;
	if(word != "GRT_MLP_FILE_V1.0"){
		file.close();
		cout<<"ERROR: FAILED TO FIND HEADER\n";
		return false;
	}

	file >> word;
	if(word != "NumInputNeurons:"){
		file.close();
		cout<<"ERROR: FAILED TO FIND NumInputNeurons\n";
		return false;
	}
	file >> numInputNeurons;

	file >> word;
	if(word != "NumHiddenNeurons:"){
		file.close();
		cout<<"ERROR: FAILED TO FIND NumHiddenNeurons\n";
		return false;
	}
	file >> numHiddenNeurons;

	file >> word;
	if(word != "NumOutputNeurons:"){
		file.close();
		cout<<"ERROR: FAILED TO FIND NumOutputNeurons\n";
		return false;
	}
	file >> numOutputNeurons;

	file >> word;
	if(word != "InputLayerActivationFunction:"){
		file.close();
		cout<<"ERROR: FAILED TO FIND InputLayerActivationFunction\n";
		return false;
	}
	file >> activationFunction;
	inputLayerActivationFunction = activationFunctionFromString(activationFunction);

	file >> word;
	if(word != "HiddenLayerActivationFunction:"){
		file.close();
		cout<<"ERROR: FAILED TO FIND HiddenLayerActivationFunction\n";
		return false;
	}
	file >> activationFunction;
	hiddenLayerActivationFunction = activationFunctionFromString(activationFunction);

	file >> word;
	if(word != "OutputLayerActivationFunction:"){
		file.close();
		cout<<"ERROR: FAILED TO FIND OutputLayerActivationFunction\n";
		return false;
	}
	file >> activationFunction;
	outputLayerActivationFunction = activationFunctionFromString(activationFunction);

	file >> word;
	if(word != "MinNumEpochs:"){
		file.close();
		cout<<"ERROR: FAILED TO FIND MinNumEpochs\n";
		return false;
	}
	file >> minNumEpochs;

	file >> word;
	if(word != "MaxNumEpochs:"){
		file.close();
		cout<<"ERROR: FAILED TO FIND MaxNumEpochs\n";
		return false;
	}
	file >> maxNumEpochs;

	file >> word;
	if(word != "NumRandomTrainingIterations:"){
		file.close();
		cout<<"ERROR: FAILED TO FIND NumRandomTrainingIterations\n";
		return false;
	}
	file >> numRandomTrainingIterations;

	file >> word;
	if(word != "ValidationSetSize:"){
		file.close();
		cout<<"ERROR: FAILED TO FIND ValidationSetSize\n";
		return false;
	}
	file >> validationSetSize;

	file >> word;
	if(word != "MinChange:"){
		file.close();
		cout<<"ERROR: FAILED TO FIND MinChange\n";
		return false;
	}
	file >> minChange;

	file >> word;
	if(word != "TrainingRate:"){
		file.close();
		cout<<"ERROR: FAILED TO FIND TrainingRate\n";
		return false;
	}
	file >> trainingRate;

	file >> word;
	if(word != "Momentum:"){
		file.close();
		cout<<"ERROR: FAILED TO FIND Momentum\n";
		return false;
	}
	file >> momentum;

	file >> word;
	if(word != "Gamma:"){
		file.close();
		cout<<"ERROR: FAILED TO FIND Gamma\n";
		return false;
	}
	file >> gamma;

	file >> word;
	if(word != "UseValidationSet:"){
		file.close();
		cout<<"ERROR: FAILED TO FIND UseValidationSet\n";
		return false;
	}
	file >> useValidationSet;

	file >> word;
	if(word != "RandomiseTrainingOrder:"){
		file.close();
		cout<<"ERROR: FAILED TO FIND RandomiseTrainingOrder\n";
		return false;
	}
	file >> randomiseTrainingOrder;

	file >> word;
	if(word != "UseScaling:"){
		file.close();
		cout<<"ERROR: FAILED TO FIND UseScaling\n";
		return false;
	}
	file >> useScaling;
    
    file >> word;
	if(word != "ClassificationMode:"){
		file.close();
		cout<<"ERROR: FAILED TO FIND ClassificationMode\n";
		return false;
	}
	file >> classificationModeActive;
    
    file >> word;
	if(word != "UseNullRejection:"){
		file.close();
		cout<<"ERROR: FAILED TO FIND UseNullRejection\n";
		return false;
	}
	file >> useNullRejection;
    
    file >> word;
	if(word != "RejectionThreshold:"){
		file.close();
		cout<<"ERROR: FAILED TO FIND RejectionThreshold\n";
		return false;
	}
	file >> nullRejectionThreshold;

	//Resize the layers
	inputLayer.resize( numInputNeurons );
	hiddenLayer.resize( numHiddenNeurons );
	outputLayer.resize( numOutputNeurons );

	//Load the neuron data
	file >> word;
	if(word != "InputLayer:"){
		file.close();
		cout<<"ERROR: FAILED TO FIND InputLayer\n";
		return false;
	}

	for(UINT i=0; i<numInputNeurons; i++){
		UINT tempNeuronID = 0;

		file >> word;
		if(word != "InputNeuron:"){
			file.close();
			cout<<"ERROR: FAILED TO FIND InputNeuron\n";
			return false;
		}
		file >> tempNeuronID;

		if( tempNeuronID != i+1 ){
			file.close();
			cout<<"ERROR: InputNeuron ID does not match!\n";
			return false;
		}

		file >> word;
		if(word != "NumInputs:"){
			file.close();
			cout<<"ERROR: FAILED TO FIND NumInputs\n";
			return false;
		}
		file >> inputLayer[i].numInputs;

		//Resize the buffers
		inputLayer[i].weights.resize( inputLayer[i].numInputs );

		file >> word;
		if(word != "Bias:"){
			file.close();
			cout<<"ERROR: FAILED TO FIND Bias\n";
			return false;
		}
		file >> inputLayer[i].bias;

		file >> word;
		if(word != "Gamma:"){
			file.close();
			cout<<"ERROR: FAILED TO FIND Gamma\n";
			return false;
		}
		file >> inputLayer[i].gamma;

		file >> word;
		if(word != "Weights:"){
			file.close();
			cout<<"ERROR: FAILED TO FIND Weights\n";
			return false;
		}

		for(UINT j=0; j<inputLayer[i].numInputs; j++){
			file >> inputLayer[i].weights[j];
		}
	}

	//Load the Hidden Layer
	file >> word;
	if(word != "HiddenLayer:"){
		file.close();
		cout<<"ERROR: FAILED TO FIND HiddenLayer\n";
		return false;
	}

	for(UINT i=0; i<numHiddenNeurons; i++){
		UINT tempNeuronID = 0;

		file >> word;
		if(word != "HiddenNeuron:"){
			file.close();
			cout<<"ERROR: FAILED TO FIND HiddenNeuron\n";
			return false;
		}
		file >> tempNeuronID;

		if( tempNeuronID != i+1 ){
			file.close();
			cout<<"ERROR: InputNeuron ID does not match!\n";
			return false;
		}

		file >> word;
		if(word != "NumInputs:"){
			file.close();
			cout<<"ERROR: FAILED TO FIND NumInputs\n";
			return false;
		}
		file >> hiddenLayer[i].numInputs;

		//Resize the buffers
		hiddenLayer[i].weights.resize( hiddenLayer[i].numInputs );

		file >> word;
		if(word != "Bias:"){
			file.close();
			cout<<"ERROR: FAILED TO FIND Bias\n";
			return false;
		}
		file >> hiddenLayer[i].bias;

		file >> word;
		if(word != "Gamma:"){
			file.close();
			cout<<"ERROR: FAILED TO FIND Gamma\n";
			return false;
		}
		file >> hiddenLayer[i].gamma;

		file >> word;
		if(word != "Weights:"){
			file.close();
			cout<<"ERROR: FAILED TO FIND Weights\n";
			return false;
		}

		for(unsigned int j=0; j<hiddenLayer[i].numInputs; j++){
			file >> hiddenLayer[i].weights[j];
		}
	}

	//Load the Output Layer
	file >> word;
	if(word != "OutputLayer:"){
		file.close();
		cout<<"ERROR: FAILED TO FIND OutputLayer\n";
		return false;
	}

	for(UINT i=0; i<numOutputNeurons; i++){
		UINT tempNeuronID = 0;

		file >> word;
		if(word != "OutputNeuron:"){
			file.close();
			cout<<"ERROR: FAILED TO FIND OutputNeuron\n";
			return false;
		}
		file >> tempNeuronID;

		if( tempNeuronID != i+1 ){
			file.close();
			cout<<"ERROR: InputNeuron ID does not match!\n";
			return false;
		}

		file >> word;
		if(word != "NumInputs:"){
			file.close();
			cout<<"ERROR: FAILED TO FIND NumInputs\n";
			return false;
		}
		file >> outputLayer[i].numInputs;

		//Resize the buffers
		outputLayer[i].weights.resize( outputLayer[i].numInputs );

		file >> word;
		if(word != "Bias:"){
			file.close();
			cout<<"ERROR: FAILED TO FIND Bias\n";
			return false;
		}
		file >> outputLayer[i].bias;

		file >> word;
		if(word != "Gamma:"){
			file.close();
			cout<<"ERROR: FAILED TO FIND Gamma\n";
			return false;
		}
		file >> outputLayer[i].gamma;

		file >> word;
		if(word != "Weights:"){
			file.close();
			cout<<"ERROR: FAILED TO FIND Weights\n";
			return false;
		}

		for(UINT j=0; j<outputLayer[i].numInputs; j++){
			file >> outputLayer[i].weights[j];
		}
	}

	if( useScaling ){
		//Resize the ranges buffers
		inputVectorRanges.resize( numInputNeurons );
		targetVectorRanges.resize( numOutputNeurons );

		//Load the ranges
		file >> word;
		if(word != "InputVectorRanges:"){
			file.close();
			cout<<"ERROR: FAILED TO FIND InputVectorRanges\n";
			return false;
		}
		for(UINT j=0; j<inputVectorRanges.size(); j++){
			file >> inputVectorRanges[j].minValue;
			file >> inputVectorRanges[j].maxValue;
		}

		file >> word;
		if(word != "OutputVectorRanges:"){
			file.close();
			cout<<"ERROR: FAILED TO FIND OutputVectorRanges\n";
			return false;
		}
		for(UINT j=0; j<targetVectorRanges.size(); j++){
			file >> targetVectorRanges[j].minValue;
			file >> targetVectorRanges[j].maxValue;
		}
	}

    initialized = true;
	trained = true;

	file.close();
	return true;
}

string MLP::activationFunctionToString(unsigned int activationFunction){
	string activationName;

	switch(activationFunction){
		case(Neuron::LINEAR):
			activationName = "LINEAR";
			break;
		case(Neuron::SIGMOID):
			activationName = "SIGMOID";
			break;
		case(Neuron::BIPOLAR_SIGMOID):
			activationName = "BIPOLAR_SIGMOID";
			break;
		default:
			activationName = "UNKNOWN";
			break;
	}

	return activationName;
}

UINT MLP::activationFunctionFromString(string activationName){
	UINT activationFunction = 0;

	if(activationName == "LINEAR" ){
		activationFunction = 0;
		return activationFunction;
	}
	if(activationName == "SIGMOID" ){
		activationFunction = 1;
		return activationFunction;
	}
	if(activationName == "BIPOLAR_SIGMOID" ){
		activationFunction = 2;
		return activationFunction;
	}
	return activationFunction;
}

bool MLP::validateActivationFunction(UINT actvationFunction){
	if( actvationFunction >= Neuron::LINEAR && actvationFunction < Neuron::NUMBER_OF_ACTIVATION_FUNCTIONS ) return true;
	return false;
}

bool MLP::setMinChange(double minChange){
	this->minChange = minChange;
	return true;
}

bool MLP::setTrainingRate(double trainingRate){
	if( trainingRate >= 0 && trainingRate <= 1.0 ){
		this->trainingRate = trainingRate;
		return true;
	}
	return false;
}

bool MLP::setMomentum(double momentum){
	if( momentum >= 0 && momentum <= 1.0 ){
		this->momentum = momentum;
		return true;
	}
	return false;
}

bool MLP::setGamma(double gamma){
	this->gamma = gamma;
	return true;
}

bool MLP::setUseValidationSet(bool useValidationSet){
	this->useValidationSet = useValidationSet;
	return true;
}

bool MLP::setRandomiseTrainingOrder(bool randomiseTrainingOrder){
	this->randomiseTrainingOrder = randomiseTrainingOrder;
	return true;
}
    
bool MLP::setUseMultiThreadingTraining(bool useMultiThreadingTraining){
    this->useMultiThreadingTraining = useMultiThreadingTraining;
    return true;
}
bool MLP::setMinNumEpochs(UINT minNumEpochs){
    if( minNumEpochs > 0 ){
        this->minNumEpochs = minNumEpochs;
        return true;
    }
    return false;
}
bool MLP::setMaxNumEpochs(UINT maxNumEpochs){
    if( maxNumEpochs > 0 ){
        this->maxNumEpochs = maxNumEpochs;
        return true;
    }
    return false;

}
bool MLP::setNumRandomTrainingIterations(UINT numRandomTrainingIterations){
    if( numRandomTrainingIterations > 0 ){
        this->numRandomTrainingIterations = numRandomTrainingIterations;
        return true;
    }
    return false;
}
bool MLP::setValidationSetSize(UINT validationSetSize){
    if( validationSetSize > 0 && validationSetSize < 100 ){
        this->validationSetSize = validationSetSize;
        return true;
    }
    return false;
}
    
} //End of namespace GRT