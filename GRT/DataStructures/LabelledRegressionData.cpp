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

#include "LabelledRegressionData.h"

namespace GRT{

LabelledRegressionData::LabelledRegressionData(UINT numInputDimensions,UINT numTargetDimensions,string datasetName,string infoText):totalNumSamples(0){
    this->numInputDimensions = numInputDimensions;
    this->numTargetDimensions = numTargetDimensions;
    this->datasetName = datasetName;
    this->infoText = infoText;
    kFoldValue = 0;
    crossValidationSetup = false;
    useExternalRanges = false;
    debugLog.setProceedingText("[DEBUG LRD]");
    errorLog.setProceedingText("[ERROR LRD]");
    warningLog.setProceedingText("[WARNING LRD]");
}

LabelledRegressionData::LabelledRegressionData(const LabelledRegressionData &rhs){
    this->datasetName = rhs.datasetName;
    this->infoText = rhs.infoText;
    this->numInputDimensions = rhs.numInputDimensions;
    this->numTargetDimensions = rhs.numTargetDimensions;
    this->totalNumSamples = rhs.totalNumSamples;
    this->kFoldValue = rhs.kFoldValue;
    this->crossValidationSetup = rhs.crossValidationSetup;
    this->useExternalRanges = rhs.useExternalRanges;
    this->externalInputRanges = rhs.externalInputRanges;
    this->externalTargetRanges = rhs.externalTargetRanges;
    this->data = rhs.data;
    this->crossValidationIndexs = rhs.crossValidationIndexs;
    debugLog.setProceedingText("[DEBUG LRD]");
    errorLog.setProceedingText("[ERROR LRD]");
    warningLog.setProceedingText("[WARNING LRD]");
}

LabelledRegressionData::~LabelledRegressionData(){}

void LabelledRegressionData::clear(){
    totalNumSamples = 0;
    kFoldValue = 0;
    crossValidationSetup = false;
    data.clear();
    crossValidationIndexs.clear();
}

bool LabelledRegressionData::setInputAndTargetDimensions(UINT numInputDimensions,UINT numTargetDimensions){
	clear();
    if( numInputDimensions > 0 && numTargetDimensions > 0 ){
        this->numInputDimensions = numInputDimensions;
        this->numTargetDimensions = numTargetDimensions;

        //Clear the external ranges
        useExternalRanges = false;
        externalInputRanges.clear();
        externalTargetRanges.clear();
        return true;
    }
    errorLog << "setInputAndTargetDimensions(UINT numInputDimensions,UINT numTargetDimensions) - The number of input and target dimensions should be greater than zero!" << endl;
    return false;
}

bool LabelledRegressionData::setDatasetName(string datasetName){

    //Make sure there are no spaces in the string
    if( datasetName.find(" ") == string::npos ){
        this->datasetName = datasetName;
        return true;
    }

    errorLog << "setDatasetName(string datasetName) - The dataset name cannot contain any spaces!" << endl;
    return false;
}

bool LabelledRegressionData::setInfoText(string infoText){
    this->infoText = infoText;
    return true;
}

bool LabelledRegressionData::addSample(vector<double> inputVector,vector<double> targetVector){
	if( inputVector.size() == numInputDimensions && targetVector.size() == numTargetDimensions ){
        data.push_back( LabelledRegressionSample(inputVector,targetVector) );
        totalNumSamples++;

        //The dataset has changed so flag that any previous cross validation setup will now not work
        crossValidationSetup = false;
        crossValidationIndexs.clear();
        return true;
    }
    errorLog << "addSample(vector<double> inputVector,vector<double> targetVector) - The inputVector size or targetVector size does not match the size of the numInputDimensions or numTargetDimensions" << endl;
    return false;
}

bool LabelledRegressionData::removeLastSample(){
	if( totalNumSamples > 0 ){
		//Remove the training example from the buffer
		data.erase(data.end()-1);
		totalNumSamples = (UINT)data.size();

        //The dataset has changed so flag that any previous cross validation setup will now not work
        crossValidationSetup = false;
        crossValidationIndexs.clear();
		return true;
	}
    warningLog << "removeLastSample() - There are no samples to remove!" << endl;
    return false;
}

bool LabelledRegressionData::setExternalRanges(vector< MinMax > externalInputRanges, vector< MinMax > externalTargetRanges, bool useExternalRanges){

    if( externalInputRanges.size() != numInputDimensions ) return false;
    if( externalTargetRanges.size() != numTargetDimensions ) return false;

    this->externalInputRanges = externalInputRanges;
    this->externalTargetRanges = externalTargetRanges;
    this->useExternalRanges = useExternalRanges;

    return true;
}

bool LabelledRegressionData::enableExternalRangeScaling(bool useExternalRanges){
    if( externalInputRanges.size() != numInputDimensions && externalTargetRanges.size() != numTargetDimensions  ){
        this->useExternalRanges = useExternalRanges;
        return true;
    }
    return false;
}

bool LabelledRegressionData::scale(double minTarget,double maxTarget){
    vector< MinMax > inputRanges = getInputRanges();
    vector< MinMax > targetRanges = getTargetRanges();
    return scale(inputRanges,targetRanges,minTarget,maxTarget);
}
bool LabelledRegressionData::scale(vector< MinMax > inputVectorRanges,vector< MinMax > targetVectorRanges,double minTarget,double maxTarget){
    if( inputVectorRanges.size() == numInputDimensions && targetVectorRanges.size() == numTargetDimensions ){

        vector< double > scaledInputVector(numInputDimensions,0);
        vector< double > scaledTargetVector(numTargetDimensions,0);
        for(UINT i=0; i<totalNumSamples; i++){

            //Scale the input vector
            for(UINT j=0; j<numInputDimensions; j++){
                scaledInputVector[j] = scale(data[i].getInputVectorValue(j),inputVectorRanges[j].minValue,inputVectorRanges[j].maxValue,minTarget,maxTarget);
            }
            //Scale the target vector
            for(UINT j=0; j<numTargetDimensions; j++){
                scaledTargetVector[j] = scale(data[i].getTargetVectorValue(j),targetVectorRanges[j].minValue,targetVectorRanges[j].maxValue,minTarget,maxTarget);
            }
            //Update the training sample with the scaled data
            data[i].set(scaledInputVector,scaledTargetVector);
        }

        return true;
    }
    return false;
}

double inline LabelledRegressionData::scale(double x,double minSource,double maxSource,double minTarget,double maxTarget){
    return (((x-minSource)*(maxTarget-minTarget))/(maxSource-minSource))+minTarget;
}

vector<MinMax> LabelledRegressionData::getInputRanges(){

    if( useExternalRanges ) return externalInputRanges;

	vector< MinMax > ranges(numInputDimensions);

	if( totalNumSamples > 0 ){
		for(UINT j=0; j<numInputDimensions; j++){
			ranges[j].minValue = data[0].getInputVectorValue(j);
			ranges[j].maxValue = data[0].getInputVectorValue(j);
			for(UINT i=0; i<totalNumSamples; i++){
				if( data[i].getInputVectorValue(j) < ranges[j].minValue ){ ranges[j].minValue = data[i].getInputVectorValue(j); }		//Search for the min value
				else if( data[i].getInputVectorValue(j) > ranges[j].maxValue ){ ranges[j].maxValue = data[i].getInputVectorValue(j); }	//Search for the max value
			}
		}
	}
	return ranges;
}

vector<MinMax> LabelledRegressionData::getTargetRanges(){

    if( useExternalRanges ) return externalTargetRanges;

    vector< MinMax > ranges(numTargetDimensions);

    if( totalNumSamples > 0 ){
        for(UINT j=0; j<numTargetDimensions; j++){
            ranges[j].minValue = data[0].getTargetVectorValue(j);
            ranges[j].maxValue = data[0].getTargetVectorValue(j);
            for(UINT i=0; i<totalNumSamples; i++){
                if( data[i].getTargetVectorValue(j) < ranges[j].minValue ){ ranges[j].minValue = data[i].getTargetVectorValue(j); }		//Search for the min value
                else if( data[i].getTargetVectorValue(j) > ranges[j].maxValue ){ ranges[j].maxValue = data[i].getTargetVectorValue(j); }	//Search for the max value
            }
        }
    }
    return ranges;
}

bool LabelledRegressionData::printStats(){
    
    cout << "DatasetName:\t" << datasetName << endl;
    cout << "DatasetInfo:\t" << infoText << endl;
    cout << "Number of Input Dimensions:\t" << numInputDimensions << endl;
    cout << "Number of Target Dimensions:\t" << numTargetDimensions << endl;
    cout << "Number of Samples:\t" << totalNumSamples << endl;
    
    vector< MinMax > inputRanges = getInputRanges();
    
    cout << "Dataset Input Dimension Ranges:\n";
    for(UINT j=0; j<inputRanges.size(); j++){
        cout << "[" << j+1 << "] Min:\t" << inputRanges[j].minValue << "\tMax: " << inputRanges[j].maxValue << endl;
    }
    
    vector< MinMax > targetRanges = getTargetRanges();
    
    cout << "Dataset Target Dimension Ranges:\n";
    for(UINT j=0; j<targetRanges.size(); j++){
        cout << "[" << j+1 << "] Min:\t" << targetRanges[j].minValue << "\tMax: " << targetRanges[j].maxValue << endl;
    }
    
    return true;
}
    
LabelledRegressionData LabelledRegressionData::partition(UINT trainingSizePercentage){

	//Partitions the dataset into a training dataset (which is kept by this instance of the LabelledRegressionData) and
	//a testing/validation dataset (which is return as a new instance of the LabelledRegressionData).  The trainingSizePercentage
	//therefore sets the size of the data which remains in this instance and the remaining percentage of data is then added to
	//the testing/validation dataset

	const UINT numTrainingExamples = (UINT) floor( double(totalNumSamples) / 100.0 * double(trainingSizePercentage) );

	LabelledRegressionData trainingSet(numInputDimensions,numTargetDimensions);
	LabelledRegressionData testSet(numInputDimensions,numTargetDimensions);
	vector< UINT > indexs( totalNumSamples );

	//Create the random partion indexs
	Random random;
	for(UINT i=0; i<totalNumSamples; i++) indexs[i] = i;
	for(UINT x=0; x<totalNumSamples*100; x++){
		UINT indexA = random.getRandomNumberInt(0,totalNumSamples);
		UINT indexB = random.getRandomNumberInt(0,totalNumSamples);
		UINT temp = indexs[ indexA ];
		indexs[ indexA ] = indexs[ indexB ];
		indexs[ indexB ] = temp;
	}

	//Add the data to the training and test sets
	for(UINT i=0; i<numTrainingExamples; i++){
		trainingSet.addSample( data[ indexs[i] ].getInputVector(), data[ indexs[i] ].getTargetVector() );
	}
	for(UINT i=numTrainingExamples; i<totalNumSamples; i++){
		testSet.addSample( data[ indexs[i] ].getInputVector(), data[ indexs[i] ].getTargetVector() );
	}

	//Overwrite the training data in this instance with the training data of the trainingSet
	data = trainingSet.getData();
	totalNumSamples = trainingSet.getNumSamples();

    //The dataset has changed so flag that any previous cross validation setup will now not work
    crossValidationSetup = false;
    crossValidationIndexs.clear();

	return testSet;
}

bool LabelledRegressionData::merge(LabelledRegressionData &regressionData){

    if( regressionData.getNumInputDimensions() != numInputDimensions ){
        errorLog << "merge(LabelledRegressionData &regressionData) - The number of input dimensions in the regressionData (" << regressionData.getNumInputDimensions() << ") does not match the number of input dimensions of this dataset (" << numInputDimensions << ")" << endl;
        return false;
    }

    if( regressionData.getNumTargetDimensions() != numTargetDimensions ){
        errorLog << "merge(LabelledRegressionData &regressionData) - The number of target dimensions in the regressionData (" << regressionData.getNumTargetDimensions() << ") does not match the number of target dimensions of this dataset (" << numTargetDimensions << ")" << endl;
        return false;
    }

    //Add the data from the labelledData to this instance
    for(UINT i=0; i<regressionData.getNumSamples(); i++){
        addSample(regressionData[i].getInputVector(), regressionData[i].getTargetVector());
    }

    //The dataset has changed so flag that any previous cross validation setup will now not work
    crossValidationSetup = false;
    crossValidationIndexs.clear();

    return true;
}

bool LabelledRegressionData::spiltDataIntoKFolds(UINT K){

    crossValidationSetup = false;
    crossValidationIndexs.clear();

    //K can not be zero
    if( K > totalNumSamples ){
        errorLog << "spiltDataIntoKFolds(UINT K) - K can not be zero!" << endl;
        return false;
    }

    //K can not be larger than the number of examples
    if( K > totalNumSamples ){
        errorLog << "spiltDataIntoKFolds(UINT K) - K can not be larger than the total number of samples in the dataset!" << endl;
        return false;
    }

    //Setup the dataset for k-fold cross validation
    kFoldValue = K;
    vector< UINT > indexs( totalNumSamples );

    //Work out how many samples are in each fold, the last fold might have more samples than the others
    UINT numSamplesPerFold = (UINT) floor( totalNumSamples/double(K) );

    //Add the random indexs to each fold
    crossValidationIndexs.resize(K);

    //Create the random partion indexs
    Random random;
    UINT indexA = 0;
    UINT indexB = 0;
    UINT temp = 0;


    //Randomize the order of the data
    for(UINT i=0; i<totalNumSamples; i++) indexs[i] = i;
    for(UINT x=0; x<totalNumSamples*1000; x++){
        //Pick two random indexs
        indexA = random.getRandomNumberInt(0,totalNumSamples);
        indexB = random.getRandomNumberInt(0,totalNumSamples);

        //Swap the indexs
        temp = indexs[ indexA ];
        indexs[ indexA ] = indexs[ indexB ];
        indexs[ indexB ] = temp;
    }

    UINT counter = 0;
    UINT foldIndex = 0;
    for(UINT i=0; i<totalNumSamples; i++){
        //Add the index to the current fold
        crossValidationIndexs[ foldIndex ].push_back( indexs[i] );

        //Move to the next fold if ready
        if( ++counter == numSamplesPerFold && foldIndex < K ){
            foldIndex++;
            counter = 0;
        }
    }

    crossValidationSetup = true;
    return true;

}

LabelledRegressionData LabelledRegressionData::getTrainingFoldData(UINT foldIndex){
    LabelledRegressionData trainingData;

    if( !crossValidationSetup ){
        errorLog << "getTrainingFoldData(UINT foldIndex) - Cross Validation has not been setup! You need to call the spiltDataIntoKFolds(UINT K,bool useStratifiedSampling) function first before calling this function!" << endl;
        return trainingData;
    }

    if( foldIndex >= kFoldValue ) return trainingData;

    trainingData.setInputAndTargetDimensions(numInputDimensions, numTargetDimensions);

    //Add the data to the training set, this will consist of all the data that is NOT in the foldIndex
    UINT index = 0;
    for(UINT k=0; k<kFoldValue; k++){
        if( k != foldIndex ){
            for(UINT i=0; i<crossValidationIndexs[k].size(); i++){

                index = crossValidationIndexs[k][i];
                trainingData.addSample( data[ index ].getInputVector(), data[ index ].getTargetVector() );
            }
        }
    }

    return trainingData;
}

LabelledRegressionData LabelledRegressionData::getTestFoldData(UINT foldIndex){
    LabelledRegressionData testData;

    if( !crossValidationSetup ) return testData;

    if( foldIndex >= kFoldValue ) return testData;

    //Add the data to the training
    testData.setInputAndTargetDimensions(numInputDimensions, numTargetDimensions);

    UINT index = 0;
    for(UINT i=0; i<crossValidationIndexs[ foldIndex ].size(); i++){

        index = crossValidationIndexs[ foldIndex ][i];
        testData.addSample( data[ index ].getInputVector(), data[ index ].getTargetVector() );
    }

    return testData;
}


bool LabelledRegressionData::saveDatasetToFile(string filename){

	std::fstream file;
	file.open(filename.c_str(), std::ios::out);

	if( !file.is_open() ){
        errorLog << "saveDatasetToFile(string filename) - Failed to open file!" << endl;
		return false;
	}

	file << "GRT_LABELLED_REGRESSION_DATA_FILE_V1.0\n";
    file << "DatasetName: " << datasetName << endl;
    file << "InfoText: " << infoText << endl;
	file << "NumInputDimensions: "<<numInputDimensions<<endl;
	file << "NumTargetDimensions: "<<numTargetDimensions<<endl;
	file << "TotalNumTrainingExamples: "<<totalNumSamples<<endl;
    file << "UseExternalRanges: " << useExternalRanges << endl;

    if( useExternalRanges ){
        for(UINT i=0; i<externalInputRanges.size(); i++){
            file << externalInputRanges[i].minValue << "\t" << externalInputRanges[i].maxValue << endl;
        }
        for(UINT i=0; i<externalTargetRanges.size(); i++){
            file << externalTargetRanges[i].minValue << "\t" << externalTargetRanges[i].maxValue << endl;
        }
    }

	file << "LabelledRegressionData:\n";

	for(UINT i=0; i<totalNumSamples; i++){
		for(UINT j=0; j<numInputDimensions; j++){
			file << data[i].getInputVectorValue(j) << "\t";
		}
		for(UINT j=0; j<numTargetDimensions; j++){
			file << data[i].getTargetVectorValue(j);
			if( j!= numTargetDimensions-1 ) file << "\t";
		}
		file << endl;
	}

	file.close();
	return true;
}

bool LabelledRegressionData::loadDatasetFromFile(string filename){

	std::fstream file;
	file.open(filename.c_str(), std::ios::in);
	clear();

	if( !file.is_open() ){
        errorLog << "loadDatasetFromFile(string filename) - Failed to open file!" << endl;
		return false;
	}

	string word;

	//Check to make sure this is a file with the Training File Format
	file >> word;
	if(word != "GRT_LABELLED_REGRESSION_DATA_FILE_V1.0"){
        errorLog << "loadDatasetFromFile(string filename) - Unknown file header!" << endl;
		file.close();
		return false;
	}

    //Get the name of the dataset
	file >> word;
	if(word != "DatasetName:"){
        errorLog << "loadDatasetFromFile(string filename) - failed to find DatasetName!" << endl;
		file.close();
		return false;
	}
	file >> datasetName;

    file >> word;
	if(word != "InfoText:"){
        errorLog << "loadDatasetFromFile(string filename) - failed to find InfoText!" << endl;
		file.close();
		return false;
	}

    //Load the info text
    file >> word;
    infoText = "";
    while( word != "NumInputDimensions:" ){
        infoText += word + " ";
        file >> word;
    }

	//Get the number of input dimensions in the training data
	if(word != "NumInputDimensions:"){
        errorLog << "loadDatasetFromFile(string filename) - Failed to find NumInputDimensions!" << endl;
		file.close();
		return false;
	}
	file >> numInputDimensions;

	//Get the number of target dimensions in the training data
	file >> word;
	if(word != "NumTargetDimensions:"){
        errorLog << "loadDatasetFromFile(string filename) - Failed to find NumTargetDimensions!" << endl;
		file.close();
		return false;
	}
	file >> numTargetDimensions;

	//Get the total number of training examples in the training data
	file >> word;
	if(word != "TotalNumTrainingExamples:"){
        errorLog << "loadDatasetFromFile(string filename) - Failed to find TotalNumTrainingExamples!" << endl;
		file.close();
		return false;
	}
	file >> totalNumSamples;

    //Check if the dataset should be scaled using external ranges
	file >> word;
	if(word != "UseExternalRanges:"){
        errorLog << "loadDatasetFromFile(string filename) - failed to find DatasetName!" << endl;
		file.close();
		return false;
	}
    file >> useExternalRanges;

    //If we are using external ranges then load them
    if( useExternalRanges ){
        externalInputRanges.resize(numInputDimensions);
        externalTargetRanges.resize(numTargetDimensions);
        for(UINT i=0; i<externalInputRanges.size(); i++){
            file >> externalInputRanges[i].minValue;
            file >> externalInputRanges[i].maxValue;
        }
        for(UINT i=0; i<externalTargetRanges.size(); i++){
            file >> externalTargetRanges[i].minValue;
            file >> externalTargetRanges[i].maxValue;
        }
    }

	//Get the main training data
	file >> word;
	if(word != "LabelledRegressionData:"){
        errorLog << "loadDatasetFromFile(string filename) - Failed to find LabelledRegressionData!" << endl;
		file.close();
		return false;
	}

	vector< double > inputVector(numInputDimensions);
	vector< double > targetVector(numTargetDimensions);
	data.resize( totalNumSamples, LabelledRegressionSample(inputVector,targetVector) );

	for(UINT i=0; i<totalNumSamples; i++){
		//Read the input vector
		for(UINT j=0; j<numInputDimensions; j++){
			file >> inputVector[j];
		}
		for(UINT j=0; j<numTargetDimensions; j++){
			file >> targetVector[j];
		}
        data[i].set(inputVector, targetVector);
	}

	file.close();
	return true;
}

bool LabelledRegressionData::saveDatasetToCSVFile(string filename){

    std::fstream file;
	file.open(filename.c_str(), std::ios::out );

	if( !file.is_open() ){
        errorLog << "saveDatasetToCSVFile(string filename) - Failed to open file!" << endl;
		return false;
	}

    //Write the data to the CSV file
    for(UINT i=0; i<totalNumSamples; i++){
		for(UINT j=0; j<numInputDimensions; j++){
			file << data[i].getInputVector()[j] << ",";
		}
        for(UINT j=0; j<numTargetDimensions; j++){
			file << data[i].getTargetVector()[j];
            if( j != numTargetDimensions-1 ) file << ",";
		}
		file << endl;
	}

	file.close();

    return true;
}

bool LabelledRegressionData::loadDatasetFromCSVFile(string filename,UINT numInputDimensions,UINT numTargetDimensions){

    fstream file;
    string value;
    clear();
    UINT numDimensions = 0;
    datasetName = "NOT_SET";
    infoText = "";

    //Clear any previous data
    clear();

    //Try and open the file
    file.open( filename.c_str(), std::ios::in );
    if( !file.is_open() ){
        return false;
    }

    //Read the first line to work out how many features are in the data to make sure it matches what we have been told is in the file
    getline( file, value );
    for(UINT i=0; i<value.size(); i++){
        if( value[i] == ',' ){
            numDimensions++;
        }
    }

    //If there are no commas in the first line then the data is not in the correct format
    if( numDimensions+1 != numInputDimensions + numTargetDimensions ){
        errorLog << "loadDatasetFromCSVFile(string filename,UINT numInputDimensions,UINT numTargetDimensions) - There are only " << numDimensions+1 << " columns in the file!" << endl;
        return false;
    }

    //Setup the labelled classification data
    setInputAndTargetDimensions(numInputDimensions, numTargetDimensions);

    //Reset the file to read from the start
    file.seekg( ios_base::beg );

    //Read the data
    bool keepParsing = true;
    UINT index = 0;
    UINT lastCommaIndex = 0;
    UINT variableCounter = 0;
    UINT lineCounter = 0;
    vector< double > inputSample(numInputDimensions,0);
    vector< double > targetSample(numTargetDimensions,0);
    while ( file.good() )
    {
        //Read an entire line of data
        getline( file, value );

        lineCounter++;

        //Parse the line
        keepParsing = true;
        index = 0;
        lastCommaIndex = 0;
        variableCounter = 0;

        if( value.length() > 0 ){
            do{
                if( value[index] == ',' ){
                    if( variableCounter < numInputDimensions ){
                        inputSample[ variableCounter ] = stringToDouble( value.substr(0,index-2) );
                        lastCommaIndex = index;
                        variableCounter++;
                    }else{
                        targetSample[ variableCounter-numInputDimensions ] = stringToDouble( value.substr(lastCommaIndex+1,index-2) );
                        lastCommaIndex = index;

                        if( ++variableCounter == numDimensions+1 ){
                            variableCounter = 0;
                            keepParsing = false;
                            if( !addSample(inputSample, targetSample) ){
                                errorLog << "loadDatasetFromCSVFile(string filename,UINT numInputDimensions,UINT numTargetSamples) - Failed to add sample from file!" << endl;
                                file.close();
                                return false;
                            }
                        }
                    }

                }


                if( ++index == value.length() ){
                    targetSample[ variableCounter-numInputDimensions ] = stringToDouble( value.substr(lastCommaIndex+1,value.length()) );
                    lastCommaIndex = index;

                    if( ++variableCounter == numDimensions+1 ){
                        variableCounter = 0;
                        keepParsing = false;
                        if( !addSample(inputSample, targetSample) ){
                            errorLog << "loadDatasetFromCSVFile(string filename,UINT numInputDimensions,UINT numTargetSamples) - Failed to add sample from file!" << endl;
                            file.close();
                            return false;
                        }
                    }

                    keepParsing = false;
                }

            }while( keepParsing );
        }

    }

    file.close();
    return true;
}

double LabelledRegressionData::stringToDouble(string value){
    std::stringstream s( value );
    double d;
    s >> d;
    return d;
}


} //End of namespace GRT

