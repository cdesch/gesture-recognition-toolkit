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

#include "LabelledClassificationData.h"

namespace GRT{

LabelledClassificationData::LabelledClassificationData(UINT numDimensions,string datasetName,string infoText):debugLog("[DEBUG LCD]"),errorLog("[ERROR LCD]"),warningLog("[WARNING LCD]"){
    this->datasetName = datasetName;
    this->numDimensions = numDimensions;
    this->infoText = infoText;
    totalNumSamples = 0;
    crossValidationSetup = false;
    useExternalRanges = false;
    if( numDimensions > 0 ) setNumDimensions( numDimensions );
}

LabelledClassificationData::LabelledClassificationData(const LabelledClassificationData &rhs):debugLog("[DEBUG LCD]"),errorLog("[ERROR LCD]"),warningLog("[WARNING LCD]"){
    this->datasetName = rhs.datasetName;
    this->infoText = rhs.infoText;
    this->numDimensions = rhs.numDimensions;
    this->totalNumSamples = rhs.totalNumSamples;
    this->kFoldValue = rhs.kFoldValue;
    this->crossValidationSetup = rhs.crossValidationSetup;
    this->useExternalRanges = rhs.useExternalRanges;
    this->externalRanges = rhs.externalRanges;
    this->classTracker = rhs.classTracker;
    this->data = rhs.data;
    this->crossValidationIndexs = rhs.crossValidationIndexs;
    this->debugLog = rhs.debugLog;
    this->errorLog = rhs.errorLog;
    this->warningLog = rhs.warningLog;
}

LabelledClassificationData::~LabelledClassificationData(){};

void LabelledClassificationData::clear(){
	totalNumSamples = 0;
	data.clear();
	classTracker.clear();
    crossValidationSetup = false;
    crossValidationIndexs.clear();
}

bool LabelledClassificationData::setNumDimensions(UINT numDimensions){

    if( numDimensions > 0 ){
        //Clear any previous training data
        clear();

        //Set the dimensionality of the data
        this->numDimensions = numDimensions;

        //Clear the external ranges
        useExternalRanges = false;
        externalRanges.clear();

        return true;
    }

    errorLog << "setNumDimensions(UINT numDimensions) - The number of dimensions of the dataset must be greater than zero!" << endl;
    return false;
}

bool LabelledClassificationData::setDatasetName(string datasetName){

    //Make sure there are no spaces in the string
    if( datasetName.find(" ") == string::npos ){
        this->datasetName = datasetName;
        return true;
    }

    errorLog << "setDatasetName(string datasetName) - The dataset name cannot contain any spaces!" << endl;
    return false;
}

bool LabelledClassificationData::setInfoText(string infoText){
    this->infoText = infoText;
    return true;
}

bool LabelledClassificationData::setClassNameForCorrespondingClassLabel(string className,UINT classLabel){

    for(UINT i=0; i<classTracker.size(); i++){
        if( classTracker[i].classLabel == classLabel ){
            classTracker[i].className = className;
            return true;
        }
    }

	errorLog << "setClassNameForCorrespondingClassLabel(string className,UINT classLabel) - Failed to find class with label: " << classLabel << endl;
    return false;
}

bool LabelledClassificationData::addSample(UINT classLabel, vector<double> sample){
	if( sample.size() != numDimensions ){
        errorLog << "addSample(UINT classLabel, vector<double> sample) - the size of the new sample (" << sample.size() << ") does not match the number of dimensions of the dataset (" << numDimensions << ")" << endl;
        return false;
    }

    //The class label must be greater than zero (as zero is used for the null rejection class label
    if( classLabel == 0 ){
        errorLog << "addSample(UINT classLabel, vector<double> sample) - the class label can not be 0!" << endl;
        return false;
    }

    //The dataset has changed so flag that any previous cross validation setup will now not work
    crossValidationSetup = false;
    crossValidationIndexs.clear();

	LabelledClassificationSample newSample(classLabel,sample);
	data.push_back( newSample );
	totalNumSamples++;

	if( classTracker.size() == 0 ){
		ClassTracker tracker(classLabel,1);
		classTracker.push_back(tracker);
	}else{
		bool labelFound = false;
		for(unsigned int i=0; i<classTracker.size(); i++){
			if( classLabel == classTracker[i].classLabel ){
				classTracker[i].counter++;
				labelFound = true;
				break;
			}
		}
		if( !labelFound ){
			ClassTracker tracker(classLabel,1);
			classTracker.push_back(tracker);
		}
	}
	return true;
}

bool LabelledClassificationData::removeLastSample(){

    if( totalNumSamples > 0 ){

        //The dataset has changed so flag that any previous cross validation setup will now not work
        crossValidationSetup = false;
        crossValidationIndexs.clear();

        //Find the corresponding class ID for the last training example
        int classLabel = data[ totalNumSamples-1 ].getClassLabel();

        //Remove the training example from the buffer
        data.erase(data.end()-1);

        totalNumSamples = (UINT)data.size();

        //Remove the value from the counter
        for(unsigned int i=0; i<classTracker.size(); i++){
            if( classTracker[i].classLabel == classLabel ){
                classTracker[i].counter--;
                break;
            }
        }

        return true;

    }else return false;

}

UINT LabelledClassificationData::eraseAllSamplesWithClassLabel(UINT classLabel){
	int numExamplesRemoved = 0;
	int numExamplesToRemove = 0;

    //The dataset has changed so flag that any previous cross validation setup will now not work
    crossValidationSetup = false;
    crossValidationIndexs.clear();

	//Find out how many training examples we need to remove
	for(unsigned int i=0; i<classTracker.size(); i++){
		if( classTracker[i].classLabel == classLabel ){
			numExamplesToRemove = classTracker[i].counter;
			classTracker.erase(classTracker.begin()+i);
			break;
		}
	}

	//Remove the samples with the matching class ID
	if( numExamplesToRemove > 0 ){
		int i=0;
		while( numExamplesRemoved < numExamplesToRemove ){
			if( data[i].getClassLabel() == classLabel ){
				data.erase(data.begin()+i);
				numExamplesRemoved++;
			}else if( ++i == data.size() ) break;
		}
	}

	totalNumSamples = (UINT)data.size();

	return numExamplesRemoved;
}

bool LabelledClassificationData::relabelAllSamplesWithClassLabel(UINT oldClassLabel,UINT newClassLabel){
    bool oldClassLabelFound = false;
    bool newClassLabelAllReadyExists = false;
    UINT indexOfOldClassLabel = 0;
    UINT indexOfNewClassLabel = 0;

    //Find out how many training examples we need to relabel
    for(UINT i=0; i<classTracker.size(); i++){
        if( classTracker[i].classLabel == oldClassLabel ){
            indexOfOldClassLabel = i;
            oldClassLabelFound = true;
        }
        if( classTracker[i].classLabel == newClassLabel ){
            indexOfNewClassLabel = i;
            newClassLabelAllReadyExists = true;
        }
    }

    //If the old class label was not found then we can't do anything
    if( !oldClassLabelFound ){
        return false;
    }

    //Relabel the old class labels
    for(UINT i=0; i<totalNumSamples; i++){
        if( data[i].getClassLabel() == oldClassLabel ){
            data[i].set(newClassLabel, data[i].getSample());
        }
    }

    //Update the class label counters
    if( newClassLabelAllReadyExists ){
        //Add the old sample count to the new sample count
        classTracker[ indexOfNewClassLabel ].counter += classTracker[ indexOfOldClassLabel ].counter;

        //Erase the old class tracker
        classTracker.erase( classTracker.begin() + indexOfOldClassLabel );
    }else{
        //Create a new class tracker
        classTracker.push_back( ClassTracker(newClassLabel,classTracker[ indexOfOldClassLabel ].counter,classTracker[ indexOfOldClassLabel ].className) );
    }

    return true;
}

bool LabelledClassificationData::setExternalRanges(vector< MinMax > externalRanges, bool useExternalRanges){

    if( externalRanges.size() != numDimensions ) return false;

    this->externalRanges = externalRanges;
    this->useExternalRanges = useExternalRanges;

    return true;
}

bool LabelledClassificationData::enableExternalRangeScaling(bool useExternalRanges){
    if( externalRanges.size() == numDimensions ){
        this->useExternalRanges = useExternalRanges;
        return true;
    }
    return false;
}

bool LabelledClassificationData::scale(double minTarget,double maxTarget){
    vector< MinMax > ranges = getRanges();
    return scale(ranges,minTarget,maxTarget);
}

bool LabelledClassificationData::scale(vector<MinMax> ranges,double minTarget,double maxTarget){
    if( ranges.size() != numDimensions ) return false;

    //Scale the training data
    for(UINT i=0; i<totalNumSamples; i++){
        for(UINT j=0; j<numDimensions; j++){
            data[i][j] = scale(data[i][j],ranges[j].minValue,ranges[j].maxValue,minTarget,maxTarget);
        }
    }

    return true;
}

double inline LabelledClassificationData::scale(double x,double minSource,double maxSource,double minTarget,double maxTarget){
    return (((x-minSource)*(maxTarget-minTarget))/(maxSource-minSource))+minTarget;
}

bool LabelledClassificationData::saveDatasetToFile(string filename){

	std::fstream file;
	file.open(filename.c_str(), std::ios::out);

	if( !file.is_open() ){
		return false;
	}

	file << "GRT_LABELLED_CLASSIFICATION_DATA_FILE_V1.0\n";
    file << "DatasetName: " << datasetName << endl;
    file << "InfoText: " << infoText << endl;
	file << "NumDimensions: " << numDimensions << endl;
	file << "TotalNumTrainingExamples: " << totalNumSamples << endl;
	file << "NumberOfClasses: " << classTracker.size() << endl;
	file << "ClassIDsAndCounters: " << endl;

	for(UINT i=0; i<classTracker.size(); i++){
		file << classTracker[i].classLabel << "\t" << classTracker[i].counter << "\t" << classTracker[i].className << endl;
	}

    file << "UseExternalRanges: " << useExternalRanges << endl;

    if( useExternalRanges ){
        for(UINT i=0; i<externalRanges.size(); i++){
            file << externalRanges[i].minValue << "\t" << externalRanges[i].maxValue << endl;
        }
    }

	file << "LabelledTrainingData:\n";

	for(UINT i=0; i<totalNumSamples; i++){
		file << data[i].getClassLabel();
		for(UINT j=0; j<numDimensions; j++){
			file << "\t" << data[i][j];
		}
		file << endl;
	}

	file.close();
	return true;
}

bool LabelledClassificationData::loadDatasetFromFile(string filename){

	std::fstream file;
	file.open(filename.c_str(), std::ios::in);
	UINT numClasses = 0;
	clear();

	if( !file.is_open() ){
        errorLog << "loadDatasetFromFile(string filename) - could not open file!" << endl;
		return false;
	}

	string word;

	//Check to make sure this is a file with the Training File Format
	file >> word;
	if(word != "GRT_LABELLED_CLASSIFICATION_DATA_FILE_V1.0"){
        errorLog << "loadDatasetFromFile(string filename) - could not find file header!" << endl;
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
    while( word != "NumDimensions:" ){
        infoText += word + " ";
        file >> word;
    }

	//Get the number of dimensions in the training data
	if(word != "NumDimensions:"){
        errorLog << "loadDatasetFromFile(string filename) - failed to find DatasetName!" << endl;
		file.close();
		return false;
	}
	file >> numDimensions;

	//Get the total number of training examples in the training data
	file >> word;
	if(word != "TotalNumTrainingExamples:"){
        errorLog << "loadDatasetFromFile(string filename) - failed to find DatasetName!" << endl;
		file.close();
		return false;
	}
	file >> totalNumSamples;

	//Get the total number of classes in the training data
	file >> word;
	if(word != "NumberOfClasses:"){
        errorLog << "loadDatasetFromFile(string filename) - failed to find DatasetName!" << endl;
		file.close();
		return false;
	}
	file >> numClasses;

	//Resize the class counter buffer and load the counters
	classTracker.resize(numClasses);

	//Get the total number of classes in the training data
	file >> word;
	if(word != "ClassIDsAndCounters:"){
        errorLog << "loadDatasetFromFile(string filename) - failed to find DatasetName!" << endl;
		file.close();
		return false;
	}

	for(UINT i=0; i<classTracker.size(); i++){
		file >> classTracker[i].classLabel;
		file >> classTracker[i].counter;
        file >> classTracker[i].className;
	}

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
        externalRanges.resize(numDimensions);
        for(UINT i=0; i<externalRanges.size(); i++){
            file >> externalRanges[i].minValue;
            file >> externalRanges[i].maxValue;
        }
    }

	//Get the main training data
	file >> word;
	if(word != "LabelledTrainingData:"){
        errorLog << "loadDatasetFromFile(string filename) - failed to find DatasetName!" << endl;
		file.close();
		return false;
	}

	LabelledClassificationSample tempSample( numDimensions );
	data.resize( totalNumSamples, tempSample );

	for(UINT i=0; i<totalNumSamples; i++){
        UINT classLabel = 0;
        vector< double > sample(numDimensions,0);
		file >> classLabel;
		for(UINT j=0; j<numDimensions; j++){
			file >> sample[j];
		}
        data[i].set(classLabel, sample);
	}

	file.close();
	return true;
}


bool LabelledClassificationData::saveDatasetToCSVFile(string filename){

    std::fstream file;
	file.open(filename.c_str(), std::ios::out );

	if( !file.is_open() ){
		return false;
	}

    //Write the data to the CSV file
    for(UINT i=0; i<totalNumSamples; i++){
		file << data[i].getClassLabel();
		for(UINT j=0; j<numDimensions; j++){
			file << "," << data[i][j];
		}
		file << endl;
	}

	file.close();

    return true;
}

bool LabelledClassificationData::loadDatasetFromCSVFile(string filename){

    fstream file;
    string value;
    clear();
    numDimensions = 0;
    datasetName = "NOT_SET";
    infoText = "";

    //Clear any previous data
    clear();

    //Try and open the file
    file.open( filename.c_str(), std::ios::in );
    if( !file.is_open() ){
        return false;
    }

    //Read the first line to work out how many features are in the data
    getline( file, value );
    for(UINT i=0; i<value.size(); i++){
        if( value[i] == ',' ){
            numDimensions++;
        }
    }

    //If there are no commas in the first line then the data is not in the correct format
    if( numDimensions == 0 ){
        return false;
    }

    //Setup the labelled classification data
    setNumDimensions( numDimensions );

    //Reset the file to read from the start
    file.seekg( ios_base::beg );

    //Read the data
    bool keepParsing = true;
    UINT index = 0;
    UINT lastCommaIndex = 0;
    UINT variableCounter = 0;
    UINT classLabel = 0;
    UINT lineCounter = 0;
    vector< double > sample(numDimensions,0);
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
                    if( variableCounter == 0 ){
                        classLabel = (UINT)stringToInt( value.substr(0,index-2) );
                        lastCommaIndex = index;
                        variableCounter++;
                    }else{
                        sample[ variableCounter-1 ] = stringToDouble( value.substr(lastCommaIndex+1,index-2) );
                        lastCommaIndex = index;
                        variableCounter++;

                        if( variableCounter == numDimensions+1 ){
                            variableCounter = 0;
                            keepParsing = false;
                            if( !addSample(classLabel, sample) ){
                                errorLog << "loadDatasetFromCSVFile(string filename) - Failed to add sample from file!" << endl;
                                file.close();
                                return false;
                            }
                        }
                    }

                }


                if( ++index == value.length() ){
                    sample[ variableCounter-1 ] = stringToDouble( value.substr(lastCommaIndex+1,value.length()) );
                    lastCommaIndex = index;
                    variableCounter++;

                    if( variableCounter == numDimensions+1 ){
                        variableCounter = 0;
                        keepParsing = false;
                        if( !addSample(classLabel, sample) ){
                            errorLog << "loadDatasetFromCSVFile(string filename) - Failed to add sample from file!" << endl;
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
    
bool LabelledClassificationData::printStats(){
    
    cout << "DatasetName:\t" << datasetName << endl;
    cout << "DatasetInfo:\t" << infoText << endl;
    cout << "Number of Dimensions:\t" << numDimensions << endl;
    cout << "Number of Samples:\t" << totalNumSamples << endl;
    cout << "Number of Classes:\t" << getNumClasses() << endl;
    cout << "ClassStats:\n";
    
    for(UINT k=0; k<getNumClasses(); k++){
        cout << "ClassLabel:\t" << classTracker[k].classLabel;
        cout << "\tNumber of Samples:\t" << classTracker[k].counter;
        cout << "\tClassName:\t" << classTracker[k].className << endl;
    }
    
    vector< MinMax > ranges = getRanges();
    
    cout << "Dataset Ranges:\n";
    for(UINT j=0; j<ranges.size(); j++){
        cout << "[" << j+1 << "] Min:\t" << ranges[j].minValue << "\tMax: " << ranges[j].maxValue << endl;
    }
    
    return true;
}

LabelledClassificationData LabelledClassificationData::partition(UINT trainingSizePercentage,bool useStratifiedSampling){

    //Partitions the dataset into a training dataset (which is kept by this instance of the LabelledClassificationData) and
	//a testing/validation dataset (which is return as a new instance of the LabelledClassificationData).  The trainingSizePercentage
	//therefore sets the size of the data which remains in this instance and the remaining percentage of data is then added to
	//the testing/validation dataset

    //The dataset has changed so flag that any previous cross validation setup will now not work
    crossValidationSetup = false;
    crossValidationIndexs.clear();

    LabelledClassificationData trainingSet(numDimensions);
    LabelledClassificationData testSet(numDimensions);
    vector< UINT > indexs( totalNumSamples );

	//Create the random partion indexs
	Random random;
    UINT indexA = 0;
    UINT indexB = 0;
    UINT temp = 0;

    if( useStratifiedSampling ){
        //Break the data into seperate classes
        vector< vector< UINT > > classData( getNumClasses() );

        //Add the indexs to their respective classes
        for(UINT i=0; i<totalNumSamples; i++){
            classData[ getClassLabelIndexValue( data[i].getClassLabel() ) ].push_back( i );
        }

        //Randomize the order of the indexs in each of the class index buffers
        for(UINT k=0; k<getNumClasses(); k++){
            UINT numSamples = (UINT)classData[k].size();
            for(UINT x=0; x<numSamples*1000; x++){
                //Pick two random indexs
                indexA = random.getRandomNumberInt(0,numSamples);
                indexB = random.getRandomNumberInt(0,numSamples);

                //Swap the indexs
                temp = classData[k][ indexA ];
                classData[k][ indexA ] = classData[k][ indexB ];
                classData[k][ indexB ] = temp;
            }
        }

        //Loop over each class and add the data to the trainingSet and testSet
        for(UINT k=0; k<getNumClasses(); k++){
            UINT numTrainingExamples = (UINT) floor( double(classData[k].size()) / 100.0 * double(trainingSizePercentage) );

            //Add the data to the training and test sets
            for(UINT i=0; i<numTrainingExamples; i++){
                trainingSet.addSample( data[ classData[k][i] ].getClassLabel(), data[ classData[k][i] ].getSample() );
            }
            for(UINT i=numTrainingExamples; i<classData[k].size(); i++){
                testSet.addSample( data[ classData[k][i] ].getClassLabel(), data[ classData[k][i] ].getSample() );
            }
        }

        //Overwrite the training data in this instance with the training data of the trainingSet
        data = trainingSet.getClassificationData();
        totalNumSamples = trainingSet.getNumSamples();
    }else{

        const UINT numTrainingExamples = (UINT) floor( double(totalNumSamples) / 100.0 * double(trainingSizePercentage) );
        //Create the random partion indexs
        Random random;
        UINT indexA = 0;
        UINT indexB = 0;
        UINT temp = 0;
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

        //Add the data to the training and test sets
        for(UINT i=0; i<numTrainingExamples; i++){
            trainingSet.addSample( data[ indexs[i] ].getClassLabel(), data[ indexs[i] ].getSample() );
        }
        for(UINT i=numTrainingExamples; i<totalNumSamples; i++){
            testSet.addSample( data[ indexs[i] ].getClassLabel(), data[ indexs[i] ].getSample() );
        }

        //Overwrite the training data in this instance with the training data of the trainingSet
        data = trainingSet.getClassificationData();
        totalNumSamples = trainingSet.getNumSamples();
    }

	return testSet;
}

bool LabelledClassificationData::merge(LabelledClassificationData &labelledData){

    if( labelledData.getNumDimensions() != numDimensions ){
        errorLog << "merge(LabelledClassificationData &labelledData) - The number of dimensions in the labelledData (" << labelledData.getNumDimensions() << ") does not match the number of dimensions of this dataset (" << numDimensions << ")" << endl;
        return false;
    }

    //The dataset has changed so flag that any previous cross validation setup will now not work
    crossValidationSetup = false;
    crossValidationIndexs.clear();

    //Add the data from the labelledData to this instance
    for(UINT i=0; i<labelledData.getNumSamples(); i++){
        addSample(labelledData[i].getClassLabel(), labelledData[i].getSample());
    }

    //Set the class names from the dataset
    vector< ClassTracker > classTracker = labelledData.getClassTracker();
    for(UINT i=0; i<classTracker.size(); i++){
        setClassNameForCorrespondingClassLabel(classTracker[i].className, classTracker[i].classLabel);
    }

    return true;
}

bool LabelledClassificationData::spiltDataIntoKFolds(UINT K,bool useStratifiedSampling){

    crossValidationSetup = false;
    crossValidationIndexs.clear();

    //K can not be zero
    if( K > totalNumSamples ){
        errorLog << "spiltDataIntoKFolds(UINT K) - K can not be zero!" << endl;
        return false;
    }

    //K can not be larger than the number of examples
    if( K > totalNumSamples ){
        errorLog << "spiltDataIntoKFolds(UINT K,bool useStratifiedSampling) - K can not be larger than the total number of samples in the dataset!" << endl;
        return false;
    }

    //K can not be larger than the number of examples in a specific class if the stratified sampling option is true
    if( useStratifiedSampling ){
        for(UINT c=0; c<classTracker.size(); c++){
            if( K > classTracker[c].counter ){
                errorLog << "spiltDataIntoKFolds(UINT K,bool useStratifiedSampling) - K can not be larger than the number of samples in any given class!" << endl;
                return false;
            }
        }
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

    if( useStratifiedSampling ){
        //Break the data into seperate classes
        vector< vector< UINT > > classData( getNumClasses() );

        //Add the indexs to their respective classes
        for(UINT i=0; i<totalNumSamples; i++){
            classData[ getClassLabelIndexValue( data[i].getClassLabel() ) ].push_back( i );
        }

        //Randomize the order of the indexs in each of the class index buffers
        for(UINT c=0; c<getNumClasses(); c++){
            UINT numSamples = (UINT)classData[c].size();
            for(UINT x=0; x<numSamples*1000; x++){
                //Pick two random indexs
                indexA = random.getRandomNumberInt(0,numSamples);
                indexB = random.getRandomNumberInt(0,numSamples);

                //Swap the indexs
                temp = classData[c][ indexA ];
                classData[c][ indexA ] = classData[c][ indexB ];
                classData[c][ indexB ] = temp;
            }
        }

        //Loop over each of the k folds, at each fold add a sample from each class
        vector< UINT >::iterator iter;
        for(UINT k=0; k<K; k++){
            for(UINT c=0; c<getNumClasses(); c++){
                //Add the index to the current fold
                iter = classData[c].begin();
                if( iter != classData[c].end() ){
                    crossValidationIndexs[ k ].push_back( *iter );
                    classData[c].erase( iter );
                }
            }
        }

    }else{
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
    }

    crossValidationSetup = true;
    return true;

}

LabelledClassificationData LabelledClassificationData::getTrainingFoldData(UINT foldIndex){
    LabelledClassificationData trainingData;

    if( !crossValidationSetup ){
        errorLog << "getTrainingFoldData(UINT foldIndex) - Cross Validation has not been setup! You need to call the spiltDataIntoKFolds(UINT K,bool useStratifiedSampling) function first before calling this function!" << endl;
       return trainingData;
    }

    if( foldIndex >= kFoldValue ) return trainingData;

    trainingData.setNumDimensions( numDimensions );

    //Add the data to the training set, this will consist of all the data that is NOT in the foldIndex
    UINT index = 0;
    for(UINT k=0; k<kFoldValue; k++){
        if( k != foldIndex ){
            for(UINT i=0; i<crossValidationIndexs[k].size(); i++){

                index = crossValidationIndexs[k][i];
                trainingData.addSample( data[ index ].getClassLabel(), data[ index ].getSample() );
            }
        }
    }

    return trainingData;
}

LabelledClassificationData LabelledClassificationData::getTestFoldData(UINT foldIndex){
    LabelledClassificationData testData;

    if( !crossValidationSetup ) return testData;

    if( foldIndex >= kFoldValue ) return testData;

    //Add the data to the training
    testData.setNumDimensions( numDimensions );

    UINT index = 0;
	for(UINT i=0; i<crossValidationIndexs[ foldIndex ].size(); i++){

        index = crossValidationIndexs[ foldIndex ][i];
		testData.addSample( data[ index ].getClassLabel(), data[ index ].getSample() );
	}

    return testData;
}

LabelledClassificationData LabelledClassificationData::getClassData(UINT classLabel){
    LabelledClassificationData classData;
    classData.setNumDimensions( this->numDimensions );

    for(UINT i=0; i<totalNumSamples; i++){
        if( data[i].getClassLabel() == classLabel ){
            classData.addSample(classLabel, data[i].getSample());
        }
    }

    return classData;
}

LabelledRegressionData LabelledClassificationData::reformatAsLabelledRegressionData(){

    //Turns the classification into a regression data to enable regression algorithms like the MLP to be used as a classifier
    //This sets the number of targets in the regression data equal to the number of classes in the classification data
    //The output of each regression training sample will then be all 0's, except for the index matching the classLabel, which will be 1
    //For this to work, the labelled classification data cannot have any samples with a classLabel of 0!
    LabelledRegressionData regressionData;

    if( totalNumSamples == 0 ){
        return regressionData;
    }

    const UINT numInputDimensions = numDimensions;
    const UINT numTargetDimensions = getNumClasses();
    regressionData.setInputAndTargetDimensions(numInputDimensions, numTargetDimensions);

    for(UINT i=0; i<totalNumSamples; i++){
        vector< double > targetVector(numTargetDimensions,0);

        //Set the class index in the target vector to 1 and all other values in the target vector to 0
        UINT classLabel = data[i].getClassLabel();

        if( classLabel > 0 ){
            targetVector[ classLabel-1 ] = 1;
        }else{
            regressionData.clear();
            return regressionData;
        }

        regressionData.addSample(data[i].getSample(),targetVector);
    }

    return regressionData;
}

UnlabelledClassificationData LabelledClassificationData::reformatAsUnlabelledClassificationData(){

    UnlabelledClassificationData unlabelledData;

    if( totalNumSamples == 0 ){
        return unlabelledData;
    }

    unlabelledData.setNumDimensions( numDimensions );

    for(UINT i=0; i<totalNumSamples; i++){
        unlabelledData.addSample( data[i].getSample() );
    }

    return unlabelledData;
}

UINT LabelledClassificationData::getMinimumClassLabel(){
    UINT minClassLabel = 99999;

    for(UINT i=0; i<classTracker.size(); i++){
        if( classTracker[i].classLabel < minClassLabel ){
            minClassLabel = classTracker[i].classLabel;
        }
    }

    return minClassLabel;
}


UINT LabelledClassificationData::getMaximumClassLabel(){
    UINT maxClassLabel = 0;

    for(UINT i=0; i<classTracker.size(); i++){
        if( classTracker[i].classLabel > maxClassLabel ){
            maxClassLabel = classTracker[i].classLabel;
        }
    }

    return maxClassLabel;
}

UINT LabelledClassificationData::getClassLabelIndexValue(UINT classLabel){
    for(UINT k=0; k<classTracker.size(); k++){
        if( classTracker[k].classLabel == classLabel ){
            return k;
        }
    }
    warningLog << "getClassLabelIndexValue(UINT classLabel) - Failed to find class label: " << classLabel << " in class tracker!" << endl;
    return 0;
}

string LabelledClassificationData::getClassNameForCorrespondingClassLabel(UINT classLabel){

    for(UINT i=0; i<classTracker.size(); i++){
        if( classTracker[i].classLabel == classLabel ){
            return classTracker[i].className;
        }
    }

    return "CLASS_LABEL_NOT_FOUND";
}

vector<MinMax> LabelledClassificationData::getRanges(){

    vector< MinMax > ranges(numDimensions);

    //If the dataset should be scaled using the external ranges then return the external ranges
    if( useExternalRanges ) return externalRanges;

    //Otherwise return the min and max values for each column in the dataset
    if( totalNumSamples > 0 ){
        for(UINT j=0; j<numDimensions; j++){
            ranges[j].minValue = data[0][0];
            ranges[j].maxValue = data[0][0];
            for(UINT i=0; i<totalNumSamples; i++){
                if( data[i][j] < ranges[j].minValue ){ ranges[j].minValue = data[i][j]; }		//Search for the min value
                else if( data[i][j] > ranges[j].maxValue ){ ranges[j].maxValue = data[i][j]; }	//Search for the max value
            }
        }
    }
    return ranges;
}

int LabelledClassificationData::stringToInt(string value){
    std::stringstream s( value );
    int i;
    s >> i;
    return i;
}

double LabelledClassificationData::stringToDouble(string value){
    std::stringstream s( value );
    double d;
    s >> d;
    return d;
}

}; //End of namespace GRT
