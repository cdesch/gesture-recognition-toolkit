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

#include "LabelledContinuousTimeSeriesClassificationData.h"

namespace GRT{

//Constructors and Destructors
LabelledContinuousTimeSeriesClassificationData::LabelledContinuousTimeSeriesClassificationData(UINT numDimensions,string datasetName,string infoText){
    
    this->numDimensions= numDimensions;
    this->datasetName = datasetName;
    this->infoText = infoText;
    
    playbackIndex  = 0;
    trackingClass = false;
    useExternalRanges = false;
    debugLog.setProceedingText("[DEBUG LTSCD]");
    errorLog.setProceedingText("[ERROR LTSCD]");
    warningLog.setProceedingText("[WARNING LTSCD]");
    
    if( numDimensions > 0 ){
        setNumDimensions(numDimensions);
    }
}

LabelledContinuousTimeSeriesClassificationData::LabelledContinuousTimeSeriesClassificationData(const LabelledContinuousTimeSeriesClassificationData &rhs){
    this->datasetName = rhs.datasetName;
    this->infoText = rhs.infoText;
	this->numDimensions = rhs.numDimensions;
	this->totalNumSamples = rhs.totalNumSamples;
    this->playbackIndex = rhs.playbackIndex;
	this->lastClassID = rhs.lastClassID;
	this->trackingClass = rhs.trackingClass;
    this->useExternalRanges = rhs.useExternalRanges;
    this->externalRanges = rhs.externalRanges;
	this->data = rhs.data;
    this->debugLog = rhs.debugLog;
    this->warningLog = rhs.warningLog;
    this->errorLog = rhs.errorLog;
}

LabelledContinuousTimeSeriesClassificationData::~LabelledContinuousTimeSeriesClassificationData(){}

void LabelledContinuousTimeSeriesClassificationData::clear(){
	totalNumSamples = 0;
    playbackIndex = 0;
	trackingClass = false;
	data.clear();
	classTracker.clear();
	timeSeriesPositionTracker.clear();
}
    
bool LabelledContinuousTimeSeriesClassificationData::setNumDimensions(UINT numDimensions){
    if( numDimensions > 0 ){
        //Clear any previous data
        clear();
        
        //Set the dimensionality of the time series data
        this->numDimensions = numDimensions;
        
        return true;
    }
    
    errorLog << "setNumDimensions(UINT numDimensions) - The number of dimensions of the dataset must be greater than zero!" << endl;
    return false;
}
    
    
bool LabelledContinuousTimeSeriesClassificationData::setDatasetName(string datasetName){
    
    //Make sure there are no spaces in the string
    if( datasetName.find(" ") == string::npos ){
        this->datasetName = datasetName;
        return true;
    }
    
    errorLog << "setDatasetName(string datasetName) - The dataset name cannot contain any spaces!" << endl;
    return false;
}
    
bool LabelledContinuousTimeSeriesClassificationData::setInfoText(string infoText){
    this->infoText = infoText;
    return true;
}
    
bool LabelledContinuousTimeSeriesClassificationData::setClassNameForCorrespondingClassLabel(string className,UINT classLabel){
    
    for(UINT i=0; i<classTracker.size(); i++){
        if( classTracker[i].classLabel == classLabel ){
            classTracker[i].className = className;
            return true;
        }
    }
    
    errorLog << "setClassNameForCorrespondingClassLabel(string className,UINT classLabel) - Failed to find class with label: " << classLabel << endl;
    return false;
}

bool LabelledContinuousTimeSeriesClassificationData::addSample(UINT classLabel, vector< double > sample){

	if( numDimensions != sample.size() ){
		errorLog << "addSample(UINT classLabel, vector<double> sample) - the size of the new sample (" << sample.size() << ") does not match the number of dimensions of the dataset (" << numDimensions << ")" << endl;
        return false;
	}

	bool searchForNewClass = true;
	if( trackingClass ){
		if( classLabel != lastClassID ){
			//The class ID has changed so update the time series tracker
			timeSeriesPositionTracker[ timeSeriesPositionTracker.size()-1 ].setEndIndex( totalNumSamples-1 );
		}else searchForNewClass = false;
	}
	
	if( searchForNewClass ){
		bool newClass = true;
		//Search to see if this class has been found before
		for(UINT k=0; k<classTracker.size(); k++){
			if( classTracker[k].classLabel == classLabel ){
				newClass = false;
				classTracker[k].counter++;
			}
		}
		if( newClass ){
			ClassTracker newCounter(classLabel,1);
			classTracker.push_back( newCounter );
		}

		//Set the timeSeriesPositionTracker start position
		trackingClass = true;
		lastClassID = classLabel;
		TimeSeriesPositionTracker newTracker(totalNumSamples,0,classLabel);
		timeSeriesPositionTracker.push_back( newTracker );
	}

	LabelledClassificationSample labelledSample(classLabel,sample);
	data.push_back( labelledSample );
	totalNumSamples++;
	return true;
}
    
bool LabelledContinuousTimeSeriesClassificationData::removeLastSample(){
    
    if( totalNumSamples > 0 ){
        
        //Find the corresponding class ID for the last training example
        int classLabel = data[ totalNumSamples-1 ].getClassLabel();
        
        //Remove the training example from the buffer
        data.erase( data.end()-1 );
        
        totalNumSamples = (UINT)data.size();
        
        //Remove the value from the counter
        for(unsigned int i=0; i<classTracker.size(); i++){
            if( classTracker[i].classLabel == classLabel ){
                classTracker[i].counter--;
                break;
            }
        }	
        
        //If we are not tracking a class then decrement the end index of the timeseries position tracker
        if( !trackingClass ){
            UINT endIndex = timeSeriesPositionTracker[ timeSeriesPositionTracker.size()-1 ].getEndIndex();
            timeSeriesPositionTracker[ timeSeriesPositionTracker.size()-1 ].setEndIndex( endIndex-1 );
        }
        
        return true;
        
    }else return false;
    
}
    
UINT LabelledContinuousTimeSeriesClassificationData::eraseAllSamplesWithClassLabel(UINT classLabel){
    int numExamplesRemoved = 0;
    int numExamplesToRemove = 0;
    
    //Find out how many training examples we need to remove    
    for(unsigned int i=0; i<classTracker.size(); i++){
        if( classTracker[i].classLabel == classLabel ){
            numExamplesToRemove = classTracker[i].counter;
            classTracker.erase(classTracker.begin()+i);
            break; //There should only be one class with this classLabel so break
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
    
    //Update the time series position tracker
    vector< TimeSeriesPositionTracker >::iterator iter = timeSeriesPositionTracker.begin();
    
    while( iter != timeSeriesPositionTracker.end() ){
        if( iter->getClassLabel() == classLabel ){
            UINT length = iter->getLength();
            //Update the start and end positions of all the following position trackers
            vector< TimeSeriesPositionTracker >::iterator updateIter = iter + 1;
            
            while( updateIter != timeSeriesPositionTracker.end() ){
                updateIter->setStartIndex( updateIter->getStartIndex() - length );
                updateIter->setEndIndex( updateIter->getEndIndex() - length );
                updateIter++;
            }
            
            //Erase the current position tracker
            iter = timeSeriesPositionTracker.erase( iter );
        }else iter++;
    }
    
    totalNumSamples = (UINT)data.size();
    
    return numExamplesRemoved;
}
    
bool LabelledContinuousTimeSeriesClassificationData::relabelAllSamplesWithClassLabel(UINT oldClassLabel,UINT newClassLabel){
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
    
    //Update the timeseries position tracker
    for(UINT i=0; i<timeSeriesPositionTracker.size(); i++){
        if( timeSeriesPositionTracker[i].getClassLabel() == oldClassLabel ){
            timeSeriesPositionTracker[i].setClassLabel( newClassLabel );
        }
    }
    
    return true;
}
    
bool LabelledContinuousTimeSeriesClassificationData::setExternalRanges(vector< MinMax > externalRanges, bool useExternalRanges){
    
    if( externalRanges.size() != numDimensions ) return false;
    
    this->externalRanges = externalRanges;
    this->useExternalRanges = useExternalRanges;
    
    return true;
}

bool LabelledContinuousTimeSeriesClassificationData::enableExternalRangeScaling(bool useExternalRanges){
    if( externalRanges.size() == numDimensions ){
        this->useExternalRanges = useExternalRanges;
        return true;
    }
    return false;
}

bool LabelledContinuousTimeSeriesClassificationData::scale(double minTarget,double maxTarget){
    vector< MinMax > ranges = getRanges();
    return scale(ranges,minTarget,maxTarget);
}

bool LabelledContinuousTimeSeriesClassificationData::scale(vector<MinMax> ranges,double minTarget,double maxTarget){
    if( ranges.size() != numDimensions ) return false;
    
    //Scale the training data
    for(UINT i=0; i<totalNumSamples; i++){
        for(UINT j=0; j<numDimensions; j++){
            data[i][j] = scale(data[i][j],ranges[j].minValue,ranges[j].maxValue,minTarget,maxTarget);
        }
    }
    
    return true;
}

double inline LabelledContinuousTimeSeriesClassificationData::scale(double x,double minSource,double maxSource,double minTarget,double maxTarget){
    return (((x-minSource)*(maxTarget-minTarget))/(maxSource-minSource))+minTarget;
}

    
bool LabelledContinuousTimeSeriesClassificationData::resetPlaybackIndex(UINT playbackIndex){
    if( playbackIndex < totalNumSamples ){
        this->playbackIndex = playbackIndex;
        return true;
    }
    return false;
}

LabelledClassificationSample LabelledContinuousTimeSeriesClassificationData::getNextSample(){    
    if( totalNumSamples == 0 ) return LabelledClassificationSample();
    
    UINT index = playbackIndex++ % totalNumSamples;
    return data[ index ];
}
    
LabelledTimeSeriesClassificationData LabelledContinuousTimeSeriesClassificationData::getAllTrainingExamplesWithClassLabel(UINT classLabel){
	LabelledTimeSeriesClassificationData classData(numDimensions);
	for(UINT x=0; x<timeSeriesPositionTracker.size(); x++){
		if( timeSeriesPositionTracker[x].getClassLabel() == classLabel && timeSeriesPositionTracker[x].getEndIndex() > 0){
			Matrix<double> timeSeries;
			for(UINT i=timeSeriesPositionTracker[x].getStartIndex(); i<timeSeriesPositionTracker[x].getEndIndex(); i++){
				timeSeries.push_back( data[ i ].getSample() );
			}
			classData.addSample(classLabel,timeSeries);
		}
	}
	return classData;
}

UINT LabelledContinuousTimeSeriesClassificationData::getMinimumClassLabel(){
    UINT minClassLabel = 99999;
    
    for(UINT i=0; i<classTracker.size(); i++){
        if( classTracker[i].classLabel < minClassLabel ){
            minClassLabel = classTracker[i].classLabel;
        }
    }
    
    return minClassLabel;
}


UINT LabelledContinuousTimeSeriesClassificationData::getMaximumClassLabel(){
    UINT maxClassLabel = 0;
    
    for(UINT i=0; i<classTracker.size(); i++){
        if( classTracker[i].classLabel > maxClassLabel ){
            maxClassLabel = classTracker[i].classLabel;
        }
    }
    
    return maxClassLabel;
}

UINT LabelledContinuousTimeSeriesClassificationData::getClassLabelIndexValue(UINT classLabel){
    for(UINT k=0; k<classTracker.size(); k++){
        if( classTracker[k].classLabel == classLabel ){
            return k;
        }
    }
    warningLog << "getClassLabelIndexValue(UINT classLabel) - Failed to find class label: " << classLabel << " in class tracker!" << endl;
    return 0;
}

string LabelledContinuousTimeSeriesClassificationData::getClassNameForCorrespondingClassLabel(UINT classLabel){
    
    for(UINT i=0; i<classTracker.size(); i++){
        if( classTracker[i].classLabel == classLabel ){
            return classTracker[i].className;
        }
    }
    
    return "CLASS_LABEL_NOT_FOUND";
}

vector<MinMax> LabelledContinuousTimeSeriesClassificationData::getRanges(){
    
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

bool LabelledContinuousTimeSeriesClassificationData::saveDatasetToFile(string filename){

	std::fstream file; 
	file.open(filename.c_str(), std::ios::out);

	if( !file.is_open() ){
        errorLog << "saveDatasetToFile(string filename) - Failed to open file!" << endl;
		return false;
	}

	if( trackingClass ){
		//The class tracker was not stopped so assume the last sample is the end
		trackingClass = false;
		timeSeriesPositionTracker[ timeSeriesPositionTracker.size()-1 ].setEndIndex( totalNumSamples-1 );
	}

	file << "GRT_LABELLED_CONTINUOUS_TIME_SERIES_CLASSIFICATION_FILE_V1.0\n";
    file << "DatasetName: " << datasetName << endl;
    file << "InfoText: " << infoText << endl;
	file << "NumDimensions: "<<numDimensions<<endl;
	file << "TotalNumSamples: "<<totalNumSamples<<endl;
	file << "NumberOfClasses: "<<classTracker.size()<<endl;
	file << "ClassIDsAndCounters: "<<endl;
	for(UINT i=0; i<classTracker.size(); i++){
		file << classTracker[i].classLabel << "\t" << classTracker[i].counter << endl;
	}

	file << "NumberOfPositionTrackers: "<<timeSeriesPositionTracker.size()<<endl;
	file << "TimeSeriesPositionTrackers: "<<endl;
	for(UINT i=0; i<timeSeriesPositionTracker.size(); i++){
		file << timeSeriesPositionTracker[i].getClassLabel() << "\t" << timeSeriesPositionTracker[i].getStartIndex() << "\t" << timeSeriesPositionTracker[i].getEndIndex() <<endl;
	}
    
    file << "UseExternalRanges: " << useExternalRanges << endl;
    
    if( useExternalRanges ){
        for(UINT i=0; i<externalRanges.size(); i++){
            file << externalRanges[i].minValue << "\t" << externalRanges[i].maxValue << endl;
        }
    }

	file << "LabelledContinuousTimeSeriesClassificationData:\n";
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

bool LabelledContinuousTimeSeriesClassificationData::loadDatasetFromFile(string filename){

	std::fstream file; 
	file.open(filename.c_str(), std::ios::in);
	UINT numClasses = 0;
	UINT numTrackingPoints = 0;
	clear();

	if( !file.is_open() ){
		errorLog<< "loadDatasetFromFile(string fileName) - Failed to open file!" << endl;
		return false;
	}

	string word;

	//Check to make sure this is a file with the Training File Format
	file >> word;
	if(word != "GRT_LABELLED_CONTINUOUS_TIME_SERIES_CLASSIFICATION_FILE_V1.0"){
		file.close();
        errorLog<< "loadDatasetFromFile(string fileName) - Failed to find file header!" << endl;
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
        errorLog<< "loadDatasetFromFile(string fileName) - Failed to find NumDimensions!" << endl;
		file.close();
		return false;
	}
	file >> numDimensions;

	//Get the total number of training examples in the training data
	file >> word;
	if(word != "TotalNumSamples:"){
        errorLog<< "loadDatasetFromFile(string fileName) - Failed to find TotalNumSamples!" << endl;
		file.close();
		return false;
	}
	file >> totalNumSamples;

	//Get the total number of classes in the training data
	file >> word;
	if(word != "NumberOfClasses:"){
        errorLog<< "loadDatasetFromFile(string fileName) - Failed to find NumberOfClasses!" << endl;
		file.close();
		return false;
	}
	file >> numClasses;

	//Resize the class counter buffer and load the counters
	classTracker.resize(numClasses);

	//Get the total number of classes in the training data
	file >> word;
	if(word != "ClassIDsAndCounters:"){
        errorLog<< "loadDatasetFromFile(string fileName) - Failed to find ClassIDsAndCounters!" << endl;
		file.close();
		return false;
	}

	for(UINT i=0; i<classTracker.size(); i++){
		file >> classTracker[i].classLabel;
		file >> classTracker[i].counter;
	}

	//Get the NumberOfPositionTrackers
	file >> word;
	if(word != "NumberOfPositionTrackers:"){
        errorLog<< "loadDatasetFromFile(string fileName) - Failed to find NumberOfPositionTrackers!" << endl;
		file.close();
		return false;
	}
	file >> numTrackingPoints;
	timeSeriesPositionTracker.resize( numTrackingPoints );

	//Get the TimeSeriesPositionTrackers
	file >> word;
	if(word != "TimeSeriesPositionTrackers:"){
        errorLog<< "loadDatasetFromFile(string fileName) - Failed to find TimeSeriesPositionTrackers!" << endl;
		file.close();
		return false;
	}

	for(UINT i=0; i<timeSeriesPositionTracker.size(); i++){
		UINT classLabel;
		UINT startIndex;
		UINT endIndex;
		file >> classLabel;
		file >> startIndex;
		file >> endIndex;
		timeSeriesPositionTracker[i].setTracker(startIndex,endIndex,classLabel);
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
	
	//Get the main time series data
	file >> word;
	if(word != "LabelledContinuousTimeSeriesClassificationData:"){
        errorLog<< "loadDatasetFromFile(string fileName) - Failed to find LabelledContinuousTimeSeriesClassificationData!" << endl;
		file.close();
		return false;
	}

	//Reset the memory
	data.resize( totalNumSamples, LabelledClassificationSample() );

	//Load each sample
	for(UINT i=0; i<totalNumSamples; i++){
		UINT classLabel = 0;
		vector<double> sample(numDimensions);

		file >> classLabel;
		for(UINT j=0; j<numDimensions; j++){
			file >> sample[j];
		}

		data[i].set(classLabel,sample);
	}

	file.close();
	return true;
}
    
bool LabelledContinuousTimeSeriesClassificationData::saveDatasetToCSVFile(string filename){
    std::fstream file; 
	file.open(filename.c_str(), std::ios::out );
    
	if( !file.is_open() ){
		return false;
	}
    
    //Write the data to the CSV file

    for(UINT i=0; i<data.size(); i++){
        file << data[i].getClassLabel();
        for(UINT j=0; j<numDimensions; j++){
            file << "," << data[i][j];
        }
        file << endl;
    }
    
	file.close();
    
    return true;
}

bool LabelledContinuousTimeSeriesClassificationData::loadDataSetFromCSVFile(string filename){
    //TODO
    datasetName = "NOT_SET";
    infoText = "";
    return false;
}

} //End of namespace GRT

