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
 The LabelledContinuousTimeSeriesClassificationData is the main data structure for recording, labeling, managing, saving, and loading datasets that can be used to test the continuous classification abilities of the GRT supervised temporal learning algorithms.
 */

#pragma once

#include "TimeSeriesPositionTracker.h"
#include "LabelledClassificationData.h"
#include "LabelledTimeSeriesClassificationData.h"

namespace GRT{

class LabelledContinuousTimeSeriesClassificationData{
public:
	
    /**
     Constructor, sets the name of the dataset and the number of dimensions of the training data.
     The name of the dataset should not contain any spaces.
	 
     @param UINT numDimensions: the number of dimensions of the training data, should be an unsigned integer greater than 0
     @param string datasetName: the name of the dataset, should not contain any spaces
     @param string infoText: some info about the data in this dataset, this can contain spaces
    */
	LabelledContinuousTimeSeriesClassificationData(UINT numDimensions=0,string datasetName = "NOT_SET",string infoText = "");
    
    /**
     Copy Constructor, copies the LabelledContinuousTimeSeriesClassificationData from the rhs instance to this instance
     
	 @param const LabelledContinuousTimeSeriesClassificationData &rhs: another instance of the LabelledContinuousTimeSeriesClassificationData class from which the data will be copied to this instance
     */
	LabelledContinuousTimeSeriesClassificationData(const LabelledContinuousTimeSeriesClassificationData &rhs);
    
    /**
     Default Destructor
     */
	~LabelledContinuousTimeSeriesClassificationData();

    /**
     Sets the equals operator, copies the data from the rhs instance to this instance
     
	 @param const LabelledContinuousTimeSeriesClassificationData &rhs: another instance of the LabelledContinuousTimeSeriesClassificationData class from which the data will be copied to this instance
	 @return a reference to this instance of LabelledContinuousTimeSeriesClassificationData
     */
	LabelledContinuousTimeSeriesClassificationData& operator= (const LabelledContinuousTimeSeriesClassificationData &rhs){
		if( this != &rhs){
            this->datasetName = rhs.datasetName;
            this->infoText = rhs.infoText;
			this->numDimensions = rhs.numDimensions;
			this->totalNumSamples = rhs.totalNumSamples;
			this->lastClassID = rhs.lastClassID;
            this->playbackIndex = rhs.playbackIndex;
		    this->trackingClass = rhs.trackingClass;
            this->useExternalRanges = rhs.useExternalRanges;
            this->externalRanges = rhs.externalRanges;
			this->data = rhs.data;
			this->classTracker = rhs.classTracker;
			this->timeSeriesPositionTracker = rhs.timeSeriesPositionTracker;
            this->debugLog = rhs.debugLog;
            this->warningLog = rhs.warningLog;
            this->errorLog = rhs.errorLog;

		}
		return *this;
	}

    /**
     Array Subscript Operator, returns the LabelledClassificationSample at index i.  
	 It is up to the user to ensure that i is within the range of [0 totalNumSamples-1]
     
	 @param const UINT &i: the index of the training sample you want to access.  Must be within the range of [0 totalNumSamples-1]
     @return a reference to the i'th LabelledClassificationSample
     */
	inline LabelledClassificationSample& operator[] (const UINT i){
		return data[i];
	}

    /**
     Clears any previous training data and counters
     */
	void clear();
    
    /**
     Sets the number of dimensions in the training data. 
	 This should be an unsigned integer greater than zero.  
	 This will clear any previous training data and counters.
     This function needs to be called before any new samples can be added to the dataset, unless the numDimensions variable was set in the 
     constructor or some data was already loaded from a file
     
	 @param UINT numDimensions: the number of dimensions of the training data.  Must be an unsigned integer greater than zero
     @return true if the number of dimensions was correctly updated, false otherwise
     */
    bool setNumDimensions(UINT numDimensions);
    
    /**
     Sets the name of the dataset.
     There should not be any spaces in the name.
     Will return true if the name is set, or false otherwise.
     
	 @return returns true if the name is set, or false otherwise
     */
    bool setDatasetName(string datasetName);
    
    /**
     Sets the info string.
	 This can be any string with information about how the training data was recorded for example.
     
	 @param string infoText: the infoText
     @return true if the infoText was correctly updated, false otherwise
     */
    bool setInfoText(string infoText);
    
    /**
     Sets the name of the class with the given class label.  
     There should not be any spaces in the className.
     Will return true if the name is set, or false if the class label does not exist.
     
	 @return returns true if the name is set, or false if the class label does not exist
     */
    bool setClassNameForCorrespondingClassLabel(string className,UINT classLabel);
    
    /**
     Adds a new labelled sample to the dataset.  
     The dimensionality of the sample should match the number of dimensions in the LabelledClassificationData.
     The class label can be zero (this should represent a null class).
     
	 @param UINT classLabel: the class label of the corresponding sample
     @param UINT vector<double> sample: the new sample you want to add to the dataset.  The dimensionality of this sample should match the number of dimensions in the LabelledClassificationData
	 @return true if the sample was correctly added to the dataset, false otherwise
     */
	bool addSample(UINT classLabel, vector< double > trainingSample);
    
    /**
     Removes the last training sample added to the dataset.
     
	 @return true if the last sample was removed, false otherwise
     */
	bool removeLastSample();
    
    /**
     Deletes from the dataset all the samples with a specific class label.
     
	 @param UINT classLabel: the class label of the samples you wish to delete from the dataset
	 @return the number of samples deleted from the dataset
     */
	UINT eraseAllSamplesWithClassLabel(UINT classLabel);
    
    /**
     Relabels all the samples with the class label A with the new class label B.
     
	 @param UINT oldClassLabel: the class label of the samples you want to relabel
     @param UINT newClassLabel: the class label the samples will be relabelled with
	 @return returns true if the samples were correctly relablled, false otherwise
     */
	bool relabelAllSamplesWithClassLabel(UINT oldClassLabel,UINT newClassLabel);
    
    /**
     Sets the external ranges of the dataset, also sets if the dataset should be scaled using these values.  
     The dimensionality of the externalRanges vector should match the number of dimensions of this dataset.
     
	 @param vector< MinMax > externalRanges: an N dimensional vector containing the min and max values of the expected ranges of the dataset.
     @param bool useExternalRanges: sets if these ranges should be used to scale the dataset, default value is false.
	 @return returns true if the external ranges were set, false otherwise
     */
    bool setExternalRanges(vector< MinMax > externalRanges, bool useExternalRanges = false);
    
    /**
     Sets if the dataset should be scaled using an external range (if useExternalRanges == true) or the ranges of the dataset (if false).
     The external ranges need to be set FIRST before calling this function, otherwise it will return false.
     
     @param bool useExternalRanges: sets if these ranges should be used to scale the dataset
	 @return returns true if the useExternalRanges variable was set, false otherwise
     */
    bool enableExternalRangeScaling(bool useExternalRanges);
    
	/**
     Scales the dataset to the new target range.
     
	 @return true if the data was scaled correctly, false otherwise
     */
    bool scale(double minTarget,double maxTarget);
    
	/**
     Scales the dataset to the new target range, using the vector of ranges as the min and max source ranges.
     
	 @return true if the data was scaled correctly, false otherwise
     */
	bool scale(vector<MinMax> ranges,double minTarget,double maxTarget);
    
	/**
     The function used to scale the data
     
	 @param double x: the input value to be scaled
	 @param double minSource: the minimum source value (that x originates from)
	 @param double maxSource: the maximum source value (that x originates from)
	 @param double minTarget: the minimum target value (that x will be scaled to)
	 @param double maxTarget: the maximum target value (that x will be scaled to)
	 @return the scaled value
     */
	double inline scale(double x,double minSource,double maxSource,double minTarget,double maxTarget);
    
    /**
     Sets the playback index to a specific index.  The index should be within the range [0 totalNumSamples-1].
     
	 @param UINT playbackIndex: the value you want to set the playback index to
	 @return true if the playback index was set correctly, false otherwise
     */
    bool resetPlaybackIndex(UINT playbackIndex);
    
    /**
     Gets the next sample, this will also increment the playback index.
     If the playback index reaches the last data sample then it will be reset to 0.
     
	 @return the LabelledClassificationSample at the current playback index
     */
    LabelledClassificationSample getNextSample();
    
    /**
     Gets all the timeseries that have a specific class label.
     
     @param UINT classLabel: the class label of the timeseries you want to find
	 @return a LabelledTimeSeriesClassificationData dataset containing any timeseries that have the matching classlabel
     */
	LabelledTimeSeriesClassificationData getAllTrainingExamplesWithClassLabel(UINT classLabel);

    /**
     Saves the labelled timeseries classification data to a custom file format.
     
	 @param string filename: the name of the file the data will be saved to
	 @return true if the data was saved successfully, false otherwise
     */
	bool saveDatasetToFile(string filename);
    
    /**
     Loads the labelled timeseries classification data from a custom file format.
     
	 @param string filename: the name of the file the data will be loaded from
	 @return true if the data was loaded successfully, false otherwise
     */
	bool loadDatasetFromFile(string filename);
    
    /**
     Saves the labelled timeseries classification data to a CSV file.
     This will save the class label as the first column and the sample data as the following N columns, where N is the number of dimensions in the data.  Each row will represent a sample.
     
	 @param string filename: the name of the file the data will be saved to
	 @return true if the data was saved successfully, false otherwise
     */
    bool saveDatasetToCSVFile(string filename);
    
    /**
     Loads the labelled timeseries classification data from a CSV file.
     This assumes the data is formatted with each row representing a sample.
     The class label should be the first column followed by the sample data as the following N columns, where N is the number of dimensions in the data.
     
	 @param string filename: the name of the file the data will be loaded from
	 @return true if the data was loaded successfully, false otherwise
     */
	bool loadDataSetFromCSVFile(string filename);
    
    /**
     Gets the name of the dataset.
     
	 @return returns the name of the dataset
     */
    string getDatasetName(){ return datasetName; }
    
    /**
     Gets the infotext for the dataset
     
	 @return returns the infotext of the dataset
     */
    string getInfoText(){ return infoText; }
    
	/**
     Gets the number of dimensions of the labelled classification data.
     
	 @return an unsigned int representing the number of dimensions in the classification data
     */
	UINT inline getNumDimensions(){ return numDimensions; }
	
	/**
     Gets the number of samples in the classification data across all the classes.
     
	 @return an unsigned int representing the total number of samples in the classification data
     */
	UINT inline getNumSamples(){ return totalNumSamples; }
	
	/**
     Gets the number of classes.
     
	 @return an unsigned int representing the number of classes
     */
	UINT inline getNumClasses(){ return (UINT)classTracker.size(); }
    
    /**
     Gets the minimum class label in the dataset. If there are no values in the dataset then the value 99999 will be returned.
     
	 @return an unsigned int representing the minimum class label in the dataset
     */
    UINT getMinimumClassLabel();
    
    /**
     Gets the maximum class label in the dataset. If there are no values in the dataset then the value 0 will be returned.
     
	 @return an unsigned int representing the maximum class label in the dataset
     */
    UINT getMaximumClassLabel();
    
    /**
     Gets the index of the class label from the class tracker.
     
	 @return an unsigned int representing the index of the class label in the class tracker
     */
    UINT getClassLabelIndexValue(UINT classLabel);
    
    /**
     Gets the name of the class with a given class label.  If the class label does not exist then the string "CLASS_LABEL_NOT_FOUND" will be returned.
     
	 @return a string containing the name of the given class label or the string "CLASS_LABEL_NOT_FOUND" if the class label does not exist
     */
    string getClassNameForCorrespondingClassLabel(UINT classLabel);
    
	/**
     Gets the ranges of the classification data.
     
	 @return a vector of minimum and maximum values for each dimension of the data
     */
	vector<MinMax> getRanges();
    
	/**
     Gets the class tracker for each class in the dataset.
     
	 @return a vector of ClassTracker, one for each class in the dataset
     */
    vector< ClassTracker > getClassTracker(){ return classTracker; }
    
    /**
     Gets the timeseries position tracker, a vector of TimeSeriesPositionTracker which indicate the start and end position of each time series in the dataset.
     
	 @return a vector of TimeSeriesPositionTracker, one for each timeseries in the dataset
     */
    vector< TimeSeriesPositionTracker > getTimeSeriesPositionTracker(){ return timeSeriesPositionTracker; }
    
	/**
     Gets the classification data.
     
	 @return a vector of LabelledClassificationSamples
     */
	vector< LabelledClassificationSample > getClassificationData(){ return data; }


private:
    string datasetName;                                     ///< The name of the dataset
    string infoText;                                        ///< Some infoText about the dataset
	UINT numDimensions;										///< The number of dimensions in the dataset
	UINT totalNumSamples;
	UINT lastClassID;
    UINT playbackIndex;
	bool trackingClass;
    bool useExternalRanges;                                 ///< A flag to show if the dataset should be scaled using the externalRanges values
    vector< MinMax > externalRanges;                        ///< A vector containing a set of externalRanges set by the user
	vector< ClassTracker > classTracker;
	vector< LabelledClassificationSample > data;
	vector< TimeSeriesPositionTracker > timeSeriesPositionTracker;
    
    DebugLog debugLog;                                      ///< Default debugging log
    ErrorLog errorLog;                                      ///< Default error log
    WarningLog warningLog;                                  ///< Default warning log
};

} //End of namespace GRT
