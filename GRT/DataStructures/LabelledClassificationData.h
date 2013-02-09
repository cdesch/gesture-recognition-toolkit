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
 The LabelledClassificationData is the main data structure for recording, labeling, managing, saving, and loading training data for supervised learning problems.
 */

#pragma once

#include "LabelledClassificationSample.h"
#include "LabelledRegressionData.h"
#include "UnlabelledClassificationData.h"

namespace GRT{

class LabelledClassificationData{
public:
    
    /**
     Constructor, sets the name of the dataset and the number of dimensions of the training data.
     The name of the dataset should not contain any spaces.
	 
     @param UINT numDimensions: the number of dimensions of the training data, should be an unsigned integer greater than 0
     @param string datasetName: the name of the dataset, should not contain any spaces
     @param string infoText: some info about the data in this dataset, this can contain spaces
     */
    LabelledClassificationData(UINT numDimensions = 0,string datasetName = "NOT_SET",string infoText = "");

	/**
     Copy Constructor, copies the LabelledClassificationData from the rhs instance to this instance
     
	 @param const LabelledClassificationData &rhs: another instance of the LabelledClassificationData class from which the data will be copied to this instance
	*/
	LabelledClassificationData(const LabelledClassificationData &rhs);

	/**
     Default Destructor
    */
	~LabelledClassificationData();

	/**
     Sets the equals operator, copies the data from the rhs instance to this instance
     
	 @param const LabelledClassificationData &rhs: another instance of the LabelledClassificationData class from which the data will be copied to this instance
	 @return a reference to this instance of LabelledClassificationData
	*/
	LabelledClassificationData& operator= (const LabelledClassificationData &rhs){
		if( this != &rhs){
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
		return *this;
	}

	/**
     Array Subscript Operator, returns the LabelledClassificationSample at index i.  
	 It is up to the user to ensure that i is within the range of [0 totalNumSamples-1]

	 @param const UINT &i: the index of the training sample you want to access.  Must be within the range of [0 totalNumSamples-1]
     @return a reference to the i'th LabelledClassificationSample
    */
	inline LabelledClassificationSample& operator[] (const UINT &i){
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
     The class label should be greater than zero (as zero is used as the default null rejection class label).

	 @param UINT classLabel: the class label of the corresponding sample
     @param UINT vector<double> sample: the new sample you want to add to the dataset.  The dimensionality of this sample should match the number of dimensions in the LabelledClassificationData
	 @return true if the sample was correctly added to the dataset, false otherwise
    */
	bool addSample(UINT classLabel, vector<double> sample);
    
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
     Saves the labelled classification data to a custom file format.

	 @param string filename: the name of the file the data will be saved to
	 @return true if the data was saved successfully, false otherwise
    */
	bool saveDatasetToFile(string filename);
	
	/**
     Loads the labelled classification data from a custom file format.

	 @param string filename: the name of the file the data will be loaded from
	 @return true if the data was loaded successfully, false otherwise
    */
	bool loadDatasetFromFile(string filename);
    
    /**
     Saves the labelled classification data to a CSV file.
     This will save the class label as the first column and the sample data as the following N columns, where N is the number of dimensions in the data.  Each row will represent a sample.
     
	 @param string filename: the name of the file the data will be saved to
	 @return true if the data was saved successfully, false otherwise
     */
	bool saveDatasetToCSVFile(string filename);
	
	/**
     Loads the labelled classification data from a CSV file.
     This assumes the data is formatted with each row representing a sample.
     The class label should be the first column followed by the sample data as the following N columns, where N is the number of dimensions in the data.
     
	 @param string filename: the name of the file the data will be loaded from
	 @return true if the data was loaded successfully, false otherwise
     */
	bool loadDatasetFromCSVFile(string filename);
    
    /**
     Prints the dataset info (such as its name and infoText) and the stats (such as the number of examples, number of dimensions, number of classes, etc.)
     to the std out.
     
     @return returns true if the dataset info and stats were printed successfully, false otherwise
     */
    bool printStats();
    
    /**
     Adds the data in the labelledData set to the current instance of the LabelledClassificationData.
     The number of dimensions in both datasets must match.
     The names of the classes from the labelledData will be added to the current instance.
     
	 @param LabelledClassificationData &labelledData: the dataset to add to this dataset
	 @return returns true if the datasets were merged, false otherwise
    */
    bool merge(LabelledClassificationData &labelledData);
    
    /**
     Partitions the dataset into a training dataset (which is kept by this instance of the LabelledClassificationData) and
	 a testing/validation dataset (which is returned as a new instance of a LabelledClassificationData).
     
	 @param UINT partitionPercentage: sets the percentage of data which remains in this instance, the remaining percentage of data is then returned as the testing/validation dataset
     @param bool useStratifiedSampling: sets if the dataset should be broken into homogeneous groups first before randomly being spilt, default value is false
	 @return a new LabelledClassificationData instance, containing the remaining data not kept but this instance
     */
	LabelledClassificationData partition(UINT partitionPercentage,bool useStratifiedSampling = false);
    
    /**
     This function prepares the dataset for k-fold cross validation and should be called prior to calling the getTrainingFold(UINT foldIndex) or getTestingFold(UINT foldIndex) functions.  It will spilt the dataset into K-folds, as long as K < M, where M is the number of samples in the dataset.
     
	 @param UINT K: the number of folds the dataset will be split into, K should be less than the number of samples in the dataset
     @param bool useStratifiedSampling: sets if the dataset should be broken into homogeneous groups first before randomly being spilt, default value is false
	 @return returns true if the dataset was split correctly, false otherwise
    */
    bool spiltDataIntoKFolds(UINT K, bool useStratifiedSampling = false);
    
    /**
     Returns the training dataset for the k-th fold for cross validation.  The spiltDataIntoKFolds(UINT K) function should have been called once before using this function.
     The foldIndex should be in the range [0 K-1], where K is the number of folds the data was spilt into.
     
	 @param UINT foldIndex: the index of the fold you want the training data for, this should be in the range [0 K-1], where K is the number of folds the data was spilt into 
	 @return returns a training dataset
    */
    LabelledClassificationData getTrainingFoldData(UINT foldIndex);
    
    /**
     Returns the test dataset for the k-th fold for cross validation.  The spiltDataIntoKFolds(UINT K) function should have been called once before using this function.
     The foldIndex should be in the range [0 K-1], where K is the number of folds the data was spilt into.
     
	 @param UINT foldIndex: the index of the fold you want the test data for, this should be in the range [0 K-1], where K is the number of folds the data was spilt into 
	 @return returns a test dataset
    */
    LabelledClassificationData getTestFoldData(UINT foldIndex);
    
    /**
     Returns the all the data with the class label set by classLabel.
     The classLabel should be a valid classLabel, otherwise the dataset returned will be empty.
     
	 @param UINT classLabel: the class label of the class you want the data for
	 @return returns a dataset containing all the data with the matching classLabel
     */
    LabelledClassificationData getClassData(UINT classLabel);
    
	/**
     Reformats the LabelledClassificationData as LabelledRegressionData to enable regression algorithms like the MLP to be used as a classifier.
	 This sets the number of targets in the regression data equal to the number of classes in the classification data.  The output target ouput of each regression sample will therefore
	 be all zeros, except for the index matching the class label which will be 1.
	 For this to work, the labelled classification data cannot have any samples with a class label of 0!

	 @return a new LabelledRegressionData instance, containing the reformated classification data
    */
	LabelledRegressionData reformatAsLabelledRegressionData();
    
    /**
     Reformats the LabelledClassificationData as UnlabelledClassificationData so the data can be used to train unsupervised training algorithms such as K-Means Clustering and Gaussian Mixture Models.
     
	 @return a new UnlabelledClassificationData instance, containing the reformated labelled classification data
     */
    UnlabelledClassificationData reformatAsUnlabelledClassificationData();
    
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
     Gets the classification data.
     
	 @return a vector of LabelledClassificationSamples
    */
	vector< LabelledClassificationSample > getClassificationData(){ return data; }

private:
    int stringToInt(string value);
    double stringToDouble(string value);
    
    string datasetName;                                     ///< The name of the dataset
    string infoText;                                        ///< Some infoText about the dataset
	UINT numDimensions;										///< The number of dimensions in the dataset
	UINT totalNumSamples;                                   ///< The total number of samples in the dataset
    UINT kFoldValue;                                        ///< The number of folds the dataset has been spilt into for cross valiation
    bool crossValidationSetup;                              ///< A flag to show if the dataset is ready for cross validation
    bool useExternalRanges;                                 ///< A flag to show if the dataset should be scaled using the externalRanges values
    vector< MinMax > externalRanges;                        ///< A vector containing a set of externalRanges set by the user
	vector< ClassTracker > classTracker;					///< A vector of ClassTracker, which keeps track of the number of samples of each class
	vector< LabelledClassificationSample > data;            ///< The labelled classification data
    vector< vector< UINT > >    crossValidationIndexs;      ///< A vector to hold the indexs of the dataset for the cross validation
    
    DebugLog debugLog;                                      ///< Default debugging log
    ErrorLog errorLog;                                      ///< Default error log
    WarningLog warningLog;                                  ///< Default warning log
        
};

} //End of namespace GRT