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
 This is the main base class that all GRT Classification algorithms should inherit from.  A large number of the
 functions in this class are virtual and simply return false as these functions must be overwridden by the inheriting
 class.
 */


#pragma once

#include "MLBase.h"
#include "../DataStructures/LabelledClassificationData.h"
#include "../DataStructures/LabelledTimeSeriesClassificationData.h"

namespace GRT{
    
#define DEFAULT_NULL_LIKELIHOOD_VALUE 0
#define DEFAULT_NULL_DISTANCE_VALUE 0

class Classifier : public MLBase
{
public:
    /**
     Default Classifier Constructor
     */
	Classifier(void);
    
    /**
     Default Classifier Destructor
     */
	virtual ~Classifier(void);
    
    /**
     This is the base clone function for the Classifier modules. This function should be overwritten by the derived class.
     
     @param const Classifier *classifier: a pointer to the Classifier base class, this should be pointing to another instance of a matching derived class
     @return returns true if the clone was successfull, false otherwise (the Classifier base class will always return flase)
     */
    virtual bool clone(const Classifier *classifier){ return false; }
    
    /**
     This copies the Classifier variables from classifierB to classifierA.
     
     @param Classifier *classifierA: a pointer to a classifier into which the values from classifierB will be copied
     @param const Classifier *classifierB: a pointer to a classifier from which the values will be copied to classifierA
     @return returns true if the copy was successfull, false otherwise
     */
    bool copyBaseVariables(Classifier *classifierA,const Classifier *classifierB);

    /**
     This is the main training interface for LabelledClassificationData. This should be overwritten by the derived class.
     
     @param LabelledClassificationData &trainingData: a reference to the training data that will be used to train the classifier
     @return returns true if the classifier was successfully trained, false otherwise
     */
    virtual bool train(LabelledClassificationData &trainingData){ return false; }
    
    /**
     This is the main training interface for LabelledTimeSeriesClassificationData. This should be overwritten by the derived class.
     
     @param LabelledTimeSeriesClassificationData &trainingData: a reference to the training data that will be used to train the classifier
     @return returns true if the classifier was successfully trained, false otherwise
     */
    virtual bool train(LabelledTimeSeriesClassificationData &trainingData){ return false; }
    
    /**
     This is the main training interface for UnlabelledClassificationData. This should be overwritten by the derived class.
     
     @param UnlabelledClassificationData &trainingData: a reference to the training data that will be used to train the classifier
     @return returns true if the classifier was successfully trained, false otherwise
     */
    virtual bool train(UnlabelledClassificationData &trainingData){ return false; }

    /**
     Returns the classifeir type as a string.
     
     @return returns the classifier type as a string
     */
    string getClassifierType() const;
    
    /**
     Returns true if nullRejection is enabled.
     
     @return returns true if nullRejection is enabled, false otherwise
     */
    bool getNullRejectionEnabled() const;
    
    /**
     Returns the current nullRejectionCoeff value.
     The nullRejectionCoeff parameter is a multipler controlling the null rejection threshold for each class.
     
     @return returns the current nullRejectionCoeff value
     */
    double getNullRejectionCoeff() const;
    
    /**
     Returns the current maximumLikelihood value.
     The maximumLikelihood value is computed during the prediction phase and is the likelihood of the most likely model.
     This value will return 0 if a prediction has not been made.
     
     @return returns the current maximumLikelihood value
     */
    double getMaximumLikelihood() const;
    
    /**
     Returns the current bestDistance value.
     The bestDistance value is computed during the prediction phase and is either the minimum or maximum distance, depending on the algorithm.
     This value will return 0 if a prediction has not been made.
     
     @return returns the current bestDistance value
     */
    double getBestDistance() const;
    
    /**
     Gets the number of classes in trained model.
     
     @return returns the number of classes in the trained model, a value of 0 will be returned if the model has not been trained
     */
    virtual UINT getNumClasses() const;
    
    /**
     Gets the predicted class label from the last prediction.
     
     @return returns the label of the last predicted class, a value of 0 will be returned if the model has not been trained
     */
    UINT getPredictedClassLabel() const;
    
    /**
     Gets a vector of the class likelihoods from the last prediction, this will be an N-dimensional vector, where N is the number of classes in the model.  
     The exact form of these likelihoods depends on the classification algorithm.
     
     @return returns a vector of the class likelihoods from the last prediction, an empty vector will be returned if the model has not been trained
     */
    vector< double > getClassLikelihoods() const;
    
    /**
     Gets a vector of the class distances from the last prediction, this will be an N-dimensional vector, where N is the number of classes in the model.  
     The exact form of these distances depends on the classification algorithm.
     
     @return returns a vector of the class distances from the last prediction, an empty vector will be returned if the model has not been trained
     */
    vector< double > getClassDistances() const;
    
    /**
     Gets a vector containing the null rejection thresholds for each class, this will be an N-dimensional vector, where N is the number of classes in the model.  
     
     @return returns a vector containing the null rejection thresholds for each class, an empty vector will be returned if the model has not been trained
     */
    vector< double > getNullRejectionThresholds() const;
    
    /**
     Gets a vector containing the label each class represents, this will be an N-dimensional vector, where N is the number of classes in the model. 
     This is useful if the model was trained with non-monotonically class labels (i.e. class labels such as [1, 3, 6, 9, 12] instead of [1, 2, 3, 4, 5]).
     
     @return returns a vector containing the class labels for each class, an empty vector will be returned if the model has not been trained
     */
    vector< UINT > getClassLabels() const;

    /**
     Sets if the classifier should use nullRejection.
     
     If set to true then the classifier will reject a predicted class label if the likelihood of the prediction is below (or above depending on the 
     algorithm) the models rejectionThreshold. If a prediction is rejected then the default null class label of 0 will be returned.
     If set to false then the classifier will simply return the most likely predicted class.
     
     @return returns true if nullRejection was updated successfully, false otherwise
     */
    bool enableNullRejection(bool useNullRejection);
    
    /**
     Sets the nullRejectionCoeff, this is a multipler controlling the null rejection threshold for each class.
     
     @return returns true if nullRejectionCoeff was updated successfully, false otherwise
     */
    virtual bool setNullRejectionCoeff(double nullRejectionCoeff);
    
    /**
     Recomputes the null rejection thresholds for each model.
     
     @return returns true if the nullRejectionThresholds were updated successfully, false otherwise
     */
    virtual bool recomputeNullRejectionThresholds(){ return false; }
    
    /**
     Indicates if the classifier can be used to classify timeseries data.
     If true then the classifier can accept training data in the LabelledTimeSeriesClassificationData format.
     
     return returns true if the classifier can be used to classify timeseries data, false otherwise
     */
    bool getTimeseriesCompatible(){ return classifierMode==TIMESERIES_CLASSIFIER_MODE; }
    
    /**
     Defines a map between a string (which will contain the name of the classifier, such as ANBC) and a function returns a new instance of that classifier
     */
    typedef std::map< string, Classifier*(*)() > StringClassifierMap;
    
    /**
     Creates a new classifier instance based on the input string (which should contain the name of a valid classifier such as ANBC).
     
     @param string const &classifierType: the name of the classifier
     @return Classifier*: a pointer to the new instance of the classifier
     */
    static Classifier* createInstanceFromString(string const &classifierType);
    
    /**
     Creates a new classifier instance based on the current classifierType string value.
     
     @return Classifier*: a pointer to the new instance of the classifier
    */
    Classifier* createNewInstance() const;
    
protected:
    string classifierType;
    bool useNullRejection;
    UINT numClasses;
    UINT predictedClassLabel;
    UINT classifierMode;
    double nullRejectionCoeff;
    double maxLikelihood;
    double bestDistance;
    vector< double > classLikelihoods;
    vector< double > classDistances;
    vector< double > nullRejectionThresholds;
    vector< UINT > classLabels;
    
    static StringClassifierMap *getMap() {
        if( !stringClassifierMap ){ stringClassifierMap = new StringClassifierMap; } 
        return stringClassifierMap; 
    }
        
    enum ClassifierModes{STANDARD_CLASSIFIER_MODE=0,TIMESERIES_CLASSIFIER_MODE};
    
private:
    static StringClassifierMap *stringClassifierMap;
    static UINT numClassifierInstances;
    
};
    
//These two functions/classes are used to register any new Classification Module with the Classifier base class
template< typename T >  Classifier *newClassificationModuleInstance() { return new T; }

template< typename T > 
class RegisterClassifierModule : Classifier { 
public:
    RegisterClassifierModule(string const &newClassificationModuleName) { 
        getMap()->insert( std::pair<string, Classifier*(*)()>(newClassificationModuleName, &newClassificationModuleInstance< T > ) );
    }
};

} //End of namespace GRT

