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
 */

#pragma once

#include "../GestureRecognitionPipeline/FeatureExtraction.h"
#include "../Util/Util.h"

namespace GRT{
    
struct AngleMagnitude{
    AngleMagnitude(){
        angle = 0;
        magnitude = 0;
    }
    double angle;
    double magnitude;
};
typedef struct AngleMagnitude AngleMagnitude;
    
class MovementTrajectoryFeatures : public FeatureExtraction{
public:
    /**
     */
    MovementTrajectoryFeatures(UINT trajectoryLength=100,UINT numCentroids=10,UINT featureMode=CENTROID_VALUE,UINT numHistogramBins=10,UINT numDimensions = 1,bool useTrajStartAndEndValues = false,bool useWeightedMagnitudeValues = true);
	
    /**
     Copy constructor, copies the MovementTrajectoryFeatures from the rhs instance to this instance.
     
     @param const MovementTrajectoryFeatures &rhs: another instance of the MovementTrajectoryFeatures class from which the data will be copied to this instance
     */
    MovementTrajectoryFeatures(const MovementTrajectoryFeatures &rhs);
    
    /**
     Default Destructor
     */
    virtual ~MovementTrajectoryFeatures();
    
    /**
     Sets the equals operator, copies the data from the rhs instance to this instance.
     
     @param const MovementTrajectoryFeatures &rhs: another instance of the MovementTrajectoryFeatures class from which the data will be copied to this instance
     @return a reference to this instance of MovementTrajectoryFeatures
     */
    MovementTrajectoryFeatures& operator=(const MovementTrajectoryFeatures &rhs);

    /**
     Sets the FeatureExtraction clone function, overwriting the base FeatureExtraction function.
     This function is used to clone the values from the input pointer to this instance of the FeatureExtraction module.
     This function is called by the GestureRecognitionPipeline when the user adds a new FeatureExtraction module to the pipeleine.
     
     @param FeatureExtraction *featureExtraction: a pointer to another instance of a MovementTrajectoryFeatures, the values of that instance will be cloned to this instance
     @return returns true if the clone was successful, false otherwise
     */
    virtual bool clone(const FeatureExtraction *featureExtraction);
    
    /**
     Sets the FeatureExtraction computeFeatures function, overwriting the base FeatureExtraction function.
     This function is called by the GestureRecognitionPipeline when any new input data needs to be processed (during the prediction phase for example).
     This function calls the MovementTrajectoryFeatures's update function.
     
     @param vector< double > inputVector: the inputVector that should be processed.  Must have the same dimensionality as the FeatureExtraction module
     @return returns true if the data was processed, false otherwise
     */
    virtual bool computeFeatures(vector< double > inputVector);
    
    /**
     Sets the FeatureExtraction reset function, overwriting the base FeatureExtraction function.
     This function is called by the GestureRecognitionPipeline when the pipelines main reset() function is called.
     This function resets the feature extraction by re-initiliazing the instance.
     
     @return true if the filter was reset, false otherwise
     */
    virtual bool reset();
    
    /**
     This saves the feature extraction settings to a file.
     This overrides the saveSettingsToFile function in the FeatureExtraction base class.
     
     @param string filename: the name of the file to save the settings to
     @return returns true if the settings were saved successfully, false otherwise
     */
    virtual bool saveSettingsToFile(string filename);
    
    /**
     This saves the feature extraction settings to a file.
     This overrides the saveSettingsToFile function in the FeatureExtraction base class.
     
     @param fstream &file: a reference to the file to save the settings to
     @return returns true if the settings were saved successfully, false otherwise
     */
    virtual bool saveSettingsToFile(fstream &file);
    
    /**
     This loads the feature extraction settings from a file.
     This overrides the loadSettingsFromFile function in the FeatureExtraction base class.
     
     @param string filename: the name of the file to load the settings from
     @return returns true if the settings were loaded successfully, false otherwise
     */
    virtual bool loadSettingsFromFile(string filename);
    
    /**
     This loads the feature extraction settings from a file.
     This overrides the loadSettingsFromFile function in the FeatureExtraction base class.
     
     @param fstream &file: a reference to the file to load the settings from
     @return returns true if the settings were loaded successfully, false otherwise
     */
    virtual bool loadSettingsFromFile(fstream &file);

    /**
     Initializes the MovementTrajectoryFeatures
     */
    bool init(UINT trajectoryLength,UINT numCentroids,UINT featureMode,UINT numHistogramBins,UINT numDimensions,bool useTrajStartAndEndValues,bool useWeightedMagnitudeValues);
    
    /**
     Computes the features from the input, this should only be called if the dimensionality of this instance was set to 1.
     
     @param double x: the value to compute features from, this should only be called if the dimensionality of the filter was set to 1
	 @return a vector containing the features, an empty vector will be returned if the features were not computed
     */
	vector< double > update(double x);
    
    /**
     Computes the features from the input, the dimensionality of x should match that of this instance.
     
     @param double x: the value to compute features from, this should only be called if the dimensionality of the filter was set to 1
	 @return a vector containing the features, an empty vector will be returned if the features were not computed
     */
    vector< double > update(vector< double > x);
    
    /**
     */
    CircularBuffer< vector< double > > getTrajectoryData();
    
    /**
     */
    Matrix< double > getCentroids();

protected:
    void cartToPolar(double x,double y,double &r, double &theta);
    
    UINT trajectoryLength;
    UINT numCentroids;
    UINT featureMode;
    UINT numHistogramBins;
    bool useTrajStartAndEndValues;
    bool useWeightedMagnitudeValues;
    CircularBuffer< vector< double > > trajectoryDataBuffer;
    Matrix< double > centroids;
    
    static RegisterFeatureExtractionModule< MovementTrajectoryFeatures > registerModule;
    
public:
    enum FeatureModes{CENTROID_VALUE=0,NORMALIZED_CENTROID_VALUE,CENTROID_DERIVATIVE,CENTROID_ANGLE_2D,CENTROID_ANGLE_3D};
};

}//End of namespace GRT