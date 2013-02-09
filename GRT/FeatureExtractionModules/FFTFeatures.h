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
#include "FFT.h"

namespace GRT{

class FFTFeatures : public FeatureExtraction
{
public:
    
	FFTFeatures(UINT fftWindowSize=FFT::FFT_WINDOW_SIZE_512,UINT numDimensions=1,bool computeMaxFreqFeature = true,bool computeMaxFreqSpectrumRatio = true,bool computeCentroidFeature = true,bool computeTopNFreqFeatures = true,UINT N = 10);
    
    /**
     Copy Constructor, copies the FFTFeatures from the rhs instance to this instance
     
	 @param const FFTFeatures &rhs: another instance of the FFTFeatures class from which the data will be copied to this instance
     */
    FFTFeatures(const FFTFeatures &rhs);
    
    /**
     Default Destructor
     */
	virtual ~FFTFeatures(void);
    
    /**
     Sets the equals operator, copies the data from the rhs instance to this instance
     
	 @param const FFTFeatures &rhs: another instance of the FFTFeatures class from which the data will be copied to this instance
	 @return a reference to this instance of FFT
     */
    FFTFeatures& operator=(const FFTFeatures &rhs);

    /**
     Sets the FeatureExtraction clone function, overwriting the base FeatureExtraction function.
     This function is used to clone the values from the input pointer to this instance of the FeatureExtraction module.
     This function is called by the GestureRecognitionPipeline when the user adds a new FeatureExtraction module to the pipeline.
     
	 @param FeatureExtraction *featureExtraction: a pointer to another instance of an FFTFeatures, the values of that instance will be cloned to this instance
	 @return true if the clone was successful, false otherwise
     */
    virtual bool clone(const FeatureExtraction *featureExtraction);
    
    /**
     Sets the FeatureExtraction computeFeatures function, overwriting the base FeatureExtraction function.
     This function is called by the GestureRecognitionPipeline when any new input data needs to be processed (during the prediction phase for example).
     
	 @param vector< double > inputVector: the inputVector that should be processed.  Must have the same dimensionality as the FeatureExtraction module
	 @return true if the data was processed, false otherwise
     */
    virtual bool computeFeatures(vector< double > inputVector);
    
    /**
     Sets the FeatureExtraction reset function, overwriting the base FeatureExtraction function.
     This function is called by the GestureRecognitionPipeline when the pipelines main reset() function is called.
     This function resets the FFTFeatures by re-initiliazing the instance.
     
	 @return true if the FFTFeatures was reset, false otherwise
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
     Initializes the FFTFeatures. 
     Should be called before calling the computeFFT(...) or computeFeatures(...) methods.
     This function is automatically called by the constructor.
     
	 @return true if the FTT was initialized, false otherwise
     */   
    bool init(UINT fftWindowSize,UINT numChannelsInFFTSignal,bool computeMaxFreqFeature,bool computeMaxFreqSpectrumRatio,bool computeCentroidFeature,bool computeTopNFreqFeatures,UINT N);
    
protected:
    UINT fftWindowSize;
    UINT numChannelsInFFTSignal;
    bool computeMaxFreqFeature;
    bool computeMaxFreqSpectrumRatio;
    bool computeCentroidFeature;
    bool computeTopNFreqFeatures;
    
    UINT N;
    double maxFreqFeature;
    double maxFreqSpectrumRatio;
    double centroidFeature;
    vector< double > topNFreqFeatures;
    
    static RegisterFeatureExtractionModule< FFTFeatures > registerModule;
    
public:
};

} //End of namespace GRT

