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
 */

#pragma once

#include "../DataStructures/LabelledTimeSeriesClassificationData.h"

namespace GRT{

class LabelledTimeSeriesClassificationSampleTrimmer{
public:

	/**
     Default Constructor.
    */
    LabelledTimeSeriesClassificationSampleTrimmer(double trimThreshold=0.1,double maximumTrimPercentage=20);

	/**
     Default Destructor
    */
	~LabelledTimeSeriesClassificationSampleTrimmer();

	LabelledTimeSeriesClassificationSampleTrimmer& operator= (const LabelledTimeSeriesClassificationSampleTrimmer &rhs){
		if( this != &rhs){
            this->trimThreshold = rhs.trimThreshold;
            this->maximumTrimPercentage = rhs.maximumTrimPercentage;
            this->debugLog = rhs.debugLog;
            this->warningLog = rhs.warningLog;
            this->errorLog = rhs.errorLog;
		}
		return *this;
	}

    bool trimTimeSeries(LabelledTimeSeriesClassificationSample &timeSeries);

private:
    double trimThreshold;
    double maximumTrimPercentage;
    DebugLog debugLog;
    WarningLog warningLog;
    ErrorLog errorLog;
    
};

} //End of namespace GRT