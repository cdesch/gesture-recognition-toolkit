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

#include "FeatureExtraction.h"
namespace GRT{
    
FeatureExtraction::StringFeatureExtractionMap* FeatureExtraction::stringFeatureExtractionMap = NULL;
UINT FeatureExtraction::numFeatureExtractionInstances = 0;
    
FeatureExtraction* FeatureExtraction::createInstanceFromString(string const &featureExtractionType){
    
    StringFeatureExtractionMap::iterator iter = getMap()->find( featureExtractionType );
    if( iter == getMap()->end() ){
        return NULL;
    }
    return iter->second();
}
    
FeatureExtraction::FeatureExtraction(void){
    featureExtractionType = "NOT_SET"; 
    initialized = false; 
    featureDataReady = false;
    numInputDimensions = 0;
    numOutputDimensions = 0;
    numFeatureExtractionInstances++;
}
    
FeatureExtraction::~FeatureExtraction(void){
    if( --numFeatureExtractionInstances == 0 ){
        delete stringFeatureExtractionMap;
        stringFeatureExtractionMap = NULL;
    }
}

bool FeatureExtraction::copyBaseVariables(FeatureExtraction *featureExtractionA,const FeatureExtraction *featureExtractionB){
    
    if( featureExtractionA == NULL || featureExtractionB == NULL ) return false;
    
    featureExtractionA->featureExtractionType = featureExtractionB->featureExtractionType;
    featureExtractionA->initialized = featureExtractionB->initialized;
    featureExtractionA->featureDataReady = featureExtractionB->featureDataReady;
    featureExtractionA->numInputDimensions = featureExtractionB->numInputDimensions;
    featureExtractionA->numOutputDimensions = featureExtractionB->numOutputDimensions;
    featureExtractionA->featureVector = featureExtractionB->featureVector;
    featureExtractionA->debugLog = featureExtractionB->debugLog;
    featureExtractionA->errorLog = featureExtractionB->errorLog;
    featureExtractionA->warningLog = featureExtractionB->warningLog;

    return true;
}

FeatureExtraction* FeatureExtraction::createNewInstance() const{
    return createInstanceFromString(featureExtractionType);
}
    
string FeatureExtraction::getFeatureExtractionType() const{ 
    return featureExtractionType; 
}

UINT FeatureExtraction::getNumInputDimensions() const{ 
    return numInputDimensions; 
}

UINT FeatureExtraction::getNumOutputDimensions() const{ 
    return numOutputDimensions; 
}

bool FeatureExtraction::getInitialized() const{ 
    return initialized; 
}
    
bool FeatureExtraction::getFeatureDataReady() const{
    return featureDataReady;
}

vector< double > FeatureExtraction::getFeatureVector() const{ 
    return featureVector; 
}
    

} //End of namespace GRT

