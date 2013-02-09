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

#include "PreProcessing.h"

namespace GRT{
    
PreProcessing::StringPreProcessingMap* PreProcessing::stringPreProcessingMap = NULL;
UINT PreProcessing::numPreProcessingInstances = 0;
    
PreProcessing* PreProcessing::createInstanceFromString(string const &preProcessingType){
    
    StringPreProcessingMap::iterator iter = getMap()->find( preProcessingType );
    if( iter == getMap()->end() ){
        return NULL;
    }
    return iter->second();
}
    
PreProcessing::PreProcessing(void){
    preProcessingType = "NOT_SET"; 
    initialized = false; 
    numInputDimensions = 0;
    numOutputDimensions = 0;
    numPreProcessingInstances++;
}
    
PreProcessing::~PreProcessing(void){
    if( --numPreProcessingInstances == 0 ){
        delete stringPreProcessingMap;
        stringPreProcessingMap = NULL;
    }
}

bool PreProcessing::copyBaseVariables(PreProcessing *preProcessingA,const PreProcessing *preProcessingB){
    
    if( preProcessingA == NULL || preProcessingB == NULL ) return false;
    
    preProcessingA->preProcessingType = preProcessingB->preProcessingType;
    preProcessingA->initialized = preProcessingB->initialized;
    preProcessingA->numInputDimensions = preProcessingB->numInputDimensions;
    preProcessingA->numOutputDimensions = preProcessingB->numOutputDimensions;
    preProcessingA->processedData = preProcessingB->processedData;
    preProcessingA->debugLog = preProcessingB->debugLog;
    preProcessingA->errorLog = preProcessingB->errorLog;
    preProcessingA->warningLog = preProcessingB->warningLog;
    return true;
}
    
PreProcessing* PreProcessing::createNewInstance() const{
    return createInstanceFromString(preProcessingType);
}
    
string PreProcessing::getPreProcessingType() const{ 
    return preProcessingType; 
}
    
UINT PreProcessing::getNumInputDimensions() const{ 
    return numInputDimensions; 
}
    
UINT PreProcessing::getNumOutputDimensions() const{ 
    return numOutputDimensions; 
}
    
bool PreProcessing::getInitialized() const{ 
    return initialized; 
}
    
vector< double > PreProcessing::getProcessedData() const{ 
    return processedData; 
}

} //End of namespace GRT

