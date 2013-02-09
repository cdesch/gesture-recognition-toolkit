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

#pragma once

#include "../Util/GRTCommon.h"

namespace GRT{

class Context
{
public:
	Context(void){ 
        contextType = "NOT_SET"; 
        initialized = false; 
        okToContinue = true;
        numInputDimensions = 0;
        numOutputDimensions = 0;
    }
	virtual ~Context(void){}
    
    //Clone method
    virtual bool clone(Context *rhs){ return false; }
    
    bool copyBaseVariables(Context *contextA,Context *contextB){
        
        if( contextA == NULL || contextB == NULL ) return false;
        
        contextA->contextType = contextB->contextType;
        contextA->initialized = contextB->initialized;
        contextA->okToContinue = contextB->okToContinue;
        contextA->numInputDimensions = contextB->numInputDimensions;
        contextA->numOutputDimensions = contextB->numOutputDimensions;
        contextA->data = contextB->data;
        contextA->debugLog = contextB->debugLog;
        contextA->errorLog = contextB->errorLog;
        contextA->warningLog = contextB->warningLog;
        return true;
    }

    virtual bool process(vector< double > inputVector){ return false; }
    virtual bool reset(){ return false; }
    
    virtual bool updateContext(bool value){ return false; }
    
    //Getters
    string getContextType(){ return contextType; }
    UINT getNumInputDimensions(){ return numInputDimensions; }
    UINT getNumOutputDimensions(){ return numOutputDimensions; }
    bool getInitialized(){ return initialized; }
    bool getOK(){ return okToContinue; }
    vector< double > getProcessedData(){ return data; }
    
    //Setters
    
protected:
    string contextType;
    bool initialized;
    bool okToContinue;
    UINT numInputDimensions;
    UINT numOutputDimensions;
    vector< double > data;
    DebugLog debugLog;
    ErrorLog errorLog;
    WarningLog warningLog;
};

} //End of namespace GRT

