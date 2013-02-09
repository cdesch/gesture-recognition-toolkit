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

#include "Regressifier.h"
namespace GRT{
    
Regressifier::StringRegressifierMap* Regressifier::stringRegressifierMap = NULL;
UINT Regressifier::numRegressifierInstances = 0;
    
Regressifier* Regressifier::createInstanceFromString(string const &regressifierType){
    
    StringRegressifierMap::iterator iter = getMap()->find( regressifierType );
    if( iter == getMap()->end() ){
        return NULL;
    }
    return iter->second();
}
    
Regressifier::Regressifier(void){
    baseType = MLBase::REGRESSIFIER;
    regressifierType = "NOT_SET";
    numOutputDimensions = 0;
    numRegressifierInstances++;
}
    
Regressifier::~Regressifier(void){
    if( --numRegressifierInstances == 0 ){
        delete stringRegressifierMap;
        stringRegressifierMap = NULL;
    }
}
    
bool Regressifier::copyBaseVariables(Regressifier *regressifierA,const Regressifier *regressifierB){
    
    if( regressifierA == NULL || regressifierB == NULL ){
        errorLog << "copyBaseVariables(Regressifier *regressifierA,Regressifier *regressifierB) - regressifierA or regressifierB is NULL!" << endl;
        return false;
    }
    
    if( !copyMLBaseVariables((MLBase*)regressifierA, (MLBase*)regressifierB) ){
        errorLog << "copyBaseVariables(Regressifier *regressifierA,Regressifier *regressifierB) - Failed to copy MLBaseVariables" << endl;
        return false;
    }
    
    regressifierA->baseType = regressifierB->baseType;
    regressifierA->regressifierType = regressifierB->regressifierType;
    regressifierA->numOutputDimensions = regressifierB->numOutputDimensions;
    
    return true;
}
    
Regressifier* Regressifier::createNewInstance() const{
    return createInstanceFromString( regressifierType );
}

string Regressifier::getRegressifierType() const{ 
    return regressifierType; 
}
    
UINT Regressifier::getNumOutputDimensions() const{ 
    if( trained ){ 
        return numOutputDimensions; 
    } 
    return 0; 
}

vector< double > Regressifier::getRegressionData() const{ 
    if( trained ){ 
        return regressionData; 
    } 
    return vector< double>(); 
}

} //End of namespace GRT

