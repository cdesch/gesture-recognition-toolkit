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

#include "Classifier.h"
namespace GRT{
    
Classifier::StringClassifierMap* Classifier::stringClassifierMap = NULL;
UINT Classifier::numClassifierInstances = 0;
    
Classifier* Classifier::createInstanceFromString(string const &classifierType){
    
    StringClassifierMap::iterator iter = getMap()->find( classifierType );
    if( iter == getMap()->end() ){
        return NULL;
    }
    return iter->second();
}
    
Classifier::Classifier(void){
    baseType = MLBase::CLASSIFIER;
    classifierMode = STANDARD_CLASSIFIER_MODE;
    classifierType = "NOT_SET";
    useNullRejection = true;
    numFeatures = 0;
    numClasses = 0;
    predictedClassLabel = 0;
    maxLikelihood = 0;
    bestDistance = 0;
    nullRejectionCoeff = 5;
    numClassifierInstances++;
}
    
Classifier::~Classifier(void){
    if( --numClassifierInstances == 0 ){
        delete stringClassifierMap;
        stringClassifierMap = NULL;
    }
}
    
bool Classifier::copyBaseVariables(Classifier *classifierA,const Classifier *classifierB){
    
    if( classifierA == NULL || classifierB == NULL ){
        errorLog << "copyBaseVariables(Classifier *classifierA,Classifier *classifierB) -ClassifierA or ClassifierB is NULL!" << endl;
        return false;
    }
    
    if( !copyMLBaseVariables((MLBase*)classifierA, (MLBase*)classifierB) ){
        errorLog << "copyBaseVariables(Classifier *classifierA,Classifier *classifierB) -Failed to copy MLBaseVariables" << endl;
        return false;
    }
    
    classifierA->classifierType = classifierB->classifierType;
    classifierA->classifierMode = classifierB->classifierMode;
    classifierA->useNullRejection = classifierB->useNullRejection;
    classifierA->numClasses = classifierB->numClasses;
    classifierA->predictedClassLabel = classifierB->predictedClassLabel;
    classifierA->nullRejectionCoeff = classifierB->nullRejectionCoeff;
    classifierA->maxLikelihood = classifierB->maxLikelihood;
    classifierA->bestDistance = classifierB->bestDistance;
    classifierA->classLabels = classifierB->classLabels;
    classifierA->classLikelihoods = classifierB->classLikelihoods;
    classifierA->classDistances = classifierB->classDistances;
    classifierA->nullRejectionThresholds = classifierB->nullRejectionThresholds;
    
    return true;
}
    
Classifier* Classifier::createNewInstance() const{
    return createInstanceFromString( classifierType );
}
    
string Classifier::getClassifierType() const{ 
    return classifierType; 
}
    
bool Classifier::getNullRejectionEnabled() const{ 
    return useNullRejection; 
}

double Classifier::getNullRejectionCoeff() const{ 
    return nullRejectionCoeff; 
}
    
double Classifier::getMaximumLikelihood() const{ 
    if( trained ) return maxLikelihood; 
    return DEFAULT_NULL_LIKELIHOOD_VALUE; 
}
    
double Classifier::getBestDistance() const{ 
    if( trained ) return bestDistance; 
    return DEFAULT_NULL_DISTANCE_VALUE; 
}

UINT Classifier::getNumClasses() const{  
    return numClasses; 
}

UINT Classifier::getPredictedClassLabel() const{ 
    if( trained ) return predictedClassLabel; 
    return 0; 
}


vector< double > Classifier::getClassLikelihoods() const{ 
    if( trained ) return classLikelihoods;
    return vector< double>(); 
}

vector< double > Classifier::getClassDistances() const{ 
    if( trained ) return classDistances; 
    return vector< double>(); 
}

vector< double > Classifier::getNullRejectionThresholds() const{ 
    if( trained ) return nullRejectionThresholds;
    return vector< double>(); 
}

vector< UINT > Classifier::getClassLabels() const{ 
    return classLabels;
}

bool Classifier::enableNullRejection(bool useNullRejection){ 
    this->useNullRejection = useNullRejection; 
    return true;
}

bool Classifier::setNullRejectionCoeff(double nullRejectionCoeff){ 
    if( nullRejectionCoeff > 0 ){ 
        this->nullRejectionCoeff = nullRejectionCoeff; 
        return true; 
    } 
    return false; 
}

} //End of namespace GRT

