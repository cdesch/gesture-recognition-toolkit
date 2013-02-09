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

#include "MLBase.h"
#include "../DataStructures/LabelledClassificationData.h"
#include "../DataStructures/LabelledTimeSeriesClassificationData.h"

namespace GRT{
    
#define DEFAULT_NULL_LIKELIHOOD_VALUE 0
#define DEFAULT_NULL_DISTANCE_VALUE 0

class Regressifier : public MLBase
{
public:
	Regressifier(void);
    
	virtual ~Regressifier(void);
    
    virtual bool clone(const Regressifier *regressifier){ return false; }
    
    virtual bool copyBaseVariables(Regressifier *regressifierA,const Regressifier *regressifierB);

    //Training methods
    virtual bool train(LabelledRegressionData &trainingData){ return false; }
    
    //Getters
    string getRegressifierType() const;

    /**
     Gets the number of dimensions in trained model.
     
     @return returns the number of dimensions in the trained model, a value of 0 will be returned if the model has not been trained
     */
    UINT getNumOutputDimensions() const;
    vector< double > getRegressionData() const;
    
    //Setters
    
    typedef std::map< string, Regressifier*(*)() > StringRegressifierMap;
    static Regressifier* createInstanceFromString(string const &regressifierType);
    Regressifier* createNewInstance() const;
    
protected:
    string regressifierType;
    UINT numOutputDimensions;
    vector< double > regressionData;
    
    static StringRegressifierMap *getMap() {
        if( !stringRegressifierMap ){ stringRegressifierMap = new StringRegressifierMap; } 
        return stringRegressifierMap; 
    }
    
private:
    static StringRegressifierMap *stringRegressifierMap;
    static UINT numRegressifierInstances;

};
    
//These two functions/classes are used to register any new Regression Module with the Regressifier base class
template< typename T >  Regressifier *newRegressionModuleInstance() { return new T; }

template< typename T > 
class RegisterRegressifierModule : Regressifier { 
public:
    RegisterRegressifierModule(string const &newRegresionModuleName) { 
        getMap()->insert( std::pair<string, Regressifier*(*)()>(newRegresionModuleName, &newRegressionModuleInstance< T > ) );
    }
};

} //End of namespace GRT

