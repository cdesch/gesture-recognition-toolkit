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

#include "FFT.h"

namespace GRT{
    
//Register the FFT module with the FeatureExtraction base class
RegisterFeatureExtractionModule< FFT > FFT::registerModule("FFT");
    
string FFT::getWindowSizeAsString(UINT windowSize){
    switch( windowSize ){
        case FFT_WINDOW_SIZE_16:
            return "FFT_WINDOW_SIZE_16";
            break;
        case FFT_WINDOW_SIZE_32:
            return "FFT_WINDOW_SIZE_32";
            break;
        case FFT_WINDOW_SIZE_64:
            return "FFT_WINDOW_SIZE_64";
            break;
        case FFT_WINDOW_SIZE_128:
            return "FFT_WINDOW_SIZE_128";
            break;
        case FFT_WINDOW_SIZE_256:
            return "FFT_WINDOW_SIZE_256";
            break;
        case FFT_WINDOW_SIZE_512:
            return "FFT_WINDOW_SIZE_512";
            break;
        case FFT_WINDOW_SIZE_1024:
            return "FFT_WINDOW_SIZE_1024";
            break;
        case FFT_WINDOW_SIZE_2048:
            return "FFT_WINDOW_SIZE_2048";
            break;
        case FFT_WINDOW_SIZE_4096:
            return "FFT_WINDOW_SIZE_4096";
            break;
        case FFT_WINDOW_SIZE_8192:
            return "FFT_WINDOW_SIZE_8192";
            break;
        case FFT_WINDOW_SIZE_16384:
            return "FFT_WINDOW_SIZE_16384";
            break;
        case FFT_WINDOW_SIZE_32768:
            return "FFT_WINDOW_SIZE_32768";
            break;
        default:
            break;
  
    }
    return "UNKNOWN FFT WINDOW SIZE";
}
    
UINT FFT::getWindowSizeAsInt(UINT windowSize){
    switch( windowSize ){
        case FFT_WINDOW_SIZE_16:
            return 16;
            break;
        case FFT_WINDOW_SIZE_32:
            return 32;
            break;
        case FFT_WINDOW_SIZE_64:
            return 64;
            break;
        case FFT_WINDOW_SIZE_128:
            return 128;
            break;
        case FFT_WINDOW_SIZE_256:
            return 256;
            break;
        case FFT_WINDOW_SIZE_512:
            return 512;
            break;
        case FFT_WINDOW_SIZE_1024:
            return 1024;
            break;
        case FFT_WINDOW_SIZE_2048:
            return 2048;
            break;
        case FFT_WINDOW_SIZE_4096:
            return 4096;
            break;
        case FFT_WINDOW_SIZE_8192:
            return 8192;
            break;
        case FFT_WINDOW_SIZE_16384:
            return 16384;
            break;
        case FFT_WINDOW_SIZE_32768:
            return 32768;
            break;
        default:
            break;
            
    }
    return 0;
}
    
FFT::FFT(unsigned int fftWindowSize,unsigned int hopSize,unsigned int numDimensions,unsigned int fftWindowFunction,bool computeMagnitude,bool computePhase):tempBuffer(NULL){ 
    featureExtractionType = "FFT"; 
    initialized = false; 
    featureDataReady = false;
    numInputDimensions = 0;
    numOutputDimensions = 0;
    
    //Setup the window size map
    windowSizeMap[ FFT_WINDOW_SIZE_16 ] = 16;
    windowSizeMap[ FFT_WINDOW_SIZE_32 ] = 32;
    windowSizeMap[ FFT_WINDOW_SIZE_64 ] = 64;
    windowSizeMap[ FFT_WINDOW_SIZE_128 ] = 128;
    windowSizeMap[ FFT_WINDOW_SIZE_256 ] = 256;
    windowSizeMap[ FFT_WINDOW_SIZE_512 ] = 512;
    windowSizeMap[ FFT_WINDOW_SIZE_1024 ] = 1024;
    windowSizeMap[ FFT_WINDOW_SIZE_2048 ] = 2048;
    windowSizeMap[ FFT_WINDOW_SIZE_4096 ] = 4096;
    windowSizeMap[ FFT_WINDOW_SIZE_8192 ] = 8192;
    windowSizeMap[ FFT_WINDOW_SIZE_16384 ] = 16384;
    windowSizeMap[ FFT_WINDOW_SIZE_32768 ] = 32768;
    
    if( hopSize > 0 && numDimensions > 0 ){
        init(fftWindowSize,hopSize,numDimensions,fftWindowFunction,computeMagnitude,computePhase);
    }
}
    
FFT::FFT(const FFT &rhs){
    this->hopSize = rhs.hopSize;
    this->dataBufferSize = rhs.dataBufferSize;
    this->fftWindowSize = rhs.fftWindowSize;
    this->fftWindowFunction = rhs.fftWindowFunction;
    this->hopCounter = rhs.hopCounter;
    this->computeMagnitude = rhs.computeMagnitude;
    this->computePhase = rhs.computePhase;
    tempBuffer = new double[ this->dataBufferSize ];
    this->dataBuffer = rhs.dataBuffer;
    this->fft = rhs.fft;
    this->windowSizeMap = rhs.windowSizeMap;
    
    copyBaseVariables((FeatureExtraction*)this, (FeatureExtraction*)&rhs);
}
    
FFT::~FFT(void){
    if( tempBuffer != NULL ){
        delete[] tempBuffer;
        tempBuffer = NULL;
    }
}
    
FFT& FFT::operator=(const FFT &rhs){
    if( this != &rhs ){
        if( tempBuffer != NULL ){
            delete[] tempBuffer;
            tempBuffer = NULL;
        }
        
        this->hopSize = rhs.hopSize;
        this->dataBufferSize = rhs.dataBufferSize;
        this->fftWindowSize = rhs.fftWindowSize;
        this->fftWindowFunction = rhs.fftWindowFunction;
        this->hopCounter = rhs.hopCounter;
        this->computeMagnitude = rhs.computeMagnitude;
        this->computePhase = rhs.computePhase;
        tempBuffer = new double[ this->dataBufferSize ];
        this->dataBuffer = rhs.dataBuffer;
        this->fft = rhs.fft;
        this->windowSizeMap = rhs.windowSizeMap;
        
        copyBaseVariables((FeatureExtraction*)this, (FeatureExtraction*)&rhs);
    }
    return *this;
}

//Clone method
bool FFT::clone(const FeatureExtraction *featureExtraction){ 
        
    if( featureExtraction == NULL ) return false;
    
    if( this->getFeatureExtractionType() == featureExtraction->getFeatureExtractionType() ){
        
        FFT *ptr = (FFT*)featureExtraction;
        //Clone the PeakDetection values 
        if( tempBuffer != NULL ){
            delete[] tempBuffer;
            tempBuffer = NULL;
        }
        
        this->hopSize = ptr->hopSize;
        this->dataBufferSize = ptr->dataBufferSize;
        this->fftWindowSize = ptr->fftWindowSize;
        this->fftWindowFunction = ptr->fftWindowFunction;
        this->hopCounter = ptr->hopCounter;
        this->computeMagnitude = ptr->computeMagnitude;
        this->computePhase = ptr->computePhase;
        tempBuffer = new double[ this->dataBufferSize ];
        this->dataBuffer = ptr->dataBuffer;
        this->fft = ptr->fft;
        this->windowSizeMap = ptr->windowSizeMap;
        
        copyBaseVariables((FeatureExtraction*)this, featureExtraction);
    }
    
    errorLog << "clone(FeatureExtraction *featureExtraction) -  FeatureExtraction Types Do Not Match!" << endl;
    
    return false;
}
    
bool FFT::saveSettingsToFile(string filename){
    
    if( !initialized ){
        errorLog << "saveSettingsToFile(string filename) - The ZeroCrossingCounter has not been initialized" << endl;
        return false;
    }
    
    std::fstream file; 
    file.open(filename.c_str(), std::ios::out);
    
    if( !saveSettingsToFile( file ) ){
        file.close();
        return false;
    }
    
    file.close();
    
    return true;
}

bool FFT::saveSettingsToFile(fstream &file){
    
    if( !file.is_open() ){
        errorLog << "saveSettingsToFile(fstream &file) - The file is not open!" << endl;
        return false;
    }
    
    file << "GRT_FFT_FILE_V1.0" << endl;
    file << "NumInputDimensions: " << numInputDimensions << endl;
    file << "NumOutputDimensions: " << numOutputDimensions << endl;
    file << "HopSize: " << hopSize << endl;
    file << "FftWindowSize: " << fftWindowSize << endl;
    file << "FftWindowFunction: " << fftWindowFunction << endl;
    file << "ComputeMagnitude: " << computeMagnitude << endl;
    file << "ComputePhase: " << computePhase << endl;
    
    return true;
}

bool FFT::loadSettingsFromFile(string filename){
    
    std::fstream file; 
    file.open(filename.c_str(), std::ios::in);
    
    if( !loadSettingsFromFile( file ) ){
        file.close();
        initialized = false;
        return false;
    }
    
    file.close();
    
    return true;
}

bool FFT::loadSettingsFromFile(fstream &file){
    
    if( !file.is_open() ){
        errorLog << "loadSettingsFromFile(fstream &file) - The file is not open!" << endl;
        return false;
    }
    
    string word;
    
    //Load the header
    file >> word;
    
    if( word != "GRT_FFT_FILE_V1.0" ){
        errorLog << "loadSettingsFromFile(fstream &file) - Invalid file format!" << endl;
        return false;     
    }
    
    //Load the NumInputDimensions
    file >> word;
    if( word != "NumInputDimensions:" ){
        errorLog << "loadSettingsFromFile(fstream &file) - Failed to read NumInputDimensions header!" << endl;
        return false;     
    }
    file >> numInputDimensions;
    
    //Load the NumOutputDimensions
    file >> word;
    if( word != "NumOutputDimensions:" ){
        errorLog << "loadSettingsFromFile(fstream &file) - Failed to read NumOutputDimensions header!" << endl;
        return false;     
    }
    file >> numOutputDimensions;
    
    file >> word;
    if( word != "HopSize:" ){
        errorLog << "loadSettingsFromFile(fstream &file) - Failed to read HopSize header!" << endl;
        return false;     
    }
    file >> hopSize;
    
    file >> word;
    if( word != "FftWindowSize:" ){
        errorLog << "loadSettingsFromFile(fstream &file) - Failed to read FftWindowSize header!" << endl;
        return false;     
    }
    file >> fftWindowSize;
    
    file >> word;
    if( word != "FftWindowFunction:" ){
        errorLog << "loadSettingsFromFile(fstream &file) - Failed to read FftWindowFunction header!" << endl;
        return false;     
    }
    file >> fftWindowFunction;
    
    file >> word;
    if( word != "ComputeMagnitude:" ){
        errorLog << "loadSettingsFromFile(fstream &file) - Failed to read ComputeMagnitude header!" << endl;
        return false;     
    }
    file >> computeMagnitude;
    
    file >> word;
    if( word != "ComputePhase:" ){
        errorLog << "loadSettingsFromFile(fstream &file) - Failed to read ComputePhase header!" << endl;
        return false;     
    }
    file >> computePhase;
    
    //Init the FFT module to ensure everything is initialized correctly
    return init(fftWindowSize,hopSize,numInputDimensions,fftWindowFunction,computeMagnitude,computePhase);
}

bool FFT::init(UINT fftWindowSize,UINT hopSize,UINT numDimensions,UINT fftWindowFunction,bool computeMagnitude,bool computePhase){
    
    initialized = false;
    
    if( !validateFFTWindowSize(fftWindowSize) ){
        errorLog << "init(UINT fftWindowSize,UINT hopSize,UINT numDimensions,UINT fftWindowFunction,bool computeMagnitude,bool computePhase) - Unknown fftWindowSize!" << endl;
        return false;
    }
    
    if( !validateFFTWindowFunction( fftWindowFunction ) ){
        errorLog << "init(UINT fftWindowSize,UINT hopSize,UINT numDimensions,UINT fftWindowFunction,bool computeMagnitude,bool computePhase) - Unkown Window Function!" << endl;
        return false;
    }
    
    if( tempBuffer != NULL ){
        delete[] tempBuffer;
        tempBuffer = NULL;
    }
    
    unsigned int fftSize = 0;
    std::map< unsigned int, unsigned int >::iterator iter = windowSizeMap.find( fftWindowSize );
    
    if( iter != windowSizeMap.end() ){
        fftSize = iter->second;
    }else{
        errorLog << "init(UINT fftWindowSize,UINT hopSize,UINT numDimensions,UINT fftWindowFunction,bool computeMagnitude,bool computePhase) - Failed to find fftSize in windowSizeMap!" << endl;
        return false;
    }
       
    this->dataBufferSize = fftSize;
    this->fftWindowSize = fftWindowSize;
    this->hopSize = hopSize;
    this->fftWindowFunction = fftWindowFunction;
    this->computeMagnitude = computeMagnitude;
    this->computePhase = computePhase;
    hopCounter = 0;
    featureDataReady = false;
    tempBuffer = new double[ dataBufferSize ];
    numInputDimensions = numDimensions;
    
    //Set the output size
    numOutputDimensions = 0;
    if( computePhase ) numOutputDimensions += fftSize * numDimensions;
    if( computeMagnitude ) numOutputDimensions += fftSize * numDimensions;
    
    //Resize the output feature vector
    featureVector.clear();
    featureVector.resize( numOutputDimensions, 0);
    
    dataBuffer.clear();
    dataBuffer.resize(dataBufferSize,vector< double >(numDimensions,0));
    fft.clear();
    fft.resize(numDimensions);
    
    for(unsigned int i=0; i<numDimensions; i++){
        if( !fft[i].init(fftSize,fftWindowFunction,computeMagnitude,computePhase) ){
            errorLog << "init(UINT fftWindowSize,UINT hopSize,UINT numDimensions,UINT fftWindowFunction,bool computeMagnitude,bool computePhase) - Failed to initialize fft!" << endl;
            return false;
        }
    }
    
    initialized = true;

    return true;
}
    
bool FFT::computeFeatures(vector< double > inputVector){ 
#ifdef GRT_SAFE_CHECKING
    if( !initialized ){
        errorLog << "computeFeatures(vector< double > inputVector) - Not initialized!" << endl;
        return false;
    }
    
    if( inputVector.size() != numInputDimensions ){
        errorLog << "computeFeatures(vector< double > inputVector) - The size of the inputVector (" << inputVector.size() << ") does not match that of the FeatureExtraction (" << numInputDimensions << ")!" << endl;
        return false;
    }
#endif
    
    return update(inputVector);
}
    
bool FFT::update(double x){
#ifdef GRT_SAFE_CHECKING
    if( !initialized ){
        errorLog << "update(double x) - Not initialized!" << endl;
        return false;
    }
    
    if( numInputDimensions != 1 ){
        errorLog << "update(double x) - The size of the input (1) does not match that of the FeatureExtraction (" << numInputDimensions << ")!" << endl;
        return false;
    }
#endif
    
    return update(vector<double>(1,x));
}

bool FFT::update(vector< double > x){
#ifdef GRT_SAFE_CHECKING
    if( !initialized ){
        errorLog << "update(vector<double> x) - Not initialized!" << endl;
        return false;
    }
    
    if( x.size() != numInputDimensions ){
        errorLog << "update(vector<double> x) - The size of the input (" << x.size() << ") does not match that of the FeatureExtraction (" << numInputDimensions << ")!" << endl;
        return false;
    }
#endif

    //Add the current input to the data buffers
    dataBuffer.push_back(x);
    
    featureDataReady = false;
    
    if( ++hopCounter == hopSize ){
        hopCounter = 0;
        //Compute the FFT for each dimension
        for(UINT j=0; j<numInputDimensions; j++){
            
            //Copy the input data for this dimension into the temp buffer
            for(UINT i=0; i<dataBufferSize; i++){
                tempBuffer[i] = dataBuffer[i][j];
            }
            
            //Compute the FFT
            if( !fft[j].computeFFT(tempBuffer) ){
                errorLog << "update(vector< double > x) - Failed to compute FFT!" << endl;
                return false;
            }
        }
        
        //Flag that the fft was computed during this update
        featureDataReady = true;
        
        //Copy the FFT data to the feature vector
        UINT index = 0;
        for(UINT j=0; j<numInputDimensions; j++){
            if( computeMagnitude ){
                double *mag = fft[j].getMagnitudeDataPtr();
                for(UINT i=0; i<fft[j].getFFTSize(); i++){
                    featureVector[index++] = *mag++;
                }
            }
            if( computePhase ){
                double *phase = fft[j].getPhaseDataPtr();
                for(UINT i=0; i<fft[j].getFFTSize(); i++){
                    featureVector[index++] = *phase++;
                }
            }
        }
    }
    
    return true;
}
    
bool FFT::reset(){ 
    if( initialized ) return init(fftWindowSize,hopSize,numInputDimensions,fftWindowFunction,computeMagnitude,computePhase);
    return false;
}
    
UINT FFT::getHopSize(){ 
    if(initialized){ return hopSize; } 
    return 0; 
}
    
UINT FFT::getDataBufferSize(){ 
    if(initialized){ return dataBufferSize; } 
    return 0; 
}
    
UINT FFT::getFFTWindowSize(){ 
    
    if( !initialized ) return 0;
    
    std::map< unsigned int, unsigned int >::iterator iter = windowSizeMap.find( fftWindowSize );
    if( iter != windowSizeMap.end() ){
        return iter->second;
    }
    errorLog << "getFFTWindowSize() - Failed to find fftSize in windowSizeMap!" << endl;
    return 0;
}
    
UINT FFT::getFFTWindowFunction(){ 
    if(initialized){ return fftWindowFunction; } 
    return 0; 
}
    
UINT FFT::getHopCounter(){ 
    if(initialized){ return hopCounter; } 
    return 0; 
}
    
bool FFT::setHopSize(UINT hopSize){
    if( hopSize > 0 ){
        this->hopSize = hopSize;
        hopCounter = 0;
        return true;
    }
    errorLog << "setHopSize(UINT hopSize) - The hopSize value must be greater than zero!" << endl;
    return false;
}

bool FFT::setFFTWindowSize(UINT fftWindowSize){
    if( validateFFTWindowSize(fftWindowSize) ){
        if( initialized ) return init(fftWindowSize, hopSize, numInputDimensions, fftWindowFunction, computeMagnitude, computePhase);
        this->fftWindowSize = fftWindowSize;
        return true;
    }
    errorLog << "setFFTWindowSize(UINT fftWindowSize) - Unkown FFT Window Size!" << endl;
    return false;

}
    
bool FFT::setFFTWindowFunction(UINT fftWindowFunction){
    if( validateFFTWindowFunction( fftWindowFunction ) ){
        this->fftWindowFunction = fftWindowFunction;
        return true;
    }
    return false;
}
    
bool FFT::setComputeMagnitude(bool computeMagnitude){
    if( initialized ) return init(fftWindowSize, hopSize, numInputDimensions, fftWindowFunction, computeMagnitude, computePhase);
    this->computeMagnitude = computeMagnitude;
    return true;
}
    
bool FFT::setComputePhase(bool computePhase){
    if( initialized ) return init(fftWindowSize, hopSize, numInputDimensions, fftWindowFunction, computeMagnitude, computePhase);
    this->computePhase = computePhase;
    return true;

}

bool FFT::isPowerOfTwo(unsigned int x){
    if (x < 2) return false;
    if (x & (x - 1)) return false;
    return true;
}
    
bool FFT::validateFFTWindowSize(UINT fftWindowSize){
    if( fftWindowSize != FFT_WINDOW_SIZE_16 && fftWindowSize != FFT_WINDOW_SIZE_32 && fftWindowSize != FFT_WINDOW_SIZE_64 &&
       fftWindowSize != FFT_WINDOW_SIZE_128 && fftWindowSize != FFT_WINDOW_SIZE_256 && fftWindowSize != FFT_WINDOW_SIZE_512 &&
       fftWindowSize != FFT_WINDOW_SIZE_1024 && fftWindowSize != FFT_WINDOW_SIZE_2048 && fftWindowSize != FFT_WINDOW_SIZE_4096 &&
       fftWindowSize != FFT_WINDOW_SIZE_8192 && fftWindowSize != FFT_WINDOW_SIZE_16384 && fftWindowSize != FFT_WINDOW_SIZE_32768 ){
        return false;
    }
    return true;
}
    
bool FFT::validateFFTWindowFunction(UINT fftWindowFunction){
    if( fftWindowFunction != RECTANGULAR_WINDOW && fftWindowFunction != BARTLETT_WINDOW && 
        fftWindowFunction != HAMMING_WINDOW && fftWindowFunction != HANNING_WINDOW ){
        return false;
    }
    return true;
}
    
    
}//End of namespace GRT