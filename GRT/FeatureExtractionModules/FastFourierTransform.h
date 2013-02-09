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

/*
 This code is based on Dominic Mazzoni's FFT c++ wrapper, which is based on a free FFT implementation
 by Don Cross (http://www.intersrv.com/~dcross/fft.html) and the FFT algorithms from Numerical Recipes.
 */

#pragma once

#include "../Util/GRTCommon.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

namespace GRT{

class FastFourierTransform{
	
public:
		
	FastFourierTransform();
    
    FastFourierTransform(const FastFourierTransform &rhs);
    
    ~FastFourierTransform();
    
    FastFourierTransform& operator=(const FastFourierTransform &rhs);
    
    bool init(unsigned int windowSize,unsigned int windowFunction = RECTANGULAR_WINDOW,bool computeMagnitude = true,bool computePhase = true);
    
    bool computeFFT(double *data);
    
    vector< double > getMagnitudeData();
    vector< double > getPhaseData();
    vector< double > getPowerData();
    double getAveragePower();
    double *getMagnitudeDataPtr();
    double *getPhaseDataPtr();
    double *getPowerDataPtr();
    
    UINT getFFTSize(){ return windowSize; }
    
protected:
    bool windowData(double *data);
    bool realFFT(double *RealIn, double *RealOut, double *ImagOut);
    bool FFT(int NumSamples,bool InverseTransform,double *RealIn, double *ImagIn, double *RealOut, double *ImagOut);
    int numberOfBitsNeeded(int PowerOfTwo);
    int reverseBits(int index, int NumBits);
    void initFFT();
    inline int fastReverseBits(int i, int NumBits);
    inline bool isPowerOfTwo(unsigned int x);
    
    unsigned int windowSize;
    unsigned int windowFunction;
    bool initialized;
    bool computeMagnitude;
    bool computePhase;
    double *fftReal;
    double *fftImag;
    double *tmpReal;
    double *tmpImag;
    double *magnitude;
    double *phase;
    double *power;
    double averagePower;
    const static int MaxFastBits = 16;
    int **gFFTBitTable;
    
public:
    enum WindowFunctionOptions{RECTANGULAR_WINDOW=0,BARTLETT_WINDOW,HAMMING_WINDOW,HANNING_WINDOW};

};
    
}//End of namespace GRT
