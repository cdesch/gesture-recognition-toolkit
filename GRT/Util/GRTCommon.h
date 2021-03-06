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

#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <float.h>
#include <math.h>
#include <time.h>
#include <map>

//Include the common classes
#include "GRTVersionInfo.h"
#include "MinMax.h"
#include "ClassTracker.h"
#include "IndexedDouble.h"
#include "DebugLog.h"
#include "ErrorLog.h"
#include "TrainingLog.h"
#include "WarningLog.h"
#include "CircularBuffer.h"
#include "Timer.h"
#include "Random.h"
#include "Util.h"

using namespace std;

namespace GRT{

//Declare any common definitions
#ifndef PI
    #define PI 3.14159265358979323846
#endif
    
#ifndef TWO_PI
	#define TWO_PI 6.28318530718
#endif
    
#define GRT_DEFAULT_NULL_CLASS_LABEL 0
#define GRT_SAFE_CHECKING true

//Declare any common typedefs, some of these are already declared in windef.h so if we are using Windows then we don't need to declare them
#ifndef __GRT_WINDOWS_BUILD__
	typedef unsigned int UINT;
	typedef signed int SINT;
	typedef unsigned long ULONG;
#endif

#ifdef __GRT_WINDOWS_BUILD__

	//NAN is not defined on Visual Studio version of math.h so define it here
	#ifndef NAN
        static const unsigned long __nan[2] = {0xffffffff, 0x7fffffff};
        #define NAN (*(const float *) __nan)
    #endif

	#ifndef INFINITY
		#define INFINITY (DBL_MAX+DBL_MAX)
	#endif

	//isnan is not defined on Visual Studio so define it here
	#ifndef isnan
		#define isnan(x) x != x
	#endif

	//isnan is not defined on Visual Studio so define it here
	#ifndef isinf
		#define isinf(x) (!isnan(x) && isnan(x - x))
	#endif
#endif
    
}; //End of namespace GRT