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

//Include the GRTVersionInfo header to find which operating system we are building for
#include "GRTVersionInfo.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>

#ifdef __GRT_WINDOWS_BUILD__
//Include any Windows specific headers
#include <windows.h>
//Hey User: Make sure you add the path to the Kernel32.lib to your lib search paths
#pragma comment(lib,"Kernel32.lib")
#endif

#ifdef __GRT_OSX_BUILD__
//Include any OSX specific headers
#include <unistd.h>
#endif

#ifdef __GRT_LINUX_BUILD__
//Include any Linux specific headers
#include <unistd.h>
#endif

namespace GRT{

class Util{
public:
    Util(){
        
    }
    ~Util(){
        
    }
    
    static bool sleep(unsigned int numMilliseconds);
    static double scale(double x,double minSource,double maxSource,double minTarget,double maxTarget);
    static std::string intToString(int i);
    static std::string intToString(unsigned int i);
    
};
    
}; //End of namespace GRT