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

#include "Util.h"

namespace GRT{
    
bool Util::sleep(unsigned int numMilliseconds){
    
#if defined( __GRT_WINDOWS_BUILD__ )
    Sleep( numMilliseconds );
    return true;
#endif
    
#if defined(__GRT_OSX_BUILD__)
    usleep( numMilliseconds * 1000 );
    return true;
#endif
    
#if defined(__GRT_LINUX_BUILD__)
    usleep( numMilliseconds * 1000 );
    return true;
#endif
    
}
    
double Util::scale(double x,double minSource,double maxSource,double minTarget,double maxTarget){
    return (((x-minSource)*(maxTarget-minTarget))/(maxSource-minSource))+minTarget;
}
    
std::string Util::intToString(int i){
    std::stringstream s;
    s << i;
    return s.str();
}
    
std::string Util::intToString(unsigned int i){
    std::stringstream s;
    s << i;
    return s.str();
}
    
}; //End of namespace GRT