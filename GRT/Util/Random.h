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

#include "../Util/GRTVersionInfo.h"
#include "Timer.h"

namespace GRT{

class Random{
public:
    Random(unsigned long long seed = 0):v(4101842887655102017LL), w(1), storedval(0.0){
        if( seed == 0 ){
            Timer t;
            seed = (unsigned long long)t.getSystemTime();
        }
        setSeed( seed );
    }
    ~Random(){
    }
    
    void setSeed(unsigned long long seed){
        v = 4101842887655102017LL;
        w = 1;
        storedval = 0;
        u = seed ^ v; int64();
        v = u; int64();
        w = v; int64();
    }
    
    inline int getRandomNumberInt(int minRange,int maxRange){
        return int( floor(getRandomNumberUniform(minRange,maxRange)) );
    }
    
    inline double getRandomNumberUniform(double minRange=0.0,double maxRange=1.0){
        return (doub()*(maxRange-minRange))+minRange;
    }
    
    double getRandomNumberGauss(double mu=0.0,double sigma=1.0){
        double v1,v2,rsq,fac;
        
        if (storedval == 0.){
            do {
                v1=2.0*doub()-1.0;
                v2=2.0*doub()-1.0;
                rsq=v1*v1+v2*v2;
            } while (rsq >= 1.0 || rsq == 0.0);
            fac=sqrt(-2.0*log(rsq)/rsq);
            storedval = v1*fac;
            return mu + sigma*v2*fac;
        } else {
            fac = storedval;
            storedval = 0.;
            return mu + sigma*fac;
        }
    }

private:
    inline unsigned long long int64() {
        u = u * 2862933555777941757LL + 7046029254386353087LL;
        v ^= v >> 17; v ^= v << 31; v ^= v >> 8;
        w = 4294957665U*(w & 0xffffffff) + (w >> 32);
        unsigned long long x = u ^ (u << 21); x ^= x >> 35; x ^= x << 4;
        return (x + v) ^ w;
    }
    inline double doub() { return 5.42101086242752217E-20 * int64(); }
    inline unsigned int int32() { return (unsigned int)int64(); } 
    
    unsigned long long u;
    unsigned long long v;
    unsigned long long w;
    double storedval;               //This is for the Gauss Box-Muller 
};

}; //End of namespace GRT