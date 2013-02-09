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

//Include the platform specific time headers
#if defined(__GRT_WINDOWS_BUILD__)
	#include <windows.h>
    #include <mmsystem.h>
#endif

#if defined(__GRT_OSX_BUILD__)
    #include <sys/time.h>
#endif

#if defined(__GRT_LINUX_BUILD__)
    #include <sys/time.h>
#endif

namespace GRT{

class Timer{
public:
    Timer(){}
    ~Timer(){}

    bool start(){
        startTime = getSystemTime();
        timerRunning = true;
        timerMode = NORMAL_MODE;
        return true;
    }

    bool start(double countDownTime){
        if( countDownTime > 0 ){
            startTime = getSystemTime();
            timerRunning = true;
            timerMode = COUNTDOWN_MODE;
            this->countDownTime = countDownTime;
            return true;
        }
        return false;
    }
    
    bool stop(){
        timerRunning = false;
        return true;
    }

    //Getters
    double getMilliSeconds(){
        if( !timerRunning ) return 0;

        unsigned long now = getSystemTime();

        switch( timerMode ){
            case NORMAL_MODE:
                return double(now-startTime);
                break;
            case COUNTDOWN_MODE:
                return (countDownTime - double(now-startTime));
                break;
            default:
                return 0;
                break;
        }

        return 0;

    }

    double getSeconds(){
        if( !timerRunning ) return 0;
        return getMilliSeconds()/1000.0;
    }

    bool running(){ return timerRunning; }
    bool timerReached(){
        if( !timerRunning ){
            return false;
        }

        if( getMilliSeconds() > 0 ) return false;
        return true;
    }

    unsigned long getSystemTime( ) {
#ifdef __GRT_OSX_BUILD__
            struct timeval now;
            gettimeofday( &now, NULL );
            return now.tv_usec/1000 + now.tv_sec*1000;
#endif
#ifdef __GRT_WINDOWS_BUILD__
			SYSTEMTIME systemTime;
            GetSystemTime(&systemTime);
			return (systemTime.wHour*60*60*1000) + (systemTime.wMinute*60*1000) + (systemTime.wSecond*1000) + systemTime.wMilliseconds;
#endif
#ifdef __GRT_LINUX_BUILD__
        struct timeval now;
        gettimeofday( &now, NULL );
        return now.tv_usec/1000 + now.tv_sec*1000;
#endif
        return 0;
    }

private:
    unsigned long startTime;
    unsigned int timerMode;
    double countDownTime;
    bool timerRunning;

    enum TimerModes{NORMAL_MODE=0,COUNTDOWN_MODE};

};

}; //End of namespace GRT
