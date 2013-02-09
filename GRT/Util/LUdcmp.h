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
 
 This code is based on the LU Decomposition code from Numerical Recipes (3rd Edition)
 
 */
#pragma once
#include "GRTCommon.h"
#include "../DataStructures/Matrix.h"

namespace GRT {
    
class LUdcmp{
	
public:
    LUdcmp(Matrix<double> a);
	~LUdcmp();
	bool solve_vector(vector <double> &b,vector <double> &x);
	bool solve(Matrix <double> &b,Matrix <double> &x);
	bool inverse(Matrix <double> &ainv);
	double det();
	bool mprove(vector <double> &b,vector <double> &x);
    bool sing;
	Matrix <double> lu;

private:
	int n;
	double d;
	vector <int> indx;
	Matrix <double> &aref;
    
    DebugLog debugLog;
    ErrorLog errorLog;
    WarningLog warningLog;
};

}//End of namespace GRT