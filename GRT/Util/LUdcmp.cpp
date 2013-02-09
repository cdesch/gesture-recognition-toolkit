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

#include "LUdcmp.h"

namespace GRT {

LUdcmp::LUdcmp(Matrix <double> a) : n( 1 ), lu(1,1), aref(a), indx(1), sing(false){
    
    debugLog.setProceedingText("[DEBUG LUdcmp]");
    errorLog.setProceedingText("[ERROR LUdcmp]");
    warningLog.setProceedingText("[WARNING LUdcmp]");
	
    n = a.getNumRows();

	lu.resize(a.getNumRows(),a.getNumCols());
	for(UINT r=0;r<a.getNumRows(); r++)
		for(UINT c=0;c<a.getNumCols(); c++)
			lu[r][c] = a[r][c];
	aref.resize(a.getNumRows(),a.getNumCols());
	for(UINT r=0;r<a.getNumRows(); r++)
		for(UINT c=0;c<a.getNumCols(); c++)
			aref[r][c] = a[r][c];

	indx.resize(a.getNumRows());

	const double TINY=1.0e-40;
	int i,imax,j,k;
	double big,temp;
	vector <double> vv(n);
	d=1.0;
	for (i=0;i<n;i++) {
		big=0.0;
		for (j=0;j<n;j++)
			if ((temp=fabs( lu[i][j] )) > big) big=temp;
		if (big == 0.0){
            sing = true;
            errorLog << "Error in LUdcmp constructor, big == 0.0" << endl;
            return;///ERRROR!!!!!!!!!!!!
		}
		vv[i] = 1.0/big;
	}
	for (k=0;k<n;k++) {
		big=0.0;
		for (i=k;i<n;i++) {
			temp=vv[i]*fabs(lu[i][k]);
			if (temp > big) {
				big=temp;
				imax=i;
			}
		}
		if (k != imax) {
			for (j=0;j<n;j++) {
				temp=lu[imax][j];
				lu[imax][j] = lu[k][j];
				lu[k][j] = temp;
			}
			d = -d;
			vv[imax]=vv[k];
		}
		indx[k]=imax;
		if (lu[k][k] == 0.0) lu[k][k] = TINY;
		for (i=k+1; i<n; i++) {
			temp = lu[i][k] /= lu[k][k];
			for (j=k+1;j<n;j++)
				lu[i][j] -= temp * lu[k][j];
		}
	}
	
}
LUdcmp::~LUdcmp(){

}
bool LUdcmp::solve_vector(vector <double> &b,vector <double> &x)
{
	int i;
	int ii=0;
	int ip;
	int j;
	double sum;
    
	if (b.size() != n || x.size() != n){
        errorLog << "solve_vector(vector <double> &b,vector <double> &x) - the size of the two vectors does not match!" << endl;
		return false;
    }
	for (i=0;i<n;i++) x[i] = b[i];
	for (i=0;i<n;i++) {
		ip=indx[i];
		sum=x[ip];
		x[ip] = x[i];
		if (ii != 0)
			for (j=ii-1;j<i;j++) sum -= lu[i][j] * x[j];
		else if (sum != 0.0)
			ii=i+1;
		x[i]=sum;
	}
	for (i=n-1;i>=0;i--) {
		sum=x[i];
		for (j=i+1;j<n;j++) sum -= lu[i][j] * x[j];
		x[i] = sum / lu[i][i];
	}
    
    return true;
}

bool LUdcmp::solve(Matrix <double> &b,Matrix <double> &x)
{
	UINT m=b.getNumCols();
	if (b.getNumRows() != n || x.getNumRows() != n || b.getNumCols() != x.getNumCols() ){
        errorLog << "solve(Matrix <double> &b,Matrix <double> &x) - the size of the two matrices does not match!" << endl;
		return false;
    }
	vector <double>  xx(n);
	for (UINT j=0; j<m; j++) {
		for(UINT i=0; i<n; i++) xx[i] = b[i][j];
		solve_vector(xx,xx);
		for(UINT i=0; i<n; i++) x[i][j] = xx[i];
	}
    return true;
}
    
bool LUdcmp::inverse(Matrix <double> &ainv)
{
	int i,j;
	ainv.resize(n,n);
	for (i=0;i<n;i++) {
		for (j=0;j<n;j++) ainv[i][j] = 0.0;
		ainv[i][i] = 1.0;
	}
	return solve(ainv,ainv);
}
    
double LUdcmp::det()
{
	double dd = d;
	for (int i=0;i<n;i++) dd *= lu[i][i];
	return dd;
}
    
bool LUdcmp::mprove(vector <double> &b,vector <double> &x)
{
	int i,j;
	vector <double> r(n);
	for (i=0;i<n;i++) {
		long double sdp = -b[i];
		for (j=0;j<n;j++)
			sdp += (long double) aref[i][j] * (long double)x[j];
		r[i]=sdp;
	}
	if( !solve_vector(r,r) ){
        return false;
    }
	for (i=0;i<n;i++) x[i] -= r[i];
    return true;
}
    
}//End of namespace GRT
