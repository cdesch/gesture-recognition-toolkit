/**
 @file
 @author  Nicholas Gillian <ngillian@media.mit.edu>
 @version 1.0
 
 @section LICENSE
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
 
 @section DESCRIPTION
 The Matrix class is a basic class for storing any type of data.  This class is a template and can therefore be used with any generic data type.
 */

#pragma once

#include "../Util/GRTCommon.h"

namespace GRT{

template <class T>
class Matrix{
public:
    /**
     Default Constructor
    */
	Matrix(){
	 rows = 0;
	 cols = 0;
     dataPtr = NULL;
	}
    
    /**
    Constructor, sets the size of the matrix to [rows cols]
     
     @param UINT rows: sets the number of rows in the matrix, must be a value greater than zero
     @param UINT cols: sets the number of columns in the matrix, must be a value greater than zero
    */
	Matrix(UINT rows,UINT cols){
      dataPtr = NULL;
      resize(rows,cols);
	}
    
    /**
     Copy Constructor, copies the values from the rhs Matrix to this Matrix instance
     
     @param const Matrix &rhs: the Matrix from which the values will be copied
    */
	Matrix(const Matrix &rhs){
		if(this!=&rhs){
			 this->dataPtr = NULL;
			 this->rows = rhs.rows;
			 this->cols = rhs.cols;
			 dataPtr = new T*[rows];
			 for(UINT i=0; i<rows; i++){
				 dataPtr[i] = new T[cols];
				 for(UINT j=0; j<cols; j++) 
					 this->dataPtr[i][j] = rhs.dataPtr[i][j];
			 }
		}
	}
    
    /**
     Destructor, cleans up any memory
    */
	~Matrix(){ 
		clear(); 
	}
    
    /**
     Defines how the data from the rhs Matrix should be copied to this Matrix
     
     @param const Matrix &rhs: another instance of a Matrix
     @return returns a pointer to this instance of the Matrix
    */
	Matrix& operator=(const Matrix &rhs){
		if(this!=&rhs){
			 this->clear();
			 this->rows = rhs.rows;
			 this->cols = rhs.cols;
			 dataPtr = new T*[rows];
			 for(UINT i=0; i<rows; i++){
				 dataPtr[i] = new T[cols];
				 for(UINT j=0; j<cols; j++) 
					 this->dataPtr[i][j] = rhs.dataPtr[i][j];
			 }
		}
		return *this;
	}
    
    /**
     Returns a pointer to the data at row r
     
     @param const UINT r: the index of the row you want, should be in the range [0 rows-1]
     @return a pointer to the data at row r
    */
	inline T* operator[](const UINT r){
     return dataPtr[r];
	}

    /**
     Gets a row vector [1 cols] from the Matrix at the row index r
     
     @param const UINT r: the index of the row, this should be in the range [0 rows-1]
     @return returns a row vector from the Matrix at the row index r
    */
	vector<T> getRowVector(const UINT r){
		vector<T> rowVector(cols);
		for(UINT c=0; c<cols; c++)
			rowVector[c] = dataPtr[r][c];
		return rowVector;
	}

    /**
     Gets a column vector [rows 1] from the Matrix at the column index c
     
     @param const UINT c: the index of the column, this should be in the range [0 cols-1]
     @return returns a column vector from the Matrix at the column index c
    */
	vector<T> getColVector(const UINT c){
		vector<T> columnVector(rows);
		for(UINT r=0; r<rows; r++)
			columnVector[r] = dataPtr[r][c];
		return columnVector;
	}
    
    /**
     Concatenates the entire matrix into a single vector and returns the vector.
     The data can either be concatenated by row or by column, by setting the respective concatByRow parameter to true of false.
     If concatByRow is true then the data in the matrix will be added to the vector row-vector by row-vector, otherwise
     the data will be added column-vector by column-vector.
     
     @param bool concatByRow: sets if the matrix data will be added to the vector row-vector by row-vector
     @return returns a vector containing the entire matrix data
     */
    vector<T> getConcatenatedVector(bool concatByRow = true){
        
        if( rows == 0 || cols == 0 ) return vector<T>();
        
        vector<T> vectorData(rows*cols);
        
        if( concatByRow ){
            for(UINT i=0; i<rows; i++){
                for(UINT j=0; j<cols; j++){
                    vectorData[ (i*cols)+j ] = dataPtr[i][j];
                }
            }
        }else{
            for(UINT j=0; j<cols; j++){
                for(UINT i=0; i<rows; i++){
                    vectorData[ (i*cols)+j ] = dataPtr[i][j];
                }
            }
        }
        
        return vectorData;
    }

    /**
     Resizes the Matrix to the new size of [r c]
     
     @param UINT r: the number of rows, must be greater than zero
     @param UINT c: the number of columns, must be greater than zero
     @return returns true or false, indicating if the resize was successful 
    */
	bool resize(UINT r,UINT c){
        //Clear any previous memory
        clear();
        if( r > 0 && c > 0 ){
            rows = r;
            cols = c;
            dataPtr = new T*[rows];
            
            //Check to see if the memory was created correctly
            if( dataPtr == NULL ){
                rows = 0;
                cols = 0;
                return false;
            }
            for(UINT i=0; i<rows; i++){
                dataPtr[i] = new T[cols];
            }
            return true;
        }
        return false;
	}

    /**
     Sets all the values in the Matrix to the input value
     
     @param T value: the value you want to set all the Matrix values to
     @return returns true or false, indicating if the set was successful 
    */
	bool setAllValues(T value){
		if(dataPtr!=NULL){
			for(UINT i=0; i<rows; i++)
				for(UINT j=0; j<cols; j++)
					dataPtr[i][j] = value;
            return true;
		}
        return false;
	}

    /**
     Adds the input sample to the end of the Matrix, extending the number of rows by 1.  The number of columns in the sample must match
     the number of columns in the Matrix, unless the Matrix size has not been set, in which case the new sample size will define the
     number of columns in the Matrix.
     
     @param vector< T > sample: the new column vector you want to add to the end of the Matrix.  Its size should match the number of columns in the Matrix
     @return returns true or false, indicating if the push was successful 
    */
	bool push_back(vector<T> sample){
		//If there is no data, but we know how many cols are in a sample then we simply create a new buffer of size 1 and add the sample
		if(dataPtr==NULL){
			cols = (UINT)sample.size();
			if( !resize(1,cols) ){
                clear();
                return false;
            }
			for(UINT j=0; j<cols; j++)
				dataPtr[0][j] = sample[j];
			return true;
		}

		//If there is data and the sample size does not match the number of columns then return false
		if(sample.size() != cols ){
			return false;
		}

		//Otherwise we copy the existing data from the data ptr into a new buffer of size (rows+1) and add the sample at the end
		T** tempDataPtr = NULL;
		tempDataPtr = new T*[ rows+1 ];
		if( tempDataPtr == NULL ){//If NULL then we have run out of memory
			return false;
		}
		for(UINT i=0; i<rows+1; i++){
			tempDataPtr[i] = new T[cols];
		}

		//Copy the original data
		for(UINT i=0; i<rows; i++)
			for(UINT j=0; j<cols; j++)
				tempDataPtr[i][j] = dataPtr[i][j];

		//Add the new sample at the end
		for(UINT j=0; j<cols; j++)
			tempDataPtr[rows][j] = sample[j];

		//Delete the original data and copy the pointer
		for(UINT i=0; i<rows; i++){
			delete[] dataPtr[i];
			dataPtr[i] = NULL;
		}
		delete[] dataPtr;
		dataPtr = tempDataPtr;
        
        //Increment the number of rows
		rows++;

		//Finally return true to signal that the data was added correctly
		return true;
	}

    /**
     Cleans up any dynamic memory and sets the number of rows and columns in the matrix to zero
    */
	void clear(){
		if(dataPtr!=NULL){
			for(UINT i=0; i<rows; i++){
                delete[] dataPtr[i];
                dataPtr[i] = NULL;
			}
			delete[] dataPtr;
			dataPtr = NULL;
		}
		rows = 0;
		cols = 0;
	}

    /**
     Gets the number of rows in the Matrix
     
     @return returns the number of rows in the Matrix
    */
	inline UINT getNumRows(){return rows;}
    
    /**
     Gets the number of columns in the Matrix
     
     @return returns the number of columns in the Matrix
    */
	inline UINT getNumCols(){return cols;}

    /**
     Saves the Matrix data to a file. This function assumes that the template T can be easily saved and loaded 
     to and from a tab seperated variable in a txt file, if not the you need to overload this function
     
     @return returns true or false, indicating if the data was saved successfully
    */
	bool saveMatrixDataToFile( string filename ){
		
		std::fstream file; 
		file.open(filename.c_str(), std::ios::out);
		
		if(!file.is_open())
		{
			cout<<"GRT_MATRIX_LOAD_ERROR: Could not open file to load data\n";
			return false;
		}
		
		std::string word;

		//Write the file header
		file << "GRT_MATRIX_DATA_FILE_V1.0\n";
		file << "Rows: " << rows << endl;
		file << "Cols: " << cols << endl;
		
		for(UINT i=0; i<rows; i++){
			for(UINT j=0; j<cols; j++){
				file << dataPtr[i][j] << "\t";
			}file << endl;
		}

		file.close();
		
		return true;
   }

    /**
     Loads the Matrix data to from file. This function assumes that the template T can be easily saved and loaded 
     to and from a tab seperated variable in a txt file, if not the you need to overload this function
     
     @return returns true or false, indicating if the data was loaded successfully
    */
	bool loadMatrixDataFromFile( string filename ){
		
		std::fstream file; 
		file.open(filename.c_str(), std::ios::in);
        
        //Delete any previous data
		clear();
		
		if(!file.is_open())
		{
			cout<<"GRT_MATRIX_LOAD_ERROR: Could not open file to load data\n";
			return false;
		}
		
		std::string word;
		
		//Check to make sure this is a file with the correct format
		file >> word;
		if(word != "GRT_MATRIX_DATA_FILE_V1.0"){
			cout<<"MATRIX_LOAD_ERROR: Incorrect file format, unknown header in data file\n";
			return false;
		}
		
		//Check and load the Number of Rows
		file >> word;
		if(word != "Rows:"){
			cout<<"MATRIX_LOAD_ERROR: Incorrect file format, can't find Number of Rows\n";
			return false;
		}
		file >> rows;
		
		//Check and load the number of columns
		file >> word;
		if(word != "Cols:"){
			cout<<"MATRIX_LOAD_ERROR: Incorrect file format, can't find Number of Columns\n";
			return false;
		}
		file >> cols;
		
		//Resize the data buffer
		resize(rows,cols);

		//Now we should have the data with each row containing a sample of the training data followed 
		//by a sample of the target data, tab seperated.
		for(UINT i=0; i<rows; i++){
			for(UINT j=0; j<cols; j++){
				T value;
				file >> value;
				dataPtr[i][j] = value;
			}
		}

		file.close();
		
		return true;
   }

private:
    
	UINT rows;      ///< The number of rows in the Matrix
	UINT cols;      ///< The number of columns in the Matrix
	T **dataPtr;    ///< A pointer to the data

};

}//End of namespace GRT

