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

//Include the Utilities
#include "Util/GRTVersionInfo.h"
#include "Util/GRTCommon.h"
#include "Util/RangeTracker.h"
#include "Util/TrainingDataRecordingTimer.h"

//Include the data structures
#include "DataStructures/Matrix.h"
#include "DataStructures/LabelledClassificationData.h"
#include "DataStructures/LabelledTimeSeriesClassificationData.h"
#include "DataStructures/LabelledContinuousTimeSeriesClassificationData.h"
#include "DataStructures/LabelledRegressionData.h"
#include "DataStructures/UnlabelledClassificationData.h"

//Include the PreProcessing Modules
#include "PreProcessingModules/Derivative.h"
#include "PreProcessingModules/LowPassFilter.h"
#include "PreProcessingModules/HighPassFilter.h"
#include "PreProcessingModules/MovingAverageFilter.h"
#include "PreProcessingModules/DoubleMovingAverageFilter.h"
#include "PreProcessingModules/SavitzkyGolayFilter.h"
#include "PreProcessingModules/DeadZone.h"

//Include the FeatureExtraction Modules
#include "FeatureExtractionModules/PeakDetection.h"
#include "FeatureExtractionModules/ZeroCrossingCounter.h"
#include "FeatureExtractionModules/FFT.h"
#include "FeatureExtractionModules/MovementTrajectoryFeatures.h"
#include "FeatureExtractionModules/MovementIndex.h"

//Include the PostProcessing Modules
#include "PostProcessingModules/ClassLabelFilter.h"
#include "PostProcessingModules/ClassLabelTimeoutFilter.h"
#include "PostProcessingModules/ClassLabelChangeFilter.h"

//Include Classification Modules
#include "ClassificationModules/ANBC/ANBC.h"
#include "ClassificationModules/KNN/KNN.h"
#include "ClassificationModules/DTW/DTW.h"
#include "ClassificationModules/SVM/SVM.h"
#include "ClassificationModules/GMM/GMM.h"
#include "ClassificationModules/LDA/LDA.h"

//Include the Regression Modules
#include "RegressionModules/ArtificialNeuralNetworks/MLP/MLP.h"

//Include the Clustering algorithms
#include "ClusteringModules/KMeans/KMeans.h"
#include "ClusteringModules/GaussianMixtureModels/GaussianMixtureModels.h"

//Include the Context Modules
#include "ContextModules/Gate.h"

//Include the Recognition Pipeline
#include "GestureRecognitionPipeline/GestureRecognitionPipeline.h"
