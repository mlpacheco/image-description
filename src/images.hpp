#include <stdlib.h>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>

#include "math.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"

using namespace std;
using namespace cv;

// Auxiliary functions
int Mat2vector(Mat mat, vector<vector<float> > &vect);
string type2str(int type);
int printContents(Mat BOWmat);
int printVector(vector<vector<float> > vect);
string joinPath(string path, string fileName);

// Feature Functions
int trainSift(vector<string> imagePaths, int numWords, string trainedPath, string outFile);
int trainCielab(vector<string> imagePaths, int numWords, string trainedPath, string outFile);
int extractSiftBOW(string trainedPath, vector<string> imagePaths, Mat &histograms, string outFile);
int extractCielabBOW(string trainedPath, vector<string> imagePaths, Mat &histograms, string outFile);
int encodeKmeansBOW(Mat pixels, Mat centers, vector<float> &BOWfeatures);

// Extraction Interfaces
int extractFeats(string trainedPath, vector<string> imagePaths,
                 vector<vector<float> > &SiftFeats, vector<vector<float> > &CielabFeats,
                 string outFile);
double intersectionScore(vector<float> histogram1, vector<float> histogram2);
