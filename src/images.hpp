#include <stdlib.h>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>

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

// SIFT Functions
int trainSift(vector<String> imagePaths, int numWords, string outfile);
int extractSiftBOW(string trainedFile, vector<string> imagePaths, Mat &histograms);

// Extraction and Similarity Interfaces
int extractFeats(string trainedFile, vector<string> imagePaths, vector<vector<float> > &extractedFeats);
double similarityScore(string image1Path, string image2Path, string trainedFile);

