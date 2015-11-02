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

int trainSift(vector<String> imagePaths, int numWords, string outfile);

int extractSift(string trainedFile, string imagePath, Mat &bowDescriptor);

double similarityScore(string image1Path, string image2Path, string trainedFile);

