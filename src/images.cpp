#include <stdlib.h>
#include <vector>
#include <string>
#include <sstream>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {

    string path = "/Users/marialeonor/Purdue/MachineLearning/Repositories/image-description/data/AbstractScenes/RenderedScenes/";
    string filename = "";
    Mat input;
    vector<KeyPoint> keypoints;
    Mat descriptor;
    Mat featuresUnclustered;
    SiftDescriptorExtractor detector;


    for(int i = 0; i < 10; i++) {
        stringstream ss;
        ss << i;
        filename = path + "Scene0_" + ss.str();
        input = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
        detector.detect(input, keypoints);
        detector.compute(input, keypoints, descriptor);
        featuresUnclustered.push_back(descriptor);
    }

    // I havent looked at the part below yet

    //Construct BOWKMeansTrainer
    //the number of bags
    int dictionarySize=200;
    //define Term Criteria
    TermCriteria tc(CV_TERMCRIT_ITER,100,0.001);
    //retries number
    int retries=1;
    //necessary flags
    int flags=KMEANS_PP_CENTERS;
    //Create the BoW (or BoF) trainer
    BOWKMeansTrainer bowTrainer(dictionarySize,tc,retries,flags);
    //cluster the feature vectors
    Mat dictionary=bowTrainer.cluster(featuresUnclustered);
    //store the vocabulary
    FileStorage fs("dictionary.yml", FileStorage::WRITE);
    fs << "vocabulary" << dictionary;
    fs.release();

}
