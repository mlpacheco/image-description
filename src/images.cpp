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

int trainSift(vector<String> imagePaths, int numWords, string outfile) {
    Mat input;
    vector<KeyPoint> keypoints;
    Mat descriptor;
    Mat featuresUnclustered;
    SiftDescriptorExtractor detector;

    for (int i = 0; i < imagePaths.size(); i++) {
        cout << imagePaths[i] << endl;
        input = imread(imagePaths[i], CV_LOAD_IMAGE_GRAYSCALE);
        detector.detect(input, keypoints);
        detector.compute(input, keypoints, descriptor);
        featuresUnclustered.push_back(descriptor);

    }
    // parameters for the algorithm
    // 100 iters
    // epsilon 0.001 ?
    TermCriteria tc(CV_TERMCRIT_ITER, 100, 0.001);
    int retries = 1;
    int flags = KMEANS_PP_CENTERS;
    // Train K-Means on numWords
    BOWKMeansTrainer bowTrainer(numWords, tc, retries, flags);
    cout << featuresUnclustered.dims << endl;
    Mat dictionary = bowTrainer.cluster(featuresUnclustered);
    // Write results;
    FileStorage fs(outfile, FileStorage::WRITE);
    fs << "vocabulary" << dictionary;
    fs.release();

    return 1;
}

int extractSift(string trainedFile, string imagePath, Mat &bowDescriptor) {

    Mat dictionary;
    FileStorage fs(trainedFile, FileStorage::READ);
    fs["vocabulary"] >> dictionary;
    fs.release();

    Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
    Ptr<FeatureDetector> detector(new SiftFeatureDetector());
    Ptr<DescriptorExtractor>  extractor(new SiftDescriptorExtractor);
    BOWImgDescriptorExtractor bowDE(extractor, matcher);
    bowDE.setVocabulary(dictionary);

    Mat img = imread(imagePath, CV_LOAD_IMAGE_GRAYSCALE);
    vector<KeyPoint> keypoints;
    detector->detect(img, keypoints);

    bowDE.compute(img, keypoints, bowDescriptor);

    return 1;

}


// right now it only measures SIFT but it needs other features next
double similarityScore(string image1Path, string image2Path, string trainedFile) {
    Mat histogram1;
    Mat histogram2;
    extractSift(trainedFile, image1Path, histogram1);
    extractSift(trainedFile, image2Path, histogram2);
    return compareHist(histogram1, histogram2, CV_COMP_INTERSECT);
}


int main(int argc, char *argv[]) {

    string path = "/Users/marialeonor/Purdue/MachineLearning/Repositories/image-description/data/AbstractScenes/RenderedScenes/";
    string filename;
    vector<string> trainImagePaths;

    for(int i = 0; i < 10; i++) {
        stringstream ss;
        ss << i;
        filename = path + "Scene0_" + ss.str() + ".png";
        trainImagePaths.push_back(filename);
    }

    cout << "imagePaths " << trainImagePaths.size() << endl;

    string trainedFile =  "/Users/marialeonor/Purdue/MachineLearning/Repositories/image-description/out/SIFTwords.yml";
    trainSift(trainImagePaths, 2, trainedFile);

    string testImage = path + "Scene0_0.png";
    for(int i = 0; i < trainImagePaths.size(); i++) {
        cout << "Similarity score: " << \
                similarityScore(testImage, trainImagePaths[i], trainedFile) \
                << endl;
    }

}
