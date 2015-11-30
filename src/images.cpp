#include "images.hpp"

/* Auxiliary Functions  */

string SIFT_FILENAME = "SIFTWords.yml";

int Mat2vector(Mat mat, vector<vector <float> > &vect) {
    /*cout << "################## Copying mat to vector " << endl;
    cout << "MAT ROWS: " << mat.rows << endl;;
    cout << "MAT COLS:  " << mat.cols << endl;*/
    for (int i = 0; i < mat.rows; i++) {
        vector<float> row;
        for (int j = 0; j < mat.cols; j++) {
            row.push_back(mat.at<float>(i,j));
        }
        vect.push_back(row);
    }
    return 1;
}


string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

int printContents(Mat BOWmat) {
    for (int i = 0; i < BOWmat.rows; i++) {
        for (int j = 0; j < BOWmat.cols; j++) {
            cout << BOWmat.at<float>(i,j) << " ";
        }
        cout << endl;
    }

    return 1;
}

int printVector(vector<vector<float> > vect) {
    for (int i = 0; i < vect.size(); i++) {
        for (int j = 0; j < vect[i].size(); j++) {
            cout << vect[i][j] << " ";
        }
        cout << endl;
    }
    return 1;
}

string joinPath(string path, string fileName) {
    string filePath;
    if (path[path.length()-1] == '/') {
        filePath = path + fileName;
    } else {
        filePath = path + "/" + fileName;
    }
    return filePath;

}


/* SIFT Functions */


int trainSift(vector<String> imagePaths, int numWords, string trainedPath, string outFile) {
    Mat input;
    vector<KeyPoint> keypoints;
    Mat descriptor; 
    Mat featuresUnclustered;
    SiftDescriptorExtractor detector;

    string outfile = joinPath(trainedPath, outFile + "_sift.yml");
    double total = imagePaths.size();
    cout << "training " << total << " images." << endl;
    for (int i = 0; i < total; i++) {
        input = imread(imagePaths[i], CV_LOAD_IMAGE_GRAYSCALE);
        detector.detect(input, keypoints);
        detector.compute(input, keypoints, descriptor);
        featuresUnclustered.push_back(descriptor);
        cout << i*100/total << "%";
        cout.flush();
        cout << "\r";
    }
    // parameters for the algorithm
    // 100 iters
    // epsilon 0.001 ?
    TermCriteria tc(CV_TERMCRIT_ITER, 100, 0.001);
    int retries = 1;
    int flags = KMEANS_PP_CENTERS;
    // Train K-Means on numWords
    BOWKMeansTrainer bowTrainer(numWords, tc, retries, flags);
    Mat dictionary = bowTrainer.cluster(featuresUnclustered);
    // Write results;
    FileStorage fs(outfile, FileStorage::WRITE);
    fs << "vocabulary" << dictionary;
    fs.release();

    return 1;
}


int extractSiftBOW(string trainedPath, vector<string> imagePaths, Mat &histograms, vector<int> &problematicImages, string outFile) {
    int a = 1;
    Mat dictionary;

    string trainedFile = joinPath(trainedPath, outFile + "_sift.yml");

    FileStorage fs(trainedFile, FileStorage::READ);
    fs["vocabulary"] >> dictionary;
    fs.release();

    Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
    Ptr<FeatureDetector> detector(new SiftFeatureDetector());
    Ptr<DescriptorExtractor>  extractor(new SiftDescriptorExtractor);
    BOWImgDescriptorExtractor bowDE(extractor, matcher);
    bowDE.setVocabulary(dictionary);
    Mat bowDescriptor;
    Mat img;
    vector<KeyPoint> keypoints;

    double total = imagePaths.size();
    //cout << "extracting " << total << " images." << endl;

    for (int i = 0; i < total; i++) {
        img = imread(imagePaths[i], CV_LOAD_IMAGE_GRAYSCALE);
        detector->detect(img, keypoints);
        bowDE.compute(img, keypoints, bowDescriptor);
        histograms.push_back(bowDescriptor);
        cout << i*100/total << "%";
        cout.flush();
        cout << '\r';

        if (i + a != histograms.rows) {
            a--;
            problematicImages.push_back(i);
        }
    }

    // printing out the contents of the Mat and its properties
    //
    //string ty =  type2str( histograms.type() );
    //printf("Matrix: %s %dx%d \n", ty.c_str(), histograms.cols, histograms.rows );
    //printContents(histograms);

    return 1;

}

/* Extracted Feats in vector for SWIG */

int extractFeats(string trainedPath, vector<string> imagePaths, vector<vector<float> > &extractedFeats, vector<int> &problematicImages, string outFile) {
    Mat SIFTfeatures;
    extractSiftBOW(trainedPath, imagePaths, SIFTfeatures, problematicImages, outFile);
    Mat2vector(SIFTfeatures, extractedFeats);
    return 1;
}

double intersectionScore(vector<float> histogram1, vector<float> histogram2) {
    return compareHist(histogram1, histogram2, CV_COMP_INTERSECT);
}


