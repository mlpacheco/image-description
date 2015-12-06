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
    //cout << "Rows: " << BOWmat.rows << " Cols: " << BOWmat.cols << endl;
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
    Size size(100,100);
    Mat dst;

    string outfile = joinPath(trainedPath, outFile + "_sift.yml");
    double total = imagePaths.size();
    cout << "Training SIFT bow model with " << total << " images." << endl;
    for (int i = 0; i < total; i++) {
        input = imread(imagePaths[i], CV_LOAD_IMAGE_GRAYSCALE);
        resize(input, dst, size);
        detector.detect(dst, keypoints);
        detector.compute(dst, keypoints, descriptor);
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
    cout << "Done" << endl;

    return 1;
}

int trainCielab(vector<String> imagePaths, int numWords, string trainedPath, string outFile) {
    Mat imgRgb;
    Mat imgLab;
    Mat featuresUnclustered;
    Size size(100,100);
    Mat resizedImg;
    Mat typeImg;
    string outfile = joinPath(trainedPath, outFile + "_cielab.yml");
    double total = imagePaths.size();
    cout << "Training CIELAB bow model with " << total << " images" << endl;
    for (int i = 0; i < total; i++) {
        // read RGB image
        imgRgb = imread(imagePaths[i]);
        // resize image
        resize(imgRgb, imgRgb, size);
        // transform to CIELAB
        cvtColor(imgRgb, imgLab, CV_BGR2Lab);
        // transform to float for kmeans
        imgLab.convertTo(typeImg, CV_32F);
        // make it a list of pixels
        typeImg = typeImg.reshape(3, typeImg.rows * typeImg.cols);
        // push features
        featuresUnclustered.push_back(typeImg);
        cout << i*100/total << "%";
        cout.flush();
        cout << "\r";
    }

    // train k-means
    Mat centers;
    Mat labels;
    TermCriteria tc(CV_TERMCRIT_ITER, 100, 0.001);
    kmeans(featuresUnclustered, numWords, labels, tc, 1, KMEANS_PP_CENTERS, centers);

    FileStorage fs(outfile, FileStorage::WRITE);
    fs << "centers" << centers;
    fs.release();
    cout << "Done" << endl;
    //printContents(centers);

    return 1;

}


int extractSiftBOW(string trainedPath, vector<string> imagePaths, Mat &histograms, string outFile) {
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

    }

    // printing out the contents of the Mat and its properties
    //
    //string ty =  type2str( histograms.type() );
    //printf("Matrix: %s %dx%d \n", ty.c_str(), histograms.cols, histograms.rows );
    //printContents(histograms);

    return 1;

}

int encodeKmeansBOW(Mat pixels, Mat centers, vector<float> &BOWfeatures) {
    vector<float> counts (centers.rows,0);
    Mat pixel;
    for (int i = 0; i < pixels.rows; i++) {
        double best_distance = 0.0;
        int best_center = -1;
        pixel = pixels.row(i).reshape(1);
        for (int j = 0; j < centers.rows; j++) {
            double dist = norm(pixel, centers.row(j));
            if (dist > best_distance) {
                best_distance = dist;
                best_center = j;
            }
        }
        counts[best_center] += 1;
    }

    /*for (int i = 0; i < counts.size(); i++) {
        cout << counts[i] << " ";
    }
    cout << endl;*/

    for (int i = 0; i < counts.size(); i++) {
        BOWfeatures.push_back(counts[i]/pixels.rows);
    }
    return 1;
}

int extractCielabBOW(string trainedPath, vector<string> imagePaths, vector<vector<float> > &histograms, string outFile) {
    Mat centers;

    string trainedFile = joinPath(trainedPath, outFile + "_cielab.yml");
    FileStorage fs(trainedFile, FileStorage::READ);
    fs["centers"] >> centers;
    fs.release();

    Mat imgRgb;
    Mat imgLab;
    Size size(100,100);
    Mat typeImg;

    double total = imagePaths.size();
    for (int i = 0; i < total; i++) {
        imgRgb = imread(imagePaths[i]);
        // resizing image
        resize(imgRgb, imgRgb, size);
        // translating to cielab coordinates
        cvtColor(imgRgb, imgLab, CV_BGR2Lab);
        // changing type for kmeans
        imgLab.convertTo(typeImg, CV_32F);
        // reshaping to have a list of pixels
        typeImg = typeImg.reshape(3, typeImg.rows * typeImg.cols);
        // extracting features for image
        vector<float> features;
        encodeKmeansBOW(typeImg, centers, features);
        histograms.push_back(features);
        cout << i*100/total << "%";
        cout.flush();
        cout << '\r';

    }

    return 1;

}

/* Extracted Feats in vector for SWIG */

int extractFeats(string trainedPath, vector<string> imagePaths,
                 vector<vector<float> > &SiftFeats, vector<vector<float> > &CielabFeats,
                 string outFile) {
    Mat SIFTfeatMat;
    cout << "Extracting SIFT words..." << endl;
    extractSiftBOW(trainedPath, imagePaths, SIFTfeatMat, outFile);
    cout << "Done." << endl;
    Mat2vector(SIFTfeatMat, SiftFeats);
    cout << "Extracting CIELAB words..." << endl;
    extractCielabBOW(trainedPath, imagePaths, CielabFeats, outFile);
    cout << "Done" << endl;
    return 1;
}

double intersectionScore(vector<float> histogram1, vector<float> histogram2) {
    return compareHist(histogram1, histogram2, CV_COMP_INTERSECT);
}


int main(int argc, char *argv[]) {
    string path = "/Users/marialeonor/Developer/image-description/data/AbstractScenes_v1.1/RenderedScenes/";
    string filename;
    vector<string> trainImagePaths;

    for (int i = 0; i < 10; i++) {
        stringstream ss;
        ss << i;
        filename = path + "Scene0_" + ss.str() + ".png";
        trainImagePaths.push_back(filename);
    }

    trainCielab(trainImagePaths, 5, ".", "TEST");
    cout << "Trained!" << endl;
    vector<vector<float> > CielabHistograms;
    extractCielabBOW(".", trainImagePaths, CielabHistograms, "TEST");
    printVector(CielabHistograms);
    //Mat histograms;
    //trainSift(trainImagePaths, 5, ".", "TEST");
    //extractSiftBOW(".", trainImagePaths, histograms, problematicImages, "TEST");
    //printContents(histograms);



    return 1;

}
