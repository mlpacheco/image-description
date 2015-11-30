%module images

%include "std_string.i"
%include "std_vector.i"

namespace std {
    %template(PathSet)  vector<string>;
    %template(Features) vector<float>;
    %template(FeaturesMatrix) vector<vector<float> >;
    %template(BadIndexes) vector<int>;
}

%{
  #include "images.hpp"
%}

int trainSift(std::vector<std::string> imagePaths,
              int numWords,
              std::string trainedPath, std::string outFile);

int extractFeats(std::string trainedPath,
                 std::vector<std::string> imagePaths,
                 std::vector<std::vector<float> > &extractedFeats,
                 std::vector<int> &problematicImages, std::string outFile);

double intersectionScore(std::vector<float> histogram1, std::vector<float> histogram2);
