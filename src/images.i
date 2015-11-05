%module images

%include "std_string.i"
%include "std_vector.i"

namespace std {
    %template(PathSet)  vector<string>;
    %template(Features) vector<float>;
    %template(FeaturesMatrix) vector<vector<float> >;
}

%{
  #include "images.hpp"
%}

int trainSift(std::vector<std::string> imagePaths,
              int numWords,
              std::string outfile);

int extractFeats(std::string trainedFile,
                 std::vector<std::string> imagePaths,
                 std::vector<std::vector<float> > &extractedFeats);

double similarityScore(std::string image1Path,
                       std::string image2Path,
                       std::string trainedFile);


