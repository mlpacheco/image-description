%module images

%include "std_string.i"
%include "std_vector.i"

namespace std {
    %template(PathSet) vector<string>;
}

%{
  #include "images.hpp"
%}

double similarityScore(std::string image1Path,
                       std::string image2Path,
                       std::string trainedFile);

int trainSift(std::vector<std::string> imagePaths,
              int numWords,
              std::string outfile);
