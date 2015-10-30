.PHONY: all clean

CPP = g++

OBJECTS = images.o

LDFLAGS = -lopencv_calib3d -lopencv_contrib -lopencv_core -lopencv_features2d -lopencv_flann \
          -lopencv_gpu -lopencv_highgui -lopencv_imgproc -lopencv_legacy -lopencv_ml -lopencv_nonfree \
          -lopencv_objdetect -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_ts \
          -lopencv_video -lopencv_videostab

INCLUDES = -I'/opt/local/include/opencv' -I'/opt/local/include'
LIB_INCLUDES = -L'/opt/local/lib'

images.o: images.cpp
		$(CPP) $(INCLUDES) $(LIB_INCLUDES) $(LDFLAGS) $< -o $@

clean:
		rm -f *.o

