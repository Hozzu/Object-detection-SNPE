LIB_INCS = -I include/ -I snpe-1.58/include/zdl/
LIB_PATH = -L lib/ -L snpe-1.58/lib/
LDFLAGS = -lSNPE -lais_client -lfastcvopt -ljson-c -ljpeg -lqcarcam_client -lopencv_core -lopencv_imgproc
CFLAGS = -O2 

all:
	$(CXX) $(CFLAGS) $(LIB_INCS) $(LIB_PATH) src/run_image.cpp src/run_camera.cpp src/main.cpp -o pkshin_detect $(LDFLAGS)
clean:
	rm -f pkshin_detect
