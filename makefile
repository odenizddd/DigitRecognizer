INCLUDES = -I/opt/homebrew/Cellar/opencv/4.8.1_4/include/opencv4 \
		   -I/Users/ozgurdenizdemir/Desktop/mnist-cpp/eigen-3.4.0

default: build run

build: main.cpp
	g++ -std=c++11 $(INCLUDES) main.cpp -L/opt/homebrew/Cellar/opencv/4.8.1_4/lib -lopencv_core -lopencv_highgui -o main

run: main
	./main