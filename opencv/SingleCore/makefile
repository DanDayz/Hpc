all:encoder

encoder : encoder.cpp decoder.cpp
	g++ `pkg-config --cflags --libs opencv` -o encoder encoder.cpp
	g++ `pkg-config --cflags --libs opencv` -o decoder decoder.cpp

clean:
	rm -rf encoder encoder.o decoder decoder *~
