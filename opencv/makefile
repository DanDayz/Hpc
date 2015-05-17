all:encoder

encoder : encoder.cpp
	g++ `pkg-config --cflags --libs opencv` -o encoder encoder.cpp

clean:
	rm -rf encoder encoder.o *~	
