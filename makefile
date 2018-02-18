# Compier to use
CC=nvcc
# CFLAGS
CFLAGS= -G -g -O3 -gencode arch=compute_32,code=sm_32

LDFLAGS= --cudart static -pg

LDLIB= -L/usr/lib/arm-linux-gnueabihf -L/usr/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_objdetect -lcublas

SOURCES=src/sr.cu

COBJECTS=$(SOURCES:.cu=.o)
DOBJECTS=$(SOURCES:.cu=.d)
EXECUTABLE=program

all: $(DOBJECTS) $(COBJECTS)
	$(CC) $(LDFLAGS) $(LDLIB) $(COBJECTS) $(DOBJETS) -o $(EXECUTABLE)
	

%.o: %.cu
	$(CC) --compile $(CFLAGS) -x cu -o $@ $<

%.d: %.cu
	$(CC) $(CFLAGS) -M -odir ""  -o $@ $<

clean:
	rm -f $(COBJECTS) $(DOBJECTS) $(EXECUTABLE)
