DEBUG=0

ifneq ($(DEBUG), 0)
CFLAGS=-O0 -g
else
CFLAGS=-O3 -lineinfo
endif

CFLAGS+=`pkg-config opencv --cflags --libs`

FILES=ex1 image

all: $(FILES)

ex1: ex1.o main.o ex1-cpu.o
	nvcc --link $(CFLAGS) $^ -o $@

image: ex1-cpu.o image.o
	nvcc --link $(CFLAGS) $^ -o $@

ex1.o: ex1.cu ex1.h
main.o: main.cu ex1.h
image.o: image.cu ex1.h

%.o: %.cu
	nvcc --compile $< $(CFLAGS) -o $@

clean::
	rm *.o $(FILES)
