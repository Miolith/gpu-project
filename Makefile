CC=nvcc
CPPFLAGS=-lm
COMPFLAGS=-std=c++11
SRC=src/*.cpp src/*.cu
EXEC_NAME=main
SRC_TEST=test/*.cu\
		 src/*.cpp\
		 src/filters.cu\
		 src/GPUfilters.cu

SRC_BENCH=bench/*.cpp\
		  test/testTools.cpp\
		  src/*.cpp\
		  src/filters.cu\
		  src/GPUfilters.cu

.PHONY: test clean all testdebug $(EXEC_NAME) bench

# Check System for OSX Comptability Support
OS=$(shell uname)

all: CPPFLAGS += -O2
all: $(EXEC_NAME)

$(EXEC_NAME):
ifeq ($(OS),Darwin) # OSX
	$(CC) $(SRC) $(COMPFLAGS) $(CPPFLAGS) -o $@
else # Other
	$(CC) $(SRC) $(CPPFLAGS) -o $@
endif

run:
	./$(EXEC_NAME) images/*

debug: CPPFLAGS += -g -G
debug: $(EXEC_NAME)

test: CPPFLAGS += -O2
test: test_compile
	./test_suite

test_compile:
ifeq ($(OS),Darwin) # OSX
	$(CC) $(SRC_TEST) $(COMPFLAGS) $(CPPFLAGS) -o test_suite
else # Other
	$(CC) $(SRC_TEST) $(CPPFLAGS) -o test_suite
endif

testdebug: CPPFLAGS += -g -G
testdebug: test_compile

bench: CPPFLAGS += -O2 -isystem benchmark/include \
	-Lbenchmark/build/src -lbenchmark -lpthread
bench:
ifeq ($(OS),Darwin) # OSX
	$(CC) $(SRC_BENCH) $(COMPFLAGS) $(CPPFLAGS) -o benchmark
else # Other
	$(CC) $(SRC_BENCH) $(CPPFLAGS) -o benchmark
endif
	./benchmark

clean:
	rm $(EXEC_NAME)

