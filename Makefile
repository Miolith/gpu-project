CC=nvcc
CPPFLAGS=-L/usr/X11R6/lib -lm -lpthread -lX11
COMPFLAGS=-std=c++11
SRC=src/*.cpp src/*.cu
EXEC_NAME=main
SRC_TEST=test/*.cu\
		 src/*.cpp\
		 src/filters.cu\
		 src/GPUfilters.cu

.PHONY: test clean all testdebug

# Check System for OSX Comptability Support
OS=$(shell uname)

all: CPPFLAGS += -O2
all: $(EXEC_NAME)

$(EXEC_NAME):
ifeq ($(OS),Darwin) # OSX
	g++ $(SRC) $(COMPFLAGS) $(CPPFLAGS) -o $@
else # Other
	$(CC) $(SRC) $(CPPFLAGS) -o $@
endif

run:
	./$(EXEC_NAME) images/*

debug: CPPFLAGS += -g #-fsanitize=address
debug: $(EXEC_NAME)

test: CPPFLAGS += -O2
test: testlol
	./test_suite

testlol:
ifeq ($(OS),Darwin) # OSX
	g++ $(SRC_TEST) $(COMPFLAGS) $(CPPFLAGS) -o test_suite
else # Other
	$(CC) $(SRC_TEST) $(CPPFLAGS) -o test_suite
endif

testdebug: CPPFLAGS += -g -G
testdebug: testlol

clean:
	rm $(EXEC_NAME)

