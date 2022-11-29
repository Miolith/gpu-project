CC=nvcc
CPPFLAGS=-lm
COMPFLAGS=-std=c++11
SRC=$(wildcard src/*.cpp src/*.cu)
EXEC_NAME=main
SRC_TST=$(wildcard test/*.cu test/*.cpp\
		 src/*.cpp\
		 src/*.cu)
SRC_TEST=$(subst src/main.cu,,${SRC_TST})
SRC_BNCH=$(wildcard bench/*.cpp\
		  test/*.cpp test/*.cu\
		  src/*.cpp\
		  src/*.cu)
SRC_BENCH_1=$(subst src/main.cu,,${SRC_BNCH})
SRC_BENCH=$(subst test/test.cu,,${SRC_BENCH_1})

OBJ_1 = $(SRC:.cpp=.o)
OBJ = $(OBJ_1:.cu=.o)
OBJ_TST_1 = $(SRC_TEST:.cpp=.o)
OBJ_TST = $(OBJ_TST_1:.cu=.o)
OBJ_BENCH_1 = $(SRC_BENCH:.cpp=.o)
OBJ_BENCH = $(OBJ_BENCH_1:.cu=.o)

.PHONY: test clean all testdebug $(EXEC_NAME) bench

# Check System for OSX Comptability Support
OS=$(shell uname)

all: CPPFLAGS += -O2
all: $(EXEC_NAME)

$(EXEC_NAME): $(OBJ)
ifeq ($(OS),Darwin) # OSX
	$(CC) $(OBJ) $(COMPFLAGS) $(CPPFLAGS) -o $@
else # Other
	$(CC) $(OBJ) $(CPPFLAGS) -o $@
endif

run:
	./$(EXEC_NAME) images/*

debug: CPPFLAGS += -g -G
debug: $(EXEC_NAME)

test: CPPFLAGS += -O2
test: test_compile
	./test_suite

test_compile: $(OBJ_TST)
ifeq ($(OS),Darwin) # OSX
	$(CC) $(OBJ_TST) $(COMPFLAGS) $(CPPFLAGS) -o test_suite
else # Other
	$(CC) $(OBJ_TST) $(CPPFLAGS) -o test_suite
endif

testdebug: CPPFLAGS += -g -G
testdebug: test_compile

bench: CPPFLAGS += -O2 \
	-Lbenchmark/build/src -lbenchmark -lpthread
bench: $(OBJ_BENCH)
ifeq ($(OS),Darwin) # OSX
	$(CC) $(OBJ_BENCH) $(COMPFLAGS) $(CPPFLAGS) -o benchmark
else # Other
	$(CC) $(OBJ_BENCH) $(CPPFLAGS) -o benchmark
endif
	./benchmark


%.o: %.cpp
ifeq ($(OS),Darwin) # OSX
	$(CC) -c $< $(COMPFLAGS) $(CPPFLAGS) -o $@
else # Other
	$(CC) -c $< $(CPPFLAGS) -o $@
endif

%.o: %.cu
ifeq ($(OS),Darwin) # OSX
	$(CC) -c $< $(COMPFLAGS) $(CPPFLAGS) -o $@
else # Other
	$(CC) -c $< $(CPPFLAGS) -o $@
endif


clean:
	-rm $(EXEC_NAME) $(OBJ) $(OBJ_TST) $(OBJ_BENCH) *.png

