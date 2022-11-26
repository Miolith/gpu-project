CC=nvcc
CPPFLAGS=-L/usr/X11R6/lib -lm -lpthread -lX11
COMPFLAGS=-std=c++11
SRC=src/*.cpp src/*.cu
EXEC_NAME=./main

# Check System for OSX Comptability Support
OS=$(shell uname)


all: $(EXEC_NAME)

$(EXEC_NAME): CPPFLAGS += -O2
$(EXEC_NAME):
ifeq ($(OS),Darwin) # OSX
	g++ $(SRC) $(COMPFLAGS) $(CPPFLAGS) -o $@
else # Other
	$(CC) $(SRC) $(CPPFLAGS) -o $@
endif

run:
	$(EXEC_NAME) images/*

debug: CPPFLAGS += -g #-fsanitize=address
debug: $(EXEC_NAME)

clean:
	rm $(EXEC_NAME)

