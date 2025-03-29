CC = mpicc
CFLAGS = -std=c99 -Wall -O0
LDFLAGS = -lm

# Source files for the two implementations
ITERATIVE_SRC = main_iterative.c Lab4_IO.c
PARALLEL_SRC = main.c Lab4_IO.c

# Object files
ITERATIVE_OBJ = $(ITERATIVE_SRC:.c=.o)
PARALLEL_OBJ = $(PARALLEL_SRC:.c=.o)

# Target executables
ITERATIVE_TARGET = main_iterative
PARALLEL_TARGET = main

all: $(ITERATIVE_TARGET) $(PARALLEL_TARGET)

$(ITERATIVE_TARGET): $(ITERATIVE_OBJ)
	$(CC) $(CFLAGS) -o $@ $(ITERATIVE_OBJ) $(LDFLAGS)

$(PARALLEL_TARGET): $(PARALLEL_OBJ)
	$(CC) $(CFLAGS) -o $@ $(PARALLEL_OBJ) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(ITERATIVE_TARGET) $(PARALLEL_TARGET) $(ITERATIVE_OBJ) $(PARALLEL_OBJ)