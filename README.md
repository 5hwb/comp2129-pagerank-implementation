# COMP2129 Pagerank Implementation

This is a C program that implements the PageRank algorithm, which aims to determine the importance of a web page (represented by a node) based on how many web pages link to it. The program was originally developed for a 2nd year university course on Operating Systems.

This program uses SSE instructions and multithreading to allow efficient use of modern multi-core CPUs for quicker computing time.

## Structure

The main source files are `pagerank.c` and `pagerank.h`. There are also some use-case tests in the `tests` folder. There is also a 'base implementation' version of pagerank.c in the `Base-implementation` folder, which contains the non-multithreaded version of the program.

## How to build

Open a terminal located at the repo's directory, and run the command `make`. To test the use-cases, run the script `testEm.sh`.
