#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include <pthread.h>
#include <immintrin.h>
#include <time.h>
// lame.
#define LEN 100000000
#define NUM_OF_RUNS 1

// SSE test by 5hwb
// clang -g -O1 -Wall -Werror -std=gnu11 -march=native -lm -pthread sse_not.c -o sse_not

/**
 * (DEBUG) Display the contents of the vector.
 */
void print_vector(double* arr) {
	//printf("=========================================\n");
	printf("[");
	for (size_t i = 0; i < LEN; i++) {
		char* format = (i == LEN - 1) ? "%.3f]\n" : "%.3f,";
		printf(format, arr[i]);
	}
}

/**
 * (DEBUG) Initialises an array
 */
void init_array(double* arr, double start, double step) {

	for (int i = 0; i < LEN; i++) {
		arr[i] = start;
		start += step;
	}
}

int main(void) {
	printf("Initialise two %d-element arrays\n", LEN);
	double* arr1 = calloc(sizeof(double), LEN);
	double* arr2 = calloc(sizeof(double), LEN);
	init_array(arr1, 1.0, 1.0);
	init_array(arr2, 3.0, 0.5);

	//*DEBUG*/print_vector(arr1);
	//*DEBUG*/print_vector(arr2);

	printf("Initialise final array\n");
	const clock_t tick = clock();
	double* DONE = calloc(sizeof(double), LEN);

	int run = 0;
	while (run < NUM_OF_RUNS) {
		//putchar('D');
		for (int i = 0; i < LEN; i += 1) {
			DONE[i] = arr1[i] + arr2[i];
		}
		run++;
	}

	const clock_t tock = clock();
	printf("Time elapsed: %.8lfs\n", (double) (tock - tick) / CLOCKS_PER_SEC);
	//*DEBUG*/print_vector(DONE);

	free(arr1);
	free(arr2);
	free(DONE);
	printf("Done.\n");
	return 0;
}

/*
	double arrf1[4] __attribute__((aligned(16))) = { 1.2, 3.5, 1.7, 2.8 };
	double arrf2[4] __attribute__((aligned(16))) = { 8.5, 4.3, 2.7, 1.2 };

	__m128 SPEED1 = _mm_loadu_ps(arrf1);
	__m128 SPEED2 = _mm_loadu_ps(arrf2);

	__m128 YEAH = _mm_add_ps(SPEED1, SPEED2);

	double* DONE = calloc(sizeof(double), 4);

	_mm_store_ps(DONE, YEAH);

	for (int i = 0; i < 4; i++) {
		printf("%f\n", DONE[i]);
	}
*/
