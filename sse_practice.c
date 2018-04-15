#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include <pthread.h>
#include <immintrin.h>
#include <time.h>
// yeah.
#define LEN 100000000
#define NUM_OF_RUNS 1

// SSE test by 5hwb
// clang -g -O1 -Wall -Werror -std=gnu11 -march=native -lm -pthread sse_practice.c -o sse_practice

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

	printf("Initialise SIMD techniques with sizeof(__m128d) = %zu\n", sizeof(__m128d));
	const clock_t tick = clock();
	double* DONE = calloc(sizeof(double), LEN);

	//double* arrf1 __attribute__((aligned(16)));
	//double* arrf2 __attribute__((aligned(16)));

	register __m128d /*SPEED1, SPEED2,*/ YEAH;

	int run = 0;
	while (run < NUM_OF_RUNS) {
		//putchar('D');
		// Check to ensure the loop length is always even
		bool is_even = (LEN % 2 == 0);
		int length = (is_even) ? LEN : LEN-1;

		// For all even indices of the array, use SSE instructions
		for (int i = 0; i < length; i += 2) {
			// Prepare the pointers to 2 double numbers!
			//arrf1 = arr1 + i;
			//arrf2 = arr2 + i;

			// Load these numbers into memory!
			//SPEED1 = _mm_load_pd(arrf1);
			//SPEED2 = _mm_load_pd(arrf2);

			// Add 2 numbers AT THE SAME TIME!
			YEAH = _mm_add_pd(*((__m128d*) (arr1 + i)), *((__m128d*) (arr2 + i)));

			// Place the result back into the final array
			_mm_store_pd(DONE + i, YEAH);
		}
		// If the array length is not even, calculate the final array element value
		if (!is_even) {
			DONE[LEN - 1] = arr1[LEN - 1] + arr2[LEN - 1];
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
	double arrf1[2] __attribute__((aligned(16))) = { 1.2, 3.5 };
	double arrf2[2] __attribute__((aligned(16))) = { 8.5, 4.3 };

	__m128d SPEED1 = _mm_loadu_pd(arrf1);
	__m128d SPEED2 = _mm_loadu_pd(arrf2);

	__m128d YEAH = _mm_add_pd(SPEED1, SPEED2);

	double* DONE = calloc(sizeof(double), 2);

	_mm_store_pd(DONE, YEAH);

	for (int i = 0; i < 2; i++) {
		printf("%f\n", DONE[i]);
	}
*/
