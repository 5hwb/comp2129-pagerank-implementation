/**
 * COMP2129 - Assignment 4
 * Name: Perryanto Hartono
 * Unikey: phar2413
 * 4/6/2016 1:58pm
 *
 * PageRank implementation (final submission)
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include <pthread.h>
#include <immintrin.h>

#include "pagerank.h"

// Global variables
page** set_of_pages; // Array of pointers to each page
size_t g_sidelen;    // Matrix side length
size_t g_totalsize;  // Total no. of elements in the matrix array
size_t g_nthreads;   // Number of threads to use

// Function pointer for the universal multithreaded function
typedef void (*operation)(size_t start, size_t end, page** page_set, double input,
		double* vec, double* mat1, double* mat2, double* res);

// Matrix operation function arguments
typedef struct {
	size_t id;        // 8 byte unique id for thread
	size_t loop_size; // 8 byte integer to determine the length of the for loop
	page** page_set;  // 8 byte array of pointers to each page
	double input;     // 8-byte double precision floating point input
	double** vec;     // 8 byte pointer to vector
	double** mat1;    // 8 byte pointer to matrix 1 (to be multiplied with vector or matrix 2)
	double** mat2;    // 8 byte pointer to matrix 2
	double** res;     // 8-byte pointer to result matrix
	operation ops;    // 8-byte pointer to function that will perform the operation on the matrix
} matrixops_args;     // TOTAL: 64 bytes

//////////////////////////////////////////////////
/////////USEFUL FUNCTIONS FOR CALCULATIONS////////
//////////////////////////////////////////////////

/**
 * Check if an outlink exists between the pages a and b.
 */
bool link_exists(page* a, page* b) {
	node* il = b->inlinks;

	// If a list of inlink pages to 'b' is found, iterate through it
	if (il) {
		node* il_traverse = il;
		while (il_traverse) {
			page* p = il_traverse->page;
			// Return true if 'a' is found
			if (p == a) {
				return true;
			}
			il_traverse = il_traverse->next;
		}
	}

	return false;
}

/**
 * Check if the difference between the norms of 2 vectors is less than EPSILON
 */
bool norm_diff_check(double* new, double* old) {
	double sum = 0;

	for (size_t i = 0; i < g_sidelen; i++) {
		sum += ((new[i] - old[i]) * (new[i] - old[i]));
	}
	double diff = sqrt(sum);

	if (diff < EPSILON) {
		return true;
	}
	return false;
}

//////////////////////////////////////////////////
/////////MATRIX - VECTOR INITIALISATION///////////
//////////////////////////////////////////////////

/**
 * Initialises a square matrix for use in calculations.
 */
double* matrix_init() {
	return _mm_malloc(g_totalsize * sizeof(double), 16);
}

/**
 * Initialises the pagerank vector with the given size and default initial values.
 */
double* pr_vector_init(double* arr) {
	for (size_t i = 0; i < g_sidelen; i++) {
		arr[i] = 1.0 / g_sidelen;
	}

	return arr;
}


//////////////////////////////////////////////////
///////HELPER FUNCTIONS FOR MULTITHREADING////////
//////////////////////////////////////////////////

/* Construct the matrix M */
void func_construct(size_t start, size_t end, page** page_set, double input,
		double* vec, double* mat1, double* mat2, double* res) {

	// Local pointer variables to pages
	page* i_page;
	page* j_page;

	for (size_t i = start; i < end; i++) {
		for (size_t j = 0; j < g_sidelen; j++) {
			i_page = page_set[i];
			j_page = page_set[j];

			// Set the values in the matrix
			if (j_page->noutlinks == 0) {
				mat1[i * g_sidelen + j] = 1.0 / g_sidelen;
			} else if (link_exists(j_page, i_page)) {
				mat1[i * g_sidelen + j] = 1.0 / j_page->noutlinks;
			} else {
				mat1[i * g_sidelen + j] = 0.0;
			}
		}
	}
}

/* Scalar addition - Adds the given scalar to the matrix (uses SSE) */
void func_sse_sadd(size_t start, size_t end, page** page_set, double input,
		double* vec, double* mat1, double* mat2, double* res) {

	// For this code, I will assume that all computers have at least SSE2.
	// AVX: 4 f.p. elements, SSE2: 2 f.p. elements
	size_t vec_len;
	#ifdef __AVX__
		vec_len = 4;
	#else
		vec_len = 2;
	#endif

	// Check to ensure the loop length is always a multiple of the SSE vector length
	bool is_even = ((end - start) % vec_len == 0);
	size_t end_checked = end - ((end - start) % vec_len);

	// This double precision f.p. int array contains the scalar values
	double scalar[vec_len] __attribute__((aligned(16)));
	for (int i = 0; i < vec_len; i++) scalar[i] = input;

	// AVX: __m256d, SSE2: __m128d
	#ifdef __AVX__
		__m256d sse_scalar = _mm256_loadu_pd(scalar);
		__m256d sse_res;
		__m256d sse_mat1;
	#else
		__m128d sse_scalar = _mm_loadu_pd(scalar);
		__m128d sse_res;
		__m128d sse_mat1;
	#endif

	for (size_t i = start; i < end_checked; i += vec_len) {
		/*=====CODE TO EXECUTE IF THE COMPUTER HAS AVX=====*/
		#ifdef __AVX__
			// Load 2 doubles into a SSE vector
			sse_mat1 = _mm256_loadu_pd(mat1 + i);

			// Perform the required operation on 2 numbers simultaneously
			sse_res = _mm256_add_pd(sse_mat1, sse_scalar);

			// Place the result back into the final array
			_mm256_storeu_pd(res + i, sse_res);

		/*=====CODE TO EXECUTE IF THE COMPUTER DOES NOT SUPPORT AVX=====*/
		#else
			// Load 2 doubles into a SSE vector
			sse_mat1 = _mm_loadu_pd(mat1 + i);

			// Perform the required operation on 2 numbers simultaneously
			sse_res = _mm_add_pd(sse_mat1, sse_scalar);

			// Place the result back into the final array
			_mm_storeu_pd(res + i, sse_res);
		#endif
	}

	// If the array length is not even, calculate the final element values
	if (!is_even) {
		for (size_t i = end_checked; i < end; i++) {
			res[i] = mat1[i] + input;
		}
	}
}

/* Scalar multiplication - Multiplies the matrix with the given scalar (uses SSE) */
void func_sse_smul(size_t start, size_t end, page** page_set, double input,
		double* vec, double* mat1, double* mat2, double* res) {

	// For this code, I will assume that all computers have at least SSE2.
	// AVX: 4 f.p. elements, SSE2: 2 f.p. elements
	size_t vec_len;
	#ifdef __AVX__
		vec_len = 4;
	#else
		vec_len = 2;
	#endif

	// Check to ensure the loop length is always a multiple of the SSE vector length
	bool is_even = ((end - start) % vec_len == 0);
	size_t end_checked = end - ((end - start) % vec_len);

	// This double precision f.p. int array contains the scalar values
	double scalar[vec_len] __attribute__((aligned(16)));
	for (int i = 0; i < vec_len; i++) scalar[i] = input;

	// AVX: __m256d, SSE2: __m128d
	#ifdef __AVX__
		__m256d sse_scalar = _mm256_loadu_pd(scalar);
		__m256d sse_res;
		__m256d sse_mat1;
	#else
		__m128d sse_scalar = _mm_loadu_pd(scalar);
		__m128d sse_res;
		__m128d sse_mat1;
	#endif

	for (size_t i = start; i < end_checked; i += vec_len) {
		/*=====CODE TO EXECUTE IF THE COMPUTER HAS AVX=====*/
		#ifdef __AVX__
			// Load 2 doubles into a SSE vector
			sse_mat1 = _mm256_loadu_pd(mat1 + i);

			// Perform the required operation on 2 numbers simultaneously
			sse_res = _mm256_mul_pd(sse_mat1, sse_scalar);

			// Place the result back into the final array
			_mm256_storeu_pd(res + i, sse_res);

		/*=====CODE TO EXECUTE IF THE COMPUTER DOES NOT SUPPORT AVX=====*/
		#else
			// Load 2 doubles into a SSE vector
			sse_mat1 = _mm_loadu_pd(mat1 + i);

			// Perform the required operation on 2 numbers simultaneously
			sse_res = _mm_mul_pd(sse_mat1, sse_scalar);

			// Place the result back into the final array
			_mm_storeu_pd(res + i, sse_res);
		#endif
	}

	// If the array length is not even, calculate the final element values
	if (!is_even) {
		for (size_t i = end_checked; i < end; i++) {
			res[i] = mat1[i] * input;
		}
	}
}

/* Matrix multiplication - Multiplies a matrix and vector with the same dimensions (uses SSE) */
void func_sse_mmul(size_t start, size_t end, page** page_set, double input,
		double* vec, double* mat1, double* mat2, double* res) {

	// For this code, I will assume that all computers have at least SSE2.
	// AVX: 4 f.p. elements, SSE2: 2 f.p. elements
	size_t vec_len;
	#ifdef __AVX__
		vec_len = 4;
	#else
		vec_len = 2;
	#endif

	// Check to ensure the loop length is always a multiple of the SSE vector length
	bool is_even = (g_sidelen % vec_len == 0);
	size_t g_sidelen_checked = g_sidelen - (g_sidelen % vec_len);

	// Local variable to collect the sum of the row-column multiplication
	register double sum;

	// AVX: __m256d, SSE2: __m128d
	#ifdef __AVX__
		__m256d sse_mat;
		__m256d sse_vec;
		__m256d sse_res;
	#else
		__m128d sse_mat;
		__m128d sse_vec;
		__m128d sse_res;
	#endif

	// Temporary variables to get and store the result
	register double *mat1_ptr;
	register double temp_res[vec_len] __attribute__((aligned(16)));

	// Multiply the pagerank vector with the matrix
	for (size_t y = start; y < end; y++) { // y = column
		sum = 0.0;

		// Set up pointer to first element in current row
		mat1_ptr = mat1 + (y * g_sidelen);

		// Go through columns and rows
		for (size_t i = 0; i < g_sidelen_checked; i += vec_len) {
			#ifdef __AVX__
				// Load values into SSE vectors
				sse_mat = _mm256_loadu_pd(mat1_ptr + i);
				sse_vec = _mm256_loadu_pd(vec + i);

				// Compute dot product of matrix row and column
				sse_res = _mm256_mul_pd(sse_mat, sse_vec);

				// Place the result in the temp array
				_mm256_storeu_pd(temp_res, sse_res);
			#else
				// Load values into SSE vectors
				sse_mat = _mm_loadu_pd(mat1_ptr + i);
				sse_vec = _mm_loadu_pd(vec + i);

				// Compute dot product of matrix row and column
				sse_res = _mm_mul_pd(sse_mat, sse_vec);

				// Place the result in the temp array
				_mm_storeu_pd(temp_res, sse_res);
			#endif

			// Multiply the rows and columns and add to the sum
			for (int i = 0; i < vec_len; i++) {
				sum += temp_res[i];
			}
		}
		// If the array length is not even, calculate the final element values
		if (!is_even) {
			for (size_t i = g_sidelen_checked; i < g_sidelen; i++) {
				sum += (mat1_ptr[i] * vec[i]);
			}
		}
		res[y] = sum;
	}
}

//////////////////////////////////////////////////
/////////////MULTITHREADED FUNCTIONS//////////////
//////////////////////////////////////////////////

/**
 * Worker function for all multithreaded functions
 */
void* multithread_worker(void* args) {

	matrixops_args* margs = (matrixops_args*) args;

	// Calculate the area of the matrix to work on,
	// based on the thread id and given loop size
	const size_t start = margs->id * (margs->loop_size / g_nthreads);
	const size_t end =
			(margs->id == g_nthreads - 1)
			? margs->loop_size
			: (margs->id + 1) * (margs->loop_size / g_nthreads);

	// Initialise the argument variables, checking if they are null first
	double input = margs->input;
	page** page_set = margs->page_set;
	double* vec  = (margs->vec == NULL)  ? NULL : *(margs->vec);
	double* mat1 = (margs->mat1 == NULL) ? NULL : *(margs->mat1);
	double* mat2 = (margs->mat2 == NULL) ? NULL : *(margs->mat2);
	double* res  = (margs->res == NULL)  ? NULL : *(margs->res);

	// Execute the function *ops - this may be a matrix multiplication, addition or something else
	(margs->ops)(start, end, page_set, input, vec, mat1, mat2, res);
	return NULL;
}

/**
 * Construct the matrix M
 */
double* matrix_construct_threaded(page** set_of_pages) {
	double* mat = matrix_init();

	// Initialise the pthread and argument variables
	pthread_t thread_ids[g_nthreads];
	matrixops_args args[g_nthreads];

	// For each thread, set the arguments to be passed to the thread function
	for (size_t i = 0; i < g_nthreads; i++) {
		args[i] = (matrixops_args) {
			.id        = i,
			.loop_size = g_sidelen,
			.page_set  = set_of_pages,
			.vec       = NULL,
			.input     = 0.0,
			.mat1      = &mat,
			.mat2      = NULL,
			.res       = NULL,
			.ops       = func_construct
		};
	}

	// Create the threads
	for (size_t i = 0; i < g_nthreads; i++) {
		pthread_create(thread_ids + i, NULL, multithread_worker, args + i);
	}

	// Wait for the threads to finish
	for (size_t i = 0; i < g_nthreads; i++) {
		pthread_join(thread_ids[i], NULL);
	}

	return mat;
}

/**
 * Scalar addition - Adds the given scalar to the matrix.
 */
double* matrix_scalaradd_threaded(double* arr, double scalar) {
	double* res = matrix_init();

	// Initialise the pthread and argument variables
	pthread_t thread_ids[g_nthreads];
	matrixops_args args[g_nthreads];

	// For each thread, set the arguments to be passed to the thread function
	for (size_t i = 0; i < g_nthreads; i++) {
		args[i] = (matrixops_args) {
			.id        = i,
			.loop_size = g_totalsize,
			.page_set  = NULL,
			.vec       = NULL,
			.input     = scalar,
			.mat1      = &arr,
			.mat2      = NULL,
			.res       = &res,
			.ops       = func_sse_sadd
		};
	}

	// Create the threads
	for (size_t i = 0; i < g_nthreads; i++) {
		pthread_create(thread_ids + i, NULL, multithread_worker, args + i);
	}

	// Wait for the threads to finish
	for (size_t i = 0; i < g_nthreads; i++) {
		pthread_join(thread_ids[i], NULL);
	}

	return res;
}

/**
 * Scalar multiplication - Multiplies the matrix with the given scalar.
 */
double* matrix_scalarmul_threaded(double* arr, double scalar) {
	double* res = matrix_init();

	// Initialise the pthread and argument variables
	pthread_t thread_ids[g_nthreads];
	matrixops_args args[g_nthreads];

	// For each thread, set the arguments to be passed to the thread function
	for (size_t i = 0; i < g_nthreads; i++) {
		args[i] = (matrixops_args) {
			.id        = i,
			.loop_size = g_totalsize,
			.page_set  = NULL,
			.vec       = NULL,
			.input     = scalar,
			.mat1      = &arr,
			.mat2      = NULL,
			.res       = &res,
			.ops       = func_sse_smul
		};
	}

	// Create the threads
	for (size_t i = 0; i < g_nthreads; i++) {
		pthread_create(thread_ids + i, NULL, multithread_worker, args + i);
	}

	// Wait for the threads to finish
	for (size_t i = 0; i < g_nthreads; i++) {
		pthread_join(thread_ids[i], NULL);
	}

	return res;
}

/**
 * Matrix multiplication - Multiplies a matrix and vector with the same dimensions.
 */
double* matrix_mulvector_threaded(double* vec, double* mat) {
	double* res = _mm_malloc(g_sidelen * sizeof(double), 16);

	// Initialise the pthread and argument variables
	pthread_t thread_ids[g_nthreads];
	matrixops_args args[g_nthreads];

	// For each thread, set the arguments to be passed to the thread function
	for (size_t i = 0; i < g_nthreads; i++) {
		args[i] = (matrixops_args) {
			.id        = i,
			.loop_size = g_sidelen,
			.page_set  = NULL,
			.vec       = &vec,
			.input     = 0.0,
			.mat1      = &mat,
			.mat2      = NULL,
			.res       = &res,
			.ops       = func_sse_mmul
		};
	}

	// Create the threads
	for (size_t i = 0; i < g_nthreads; i++) {
		pthread_create(thread_ids + i, NULL, multithread_worker, args + i);
	}

	// Wait for the threads to finish
	for (size_t i = 0; i < g_nthreads; i++) {
		pthread_join(thread_ids[i], NULL);
	}

	return res;
}

/**
 * Construct the matrix M_hat = d*M + ((1 - d)/N)E
 */
double* matrix_construct_hat(double* M, double dampener) {
	double* dM = matrix_scalarmul_threaded(M, dampener);
	double* res = matrix_scalaradd_threaded(dM, (1 - dampener) / g_sidelen);
	_mm_free(dM);
	return res;
}

//////////////////////////////////////////////////
///////PAGERANK FUNCTION THAT DOES EVERYTHING/////
//////////////////////////////////////////////////

void pagerank(node* list, size_t npages, size_t nedges, size_t nthreads, double dampener) {
	// Initialise the array set of pages
	set_of_pages = malloc(sizeof(page*) * npages);

	// Set global variable values
	g_sidelen = npages;
	g_totalsize = g_sidelen*g_sidelen;
	g_nthreads = nthreads;

	// Go through the linked list of pages
	node* going_through = list;
	while (going_through) {
		page* pg = going_through->page;

		// Add the page to the array set of pages before moving on
		set_of_pages[pg->index] = pg;
		going_through = going_through->next;
	}

	// Calculate the M matrix
	double* M = matrix_construct_threaded(set_of_pages);

	// Calculate the M_hat matrix
	double* M_hat = matrix_construct_hat(M, dampener);

	// Free M, it is not needed beyond this point.
	_mm_free(M);

	// Initialise the vector of pagerank results
	double* pr_vector = _mm_malloc(g_sidelen * sizeof(double), 16);
	pr_vector = pr_vector_init(pr_vector);

	/* Multiply matrix M_hat with the vector of pagerank results until the difference
	 * between the previous and the current iteration drops to below 0.005 */
	bool has_converged = false;
	double* pr_vector_next = NULL;
	while (!has_converged) {
		pr_vector_next = matrix_mulvector_threaded(pr_vector, M_hat);
		has_converged = norm_diff_check(pr_vector_next, pr_vector);

		// Move on to the next iteration
		_mm_free(pr_vector);
		pr_vector = pr_vector_next;
	}


	// Display the final result!
	for (int i = 0; i < g_sidelen; i++) {
		printf("%s %.8lf\n", set_of_pages[i]->name, pr_vector[i]);
	}

	// Free all remaining allocated memory
	_mm_free(M_hat);
	free(set_of_pages);
	_mm_free(pr_vector);
}

/*
######################################
### DO NOT MODIFY BELOW THIS POINT ###
######################################
*/

int main(int argc, char** argv) {

	/*
	######################################################
	### DO NOT MODIFY THE MAIN FUNCTION OR HEADER FILE ###
	######################################################
	*/

	config conf;

	init(&conf, argc, argv);

	node* list = conf.list;
	size_t npages = conf.npages;
	size_t nedges = conf.nedges;
	size_t nthreads = conf.nthreads;
	double dampener = conf.dampener;

	pagerank(list, npages, nedges, nthreads, dampener);

	release(list);

	return 0;
}
