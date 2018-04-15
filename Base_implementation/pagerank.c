/**
 * COMP2129 - Assignment 4
 * Name: Perryanto Hartono
 * Unikey: phar2413
 * 27/5/2016 6:29pm
 *
 * PageRank implementation
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include <pthread.h>
#include <immintrin.h>

#include "pagerank.h"

page** set_of_pages; // Array of pointers to each page
size_t g_dim;        // Matrix dimension
size_t g_size;       // Total no. of elements in the matrix array

//////////-----------------------------///////////
///////////////DEBUGGING FUNCTIONS////////////////
//////////REMOVE WHEN ASIGNMENT IS DONE///////////
//////////-----------------------------///////////
/**
 * (DEBUG) Display the contents of the vector.
 */
void print_vector(double* arr) {
	printf("=========================================\n");
	printf("[");
	for (size_t i = 0; i < g_dim; i++) {
		char* format = (i == g_dim - 1) ? "%.8lf]\n" : "%.8lf,";
		printf(format, arr[i]);
	}
}

/**
 * (DEBUG) Display the contents of the matrix.
 */
void print_matrix(double* mat) {
	printf("=========================================\n");
	for (size_t y = 0; y < g_dim; y++) {
		for (size_t x = 0; x < g_dim; x++) {
			char* format = (x == g_dim - 1) ? "%.8lf\n" : "%.8lf ";
			printf(format, mat[y * g_dim + x]);
		}
	}
}

//////////////////////////////////////////////////
/////////USEFUL FUNCTIONS FOR CALCULATIONS////////
//////////////////////////////////////////////////

/**
 * Initialises a square matrix for use in calculations.
 */
double* matrix_init() {
	return calloc(g_size, sizeof(double));
}

/**
 * Initialises a square matrix with all values set to 1.
 */
double* matrix_init_uniform() {
	double* mat = matrix_init();
	for (size_t i = 0; i < g_size; i++) {
		mat[i] = 1.0;
	}
	return mat;
}

/**
 * Initialises the pagerank vector with the given size and default initial values.
 */
double* pr_vector_init(double* arr) {
	for (size_t i = 0; i < g_dim; i++) {
		arr[i] = 1.0 / g_dim;
	}

	return arr;
}

/**
 * Scalar multiplication - Multiplies the matrix with the given scalar.
 */
double* matrix_scalarmul(double* arr, double scalar) {
	double* res = matrix_init();

	for (size_t i = 0; i < g_size; i++) {
		res[i] = arr[i] * scalar;
	}
	return res;
}

/**
 * Matrix addition - Adds corresponding values of 2 matrices together.
 */
double* matrix_matadd(double* arr1, double* arr2) {
	double* res = matrix_init();

	for (size_t i = 0; i < g_size; i++) {
		res[i] = arr1[i] + arr2[i];
	}
	return res;
}

/**
 * Matrix multiplication - Multiplies a matrix and vector with the same dimensions.
 */
double* matrix_mulvector(double* vec, double* mat) {
	double* res = calloc(g_dim, sizeof(double));

	// Multiply the pagerank vector with the matrix
	for (size_t y = 0; y < g_dim; y++) { // y = column
		double sum = 0.0;

		// Go through columns and rows
		for (size_t i = 0; i < g_dim; i++) {
			// Calculate coordinates of row
			size_t rx = i, ry = y;

			// Multiply the rows and columns and add to the sum
			sum += (mat[ry * g_dim + rx] * vec[i]);
		}
		res[y] = sum;
	}

	return res;
}

/**
 * Check if an outlink exists between the pages a and b.
 */
bool link_exists(page* a, page* b) {
	//*DEBUG*///printf("COMPARING %p %s AND %p %s\n", a, a->name, b, b->name);
	node* il = b->inlinks;

	// If a list of inlink pages is found, iterate through it
	if (il) {
		node* il_traverse = il;
		while (il_traverse) {
			page* p = il_traverse->page;
			//*DEBUG*///printf("%p %s, ", p, p->name);
			if (p == a) {
				//*DEBUG*/printf("LINK FOUND\n");
				return true;
			}
			il_traverse = il_traverse->next;
		}
		//*DEBUG*///printf("\n");
	}

	//*DEBUG*/printf("NO LINKS\n");
	return false;
}

/**
 * Check if the difference between the norms of 2 vectors is less than EPSILON
 */
bool norm_diff_check(double* new, double* old) {
	double sum = 0;

	for (size_t i = 0; i < g_dim; i++) {
		sum += ((new[i] - old[i]) * (new[i] - old[i]));
	}
	double diff = sqrt(sum);
	//*DEBUG*/printf("diff = %.8lf\n", diff);

	if (diff < EPSILON) {
		return true;
	}
	return false;
}

/**
 * Construct the matrix M
 */
double* matrix_construct(page** set_of_pages) {
	double* mat = matrix_init();
	for (size_t i = 0; i < g_dim; i++) {
		for (size_t j = 0; j < g_dim; j++) {
			page* i_page = set_of_pages[i];
			page* j_page = set_of_pages[j];
			//*DEBUG*/printf("%s | %s\n", i_page->name, j_page->name);

			// Set the values in the matrix
			if (j_page->noutlinks == 0) {
				mat[i * g_dim + j] = 1.0 / g_dim;
			} else if (link_exists(j_page, i_page)) {
				mat[i * g_dim + j] = 1.0 / j_page->noutlinks;
			} else {
				mat[i * g_dim + j] = 0.0;
			}
		}
	}

	return mat;
}

/**
 * Construct the matrix M_hat = d*M + ((1 - d)/N)E
 */
double* matrix_construct_hat(double* M, double* E, double dampener) {
	double* dM = matrix_scalarmul(M, dampener);
	double* E_mul = matrix_scalarmul(E, (1 - dampener) / g_dim);

	double* res = matrix_matadd(dM, E_mul);
	free(dM);
	free(E_mul);
	return res;
}

/*
// Go through the linked list of pages
node* going_through = list;
while (going_through) {
	page* pg = going_through->page;

	printf("Name=%s, index=%zu, noutlinks=%zu, inlinks:", pg->name, pg->index, pg->noutlinks);

	// Add the page to the array set of pages
	set_of_pages[pg->index] = pg;

	// If a list of inlink pages is found, iterate through it
	node* il = pg->inlinks;
	if (il) {
		node* il_traverse = il;
		while (il_traverse) {
			printf("%s,", il_traverse->page->name);
			il_traverse = il_traverse->next;
		}
		printf("\n");
	} else {
		printf("none found\n");
	}

	going_through = going_through->next;
}
*/

//////////////////////////////////////////////////
///////PAGERANK FUNCTION THAT DOES EVERYTHING/////
//////////////////////////////////////////////////

void pagerank(node* list, size_t npages, size_t nedges, size_t nthreads, double dampener) {
	//*DEBUG*/printf("==============START OF PROGRAM================\n");
	//*DEBUG*///printf("list=%p, npages=%zu, nedges=%zu, nthreads=%zu, dampener=%.8lf\n", list, npages, nedges, nthreads, dampener);
	// Initialise the array set of pages
	set_of_pages = malloc(sizeof(page*) * npages);

	// Set global dimension values
	g_dim = npages;
	g_size = g_dim*g_dim;

	// Go through the linked list of pages
	node* going_through = list;
	while (going_through) {
		page* pg = going_through->page;

		//*DEBUG*/printf("Name=%s, index=%zu, noutlinks=%zu\n", pg->name, pg->index, pg->noutlinks);
		// Add the page to the array set of pages before moving on
		set_of_pages[pg->index] = pg;
		going_through = going_through->next;
	}

	/* DEBUG whether set_of_pages is initialised as expected
	for (int i = 0; i < npages; i++) {
		printf("SETOFPAGES: %s\n", set_of_pages[i]->name);
	}*/

	// Calculate the M matrix
	double* M = matrix_construct(set_of_pages);
	//*DEBUG*/print_matrix(M);

	// Calculate the M_hat matrix
	double* E = matrix_init_uniform();
	double* M_hat = matrix_construct_hat(M, E, dampener);
	//*DEBUG*/print_matrix(M_hat);

	// Initialise the vector of pagerank results
	double* pr_vector = calloc(g_dim, sizeof(double));
	pr_vector = pr_vector_init(pr_vector);
	//*DEBUG*/print_vector(pr_vector);

	/* Multiply matrix M_hat with the vector of pagerank results until the difference
	 * between the previous and the current iteration drops to below 0.005 */
	int qwe = 0;
	bool has_converged = false;
	double* pr_vector_next = NULL;
	while (!has_converged) {
		pr_vector_next = matrix_mulvector(pr_vector, M_hat);
		has_converged = norm_diff_check(pr_vector_next, pr_vector);
		//*DEBUG*/print_vector(pr_vector);
		//*DEBUG*/printf("iteration no.%d, epsilon=%.8lf, condition=%s\n", qwe, EPSILON, has_converged ? "true" : "false");

		// Move on to the next iteration
		free(pr_vector);
		pr_vector = pr_vector_next;
		qwe++;
	}


	// Display the final result
	for (int i = 0; i < g_dim; i++) {
		printf("%s %.8lf\n", set_of_pages[i]->name, pr_vector[i]);
	}

	// Free all allocated memory
	free(M);
	free(M_hat);
	free(E);
	free(set_of_pages);
	free(pr_vector);
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
