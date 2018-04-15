#include <stdio.h>
#include <immintrin.h>

int main(void) {
    #ifdef __m256d
    printf("This hardware is AVX compatible\n");
    // AVX stuff here
    #else
    printf("This hardware is ----not -------AVX compatible\n");
    // Non-AVX stuff here
    #endif
    return 0;
}
