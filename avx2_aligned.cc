#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <immintrin.h>
#include <malloc.h>
#include <cstring>

using namespace std;

void gen_mat(double* A, double* b, int n) {
    srand(12345);
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            if (i != j) {
                A[i*n + j] = (rand() % 2000 - 1000) / 1000.0;
                sum += fabs(A[i*n + j]);
            }
        }
        A[i*n + i] = sum + 1.0;
        b[i] = (rand() % 2000 - 1000) / 1000.0;
    }
}

void gauss_avx2_aligned(double* A, double* b, int n) {
    for (int k = 0; k < n; k++) {
        double pivot = A[k*n + k];
        double inv_pivot = 1.0 / pivot;
        __m256d v_inv_pivot = _mm256_set1_pd(inv_pivot);

        int j = k + 1;
        while (j < n && ((uintptr_t)&A[k*n + j] % 32 != 0)) {
            A[k*n + j] *= inv_pivot;
            j++;
        }
        for (; j <= n - 4; j += 4) {
            __m256d v = _mm256_load_pd(&A[k*n + j]);
            v = _mm256_mul_pd(v, v_inv_pivot);
            _mm256_store_pd(&A[k*n + j], v);
        }
        for (; j < n; j++) {
            A[k*n + j] *= inv_pivot;
        }
        b[k] *= inv_pivot;
        A[k*n + k] = 1.0;

        for (int i = k + 1; i < n; i++) {
            double factor = A[i*n + k];
            __m256d v_factor = _mm256_set1_pd(factor);

            j = k + 1;
            while (j < n && ((uintptr_t)&A[i*n + j] % 32 != 0)) {
                A[i*n + j] -= factor * A[k*n + j];
                j++;
            }
            for (; j <= n - 4; j += 4) {
                __m256d v1 = _mm256_load_pd(&A[i*n + j]);
                __m256d v2 = _mm256_load_pd(&A[k*n + j]);
                __m256d res = _mm256_fnmadd_pd(v_factor, v2, v1);
                _mm256_store_pd(&A[i*n + j], res);
            }
            for (; j < n; j++) {
                A[i*n + j] -= factor * A[k*n + j];
            }
            b[i] -= factor * b[k];
            A[i*n + k] = 0.0;
        }
    }

    for (int i = n - 1; i >= 0; i--) {
        for (int j = i + 1; j < n; j++) {
            b[i] -= A[i*n + j] * b[j];
        }
    }
}

bool check_res(const double* orig_A, 
               const double* orig_b, 
               const double* x, int n) {
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += orig_A[i*n + j] * x[j];
        }
        if (fabs(sum - orig_b[i]) > 1e-6) {
            cout << "Error at row " << i << ": " << sum << " vs " << orig_b[i] << endl;
            return false;
        }
    }
    return true;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <matrix_size>" << endl;
        return 1;
    }
    
    int n = atoi(argv[1]);
    
    double* A = (double*)_aligned_malloc(n * n * sizeof(double), 32);
    double* orig_A = (double*)_aligned_malloc(n * n * sizeof(double), 32);
    double* b = (double*)_aligned_malloc(n * sizeof(double), 32);
    double* orig_b = (double*)_aligned_malloc(n * sizeof(double), 32);
    
    gen_mat(A, b, n);
    memcpy(orig_A, A, n * n * sizeof(double));
    memcpy(orig_b, b, n * sizeof(double));

    auto start = chrono::high_resolution_clock::now();
    gauss_avx2_aligned(A, b, n);
    auto end = chrono::high_resolution_clock::now();
    int time_ms = chrono::duration_cast<chrono::milliseconds>(end - start).count();

    if (check_res(orig_A, orig_b, b, n)) {
        cout << "AVX2 aligned version passed" << endl;
        cout << "Size: " << n << ", Time: " << time_ms << " ms" << endl;
    } else {
        cout << "AVX2 aligned version failed" << endl;
    }

    _aligned_free(A);
    _aligned_free(orig_A);
    _aligned_free(b);
    _aligned_free(orig_b);

    return 0;
}