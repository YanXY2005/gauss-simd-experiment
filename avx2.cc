#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <immintrin.h>

using namespace std;

void gen_mat(vector<double>& A, vector<double>& b, int n) {
    srand(12345);
    A.resize(n * n);
    b.resize(n);
    
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

void gauss_avx2(vector<double>& A, vector<double>& b, int n) {
    for (int k = 0; k < n; k++) {
        double pivot = A[k*n + k];
        double inv_pivot = 1.0 / pivot;
        __m256d v_inv_pivot = _mm256_set1_pd(inv_pivot);

        int j = k + 1;
        for (; j <= n - 16; j += 16) {
            __m256d a1 = _mm256_loadu_pd(&A[i*n+j]);
            __m256d k1 = _mm256_loadu_pd(&A[k*n+j]);
            __m256d a2 = _mm256_loadu_pd(&A[i*n+j+4]);
            __m256d k2 = _mm256_loadu_pd(&A[k*n+j+4]);
            _mm256_storeu_pd(&A[i*n+j], _mm256_fnmadd_pd(v_factor, k1, a1));
            _mm256_storeu_pd(&A[i*n+j+4], _mm256_fnmadd_pd(v_factor, k2, a2));
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
            for (; j <= n - 4; j += 4) {
                __m256d v1 = _mm256_loadu_pd(&A[i*n + j]);
                __m256d v2 = _mm256_loadu_pd(&A[k*n + j]);
                __m256d res = _mm256_fnmadd_pd(v_factor, v2, v1);
                _mm256_storeu_pd(&A[i*n + j], res);
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

bool check_res(const vector<double>& orig_A, 
               const vector<double>& orig_b, 
               const vector<double>& x, int n) {
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
    
    vector<double> A, orig_A;
    vector<double> b, orig_b;
    
    gen_mat(A, b, n);
    orig_A = A;
    orig_b = b;

    auto start = chrono::high_resolution_clock::now();
    gauss_avx2(A, b, n);
    auto end = chrono::high_resolution_clock::now();
    int time_ms = chrono::duration_cast<chrono::milliseconds>(end - start).count();

    if (check_res(orig_A, orig_b, b, n)) {
        cout << "AVX2 version passed" << endl;
        cout << "Size: " << n << ", Time: " << time_ms << " ms" << endl;
    } else {
        cout << "AVX2 version failed" << endl;
    }

    return 0;
}