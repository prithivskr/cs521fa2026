#include <chrono>
#include "../include/utils.h"

#define NUM_RUNS 2

#define CHECK(name) \
  std::cout << "checking " << #name << std::endl;		\
  initialize(refC, Ref::M * Ref::N);				\
  name(ref.A, ref.B, refC, Ref::M, Ref::N, Ref::K);		\
  if (!ref.checkRef(refC)){					\
    std::cerr << #name << ": check ref failed!" << std::endl;	\
  };								
  
#define TIME(name) \
  for (int i = 0; i < 1; i++)						\
    {									\
      name(A, B, C, M, N, K);						\
    }									\
  std::chrono::duration<double, std::milli> time_##name(0);		\
  for (int i = 0; i < NUM_RUNS; i++)					\
    {									\
      initialize(C, M * N);						\
      auto start_time_ ## name = std::chrono::high_resolution_clock::now(); \
      name(A, B, C, M, N, K);						\
      auto end_time_ ## name = std::chrono::high_resolution_clock::now(); \
      time_ ## name += end_time_ ## name - start_time_ ## name;		\
    }									\
std::chrono::duration<double, std::milli> duration_ ## name = time_ ## name/float(NUM_RUNS); \
  std::cout << "Time taken for GEMM (CPU," << #name <<"): " << duration_ ## name.count() << "ms" << std::endl; 

#define T_SZ 36

using namespace std;

// reference CPU implementation of the GEMM kernel
// note that this implementation is naive and will run for longer for larger
// graphs
void gemm_cpu_o0(float* A, float* B, float *C, int M, int N, int K) {
  for (int j = 0; j < N; j++) {
    for (int i = 0; i < M; i++) {
      for (int k = 0; k < K; k++) {
	    C[i * N + j]  += A[i * K + k]  * B[k * N + j];
      }
    }
  }
}

// Your optimized implementations go here
// note that for o4 you don't have to change the code, but just the compiler
// flags. So, you can use o3's code for that part
void gemm_cpu_o1(float *A, float *B, float *C, int M, int N, int K) {
  for (int i = 0; i < M; i++) {
    for (int k = 0; k < K; k++) {
      for (int j = 0; j < N; j++) {
        C[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
  }
}

void gemm_cpu_o2(float *A, float *B, float *C, int M, int N, int K) {
  for (int i = 0; i < M; i++) {
    for (int kk = 0; kk < K; kk += T_SZ) {
      for (int jj = 0; jj < N; jj += T_SZ) {
        int ke = min(kk + T_SZ, K);

        for (int k = kk; k < ke; k++) {
          float a = A[i * K + k];
          int je = min(jj + T_SZ, N);

          for (int j = jj; j < je; j++) {
            C[i * N + j] += a * B[k * N + j];
          }
        }
      }
    }
  }
}


/*
 * Uses fma instructions
 *
 * Command:
 * objdump -D ./mp1_cpu | grep -i -E "vfmadd|vfmsub|vfnmadd|fma"
 *
 * Output:
 *
    182f:	c4 e2 e9 a9 85 d8 fe 	vfmadd213sd -0x128(%rbp),%xmm2,%xmm0
    1970:	c4 e2 d1 a9 85 d8 fe 	vfmadd213sd -0x128(%rbp),%xmm5,%xmm0
    1ab1:	c4 e2 f1 a9 85 d8 fe 	vfmadd213sd -0x128(%rbp),%xmm1,%xmm0
    1c85:	c4 e2 c9 a9 85 e8 fe 	vfmadd213sd -0x118(%rbp),%xmm6,%xmm0
    22de:	c4 e2 6d a8 04 01    	vfmadd213ps (%rcx,%rax,1),%ymm2,%ymm0
    232f:	c4 c2 61 98 04 87    	vfmadd132ps (%r15,%rax,4),%xmm3,%xmm0
    235b:	c4 e2 71 a9 06       	vfmadd213ss (%rsi),%xmm1,%xmm0
    2383:	c4 e2 71 a9 06       	vfmadd213ss (%rsi),%xmm1,%xmm0
    23a3:	c4 c2 59 99 0c 97    	vfmadd132ss (%r15,%rdx,4),%xmm4,%xmm1
    25f4:	c4 e2 71 b9 02       	vfmadd231ss (%rdx),%xmm1,%xmm0
    2775:	c4 e2 75 a8 04 3a    	vfmadd213ps (%rdx,%rdi,1),%ymm1,%ymm0
    27c5:	c4 c2 61 98 04 be    	vfmadd132ps (%r14,%rdi,4),%xmm3,%xmm0
    2804:	c4 c2 59 99 04 9e    	vfmadd132ss (%r14,%rbx,4),%xmm4,%xmm0
    283b:	c4 c2 51 99 04 9e    	vfmadd132ss (%r14,%rbx,4),%xmm5,%xmm0
    286b:	c4 c2 49 99 04 be    	vfmadd132ss (%r14,%rdi,4),%xmm6,%xmm0
    28f0:	c4 e2 69 99 43 fc    	vfmadd132ss -0x4(%rbx),%xmm2,%xmm0
    2b35:	c4 c2 71 a9 04 8e    	vfmadd213ss (%r14,%rcx,4),%xmm1,%xmm0
    2c26:	c4 c2 6d a8 44 0d 00 	vfmadd213ps 0x0(%r13,%rcx,1),%ymm2,%ymm0
    2c8d:	c4 c2 61 98 04 8c    	vfmadd132ps (%r12,%rcx,4),%xmm3,%xmm0
    2ccb:	c4 e2 71 a9 03       	vfmadd213ss (%rbx),%xmm1,%xmm0
    2d00:	c4 e2 71 a9 03       	vfmadd213ss (%rbx),%xmm1,%xmm0
    2d31:	c4 c2 59 99 0c 8c    	vfmadd132ss (%r12,%rcx,4),%xmm4,%xmm1
 */
void gemm_cpu_o3(float *A, float *B, float *C, int M, int N, int K) {
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < M; i++) {
    for (int kk = 0; kk < K; kk += T_SZ) {
      for (int jj = 0; jj < N; jj += T_SZ) {
        int ke = min(kk + T_SZ, K);

        for (int k = kk; k < ke; k++) {
          float a = A[i * K + k];
          int je = min(jj + T_SZ, N);

          #pragma omp simd
          for (int j = jj; j < je; j++) {
            C[i * N + j] += a * B[k * N + j];
          }
        }
      }
    }
  }
}


int main(int argc, char* argv[]) {
	if (argc < 3) {
	  std::cout << "Usage: mp1 <M> <N> <K>" << std::endl;
	  return 1;
	}

	int M = atoi(argv[1]);
	int N = atoi(argv[2]);
	int K = atoi(argv[3]);

	float* A = new float[M * K]();
	float* B = new float[K * N]();
	float* C = new float[M * N]();

	fillRandom(A, M * K);
	fillRandom(B, K * N);

	// Check if the kernel results are correct
	// note that even if the correctness check fails all optimized kernels will run.
	// We are not exiting the program at failure at this point.
	// It is a good idea to add more correctness checks to your code.
	// We may (at discretion) verify that your code is correct.
	float* refC = new float[Ref::M * Ref::N]();
	auto ref = Ref();
	CHECK(gemm_cpu_o0)
	CHECK(gemm_cpu_o1)
	CHECK(gemm_cpu_o2)
	CHECK(gemm_cpu_o3)
	delete[] refC;

	TIME(gemm_cpu_o0)
	TIME(gemm_cpu_o1)
	TIME(gemm_cpu_o2)
	TIME(gemm_cpu_o3)

	delete[] A;
	delete[] B;
	delete[] C;

	return 0;
}
