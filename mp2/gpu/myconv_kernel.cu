#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <torch/extension.h>

// example
#define TILE_H 8
#define TILE_W 8
#define TILE_C 16

// Kernel declaration
__global__ void gemm_gpu_o4_kernel(
    const float *__restrict__ x, // input: N x C x H x W
    const float *__restrict__ w, // weights: C_out x C_in x KH x KW
    float *__restrict__ out,     // output: N x C x H x W
    int N, int C_in, int H, int W, int C_out, int KH, int KW, int stride,
    int pad, int out_h, int out_w) {
  extern __shared__ float shmem[]; // shared memory for partial sums

  // owned output tile
  int blck_h = blockIdx.y * TILE_H;
  int blck_w = blockIdx.x * TILE_W;

  int oh = blck_h + threadIdx.y;
  int ow = blck_w + threadIdx.x;

  int n = blockIdx.z / C_out;
  int co = blockIdx.z % C_out;

  bool in_bounds = (oh < out_h && ow < out_w);

  int h_tile_id = (TILE_H - 1) * stride + KH;
  int w_tile_id = (TILE_W - 1) * stride + KW;

  int xtn = TILE_C * h_tile_id * w_tile_id;
  int wtn = TILE_C * KH * KW;

  float *sh_x = shmem;
  float *sh_w = shmem + xtn;

  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * blockDim.y;

  float acc = 0;

  for (int ci_base = 0; ci_base < C_in; ci_base += TILE_C) {

    int input_y0 = blck_h * stride - pad;
    int input_x0 = blck_w * stride - pad;

    int t_sz = h_tile_id * w_tile_id;

    for (int i = tid; i < xtn; i += num_threads) {
      int tc = i / t_sz;

      int rem = i % t_sz;
      int sy = rem / w_tile_id;
      int sx = rem % w_tile_id;

      int ci = ci_base + tc;

      int iy = input_y0 + sy;
      int ix = input_x0 + sx;

      if (ci < C_in && iy >= 0 && iy < H && ix >= 0 && ix < W) {

        sh_x[i] = x[((n * C_in + ci) * H + iy) * W + ix];

      } else {
        sh_x[i] = 0;
      }
    }

    for (int i = tid; i < wtn; i += num_threads) {
      int tc = i / (KH * KW);

      int rem = i % (KH * KW);
      int kh = rem / KW;
      int kw = rem % KW;

      int ci = ci_base + tc;

      if (ci < C_in) {
        sh_w[i] = w[((co * C_in + ci) * KH + kh) * KW + kw];
      } else {
        sh_w[i] = 0;
      }
    }

    __syncthreads();

    if (in_bounds) {
      for (int tc = 0; tc < TILE_C; ++tc) {
        for (int kh = 0; kh < KH; ++kh) {
          for (int kw = 0; kw < KW; ++kw) {
            int sy = threadIdx.y * stride + kh;
            int sx = threadIdx.x * stride + kw;

            float xv = sh_x[(tc * h_tile_id + sy) * w_tile_id + sx];
            float wv = sh_w[(tc * KH + kh) * KW + kw];

            acc += xv * wv;
          }
        }
      }
    }

    __syncthreads();
  }

  if (in_bounds) {
    out[((n * C_out + co) * out_h + oh) * out_w + ow] = acc;
  }
}

// Function for Python binding
torch::Tensor conv_cuda(torch::Tensor x, torch::Tensor w, int stride, int pad) {
  int N = x.size(0);
  int C_in = x.size(1);
  int H = x.size(2);
  int W = x.size(3);

  int C_out = w.size(0);
  int KH = w.size(2);
  int KW = w.size(3);

  int out_h = (H + 2 * pad - KH) / stride + 1;
  int out_w = (W + 2 * pad - KW) / stride + 1;

  auto out = torch::zeros({N, C_out, out_h, out_w}, x.options());

  dim3 block(8, 8);
  dim3 grid((out_w + block.x - 1) / block.x, (out_h + block.y - 1) / block.y,
            N * C_out);

  int h_tile_id = (TILE_H - 1) * stride + KH;
  int w_tile_id = (TILE_W - 1) * stride + KW;
  size_t shared_bytes =
      TILE_C * (h_tile_id * w_tile_id + KH * KW) * sizeof(float);

  gemm_gpu_o4_kernel<<<grid, block, shared_bytes>>>(
      x.data_ptr<float>(), w.data_ptr<float>(), out.data_ptr<float>(), N, C_in,
      H, W, C_out, KH, KW, stride, pad, out_h, out_w);

  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("conv_cuda", &conv_cuda, "Custom Conv2D (CUDA)");
}
