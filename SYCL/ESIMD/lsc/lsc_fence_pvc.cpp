//==------- lsc_fence_pvc.cpp - DPC++ ESIMD on-device test -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test checks lsc_fence intrinsic.

// REQUIRES: gpu-intel-pvc
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include "../esimd_test_utils.hpp"

#include <CL/sycl.hpp>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <sycl/ext/intel/experimental/esimd.hpp>

int main() {
  using namespace cl::sycl;
  using namespace sycl::ext::intel::experimental::esimd;
  auto size = size_t{512};
  unsigned constexpr SIMDSize = 8;

  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());
  auto device = q.get_device();
  std::cout << "Device name: " << device.get_info<info::device::name>()
            << std::endl;

  auto *res_vec = malloc_shared<int>(size, q);
  std::fill(res_vec, res_vec + size, 0);

  try {
    q.submit([&](handler &h) {
      h.parallel_for<class SimplestKernel>(
          range<2>{size / SIMDSize, 2}, [=](id<2> id) SYCL_ESIMD_KERNEL {
            // Basically this kernel is an example from wiki:
            // https://en.wikipedia.org/wiki/Memory_barrier#Example
            slm_init(8192);
            auto offset = id[0] * SIMDSize;
            auto byte_offset = offset * sizeof(int);
            auto cond_offset = size * sizeof(int) + byte_offset;
            if (id[1] % 2 == 0) {
              // First thread: write data and condition
              // and provoke gpu to reorder instructions
              auto data = simd<int, SIMDSize>(offset, 1);
              lsc_slm_block_store<int, SIMDSize>(byte_offset, data * 10);
              lsc_slm_block_store<int, SIMDSize>(byte_offset, data * 5);
              lsc_slm_block_store<int, SIMDSize>(byte_offset, data);
              // Protect from reordering for the last two instructions
              lsc_fence<lsc_memory_kind::shared_local>();
              lsc_slm_block_store<int, SIMDSize>(cond_offset,
                                                 simd<int, SIMDSize>(1));
            } else {
              auto condition = simd<int, SIMDSize>(0);
              int imax = 1000;
              int i = 0;
              while (condition[0] == 0 && i < imax) {
                condition = lsc_slm_block_load<int, SIMDSize>(cond_offset);
                ++i;
              }
              // Protect from reordering for the while cycle and data read
              lsc_fence<lsc_memory_kind::shared_local>();
              auto data = lsc_slm_block_load<int, SIMDSize>(byte_offset);
              lsc_block_store<int, SIMDSize>(res_vec + offset, data);
            }
          });
    });
    q.wait();
  } catch (sycl::exception e) {
    std::cout << "SYCL exception caught: " << e.what();
    free(res_vec, q);
    return 1;
  }

  auto error = 0;
  for (auto i = 0; i != size; ++i) {
    error += std::abs(res_vec[i] - i);
  }
  std::cout << (error != 0 ? "FAILED" : "PASSED") << std::endl;
  free(res_vec, q);
  return error;
}
