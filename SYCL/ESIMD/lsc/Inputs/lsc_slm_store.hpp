//==------- lsc_slm_store.hpp - DPC++ ESIMD on-device test -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <sycl/ext/intel/experimental/esimd.hpp>

#include <iostream>

#include "common.hpp"

using namespace cl::sycl;
using namespace sycl::ext::intel::experimental::esimd;

template <int case_num, typename T, uint32_t Groups, uint32_t Threads,
          uint16_t VL, uint16_t VS, bool transpose,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none>
bool test(uint32_t pmask = 0xffffffff) {
  static_assert((VL == 1) || !transpose, "Transpose must have exec size 1");
  if constexpr (DS == lsc_data_size::u8u32 || DS == lsc_data_size::u16u32) {
    static_assert(!transpose, "Conversion types may not use vector");
    static_assert(VS == 1, "Only D32 and D64 support vector load");
  }

  static_assert(DS != lsc_data_size::u16u32h, "D16U32h not supported in HW");
  static_assert(sizeof(T) >= 4,
                "D8 and D16 are valid only in 2D block load/store");

  uint16_t Size = Groups * Threads * VL * VS;

  T vmask = (T)-1;
  if constexpr (DS == lsc_data_size::u8u32)
    vmask = (T)0xff;
  if constexpr (DS == lsc_data_size::u16u32)
    vmask = (T)0xffff;
  if constexpr (DS == lsc_data_size::u16u32h)
    vmask = (T)0xffff0000;

  T old_val = get_rand<T>();
  T new_val = get_rand<T>();

  auto GPUSelector = gpu_selector{};
  auto q = queue{GPUSelector};
  auto dev = q.get_device();
  std::cout << "Running case #" << case_num << " on "
            << dev.get_info<info::device::name>() << "\n";
  auto ctx = q.get_context();

  // workgroups
  cl::sycl::range<1> GlobalRange{Groups};
  // threads in each group
  cl::sycl::range<1> LocalRange{Threads};
  cl::sycl::nd_range<1> Range{GlobalRange * LocalRange, LocalRange};

  T *out = static_cast<T *>(sycl::malloc_shared(Size * sizeof(T), dev, ctx));
  for (int i = 0; i < Size; i++)
    out[i] = 0;

  std::vector<uint32_t> p(VL, 0);
  if constexpr (!transpose)
    for (int i = 0; i < VL; i++)
      p[i] = (pmask >> i) & 1;

  try {
    buffer<uint32_t, 1> bufp(p.data(), p.size());

    auto e = q.submit([&](handler &cgh) {
      auto accp = bufp.template get_access<access::mode::read>(cgh);
      cgh.parallel_for<KernelID<case_num>>(
          Range, [=](cl::sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL {
            constexpr uint16_t gran = 4;    // using oword write x1 to init SLM
            constexpr uint16_t slm_blocks = // number of owords per Thread
                VL * VS / gran + ((VL * VS) % gran ? 1 : 0);
            constexpr uint16_t slm_ud_per_group = slm_blocks * Threads * gran;
            constexpr uint16_t slm_size_per_group =
                slm_ud_per_group * sizeof(T);
            constexpr uint16_t slm_size = slm_size_per_group * Groups;

            uint16_t globalID = ndi.get_global_id(0);
            uint32_t elem_off = globalID * VL * VS;
            uint32_t byte_off = elem_off * sizeof(T);

            slm_init(slm_size);
            if (ndi.get_local_id(0) == 0) {
              uint32_t group_off = ndi.get_group(0) * slm_size_per_group;
              simd<T, gran> slm_val(old_val);
              for (int i = 0; i < slm_size_per_group; i += gran * sizeof(T))
                slm_block_store<T, gran>(i + group_off, slm_val);
            }

            barrier();

            if constexpr (transpose) {
              simd<T, VS> vals(new_val + elem_off, 1);
              lsc_slm_block_store<T, VS, DS, L1H, L3H>(byte_off, vals);

              barrier();

              auto ret = lsc_slm_block_load<T, VS, DS, L1H, L3H>(byte_off);
              lsc_block_store<T, VS>(out + elem_off, ret);
            } else {
              T val = new_val + elem_off;
              simd<T, VS * VL> vals;
              for (int i = 0; i < VL; i++)
                for (int j = 0; j < VS; j++)
                  vals.template select<1, 1>(i + j * VL) = val++;

              simd<uint16_t, VL> pred = lsc_block_load<uint32_t, VL>(accp, 0);
              simd<uint32_t, VL> offset(byte_off, VS * sizeof(T));

              lsc_slm_scatter<T, VS, DS, L1H, L3H, VL>(offset, vals, pred);

              barrier();

              auto ret = lsc_slm_gather<T, VS, lsc_data_size::default_size,
                                        cache_hint::none, cache_hint::none, VL>(
                  offset);
              lsc_scatter<T, VS, lsc_data_size::default_size, cache_hint::none,
                          cache_hint::none, VL>(out, offset, ret);
            }
          });
    });
    e.wait();
  } catch (cl::sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    sycl::free(out, ctx);
    return false;
  }

  bool passed = true;

  if constexpr (transpose) {
    for (int i = 0; i < Size; i++) {
      T e = new_val + i;
      if (out[i] != e) {
        passed = false;
        std::cout << "out[" << i << "] = 0x" << std::hex << (uint64_t)out[i]
                  << " vs etalon = 0x" << (uint64_t)e << std::dec << std::endl;
      }
    }
  } else {
    for (int i = 0; i < Size; i++) {
      T e = (pmask >> ((i / VS) % VL)) & 1
                ? ((new_val + i) & vmask) | (old_val & ~vmask)
                : old_val;
      if (out[i] != e) {
        passed = false;
        std::cout << "out[" << i << "] = 0x" << std::hex << (uint64_t)out[i]
                  << " vs etalon = 0x" << (uint64_t)e << std::dec << std::endl;
      }
    }
  }

  if (!passed)
    std::cout << "Case #" << case_num << " FAILED" << std::endl;

  sycl::free(out, ctx);

  return passed;
}
