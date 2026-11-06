// Copyright (C) 2026 Tencent.

#ifndef SRC_UTILS_TMA_CUH_
#define SRC_UTILS_TMA_CUH_

#include "cute/tensor.hpp"

namespace hpc {

__device__ __forceinline__ void tma_descriptor_replace_shapes_in_shared_mem(
    cute::TmaDescriptor &smem_desc, cute::array<uint32_t, 5> const &prob_shape) {
#if (__CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 3))
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  uint32_t smem_int_desc = cute::cast_smem_ptr_to_uint(&smem_desc);
  uint64_t const smem_int64_desc = 0;
  asm volatile("cvt.u64.u32 %0, %1;" ::"l"(smem_int64_desc), "r"(smem_int_desc));
  asm volatile(
      "tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 0, %1;" ::"l"(smem_int64_desc),
      "r"(prob_shape[0]));
  asm volatile(
      "tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 1, %1;" ::"l"(smem_int64_desc),
      "r"(prob_shape[1]));
  asm volatile(
      "tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 2, %1;" ::"l"(smem_int64_desc),
      "r"(prob_shape[2]));
  asm volatile(
      "tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 3, %1;" ::"l"(smem_int64_desc),
      "r"(prob_shape[3]));
  asm volatile(
      "tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 4, %1;" ::"l"(smem_int64_desc),
      "r"(prob_shape[4]));
#endif
#endif
}

template <typename Tma, typename GTensor, bool kUpdateShape = true>
__device__ __forceinline__ void update_tma_gtensor(cute::TmaDescriptor &smem_tma_desc,
                                                   const GTensor &gtensor) {
  cute::array<uint32_t, 5> shape{1, 1, 1, 1, 1};
  cute::array<uint64_t, 5> stride{0, 0, 0, 0, 0};

  cute::detail::fill_tma_gmem_shape_stride(Tma{}, gtensor, shape, stride);

  const void *gmem_ptr = gtensor.data().get();
  cute::tma_descriptor_replace_addr_in_shared_mem(smem_tma_desc, gmem_ptr);

  if constexpr (kUpdateShape) {
    tma_descriptor_replace_shapes_in_shared_mem(smem_tma_desc, shape);
  } else {
    // update stride to byte representation
    using T = typename GTensor::value_type;
    for (auto &s : stride) {
      s = (s * cute::sizeof_bits_v<T>) / 8;
    }
    tma_descriptor_replace_dims_strides_in_shared_mem(smem_tma_desc, shape, stride);
  }
}

__device__ __forceinline__ void cp_async_g2s(void *smem_ptr, const void *gmem_ptr, int bytes,
                                             const uint64_t *bar_ptr) {
  int smem_int = cute::cast_smem_ptr_to_uint(smem_ptr);
  int bar_int = cute::cast_smem_ptr_to_uint(bar_ptr);
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(
          smem_int),
      "l"(gmem_ptr), "r"(bytes), "r"(bar_int));
}

// Rewrite a whole-tile descriptor into the shape cp.async.bulk.tensor's tile::gather4 wants.
// rank is 2: dim0 is the contiguous K, dim1 is the row being gathered. sizes and box are element
// counts, strides are bytes and exclude dim0 (whose stride is implicitly the element size). The
// box's dim1 must be 1 -- a hard requirement of gather4, whose row numbers come from the PTX
// {column, row0..row3}; out-of-range rows are zeroed by OOB_FILL_NONE. Swizzle 128B matches
// Layout_K_SW128_Atom.
template <typename Tma, typename T>
void make_tma_gather4_desc(Tma &tma, const T *ptr, uint64_t m, uint64_t k, uint32_t kTileK,
                           CUtensorMapL2promotion l2_promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE) {
  constexpr uint32_t rank = 2;
  uint64_t sizes[rank] = {k, m};
  uint64_t strides[rank - 1] = {k * sizeof(T)};
  uint32_t box_sizes[rank] = {kTileK, 1};
  uint32_t elem_strides[rank] = {1, 1};

  cuTensorMapEncodeTiled(&tma.tma_desc_, CU_TENSOR_MAP_DATA_TYPE_UINT8, rank,
                         const_cast<void *>(reinterpret_cast<const void *>(ptr)), sizes, strides,
                         box_sizes, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                         CU_TENSOR_MAP_SWIZZLE_128B, l2_promotion,
                         CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

// One tile::gather4: four independent row numbers in a single instruction, with the coordinates
// given as {column, row0..row3}. A row number of -1 is out of range, which makes TMA zero-fill
// that row while still counting its bytes on the barrier. kMmaSM == 2 targets the peer CTA's
// shared memory through mapa; kMmaSM == 1 (the only form sm120 can use, since ptxas rejects
// .shared::cluster there) writes this CTA's own.
template <int kMmaSM>
__device__ __forceinline__ void tma_gather4_load(void *smem_dst, const CUtensorMap *tmap, int col,
                                                 int row0, int row1, int row2, int row3,
                                                 uint64_t *mbar, uint64_t cache_hint) {
  // cute's accessor rather than __cvta_generic_to_shared: this header is also pulled into
  // host-only .cc translation units, where the intrinsic is not declared.
  uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_dst);
  uint64_t tmap_addr = reinterpret_cast<uint64_t>(tmap);
  uint32_t mbar_addr = cute::cast_smem_ptr_to_uint(mbar);

  if constexpr (kMmaSM == 2) {
    asm volatile(
        "{\n\t"
        ".reg .b32 remAddr32;\n\t"
        "mapa.shared::cluster.u32  remAddr32, %7, 0;\n\t"
        "cp.async.bulk.tensor.2d.cta_group::2.tile::gather4.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3, %4, %5, %6}], "
        "[remAddr32], %8;\n\t"
        "}"
        :
        : "r"(smem_addr), "l"(tmap_addr), "r"(col), "r"(row0), "r"(row1), "r"(row2), "r"(row3),
          "r"(mbar_addr), "l"(cache_hint)
        : "memory");
  } else {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
        ".mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;\n"
        :
        : "r"(smem_addr), "l"(tmap_addr), "r"(col), "r"(row0), "r"(row1), "r"(row2), "r"(row3),
          "r"(mbar_addr), "l"(cache_hint)
        : "memory");
  }
}

}  // namespace hpc

#endif  // SRC_UTILS_TMA_CUH_
