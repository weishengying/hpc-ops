// Copyright (C) 2026 Tencent.

#ifndef SRC_FUSE_MOE_SM120_MXFP8_GATE_UP_FUSE_ACT_MUL_QUANT_CUH_
#define SRC_FUSE_MOE_SM120_MXFP8_GATE_UP_FUSE_ACT_MUL_QUANT_CUH_

#include <cuda.h>
#include <cuda_fp8.h>

#include "cute/atom/mma_traits_sm120.hpp"
#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/fast_math.h"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace fuse_moe {
namespace kernels {
namespace mxfp8 {

__device__ __forceinline__ void get_next_tile_horizon(const int *tiles_ptr, int iblock,
                                                      int num_group, int &igroup, int &itile_n,
                                                      int &itile_m, int &itile_mg, int &sum_tile_m,
                                                      cutlass::FastDivmod flat_divider) {
  int num_tile_m;

  flat_divider(itile_mg, itile_n, iblock);
  for (int i = igroup; i < num_group; i++) {
    num_tile_m = tiles_ptr[i];
    sum_tile_m += num_tile_m;
    if (itile_mg < sum_tile_m) {
      igroup = i;
      sum_tile_m = sum_tile_m - num_tile_m;
      itile_m = itile_mg - sum_tile_m;
      return;
    }
  }
  igroup = -1;
}

template <typename Config, int kNumTopk, typename TmaA, typename TmaB, typename TmaY,
          typename TmaSFB>
__global__ void __launch_bounds__(Config::kNumThreads, 1) gate_up_fuse_act_mul_quant_kernel(
    const __grid_constant__ TmaA tma_a, const __grid_constant__ TmaB tma_b,
    const __grid_constant__ TmaY tma_y, const __grid_constant__ TmaSFB tma_sfb,
    typename Config::TSF *y_scale_ptr, const typename Config::TSF *x_scale_ptr,
    const int *seqlens_ptr, const int *cu_seqlens_ptr, const int *tiles_ptr,
    const int *permuted_idx_to_raw_idx_ptr, int num_group, int m_pad, int n, int k, float limit,
    cutlass::FastDivmod tile_n_divider) {
  using namespace cute;  // NOLINT

  using TA = typename Config::TA;
  using TB = typename Config::TB;
  using TC = typename Config::TC;
  using TSF = typename Config::TSF;

  using SLayoutA = typename Config::SLayoutA;
  using SLayoutB = typename Config::SLayoutB;
  using SLayoutC = typename Config::SLayoutC;
  using SLayoutSFA = typename Config::SLayoutSFA;
  using SLayoutSFB = typename Config::SLayoutSFB;
  using SLayoutSFARead = typename Config::SLayoutSFARead;
  using SLayoutSFBRead = typename Config::SLayoutSFBRead;
  using TiledMma = typename Config::TiledMma;
  using TiledCopyA = typename Config::TiledCopyA;
  using TiledCopyB = typename Config::TiledCopyB;

  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;
  constexpr int kTileK = Config::kTileK;
  constexpr int kStageAB = Config::kStageAB;
  constexpr int kStageTask = Config::kStageTask;
  constexpr int kMathThreads = Config::kMathThreads;
  constexpr int kSFBlockM = Config::kSFBlockM;
  constexpr int kSFBlockBytes = Config::kSFBlockBytes;
  constexpr int kSFPerTile = Config::kSFPerTile;

  // idx
  int idx = threadIdx.x;
  int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
  int elected = cute::elect_one_sync();
  bool is_leader_in_block = (iwarp == 0) && elected;
  int iblock = blockIdx.x;

  // barrier
  __shared__ uint64_t shmab_readable[kStageAB];
  __shared__ uint64_t shmab_writable[kStageAB];

  __shared__ uint64_t task_readable[kStageTask];
  __shared__ uint64_t task_writable[kStageTask];
  __shared__ int task_shm[kStageTask][4];

  __shared__ float max_shm[4 * kTileM];

  // shm
  extern __shared__ uint8_t shm_data[] alignas(1024);
  auto *shm_a = reinterpret_cast<TA *>(shm_data);
  auto *shm_b = reinterpret_cast<TB *>(shm_a + cosize(SLayoutA{}));
  auto *shm_y = reinterpret_cast<TC *>(shm_b + cosize(SLayoutB{}));
  auto *shm_sfa = reinterpret_cast<TSF *>(shm_y + cosize(SLayoutC{}));
  auto *shm_sfb = reinterpret_cast<TSF *>(shm_sfa + cosize(SLayoutSFA{}));

  auto sA = make_tensor(make_smem_ptr(shm_a), SLayoutA{});
  auto sB = make_tensor(make_smem_ptr(shm_b), SLayoutB{});
  auto sY = make_tensor(make_smem_ptr(shm_y), SLayoutC{});
  auto sSFA = make_tensor(make_smem_ptr(shm_sfa), SLayoutSFA{});
  auto sSFB = make_tensor(make_smem_ptr(shm_sfb), SLayoutSFB{});
  // gather4's destination wants the un-swizzled offset -- TMA applies the swizzle itself from the
  // descriptor -- so take SLayoutA's inner layout.
  auto sAraw = make_tensor(make_smem_ptr(shm_a), SLayoutA{}.layout_b());

  // gmem
  auto gB = tma_b.get_tma_tensor(make_shape(n, k, num_group));
  auto gSFB = tma_sfb.get_tma_tensor(
      make_shape(Int<4>{}, Int<128>{}, n / kSFBlockM, k / kTileK, num_group));

  // tma
  auto btma_b = tma_b.get_slice(0);
  auto btma_sfb = tma_sfb.get_slice(0);

  auto tBg = btma_b.partition_S(gB);  // (V, N, K, G)
  auto tBs = btma_b.partition_D(sB);  // (V, N, K, kStageAB)

  auto tSFBg = btma_sfb.partition_S(gSFB);  // (V, 1, 1, N / 128, K, G)
  auto tSFBs = btma_sfb.partition_D(sSFB);  // (V, 1, 1, kStageAB)

  int num_tile_k = k / kTileK;

  // B counts gmem bytes: 4 bits per fp4 element, same as kTmaTransBytesW in
  // src/group_gemm/sm100/mxfp8/group_gemm_mxfp8.cuh.
  constexpr int kTransactionBytes =
      (sizeof_bits_v<TA> * cosize(SLayoutA{}(_, _, 0)) +                   // NOLINT
       sizeof_bits_v<typename Config::TW> * cosize(SLayoutB{}(_, _, 0)) +  // NOLINT
       sizeof_bits_v<TSF> * cosize(SLayoutSFB{}(_, _, 0))) /               // NOLINT
      8;

  constexpr int kNumMmaWarps = Config::kNumMmaWarps;

  // barrier
  if (is_leader_in_block) {
#pragma unroll
    for (int i = 0; i < kStageAB; ++i) {
      initialize_barrier(shmab_readable[i], 1 + 32);
      initialize_barrier(shmab_writable[i], kNumMmaWarps * 32);
    }
#pragma unroll
    for (int i = 0; i < kStageTask; ++i) {
      initialize_barrier(task_readable[i], 1);
      initialize_barrier(task_writable[i], kNumMmaWarps * 32 + 1);
    }

    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();

  // PDL
  cudaGridDependencySynchronize();

  if (idx >= kMathThreads) {
    cutlass::arch::warpgroup_reg_dealloc<Config::kLoadRegisters>();
  }

  // main
  if (iwarp == 8) {
    // scheduler warp
    if (elected) {
      int istage_task = 0;
      int phase_task = 1;

      int igroup = 0;
      int itile_m = 0;
      int itile_n = 0;
      int itile_mg = 0;
      int sum_tile_m = 0;

      while (true) {
        wait_barrier(task_writable[istage_task], phase_task);

        iblock += gridDim.x;
        get_next_tile_horizon(tiles_ptr, iblock, num_group, igroup, itile_n, itile_m, itile_mg,
                              sum_tile_m, tile_n_divider);
        task_shm[istage_task][0] = igroup;
        task_shm[istage_task][1] = itile_m;
        task_shm[istage_task][2] = itile_n;
        task_shm[istage_task][3] = itile_mg;

        arrive_barrier(task_readable[istage_task]);
        if (igroup < 0) {
          break;
        }
        if (++istage_task == kStageTask) {
          istage_task = 0;
          phase_task ^= 1;
        }
      }  // while
    }
  } else if (iwarp == 9) {
    // tma warp
    int ilane = idx % 32;
    int ismem_write = 0;
    int phase = 1;
    int istage_task = 0;
    int phase_task = 0;
    int num_sf_k = k / Config::kSFVecSize;

    constexpr int kGather4Rows = Config::kGather4Rows;
    const auto *desc_a = reinterpret_cast<const CUtensorMap *>(tma_a.get_tma_descriptor());
    bool issuing_a = ilane < kTileM / kGather4Rows;

    int igroup = 0;
    int itile_m = 0;
    int itile_n = 0;
    int itile_mg = 0;
    int sum_tile_m = 0;
    get_next_tile_horizon(tiles_ptr, iblock, num_group, igroup, itile_n, itile_m, itile_mg,
                          sum_tile_m, tile_n_divider);

    while (igroup >= 0) {
      int seqlen = seqlens_ptr[igroup];
      int cu_seqlen = cu_seqlens_ptr[igroup];
      int itoken[kGather4Rows];
#pragma unroll
      for (int i = 0; i < kGather4Rows; i++) {
        int irow = itile_m * kTileM + ilane * kGather4Rows + i;
        int route =
            (issuing_a && irow < seqlen) ? permuted_idx_to_raw_idx_ptr[cu_seqlen + irow] : -1;
        itoken[i] = route >= 0 ? route / kNumTopk : -1;
      }

#pragma unroll 1
      for (int itile_k = 0; itile_k < num_tile_k; itile_k++) {
        wait_barrier(shmab_writable[ismem_write], phase);

        // load B + SFB
        if (elected) {
          set_barrier_transaction_bytes(shmab_readable[ismem_write], kTransactionBytes);
          cute::copy(tma_b.with(shmab_readable[ismem_write], 0, TMA::CacheHintSm90::EVICT_FIRST),
                     tBg(_, itile_n, itile_k, igroup), tBs(_, 0, 0, ismem_write));
          cute::copy(tma_sfb.with(shmab_readable[ismem_write], 0, TMA::CacheHintSm90::EVICT_FIRST),
                     tSFBg(_, 0, 0, itile_n, itile_k, igroup), tSFBs(_, 0, 0, ismem_write));
        }

        // load A + SFA
        if (issuing_a) {
          tma_gather4_load<1>(&sAraw(ilane * kGather4Rows, 0, ismem_write), desc_a,
                              itile_k * kTileK, itoken[0], itoken[1], itoken[2], itoken[3],
                              &shmab_readable[ismem_write],
                              static_cast<uint64_t>(TMA::CacheHintSm90::EVICT_NORMAL));
#pragma unroll
          for (int i = 0; i < kGather4Rows; i++) {
            const auto *src = x_scale_ptr +
                              static_cast<int64_t>(itoken[i] >= 0 ? itoken[i] : 0) * num_sf_k +
                              itile_k * kSFPerTile;
            cp_async_4b(&sSFA(ilane * kGather4Rows + i, 0, ismem_write), src,
                        itoken[i] >= 0 ? kSFPerTile : 0);
          }
        }

        cutlass::arch::cpasync_barrier_arrive_noinc(&shmab_readable[ismem_write]);

        ++ismem_write;
        if (ismem_write == kStageAB) {
          ismem_write = 0;
          phase ^= 1;
        }
      }  // for itile_k

      // fetch the next task
      wait_barrier(task_readable[istage_task], phase_task);
      igroup = task_shm[istage_task][0];
      itile_m = task_shm[istage_task][1];
      itile_n = task_shm[istage_task][2];
      itile_mg = task_shm[istage_task][3];
      if (elected) {
        arrive_barrier(task_writable[istage_task]);
      }
      if (++istage_task == kStageTask) {
        istage_task = 0;
        phase_task ^= 1;
      }
    }  // while
  } else if (iwarp < 8) {
    // MMA warp
    cutlass::arch::warpgroup_reg_alloc<Config::kMathRegisters>();

    int ismem_read = 0;
    int phase_shmab = 0;

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_slice(idx);
    auto tCrA = thr_mma.partition_fragment_A(sA(_, _, 0));
    auto tCrB = thr_mma.partition_fragment_B(sB(_, _, 0));
    auto acc = partition_fragment_C(tiled_mma, Shape<Int<kTileM>, Int<kTileN>>{});
    auto tIC = thr_mma.partition_C(make_identity_tensor(Shape<Int<kTileM>, Int<kTileN>>{}));
    CUTE_STATIC_ASSERT_V(size(acc) == Int<kTileM * kTileN / (kNumMmaWarps * 32)>{});

    TiledCopyA tiled_copy_a;
    TiledCopyB tiled_copy_b;
    auto thr_copy_a = tiled_copy_a.get_slice(idx);
    auto thr_copy_b = tiled_copy_b.get_slice(idx);

    auto tAsA = thr_copy_a.partition_S(sA);
    auto tBsB = thr_copy_b.partition_S(sB);

    auto tArA = thr_copy_a.retile_D(tCrA);
    auto tBrB = thr_copy_b.retile_D(tCrB);

    constexpr int kNumM = size<1>(decltype(tCrA){});
    constexpr int kNumN = size<1>(decltype(tCrB){});
    int ilane = idx % 32;
    int warp_m = idx / 32 / 4;
    int warp_n = idx / 32 % 4;
    auto sSFAr = make_tensor(make_smem_ptr(shm_sfa), SLayoutSFARead{});
    auto sSFBr = make_tensor(make_smem_ptr(shm_sfb), SLayoutSFBRead{});

    int sfa_row[kNumM];
    int sfb_row[kNumN];
#pragma unroll
    for (int im = 0; im < kNumM; im++) {
      sfa_row[im] = im * 32 + warp_m * 16 + ilane / 4 + (ilane % 2) * 8;
    }
#pragma unroll
    for (int in = 0; in < kNumN; in++) {
      sfb_row[in] = warp_n * 32 + in * 8 + ilane / 4;
    }

    // y carries n / 2 / 32 scales per row.
    int num_sf_y = n / 2 / Config::kSFVecSize;

    int istage_task = 0;
    int phase_task = 0;

    int igroup = 0;
    int itile_m = 0;
    int itile_n = 0;
    int itile_mg = 0;
    int sum_tile_m = 0;
    get_next_tile_horizon(tiles_ptr, iblock, num_group, igroup, itile_n, itile_m, itile_mg,
                          sum_tile_m, tile_n_divider);

    while (igroup >= 0) {
      clear(acc);

      for (int itile_k = 0; itile_k < num_tile_k; itile_k++) {
        // wait tma load ready
        wait_barrier(shmab_readable[ismem_read], phase_shmab);

        uint32_t sfa4[kNumM];
        uint32_t sfb4[kNumN];

        // load sfa
#pragma unroll
        for (int im = 0; im < kNumM; im++) {
          sfa4[im] = *reinterpret_cast<const uint32_t *>(&sSFAr(sfa_row[im], 0, ismem_read));
        }
#pragma unroll
        for (int in = 0; in < kNumN; in++) {
          sfb4[in] = *reinterpret_cast<const uint32_t *>(
              &sSFBr(sfb_row[in] % 32, sfb_row[in] / 32 * 4, ismem_read));
        }

        // load sfb
#pragma unroll
        for (int ik = 0; ik < kTileK / 32; ik++) {
          cute::copy(tiled_copy_a, tAsA(_, _, ik, ismem_read), tArA(_, _, ik));
          cute::copy(tiled_copy_b, tBsB(_, _, ik, ismem_read), tBrB(_, _, ik));
          if constexpr (Config::kIsFp4W) {
            // ldmatrix .b4x16_p64 puts each fp4 in a byte's low 4 bits, mxf8f6f4 wants it in the
            // middle 4: shift left by 2 (same as cute's fp4_shift_B).
            cute::transform(recast<uint32_t>(tCrB(_, _, ik)), [](uint32_t &v) { return v << 2; });
          }
          // Every (im, in) has its own scales, so they are set per MMA.
#pragma unroll
          for (int im = 0; im < kNumM; im++) {
#pragma unroll
            for (int in = 0; in < kNumN; in++) {
              cute::gemm(tiled_mma.with(sfa4[im], sfb4[in], ik), tCrA(_, im, ik), tCrB(_, in, ik),
                         acc(_, im, in));
            }
          }
        }

        // tell TMA this stage may be overwritten
        cutlass::arch::fence_view_async_shared();
        arrive_barrier(shmab_writable[ismem_read]);

        ++ismem_read;
        if (ismem_read == kStageAB) {
          ismem_read = 0;
          phase_shmab ^= 1;
        }
      }  // for itile_k

      // epilogue: act = silu(gate) * up, where in = 0,1 are gate columns and in = 2,3 the matching
      // up columns (+16). acc's v = 0,1 sit on row q and v = 2,3 on row q + 8.
      float act[4][kNumM][2];
#pragma unroll
      for (int im = 0; im < kNumM; im++) {
#pragma unroll
        for (int j = 0; j < 2; j++) {
#pragma unroll
          for (int v = 0; v < 4; v++) {
            float g = fminf(limit, acc(v, im, j));
            float u = fminf(fmaxf(acc(v, im, j + 2), -limit), limit);
            act[v][im][j] = silu(g) * u;
          }
        }
      }

      // One quantization group is 32 act values per row: this warp's 16 columns reduce within the
      // quad first, then merge with the neighbouring warp (warp_n ^ 1).
      float amax[kNumM][2];
#pragma unroll
      for (int im = 0; im < kNumM; im++) {
#pragma unroll
        for (int h = 0; h < 2; h++) {
          float v = fmaxf(fmaxf(fabsf(act[2 * h][im][0]), fabsf(act[2 * h + 1][im][0])),
                          fmaxf(fabsf(act[2 * h][im][1]), fabsf(act[2 * h + 1][im][1])));
          v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFF, v, 1));
          v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFF, v, 2));
          amax[im][h] = v;
          if (ilane % 4 == 0) {
            max_shm[warp_n * kTileM + get<0>(tIC(2 * h, im, 0))] = v;
          }
        }
      }
      cutlass::arch::NamedBarrier::sync(kNumMmaWarps * 32, 1);

#pragma unroll
      for (int im = 0; im < kNumM; im++) {
#pragma unroll
        for (int h = 0; h < 2; h++) {
          int row = get<0>(tIC(2 * h, im, 0));
          float row_amax = fmaxf(amax[im][h], max_shm[(warp_n ^ 1) * kTileM + row]);

          constexpr float kE4m3MaxInv = 1.f / 448.f;
          __nv_fp8_e8m0 sf;
          sf.__x = __nv_cvt_float_to_e8m0(row_amax * kE4m3MaxInv, __NV_SATFINITE, cudaRoundPosInf);
          float sfv = static_cast<float>(sf);
          float inv = (sfv != 0.f) ? rcpf_ftz(sfv) : 0.f;

          // Each quantization group's scale is written by the even-warp_n warps.
          // y_scale is layout_128x4: row r's c-th scale sits in block (r / 128, c / 4) at
          // (r % 32) * 16 + (r % 128 / 32) * 4 + c % 4.
          if (warp_n % 2 == 0 && ilane % 4 == 0) {
            int64_t r = static_cast<int64_t>(itile_mg) * kTileM + row;
            int c = itile_n * 2 + warp_n / 2;
            int64_t iblk = r / kSFBlockM * (num_sf_y / Config::kSFPerTile) + c / Config::kSFPerTile;
            reinterpret_cast<uint8_t *>(
                y_scale_ptr)[iblk * kSFBlockBytes + r % 32 * 16 + r % kSFBlockM / 32 * 4 +
                             c % Config::kSFPerTile] = sf.__x;
          }

#pragma unroll
          for (int j = 0; j < 2; j++) {
            // The gate column c's position within act: the first 16 of every 32 columns.
            int c = get<1>(tIC(2 * h, im, j));
            int icol = c / 32 * 16 + c % 16;
            __nv_fp8x2_storage_t q = __nv_cvt_float2_to_fp8x2(
                make_float2(act[2 * h][im][j] * inv, act[2 * h + 1][im][j] * inv), __NV_SATFINITE,
                __NV_E4M3);
            *reinterpret_cast<__nv_fp8x2_storage_t *>(&sY(row, icol)) = q;
          }
        }
      }
      tma_store_fence();
      cutlass::arch::NamedBarrier::sync(kNumMmaWarps * 32, 1);
      if (idx == 0) {
        auto gY = tma_y.get_tma_tensor(make_shape(m_pad, n / 2));
        auto btma_y = tma_y.get_slice(0);
        auto tYg = btma_y.partition_S(gY);
        auto tYs = btma_y.partition_D(sY);
        cute::copy(tma_y, tYs(_, 0, 0), tYg(_, itile_mg, itile_n));
        tma_store_arrive();
        tma_store_wait<0>();
      }
      // Without this some warps race ahead into the next task.
      cutlass::arch::NamedBarrier::sync(kNumMmaWarps * 32, 1);

      // fetch the next task
      wait_barrier(task_readable[istage_task], phase_task);
      igroup = task_shm[istage_task][0];
      itile_m = task_shm[istage_task][1];
      itile_n = task_shm[istage_task][2];
      itile_mg = task_shm[istage_task][3];
      arrive_barrier(task_writable[istage_task]);
      if (++istage_task == kStageTask) {
        istage_task = 0;
        phase_task ^= 1;
      }
    }
  }  // mma warp

  cudaTriggerProgrammaticLaunchCompletion();
}

}  // namespace mxfp8
}  // namespace kernels
}  // namespace fuse_moe
}  // namespace hpc

#endif  // SRC_FUSE_MOE_SM120_MXFP8_GATE_UP_FUSE_ACT_MUL_QUANT_CUH_
