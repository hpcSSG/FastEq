/***************************************************************************************************
 * fasteq eqlinear backends/fp64/collective/sm90_mma_fp64_array_tma_warpspecialized.hpp
 *
 * M2: SM90 grouped (ptr-array) warp-specialised FP64 mainloop.  TMA lands double tiles of A and B
 * in shared memory; the MMA warpgroup copies both operands into registers and issues the composite
 * 32x32x16 DMMA atom (2x4 sub-atoms of mma.sync.m16n8k16.f64).
 *
 * Derived from CUTLASS's
 *   include/cutlass/gemm/collective/sm90_mma_tma_gmma_rs_warpspecialized.hpp
 * (CUTLASS v4.8.0, BSD-3-Clause, Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES).
 * The pipeline / scheduler / epilogue contracts are upstream's, untouched; the differences are:
 *
 *  1. Both operands are register-sourced (DMMA has no smem operand descriptor), so there are two
 *     smem->register copies per K-tile.  FP64's register layout makes those 8-byte LDS accesses;
 *     `ldmatrix` does not cover 64-bit elements.
 *  2. No tf32 split and no B_res: A and B are plain double tensors, two TMA descriptors per SM,
 *     and TmaTransactionBytes counts A + B.
 *  3. The TiledMma is the composite `Op32x32x16` atom (fp64_dmma_atom.hpp), not a GMMA atom; the
 *     stock array kernel only sees that `size(TiledMma) == 256` and the C layout.
 *
 * The host-side operand layouts are the same as M1: A is (M,K) K-major, B is (N,K) K-major
 * (`ldb` = the packed row stride), C/D row-major with the row pointer = out + j*V.
 **************************************************************************************************/
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/trace.h"
#include "cutlass/arch/barrier.h"
#include "cutlass/cuda_host_adapter.hpp"

#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/algorithm/functional.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"

#include "dispatch_policy_fp64.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  int Stages,
  class ClusterShape,
  class KernelSchedule,
  bool ComposedA,
  class TileShape_,
  class ElementA_,
  class StrideA_,
  class ElementB_,
  class StrideB_,
  class TiledMma_,
  class GmemTiledCopyA_,
  class SmemLayoutAtomA_,
  class SmemCopyAtomA_,
  class TransformA_,
  class GmemTiledCopyB_,
  class SmemLayoutAtomB_,
  class SmemCopyAtomB_,
  class TransformB_>
struct CollectiveMma<
    MainloopSm90ArrayTmaDmmaWarpSpecializedFp64<Stages, ClusterShape, KernelSchedule, ComposedA>,
    TileShape_,
    ElementA_,
    StrideA_,
    ElementB_,
    StrideB_,
    TiledMma_,
    GmemTiledCopyA_,
    SmemLayoutAtomA_,
    SmemCopyAtomA_,
    TransformA_,
    GmemTiledCopyB_,
    SmemLayoutAtomB_,
    SmemCopyAtomB_,
    TransformB_>
{
  //
  // Type Aliases
  //
  using DispatchPolicy = MainloopSm90ArrayTmaDmmaWarpSpecializedFp64<Stages, ClusterShape, KernelSchedule, ComposedA>;
  static constexpr bool kComposedA = DispatchPolicy::ComposedA;
  using TileShape = TileShape_;
  using ElementA = ElementA_;
  using StrideA = StrideA_;
  using InternalStrideA = cute::remove_pointer_t<StrideA>;
  using ElementB = ElementB_;
  using StrideB = StrideB_;
  using InternalStrideB = cute::remove_pointer_t<StrideB>;
  using TiledMma = TiledMma_;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ArchTag = typename DispatchPolicy::ArchTag;

  using MainloopPipeline = cutlass::PipelineTmaAsync<DispatchPolicy::Stages>;
  using PipelineState = cutlass::PipelineState<DispatchPolicy::Stages>;

  using PipelineParams = typename MainloopPipeline::Params;
  using CtaShape_MNK = decltype(shape_div(TileShape{}, ClusterShape{}));

  static constexpr int NumProducerThreadEvents = 1;

  /// Threads that run the GMMA mainloop (== size(TiledMma{}), 256 for the 2-warpgroup cooperative
  /// schedule).  `mma()` is entered by exactly these threads, so it doubles as the width of the
  /// split barrier.
  static constexpr int NumMmaThreads = size(TiledMma{});

  static_assert(rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  // Tile along modes in a way that maximizes the TMA box size.
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{}),
      cute::conditional_t< ::cutlass::gemm::detail::is_major<0,StrideA>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{}),
      cute::conditional_t< ::cutlass::gemm::detail::is_major<0,StrideB>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));

  static_assert(DispatchPolicy::Stages >= 2, "Specialization requires Stages set to value 2 or more.");
  // Both operands are register-sourced: DMMA has no smem operand descriptor.  `FrgTypeA/B` being
  // plain value types is what makes `make_fragment_A/B` allocate registers, and the non-void copy
  // atoms are the other half of the pair.
  static_assert(!cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value &&
                !cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeB>::value,
                "The FP64 DMMA mainloop needs register-sourced A and B.");
  static_assert(cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>,
      "GmemTiledCopy - invalid SM90 TMA copy atom specified.");
  static_assert(cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>,
      "GmemTiledCopy - invalid SM90 TMA copy atom specified.");
  static_assert(cute::is_same_v<ElementA, double> && cute::is_same_v<ElementB, double>,
      "The FP64 DMMA mainloop takes plain double operands (no split, no K-concatenation).");

  // A and B land in smem as raw 64-bit words; the TMA must not reinterpret them.
  static constexpr bool ConvertF32toTF32A = false;
  static constexpr bool ConvertF32toTF32B = false;
  using InternalElementA = uint_bit_t<sizeof_bits_v<ElementA>>;
  using InternalElementB = uint_bit_t<sizeof_bits_v<ElementB>>;

  // Assumption: StrideA is congruent with Problem_MK
  using TMA_A = decltype(make_tma_copy(
      GmemTiledCopyA{},
      make_tensor(static_cast<InternalElementA const*>(nullptr), repeat_like(InternalStrideA{}, int32_t(0)), InternalStrideA{}),
      SmemLayoutA{}(_,_,Int<0>{}),
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
      size<1>(ClusterShape{})));  // mcast along N mode for this M load, if any
  // Assumption: StrideB is congruent with Problem_NK
  using TMA_B = decltype(make_tma_copy(
      GmemTiledCopyB{},
      make_tensor(static_cast<InternalElementB const*>(nullptr), repeat_like(InternalStrideB{}, int32_t(0)), InternalStrideB{}),
      SmemLayoutB{}(_,_,Int<0>{}),
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),
      size<0>(ClusterShape{}))); // mcast along M mode for this N load, if any

  struct SharedStorage {
    struct TensorStorage : cute::aligned_struct<128, _0> {
      cute::array_aligned<typename TiledMma::ValTypeA, cute::cosize_v<SmemLayoutA>> smem_A;
      cute::array_aligned<typename TiledMma::ValTypeB, cute::cosize_v<SmemLayoutB>> smem_B;
    } tensors;

    struct TensorMapStorage : cute::aligned_struct<128, _0> {
      cute::TmaDescriptor smem_tensormap_A;
      cute::TmaDescriptor smem_tensormap_B;
    } tensormaps;

    using PipelineStorage = typename MainloopPipeline::SharedStorage;
    PipelineStorage pipeline;
  };
  using TensorStorage = typename SharedStorage::TensorStorage;
  using TensorMapStorage = typename SharedStorage::TensorMapStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;

  static constexpr bool IsGroupedGemmKernel = !cute::is_same_v<InternalStrideA, StrideA>;

  // Host side kernel arguments
  struct Arguments {
    ElementA const** ptr_A;  // one fp64 copy
    StrideA dA;
    ElementB const** ptr_B;  // fp64 B, K-major
    StrideB dB;
    /// Composed-A only: the per-path K extent (U) and the largest path count of any group.
    /// The flat mainloop ignores both.
    int u = 0;
    int m_max = 1;
  };

  // L2 residency hints for the two operands, selected at compile time by EQLIN_TMA_HINT.
  // A is streamed once per group and never revisited; B is re-read by every M-tile of the same
  // group, so the two want opposite treatment.  The hint is a runtime operand on the TMA
  // instruction rather than a tensormap field, so leaving EQLIN_TMA_HINT at 0 costs nothing.
  // These are accessors rather than constants because `with()` binds the hint by reference, and
  // taking the address of a host `constexpr` variable from device code is not allowed.
  CUTLASS_HOST_DEVICE static constexpr TMA::CacheHintSm90 cache_hint_a() {
#if EQLIN_TMA_HINT
    return TMA::CacheHintSm90::EVICT_FIRST;
#else
    return TMA::CacheHintSm90::EVICT_NORMAL;
#endif
  }
  CUTLASS_HOST_DEVICE static constexpr TMA::CacheHintSm90 cache_hint_b() {
#if EQLIN_TMA_HINT
    return TMA::CacheHintSm90::EVICT_LAST;
#else
    return TMA::CacheHintSm90::EVICT_NORMAL;
#endif
  }

  // Device side kernel params
  struct Params {
    TMA_A tma_load_a;
    TMA_B tma_load_b;
    uint32_t tma_transaction_bytes = TmaTransactionBytes;
    void* tensormaps;
    InternalElementA const** ptr_A;
    StrideA dA;
    InternalElementB const** ptr_B;
    StrideB dB;
    int u = 0;
    int m_max = 1;
  };

  //
  // Methods
  //

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
      ProblemShape problem_shapes,
      Arguments const& args,
      void* workspace) {
    // These tensor shapes (only applicable for grouped gemm) and pointers are only used to create tensormap/tma desc.
    // These will be replaced with correct values before the initial tma load.
    auto init_shape = repeat_like(typename ProblemShape::UnderlyingProblemShape{}, int32_t(1));
    auto init_M = get<0>(init_shape);
    auto init_N = get<1>(init_shape);
    auto init_K = get<2>(init_shape);
    // Batches/Groups are managed by using appropriate pointers to input matrices
    const uint32_t init_L = 1;
    // NOTE: Since TMA desc creation with nullptr not possible until 12.6, we use an initial address even when tensor addresses are on device. This address is never used.
    InternalElementA const* ptr_A_first_batch = reinterpret_cast<InternalElementA const*>(reinterpret_cast<uint64_t>(args.ptr_A) & 0xFFFFFFFFFFFFFFF0);  // Address must be 16B-aligned
    InternalElementB const* ptr_B_first_batch = reinterpret_cast<InternalElementB const*>(reinterpret_cast<uint64_t>(args.ptr_B) & 0xFFFFFFFFFFFFFFF0);  // Address must be 16B-aligned

    InternalStrideA stride_a;
    InternalStrideB stride_b;
    if constexpr (IsGroupedGemmKernel) {
      // Strides for Grouped Gemm will be replaced prior to the first access regardless.
      stride_a = InternalStrideA{};
      stride_b = InternalStrideB{};
    }
    else {
      // Tensor shapes for Ptr-Array are initialized correctly only here.
      auto problem_shape_MNK = problem_shapes.get_host_problem_shape(0);
      init_M = get<0>(problem_shape_MNK);
      init_N = get<1>(problem_shape_MNK);
      init_K = get<2>(problem_shape_MNK);

      stride_a = args.dA;
      stride_b = args.dB;
    }
    Tensor tensor_b = make_tensor(ptr_B_first_batch, make_layout(make_shape(init_N,init_K,init_L), stride_b));
    // The initial A tensor is (M, K, L) with unit extents for both flavours; the *stride type*
    // (flat K vs composed (U, m)) is a template parameter, so the descriptor type stays the same
    // for this dummy and for the per-group updates.
    Tensor tensor_a = make_tensor(ptr_A_first_batch, make_layout(make_shape(init_M,init_K,init_L), stride_a));
    TMA_A tma_load_a = make_tma_copy(
        GmemTiledCopyA{},
        tensor_a,
        SmemLayoutA{}(_,_,Int<0>{}),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
        size<1>(ClusterShape{})); // mcast along N mode for this M load, if any
    TMA_B tma_load_b = make_tma_copy(
        GmemTiledCopyB{},
        tensor_b,
        SmemLayoutB{}(_,_,Int<0>{}),
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),
        size<0>(ClusterShape{})); // mcast along M mode for this N load, if any

    void* tensormaps = workspace;

    return {
      tma_load_a,
      tma_load_b,
      TmaTransactionBytes,
      tensormaps,
      reinterpret_cast<InternalElementA const**>(args.ptr_A),
      args.dA,
      reinterpret_cast<InternalElementB const**>(args.ptr_B),
      args.dB,
      args.u,
      args.m_max
    };
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args, int sm_count) {
    // Two input tensormaps (A, B) per SM.
    constexpr uint32_t NumInputTensors = 2;
    constexpr size_t SizeOfCuTensorMap = sizeof(cute::TmaDescriptor);
    // Allocate gmem space for input tensormaps per each SM, A tensormap copies followed by B tensormap copies
    return (NumInputTensors * SizeOfCuTensorMap * sm_count);
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream, CudaHostAdapter* cuda_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }

  template<class ProblemShape>
  static bool
  can_implement(
      ProblemShape problem_shapes,
      Arguments const& args) {
    constexpr int tma_alignment_bits = 128;
    constexpr int min_tma_aligned_elements_A = tma_alignment_bits / cutlass::sizeof_bits<ElementA>::value;
    constexpr int min_tma_aligned_elements_B = tma_alignment_bits / cutlass::sizeof_bits<ElementB>::value;

    bool implementable = true;
    if (problem_shapes.is_host_problem_shape_available()) {
      // Check alignment for all problem sizes
      for (int i = 0; i < problem_shapes.groups(); i++) {
        auto problem_shape_MNKL = append<4>(problem_shapes.get_host_problem_shape(i), 1);
        auto [M,N,K,L] = problem_shape_MNKL;
        implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_A>(cute::make_shape(M,K,L), InternalStrideA{});
        implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_B>(cute::make_shape(N,K,L), InternalStrideB{});
      }
    }

    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for TMA.\n");
    }
    return implementable;
  }

  static constexpr int K_PIPE_MAX = DispatchPolicy::Stages;
  static constexpr int K_PIPE_MMAS = 1;
  // A + B bytes per stage.
  static constexpr uint32_t TmaTransactionBytes =
        cutlass::bits_to_bytes(size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) * static_cast<uint32_t>(sizeof_bits<ElementA>::value))+
        cutlass::bits_to_bytes(size<0>(SmemLayoutB{}) * size<1>(SmemLayoutB{}) * static_cast<uint32_t>(sizeof_bits<ElementB>::value));

  // Set up the data needed by this collective for load and mma.
  // Returns a tuple of tensors. The collective and the kernel layer have the contract that the
  // returned tuple must contain at least two elements, with the first two elements being:
  // gA_mkl - The tma tensor, A after a local tile so it has shape  (BLK_M,BLK_K,m,k,l)
  // gB_nkl - The tma tensor, B after a local tile so it has shape  (BLK_N,BLK_K,n,k,l)
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto
  load_init(ProblemShape_MNKL const& problem_shape_MNKL, Params const& mainloop_params) const {
    using X = Underscore;
    // Separate out problem shape for convenience
    auto [M,N,K,L] = problem_shape_MNKL;
    const int32_t init_L = 1;

    // TMA requires special handling of strides to deal with coord codomain mapping
    // Represent the full tensors -- get these from TMA
    Tensor mB_nkl = mainloop_params.tma_load_b.get_tma_tensor(make_shape(N,K,init_L));                            // (n,k,l)

    // Make tiled views, defer the slice
    Tensor gB_nkl = local_tile(mB_nkl, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});  // (BLK_N,BLK_K,n,k,l)

    if constexpr (kComposedA) {
      // A is (M, U, m): the path index is the batch mode.  The flat K = m*U tiles are decoded in
      // `load` into (m_path, u_tile); this view keeps (BLK_M, BLK_K, 1, M_tiles, U_tiles, m).
      Tensor mA_mum = mainloop_params.tma_load_a.get_tma_tensor(
          make_shape(M, mainloop_params.u, mainloop_params.m_max));
      Tensor gA = local_tile(mA_mum,
                             make_tile(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<1>{}),
                             make_coord(_,_,_));
      return cute::make_tuple(gA, gB_nkl);
    } else {
      Tensor mA_mkl = mainloop_params.tma_load_a.get_tma_tensor(make_shape(M,K,init_L));                          // (m,k,l)
      Tensor gA_mkl = local_tile(mA_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});  // (BLK_M,BLK_K,m,k,l)
      return cute::make_tuple(gA_mkl, gB_nkl);
    }
  }

  // Perform a collective-scoped matrix multiply-accumulate
  // Producer Perspective
  template <
    class TensorA, class TensorB,
    class TensorMapA, class TensorMapB,
    class KTileIterator, class BlockCoord
  >
  CUTLASS_DEVICE void
  load(
      Params const& mainloop_params,
      MainloopPipeline pipeline,
      PipelineState smem_pipe_write,
      cute::tuple<TensorA, TensorB> const& load_inputs,
      cute::tuple<TensorMapA, TensorMapB> const& input_tensormaps,
      BlockCoord const& blk_coord,
      KTileIterator k_tile_iter, int k_tile_count,
      int thread_idx,
      uint32_t block_rank_in_cluster,
      TensorStorage& shared_tensors) {
    int lane_predicate = cute::elect_one_sync();

    if (lane_predicate) {
      Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});        // (BLK_M,BLK_K,PIPE)
      Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{});        // (BLK_N,BLK_K,PIPE)

      //
      // Prepare the TMA loads for A and B
      //

      constexpr uint32_t cluster_shape_x = get<0>(typename DispatchPolicy::ClusterShape());
      uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};

      Tensor gA_mkl = get<0>(load_inputs);
      Tensor gB_nkl = get<1>(load_inputs);

      auto block_tma_a = mainloop_params.tma_load_a.get_slice(cluster_local_block_id.y);
      auto block_tma_b = mainloop_params.tma_load_b.get_slice(cluster_local_block_id.x);

      // Partition the inputs based on the current block coordinates.
      auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;
      Tensor gB = gB_nkl(_,_,n_coord,_,l_coord);                                                     // (BLK_N,BLK_K,k)

      Tensor tAsA = block_tma_a.partition_D(sA);                                              // (TMA,TMA_M,TMA_K,PIPE)
      Tensor tBgB = block_tma_b.partition_S(gB);                                                 // (TMA,TMA_N,TMA_K,k)
      Tensor tBsB = block_tma_b.partition_D(sB);                                              // (TMA,TMA_N,TMA_K,PIPE)

      uint16_t mcast_mask_a = 0;
      uint16_t mcast_mask_b = 0;

      // Issue TmaLoads
      // Maps the tile -> block, value
      if constexpr (cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>) {
        auto block_layout = Layout<typename DispatchPolicy::ClusterShape>{}; // (m,n) -> block_id
        for (int n = 0; n < size<1>(block_layout); ++n) {
          mcast_mask_a |= (uint16_t(1) << block_layout(cluster_local_block_id.x,n,Int<0>{}));
        }
      }

      if constexpr (cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>) {
        auto block_layout = Layout<typename DispatchPolicy::ClusterShape>{}; // (m,n) -> block_id
        for (int m = 0; m < size<0>(block_layout); ++m) {
          mcast_mask_b |= (uint16_t(1) << block_layout(m,cluster_local_block_id.y,Int<0>{}));
        }
      }

      if constexpr (kComposedA) {
        // Flat K = m*U tiles; tile t belongs to path t/(U/BK), u-slice t%(U/BK).  The A source
        // view carries the path in its batch (L) mode, so the descriptor is still rank 3.
        int const u_tiles = mainloop_params.u / int(size<2>(TileShape{}));
        CUTLASS_PRAGMA_NO_UNROLL
        for ( ; k_tile_count > 0; --k_tile_count)
        {
          pipeline.producer_acquire(smem_pipe_write);
          using BarrierType = typename MainloopPipeline::ProducerBarrierType;
          BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);
          int write_stage = smem_pipe_write.index();

          int const k = *k_tile_iter;
          int const m_path = k / u_tiles;
          int const u_tile = k % u_tiles;
          Tensor gA = gA_mkl(_,_,Int<0>{},m_coord,u_tile,m_path);                               // (BLK_M,BLK_K)
          Tensor tAgA = block_tma_a.partition_S(gA);                                            // (TMA,TMA_M,TMA_K)

          copy(mainloop_params.tma_load_a.with(get<0>(input_tensormaps), *tma_barrier, mcast_mask_a, cache_hint_a()), tAgA, tAsA(_,_,_,write_stage));
          copy(mainloop_params.tma_load_b.with(get<1>(input_tensormaps), *tma_barrier, mcast_mask_b, cache_hint_b()), tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,write_stage));
          ++k_tile_iter;
          ++smem_pipe_write;
        }
      } else {
        Tensor gA = gA_mkl(_,_,m_coord,_,l_coord);                                                   // (BLK_M,BLK_K,k)
        Tensor tAgA = block_tma_a.partition_S(gA);                                               // (TMA,TMA_M,TMA_K,k)
        CUTLASS_PRAGMA_NO_UNROLL
        for ( ; k_tile_count > 0; --k_tile_count)
        {
          pipeline.producer_acquire(smem_pipe_write);
          using BarrierType = typename MainloopPipeline::ProducerBarrierType;
          BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);
          int write_stage = smem_pipe_write.index();
          copy(mainloop_params.tma_load_a.with(get<0>(input_tensormaps), *tma_barrier, mcast_mask_a, cache_hint_a()), tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,write_stage));
          copy(mainloop_params.tma_load_b.with(get<1>(input_tensormaps), *tma_barrier, mcast_mask_b, cache_hint_b()), tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,write_stage));
          ++k_tile_iter;
          ++smem_pipe_write;
        }
      }
    }
  }

  // Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
  CUTLASS_DEVICE void
  load_tail(MainloopPipeline pipeline, PipelineState smem_pipe_write) {
    int lane_predicate = cute::elect_one_sync();

    // Issue the epilogue waits
    if (lane_predicate) {
      // This helps avoid early exit of blocks in Cluster.
      // Waits for all stages to either be released (all
      // Consumer UNLOCKs), or if the stage was never used
      // then it would just be acquired since the phase was
      // still inverted from make_producer_start_state.
      pipeline.producer_tail(smem_pipe_write);
    }
  }

  /// Stage one K-tile's A and B into registers.  DMMA has no smem operand descriptor, so both
  /// operands need a smem->register copy; the DMMA fragment layouts make those 8-byte LDS
  /// accesses (`ldmatrix` does not cover 64-bit elements).  `mma.sync` reads its register operands
  /// at issue -- unlike wgmma's asynchronous read -- so a single register buffer per operand is
  /// safe without the drain the tf32 RS mainloop needs.
  template <class SmemTiledCopyA, class SmemThrCopyA, class TensorACopyView,
            class SmemTiledCopyB, class SmemThrCopyB, class TensorBCopyView,
            class TensorA, class TensorB>
  CUTLASS_DEVICE void
  stage_regs(SmemTiledCopyA const& smem_tiled_copy_A,
             SmemThrCopyA   const& smem_thr_copy_A,
             TensorACopyView const& tCsA_copy_view,
             SmemTiledCopyB const& smem_tiled_copy_B,
             SmemThrCopyB   const& smem_thr_copy_B,
             TensorBCopyView const& tCsB_copy_view,
             int             read_stage,
             TensorA&        tCrA,
             TensorB&        tCrB) {
    copy(smem_tiled_copy_A, tCsA_copy_view(_,_,_,read_stage), smem_thr_copy_A.retile_D(tCrA));
    copy(smem_tiled_copy_B, tCsB_copy_view(_,_,_,read_stage), smem_thr_copy_B.retile_D(tCrB));
  }

  /// All of a K-tile's DMMA work.  One `cute::gemm` per K-block; the composite atom expands each
  /// into its 2x4 sub-atom instructions.  The caller does `consumer_release` once afterwards.
  template <class TensorAK, class TensorBK, class FrgTensorC>
  CUTLASS_DEVICE void
  mma_k_tile(TiledMma& tiled_mma,
             TensorAK const& tCrA_k, TensorBK const& tCrB_k,
             FrgTensorC& accum) {
    CUTLASS_PRAGMA_UNROLL
    for (int k_block = 0; k_block < size<2>(tCrA_k); ++k_block) {
      cute::gemm(tiled_mma, tCrA_k(_,_,k_block), tCrB_k(_,_,k_block), accum);
    }
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Consumer Perspective
  template <
    class FrgTensorC
  >
  CUTLASS_DEVICE void
  mma(MainloopPipeline pipeline,
      PipelineState smem_pipe_read,
      FrgTensorC& accum,
      int k_tile_count,
      int thread_idx,
      TensorStorage& shared_tensors,
      Params const& mainloop_params) {
    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
    static_assert(rank(SmemLayoutA{}) == 3, "Smem layout must be rank 3.");
    static_assert(rank(SmemLayoutB{}) == 3, "Smem layout must be rank 3.");
    static_assert(!cute::is_void_v<SmemCopyAtomA>,
      "The FP64 DMMA mainloop sources A from registers, so it needs a smem->register copy atom.");
    static_assert(!cute::is_void_v<SmemCopyAtomB>,
      "The FP64 DMMA mainloop sources B from registers, so it needs a smem->register copy atom.");

    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});          // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{});          // (BLK_N,BLK_K,PIPE)

    TiledMma tiled_mma;
    // DMMA operands are per-thread register fragments; the thread slice (not the warpgroup slice,
    // which is for smem descriptors) is what partitions them.
    auto mma_thread_slice = tiled_mma.get_thread_slice(thread_idx);
    Tensor tCrA_one = mma_thread_slice.partition_fragment_A(sA(_,_,Int<0>{}));                 // (MMA,MMA_M,MMA_K)
    Tensor tCrB_one = mma_thread_slice.partition_fragment_B(sB(_,_,Int<0>{}));                 // (MMA,MMA_N,MMA_K)
    // Both are plain register arrays.  Build them from `shape(tCrA/B_one)` so they inherit the
    // atom's layout: a hand-rolled layout (or a runtime-indexed tensor, which materialises in local
    // memory) is enough to hand the tensor cores their operands in the wrong register order.
    Tensor tCrA = make_tensor<double>(shape(tCrA_one));
    Tensor tCrB = make_tensor<double>(shape(tCrB_one));

    auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(thread_idx);
    Tensor tCsA_copy_view  = smem_thr_copy_A.partition_S(sA);                                 // (CPY,CPY_M,CPY_K,PIPE)
    auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
    auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(thread_idx);
    Tensor tCsB_copy_view  = smem_thr_copy_B.partition_S(sB);                                 // (CPY,CPY_N,CPY_K,PIPE)

    CUTE_STATIC_ASSERT_V(size<1>(tCrA_one) == size<1>(accum));                                 // M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB_one) == size<2>(accum));                                 // N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA_one) == size<2>(tCrB_one));                              // K
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));                        // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));                        // PIPE
    // The register fragments and the smem->register copies must agree mode for mode.
    CUTE_STATIC_ASSERT_V(size<2>(tCsA_copy_view) == size<2>(smem_thr_copy_A.retile_D(tCrA_one)));  // CPY_K
    CUTE_STATIC_ASSERT_V(size<2>(tCsB_copy_view) == size<2>(smem_thr_copy_B.retile_D(tCrB_one)));  // CPY_K

    //
    // PIPELINED MAIN LOOP
    //
    static_assert((0 <= K_PIPE_MMAS) && (K_PIPE_MMAS <  K_PIPE_MAX),
        "ERROR : Incorrect number of MMAs in flight");

    // mma.sync has no ScaleOut::Zero: zero the accumulator before the first K-tile.
    clear(accum);

    // We release buffers to producer warps(dma load) with some mmas in flight
    PipelineState smem_pipe_release = smem_pipe_read;

    // Prologue
    int prologue_mma_count = min(K_PIPE_MMAS, k_tile_count);
    if (k_tile_count > 0) {
      // WAIT on smem_pipe_read until its data are available (phase bit flips from rdPhaseBit value)
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);

      int read_stage = smem_pipe_read.index();
      stage_regs(smem_tiled_copy_A, smem_thr_copy_A, tCsA_copy_view,
                 smem_tiled_copy_B, smem_thr_copy_B, tCsB_copy_view,
                 read_stage, tCrA, tCrB);
      mma_k_tile(tiled_mma, tCrA, tCrB, accum);

      ++smem_pipe_read;
    }

    CUTLASS_PRAGMA_UNROLL
    for (int k_tile_prologue = prologue_mma_count - 1; k_tile_prologue > 0; --k_tile_prologue)
    {
      // WAIT on smem_pipe_read until its data are available (phase bit flips from rdPhaseBit value)
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);

      int read_stage = smem_pipe_read.index();
      stage_regs(smem_tiled_copy_A, smem_thr_copy_A, tCsA_copy_view,
                 smem_tiled_copy_B, smem_thr_copy_B, tCsB_copy_view,
                 read_stage, tCrA, tCrB);
      mma_k_tile(tiled_mma, tCrA, tCrB, accum);

      ++smem_pipe_read;
    }

    // Mainloop
    k_tile_count -= prologue_mma_count;

    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 0; --k_tile_count)
    {
      // WAIT on smem_pipe_read until its data are available (phase bit flips from rdPhaseBit value)
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);

      int read_stage = smem_pipe_read.index();
      stage_regs(smem_tiled_copy_A, smem_thr_copy_A, tCsA_copy_view,
                 smem_tiled_copy_B, smem_thr_copy_B, tCsB_copy_view,
                 read_stage, tCrA, tCrB);
      mma_k_tile(tiled_mma, tCrA, tCrB, accum);

      // UNLOCK smem_pipe_release, done _computing_ on it.  mma.sync is synchronous, so the
      // stage's smem has been read by the time this point is reached.
      pipeline.consumer_release(smem_pipe_release);

      // Advance smem_pipe_read and smem_pipe_release
      ++smem_pipe_read;
      ++smem_pipe_release;
    }
  }

  /// Perform a Consumer Epilogue to release all buffers
  CUTLASS_DEVICE void
  mma_tail(MainloopPipeline pipeline, PipelineState smem_pipe_release, int k_tile_count) {
    // Prologue GMMAs
    int prologue_mma_count = min(K_PIPE_MMAS, k_tile_count);
    k_tile_count -= prologue_mma_count;

    smem_pipe_release.advance(k_tile_count);

    // mma.sync is synchronous; there is no outstanding MMA batch to drain.
    for (int count = 0; count < prologue_mma_count; ++count) {
      pipeline.consumer_release(smem_pipe_release);                 // UNLOCK smem_pipe_release, done _computing_ on it
      ++smem_pipe_release;
    }
  }

  //
  // Methods to perform different parts of TMA/Tensormap modifications
  //

  CUTLASS_DEVICE auto
  tensormaps_init(
      Params const& mainloop_params,
      TensorMapStorage& shared_tensormaps,
      int32_t sm_count,
      int32_t sm_idx) {
    cute::TmaDescriptor* gmem_tensormap = reinterpret_cast<cute::TmaDescriptor*>(mainloop_params.tensormaps);

    cute::TmaDescriptor* tma_desc_a = &gmem_tensormap[sm_idx];
    cute::TmaDescriptor* tma_desc_b = &gmem_tensormap[sm_idx + sm_count];

    if (cute::elect_one_sync()) {
      // Bringing tensormaps from params to smem for modification later
      Tensor pA_tensormap = make_tensor(mainloop_params.tma_load_a.get_tma_descriptor(), Int<1>{}, Int<1>{});
      Tensor sA_tensormap = make_tensor(make_smem_ptr(&shared_tensormaps.smem_tensormap_A), Int<1>{}, Int<1>{});
      Tensor pB_tensormap = make_tensor(mainloop_params.tma_load_b.get_tma_descriptor(), Int<1>{}, Int<1>{});
      Tensor sB_tensormap = make_tensor(make_smem_ptr(&shared_tensormaps.smem_tensormap_B), Int<1>{}, Int<1>{});

      copy(recast<uint128_t>(pA_tensormap), recast<uint128_t>(sA_tensormap));
      copy(recast<uint128_t>(pB_tensormap), recast<uint128_t>(sB_tensormap));
    }
    __syncwarp();

    return cute::make_tuple(tma_desc_a, tma_desc_b);
  }

  // Replace address for the global tensor (to be done by single thread)
  CUTLASS_DEVICE
  void
  tensormaps_replace_global_address(
      TensorMapStorage& shared_tensormaps,
      Params const& mainloop_params,
      int32_t next_batch) {
    // Replacing global_address for the next batch
    cute::tma_descriptor_replace_addr_in_shared_mem(shared_tensormaps.smem_tensormap_A,
                                                    mainloop_params.ptr_A[next_batch]);
    cute::tma_descriptor_replace_addr_in_shared_mem(shared_tensormaps.smem_tensormap_B,
                                                    mainloop_params.ptr_B[next_batch]);
  }

  // Replace dim and strides for the global tensor - used only for Grouped GEMM (to be done by single thread)
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE
  void
  tensormaps_replace_global_tensor_properties(
      TensorMapStorage& shared_tensormaps,
      Params const& mainloop_params,
      int32_t next_group,
      ProblemShape_MNKL problem_shape_mnkl) {
    const auto M = get<0>(problem_shape_mnkl);
    const auto N = get<1>(problem_shape_mnkl);
    const auto K = get<2>(problem_shape_mnkl);
    // Replace all dims for consistency
    constexpr int MaxTensorRank = 5;
    cute::array<uint32_t, MaxTensorRank> prob_shape_A  = {1,1,1,1,1};
    cute::array<uint64_t, MaxTensorRank> prob_stride_A = {0,0,0,0,0};
    cute::array<uint32_t, MaxTensorRank> prob_shape_B  = {1,1,1,1,1};
    cute::array<uint64_t, MaxTensorRank> prob_stride_B = {0,0,0,0,0};

    InternalElementA const* ptr_A = nullptr;
    // Composed A: the descriptor's dims are (M, U, m_g) with the path in the batch mode; the
    // flat path keeps (M, K, 1).  m_g is exactly K/U (the problem's flat contraction length
    // divided by the per-path extent).
    Tensor tensor_a = [&]() {
      if constexpr (kComposedA) {
        int const m_g = int(K) / mainloop_params.u;
        return make_tensor(ptr_A, make_shape(M, mainloop_params.u, m_g), mainloop_params.dA[next_group]);
      } else {
        return make_tensor(ptr_A, make_shape(M,K,Int<1>{}), mainloop_params.dA[next_group]);
      }
    }();

    InternalElementB const* ptr_B = nullptr;
    Tensor tensor_b = make_tensor(ptr_B, make_shape(N,K,Int<1>{}), mainloop_params.dB[next_group]);

    cute::detail::fill_tma_gmem_shape_stride(mainloop_params.tma_load_a, tensor_a,
                                             prob_shape_A, prob_stride_A);
    cute::detail::fill_tma_gmem_shape_stride(mainloop_params.tma_load_b, tensor_b,
                                             prob_shape_B, prob_stride_B);

    // Convert strides to byte strides
    for (uint64_t& stride : prob_stride_A) {
      stride = (stride * sizeof_bits_v<InternalElementA>) / 8;
    }
    for (uint64_t& stride : prob_stride_B) {
      stride = (stride * sizeof_bits_v<InternalElementB>) / 8;
    }

    cute::tma_descriptor_replace_dims_strides_in_shared_mem(shared_tensormaps.smem_tensormap_A,
                                                            prob_shape_A,
                                                            prob_stride_A);
    cute::tma_descriptor_replace_dims_strides_in_shared_mem(shared_tensormaps.smem_tensormap_B,
                                                            prob_shape_B,
                                                            prob_stride_B);
  }

  template <class TensorMapA, class TensorMapB, class ProblemShape_MNKL>
  CUTLASS_DEVICE
  void
  tensormaps_perform_update(
      TensorMapStorage& shared_tensormaps,
      Params const& mainloop_params,
      cute::tuple<TensorMapA, TensorMapB> const& input_tensormaps,
      ProblemShape_MNKL problem_shape_mnkl,
      int32_t next_batch) {
    if (cute::elect_one_sync()) {
      // Replacing global_address for the next batch
      tensormaps_replace_global_address(shared_tensormaps, mainloop_params, next_batch);

      if constexpr (IsGroupedGemmKernel) {
        // Replacing global dims and strides for the next batch
        tensormaps_replace_global_tensor_properties(shared_tensormaps,
          mainloop_params, next_batch, problem_shape_mnkl);
      }
    }
  }

  template <class TensorMapA, class TensorMapB>
  CUTLASS_DEVICE
  void
  tensormaps_cp_fence_release (
      TensorMapStorage& shared_tensormaps,
      cute::tuple<TensorMapA, TensorMapB> const& input_tensormaps) {
    if (cute::elect_one_sync()) {
      cute::tma_desc_commit_group();
      cute::tma_desc_wait_group();
    }
    // Entire warp must do this (i.e. it's aligned)
    tma_descriptor_cp_fence_release(get<0>(input_tensormaps), shared_tensormaps.smem_tensormap_A);
    tma_descriptor_cp_fence_release(get<1>(input_tensormaps), shared_tensormaps.smem_tensormap_B);
  }

  // The entire warp must call this function collectively (that is, the instructions are aligned)
  template <class TensorMapA, class TensorMapB>
  CUTLASS_DEVICE
  void
  tensormaps_fence_acquire(cute::tuple<TensorMapA, TensorMapB> const& input_tensormaps) {
    cute::tma_descriptor_fence_acquire(get<0>(input_tensormaps));
    cute::tma_descriptor_fence_acquire(get<1>(input_tensormaps));
  }

  template <class InputTensors, class ProblemShape_MNKL>
  CUTLASS_DEVICE
  InputTensors
  tensors_perform_update(
      InputTensors const& input_tensors,
      [[maybe_unused]] Params const& mainloop_params,
      [[maybe_unused]] ProblemShape_MNKL problem_shape_mnkl,
      [[maybe_unused]] int32_t next_batch) {
    return input_tensors;
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
