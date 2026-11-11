/***************************************************************************************************
 * fasteq eqlinear backends/tf32/collective/sm90_mma_tf32x3_array_tma_gmma_ss_warpspecialized.hpp
 *
 * M3-2 stage 1: SM90 grouped (ptr-array) warp-specialised mainloop that computes the tf32x3
 * emulation of an fp32 GEMM while reading A from global memory *exactly once*.
 *
 * Derived from CUTLASS's
 *   include/cutlass/gemm/collective/sm90_mma_array_tma_gmma_ss_warpspecialized.hpp
 * (CUTLASS v4.8.0, BSD-3-Clause, Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES).
 * Only the parts listed below differ; everything else is the upstream code, kept verbatim so the
 * pipeline / scheduler / epilogue contracts stay identical.
 *
 * --- Why ---
 *
 * The M2-A1 kernel expressed tf32x3 as a K-concatenation:  C = [A_big|A_res|A_big] @ [B_big;B_big;B_res].
 * That is algebraically correct but materialises A **three times** in global memory (1.12 GB written
 * by the prep kernel and 1.12 GB read back by the GEMM on LARGE_P3/B5888, against 0.37 GB of real
 * input).  End to end the operator moved 2.73 GB where it needs 0.46 GB.
 *
 * Here A is stored once as a plain fp32 permute and the three products are issued as three
 * `cute::gemm` calls on the *same* shared-memory stage:
 *
 *     pass 1  A_big . B_big     (ScaleOut::Zero, set up by the prologue)
 *     pass 2  A_res . B_big     (ScaleOut::One)
 *     pass 3  A_big . B_res     (ScaleOut::One)
 *
 * A_big / A_res are produced in shared memory by `split_a()` from the fp32 tile the TMA just
 * landed; B_big / B_res are pre-split on the host (weights, a few MB, off the inference path) and
 * arrive as two separate global tensors with identical shapes and strides.
 *
 * --- The four deltas from upstream ---
 *
 *  1. `ConvertF32toTF32A` is forced false, so the TMA copies raw fp32 bits -- the residual cannot
 *     be recovered once the low 13 mantissa bits are gone.
 *  2. `SharedStorage` grows `smem_A_res` and `smem_B_res`; `Arguments`/`Params` grow the
 *     `ptr_B_res` tensor and its TMA descriptor, and the per-SM tensormap workspace holds three
 *     descriptors instead of two.  `dB` is shared: the two B tensors are laid out identically.
 *  3. `load()` issues a third TMA load per stage, and `TmaTransactionBytes` accounts for it.
 *  4. `mma()` splits the A stage and issues three `cute::gemm` calls inside one
 *     `warpgroup_arrive()/warpgroup_commit_batch()` window, with a named barrier between the
 *     split and the GMMAs.  `consumer_release` stays where upstream put it -- after the last pass
 *     -- so the stage lives across all three.
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

#include "dispatch_policy_tf32x3.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  int Stages,
  class ClusterShape,
  class KernelSchedule,
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
    MainloopSm90ArrayTmaGmmaWarpSpecializedTf32x3Rs<Stages, ClusterShape, KernelSchedule>,
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
  using DispatchPolicy = MainloopSm90ArrayTmaGmmaWarpSpecializedTf32x3Rs<Stages, ClusterShape, KernelSchedule>;
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
  // RS: A is a register operand, so only B is a smem descriptor.  `FrgTypeA` being a plain value
  // type (`tfloat32_t`) is exactly what makes `make_fragment_A` allocate registers instead of
  // descriptors, and the `is_void_v<SmemCopyAtomA>` assert in `mma()` is the other half of the pair.
  static_assert(!cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value &&
                cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeB>::value,
                "The RS mainloop needs a register-sourced A and a smem_desc-sourced B.");
  static_assert(cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>,
      "GmemTiledCopy - invalid SM90 TMA copy atom specified.");
  static_assert(cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>,
      "GmemTiledCopy - invalid SM90 TMA copy atom specified.");
  static_assert(cute::is_same_v<ElementB, tfloat32_t>,
      "The B operand must already be tf32: its big/res split is done on the host.");

  // Delta 1: A lands in smem as raw fp32.  Upstream forces tf32 (round) for ElementA == float; the
  // low 13 mantissa bits are exactly what `split_a()` needs to build the residual from, so the TMA
  // must not touch them.
  static constexpr bool ConvertF32toTF32A = false;
  static constexpr bool ConvertF32toTF32B = cute::is_same_v<float, ElementB>;
  using InternalElementA = cute::conditional_t<ConvertF32toTF32A, tfloat32_t, uint_bit_t<sizeof_bits_v<ElementA>>>;
  using InternalElementB = cute::conditional_t<ConvertF32toTF32B, tfloat32_t, uint_bit_t<sizeof_bits_v<ElementB>>>;

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
  /// Delta 2: the residual half of B.  Same box, same strides, different base address.
  using TMA_B_res = TMA_B;

  struct SharedStorage {
    struct TensorStorage : cute::aligned_struct<128, _0> {
      cute::array_aligned<typename TiledMma::ValTypeA, cute::cosize_v<SmemLayoutA>> smem_A;
      cute::array_aligned<typename TiledMma::ValTypeB, cute::cosize_v<SmemLayoutB>> smem_B;
      // RS: only B still needs a second tile.  A_res never lands in shared memory -- it is produced
      // in registers from the register copy of A (see `split_A_regs`), so `stage_bytes_` in the
      // builder is `mk + 2*nk`, not `2*mk + 2*nk`.
      cute::array_aligned<typename TiledMma::ValTypeB, cute::cosize_v<SmemLayoutB>> smem_B_res;
    } tensors;

    struct TensorMapStorage : cute::aligned_struct<128, _0> {
      cute::TmaDescriptor smem_tensormap_A;
      cute::TmaDescriptor smem_tensormap_B;
      cute::TmaDescriptor smem_tensormap_B_res;
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
    ElementA const** ptr_A;      // raw fp32, one copy
    StrideA dA;
    ElementB const** ptr_B;      // tf32 big part
    StrideB dB;                  // shared by ptr_B and ptr_B_res
    ElementB const** ptr_B_res;  // tf32 residual part, same layout as ptr_B
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
    TMA_B_res tma_load_b_res;
    uint32_t tma_transaction_bytes = TmaTransactionBytes;
    void* tensormaps;
    InternalElementA const** ptr_A;
    StrideA dA;
    InternalElementB const** ptr_B;
    StrideB dB;
    InternalElementB const** ptr_B_res;
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
    Tensor tensor_a = make_tensor(ptr_A_first_batch, make_layout(make_shape(init_M,init_K,init_L), stride_a));
    Tensor tensor_b = make_tensor(ptr_B_first_batch, make_layout(make_shape(init_N,init_K,init_L), stride_b));
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
    // B_res uses the same layout, so reusing tensor_b (same shape/stride) is exact.
    TMA_B_res tma_load_b_res = make_tma_copy(
        GmemTiledCopyB{},
        tensor_b,
        SmemLayoutB{}(_,_,Int<0>{}),
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),
        size<0>(ClusterShape{}));

    void* tensormaps = workspace;

    return {
      tma_load_a,
      tma_load_b,
      tma_load_b_res,
      TmaTransactionBytes,
      tensormaps,
      reinterpret_cast<InternalElementA const**>(args.ptr_A),
      args.dA,
      reinterpret_cast<InternalElementB const**>(args.ptr_B),
      args.dB,
      reinterpret_cast<InternalElementB const**>(args.ptr_B_res)
    };
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args, int sm_count) {
    // Delta 2: three input tensormaps (A, B_big, B_res) per SM instead of two.
    constexpr uint32_t NumInputTensors = 3;
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
  // Delta 3: A + B_big + B_res bytes per stage.
  static constexpr uint32_t TmaTransactionBytes =
        cutlass::bits_to_bytes(size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) * static_cast<uint32_t>(sizeof_bits<ElementA>::value))+
        cutlass::bits_to_bytes(size<0>(SmemLayoutB{}) * size<1>(SmemLayoutB{}) * static_cast<uint32_t>(sizeof_bits<ElementB>::value))+
        cutlass::bits_to_bytes(size<0>(SmemLayoutB{}) * size<1>(SmemLayoutB{}) * static_cast<uint32_t>(sizeof_bits<ElementB>::value));

  // Set up the data needed by this collective for load and mma.
  // Returns a tuple of tensors. The collective and the kernel layer have the contract that the
  // returned tuple must contain at least two elements, with the first two elements being:
  // gA_mkl - The tma tensor, A after a local tile so it has shape  (BLK_M,BLK_K,m,k,l)
  // gB_nkl - The tma tensor, B after a local tile so it has shape  (BLK_N,BLK_K,n,k,l)
  // The rest of the tensors can be specified as needed by this collective.
  //
  // Delta 3: a third element, the B_res view, is appended.  The stock kernel only ever reads
  // elements 0 and 1 by index (`get<0>` / `get<1>`), so a wider tuple is source-compatible.
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto
  load_init(ProblemShape_MNKL const& problem_shape_MNKL, Params const& mainloop_params) const {
    using X = Underscore;
    // Separate out problem shape for convenience
    auto [M,N,K,L] = problem_shape_MNKL;
    const int32_t init_L = 1;

    // TMA requires special handling of strides to deal with coord codomain mapping
    // Represent the full tensors -- get these from TMA
    Tensor mA_mkl = mainloop_params.tma_load_a.get_tma_tensor(make_shape(M,K,init_L));                            // (m,k,l)
    Tensor mB_nkl = mainloop_params.tma_load_b.get_tma_tensor(make_shape(N,K,init_L));                            // (n,k,l)
    Tensor mB_res_nkl = mainloop_params.tma_load_b_res.get_tma_tensor(make_shape(N,K,init_L));                    // (n,k,l)

    // Make tiled views, defer the slice
    Tensor gA_mkl = local_tile(mA_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});  // (BLK_M,BLK_K,m,k,l)
    Tensor gB_nkl = local_tile(mB_nkl, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});  // (BLK_N,BLK_K,n,k,l)
    Tensor gB_res_nkl = local_tile(mB_res_nkl, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});  // (BLK_N,BLK_K,n,k,l)

    return cute::make_tuple(gA_mkl, gB_nkl, gB_res_nkl);
  }

  // Perform a collective-scoped matrix multiply-accumulate
  // Producer Perspective
  template <
    class TensorA, class TensorB, class TensorBRes,
    class TensorMapA, class TensorMapB, class TensorMapBRes,
    class KTileIterator, class BlockCoord
  >
  CUTLASS_DEVICE void
  load(
      Params const& mainloop_params,
      MainloopPipeline pipeline,
      PipelineState smem_pipe_write,
      cute::tuple<TensorA, TensorB, TensorBRes> const& load_inputs,
      cute::tuple<TensorMapA, TensorMapB, TensorMapBRes> const& input_tensormaps,
      BlockCoord const& blk_coord,
      KTileIterator k_tile_iter, int k_tile_count,
      int thread_idx,
      uint32_t block_rank_in_cluster,
      TensorStorage& shared_tensors) {
    int lane_predicate = cute::elect_one_sync();

    if (lane_predicate) {
      Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});        // (BLK_M,BLK_K,PIPE)
      Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{});        // (BLK_N,BLK_K,PIPE)
      Tensor sB_res = make_tensor(make_smem_ptr(shared_tensors.smem_B_res.data()), SmemLayoutB{});  // (BLK_N,BLK_K,PIPE)

      //
      // Prepare the TMA loads for A and B
      //

      constexpr uint32_t cluster_shape_x = get<0>(typename DispatchPolicy::ClusterShape());
      uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};

      Tensor gA_mkl = get<0>(load_inputs);
      Tensor gB_nkl = get<1>(load_inputs);
      Tensor gB_res_nkl = get<2>(load_inputs);

      auto block_tma_a = mainloop_params.tma_load_a.get_slice(cluster_local_block_id.y);
      auto block_tma_b = mainloop_params.tma_load_b.get_slice(cluster_local_block_id.x);
      auto block_tma_b_res = mainloop_params.tma_load_b_res.get_slice(cluster_local_block_id.x);

      // Partition the inputs based on the current block coordinates.
      auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;
      Tensor gA = gA_mkl(_,_,m_coord,_,l_coord);                                                     // (BLK_M,BLK_K,k)
      Tensor gB = gB_nkl(_,_,n_coord,_,l_coord);                                                     // (BLK_N,BLK_K,k)
      Tensor gB_res = gB_res_nkl(_,_,n_coord,_,l_coord);                                             // (BLK_N,BLK_K,k)

      // Applies the mapping from block_tma_a
      Tensor tAgA = block_tma_a.partition_S(gA);                                                 // (TMA,TMA_M,TMA_K,k)
      Tensor tAsA = block_tma_a.partition_D(sA);                                              // (TMA,TMA_M,TMA_K,PIPE)

      Tensor tBgB = block_tma_b.partition_S(gB);                                                 // (TMA,TMA_N,TMA_K,k)
      Tensor tBsB = block_tma_b.partition_D(sB);                                              // (TMA,TMA_N,TMA_K,PIPE)

      Tensor tBgB_res = block_tma_b_res.partition_S(gB_res);                                     // (TMA,TMA_N,TMA_K,k)
      Tensor tBsB_res = block_tma_b_res.partition_D(sB_res);                                  // (TMA,TMA_N,TMA_K,PIPE)

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

      // Mainloop
      CUTLASS_PRAGMA_NO_UNROLL
      for ( ; k_tile_count > 0; --k_tile_count)
      {
        // LOCK smem_pipe_write for _writing_
        pipeline.producer_acquire(smem_pipe_write);

        //
        // Copy gmem to smem for *k_tile_iter
        //

        using BarrierType = typename MainloopPipeline::ProducerBarrierType;
        BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);

        int write_stage = smem_pipe_write.index();
        copy(mainloop_params.tma_load_a.with(get<0>(input_tensormaps), *tma_barrier, mcast_mask_a, cache_hint_a()), tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,write_stage));
        copy(mainloop_params.tma_load_b.with(get<1>(input_tensormaps), *tma_barrier, mcast_mask_b, cache_hint_b()), tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,write_stage));
        copy(mainloop_params.tma_load_b_res.with(get<2>(input_tensormaps), *tma_barrier, mcast_mask_b, cache_hint_b()), tBgB_res(_,_,_,*k_tile_iter), tBsB_res(_,_,_,write_stage));
        ++k_tile_iter;

        // Advance smem_pipe_write
        ++smem_pipe_write;
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

  /// M4-1: build A_res *in registers* from the register copy of A_big.
  ///
  /// A_big is *not* materialized: a tf32 WGMMA reads only the high 19 bits of its 32-bit operand,
  /// which is exactly the round-toward-zero `NumericConverterFastF32` would have written back, so
  /// the raw fp32 value already *is* A_big.  Only the residual has to be produced.
  ///
  /// `cutlass::NumericConverterFastF32` is the same primitive the host-side split (`split_pack.cu`),
  /// the Ampere M1 path and the SS fork's `split_A_stage` use, and this loop is a line-for-line
  /// port of that one -- so the three products stay numerically identical.  What changed is where
  /// A_res lands: registers, not a second shared-memory tile.  That removes the split's smem read
  /// and write, its `fence_view_async_shared` and its named barrier, *and* the A-side GMMA operand
  /// reads -- A is fetched from smem exactly once per K-tile now instead of three times.
  ///
  /// `tfloat32_t`'s own converting constructor and `operator float()` are both lossy
  /// (`round_half_ulp_truncate` in, `& ~0x1fff` out), so the raw 32 bits are moved through the
  /// public `storage` member -- a round trip through those operators would silently truncate the
  /// residual to zero and the third product would stop contributing.
  template <class TensorA, class TensorARes>
  CUTLASS_DEVICE void
  split_A_regs(TensorA& tCrA_k, TensorARes& tCrA_res_k) {
    cutlass::NumericConverterFastF32<> to_fast_f32;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(tCrA_k); ++i) {
      float const a = __uint_as_float(tCrA_k(i).storage);
      tfloat32_t r;
      r.storage = __float_as_uint(static_cast<float>(to_fast_f32(a)[1]));
      tCrA_res_k(i) = r;
    }
  }

  /// Stage this K-tile's A into registers and derive A_res there.
  ///
  /// These registers are a single buffer, not a pipeline: `wgmma.mma_async` may still be reading
  /// the previous K-tile's values when the next `copy` arrives, and `K_PIPE_MMAS` deliberately
  /// leaves that many batches in flight.  The stock SM90 RS mainloop overwrites a single-buffered
  /// tCrA the same way and is safe only because it asserts `size<2>(tCrA) > 2` K-blocks of slack,
  /// so that whatever reads slot s has retired by the time slot s is written again; TB_K = 16
  /// gives MMA_K = 2, which is below that.  Draining is what restores the invariant, and it costs
  /// one MMA/compute hand-off per K-tile against the ~100x more work the GMMAs represent.
  template <class SmemTiledCopyA, class SmemThrCopyA, class TensorACopyView, class TensorA,
            class TensorARes>
  CUTLASS_DEVICE void
  stage_A_regs(SmemTiledCopyA const& smem_tiled_copy_A,
               SmemThrCopyA   const& smem_thr_copy_A,
               TensorACopyView const& tCsA_copy_view,
               int             read_stage,
               TensorA&        tCrA,
               TensorARes&     tCrA_res) {
    warpgroup_wait<0>();
    copy(smem_tiled_copy_A, tCsA_copy_view(_,_,_,read_stage), smem_thr_copy_A.retile_D(tCrA));
    split_A_regs(tCrA, tCrA_res);
  }

  /// Passes 2 and 3 of a K-tile: A_res . B_big and A_big . B_res.  Kept separate from pass 1
  /// because the prologue has to unroll pass 1 itself to flip `accumulate_` to One.
  ///
  /// `tCrA_k` / `tCrA_res_k` are the parity-selected register buffers for this K-tile (already
  /// sliced out of the 2-deep staging array); B stays smem-sourced and is still indexed by stage.
  template <class TensorAK, class TensorAResK, class TensorB, class TensorBRes, class FrgTensorC>
  CUTLASS_DEVICE void
  mma_res_passes(TiledMma& tiled_mma,
                 TensorAK const& tCrA_k, TensorAResK const& tCrA_res_k,
                 TensorB const& tCrB, TensorBRes const& tCrB_res,
                 FrgTensorC& accum, int read_stage) {
    CUTLASS_PRAGMA_UNROLL
    for (int k_block = 0; k_block < size<2>(tCrA_k); ++k_block) {
      cute::gemm(tiled_mma, tCrA_res_k(_,_,k_block), tCrB(_,_,k_block,read_stage), accum);
    }
    CUTLASS_PRAGMA_UNROLL
    for (int k_block = 0; k_block < size<2>(tCrA_k); ++k_block) {
      cute::gemm(tiled_mma, tCrA_k(_,_,k_block), tCrB_res(_,_,k_block,read_stage), accum);
    }
  }

  /// All three tf32x3 products for one K-tile.  All of them go into one warpgroup batch so the
  /// K-tile's shared-memory stage stays live across them and `consumer_release` (which the caller
  /// does once, after this returns) only fires when the last pass has been issued.
  template <class TensorAK, class TensorAResK, class TensorB, class TensorBRes, class FrgTensorC>
  CUTLASS_DEVICE void
  mma_k_tile(TiledMma& tiled_mma,
             TensorAK const& tCrA_k, TensorAResK const& tCrA_res_k,
             TensorB const& tCrB, TensorBRes const& tCrB_res,
             FrgTensorC& accum, int read_stage) {
    CUTLASS_PRAGMA_UNROLL
    for (int k_block = 0; k_block < size<2>(tCrA_k); ++k_block) {
      cute::gemm(tiled_mma, tCrA_k(_,_,k_block), tCrB(_,_,k_block,read_stage), accum);
    }
    mma_res_passes(tiled_mma, tCrA_k, tCrA_res_k, tCrB, tCrB_res, accum, read_stage);
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
      "The RS tf32x3 mainloop sources A from registers, so it needs a smem->register copy atom.");
    static_assert(cute::is_void_v<SmemCopyAtomB>,
      "SM90 GMMA mainloops cannot have a non-void copy atom for smem sourced instructions.");
    // A_res lives in registers now, so the smem_A_res write-after-read distance the SS fork worries
    // about is gone and no register double buffer replaces it: `wgmma.mma_async` consumes its
    // register operands at *issue*, so overwriting them for the next K-tile cannot disturb GMMAs
    // already issued.  (The stock SM90 RS mainloop relies on the same thing when it prefetches
    // k_block+1 over the registers of k_block.)  Only the K_PIPE_MMAS pipeline lag needs smem to
    // stay put, and that is what `consumer_release` handles.

    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});          // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{});          // (BLK_N,BLK_K,PIPE)

    //
    // Define C accumulators and A/B partitioning
    //

    // Layout of warp group to thread mapping

    // Only the B half of the SS fork's check survives: an `rs_` atom's ALayout is the *register*
    // layout, which has no warpgroup-strided leading mode to assert about.
    static_assert(stride<0>(typename TiledMma::BLayout{}) == 0 and
                  size<0>(typename TiledMma::BLayout{}) == NumThreadsPerWarpGroup,
                  "Stride of the first mode must be 0 and the size of the mode must be NumThreadsPerWarpGroup");

    constexpr int MmaWarpGroups = size(TiledMma{}) / NumThreadsPerWarpGroup;
    Layout warp_group_thread_layout = make_layout(Int<MmaWarpGroups>{},
                                                  Int<NumThreadsPerWarpGroup>{});

    int warp_group_idx = __shfl_sync(0xFFFFFFFF, thread_idx / NumThreadsPerWarpGroup, 0);

    TiledMma tiled_mma;
    auto thread_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx));

    // B stays smem-sourced, so its fragments are per-warpgroup descriptors.
    Tensor tCsA = thread_mma.partition_A(sA);                                                 // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCsB = thread_mma.partition_B(sB);                                                 // (MMA,MMA_N,MMA_K,PIPE)
    Tensor tCrB = thread_mma.make_fragment_B(tCsB);                                           // (MMA,MMA_N,MMA_K,PIPE)

    // A is register-sourced, so its fragments are per-*thread* and only one K-tile wide.  Mirrors
    // `partition_fragment_A` in the stock RS collective: the warpgroup slice is right for smem
    // descriptors, the thread slice is right for registers.
    auto mma_thread_slice  = tiled_mma.get_thread_slice(thread_idx);
    Tensor tCrA_one = mma_thread_slice.partition_fragment_A(sA(_,_,Int<0>{}));                 // (MMA,MMA_M,MMA_K)
    // A_big and A_res for the K-tile being consumed.  Both are built from `shape(tCrA_one)` so they
    // inherit its layout: an `rs_` A fragment is a plain register array, and the GMMAs read each
    // thread's four values in the order that layout defines.  Hand-rolling a layout here (or
    // indexing either tensor with a runtime coordinate, which materialises the array in local
    // memory) is enough to hand the tensor cores their operands in the wrong register order -- do
    // not reintroduce a parity/deep-pipeline buffer.
    Tensor tCrA     = make_tensor<tfloat32_t>(shape(tCrA_one));
    Tensor tCrA_res = make_tensor<tfloat32_t>(shape(tCrA_one));

    // smem -> register copy for A: now the *only* A traffic left on the L1/TEX pipe.
    auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(thread_idx);
    Tensor tCsA_copy_view  = smem_thr_copy_A.partition_S(sA);                                 // (CPY,CPY_M,CPY_K,PIPE)

    // Delta 2: descriptor views over the residual tiles, laid out identically to their big parts.
    Tensor sB_res = make_tensor(make_smem_ptr(shared_tensors.smem_B_res.data()), SmemLayoutB{});
    Tensor tCsB_res = thread_mma.partition_B(sB_res);                                         // (MMA,MMA_N,MMA_K,PIPE)
    Tensor tCrB_res = thread_mma.make_fragment_B(tCsB_res);                                   // (MMA,MMA_N,MMA_K,PIPE)

    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(accum));                                                         // M
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<2>(accum));                                                         // N
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));                                                          // K
    CUTE_STATIC_ASSERT_V(size<3>(tCsA) == size<3>(tCsB));                                                       // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));                                         // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));                                         // PIPE
    // The A register fragment and the smem->register copy must agree mode for mode (the same two
    // checks the stock RS collective makes on its `tCrA_copy_view`).
    CUTE_STATIC_ASSERT_V(size<2>(tCrA_one) == size<2>(tCsA));                                                   // K
    CUTE_STATIC_ASSERT_V(size<2>(tCsA_copy_view) == size<2>(smem_thr_copy_A.retile_D(tCrA_one)));            // CPY_K

    //
    // PIPELINED MAIN LOOP
    //
    static_assert((0 <= K_PIPE_MMAS) && (K_PIPE_MMAS <  K_PIPE_MAX),
        "ERROR : Incorrect number of MMAs in flight");

    // We release buffers to producer warps(dma load) with some mmas in flight
    PipelineState smem_pipe_release = smem_pipe_read;

    // Prologue GMMAs
    int prologue_mma_count = min(K_PIPE_MMAS, k_tile_count);
    tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
    warpgroup_fence_operand(accum);
    if (k_tile_count > 0) {
      // WAIT on smem_pipe_read until its data are available (phase bit flips from rdPhaseBit value)
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);

      int read_stage = smem_pipe_read.index();
      stage_A_regs(smem_tiled_copy_A, smem_thr_copy_A, tCsA_copy_view, read_stage, tCrA, tCrA_res);

      warpgroup_arrive();
      // Unroll the K mode manually to set scale D to 1
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA_one); ++k_block) {
        // (V,M,K) x (V,N,K) => (V,M,N)
        cute::gemm(tiled_mma, tCrA(_,_,k_block), tCrB(_,_,k_block,read_stage), accum);
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
      }
      // Pass 1 is already complete above; only the two residual passes are left.
      mma_res_passes(tiled_mma, tCrA, tCrA_res, tCrB, tCrB_res, accum, read_stage);

      warpgroup_commit_batch();

      ++smem_pipe_read;
    }

    warpgroup_fence_operand(accum);
    CUTLASS_PRAGMA_UNROLL
    for (int k_tile_prologue = prologue_mma_count - 1; k_tile_prologue > 0; --k_tile_prologue)
    {
      // WAIT on smem_pipe_read until its data are available (phase bit flips from rdPhaseBit value)
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);

      int read_stage = smem_pipe_read.index();
      stage_A_regs(smem_tiled_copy_A, smem_thr_copy_A, tCsA_copy_view, read_stage, tCrA, tCrA_res);

      warpgroup_arrive();
      mma_k_tile(tiled_mma, tCrA, tCrA_res, tCrB, tCrB_res, accum, read_stage);
      warpgroup_commit_batch();

      ++smem_pipe_read;
    }

    warpgroup_fence_operand(accum);
    // Mainloop GMMAs
    k_tile_count -= prologue_mma_count;

    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 0; --k_tile_count)
    {
      // WAIT on smem_pipe_read until its data are available (phase bit flips from rdPhaseBit value)
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);

      //
      // Compute on k_tile
      //

      int read_stage = smem_pipe_read.index();
      warpgroup_fence_operand(accum);

      stage_A_regs(smem_tiled_copy_A, smem_thr_copy_A, tCsA_copy_view, read_stage, tCrA, tCrA_res);

      warpgroup_arrive();
      mma_k_tile(tiled_mma, tCrA, tCrA_res, tCrB, tCrB_res, accum, read_stage);
      warpgroup_commit_batch();

      /// Wait on the GMMA barrier for K_PIPE_MMAS (or fewer) outstanding to ensure smem_pipe_write is consumed
      warpgroup_wait<K_PIPE_MMAS>();
      warpgroup_fence_operand(accum);

      // UNLOCK smem_pipe_release, done _computing_ on it
      pipeline.consumer_release(smem_pipe_release);

      // Advance smem_pipe_read and smem_pipe_release
      ++smem_pipe_read;
      ++smem_pipe_release;
    }

    warpgroup_fence_operand(accum);
  }

  /// Perform a Consumer Epilogue to release all buffers
  CUTLASS_DEVICE void
  mma_tail(MainloopPipeline pipeline, PipelineState smem_pipe_release, int k_tile_count) {
    // Prologue GMMAs
    int prologue_mma_count = min(K_PIPE_MMAS, k_tile_count);
    k_tile_count -= prologue_mma_count;

    smem_pipe_release.advance(k_tile_count);

    // Wait on all GMMAs to complete
    warpgroup_wait<0>();

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
    cute::TmaDescriptor* tma_desc_b_res = &gmem_tensormap[sm_idx + 2 * sm_count];

    if (cute::elect_one_sync()) {
      // Bringing tensormaps from params to smem for modification later
      Tensor pA_tensormap = make_tensor(mainloop_params.tma_load_a.get_tma_descriptor(), Int<1>{}, Int<1>{});
      Tensor sA_tensormap = make_tensor(make_smem_ptr(&shared_tensormaps.smem_tensormap_A), Int<1>{}, Int<1>{});
      Tensor pB_tensormap = make_tensor(mainloop_params.tma_load_b.get_tma_descriptor(), Int<1>{}, Int<1>{});
      Tensor sB_tensormap = make_tensor(make_smem_ptr(&shared_tensormaps.smem_tensormap_B), Int<1>{}, Int<1>{});
      Tensor pB_res_tensormap = make_tensor(mainloop_params.tma_load_b_res.get_tma_descriptor(), Int<1>{}, Int<1>{});
      Tensor sB_res_tensormap = make_tensor(make_smem_ptr(&shared_tensormaps.smem_tensormap_B_res), Int<1>{}, Int<1>{});

      copy(recast<uint128_t>(pA_tensormap), recast<uint128_t>(sA_tensormap));
      copy(recast<uint128_t>(pB_tensormap), recast<uint128_t>(sB_tensormap));
      copy(recast<uint128_t>(pB_res_tensormap), recast<uint128_t>(sB_res_tensormap));
    }
    __syncwarp();

    return cute::make_tuple(tma_desc_a, tma_desc_b, tma_desc_b_res);
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
    cute::tma_descriptor_replace_addr_in_shared_mem(shared_tensormaps.smem_tensormap_B_res,
                                                    mainloop_params.ptr_B_res[next_batch]);
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
    Tensor tensor_a = make_tensor(ptr_A, make_shape(M,K,Int<1>{}), mainloop_params.dA[next_group]);

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
    // B_res shares B's shape and strides by construction (same `dB` row).
    cute::tma_descriptor_replace_dims_strides_in_shared_mem(shared_tensormaps.smem_tensormap_B_res,
                                                            prob_shape_B,
                                                            prob_stride_B);
  }

  template <class TensorMapA, class TensorMapB, class TensorMapBRes, class ProblemShape_MNKL>
  CUTLASS_DEVICE
  void
  tensormaps_perform_update(
      TensorMapStorage& shared_tensormaps,
      Params const& mainloop_params,
      cute::tuple<TensorMapA, TensorMapB, TensorMapBRes> const& input_tensormaps,
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

  template <class TensorMapA, class TensorMapB, class TensorMapBRes>
  CUTLASS_DEVICE
  void
  tensormaps_cp_fence_release (
      TensorMapStorage& shared_tensormaps,
      cute::tuple<TensorMapA, TensorMapB, TensorMapBRes> const& input_tensormaps) {
    if (cute::elect_one_sync()) {
      cute::tma_desc_commit_group();
      cute::tma_desc_wait_group();
    }
    // Entire warp must do this (i.e. it's aligned)
    tma_descriptor_cp_fence_release(get<0>(input_tensormaps), shared_tensormaps.smem_tensormap_A);
    tma_descriptor_cp_fence_release(get<1>(input_tensormaps), shared_tensormaps.smem_tensormap_B);
    tma_descriptor_cp_fence_release(get<2>(input_tensormaps), shared_tensormaps.smem_tensormap_B_res);
  }

  // The entire warp must call this function collectively (that is, the instructions are aligned)
  template <class TensorMapA, class TensorMapB, class TensorMapBRes>
  CUTLASS_DEVICE
  void
  tensormaps_fence_acquire(cute::tuple<TensorMapA, TensorMapB, TensorMapBRes> const& input_tensormaps) {
    cute::tma_descriptor_fence_acquire(get<0>(input_tensormaps));
    cute::tma_descriptor_fence_acquire(get<1>(input_tensormaps));
    cute::tma_descriptor_fence_acquire(get<2>(input_tensormaps));
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
