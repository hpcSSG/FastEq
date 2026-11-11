/***************************************************************************************************
 * fasteq eqlinear backends/tf32/collective/collective_builder_tf32x3.hpp
 *
 * Route `CollectiveBuilder` to the tf32x3 mainloops.
 *
 * A copy of the SM90 array GEMM arm of CUTLASS's `sm90_gmma_builder.inl` (v4.8.0, BSD-3-Clause),
 * keyed on two custom schedule tags so it never competes with the stock specialization.  Two
 * mainloops share everything here except three axes:
 *
 *  1. **Where A comes from.**  SS (M3-2): A is a smem descriptor and the mainloop derives A_res
 *     into a fourth smem tile.  RS (M4-1): a `rs_` WGMMA takes A from registers, so A needs a
 *     smem->register copy atom and no A_res tile exists.
 *  2. **The GMMA operand selector** -- `ss_op_selector` vs `rs_op_selector`.  Both are pinned to
 *     tf32 and to Cooperative (2 M-warpgroups), which is what the mainloop's
 *     `AtomLayoutMNK = (2,1,1)` split assumes.
 *  3. **The per-stage shared-memory footprint**, which follows from (1): four tiles (2*mk + 2*nk)
 *     or three (mk + 2*nk).  Getting this out of sync with the mainloop's actual `SharedStorage`
 *     desynchronises `DispatchPolicy::Stages` from the smem it indexes -- a correctness bug, not a
 *     perf loss -- so both sides read the same `tf32x3_stage_bytes`.
 *
 * The carveout reserves three TMA descriptors in both cases (A_res was never a TMA input) and the
 * operands always arrive as (float A, tf32 B): A raw so its residual survives the TMA, B because
 * its split is done once on the host.
 **************************************************************************************************/
#pragma once

#include "cutlass/gemm/collective/collective_builder.hpp"

#include "dispatch_policy_tf32x3.hpp"
#include "sm90_mma_tf32x3_array_tma_gmma_ss_warpspecialized.hpp"
#include "sm90_mma_tf32x3_rs_array_tma_gmma_warpspecialized.hpp"

namespace cutlass::gemm::collective {

namespace detail {

// One stage of an SM90 tf32x3 mainloop, minus the pipeline barrier.  Mirrors
// `compute_stage_count_or_override` in sm90_gmma_builder.inl.  Both the builder and the mainloop's
// `SharedStorage` are sized from this one function so they cannot drift.
template <bool AFromRegisters, class TileShapeMNK>
constexpr int
tf32x3_stage_bytes() {
  constexpr int mk_bytes = cutlass::bits_to_bytes(32 * size<0>(TileShapeMNK{}) * size<2>(TileShapeMNK{}));
  constexpr int nk_bytes = cutlass::bits_to_bytes(32 * size<1>(TileShapeMNK{}) * size<2>(TileShapeMNK{}));
  if constexpr (AFromRegisters) {
    return mk_bytes + 2 * nk_bytes;  // A_big + B_big + B_res; A_res lives in registers
  } else {
    return 2 * mk_bytes + 2 * nk_bytes;  // A_big + A_res + B_big + B_res
  }
}

template <bool AFromRegisters, int capacity_bytes_, class TileShapeMNK, int alignment = 128,
          int carveout_bytes_>
constexpr int
tf32x3_stage_count_or_override(StageCountAutoCarveout<carveout_bytes_> stage_count) {
  constexpr auto mainloop_pipeline_bytes = sizeof(typename cutlass::PipelineTmaAsync<1>::SharedStorage);
  constexpr int stage_bytes = cutlass::round_up(tf32x3_stage_bytes<AFromRegisters, TileShapeMNK>(), alignment) +
      static_cast<int>(mainloop_pipeline_bytes);

  constexpr int carveout_bytes = cutlass::round_up(carveout_bytes_, alignment);
  constexpr int capacity_bytes = capacity_bytes_ / alignment * alignment;

  return (capacity_bytes - carveout_bytes) / stage_bytes;
}

template <bool AFromRegisters, int capacity_bytes, class TileShapeMNK, int alignment = 128, int stages>
constexpr int
tf32x3_stage_count_or_override(cute::Int<stages> stage_count) { return stages; }

// GMMA operand selector.  Split into specializations rather than a `conditional_t` so that only the
// selected selector is ever instantiated -- both carry `static_assert`s that the other mode would
// trip (an `rs_` atom has a minimum K-block count an `ss_` atom does not).
template <bool AFromRegisters, class ElementAccumulator, class TileShape_MNK>
struct Tf32x3MmaSelector;

template <class ElementAccumulator, class TileShape_MNK>
struct Tf32x3MmaSelector<false, ElementAccumulator, TileShape_MNK> {
  using type = decltype(cute::GMMA::ss_op_selector<
      tfloat32_t, tfloat32_t, ElementAccumulator, TileShape_MNK,
      cute::GMMA::Major::K, cute::GMMA::Major::K>());
};

template <class ElementAccumulator, class TileShape_MNK>
struct Tf32x3MmaSelector<true, ElementAccumulator, TileShape_MNK> {
  using type = decltype(cute::GMMA::rs_op_selector<
      tfloat32_t, tfloat32_t, ElementAccumulator, TileShape_MNK,
      cute::GMMA::Major::K, cute::GMMA::Major::K>());
};

// The mainloop tag is what `CollectiveMma` partial-specializes on, so it -- not just the copy atom
// and the stage footprint -- has to follow `AFromRegisters`.
template <bool AFromRegisters, int Stages, class ClusterShape, class ScheduleTag>
struct Tf32x3MainloopTag;

template <int Stages, class ClusterShape, class ScheduleTag>
struct Tf32x3MainloopTag<false, Stages, ClusterShape, ScheduleTag> {
  using type = MainloopSm90ArrayTmaGmmaWarpSpecializedTf32x3<Stages, ClusterShape, ScheduleTag>;
};

template <int Stages, class ClusterShape, class ScheduleTag>
struct Tf32x3MainloopTag<true, Stages, ClusterShape, ScheduleTag> {
  using type = MainloopSm90ArrayTmaGmmaWarpSpecializedTf32x3Rs<Stages, ClusterShape, ScheduleTag>;
};

/// Everything the SS and RS tf32x3 builders share.  `CollectiveMma` resolves to the right mainloop
/// on its own: the two differ only in the policy tag, which is the partial specialization key.
template <
  bool AFromRegisters,
  class ScheduleTag,
  class ElementA,
  class GmemLayoutATag,
  int AlignmentA,
  class ElementB,
  class GmemLayoutBTag,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType
>
struct Tf32x3BuilderTraits {
  static_assert(is_static<TileShape_MNK>::value);
  static_assert(is_static<ClusterShape_MNK>::value);
#ifndef CUTLASS_SM90_COLLECTIVE_BUILDER_SUPPORTED
  static_assert(cutlass::detail::dependent_false<ElementA>, "Unsupported Toolkit for SM90 Collective Builder\n");
#endif
  static_assert(detail::is_aligned<ElementA, AlignmentA, ElementB, AlignmentB, detail::tma_alignment_bytes>(),
                "Should meet TMA alignment requirement\n");
  static_assert(cute::is_same_v<ElementA, float>,
                "The tf32x3 mainloop needs raw fp32 A; its residual is destroyed by a TMA-side tf32 convert.");
  static_assert(cute::is_same_v<ElementB, cutlass::tfloat32_t>,
                "The tf32x3 mainloop expects a pre-split tf32 B (big and residual tensors).");
  static_assert(cute::is_same_v<GmemLayoutATag, layout::RowMajor *>,
                "Only the row-major (K-major) A layout is wired up.");
  static_assert(cute::is_same_v<GmemLayoutBTag, layout::ColumnMajor *>,
                "Only the column-major (K-major) B layout is wired up.");

  static constexpr cute::GMMA::Major GmmaMajorA = cute::GMMA::Major::K;
  static constexpr cute::GMMA::Major GmmaMajorB = cute::GMMA::Major::K;

  using AtomLayoutMNK = Layout<Shape<_2,_1,_1>>;  // Cooperative: two MMA warpgroups own M halves

  using TiledMma = decltype(cute::make_tiled_mma(
      typename Tf32x3MmaSelector<AFromRegisters, ElementAccumulator, TileShape_MNK>::type(),
      AtomLayoutMNK{}));

  using GmemTiledCopyA = decltype(detail::sm90_cluster_shape_to_tma_atom(shape<1>(ClusterShape_MNK{})));
  using GmemTiledCopyB = decltype(detail::sm90_cluster_shape_to_tma_atom(shape<0>(ClusterShape_MNK{})));

  using SmemLayoutAtomA = decltype(detail::ss_smem_selector<
      GmmaMajorA, tfloat32_t, decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutAtomB = decltype(detail::ss_smem_selector<
      GmmaMajorB, tfloat32_t, decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());

  static constexpr size_t TensorMapStorage = sizeof(cute::TmaDescriptor) * 3;  // A, B_big, B_res
  static constexpr size_t SchedulerPipelineStorage =
      sizeof(cutlass::PipelineDetail::PipelineAsyncSharedStorage<8>);
  static constexpr int KernelSmemCarveout = static_cast<int>(TensorMapStorage + SchedulerPipelineStorage);
  static constexpr int Sm90ReducedSmemCapacityBytes = detail::sm90_smem_capacity_bytes - KernelSmemCarveout;

  static constexpr int PipelineStages =
      detail::tf32x3_stage_count_or_override<AFromRegisters, Sm90ReducedSmemCapacityBytes, TileShape_MNK>(
          StageCountType{});

  using DispatchPolicy = typename Tf32x3MainloopTag<
      AFromRegisters, PipelineStages, ClusterShape_MNK, ScheduleTag>::type;

  // Only the RS mainloop retiles A; an `ss_` atom reaches into shared memory itself.
  using SmemCopyAtomA = cute::conditional_t<AFromRegisters,
                                            Copy_Atom<cute::AutoVectorizingCopy, tfloat32_t>, void>;
  using SmemCopyAtomB = void;

  using CollectiveOp = CollectiveMma<
      DispatchPolicy,
      TileShape_MNK,
      ElementA,
      TagToStrideA_t<GmemLayoutATag>,
      ElementB,
      TagToStrideB_t<GmemLayoutBTag>,
      TiledMma,
      GmemTiledCopyA,
      SmemLayoutAtomA,
      SmemCopyAtomA,
      cute::identity,
      GmemTiledCopyB,
      SmemLayoutAtomB,
      SmemCopyAtomB,
      cute::identity
    >;
};

}  // namespace detail

/// SS (M3-2): in-kernel split, A_res materialised into shared memory.
template <
  class ElementA,
  class GmemLayoutATag,
  int AlignmentA,
  class ElementB,
  class GmemLayoutBTag,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
    arch::Sm90,
    arch::OpClassTensorOp,
    ElementA,
    GmemLayoutATag,
    AlignmentA,
    ElementB,
    GmemLayoutBTag,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType,
    cute::enable_if_t<
      cute::is_same_v<KernelScheduleType, KernelPtrArrayTf32x3TmaWarpSpecializedCooperative>>
> : detail::Tf32x3BuilderTraits<
      /*AFromRegisters=*/false,
      KernelPtrArrayTf32x3TmaWarpSpecializedCooperative,
      ElementA, GmemLayoutATag, AlignmentA,
      ElementB, GmemLayoutBTag, AlignmentB,
      ElementAccumulator, TileShape_MNK, ClusterShape_MNK, StageCountType>
{ };

/// RS (M4-1): A from registers, A_res derived there, no A_res shared-memory tile.
template <
  class ElementA,
  class GmemLayoutATag,
  int AlignmentA,
  class ElementB,
  class GmemLayoutBTag,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
    arch::Sm90,
    arch::OpClassTensorOp,
    ElementA,
    GmemLayoutATag,
    AlignmentA,
    ElementB,
    GmemLayoutBTag,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType,
    cute::enable_if_t<
      cute::is_same_v<KernelScheduleType, KernelPtrArrayTf32x3RsTmaWarpSpecializedCooperative>>
> : detail::Tf32x3BuilderTraits<
      /*AFromRegisters=*/true,
      KernelPtrArrayTf32x3RsTmaWarpSpecializedCooperative,
      ElementA, GmemLayoutATag, AlignmentA,
      ElementB, GmemLayoutBTag, AlignmentB,
      ElementAccumulator, TileShape_MNK, ClusterShape_MNK, StageCountType>
{ };

}  // namespace cutlass::gemm::collective
