/***************************************************************************************************
 * fasteq eqlinear tf32/csrc/collective/collective_builder_tf32x3_staged.hpp
 *
 * P1: builder wiring for the numeric fork of the tf32x3 RS mainloop.
 *
 * Everything that is not specific to the fork is reused from the frozen
 * `collective_builder_tf32x3.hpp`: `Tf32x3BuilderTraits` is generic over the schedule tag, so all
 * this file adds is (a) the `Tf32x3MainloopTag` specialization that maps the fork's tag to the
 * fork's mainloop, and (b) a `CollectiveBuilder` specialization that routes the fork's tag into
 * those shared traits.  No code is duplicated.
 *
 * The fork's `Tf32x3MainloopTag` specialization is *more* specialized than the frozen
 * `<true, Stages, ClusterShape, ScheduleTag>` one (its last argument is a concrete template), so
 * partial-ordering picks it for the fork's tag only -- the RS and SS paths are untouched.
 **************************************************************************************************/
#pragma once

#include "collective_builder_tf32x3.hpp"
#include "dispatch_policy_tf32x3_staged.hpp"
#include "sm90_mma_tf32x3_rs_stage_array_tma_gmma_warpspecialized.hpp"

namespace cutlass::gemm::collective {

using namespace cute;

namespace detail {

/// Map the fork's schedule tag onto the fork's mainloop policy.
template <int Stages, class ClusterShape, bool RnBig, bool Fold>
struct Tf32x3MainloopTag<true, Stages, ClusterShape,
                         KernelPtrArrayTf32x3RsStageTmaWarpSpecializedCooperative<RnBig, Fold>> {
  using type = MainloopSm90ArrayTmaGmmaWarpSpecializedTf32x3Stage<
      Stages, ClusterShape, KernelPtrArrayTf32x3RsStageTmaWarpSpecializedCooperative<RnBig, Fold>>;
};

}  // namespace detail

namespace detail {

/// True for every instantiation of the fork's schedule tag.  A `std::is_base_of` test cannot be
/// used here: `Tag<true>` and `Tag<false>` are unrelated types, not a base/derived pair.
template <class T>
struct IsStagedRsSchedule : cute::false_type {};
template <bool RnBig, bool Fold>
struct IsStagedRsSchedule<KernelPtrArrayTf32x3RsStageTmaWarpSpecializedCooperative<RnBig, Fold>>
    : cute::true_type {};

}  // namespace detail

/// Numeric fork of the RS path: same tile, same operand layout (`kFp32RegSplit`), different split
/// point.  `AFromRegisters = true` keeps the RS Mma selector and the register-source copy atom.
/// `RnBig` is not a template parameter of this specialization -- it rides in the schedule tag, so
/// it reaches `Tf32x3BuilderTraits` (whose `ScheduleTag` argument is the tag type) unchanged.
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
    cute::enable_if_t<detail::IsStagedRsSchedule<KernelScheduleType>::value>
> : detail::Tf32x3BuilderTraits<
      /*AFromRegisters=*/true,
      KernelScheduleType,
      ElementA, GmemLayoutATag, AlignmentA,
      ElementB, GmemLayoutBTag, AlignmentB,
      ElementAccumulator, TileShape_MNK, ClusterShape_MNK, StageCountType>
{ };

}  // namespace cutlass::gemm::collective
