/***************************************************************************************************
 * fasteq eqlinear tf32/csrc/collective/dispatch_policy_tf32x3_staged.hpp
 *
 * P1: schedule and mainloop tags for the numeric fork of the tf32x3 RS mainloop.
 *
 * These are separate *types* rather than flags for the reason the frozen M4-1 header already gives:
 * `CollectiveMma` is matched by partial specialization on its policy argument, so two mainloops
 * cannot share one policy type no matter how their trailing template parameters differ.
 *
 * The schedule tag **derives** from `KernelPtrArrayTf32x3RsTmaWarpSpecializedCooperative` (which
 * derives from `KernelPtrArrayTf32x3TmaWarpSpecializedCooperative`, which derives from CUTLASS's
 * `KernelPtrArrayTmaWarpSpecializedCooperative`).  That chain is what makes the stock SM90 array
 * kernel pick this collective up:
 *
 *     enable_if_t<is_base_of_v<KernelPtrArrayTmaWarpSpecializedCooperative,
 *                              CollectiveMainloop::DispatchPolicy::Schedule>>
 *
 * so the kernel, scheduler, epilogue and workspace handling stay the stock ones.
 **************************************************************************************************/
#pragma once

#include "dispatch_policy_tf32x3.hpp"

namespace cutlass::gemm {

/// The two numeric switches ride in the *tag* rather than as mainloop template arguments so that
/// `Tf32x3MainloopTag`'s partial specialization can key on them (the builder's trait is frozen and
/// only sees the schedule tag).
///
/// `RnBig` -- round-to-nearest split point instead of truncation.
/// `Fold`  -- allocate a second accumulator and grade the accumulation every `fold_k` K-tiles
///            (`fold_k` is a runtime field, so a ladder needs no rebuild).  `Fold=false` keeps the
///            frozen RS register footprint, which is why it is a template argument at all.
template <bool RnBig_, bool Fold_ = false>
struct KernelPtrArrayTf32x3RsStageTmaWarpSpecializedCooperative
    : KernelPtrArrayTf32x3RsTmaWarpSpecializedCooperative {
  static constexpr bool RnBig = RnBig_;
  static constexpr bool Fold = Fold_;
};

template <
  int Stages_,
  class ClusterShape_ = Shape<_1, _1, _1>,
  class KernelSchedule = KernelPtrArrayTf32x3RsStageTmaWarpSpecializedCooperative<false, false>
>
struct MainloopSm90ArrayTmaGmmaWarpSpecializedTf32x3Stage {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using Schedule = KernelSchedule;
  using ArchTag = arch::Sm90;
};

}  // namespace cutlass::gemm
