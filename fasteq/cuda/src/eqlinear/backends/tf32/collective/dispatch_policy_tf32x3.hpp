/***************************************************************************************************
 * fasteq eqlinear backends/tf32/collective/dispatch_policy_tf32x3.hpp
 *
 * M3-2: schedule tags for the in-kernel tf32x3 mainloop.
 *
 * `KernelPtrArrayTf32x3TmaWarpSpecializedCooperative` *derives* from CUTLASS's
 * `KernelPtrArrayTmaWarpSpecializedCooperative`.  That is deliberate: the stock SM90 array kernel
 * (`sm90_gemm_array_tma_warpspecialized_cooperative.hpp:70`) selects itself with
 *
 *     enable_if_t<is_base_of_v<KernelPtrArrayTmaWarpSpecializedCooperative,
 *                              CollectiveMainloop::DispatchPolicy::Schedule>>
 *
 * so deriving lets the custom collective reuse the stock kernel, scheduler, epilogue plumbing and
 * workspace handling untouched -- only the mainloop's operand traffic changes.
 **************************************************************************************************/
#pragma once

#include "cutlass/gemm/dispatch_policy.hpp"

namespace cutlass::gemm {

/// Same warp-specialised / persistent / grouped behaviour as the stock cooperative array kernel,
/// but the mainloop takes raw fp32 A and does the tf32 big/res split in shared memory.
struct KernelPtrArrayTf32x3TmaWarpSpecializedCooperative
    : KernelPtrArrayTmaWarpSpecializedCooperative { };

template <
  int Stages_,
  class ClusterShape_ = Shape<_1, _1, _1>,
  class KernelSchedule = KernelPtrArrayTf32x3TmaWarpSpecializedCooperative
>
struct MainloopSm90ArrayTmaGmmaWarpSpecializedTf32x3 {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using Schedule = KernelSchedule;
  using ArchTag = arch::Sm90;
};

/// M4-1: same kernel/scheduler/epilogue as the SS mainloop above, but the tf32x3 mainloop takes A
/// from *registers* (an `rs_` WGMMA) and derives A_res there, so no A_res tile is ever staged.
///
/// The schedule tag derives from the SS one, which itself derives from CUTLASS's cooperative array
/// tag -- that is what keeps the stock kernel selection working.  The *mainloop* tag has to be a
/// separate type rather than a flag: `CollectiveMma` is matched by partial specialization on its
/// policy argument, and two specializations with the same argument list are ill-formed no matter
/// how the trailing template parameters differ.
struct KernelPtrArrayTf32x3RsTmaWarpSpecializedCooperative
    : KernelPtrArrayTf32x3TmaWarpSpecializedCooperative { };

template <
  int Stages_,
  class ClusterShape_ = Shape<_1, _1, _1>,
  class KernelSchedule = KernelPtrArrayTf32x3RsTmaWarpSpecializedCooperative
>
struct MainloopSm90ArrayTmaGmmaWarpSpecializedTf32x3Rs {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using Schedule = KernelSchedule;
  using ArchTag = arch::Sm90;
};

}  // namespace cutlass::gemm
