/***************************************************************************************************
 * fasteq eqlinear fp64/csrc/collective/dispatch_policy_fp64.hpp
 *
 * Schedule tags for the FP64 DMMA TMA mainloop.
 *
 * `KernelPtrArrayFp64TmaWarpSpecializedCooperative` derives from CUTLASS's
 * `KernelPtrArrayTmaWarpSpecializedCooperative` so the stock SM90 array kernel selects itself:
 * sm90_gemm_array_tma_warpspecialized_cooperative.hpp:92 enables on
 *     is_base_of_v<KernelPtrArrayTmaWarpSpecializedCooperative, DispatchPolicy::Schedule>
 * Only the mainloop's operand path changes; the stock kernel, tile scheduler, epilogue and
 * workspace handling are reused untouched.
 **************************************************************************************************/
#pragma once

#include "cutlass/gemm/dispatch_policy.hpp"

namespace cutlass::gemm {

struct KernelPtrArrayFp64TmaWarpSpecializedCooperative
    : KernelPtrArrayTmaWarpSpecializedCooperative { };
struct KernelPtrArrayFp64TmaWarpSpecializedPingpong
    : KernelPtrArrayTmaWarpSpecializedPingpong { };

/// Mainloop tag.  Separate type from any GMMA tag because `CollectiveMma` is matched by partial
/// specialization on the policy argument.
///
/// `ComposedA` selects how the A operand's K axis is described:
///   false -- flat (M, K) with a single K stride (the packed A_prep path);
///   true  -- (M, U, m) with the path index in the descriptor's *batch* (L) mode and stride
///            i_g*U, so a multi-path group reads x in place (no permute).  The mainloop still
///            iterates the flat K = m*U tiles and decodes each into (m_path, u_tile).
template <
  int Stages_,
  class ClusterShape_ = Shape<_1, _1, _1>,
  class KernelSchedule = KernelPtrArrayFp64TmaWarpSpecializedCooperative,
  bool ComposedA_ = false
>
struct MainloopSm90ArrayTmaDmmaWarpSpecializedFp64 {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using Schedule = KernelSchedule;
  using ArchTag = arch::Sm90;
  constexpr static bool ComposedA = ComposedA_;
};

}  // namespace cutlass::gemm
