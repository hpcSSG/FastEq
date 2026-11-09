/***************************************************************************************************
 * fasteq eqlinear backends/tf32/collective/tile_scheduler_streamk_group.hpp
 *
 * M7: a *grouped* (ptr-array) tile scheduler that can split the K axis across CTAs.
 *
 * Why this file exists
 * --------------------
 * CUTLASS v4.8.0 ships an SM90 stream-K scheduler, but it is not a base class and not grouped:
 * `PersistentTileSchedulerSm90StreamK` holds `PersistentTileSchedulerSm90` as a private alias and
 * speaks the non-array cooperative kernel's scheduling API (`Pipeline = PipelineEmpty`,
 * `advance_to_next_work(uint32_t)`, `SharedStorage::data() == nullptr`).  The ptr-array cooperative
 * kernel (`sm90_gemm_array_tma_warpspecialized_cooperative.hpp`) drives its scheduler through a
 * pipeline instead -- and the grouped path is the only one with a scheduler *warp*, which pre-walks
 * the work items and pushes them into a `SchedulerPipelineStageCount`-deep shared-memory ring that
 * the mainloop/epilogue warps drain via `fetch_next_work`.  The two APIs are incompatible, so the
 * stock stream-K scheduler cannot be reused by swapping its decorated base.  Nothing in the tree
 * combines stream-K with a `GroupProblemShape`.
 *
 * What is *not* missing is the kernel side: the array kernel already has every stream-K call site
 * (`get_work_k_tile_count` :688, `get_work_k_tile_start` :689, `compute_epilogue` :844,
 * `reduction_subtile_idx()` :872, `TileScheduler::fixup` :1012), and the grouped scheduler already
 * declares `fixup` with the matching signature -- it is simply an empty body
 * (`sm90_tile_scheduler_group.hpp:493-498`), because the stock grouped decomposition never splits an
 * output tile.
 *
 * So the work is confined to this directory: a scheduler that keeps the grouped scheduler's
 * group-aware, pipeline-based scheduling API and adds (a) a K-split decomposition and (b) a real
 * `fixup`.  Nothing under `3rdparty/` is touched -- that tree is gitignored and reproducible from its
 * pinned commit, so a change there would be invisible to git and lost on re-vendor.
 *
 * How it is selected
 * ------------------
 * The array kernel's `static_assert` requires the *tag* passed to `GemmUniversal` to derive from
 * `cutlass::gemm::GroupScheduler`, and then routes that tag through `TileSchedulerSelector`.
 * `StreamKGroupScheduler` below is therefore a tag derived from `GroupScheduler`, and the selector
 * specialization at the bottom of this file maps it to the real scheduler class.
 *
 * Why the index layer is a walk and not a second heuristic
 * -------------------------------------------------------
 * `GroupProblemShape<Shape<int,int,int>>` here carries `num_groups == n` -- one group *per problem*,
 * each with its own K (`sm90_grouped_gemm.cu`: `host_problem_[j] = {B, V, rows[j].K}`).  The tile
 * space is therefore group-major, and every group owns `ceil(B/TB_M) * ceil(V/TB_N)` tiles whose
 * `k_tiles` differ from one group to the next.
 *
 * CUTLASS's decomposition arithmetic (`get_num_sk_tiles`, `get_num_sk_units`) is already pure and
 * takes per-shape scalars; what it cannot express is a problem set whose K varies.  Rather than
 * write a second heuristic beside it, this file keeps those functions and supplies them the two
 * aggregate scalars that generalize them exactly -- the *minimum* `k_tiles` over the tiles (the gate
 * asks "is there enough work to be worth splitting") and the *sum* of `k_tiles` over the tiles (the
 * unit count asks "how much work is there in total").  When every tile has the same `k_tiles` both
 * reduce to CUTLASS's own expression.
 *
 * What is genuinely new is only the index layer, and its shape follows from that heterogeneity.  The
 * non-grouped scheduler turns a flat k-iteration into (batch, tile, k) with `divmod_batch_`, because
 * its L batches all share one shape; here the divisor between one group and the next differs, so the
 * same conversion is a *walk* over the groups (`locate_sk_iter`).  Sixteen groups, once per work
 * item, on one warp -- it is not in a per-k-tile path.
 *
 * The work item space
 * -------------------
 * `decomp_.sk_units` split units come first, then `decomp_.dp_tiles` whole-tile (data-parallel)
 * items.  That order is the stock stream-K scheduler's (`sm90_tile_scheduler_stream_k.hpp:1030-1045`
 * orders the same two regions the same way) and it is the useful one: the split units exist because
 * a wave is short of full, so they are the items that want every CTA.
 *
 * A split unit covers `iters_per_unit` (or one more, for the first `big_units`) consecutive
 * k-iterations of the whole split region, which in general runs across an output tile boundary.
 * The array kernel processes exactly one output tile per work item -- it never calls
 * `continue_current_work` itself -- but its grouped scheduler's `fetch_next_work` does, *before*
 * draining the scheduler pipeline, so a unit's later tiles are handed to the same CTA without a
 * round trip through the ring.  That is the hook this file uses, and it is why the ring accounting
 * is untouched: a continuation consumes no ring slot.
 *
 * The reduction
 * -------------
 * A split tile's C is the sum of its fragments, and the array kernel gives the scheduler exactly one
 * hook to combine them: `fixup`, called by each consumer warpgroup with its accumulators, between
 * the mainloop and the epilogue.  The mechanism is the stock stream-K one -- a scratch copy of the
 * tile in the workspace, a per-(tile, warpgroup) lock, and a lock value that is the number of
 * k-iterations that have arrived so far -- so that the fragment starting at `K_idx` waits for
 * exactly `K_idx` and no arrival order has to be assumed.  What differs is only how a tile is named:
 * the stock scheduler linearizes (M, N, L) back to a tile index, this one carries the group-major
 * tile index in the work item, because the group-major index is what the split region is counted in
 * and inverting it would mean walking the groups again.
 **************************************************************************************************/
#pragma once

#include <cstdio>
#include <cstdlib>

#include "cutlass/barrier.h"
#include "cutlass/block_striped.h"
#include "cutlass/cutlass.h"
#include "cutlass/workspace.h"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/gemm/kernel/sm90_tile_scheduler_group.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_detail.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"

namespace eqlin {

/// Tag for the grouped stream-K scheduler.  Derived from `GroupScheduler` rather than being a bare
/// tag because the array kernel's `static_assert` tests the *tag* for that base
/// (`sm90_gemm_array_tma_warpspecialized_cooperative.hpp:130-136`).
struct StreamKGroupScheduler : cutlass::gemm::GroupScheduler {};

using DecompositionMode = cutlass::gemm::kernel::detail::DecompositionMode;

/// The decomposition, computed host-side from the full problem set.
///
/// This is deliberately a plain aggregate of scalars rather than a mode flag plus state: the
/// scheduler reads only these numbers at runtime, exactly as `PersistentTileSchedulerSm90StreamK`
/// does (it has no stored `decomposition_mode` either -- the mode is baked into
/// `divmod_splits_.divisor` / `sk_units_` / `sk_tiles_`).
struct GroupStreamKDecomposition {
  DecompositionMode mode = DecompositionMode::DataParallel;

  /// Output tiles, counted over the whole problem set (group-major).
  uint64_t output_tiles = 0;
  /// Tiles `[0, sk_tiles)` are the split ones, each computed by several units; tiles
  /// `[sk_tiles, output_tiles)` are whole-tile work, one item each.  That is CUTLASS's split --
  /// `get_num_sk_tiles` returns `output_tiles - dp_tiles` and `assign_work` maps its data-parallel
  /// items to `linear_idx - sk_units + sk_tiles` (`sm90_tile_scheduler_stream_k.hpp:1039`) -- and it
  /// is kept rather than inverted so the two files cannot disagree about which tile is which.
  uint32_t sk_tiles = 0;
  uint64_t dp_tiles = 0;
  uint32_t sk_units = 0;

  /// Total k-iterations of the split tiles, and the balanced split of them into `sk_units`.
  uint64_t sk_iters = 0;
  uint32_t iters_per_unit = 0;
  uint32_t big_units = 0;

  /// k-iterations of the whole problem set that precede the first split tile.  The split region's
  /// local iteration `i` is global iteration `dp_iters + i`.
  uint64_t dp_iters = 0;

  uint32_t ctas_per_wave = 0;
};

namespace detail {

/// k_tiles of one output tile is `ceil(K / TB_K)`; `tb_k` is the tile's K extent.
template <int TB_K>
CUTLASS_HOST_DEVICE uint32_t k_tiles_of(int64_t K) {
  return static_cast<uint32_t>((K + TB_K - 1) / TB_K);
}

/// Number of output tiles owned by one group (problem), matching
/// `PersistentTileSchedulerSm90Group::get_tiled_cta_shape_mnl`: M is rounded up by the cluster
/// extent, N likewise, and there is a single batch per group.
///
/// Device-callable because the split decode walks the groups on the device (`locate_sk_iter`) --
/// from the *device* problem set, which is already resident.  Building the same table on the host
/// and uploading it was measured and rejected: this operator re-runs `initialize()` on every
/// `launch()`, and the benchmark wraps the whole launch in CUDA events, so three `cudaMemcpyAsync`
/// of an O(#problems) table land inside the timed region.  That cost +23% at SMALL_P1/B512 and was
/// still +3% at LARGE_P3/B5888; it was a fixed per-launch cost, not a per-work-item one.
template <int TB_M, int TB_N, int CM, int CN>
CUTLASS_HOST_DEVICE uint64_t tiles_of_group(int64_t M, int64_t N) {
  auto blocks_m = (M + TB_M - 1) / TB_M;
  auto blocks_n = (N + TB_N - 1) / TB_N;
  if (blocks_m <= 0) blocks_m = 1;
  if (blocks_n <= 0) blocks_n = 1;
  blocks_m = ((blocks_m + CM - 1) / CM) * CM;
  blocks_n = ((blocks_n + CN - 1) / CN) * CN;
  return static_cast<uint64_t>(blocks_m) * static_cast<uint64_t>(blocks_n);
}

/// The gate and the unit count, generalized from `PersistentTileSchedulerSm90StreamKParams:
/// get_num_sk_tiles` :1067-1116 and `get_num_sk_units` :1118-1141.
///
/// The two generalizations are forced by the grouped shape and are exactly the substitutions that
/// make the CUTLASS expressions a special case:
///   * the gate `k_tiles_per_output_tile <= min_iters_per_sk_unit_` becomes "the *smallest* tile
///     has too little work to split" -- if any tile is splittable the region is worth splitting,
///     and if the smallest is not, splitting cannot pay for its own reduction anywhere;
///   * `k_tiles_per_output_tile * sk_tiles` becomes the true sum of `k_tiles` over the split tiles.
/// Phase 1: which output tiles belong to the split region.
///
/// This is `get_num_sk_tiles` (`tile_scheduler_params.h:1067-1116`) with the one substitution the
/// grouped shape forces: the scalar `k_tiles_per_output_tile` in its gate becomes `min_k_tiles`,
/// the smallest `k_tiles` over all output tiles.  Deliberately free of any per-tile data, so the
/// gate stays a function of the shape alone.
template <uint32_t MinItersPerSkUnit>
inline GroupStreamKDecomposition
select_sk_region(uint64_t output_tiles,
                 uint32_t min_k_tiles,
                 uint64_t ctas_per_wave,
                 DecompositionMode requested) {
  GroupStreamKDecomposition d;
  d.output_tiles = output_tiles;
  d.ctas_per_wave = static_cast<uint32_t>(ctas_per_wave);
  d.dp_tiles = output_tiles;  // until proven otherwise

  if (requested == DecompositionMode::DataParallel || requested == DecompositionMode::SplitK) {
    return d;
  }

  uint64_t const full_waves = output_tiles / ctas_per_wave;
  uint64_t const total_waves = (output_tiles + ctas_per_wave - 1) / ctas_per_wave;

  // The first two waves' worth of tiles are the split ones; everything after them is data-parallel
  // (and there is no data-parallel region at all unless there are more than two full waves).
  uint64_t const dp_waves = full_waves > 1 ? full_waves - 1 : 0;
  uint64_t const dp_tiles = dp_waves * ctas_per_wave;
  uint64_t const sk_tiles = output_tiles - dp_tiles;

  // No wave quantization => nothing to split; and a tile with too few k-iterations is not worth
  // splitting even if the waves do quantize.
  if (full_waves == total_waves || min_k_tiles <= MinItersPerSkUnit) {
    return d;
  }

  if (requested == DecompositionMode::Heuristic) {
    // CUTLASS's rudimentary heuristic, kept verbatim: do not split if the tail wave is more than
    // half full.
    uint64_t const tail_tiles = output_tiles - full_waves * ctas_per_wave;
    if (2 * tail_tiles >= ctas_per_wave) {
      return d;
    }
  }

  d.dp_tiles = dp_tiles;
  d.sk_tiles = static_cast<uint32_t>(sk_tiles);
  return d;
}

/// Phase 2: how finely the split region's k-iterations are carved up.
///
/// This is `get_num_sk_units` (`tile_scheduler_params.h:1118-1141`).  Its `k_tiles_per_output_tile
/// * sk_tiles` -- the k-iterations of the split region -- is the one quantity a heterogeneous
/// problem set cannot express as a product, so it is passed in already summed.  The "big units
/// first" remainder arithmetic is CUTLASS's, verbatim.
template <uint32_t MinItersPerSkUnit>
inline void
size_sk_units(GroupStreamKDecomposition& d, uint64_t sk_k_tiles_total, uint64_t ctas_per_wave) {
  if (d.sk_tiles == 0) {
    return;
  }
  uint64_t const min_sized_units = sk_k_tiles_total / MinItersPerSkUnit;
  uint64_t const sk_units = min_sized_units < ctas_per_wave ? min_sized_units : ctas_per_wave;
  if (sk_units == 0) {
    d.dp_tiles = d.output_tiles;
    d.sk_tiles = 0;
    return;
  }
  d.sk_units = static_cast<uint32_t>(sk_units);
  d.sk_iters = sk_k_tiles_total;
  d.iters_per_unit = static_cast<uint32_t>(sk_k_tiles_total / sk_units);
  d.big_units = static_cast<uint32_t>(sk_k_tiles_total % sk_units);
  d.mode = DecompositionMode::StreamK;
}

}  // namespace detail

/// The scheduler's kernel-side Params.
///
/// *Inherits* `PersistentTileSchedulerSm90GroupParams` rather than holding one by value: the array
/// kernel reads `params.scheduler.raster_order_` directly
/// (`sm90_gemm_array_tma_warpspecialized_cooperative.hpp:408`), so the stock fields have to sit at
/// the top level of this type rather than behind a member.
///
/// The only addition is the decomposition, and it is a fixed-size bundle of scalars -- whatever the
/// scheduler needs to know *per problem* it reads from `problem_shapes_`, which is already resident
/// on the device.  That is not just a preference: this operator re-runs `initialize()` on every
/// `launch()` and the timing harness wraps the whole launch, so a host-built per-problem table would
/// put its upload inside the measured region -- measured at +23% on a 25 us kernel.
template <class GroupProblemShape>
struct GroupStreamKSchedulerParams
  : cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90GroupParams<GroupProblemShape> {
  using BaseParams =
    cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90GroupParams<GroupProblemShape>;

  GroupStreamKDecomposition decomp{};

  /// The scheduler's own slice of the kernel workspace: `[partial sums][locks]`, sized and laid out
  /// by `reduction_bytes` / `barrier_bytes` below.  The base has no such member -- the stock grouped
  /// decomposition never writes a partial -- so it is declared here rather than inherited.
  void* reduction_workspace_ = nullptr;
};

/// The grouped stream-K scheduler.
///
/// Derives from `PersistentTileSchedulerSm90Group` for the pipeline-based scheduling API the array
/// kernel drives, and for its group-aware tile decode (`get_work_idx_m_and_n` already knows about
/// the swizzle, the cluster divmods and the per-group shapes).
///
/// The linear index space is what changes: the base's is `[0, blocks_across_problem_)` with one
/// whole output tile per index, this one's is `[0, sk_units)` of split units followed by one item
/// per data-parallel tile.  Everything the base schedules funnels through
/// `get_current_work_for_linear_idx`, so that plus `continue_current_work` -- the hook
/// `fetch_next_work` consults before draining the ring, and the reason a unit may cover more tiles
/// than the array kernel would otherwise allow -- are the two behavioural overrides.  The rest is
/// the static stream-K hooks and the workspace sizing.
///
/// `WorkTileInfo`'s first four members must stay `M_idx, N_idx, L_idx, is_valid_tile` in that exact
/// order: `get_work_idx_m_and_n` returns its result by brace-init
/// (`{minor, major, group_idx, valid}`), so a field inserted ahead of them would silently swap
/// `L_idx` for the K index with no compile error.
template <class GroupProblemShape, int SchedulerPipelineStageCount, class TileShape,
          class ClusterShape>
class GroupStreamKScheduler
  : public cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Group<GroupProblemShape,
                                                                            SchedulerPipelineStageCount> {
  using Base = cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Group<
    GroupProblemShape, SchedulerPipelineStageCount>;

  /// Where the split arithmetic and the reduction-sizing arithmetic come from.  Referenced rather
  /// than restated, so this file and CUTLASS cannot drift apart.
  using StreamKParams = cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams;

  /// CUTLASS's own floor on stream-K unit size (`tile_scheduler_params.h:470`).
  static constexpr uint32_t MinItersPerSkUnit = StreamKParams::min_iters_per_sk_unit_;

public:
  using Params = GroupStreamKSchedulerParams<GroupProblemShape>;
  static constexpr bool IsDynamicPersistent = false;

  using Pipeline = typename Base::Pipeline;
  using PipelineStorage = typename Base::PipelineStorage;
  using PipelineState = typename Base::PipelineState;
  using ThrottlePipeline = typename Base::ThrottlePipeline;
  using ThrottlePipelineStorage = typename Base::ThrottlePipelineStorage;
  using RasterOrder = typename Base::RasterOrder;
  using RasterOrderOptions = typename Base::RasterOrderOptions;

  struct WorkTileInfo {
    // Order is load-bearing -- see the class comment.
    int32_t M_idx = 0;
    int32_t N_idx = 0;
    int32_t L_idx = 0;  // group (problem) index
    int32_t is_valid_tile = 0;

    /// The K range this work item covers, for a split fragment.
    ///
    /// `k_tile_count == 0` is the data-parallel marker and not an empty range: a whole-tile item's
    /// count is its tile's full K extent, which the base derives from the `ProblemShape_MNKL` the
    /// caller already holds (`sm90_tile_scheduler_group.hpp:526`), so writing it here as well would
    /// re-read the device problem set for a number the kernel has in hand -- measured as a per-work
    /// -item cost in whole-kernel instructions.  A fragment always covers at least one k-tile, so
    /// the marker is unambiguous.
    int32_t K_idx = 0;
    uint32_t k_tile_count = 0;
    uint32_t k_tile_remaining = 0;

    /// The end (exclusive) of this work item's whole stream-K unit, in the split region's flat
    /// k-iteration space; 0 for a data-parallel item.  It rides in the work item -- rather than in
    /// the scheduler, as the stock stream-K scheduler keeps its `unit_iter_start_` -- because the
    /// continuation is computed on the *consumer* side: the array kernel's mainloop and epilogue
    /// warps each own a scheduler object whose `current_work_linear_idx_` is never advanced, since
    /// it is the scheduler warp that walks the index space.  Only what travels through the ring is
    /// in step with the item being consumed.
    uint32_t unit_iter_end = 0;

    /// Whether this fragment covers the *end* of its output tile, i.e. whether this CTA is the last
    /// one to contribute to it and therefore owns its epilogue.  True for a whole-tile work item as
    /// well; see `compute_epilogue` and `requires_fixup` for how the two use it.
    bool is_final_split = false;

    /// The group-major index of the output tile this fragment belongs to.  It is the reduction's
    /// only key: the stock scheduler recovers a tile index from (M, N, L)
    /// (`sm90_tile_scheduler_stream_k.hpp:596-608`), but here the tile index is the split region's
    /// own currency and inverting the decode would mean walking the groups a second time.
    int32_t sk_tile_id = 0;

    CUTLASS_HOST_DEVICE bool is_valid() const { return is_valid_tile != 0; }

    /// Whether this work item is a *partial* sum: a split fragment that does not cover its whole
    /// output tile.  A whole-tile item is marked by `k_tile_count == 0`, and a fragment that happens
    /// to cover a whole tile (`K_idx == 0 && is_final_split`) is not partial either -- both compute a
    /// complete C and need no workspace.  That second case is not hypothetical: `sk_units` is capped
    /// at the CTA count, so on a problem set whose tiles are few and wide a unit can cover one whole.
    CUTLASS_HOST_DEVICE bool is_partial_sum() const {
      return k_tile_count != 0 && !(K_idx == 0 && is_final_split);
    }

    CUTLASS_HOST_DEVICE int32_t reduction_subtile_idx() const { return -1; }

    CUTLASS_HOST_DEVICE static WorkTileInfo invalid_work_tile() { return WorkTileInfo{}; }
  };

  using SchedulerResponse = WorkTileInfo;

  class SharedStorage {
  public:
    CUTLASS_DEVICE PipelineStorage pipeline() { return pipeline_; }
    CUTLASS_DEVICE ThrottlePipelineStorage throttle_pipeline() { return ThrottlePipelineStorage{}; }
    CUTLASS_DEVICE SchedulerResponse* data() { return data_; }

  private:
    alignas(16) PipelineStorage pipeline_;
    alignas(16) SchedulerResponse data_[SchedulerPipelineStageCount];
  };

  /// The scheduler's host interface.  `decomp` is *input*, not output: the host plan computes it
  /// once from the problem set it already holds and passes it in, because the two functions that
  /// need it to size the workspace are called with a default-constructed problem shape -- see
  /// `get_workspace_size`.  Carrying it here also keeps the decomposition off the per-launch host
  /// path: the plan re-runs `initialize()` on every `launch()`, and this operator's kernels are
  /// short enough (25 us at the smallest point) for a 16-group host walk to be visible in a timing
  /// window that brackets the launch.
  struct Arguments {
    int max_swizzle_size = 1;
    RasterOrderOptions raster_order = RasterOrderOptions::AlongM;
    DecompositionMode decomposition_mode = DecompositionMode::Heuristic;
    GroupStreamKDecomposition decomp{};
  };

  /// The stream-K extras, unpacked from Params in the constructor.  Everything else the scheduler
  /// reads -- the shapes, the cta/cluster geometry, the swizzle -- is already in the base's own
  /// `scheduler_params` copy, so only what the base has no notion of is kept here.
  GroupStreamKDecomposition decomp_;

  //
  // Host-side decomposition
  //

  static constexpr int TB_M = cute::size<0>(TileShape{});
  static constexpr int TB_N = cute::size<1>(TileShape{});
  static constexpr int TB_K = cute::size<2>(TileShape{});
  static constexpr int CM = cute::size<0>(ClusterShape{});
  static constexpr int CN = cute::size<1>(ClusterShape{});

  /// The reduction runs one *warpgroup* at a time -- each consumer warpgroup reduces its own slice
  /// of the tile through its own lock -- so the barrier and the striped reduce both work in units of
  /// `NumThreadsPerWarpGroup`, exactly as the stock stream-K scheduler's do.
  static constexpr uint32_t ReductionThreads = cutlass::NumThreadsPerWarpGroup;
  using BarrierType = typename cutlass::NamedBarrierManager<1>::T;

  /// Whether this instantiation may split at all.  Two structural limits, both of which the split
  /// path above would have to grow a mechanism to cross:
  ///
  /// * a cluster.  CUTLASS shares one K range across a cluster's CTAs (`div_cluster_size` and the
  ///   per-cluster `k_tiles_per_sk_unit`, `sm90_tile_scheduler_stream_k.hpp:1063-1087`), because the
  ///   cluster multicasts one operand load and so cannot have its CTAs on different k-ranges of
  ///   different tiles.  That mechanism is not here.
  /// * a swizzle.  The tile index space this file counts (`tiles_of_group`) mirrors the base's
  ///   `get_tiled_cta_shape_mnl`, which does not pad; the base's *decode* additionally rounds each
  ///   group up by the swizzle size (`sm90_tile_scheduler_group.hpp:365-372`).  The two agree only
  ///   while the swizzle cannot pad -- which is always the case at `max_swizzle_size <= 1`, the
  ///   default and the only setting any result in this repository was measured at.
  static constexpr bool CanSplit = (CM * CN == 1);

  /// The whole decomposition, as a pure function of the problem set and the tile shape.  Sizing,
  /// filling and `to_underlying_arguments` all call this, so they cannot disagree.
  static GroupStreamKDecomposition
  compute_decomposition(GroupProblemShape const& shapes, uint64_t ctas_per_wave,
                        int max_swizzle_size, DecompositionMode requested) {
    DecompositionMode const effective =
      (CanSplit && max_swizzle_size <= 1) ? requested : DecompositionMode::DataParallel;

    uint64_t const groups = static_cast<uint64_t>(shapes.groups());

    uint64_t tiles_total = 0;
    uint32_t min_k_tiles = UINT32_MAX;
    for (uint64_t g = 0; g < groups; ++g) {
      auto shape = shapes.get_host_problem_shape(static_cast<int32_t>(g));
      tiles_total += detail::tiles_of_group<TB_M, TB_N, CM, CN>(cute::get<0>(shape),
                                                                cute::get<1>(shape));
      uint32_t const kt = detail::k_tiles_of<TB_K>(cute::get<2>(shape));
      if (kt < min_k_tiles) min_k_tiles = kt;
    }
    if (groups == 0) {
      min_k_tiles = 0;
    }

    GroupStreamKDecomposition d = detail::select_sk_region<MinItersPerSkUnit>(
      tiles_total, min_k_tiles, ctas_per_wave, effective);
    if (d.sk_tiles == 0) {
      return d;
    }

    // Walk the groups once more to sum, over the tiles in the split region and over those in the
    // data-parallel region, their k-iterations -- the one quantity a heterogeneous problem set
    // cannot express as a per-tile constant times a tile count.  The split region is the *first*
    // `sk_tiles` tiles and the groups own them in order, so this cuts each group's tile run at
    // `sk_tiles` and adds the two halves to the two totals.
    uint64_t tile_cursor = 0, dp_iters = 0, sk_iters = 0;
    for (uint64_t g = 0; g < groups; ++g) {
      auto shape = shapes.get_host_problem_shape(static_cast<int32_t>(g));
      uint64_t const tiles = detail::tiles_of_group<TB_M, TB_N, CM, CN>(cute::get<0>(shape),
                                                                       cute::get<1>(shape));
      uint64_t const kt = detail::k_tiles_of<TB_K>(cute::get<2>(shape));
      uint64_t const hi = tile_cursor + tiles;
      uint64_t const cut = d.sk_tiles;

      uint64_t const sk_hi = hi < cut ? hi : cut;
      if (sk_hi > tile_cursor) {
        sk_iters += (sk_hi - tile_cursor) * kt;
      }
      uint64_t const dp_lo = tile_cursor > cut ? tile_cursor : cut;
      if (hi > dp_lo) {
        dp_iters += (hi - dp_lo) * kt;
      }
      tile_cursor = hi;
    }
    d.dp_iters = dp_iters;

    detail::size_sk_units<MinItersPerSkUnit>(d, sk_iters, ctas_per_wave);
    return d;
  }

  template <class TShape, class CShape>
  static Params
  to_underlying_arguments(GroupProblemShape problem_shapes, TShape tile_shape, CShape cluster_shape,
                          cutlass::KernelHardwareInfo const& hw_info, Arguments const& arguments,
                          void* workspace = nullptr,
                          [[maybe_unused]] const uint32_t epilogue_subtile = 1,
                          [[maybe_unused]] uint32_t ktile_start_alignment_count = 1u) {
    static_assert(cute::is_static<TShape>::value);
    static_assert(cute::is_static<CShape>::value);

    Params params;
    params.initialize(Base::get_tiled_cta_shape_mnl(problem_shapes, hw_info, tile_shape,
                                                    cluster_shape),
                      problem_shapes, cutlass::gemm::to_gemm_coord(tile_shape),
                      cutlass::gemm::to_gemm_coord(cluster_shape), hw_info, arguments.max_swizzle_size,
                      arguments.raster_order);

    params.decomp = arguments.decomp;
    params.reduction_workspace_ = workspace;

    if (nullptr != nullptr) {
      fprintf(stderr,
              "[streamk] tiles=%llu sk_tiles=%u dp_tiles=%llu sk_units=%u iters=%llu "
              "iters/unit=%u big_units=%u ctas/wave=%u mode=%d\n",
              (unsigned long long)params.decomp.output_tiles, params.decomp.sk_tiles,
              (unsigned long long)params.decomp.dp_tiles, params.decomp.sk_units,
              (unsigned long long)params.decomp.sk_iters, params.decomp.iters_per_unit,
              params.decomp.big_units, params.decomp.ctas_per_wave, (int)params.decomp.mode);
    }
    return params;
  }

  /// The launch geometry, and with a split decomposition the grid is where the split's parallelism
  /// comes from: the base clamps the grid to `min(sm_count, <problem blocks>)`
  /// (`tile_scheduler_params.h:1748`), and the problem blocks are the *tiles* -- at B=512 that is 64
  /// CTAs on 132 SMs, which is the whole reason there is something to gain here.  Passing the unit
  /// count keeps the same clamp applied to the number of work items instead, so the split region
  /// gets a CTA per unit (up to the SM count) rather than a CTA per tile.
  template <class TShape, class CShape>
  CUTLASS_HOST_DEVICE static dim3
  get_grid_shape([[maybe_unused]] Params const& params, GroupProblemShape const& problem_shapes,
                 TShape tile_shape, CShape cluster_shape, cutlass::KernelHardwareInfo hw_info,
                 Arguments arguments, bool truncate_by_problem_size = true) {
    dim3 problem_blocks =
      Base::get_tiled_cta_shape_mnl(problem_shapes, hw_info, tile_shape, cluster_shape);
    if (params.decomp.sk_units != 0) {
      problem_blocks = dim3(static_cast<uint32_t>(params.decomp.sk_units + params.decomp.dp_tiles),
                            1, 1);
    }
    return Base::Params::get_grid_shape(problem_blocks, cutlass::gemm::to_gemm_coord(cluster_shape),
                                        hw_info, arguments.max_swizzle_size, arguments.raster_order,
                                        truncate_by_problem_size);
  }

  static bool can_implement(Arguments const&, cutlass::KernelHardwareInfo const&) { return true; }

  //
  // Workspace
  //

  /// The CUTLASS stream-K arithmetic, reused rather than restated: one full `TileShape` of
  /// accumulator per split tile, L2-aligned, and a lock per (split tile, consumer warpgroup).
  template <class ElementAccumulator>
  CUTLASS_HOST_DEVICE static size_t reduction_bytes(uint64_t sk_tiles) {
    return StreamKParams::get_reduction_workspace_size(
      sk_tiles, cutlass::gemm::to_gemm_coord(TileShape{}),
      cutlass::sizeof_bits<ElementAccumulator>::value);
  }

  CUTLASS_HOST_DEVICE static size_t barrier_bytes(uint64_t sk_tiles, uint32_t mma_warp_groups) {
    return StreamKParams::get_barrier_workspace_size(
      sk_tiles, mma_warp_groups, cutlass::sizeof_bits<BarrierType>::value);
  }

  /// Both statics below size from `Arguments` and deliberately *not* from the problem shape they are
  /// handed.  The array kernel calls them with `typename ProblemShape::UnderlyingProblemShape{}` --
  /// a default-constructed, all-zero shape (`sm90_gemm_array_tma_warpspecialized_cooperative.hpp:305`
  /// and `:362`) -- so a decomposition recomputed here would count the tiles of an empty problem set
  /// and hand `fixup` a buffer of the wrong size.  The stock stream-K scheduler has the same
  /// signature and the same caller, but it is never instantiated with the ptr-array kernel (see the
  /// file header), which is why the mismatch has no symptom upstream.  The host plan holds the real
  /// problem set, so it computes the decomposition once and passes it in `Arguments`; all that is
  /// left for these two to do is arithmetic on `sk_tiles`.
  template <class ProblemShape_, class ElementAccumulator>
  static size_t
  get_workspace_size(Arguments const& args, ProblemShape_, cutlass::KernelHardwareInfo const&,
                     uint32_t mma_warp_groups, const uint32_t = 1, uint32_t = 1) {
    uint64_t const sk_tiles = args.decomp.sk_tiles;
    if (sk_tiles == 0) {
      return 0;
    }
    if (nullptr != nullptr) {
      fprintf(stderr, "[streamk] ws sk_tiles=%llu redu=%zu barrier=%zu warp_groups=%u tile_c=%d\n",
              (unsigned long long)sk_tiles, reduction_bytes<ElementAccumulator>(sk_tiles),
              barrier_bytes(sk_tiles, mma_warp_groups), mma_warp_groups, TB_M * TB_N);
    }
    return reduction_bytes<ElementAccumulator>(sk_tiles) + barrier_bytes(sk_tiles, mma_warp_groups);
  }

  /// Only the locks are cleared.  The partial sums need no clearing because the fragment at `K_idx`
  /// 0 *stores* into the workspace instead of accumulating onto whatever was there.  Clearing here,
  /// in the call the kernel makes on every `initialize()`, is what keeps the locks zero at the start
  /// of every launch -- the plan re-runs that call per launch, so a one-time clear would not do.
  template <class ProblemShape_, class ElementAccumulator>
  static cutlass::Status
  initialize_workspace(Arguments const& args, void* workspace, cudaStream_t stream, ProblemShape_,
                       cutlass::KernelHardwareInfo const&, uint32_t mma_warp_groups,
                       const uint32_t = 1, uint32_t = 1,
                       cutlass::CudaHostAdapter* cuda_adapter = nullptr) {
    uint64_t const sk_tiles = args.decomp.sk_tiles;
    if (sk_tiles == 0) {
      return cutlass::Status::kSuccess;
    }
    uint8_t* locks =
      reinterpret_cast<uint8_t*>(workspace) + reduction_bytes<ElementAccumulator>(sk_tiles);
    return cutlass::zero_workspace(locks, barrier_bytes(sk_tiles, mma_warp_groups), stream,
                                   cuda_adapter);
  }

  //
  // Device-side scheduling
  //

  GroupStreamKScheduler() = default;

  /// The base constructor is reused verbatim: it seeds `current_work_linear_idx_`,
  /// `current_group_info_`, `scheduler_params` and the two-deep problem-shape prefetch.  The
  /// response pointer is cast because the base types the ring element as *its* `WorkTileInfo` while
  /// the kernel allocates the ring in ours; the base only ever stores it as `void*` and
  /// dereferences it through the callback's type, so no base-typed access survives.
  CUTLASS_DEVICE explicit GroupStreamKScheduler(Params const& params_, void* response_ptr)
    : Base(params_, reinterpret_cast<typename Base::SchedulerResponse*>(response_ptr)),
      decomp_(params_.decomp) {}

  /// A tile index resolved to a whole-tile work item.
  ///
  /// This is the base's own group-aware decode, so the split region lands on exactly the tiles the
  /// data-parallel region would.  The K fields are left default, i.e. `k_tile_count == 0`: a tile
  /// decoded this way covers all of its own K.  See `assign_sk_fragment` for why every caller must
  /// visit tiles in increasing order.
  CUTLASS_DEVICE WorkTileInfo decode_tile(uint64_t tile_idx) {
    return Base::template get_work_idx_m_and_n<WorkTileInfo>(
      tile_idx, this->current_group_info_, this->scheduler_params.problem_shapes_,
      this->cached_problem_shapes_, this->scheduler_params.cta_shape_,
      this->scheduler_params.cluster_shape_, this->scheduler_params.divmod_cluster_shape_major_,
      this->scheduler_params.divmod_cluster_shape_minor_, this->scheduler_params.divmod_cta_shape_m_,
      this->scheduler_params.divmod_cta_shape_n_, this->scheduler_params.max_swizzle_size_,
      this->scheduler_params.raster_order_);
  }

  /// A point in the split region's flat k-iteration space, resolved to the tile containing it.
  struct SkPoint {
    int32_t tile_id = -1;          ///< the tile's group-major index
    uint64_t tile_iter_start = 0;  ///< the tile's first iteration in that space
    uint32_t k_tiles = 0;          ///< the tile's k extent
  };

  /// Resolves a flat k-iteration of the split region to its output tile.
  ///
  /// A walk, not a division, and that is the whole content of "the problem set is heterogeneous":
  /// the region spans whole groups and `k_tiles` changes from one group to the next, so there is no
  /// single divisor to take the iteration apart with.  Reading the shapes off the device problem set
  /// is what the loop body does; it runs once per work item, on one warp, next to the same loads the
  /// base's decode makes.
  CUTLASS_DEVICE SkPoint locate_sk_iter(uint64_t iter) const {
    uint64_t tile_cursor = 0, iter_cursor = 0;
    int32_t const groups = this->scheduler_params.problem_shapes_.groups();
    for (int32_t g = 0; g < groups; ++g) {
      auto shape = this->scheduler_params.problem_shapes_.get_problem_shape(g);
      uint64_t const tiles =
        detail::tiles_of_group<TB_M, TB_N, CM, CN>(cute::get<0>(shape), cute::get<1>(shape));
      uint32_t const kt = detail::k_tiles_of<TB_K>(cute::get<2>(shape));
      uint64_t const span = tiles * uint64_t(kt);
      if (iter < iter_cursor + span) {
        uint64_t const local = iter - iter_cursor;
        uint64_t const in_tile = local / kt;
        return SkPoint{static_cast<int32_t>(tile_cursor + in_tile),
                       iter_cursor + in_tile * uint64_t(kt), kt};
      }
      tile_cursor += tiles;
      iter_cursor += span;
    }
    // Unreachable: every caller is bounded by `sk_iters`.  A `tile_id` of -1 makes that visible
    // through the decode rather than by reading past the problem set.
    return SkPoint{};
  }

  /// The slice of the split region's flat iteration space that unit `u` covers.
  ///
  /// "Big units first": the remainder goes to the first `big_units` units rather than being spread,
  /// which is the arithmetic of `divmod_k_tiles_per_sk_big_unit_` in
  /// `sm90_tile_scheduler_stream_k.hpp:1101-1116`.  Here the total is already summed over a
  /// heterogeneous problem set, so the split is over iterations directly.
  CUTLASS_DEVICE void sk_unit_range(uint64_t u, uint64_t& start, uint32_t& len) const {
    uint64_t const big = decomp_.big_units;
    start = u * decomp_.iters_per_unit + (u < big ? u : big);
    len = decomp_.iters_per_unit + (u < big ? 1u : 0u);
  }

  /// Fills in the next fragment of the current work item's unit.
  ///
  /// `wt.k_tile_remaining` counts the unit's iterations from the fragment that is about to be
  /// assigned to the end of the unit, so the fragment starts at `unit_iter_end - k_tile_remaining`
  /// and runs to the next tile boundary, or to the end of the unit if that comes first.
  ///
  /// The stock stream-K scheduler walks a unit back-to-front instead
  /// (`sm90_tile_scheduler_stream_k.hpp:925-933`: doing so leaves each tile's *ending* portion to be
  /// computed later, "to reduce the likelihood of blocking on other work" in the reduction).  The
  /// direction is swapped here for a structural reason rather than a preference: this scheduler
  /// decodes tiles through the base's group cursor, and that cursor only ever moves forward
  /// (`sm90_tile_scheduler_group.hpp:346-380`).  A unit's tiles visited back-to-front would be
  /// decoded against a stale `start_linear_idx` -- silently, onto the wrong output tile.  Front to
  /// back keeps every thread's tile index monotonic, which is exactly what a monotonic cursor needs.
  /// What is given up is the contention heuristic, and it is load-bearing under the ordered reduction
  /// in `fixup`: a unit's forward walk reaches a tile's *tail* before the next tile's head, so the
  /// fragment that ends tile T waits on tile T's own starting fragment -- which a straddling unit
  /// computes only after the fragment that ends tile T-1.
  CUTLASS_DEVICE void assign_sk_fragment(WorkTileInfo& wt) {
    uint64_t const begin = uint64_t(wt.unit_iter_end) - wt.k_tile_remaining;
    SkPoint const p = locate_sk_iter(begin);
    uint64_t const tile_hi = p.tile_iter_start + p.k_tiles;
    uint64_t const end = tile_hi < uint64_t(wt.unit_iter_end) ? tile_hi : uint64_t(wt.unit_iter_end);

    WorkTileInfo const decoded = decode_tile(uint64_t(p.tile_id));
    wt.M_idx = decoded.M_idx;
    wt.N_idx = decoded.N_idx;
    wt.L_idx = decoded.L_idx;
    wt.is_valid_tile = decoded.is_valid_tile;
    wt.K_idx = static_cast<int32_t>(begin - p.tile_iter_start);
    wt.k_tile_count = static_cast<uint32_t>(end - begin);
    wt.sk_tile_id = p.tile_id;
    // `end < tile_hi` only when the unit ran out inside this tile, i.e. exactly when some later unit
    // covers the tile's tail -- and that later unit, not this one, is the one whose C is complete.
    wt.is_final_split = (end == tile_hi);
  }

  /// The one override: everything the base schedules funnels through here.
  CUTLASS_DEVICE WorkTileInfo get_current_work_for_linear_idx(uint64_t linear_idx) {
    if (decomp_.sk_units == 0) {
      // Data-parallel decomposition: the linear index *is* the output tile index, and the base's
      // bound is the only thing that ends the walk.
      if (this->scheduler_params.pre_processed_problem_shapes &&
          linear_idx >= this->scheduler_params.blocks_across_problem_) {
        return WorkTileInfo::invalid_work_tile();
      }
      return decode_tile(linear_idx);
    }

    // Split decomposition: `sk_units` units, then one item per data-parallel tile.  The tiles of
    // both regions are visited in increasing index order -- the units cover the split region's
    // iterations in order, and the split region is the leading tile range.
    if (linear_idx < decomp_.sk_units) {
      WorkTileInfo wt;
      uint64_t start;
      uint32_t len;
      sk_unit_range(linear_idx, start, len);
      wt.unit_iter_end = static_cast<uint32_t>(start + len);
      wt.k_tile_remaining = len;
      assign_sk_fragment(wt);
      return wt;
    }

    uint64_t const unit_span = linear_idx - decomp_.sk_units;
    if (unit_span >= decomp_.dp_tiles) {
      return WorkTileInfo::invalid_work_tile();
    }
    return decode_tile(uint64_t(decomp_.sk_tiles) + unit_span);
  }

  // The four methods below are the base's own bodies, restated only because their default callbacks
  // and parameters name `WorkTileInfo` -- and inside the base that name is *its* four-field work
  // item, which would silently drop the K range at the boundary.  The behaviour is otherwise
  // identical, so `get_current_work_for_linear_idx` above stays the single point of difference.

  template <class ClusterShape_, typename CallbackBeforeCommit = WorkTileInfo (*)(WorkTileInfo)>
  CUTLASS_DEVICE auto
  initial_work_tile_info(ClusterShape_,
                         CallbackBeforeCommit callback_before_commit =
                           [](WorkTileInfo response) { return response; }) {
    auto work_tile = get_current_work_for_linear_idx(this->current_work_linear_idx_);
    using WorkTileWithCallbackInfo = decltype(callback_before_commit(work_tile));
    WorkTileWithCallbackInfo work_tile_with_callback_info = work_tile;
    if (work_tile_with_callback_info.is_valid()) {
      work_tile_with_callback_info = callback_before_commit(work_tile);
    }
    return work_tile_with_callback_info;
  }

  template <typename TileSchedulerPipeline, typename TileSchedulerPipelineState,
            typename CallbackBeforeCommit = WorkTileInfo (*)(WorkTileInfo)>
  CUTLASS_DEVICE auto
  advance_to_next_work(TileSchedulerPipeline& scheduler_pipeline,
                       TileSchedulerPipelineState scheduler_pipe_producer_state,
                       uint32_t advance_count = 1,
                       CallbackBeforeCommit callback_before_commit =
                         [](WorkTileInfo info) { return info; }) {
    this->current_work_linear_idx_ += this->total_grid_size_ * uint64_t(advance_count);
    auto work_tile = get_current_work_for_linear_idx(this->current_work_linear_idx_);
    using WorkTileWithCallbackInfo = decltype(callback_before_commit(work_tile));
    WorkTileWithCallbackInfo work_tile_with_callback_info = work_tile;
    scheduler_pipeline.producer_acquire(scheduler_pipe_producer_state);
    if (work_tile_with_callback_info.is_valid()) {
      work_tile_with_callback_info = callback_before_commit(work_tile);
    }
    if (cute::elect_one_sync()) {
      reinterpret_cast<WorkTileWithCallbackInfo*>(this->response_ptr_)
        [scheduler_pipe_producer_state.index()] = work_tile_with_callback_info;
      cutlass::arch::fence_view_async_shared();
      scheduler_pipeline.producer_commit(scheduler_pipe_producer_state);
    }
    return cute::make_tuple(work_tile_with_callback_info, true);
  }

  template <typename WorkTileWithCallbackInfo, typename TileSchedulerPipeline,
            typename TileSchedulerPipelineState>
  CUTLASS_DEVICE auto
  fetch_next_work(WorkTileWithCallbackInfo work_tile_with_callback_info,
                  TileSchedulerPipeline& scheduler_pipeline,
                  TileSchedulerPipelineState scheduler_pipe_consumer_state) {
    if (continue_current_work(work_tile_with_callback_info)) {
      // `false`, not the base's `true`: a continuation is computed locally from the item just
      // consumed and takes no slot out of the scheduler ring, so the caller must not advance its
      // consumer state.  The base returns `true` here because the stock grouped scheduler never
      // continues (`sm90_tile_scheduler_group.hpp:505`) and the path is dead upstream; for the
      // non-array stream-K scheduler, which is where the shape of this method comes from, the
      // pipeline is `PipelineEmpty` (`sm90_tile_scheduler_stream_k.hpp:708`), so the flag costs
      // nothing there either.  Under a real pipeline it would desynchronise the producer, which
      // commits one ring slot per work item, from a consumer that had skipped one.
      return cute::make_tuple(work_tile_with_callback_info, false);
    }
    scheduler_pipeline.consumer_wait(scheduler_pipe_consumer_state);
    work_tile_with_callback_info =
      reinterpret_cast<WorkTileWithCallbackInfo*>(this->response_ptr_)
        [scheduler_pipe_consumer_state.index()];
    cutlass::arch::fence_view_async_shared();
    scheduler_pipeline.consumer_release(scheduler_pipe_consumer_state);
    return cute::make_tuple(work_tile_with_callback_info, true);
  }

  //
  // Stream-K hooks
  //

  /// Hands the same work item its unit's next tile, when the unit has one.
  ///
  /// The comparison is `> k_tile_count` rather than "leftover is non-zero" because
  /// `k_tile_remaining` is what is left *including* the fragment just finished: equal means this
  /// fragment ended at the unit's end.  A whole-tile work item has both at zero and so never
  /// continues, which is what preserves the array kernel's one-tile-per-item structure for the
  /// data-parallel region.
  CUTLASS_DEVICE bool continue_current_work(WorkTileInfo& wt) {
    if (!wt.is_valid() || wt.k_tile_remaining <= wt.k_tile_count) {
      return false;
    }
    wt.k_tile_remaining -= wt.k_tile_count;
    assign_sk_fragment(wt);
    return wt.is_valid();
  }

  /// Only the CTA holding a tile's final fragment stores it.  Every other fragment of a split tile
  /// leaves its partial in the workspace for `fixup` to combine, and its epilogue is skipped
  /// outright -- which is most of what a split costs back: the tile's D is written once, not once
  /// per fragment.
  CUTLASS_HOST_DEVICE static bool
  compute_epilogue(WorkTileInfo const& work_tile_info, Params const&) {
    return work_tile_info.is_valid() &&
           (work_tile_info.k_tile_count == 0 || work_tile_info.is_final_split);
  }

  CUTLASS_HOST_DEVICE static bool
  valid_warpgroup_in_work_tile(WorkTileInfo const&) {
    return true;
  }

  /// Combines the fragments of a split tile.
  ///
  /// The mechanism is the stock stream-K scheduler's, minus the branches it needs for
  /// decompositions this file never produces (no separate-reduction units, and one reduction mode):
  ///
  ///   * the workspace holds one C tile per split tile, sliced by consumer warpgroup, so a
  ///     warpgroup's slice is private to it and its lock can be too;
  ///   * the lock is a count of k-iterations that have arrived, so "wait for the value I start at"
  ///     is the whole synchronisation -- the fragment at `K_idx` waits for `K_idx`, the one at
  ///     `K_idx == 0` stores instead of accumulating, and no arrival order has to be assumed;
  ///   * every accumulating fragment goes through `BlockStripedReduce::reduce`, which is one
  ///     `atomicAdd` per element, so a fragment that is not the tile's first pays a full C tile of
  ///     read-modify-write traffic through L2 and one that is not the tile's last pays a full C tile
  ///     of it in atomics.  That traffic, not the mainloop, is what a split of these shapes costs.
  ///
  /// Grouped, this is the one place the heterogeneous problem set shows through: the tile is named
  /// by its group-major index, which the work item carries, rather than recovered from (M, N, L).
  template <class FrgTensorC>
  CUTLASS_DEVICE static void
  fixup(Params const& params, WorkTileInfo const& work_tile_info, FrgTensorC& accumulators,
        uint32_t num_barriers, uint32_t barrier_idx) {
    if (!work_tile_info.is_partial_sum()) {
      return;
    }

    using ElementAccumulator = typename FrgTensorC::value_type;
    using AccumulatorArrayT = cutlass::Array<ElementAccumulator, cute::size(FrgTensorC{})>;
    using BlockStripedReduceT = cutlass::BlockStripedReduce<ReductionThreads, AccumulatorArrayT>;

    // `StreamkBarrier0` and a width of two is the stock choice
    // (`sm90_tile_scheduler_stream_k.hpp:415-417`); the barrier only serialises a warpgroup's own
    // accesses to its own lock slot, so nothing about it depends on the grouped shape.
    constexpr uint32_t BarrierOffset =
      static_cast<uint32_t>(cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);
    using BarrierManager = cutlass::NamedBarrierManager<ReductionThreads, BarrierOffset, 2>;

    uint64_t const lock_idx = uint64_t(work_tile_info.sk_tile_id) * num_barriers + barrier_idx;
    uint64_t const tile_offset =
      uint64_t(TB_M) * TB_N * uint64_t(work_tile_info.sk_tile_id) +
      uint64_t(cute::size(accumulators)) * barrier_idx * ReductionThreads;

    auto* tile_workspace = reinterpret_cast<AccumulatorArrayT*>(
      reinterpret_cast<ElementAccumulator*>(params.reduction_workspace_) + tile_offset);
    auto* locks = reinterpret_cast<BarrierType*>(
      reinterpret_cast<uint8_t*>(params.reduction_workspace_) +
      reduction_bytes<ElementAccumulator>(params.decomp.sk_tiles));
    auto* accumulator_array = reinterpret_cast<AccumulatorArrayT*>(accumulators.data());

    uint32_t const thread_in_group = threadIdx.x % ReductionThreads;

    if (work_tile_info.is_final_split) {
      // This fragment ends the tile, so it is the one that owns the epilogue: it waits for the
      // running count to reach its own start, which is where the fragments before it leave it.
      BarrierManager::wait_eq(barrier_idx, locks, thread_in_group, static_cast<int>(lock_idx),
                              work_tile_info.K_idx);
      BlockStripedReduceT::load_add(*accumulator_array, tile_workspace, thread_in_group);
      return;
    }

    if (work_tile_info.K_idx == 0) {
      BlockStripedReduceT::store(tile_workspace, *accumulator_array, thread_in_group);
    }
    else {
      BarrierManager::wait_eq(barrier_idx, locks, thread_in_group, static_cast<int>(lock_idx),
                              work_tile_info.K_idx);
      BlockStripedReduceT::reduce(tile_workspace, *accumulator_array, thread_in_group);
    }
    BarrierManager::arrive_inc(barrier_idx, locks, thread_in_group, static_cast<int>(lock_idx),
                               static_cast<int>(work_tile_info.k_tile_count));
  }

  /// A split fragment reports its own count; a whole-tile item is the base's own arithmetic on the
  /// shape the caller passes in.  Both this and `get_work_k_tile_start` are constant-folding-friendly
  /// on the data-parallel path only to the extent the compiler can see `k_tile_count == 0`; the
  /// measured whole-kernel cost of that not folding is why the data-parallel shapes are dispatched
  /// to the stock `GroupScheduler` at plan time rather than run through this class.
  template <class ProblemShape_MNKL, class TShape>
  CUTLASS_HOST_DEVICE static uint32_t
  get_work_k_tile_count(WorkTileInfo const& work_tile_info, ProblemShape_MNKL problem_shape,
                        TShape tile_shape) {
    if (work_tile_info.k_tile_count != 0) {
      return work_tile_info.k_tile_count;
    }
    return cute::size(cute::ceil_div(cute::get<2>(problem_shape), cute::get<2>(tile_shape)));
  }

  CUTLASS_HOST_DEVICE static uint32_t
  get_work_k_tile_start(WorkTileInfo const& work_tile_info) {
    return work_tile_info.k_tile_count == 0 ? 0u
                                            : static_cast<uint32_t>(work_tile_info.K_idx);
  }
};

}  // namespace eqlin

namespace cutlass::gemm::kernel::detail {

/// Selects the grouped stream-K scheduler for `eqlin::StreamKGroupScheduler`.
///
/// This specialization is *required*, not optional: the array kernel rewrites a non-void
/// `TileScheduler_` into `SchedulerTag` and always routes it through `TileSchedulerSelector`
/// (`sm90_gemm_array_tma_warpspecialized_cooperative.hpp:138-155`), so passing the scheduler class
/// directly to `GemmUniversal` would fall through to the primary template's `static_assert`.
/// Specializing on our own tag type is the supported extension point.
template <class TileShape, class ClusterShape, uint32_t SchedulerPipelineStageCount,
          class GroupProblemShape>
struct TileSchedulerSelector<eqlin::StreamKGroupScheduler, arch::Sm90, TileShape, ClusterShape,
                             SchedulerPipelineStageCount, GroupProblemShape> {
  using Scheduler = eqlin::GroupStreamKScheduler<GroupProblemShape, SchedulerPipelineStageCount,
                                                 TileShape, ClusterShape>;
};

}  // namespace cutlass::gemm::kernel::detail
