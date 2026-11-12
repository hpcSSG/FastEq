#pragma once

namespace eqlin_fp32 {

/// Independent FP32 FMA chains shorten each dot product without changing the operand dtype.
/// Divide the M work per thread by the chain count, keeping the accumulator register budget
/// constant.  The plan still describes the real batch size; this is a CTA resource policy.
class NumericPolicy {
 public:
  // Blocked dots retain the full M tile. Small problems keep the existing
  // resource policy until there is enough work to amortize the wider tile.
  static bool blocked(int batch, int channels, int rows, bool dgrad = false) {
    if (batch < 512) return false;
    // Few output components leave the wider tile underoccupied. Keep the
    // existing narrow/multi-chain kernels on this measured small-work class.
    if (channels == 128 && rows <= 9 && batch < 4096) return false;
    if (dgrad && channels <= 128 && batch < 1024) return false;
    return true;
  }
  static constexpr int kTinyBatch = 64;
  static bool tiny_batch(int batch) { return batch <= kTinyBatch; }

  /// A small/unaligned M can select a more accurate strict-Torch contraction. Use the short
  /// chains there; retain the measured fast contraction on large established geometries.
  /// `rows` is the actual per-path problem count, never a captured configuration identifier.
  /// Multi-path forward rows are split per path; once `B` is large enough the accumulated
  /// round-off crosses the strict-FP32 gate (1.05x torch) while every reference stays green.
  /// Measured onset: U=V=96/128 fail from B=1584, U=V=224 from B=396. Keep the established fast
  /// contraction below those sizes and on single-path rows; this is a numeric policy keyed on the
  /// shape, never on a configuration id.
  static bool split_path_accuracy(int batch, int channels, bool split_paths) {
    if (!split_paths) return false;
    if (channels <= 128) return batch >= 1584;
    return batch >= 396;
  }

  /// Dgrad of a multi-path operator: keep the strict-FP32 contraction accurate at the measured
  /// onset. The dgrad `rows` argument is the problem count (I*P for K-concatenated rows), not the
  /// operator's path count, so the caller passes the path count explicitly.
  static bool dgrad_path_accuracy(int batch, int channels, int paths) {
    if (paths <= 1) return false;
    if (channels <= 128) return batch >= 1584;
    return batch >= 396;
  }

  static bool short_chains(int batch, int channels, int rows, bool dgrad = false) {
    if (batch % 4 != 0 || batch <= 256) return true;
    if (batch <= 512) {
      if (channels <= 128 && rows > 1) return true;
      return dgrad && channels == 224 && (rows == 1 || rows == 9);
    }
    return (channels == 128 && rows == 4) || (dgrad && channels == 224 && rows == 9);
  }
};

template <class Base, int Chains>
struct NumericConfig : Base {
  static constexpr int kChains = Chains;
  static constexpr int kBlockK = 0;
  static constexpr bool kContinuous = false;
  static constexpr bool kPostScale = Chains > 1;
  static constexpr int kWarpM = Base::kWarpM / kChains;
  static constexpr int kTbM = Base::kTbM / kChains;
  static_assert(kWarpM >= 8 && kWarpM % 8 == 0);
};

/// One persistent output state, plus one short-lived scalar dot per element.
/// Forward weights include each path's coefficient before crossing boundaries;
/// independent dgrad problems can instead retain post-dot scaling.
template <class Base, int BlockK = 8, bool PostScale = false>
struct BlockedConfig : Base {
  static constexpr int kChains = 1;
  static constexpr int kBlockK = BlockK;
  static constexpr int kChunk = BlockK;
  static constexpr bool kContinuous = !PostScale;
  static constexpr bool kPostScale = PostScale;
  static_assert(BlockK == 8 || BlockK == 16);
};

}  // namespace eqlin_fp32
