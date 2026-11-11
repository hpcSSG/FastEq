#pragma once

namespace eqlin_fp32 {

/// Independent FP32 FMA chains shorten each dot product without changing the operand dtype.
/// Divide the M work per thread by the chain count, keeping the accumulator register budget
/// constant.  The plan still describes the real batch size; this is a CTA resource policy.
class NumericPolicy {
 public:
  static constexpr int kTinyBatch = 64;
  static bool tiny_batch(int batch) { return batch <= kTinyBatch; }

  /// A small/unaligned M can select a more accurate strict-Torch contraction. Use the short
  /// chains there; retain the measured fast contraction on large established geometries.
  /// `rows` is the actual per-path problem count, never a captured configuration identifier.
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
  static constexpr int kWarpM = Base::kWarpM / kChains;
  static constexpr int kTbM = Base::kTbM / kChains;
  static_assert(kWarpM >= 8 && kWarpM % 8 == 0);
};

}  // namespace eqlin_fp32
