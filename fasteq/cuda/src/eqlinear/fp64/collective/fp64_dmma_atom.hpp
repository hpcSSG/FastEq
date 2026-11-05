/***************************************************************************************************
 * fasteq eqlinear fp64/csrc/collective/fp64_dmma_atom.hpp
 *
 * A composite warp-level FP64 DMMA atom: 32x32x16, built from 2(M) x 4(N) sub-atoms of
 * cute::SM90::MMA_16x8x16_F64F64F64F64_TN.
 *
 * Why a composite atom.  The stock SM90 cooperative array kernel requires
 *     size(TiledMma) == 256, TileShape M >= 128, NumMmaWarpGroups == 2
 * (sm90_gemm_array_tma_warpspecialized_cooperative.hpp:442-445).  A raw 16x8 DMMA atom gives a
 * 128x8 tile at 256 threads, so the N extent has to come from somewhere: each warp owns a 32x32
 * tile (2x4 atoms) and the CTA tiles 4x2 warps -> 128x64 at 256 threads.
 *
 * Register grouping.  MMA_Atom::call() unpacks the fragments element by element (mma_traits.hpp
 * `explode`), so the register arrays are grouped to keep one sub-atom's operands contiguous:
 *     A: double[2][8]   grouped by mi
 *     B: double[4][4]   grouped by ni
 *     C: double[8][4]   grouped by chunk = mi + 2*ni
 * The value-mode order in the layouts below is chosen so that CuTe's linear value index groups
 * exactly that way (v modes fastest, then mi/ni slowest).
 *
 * Flat-coordinate convention (same as the sub-atom's traits): the atom's operand tile is
 * linearised as m + 32*k for A (M-major), n + 32*k for B (N-major), m + 32*n for C.  The sub-atom
 * strides were re-derived in those 32-wide coordinates.
 **************************************************************************************************/
#pragma once

#include <cute/arch/mma_sm90.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits.hpp>

namespace eqlin::fp64 {

/// 32x32x16 FP64 DMMA: 2x4 sub-atoms of 16x8x16.
struct Op32x32x16 {
  using DRegisters = double[8][4];
  using ARegisters = double[2][8];
  using BRegisters = double[4][4];
  using CRegisters = double[8][4];

  CUTE_HOST_DEVICE static void fma(
      double (&d0)[4], double (&d1)[4], double (&d2)[4], double (&d3)[4],
      double (&d4)[4], double (&d5)[4], double (&d6)[4], double (&d7)[4],
      double const (&a0)[8], double const (&a1)[8],
      double const (&b0)[4], double const (&b1)[4],
      double const (&b2)[4], double const (&b3)[4],
      double const (&c0)[4], double const (&c1)[4], double const (&c2)[4], double const (&c3)[4],
      double const (&c4)[4], double const (&c5)[4], double const (&c6)[4], double const (&c7)[4]) {
    double *d[8] = {&d0[0], &d1[0], &d2[0], &d3[0], &d4[0], &d5[0], &d6[0], &d7[0]};
    double const *a[2] = {&a0[0], &a1[0]};
    double const *b[4] = {&b0[0], &b1[0], &b2[0], &b3[0]};
    double const *c[8] = {&c0[0], &c1[0], &c2[0], &c3[0], &c4[0], &c5[0], &c6[0], &c7[0]};
    CUTLASS_PRAGMA_UNROLL
    for (int mi = 0; mi < 2; ++mi) {
      CUTLASS_PRAGMA_UNROLL
      for (int ni = 0; ni < 4; ++ni) {
        double *dd = d[mi + 2 * ni];
        double const *cc = c[mi + 2 * ni];
        cute::SM90::MMA_16x8x16_F64F64F64F64_TN::fma(
            dd[0], dd[1], dd[2], dd[3],
            a[mi][0], a[mi][1], a[mi][2], a[mi][3],
            a[mi][4], a[mi][5], a[mi][6], a[mi][7],
            b[ni][0], b[ni][1], b[ni][2], b[ni][3],
            cc[0], cc[1], cc[2], cc[3]);
      }
    }
  }
};


/// 32x32x8 FP64 DMMA: 2x4 sub-atoms of 16x8x8.  Half the per-thread operands of the k16 atom
/// (8 A + 8 B doubles instead of 16 + 16), which is what makes register double buffering fit
/// inside the 170-register budget of a 3-CTA/SM kernel; two of these cover one BK=16 stage.
struct Op32x32x8 {
  using DRegisters = double[8][4];
  using ARegisters = double[2][4];
  using BRegisters = double[4][2];
  using CRegisters = double[8][4];

  CUTE_HOST_DEVICE static void fma(
      double (&d0)[4], double (&d1)[4], double (&d2)[4], double (&d3)[4],
      double (&d4)[4], double (&d5)[4], double (&d6)[4], double (&d7)[4],
      double const (&a0)[4], double const (&a1)[4],
      double const (&b0)[2], double const (&b1)[2],
      double const (&b2)[2], double const (&b3)[2],
      double const (&c0)[4], double const (&c1)[4], double const (&c2)[4], double const (&c3)[4],
      double const (&c4)[4], double const (&c5)[4], double const (&c6)[4], double const (&c7)[4]) {
    double *d[8] = {&d0[0], &d1[0], &d2[0], &d3[0], &d4[0], &d5[0], &d6[0], &d7[0]};
    double const *a[2] = {&a0[0], &a1[0]};
    double const *b[4] = {&b0[0], &b1[0], &b2[0], &b3[0]};
    double const *c[8] = {&c0[0], &c1[0], &c2[0], &c3[0], &c4[0], &c5[0], &c6[0], &c7[0]};
    CUTLASS_PRAGMA_UNROLL
    for (int mi = 0; mi < 2; ++mi) {
      CUTLASS_PRAGMA_UNROLL
      for (int ni = 0; ni < 4; ++ni) {
        double *dd = d[mi + 2 * ni];
        double const *cc = c[mi + 2 * ni];
        cute::SM90::MMA_16x8x8_F64F64F64F64_TN::fma(
            dd[0], dd[1], dd[2], dd[3],
            a[mi][0], a[mi][1], a[mi][2], a[mi][3],
            b[ni][0], b[ni][1],
            cc[0], cc[1], cc[2], cc[3]);
      }
    }
  }
};

}  // namespace eqlin::fp64

namespace cute {

template <>
struct MMA_Traits<eqlin::fp64::Op32x32x16> {
  using ValTypeD = double;
  using ValTypeA = double;
  using ValTypeB = double;
  using ValTypeC = double;

  using Shape_MNK = Shape<_32, _32, _16>;
  using ThrID = Layout<_32>;

  using ALayout = Layout<Shape<Shape<_4, _8>, Shape<_2, _4, _2>>,
                         Stride<Stride<_32, _1>, Stride<_8, _128, _16>>>;
  using BLayout = Layout<Shape<Shape<_4, _8>, Shape<_4, _4>>,
                         Stride<Stride<_32, _1>, Stride<_128, _8>>>;
  using CLayout = Layout<Shape<Shape<_4, _8>, Shape<_2, _2, _2, _4>>,
                         Stride<Stride<_64, _1>, Stride<_32, _8, _16, _256>>>;
};

/// 32x32x8: the k16 traits with the k mode halved (v1 in {0,1}) and C unchanged.
template <>
struct MMA_Traits<eqlin::fp64::Op32x32x8> {
  using ValTypeD = double;
  using ValTypeA = double;
  using ValTypeB = double;
  using ValTypeC = double;

  using Shape_MNK = Shape<_32, _32, _8>;
  using ThrID = Layout<_32>;

  using ALayout = Layout<Shape<Shape<_4, _8>, Shape<_2, _2, _2>>,
                         Stride<Stride<_32, _1>, Stride<_8, _128, _16>>>;
  using BLayout = Layout<Shape<Shape<_4, _8>, Shape<_2, _4>>,
                         Stride<Stride<_32, _1>, Stride<_128, _8>>>;
  using CLayout = Layout<Shape<Shape<_4, _8>, Shape<_2, _2, _2, _4>>,
                         Stride<Stride<_64, _1>, Stride<_32, _8, _16, _256>>>;
};

}  // namespace cute
