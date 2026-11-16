#pragma once

#include "grouped_gemm_fp32.cuh"
#include "bwd.h"

namespace eqlin_fp32 {

// M <= NumericPolicy::kTinyBatch only. The runner owns [P,U,V] weights and A storage.
// rbase/pg are either both empty (one path per output) or both have J entries.
// set_scales takes one original double CG per entry of rows, in problem order.
std::unique_ptr<GroupedGemmPlan::ImplBase> make_tiny_forward_plan(
    int B, int U, int V, int J, long long lda, std::vector<RowPlan> const &rows,
    std::vector<int> const &rbase, std::vector<int> const &pg,
    float const *a, float const *w);

// Dgrad only: runner-owned weights have already been transposed to [P,K,N].
// Each problem is one output group; its b/ldb are deliberately ignored.
std::unique_ptr<BwdPlan> make_tiny_dgrad_plan(
    int B, int N, std::vector<BwdProblem> const &problems, float const *weights);

// One-time prep into a separate runner-owned buffer: [P,U,V] -> [P,V,U].
int launch_tiny_weight_transpose(float const *w, float *transposed,
                                 int P, int U, int V, cudaStream_t stream);

}  // namespace eqlin_fp32
