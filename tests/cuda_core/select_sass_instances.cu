// Compile with nvcc --cubin for the before/after K5c disassembly comparison.
// Instantiate each materialized-row kernel and both strided block-max kernels.
#include "cuda/core/select_score.cuh"

namespace strix::cuda::core::select::detail {
template __global__ void score_kernel<ScoreForm::relu_tree>(const ScoreRow*, ScoreSpec);
template __global__ void score_kernel<ScoreForm::weighted_fma>(const ScoreRow*, ScoreSpec);
template __global__ void score_kernel<ScoreForm::weighted_bf16>(const ScoreRow*, ScoreSpec);
template __global__ void score_kernel<ScoreForm::weighted_bf16x3>(const ScoreRow*, ScoreSpec);
template __global__ void score_kernel<ScoreForm::weighted_fp4>(const ScoreRow*, ScoreSpec);
template __global__ void score_kernel<ScoreForm::weighted_fp8>(const ScoreRow*, ScoreSpec);
template __global__ void score_kernel<ScoreForm::block_max>(const ScoreRow*, ScoreSpec);
template __global__ void score_kernel<ScoreForm::block_max_tiled>(const ScoreRow*, ScoreSpec);
template __global__ void score_strided_kernel<ScoreForm::block_max>(StridedScoreArgs, unsigned, unsigned);
template __global__ void score_strided_kernel<ScoreForm::block_max_tiled>(StridedScoreArgs, unsigned, unsigned);
template __global__ void pick_kernel<KeyOrder::ordered>(const PickRow*, PickSpec);
template __global__ void pick_kernel<KeyOrder::canonical_float>(const PickRow*, PickSpec);
template __global__ void pick_kernel<KeyOrder::raw_float>(const PickRow*, PickSpec);
template __global__ void pick_kernel<KeyOrder::eligible_float>(const PickRow*, PickSpec);
} // namespace strix::cuda::core::select::detail
