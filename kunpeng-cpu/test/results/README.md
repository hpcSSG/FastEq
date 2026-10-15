# 历史结果
inference_e3nn_results.txt 原样来自用户的集成包，不是本次重构后重新运行的结果。
原脚本主要使用 float32，包含前向和输入梯度误差及耗时；没有完整环境版本快照。
FCTP/Linear 测试的是 packed 路径；SC 的参照是 PyTorch einsum，不能统称为 e3nn 测试。
这些数据不能替代本仓库在目标机器上的回归测试，也不能证明完整训练或 cuEquivariance API 兼容。
