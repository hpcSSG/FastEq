import torch

x = torch.randn(4,96,96)

# 一步转置（对每个 slice 的最后两个维度同时转置）
y = x.transpose(-2, -1)   # 或 x.transpose(1,2) 或 x.T （对 3D tensor, x.T 等价于 transpose(-2,-1)）

# 等价于对每个 slice 做 transpose 再 stack
slices = [x[i].transpose(-2, -1) for i in range(x.shape[0])]  # 列表中每项是 (96,96) 的 view
y2 = torch.stack(slices, dim=0)  # (4,96,96) — 注意：stack 会分配新内存（复制）

# 数值相等
print(torch.allclose(y, y2))  # True

