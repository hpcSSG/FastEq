import time
import torch
import torch.nn as nn

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 10),
        )

    def forward(self, x):
        return self.seq(x)

device = torch.device("cuda")
model = TinyNet().to(device).eval()

BATCH = 4096
inp_static = torch.empty(BATCH, 1024, device=device, dtype=torch.float32)
out_static = torch.empty(BATCH, 10, device=device, dtype=torch.float32)

# warmup for allocator / autotune
for _ in range(5):
    tmp = model(torch.randn_like(inp_static))
torch.cuda.synchronize()

g = torch.cuda.CUDAGraph()

# capture
# 注意：graph capture 里面必须用事先分配好的 Tensor (inp_static)
with torch.cuda.graph(g):
    out_static = model(inp_static)

# benchmark loop
repeats = 20
torch.cuda.synchronize()
t0 = time.perf_counter() * 1000.0  # ms

for _ in range(repeats):
    # 模拟新输入
    new_inp = torch.randn_like(inp_static)
    inp_static.copy_(new_inp)

    g.replay()

torch.cuda.synchronize()
t1 = time.perf_counter() * 1000.0
avg_ms = (t1 - t0) / repeats
print(f"avg latency per batch: {avg_ms:.4f} ms")

