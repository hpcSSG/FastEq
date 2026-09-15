"""Source-ordered gradient of each reciprocal standard deviation."""
import triton
import triton.language as tl
from ._torch_norm_order import configured_m_sum


@triton.jit
def _flat_sum(value, atom, START: tl.constexpr, M: tl.constexpr,
              C: tl.constexpr, CFG: tl.constexpr):
    BK: tl.constexpr = value.shape[0]
    BC: tl.constexpr = value.shape[1]
    i = tl.arange(0, BK * BC)
    offset = (START + i // C) * BC + i % C
    selected = tl.gather(value.reshape((BK * BC,)), offset % (BK * BC), 0)
    selected = tl.where(i < M, selected, 0.)
    return tl.sum(configured_m_sum(selected[:, None], atom, 0, M, CFG), 0)


@triton.jit
def source_r_gradient(z, dy, gamma, atom, INDEX_START: tl.constexpr,
                      START: tl.constexpr, END: tl.constexpr, C: tl.constexpr,
                      L: tl.constexpr, ORDER: tl.constexpr,
                      M_CONFIG: tl.constexpr, C_CONFIG: tl.constexpr,
                      Q_CONFIG: tl.constexpr, HAS_WEIGHT: tl.constexpr,
                      DEGREE_CONFIG: tl.constexpr):
    BK: tl.constexpr = z.shape[0]
    BC: tl.constexpr = z.shape[1]
    if ORDER == 'broadcast':
        result = tl.full((), 0., tl.float32)
        k = INDEX_START + tl.arange(0, BK)[:, None]
        for l in tl.static_range(L, -1, -1):
            if (l * l >= START) & (l * l < END):
                if HAS_WEIGHT:
                    ds = configured_m_sum(dy * z, atom, l * l - INDEX_START, 2 * l + 1, ((M_CONFIG >> (12 * l)) & 4095))
                    weight = tl.sum(tl.where(k == l * l, gamma, 0.), 0)
                    dr = configured_m_sum((ds * weight)[:, None], atom, 0, C, C_CONFIG)
                    result += tl.sum(dr, 0)
                else:
                    result += _flat_sum(dy * z, atom, l * l - INDEX_START,
                        (2 * l + 1) * C, C, ((DEGREE_CONFIG >> (12 * l)) & 4095))
    else:
        result = _flat_sum((dy * z) * gamma, atom, START - INDEX_START,
                           (END - START) * C, C, Q_CONFIG)
    return result
