# Cost and capacity board

- Host: Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39 / x86_64 / 12 logical CPUs / ISA `avx2`
- Measured: 2026-09-20T22:12:57Z
- Protocol: in-process `recommend_sync`, prebuilt index snapshot, 64-request warmup,
  N trials with the coefficient of variation reported, synthetic clustered embeddings.
- Single-host benchmark. Not production, not multi-region, not ByteDance scale.

## Kernels, graph retrieval (65,536 items, dim 64, ef=64, retrieve_k=64)

| kernel | p99 us | p99 CV | QPS | response recall@10 | RSS MiB |
|---|---|---|---|---|---|
| scalar | 197.2 | 0.109 | 8487 | 0.5383 | 35.8 |
| simd | 117.2 | 0.084 | 12873 | 0.5383 | 35.8 |
| soa_strided | 742.4 | 0.195 | 2177 | 0.5383 | 51.9 |
| blocked | 100.0 | 0.029 | 14521 | 0.5383 | 51.9 |
| int8 | 62.4 | 0.031 | 22767 | 0.5367 | 40.1 |

## Kernels, exact scan (65,536 items) -- where layout and dtype actually bind

| kernel | p99 us | QPS | catalog MiB |
|---|---|---|---|
| scalar | 1746.3 | 650 | 16.0 |
| simd | 880.5 | 1929 | 16.0 |
| soa_strided | 11389.1 | 95 | 32.0 |
| blocked | 543.2 | 2397 | 32.0 |
| int8 | 331.1 | 4028 | 20.2 |

## Latency / recall tradeoff (ef sweep, 65,536 items)

| ef | p99 us | QPS | retrieve recall@10 | response recall@10 | hops |
|---|---|---|---|---|---|
| 16 | 39.8 | 45292 | 0.2273 | 0.2273 | 22 |
| 24 | 53.0 | 32820 | 0.3047 | 0.3047 | 32 |
| 32 | 65.6 | 24925 | 0.3609 | 0.3609 | 40 |
| 48 | 76.0 | 19505 | 0.4594 | 0.4594 | 56 |
| 64 | 106.8 | 14366 | 0.5383 | 0.5383 | 72 |
| 96 | 145.0 | 10207 | 0.6406 | 0.6406 | 102 |
| 128 | 195.2 | 7737 | 0.7219 | 0.7219 | 133 |
| 192 | 256.4 | 5506 | 0.8266 | 0.8266 | 195 |
| 256 | 358.6 | 4078 | 0.8820 | 0.8820 | 258 |
| 384 | 551.0 | 2767 | 0.9484 | 0.9484 | 385 |

## Blocked AoSoA vs AoS+SIMD across dim (200,000 items, exact scan)

| dim | simd p99 us | blocked p99 us | blocked/simd |
|---|---|---|---|
| 8 | 1031 | 291 | 0.28x |
| 16 | 950 | 438 | 0.46x |
| 24 | 1047 | 740 | 0.71x |
| 32 | 1562 | 1211 | 0.78x |
| 48 | 2203 | 1706 | 0.77x |
| 64 | 2620 | 1994 | 0.76x |
| 96 | 3171 | 2921 | 0.92x |
| 128 | 4507 | 4875 | 1.08x |

Do not restate any of this as production QPS, multi-region serving, or trained-model quality.
