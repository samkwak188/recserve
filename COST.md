# Cost and capacity board

- Host: Windows-11-10.0.26200-SP0 / ARM64 / 12 logical CPUs / ISA `neon+sdot`
- Measured: 2026-09-17T20:52:56Z
- Protocol: in-process `recommend_sync`, prebuilt index snapshot, 64-request warmup,
  N trials with the coefficient of variation reported, synthetic clustered embeddings.
- Single-host benchmark. Not production, not multi-region, not ByteDance scale.

## Index correctness (vs hnswlib, identical data and parameters)

| ef | hnswlib recall@10 | RecServe recall@10 | delta |
|---|---|---|---|
| 32 | 0.3684 | 0.3742 | +0.0059 |
| 64 | 0.5445 | 0.5426 | -0.0020 |
| 128 | 0.7285 | 0.7273 | -0.0012 |
| 256 | 0.8902 | 0.8820 | -0.0082 |

Max absolute delta 0.0082 over 65,536 items, M=16, efConstruction=100.

## Kernels, graph retrieval (65,536 items, dim 64, ef=64, retrieve_k=64)

| kernel | p99 us | p99 CV | QPS | response recall@10 | RSS MiB |
|---|---|---|---|---|---|
| scalar | 221.6 | 0.130 | 7646 | 0.5453 | 38.7 |
| simd | 124.0 | 0.165 | 12939 | 0.5453 | 38.8 |
| soa_strided | 565.3 | 0.094 | 3090 | 0.5453 | 54.6 |
| blocked | 106.6 | 0.133 | 14031 | 0.5453 | 54.8 |
| int8 | 60.4 | 0.227 | 25250 | 0.5344 | 43.1 |

## Kernels, exact scan (65,536 items) -- where layout and dtype actually bind

| kernel | p99 us | QPS | catalog MiB |
|---|---|---|---|
| scalar | 1930.0 | 566 | 16.0 |
| simd | 545.4 | 2543 | 16.0 |
| soa_strided | 4683.5 | 228 | 32.0 |
| blocked | 607.0 | 2099 | 32.0 |
| int8 | 455.5 | 2944 | 20.2 |

## Kernels, exact scan (1,000,000 items) -- past every cache

| kernel | p99 ms | QPS | catalog MiB |
|---|---|---|---|
| scalar | 27.66 | 37.0 | 244.1 |
| simd | 5.96 | 174.5 | 244.1 |
| soa_strided | 51.04 | 28.9 | 488.3 |
| blocked | 7.41 | 140.5 | 488.3 |
| int8 | 5.26 | 196.8 | 309.0 |

## Latency / recall tradeoff (ef sweep, 65,536 items)

| ef | p99 us | QPS | retrieve recall@10 | response recall@10 | hops |
|---|---|---|---|---|---|
| 16 | 36.6 | 43010 | 0.2352 | 0.2352 | 22 |
| 24 | 46.0 | 32576 | 0.3289 | 0.3289 | 32 |
| 32 | 59.0 | 24862 | 0.3797 | 0.3797 | 40 |
| 48 | 84.0 | 18151 | 0.4680 | 0.4680 | 56 |
| 64 | 99.4 | 14224 | 0.5453 | 0.5453 | 72 |
| 96 | 156.3 | 10114 | 0.6344 | 0.6344 | 103 |
| 128 | 201.7 | 8068 | 0.7125 | 0.7125 | 133 |
| 192 | 276.2 | 5518 | 0.8242 | 0.8242 | 195 |
| 256 | 386.8 | 3857 | 0.8781 | 0.8781 | 258 |
| 384 | 538.7 | 2743 | 0.9422 | 0.9422 | 385 |

## Blocked AoSoA vs AoS+SIMD across dim (200,000 items, exact scan)

| dim | simd p99 us | blocked p99 us | blocked/simd |
|---|---|---|---|
| 8 | 733 | 406 | 0.55x |
| 16 | 805 | 547 | 0.68x |
| 24 | 934 | 647 | 0.69x |
| 32 | 1037 | 799 | 0.77x |
| 48 | 1100 | 1130 | 1.03x |
| 64 | 1719 | 1591 | 0.93x |
| 96 | 1685 | 2278 | 1.35x |
| 128 | 2267 | 2991 | 1.32x |

Blocked wins clearly at dim <= 32 (0.55x at the low end), is within noise between dim 32 and 96, and loses from dim 96 up (1.35x).
Below the crossover the per-item horizontal reduction dominates and blocking
amortises it across 8 items; above it, broadcasting q[d] once per dimension
costs more than the reduction it removes.

## Feature store under live ingest (8 serving threads, median of 3)

| events/s | store | serve QPS | p50 us | p99 us | p99.9 us | freshness p99 ms |
|---|---|---|---|---|---|---|
| 10,000 | mutex | 158858 | 37 | 87 | 3015 | 0 |
| 10,000 | snapshot | 169228 | 36 | 55 | 2918 | 104 |
| 100,000 | mutex | 148422 | 39 | 114 | 3100 | 0 |
| 100,000 | snapshot | 158341 | 38 | 51 | 2979 | 111 |
| 500,000 | mutex | 139807 | 40 | 139 | 3162 | 0 |
| 500,000 | snapshot | 155413 | 39 | 51 | 3031 | 113 |

### Publish interval: the price of not holding a lock

| publish ms | p99 us | p99.9 us | freshness p50 ms | freshness p99 ms | publish p99 us |
|---|---|---|---|---|---|
| 10 | 52 | 3123 | 15 | 25 | 7301 |
| 50 | 52 | 3232 | 33 | 64 | 7329 |
| 100 | 53 | 3213 | 61 | 113 | 7208 |
| 500 | 54 | 3260 | 255 | 504 | 8988 |

## Online/offline feature skew vs publish interval

| publish ms | unpublished events | items differing | max dCTR | mean dCTR |
|---|---|---|---|---|
| 100 | 99 | 74 | 1.0000 | 0.000196 |
| 1000 | 999 | 553 | 1.0000 | 0.001464 |
| 5000 | 4,999 | 1,919 | 1.0000 | 0.005549 |
| 30000 | 19,999 | 5,065 | 1.0000 | 0.019699 |

## Closed-loop agent, from a misconfigured start

| planner | start p99 us | best p99 us | improvement | confirmed | accepted | rejected |
|---|---|---|---|---|---|---|
| heuristic | 656.5 | 230.4 | +64.9% | True | 2 | 10 |
| random | 841.2 | 351.4 | +58.2% | True | 3 | 9 |
| grid | 629.9 | 637.6 | -1.2% | False | 2 | 10 |

Smallest change this host can resolve at this protocol: 17.5% to 64.7% (2 sigma, calibrated per run against the START config -- a slow,
misconfigured start is noisier in absolute terms, so its floor is wider.)

On an already-tuned config all planners accept nothing. The best remaining
change is retrieve_k 64 -> 16; a separate 40-run interleaved A/B measures it
at +2.9% with t = 2.10, below the floor, so declining to claim it is correct.

## Capacity and cost (1,000,000 qps, 100,000,000 items, 3 replicas, 65% utilisation)

Prices: https://aws.amazon.com/ec2/pricing/on-demand/ (2026-09-17, linux on-demand, no reserved or spot discount). On-demand list is a ceiling, not a quote.

| kernel | instance | QPS/core | B/item | hosts | bound by | $/hour | $/M requests |
|---|---|---|---|---|---|---|---|
| scalar | c7g.4xlarge | 7518 | 388 | 13 | compute | 7.54 | 0.0021 |
| scalar | c6i.4xlarge | 7518 | 388 | 13 | compute | 8.84 | 0.0025 |
| scalar | r7g.4xlarge | 7518 | 388 | 13 | compute | 11.14 | 0.0031 |
| simd | c7g.4xlarge | 14104 | 388 | 7 | compute | 4.06 | 0.0011 |
| simd | c6i.4xlarge | 14104 | 388 | 7 | compute | 4.76 | 0.0013 |
| simd | r7g.4xlarge | 14104 | 388 | 7 | compute | 6.00 | 0.0017 |
| int8 | c7g.4xlarge | 26008 | 200 | 4 | compute | 2.32 | 0.0006 |
| int8 | c6i.4xlarge | 26008 | 200 | 4 | compute | 2.72 | 0.0008 |
| int8 | r7g.4xlarge | 26008 | 200 | 4 | compute | 3.43 | 0.0010 |

- **c7g.4xlarge**: int8 takes 7 hosts to 4, $2,964 to $1,694/month (saves $1,270), at a cost of 0.0109 recall@10.
- **c6i.4xlarge**: int8 takes 7 hosts to 4, $3,475 to $1,986/month (saves $1,489), at a cost of 0.0109 recall@10.
- **r7g.4xlarge**: int8 takes 7 hosts to 4, $4,378 to $2,502/month (saves $1,876), at a cost of 0.0109 recall@10.

Do not restate any of this as production QPS, multi-region serving, or trained-model quality.
