# Cost and capacity board

- Host: Windows-11-10.0.26200-SP0 / ARM64 / 12 logical CPUs / ISA `neon+sdot`
- Measured: 2026-09-18T08:22:04Z
- Protocol: in-process `recommend_sync`, prebuilt index snapshot, 64-request warmup,
  N trials with the coefficient of variation reported, synthetic clustered embeddings.
- Single-host benchmark. Not production, not multi-region, not ByteDance scale.

## Index correctness (vs hnswlib, identical data and parameters)

| ef | hnswlib recall@10 | RecServe recall@10 | delta |
|---|---|---|---|
| 32 | 0.3816 | 0.3738 | -0.0078 |
| 64 | 0.5609 | 0.5457 | -0.0152 |
| 128 | 0.7469 | 0.7293 | -0.0176 |
| 256 | 0.8883 | 0.8820 | -0.0063 |

Max absolute delta 0.0176 over 65,536 items, M=16, efConstruction=100.

## Kernels, graph retrieval (65,536 items, dim 64, ef=64, retrieve_k=64)

| kernel | p99 us | p99 CV | QPS | response recall@10 | RSS MiB |
|---|---|---|---|---|---|
| scalar | 242.0 | 0.093 | 7087 | 0.5453 | 38.6 |
| simd | 141.2 | 0.079 | 12263 | 0.5453 | 38.8 |
| soa_strided | 616.4 | 0.045 | 3009 | 0.5453 | 54.7 |
| blocked | 122.2 | 0.108 | 13624 | 0.5453 | 54.8 |
| int8 | 81.6 | 0.055 | 24512 | 0.5344 | 43.1 |

## Kernels, exact scan (65,536 items) -- where layout and dtype actually bind

| kernel | p99 us | QPS | catalog MiB |
|---|---|---|---|
| scalar | 1956.7 | 565 | 16.0 |
| simd | 507.1 | 2520 | 16.0 |
| soa_strided | 7946.1 | 214 | 32.0 |
| blocked | 576.0 | 2148 | 32.0 |
| int8 | 408.9 | 3142 | 20.2 |

## Kernels, exact scan (1,000,000 items) -- past every cache

| kernel | p99 ms | QPS | catalog MiB |
|---|---|---|---|
| scalar | 32.95 | 36.9 | 244.1 |
| simd | 5.86 | 175.3 | 244.1 |
| soa_strided | 38.78 | 29.7 | 488.3 |
| blocked | 7.13 | 144.2 | 488.3 |
| int8 | 4.93 | 209.4 | 309.0 |

## Latency / recall tradeoff (ef sweep, 65,536 items)

| ef | p99 us | QPS | retrieve recall@10 | response recall@10 | hops |
|---|---|---|---|---|---|
| 16 | 44.2 | 41312 | 0.2352 | 0.2352 | 22 |
| 24 | 63.2 | 28154 | 0.3289 | 0.3289 | 32 |
| 32 | 80.6 | 22297 | 0.3797 | 0.3797 | 40 |
| 48 | 111.6 | 15852 | 0.4680 | 0.4680 | 56 |
| 64 | 119.2 | 13523 | 0.5453 | 0.5453 | 72 |
| 96 | 172.9 | 9647 | 0.6344 | 0.6344 | 103 |
| 128 | 219.2 | 7400 | 0.7125 | 0.7125 | 133 |
| 192 | 307.0 | 5091 | 0.8242 | 0.8242 | 195 |
| 256 | 356.1 | 4107 | 0.8781 | 0.8781 | 258 |
| 384 | 523.4 | 2732 | 0.9422 | 0.9422 | 385 |

## Blocked AoSoA vs AoS+SIMD across dim (200,000 items, exact scan)

| dim | simd p99 us | blocked p99 us | blocked/simd |
|---|---|---|---|
| 8 | 677 | 413 | 0.61x |
| 16 | 757 | 529 | 0.70x |
| 24 | 924 | 616 | 0.67x |
| 32 | 964 | 813 | 0.84x |
| 48 | 1210 | 1300 | 1.07x |
| 64 | 1353 | 1558 | 1.15x |
| 96 | 1756 | 2259 | 1.29x |
| 128 | 2288 | 2910 | 1.27x |

Blocked wins clearly at dim <= 32 (0.61x at the low end), is within noise between dim 32 and 64, and loses from dim 64 up (1.29x).
Below the crossover the per-item horizontal reduction dominates and blocking
amortises it across 8 items; above it, broadcasting q[d] once per dimension
costs more than the reduction it removes.

## Feature store under live ingest (8 serving threads, median of 3)

| events/s | store | serve QPS | p50 us | p99 us | p99.9 us | freshness p99 ms |
|---|---|---|---|---|---|---|
| 10,000 | mutex | 194864 | 38 | 80 | 182 | 0 |
| 10,000 | snapshot | 209610 | 37 | 48 | 121 | 102 |
| 100,000 | mutex | 184041 | 40 | 91 | 351 | 0 |
| 100,000 | snapshot | 200734 | 39 | 49 | 70 | 101 |
| 500,000 | mutex | 171189 | 41 | 104 | 949 | 0 |
| 500,000 | snapshot | 195081 | 41 | 50 | 73 | 116 |

### Publish interval: the price of not holding a lock

| publish ms | p99 us | p99.9 us | freshness p50 ms | freshness p99 ms | publish p99 us |
|---|---|---|---|---|---|
| 10 | 50 | 72 | 0 | 26 | 520 |
| 50 | 50 | 77 | 30 | 63 | 1275 |
| 100 | 50 | 72 | 53 | 113 | 1442 |
| 500 | 50 | 74 | 250 | 499 | 3919 |

## Online/offline feature skew vs publish interval

| publish ms | unpublished events | items differing | max dCTR | mean dCTR |
|---|---|---|---|---|
| 100 | 99 | 74 | 1.0000 | 0.000196 |
| 1000 | 999 | 553 | 1.0000 | 0.001464 |
| 5000 | 4,999 | 1,919 | 1.0000 | 0.005549 |
| 30000 | 19,999 | 5,065 | 1.0000 | 0.019699 |

## Recommendation quality on real embeddings (ml25m: 40,858 items, 4,096 users)

ALS on MovieLens, per-user temporal split, items already seen in training
filtered from the returned list. recall@10 ceiling is 0.652 (mean held-out set 15.3 items).

| method | recall@10 | ndcg@10 |
|---|---|---|
| random | 0.0003 | 0.0006 |
| most_popular | 0.0503 | 0.0630 |
| als_exact | 0.0720 | 0.0739 |
| **served through recserve** | **0.0720** | **0.0739** |

ALS beats most-popular by +0.0217 recall (+43.0%). The C++
service and an independent numpy implementation of the same metric agree to
0.00000.

| ef | recall@10 | ndcg@10 | retrieve recall | p99 us |
|---|---|---|---|---|
| 16 | 0.0714 | 0.0736 | 0.9946 | 195 |
| 32 | 0.0714 | 0.0736 | 0.9946 | 183 |
| 64 | 0.0714 | 0.0736 | 0.9946 | 191 |
| 128 | 0.0714 | 0.0736 | 0.9946 | 190 |
| 256 | 0.0714 | 0.0736 | 0.9946 | 185 |

On trained embeddings the graph finds essentially all of the exact top-k at
small ef. On synthetic uniform vectors the same ef reaches about half, which
is a property of the data, not of the index.

## Sharded retrieval: tail amplification (12 cores)

A request finishes when its SLOWEST shard replies, so end-to-end latency is
the maximum of N samples, not the mean (Dean & Barroso, "The Tail at Scale",
CACM 56(2), 2013). Shards are threads in one process: no network, no separate
failure domain, so this is a LOWER BOUND on what a real deployment would see.

| shards | items/shard | shard p99 us | e2e p99 us | merge p99 us | amplification | recall@10 |
|---|---|---|---|---|---|---|
| 1 | 1,000,000 | 255 | 292 | 9 | 1.15x | 0.1605 |
| 2 | 500,000 | 258 | 361 | 10 | 1.40x | 0.2489 |
| 4 | 250,000 | 249 | 415 | 12 | 1.67x | 0.3670 |
| 8 | 125,000 | 244 | 666 | 13 | 2.73x | 0.5108 |
| 16 * | 62,500 | 214 | 863 | 16 | 4.03x | 0.6611 |
| 32 * | 31,250 | 210 | 1377 | 19 | 6.56x | 0.7994 |

\* more shards than cores: those rows include CPU queueing on top of the
structural tail effect, so read them as an upper bound rather than as a
clean measurement of amplification.

Recall rises with shard count because each shard searches its own smaller
graph at the same ef and every shard contributes its own top-k, so total
candidates examined scales with N. Sharding buys accuracy and costs tail
latency; that trade is the actual decision, not whether scatter-gather works.

## Closed-loop agent, from a misconfigured start

| planner | start p99 us | best p99 us | improvement | confirmed | accepted | rejected |
|---|---|---|---|---|---|---|
| heuristic | 641.4 | 248.0 | +61.3% | True | 2 | 10 |
| random | 651.9 | 403.7 | +38.1% | True | 1 | 11 |
| grid | 589.5 | 577.7 | +0.0% | True | 0 | 12 |

Smallest change this host can resolve at this protocol: 16.0% to 20.5% (2 sigma, calibrated per run against the START config -- a slow,
misconfigured start is noisier in absolute terms, so its floor is wider.)

On an already-tuned config all planners accept nothing. The best remaining
change is retrieve_k 64 -> 16; a separate 40-run interleaved A/B measures it
at +2.9% with t = 2.10, below the floor, so declining to claim it is correct.

## Capacity and cost (1,000,000 qps, 100,000,000 items, 3 replicas, 65% utilisation)

Prices: https://aws.amazon.com/ec2/pricing/on-demand/ (2026-09-17, linux on-demand, no reserved or spot discount). On-demand list is a ceiling, not a quote.

| kernel | instance | QPS/core | B/item | hosts | bound by | $/hour | $/M requests |
|---|---|---|---|---|---|---|---|
| scalar | c7g.4xlarge | 7378 | 388 | 14 | compute | 8.12 | 0.0023 |
| scalar | c6i.4xlarge | 7378 | 388 | 14 | compute | 9.52 | 0.0026 |
| scalar | r7g.4xlarge | 7378 | 388 | 14 | compute | 12.00 | 0.0033 |
| simd | c7g.4xlarge | 13002 | 388 | 8 | compute | 4.64 | 0.0013 |
| simd | c6i.4xlarge | 13002 | 388 | 8 | compute | 5.44 | 0.0015 |
| simd | r7g.4xlarge | 13002 | 388 | 8 | compute | 6.85 | 0.0019 |
| int8 | c7g.4xlarge | 25941 | 200 | 4 | compute | 2.32 | 0.0006 |
| int8 | c6i.4xlarge | 25941 | 200 | 4 | compute | 2.72 | 0.0008 |
| int8 | r7g.4xlarge | 25941 | 200 | 4 | compute | 3.43 | 0.0010 |

- **c7g.4xlarge**: int8 takes 8 hosts to 4, $3,387 to $1,694/month (saves $1,694), at a cost of 0.0109 recall@10.
- **c6i.4xlarge**: int8 takes 8 hosts to 4, $3,971 to $1,986/month (saves $1,986), at a cost of 0.0109 recall@10.
- **r7g.4xlarge**: int8 takes 8 hosts to 4, $5,004 to $2,502/month (saves $2,502), at a cost of 0.0109 recall@10.

Do not restate any of this as production QPS, multi-region serving, or trained-model quality.
