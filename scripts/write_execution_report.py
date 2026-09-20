#!/usr/bin/env python3
"""Render the execution handoff from saved measurements, without re-benchmarking."""
import datetime
import json
import pathlib
import statistics

root = pathlib.Path(__file__).resolve().parents[1]
def read(name):
    return json.loads((root/name).read_text())
gpu = read('results/gpu-crossover.json')
service = read('results/service-capacity.json')
ann = read('results/ann-high-recall.json')
quality = read('results/wsl-quality.json')
regression = read('results/wsl-regression.json')
ci = read('.cache/logs/hosted-ci.json')
if not ci['runs'] or any(r['conclusion'] != 'success' or any(j['conclusion'] != 'success' for j in r['jobs']) for r in ci['runs']):
    raise RuntimeError('hosted CI is not completely successful; preserve the previous report')
checks = ['baseline-tests', 'python-unit', 'test-none', 'test-address', 'test-undefined', 'test-thread',
          'final-cpu-test', 'final-gpu-test', 'hnsw-identical-queries', 'movielens-quality', 'real-gpu-quality',
          'bundle-create', 'container-build', 'container-smoke', 'paired-regression', 'final-cpu-quick',
          'gpu-campaign', 'ann-high-recall', 'service-capacity', 'recheck-test-build',
          'recheck-test-build-gpu', 'recheck-test-build-check-thread', 'container-build-final']
receipts = {}
for name in checks:
    path = root/f'.cache/logs/{name}.json'
    if not path.exists():
        raise RuntimeError('missing validation receipt: ' + name)
    receipt = json.loads(path.read_text())
    if receipt.get('exit_code') != 0:
        raise RuntimeError('validation is not complete/passing: ' + name)
    receipts[name] = receipt
evidence = dict(recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                runtime_commit='7e0eb632da22f42a96a006b1652afc5b19c48ed8', receipts=receipts,
                telemetry_checkpoint='4651d0d8fa66d8b93732a2b21cd8b330eb9f70e1',
                hosted_ci=ci, gpu_sanitizer=dict(status='blocked', reason='Windows WDDM GPU debugger interface disabled',
                receipt=read('.cache/logs/gpu-memcheck.json')),
                limitations=['no one-hour soak', 'no durable event-to-server bridge', 'no serving eligibility parity',
                             'no trained reranker or global time split', 'no authenticated public API or online pilot',
                             'no GPU runtime container or registered GPU runner', 'no measured cloud cost'])
(root/'results/execution-validation.json').write_text(json.dumps(evidence, indent=2)+'\n')

lines = ['# Execution report: RTX/WSL continuation', '',
         'Runtime checkpoint: `7e0eb63`, 2026-09-20. This is a verified local prototype, **not a production-ready pipeline**.', '',
         'Full acceptance gates: [execution plan](../EXECUTION_PLAN.md). Reproduction and operational limits: [operations](../OPERATIONS.md).', '',
         '## Verification', '',
         '- All 16 hosted jobs passed: [runtime CI](https://github.com/samkwak188/recserve/actions/runs/35541070732).',
         '- Telemetry/dependency follow-up `4651d0d` also passed all 16 jobs: [follow-up CI](https://github.com/samkwak188/recserve/actions/runs/35542148547).',
         '- CPU-only/AVX2, ASan, UBSan and TSan pass; WSL TSan used process-only ASLR disabling.',
         '- Six local CUDA-build CTest suites pass, including actual RTX correctness and socket microbatching.',
         '- Bundle tampering/identity mapping, malformed frames/artifacts, slow clients, bounded overload, recovery, shutdown, deadline expiry and injected exact CPU fallback are tested.',
         '- Non-root/read-only/loopback CPU container served the trained model; its temporary validation container/network were removed.',
         '- Compute Sanitizer is **blocked**, not passed: Windows GPU debugger interface requires owner/admin approval.',
         f"- Matched-ISA paired regression: median p99 ratio **{regression['p99_ratio']:.3f}x**, recall delta **{-regression['recall_drop']:.4f}**; gate passed.", '',
         'Machine: Ryzen 5 5600X, 12 logical CPUs, RTX 3060 12 GiB; Ubuntu 24.04 WSL, GCC 13.3, CUDA 13.1.115, driver 591.86.',
         'Detailed command receipts and source/binary/fixture provenance: [validation JSON](execution-validation.json) and [GPU raw measurements](gpu-crossover.json).', '',
         '## Observed GPU crossover', '',
         'Top-10 exact retrieval, dim64 synthetic catalog, three independent processes per point, 200 timed preformed batches/process after five warmups. Each backend uses the same 128 independent quality queries.',
         'The CPU exact baseline is single-threaded AVX2 FP32 `simd`, not an all-core CPU/GEMM baseline or a claim of best-possible CPU performance. GPU batch latency excludes queue formation, feature lookup, final ranking and network. These sampled p99 values are descriptive, not a production tail bound.', '',
         '| Items | Batch | GPU QPS | Speedup vs CPU exact | GPU batch p99 ms |',
         '|---:|---:|---:|---:|---:|']
for row in gpu['summary']:
    if row['batch'] in (1, 8, 128):
        lines.append(f"| {row['items']:,} | {row['batch']} | {row['gpu_qps']:,.0f} | {row['gpu_vs_exact']:.2f}x | {row['gpu_p99_batch_us']/1000:.3f} |")
minimum = min(r['recall'] for r in gpu['measurements'] if r['backend'] == 'cuda')
memory = max(r['device_bytes'] for r in gpu['measurements'])/(1024**2)
lines += ['', f'GPU retrieve recall minimum: **{minimum:.6f}** on these probes. Maximum explicitly allocated device workspace/catalog across this sweep: **{memory:.1f} MiB** (not total process/driver VRAM).',
          'At batch1 the observed GPU/CPU-exact crossover lies between 16K and 64K items. At 4K items batch8 is effectively tied (~0.994x), not a demonstrated win. Large batches improve throughput but can violate a 5 ms online latency target.', '',
          '### CPU ANN must be compared at stated quality', '',
          'The main sweep uses HNSW ef384. At 1M synthetic items its recall is only 0.5695, so its QPS is not an equal-quality substitute for GPU exact retrieval. A second fixed-fixture sweep reports:', '',
          '| HNSW ef | Minimum recall@10 | Median QPS, batch1 |', '|---:|---:|---:|']
for ef in sorted({r['ef'] for r in ann['measurements']}):
    group = [r for r in ann['measurements'] if r['ef'] == ef]
    lines.append(f"| {ef} | {min(r['recall'] for r in group):.6f} | {statistics.median(r['qps'] for r in group):,.1f} |")
lines += ['', 'This is a coarse ANN grid and a synthetic low-recall stress case, not an optimized equal-quality real-data benchmark. The measured recall1.0 is finite-probe evidence, not a guarantee over every query. [Raw ANN results](ann-high-recall.json).', '',
          '## Actual network service capacity', '',
          'Verified MovieLens bundle: 6,298 items, dim32, 579 query rows; 8 server workers; GPU maxbatch8/wait100us; one TCP connection/request; fixed intended arrivals; 5 ms client deadline; 15 seconds per point.',
          'Latency includes client scheduling, queueing and network. All attempted arrivals are accounted for. Reported p99 is for successful requests, so failure rate must be read alongside it. This is a short sweep without an explicit service warmup, not a one-hour soak or proven saturation capacity.', '',
          '| Backend | Offered QPS | Attempted | Successful p99 ms | Failure % | Provisional SLO met |',
          '|---|---:|---:|---:|---:|---|']
for row in service['rows']:
    value = row['success_p99_us']
    lines.append(f"| {row['backend']} | {row['config']['rate']} | {row['attempted']} | {value/1000:.3f} | {row['failure_rate']*100:.3f} | {'yes' if row['provisional_slo_met'] else 'no'} |")
lines += ['', '**Decision:** keep CPU HNSW as the local default. GPU misses at low/moderate arrival rates must be investigated and retested before an automatic routing policy or GPU SLO claim. The sweep identifies symptoms, not their proven cause.',
          'The existing offline evaluation filters seen items; the deployed TCP path does not yet have those filters or fresh durable features. Therefore this is a serving-system test on real embeddings, not proof of production recommendation utility. [Raw capacity results](service-capacity.json).',
          'A post-campaign telemetry correction separates clean EOF disconnects from I/O errors. The old raw server I/O counter includes normal one-request connection closures; the client-side failure rates in this table do not. The correction is covered by CPU/GPU and thread-sanitized socket tests.', '',
          '## Real-data quality', '']
popular = quality['baselines']['most_popular']['recall']
als = quality['baselines']['als_exact']['recall']
lines += [f'ALS recall@10 **{als:.5f}**, popularity **{popular:.5f}**: **{(als/popular-1)*100:.1f}% relative offline lift**. C++ and NumPy recall agree within the recorded precision. CUDA also reproduced the measured held-out recall.',
          'This uses a per-user temporal holdout, not a globally time-isolated train/validation/test design. No confidence interval, new learned reranker, causal engagement lift or real-user experiment is claimed. [Raw quality report](wsl-quality.json).', '',
          '## Remaining release gates', '',
          'NVIDIA sanitizer approval; long-duration/failure-drill validation; selected pilot objective; durable transactional feature recovery and event-to-response wiring; serving-time eligibility/cold-start parity; learned and temporally validated ranking; authenticated deployment, backup/rollback, and opt-in user outcomes.',
          'No cloud instance was rented, no public endpoint exposed, and no real-user impact or cloud cost saving inferred. The historical ARM board is unchanged.', '',
          'The GPU campaign began before the runtime commit was created, so its recorded Git parent is `86bbbc2`; its source and executable hashes identify the measured worktree. That runtime source was subsequently committed as `7e0eb63`. New WSL artifacts do not replace historical ARM measurements.', '']
(root/'results/EXECUTION_REPORT.md').write_text('\n'.join(lines))
print(json.dumps(dict(report='results/EXECUTION_REPORT.md', receipts=len(receipts), gpu_processes=len(gpu['measurements']), service_points=len(service['rows']))))
