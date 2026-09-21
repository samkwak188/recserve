"""Generate the readable investigation from raw measurements, not copied numbers."""
import json
from pathlib import Path


def main():
    source = Path('results/gpu-lowrate-investigation.json')
    study = json.loads(source.read_text())
    groups = {}
    for row in study['rows']:
        groups.setdefault((row['backend'], row['batch'], row['warmed']), []).append(row)
    lines = ['# Low-rate GPU investigation', '',
             'Single-host WSL experiment; no trading or whole-pilot latency claim.', '',
             'Each cell uses three independent service processes, 100 offered QPS for five seconds',
             'and a 5 ms client deadline. Warm cells first run 2 seconds at 1,000 QPS.',
             'Cold means a fresh process without request warmup, not a reset GPU or cold OS.',
             'Policy order is interleaved with a fixed seed. Stage instrumentation is enabled.', '',
             '| Backend / max batch | Warmed | Failed / attempted | Successful p99 range (ms) |',
             '|---|---|---:|---:|']
    for (backend, batch, warm), rows in sorted(groups.items()):
        attempted = sum(r['client']['attempted'] for r in rows)
        failed = sum(r['client']['attempted']-r['client']['counts'].get('ok', 0) for r in rows)
        tails = [r['client']['success_p99_us']/1000 for r in rows]
        lines.append(f'| {backend} / {batch} | {warm} | {failed}/{attempted} | {min(tails):.3f}-{max(tails):.3f} |')
    gpu = [r for r in study['rows'] if r['backend'] == 'cuda']
    first_wall = [r['first_batch_wall_us']/1000 for r in gpu]
    first_device = [r['first_batch_device_us']/1000 for r in gpu]
    lines += ['', '## Stage evidence and interpretation', '',
              f'- First CUDA call host wall time: {min(first_wall):.3f}-{max(first_wall):.3f} ms.',
              f'- First CUDA call summed device-event stages (H2D/compute/D2H): {min(first_device):.3f}-{max(first_device):.3f} ms.',
              '- The long first call is directly observed. Device-event intervals and host wall',
              '  measurements disagree at startup; they are not a trustworthy additive decomposition.',
              '  Do not subtract them to claim host-only overhead or pure kernel time. Event intervals',
              '  can include gaps between stream submissions. A timeline/clock investigation is needed',
              '  before attributing this delay to module loading, scheduling or a specific kernel.',
              '  [CUDA event timing reference](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html).',
              '- Compare cold/warm failure counts before deciding whether warmup helps; remaining',
              '  warm misses and sample size prevent a production latency guarantee.',
              '- Batch1 versus batch8 at low rate tests batching policy, not GPU throughput capacity.',
              '  Per-batch stage distributions must not be added to per-request p99 values.',
              '- Queue histograms cover dispatched GPU jobs; expiry counters also include jobs',
              '  discarded before dispatch. Never infer zero queue trouble from survivor histograms.',
              '- Keep CPU HNSW as the default. No automatic router, driver setting, power-state',
              '  change or startup policy was enabled from this experiment.', '',
              '## Reproduce / remaining work', '',
              '`bash scripts/check_ownership_runtime.sh` runs the local sanitizer/CUDA matrix',
              'before the experiment. `python3 scripts/summarize_gpu_investigation.py` generates',
              'this report. See [raw data](gpu-lowrate-investigation.json) for command parameters,',
              'source/binary/manifest hashes, individual runs and cumulative stage buckets.', '',
              'Next: profile first-call host/runtime operations, test readiness warmup as an explicit',
              'policy under idle gaps and restarts, perform longer burst/soak tests, then compare',
              'quality-matched optimized multicore/GEMM baselines on realistic larger catalogs.',
              'Compute Sanitizer remains a separate unpassed approval gate.', '']
    Path('results/GPU_INVESTIGATION.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines[8:8+len(groups)+3]))


if __name__ == '__main__':
    main()
