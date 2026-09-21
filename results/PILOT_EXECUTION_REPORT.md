# Pilot execution report

Local implementation and evidence; not a production or user-impact claim.

## Completed

- External-ID API over actual C++ retrieval, training-seen/feedback/availability
  filtering, bounded replenishment and labeled cold-start/upstream-failure fallback.
- Transactional journal/projection, explicit shown/outcome attribution, duplicate
  and conflicting-event behavior, and generation-fenced response publication.
- Actual process death before/after commit, independent projection replay,
  concurrent retries, backup restore, and full C++/API restart verification.
- Bounded HTTP admission and absolute connection deadlines, tested on Windows
  and Linux. CPU/GPU stage histograms and cold/warm first-call investigation.
- Supported README separated from preserved historical experiments.
- Source-backed [trading applicability](../docs/TRADING_APPLICABILITY.md).

## Executed checks

- 25 Python discovery tests passed in WSL.
- 18 pilot tests passed on native Windows Python 3.13.
- Seven CPU CTest suites and eight CUDA-build suites passed on this workstation.
- ASan, UBSan and TSan configurations passed. TSan uses process-scoped setarch -R.
- Synthetic and trained MovieLens demos passed; a dismissal survived both process
  restarts and its replay did not increment the feature generation again.
- Final CPU image built with pilot tests and served a real-model request as UID
  10001 under read-only-root, PID and loopback restrictions. The runtime image
  contains the C++ service, not a packaged/deployed pilot API stack.
- Eighteen independent GPU/CPU low-rate experiments completed; see the
  [generated investigation](GPU_INVESTIGATION.md), including clock caveats.

The first pilot CI run exposed a Windows backup-test handle leak and omitted
container test inputs. Both were reproduced/repaired. Portability fix cbeb208
passed all 16 hosted jobs in [run 35555320380](https://github.com/samkwak188/recserve/actions/runs/35555320380).
Later commit CI status must be checked separately; this is not a blanket CI claim.

## Reproduce

```powershell
.\scripts\Check-Pilot.ps1
```

```bash
bash scripts/check_ownership_runtime.sh tests-only
# To repeat the bounded GPU experiment as well:
bash scripts/check_ownership_runtime.sh
```

All commands run through local scripts. Full logs remain in `.cache/logs`;
[validation receipts](pilot-validation.json) record timestamps, exit codes and
source hashes. Acceptance results: [MovieLens](pilot-movielens.json),
[synthetic](pilot-synthetic.json), [container](pilot-container-validation.json).

## Not completed / next decisions

- Real-user enrollment, consent/retention/deletion, authentication/TLS, UI and
  user outcomes. No person was recruited or account contacted.
- Whole-pilot long soak, storage exhaustion/machine-loss drills and migration.
- Kafka-to-RCU durable feature bridge, event-time windows and distributed fencing.
  SQLite is an explicit smaller architecture, not completion of those tickets.
- Globally isolated model evaluation, learned reranker, confidence intervals and
  optimized multicore/GEMM quality-matched benchmarking.
- CUDA timeline/root-cause attribution, Compute Sanitizer approval and a trusted
  hardware CI runner. Functional tests do not waive sanitizer approval.
- Financial data rights, order-book reconstruction, calibrated queue/fill models,
  transaction-cost/latency evaluation and execution controls. No live trades,
  financial subscriptions or profitable-arbitrage claims were produced.

Choose the next domain before expanding: a consented movie pilot or a separate
licensed-data, offline queue-fill study. Human understanding/review remains an
owner task and is not established by these automated tests.
