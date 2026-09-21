# RecServe for trading research: what transfers, what does not

Assessment date: 2026-09-20. This is an engineering research assessment, not
investment advice or evidence of a profitable strategy. No broker, market-data
subscription, order-entry connection, live order or paper fill was created.

## Verdict

The strongest fit is **historical market-state retrieval for queue/fill-risk
research**. The implementation can accelerate finding similar past states and
serving a separately trained estimate. It does not currently model an order book,
predict fills, enforce trading risk, or detect executable arbitrage.

RecServe implements known HNSW/inner-product retrieval techniques plus serving
and measurement machinery. That systems implementation can be original project
work without claiming the underlying HNSW algorithm was invented here. The
[original HNSW paper](https://arxiv.org/abs/1603.09320) describes the search method;
it does not establish a financial signal.

## Component-by-component transfer

| Existing component | Plausible reuse | Required change / invalid shortcut |
|---|---|---|
| HNSW, `include/recserve/index.hpp` | Retrieve similar historical book states | Define/train a market-state representation and validate distance; never ANN-search for exact matching quotes or orders |
| SIMD/int8 dot products | Small linear fill/adverse-selection models; exact state-search oracle | Fit market labels; quantify probability/calibration changes from quantization |
| CUDA exact top-K | Offline scenario lookup and many-state replay batches | Current millisecond service results do not prove event-to-order latency; don't batch urgent orders merely for QPS |
| RCU snapshot store | Publish read-only model parameters or coherent analytical features | Order-book mutation/replay needs ordered ownership; copying whole books per event is not justified |
| Bounded queues/deadlines | Bound analytical work and reject stale predictions | Dropping an order-book delta corrupts future state; overload/gaps must invalidate the book and trigger resynchronization |
| Versioned bundles/journals | Reproducible models, audit records, immutable replay inputs | Financial order state needs venue acknowledgements, reconciliation and session-aware IDs; feedback idempotency is not order safety |
| Open-loop load generator | Measure offered load, deadline misses and overload | Replay actual bursts and distinguish hardware receive, application, decision and send timestamps |

## 1. Low-latency trading: useful infrastructure, unproven execution path

Separate three paths:

- **Offline research:** train/evaluate models, build state indexes, run GPU batches.
- **Analytical serving:** retrieve past states or score a market-state vector.
- **Order execution:** ordered market-data handling, deterministic risk checks,
  order-entry protocol, acknowledgements, inventory and kill switch.

Only portions of the first two resemble the current project. The pilot HTTP/
SQLite layer is deliberately not proposed for the latency-critical execution
path. Existing TCP requests carry a fixed user-query row, not a changing market
feature vector. A future research adapter must accept validated vectors or run
the C++ retrieval library in-process; replacing movies with ticker IDs is not
sufficient.

Measured CPU service p99 of 1.800 ms at 2,000 offered QPS was a 15-second local
MovieLens sweep. Large-catalog GPU throughput excluded network and batch formation.
Neither result includes feed decoding, book updates, execution risk or venue
latency. WSL desktop measurements cannot substantiate competitive HFT claims.

ITCH is a market-data feed, not an order-entry API. It specifies sequenced
order lifecycle messages, fixed-point price fields and nanosecond-since-midnight
timestamps. Its stock-locate IDs can change between days, so raw row IDs cannot
silently cross sessions. A real adapter needs the transport's sequence/session
information as well as the ITCH payload. Source:
[Nasdaq TotalView-ITCH 5.0 specification](https://www.nasdaqtrader.com/content/technicalsupport/specifications/dataproducts/NQTVITCHSpecification.pdf).

## 2. Queue modeling: the best bounded follow-on project

Distinguish **service queues** (pending requests/GPU batches) from **exchange
queues** (price-time/pro-rata order priority). RecServe currently measures the
former, not the latter.

Proposed research question:

> Does nearest-neighbor retrieval of historical order-book states improve
> out-of-sample fill-probability calibration over a simple baseline, within a
> measured inference-time budget?

Candidate input at local decision time t: spread in ticks, bid/ask depth,
imbalance, volume ahead of a hypothetical order, recent signed trade flow,
cancel/add intensity, volatility and time-of-day. Fit all transformations on
training days only. RecServe's cosine-style normalized fixtures may discard
important absolute depth/size information; do not blindly reuse normalization.

Candidate label: full fill before horizon H, after an explicit submission delay,
conditional on the hypothetical order placement/cancellation policy. Also
measure partial fill, time to fill and adverse price movement after fill.
Use only observations available by the decision's **local receive time**, not
future exchange events that had not arrived locally yet. Separate horizon,
latency and queue assumptions in the recorded configuration.

In a FIFO toy queue, total marketable volume exceeding initial volume ahead is
not generally sufficient to reconstruct a real fill: cancellations' positions,
partial fills, hidden liquidity and venue priority rules matter. Market-by-price
data does not reveal every order's priority. With market-by-order data, replay
add/cancel/execute/replace semantics and exclude auctions/halts unless modeled.
Do not report simulated fills as actual executions.

A model based on state-dependent event intensities is a relevant baseline:
the [queue-reactive model](https://arxiv.org/abs/1312.0563) treats the book as a
Markov queuing system during periods of fixed reference price. Simple imbalance
prediction is another baseline, motivated by
[Gould and Bonart](https://arxiv.org/abs/1512.03492). Their findings do not establish
predictability or profitability on a new instrument, period or latency budget.

Precise research acceptance order:

1. Choose one venue/instrument and obtain permitted order-level historical data.
   Record license, schema, sessions, source hashes, feed gaps and timestamps.
2. Reconstruct a deterministic book. Test duplicate/session reset, missing sequence,
   cancellation ahead/behind, replacement priority and partial execution against
   hand-worked fixtures. Gaps invalidate subsequent state until a valid recovery.
3. Split by whole ordered trading days. Purge examples whose future label horizon
   overlaps a validation/test boundary; keep all preprocessing train-only.
4. Compare constant/base-rate, imbalance/logistic, queue-reactive, exact nearest
   neighbors and HNSW-assisted predictions on identical examples. Hyperparameters
   come from validation, not the final test interval.
5. Report Brier/log loss, reliability bins, fill/partial-fill errors and conditional
   adverse selection. Use day/session-level uncertainty estimates rather than
   treating highly correlated ticks as independent observations.
6. Measure recall versus exact state search AND prediction/calibration change.
   High neighbor recall alone is not a quality or profitability result.
7. Compare single-thread/multicore CPU with GPU at the actual inference budget.
   Publish all failed/stale predictions and market-data replay speeds separately.

Stop if a simpler baseline is equally good and cheaper. That is a valid research
result and a stronger engineering decision than forcing ANN into the hot path.

## 3. Arbitrage: retrieval is not the detector

For a simple same-instrument two-venue screen, exact lookup of current executable
prices and depth is the starting point. Candidate gross edge at quantity q is
`q * (sell_bid - buy_ask)`, with q bounded by reachable depth, capital, inventory
and venue limits. Then subtract explicit buy/sell fees, execution slippage and
other applicable costs. Use integer ticks/fixed-point accounting, not embedding
similarity or unconstrained floating-point scores.

Illustrative arithmetic only: a $0.02/unit quoted edge on 100 units is $2 gross.
Two $0.003/unit fees cost $0.60; a $0.01/unit slippage allowance costs $1.00,
leaving $0.40 **before** leg risk, financing/borrow, transfer and other costs.
These are hypothetical inputs, not current venue prices/fees or expected profit.

Two legs on different venues are not atomic. One can fill while the other is
rejected, repriced or only partially filled. Model pre-positioned funds/inventory,
quote age, local arrival delay, order acknowledgement, partial fills, hedging cost
and forced unwind. Do not simulate buying at one venue and instantly transferring
the asset to fund a simultaneous sale elsewhere.

For triangular currency conversion, negative-log rate cycle detection can screen
candidates, but actual executable size, bid/ask direction, fees and sequential
fill constraints still require exact evaluation. For statistical/pairs trading,
retrieving similar regimes is plausible but is not risk-free arbitrage; predictive
relationships and transaction costs require their own out-of-sample evidence.

Potential RecServe role: estimate fill/adverse-selection risk or retrieve historical
stress scenarios **after** exact candidate construction, not replace quote matching,
inventory accounting or hard risk controls.

## Before any live integration

Require separate authorization and venue/data choices; define max order/notional,
position and loss limits, stale-book rejection, order-rate limits, kill switch,
duplicate-order prevention, reconciled broker state and a no-trade default on
uncertainty. No generated backtest is authorization to place an order.

For U.S. securities broker-dealers with market access, Rule 15c3-5 imposes specific
financial/regulatory controls; applicability depends on the participant and access
arrangement. This is not a compliance determination. Primary reference:
[SEC market-access risk-control FAQ](https://www.sec.gov/files/faq-15c-5-risk-management-controls-bd.htm).

## Recommendation for this repository

Finish and review the movie pilot as the supported application. If finance is the
chosen next domain, build **one offline queue-fill research adapter** under a
clearly experimental boundary, using licensed data and the gates above. Keep the
retrieval/measurement library reusable. Do not advertise RecServe as a trading or
arbitrage engine until the financial data model and execution evidence exist.
