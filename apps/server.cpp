#include "recserve/engine.hpp"
#include "recserve/net.hpp"
#include "recserve/rss.hpp"
#include "recserve/gpu_batcher.hpp"
#include <csignal>
#include <iostream>
#include <sstream>

using namespace recserve;
namespace {
volatile std::sig_atomic_t interrupted = 0;
void stop_signal(int) { interrupted = 1; }
struct Counters {
  std::atomic<std::uint64_t> accepted{0}, rejected{0}, active{0}, queued{0}, requests{0}, ok{0},
      timeout{0}, bad{0}, unavailable{0}, io_failure{0}, clean_disconnect{0}, duration_us{0};
  DurationHistogram connection_queue, request_compute;
};
struct Connection { socket_t socket; std::uint64_t accepted_us; };
}

int run(int argc, char** argv) {
  int port = 9400, metrics_port = 9401, items = 4096, dim = 64, workers = 8, queue_limit = 64;
  int io_ms = 250, run_seconds = 0;
  int batch_size = 8, batch_wait_us = 100;
  bool synthetic = false;
  std::string bind_address = "127.0.0.1", catalog, index, queries, backend = "hnsw";
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    if (flag == "--synthetic") { synthetic = true; continue; }
    if (i + 1 >= argc) throw std::invalid_argument("missing value for " + flag);
    const std::string value = argv[++i];
    if (flag == "--port") port = std::stoi(value);
    else if (flag == "--metrics-port") metrics_port = std::stoi(value);
    else if (flag == "--bind") bind_address = value;
    else if (flag == "--items") items = std::stoi(value);
    else if (flag == "--dim") dim = std::stoi(value);
    else if (flag == "--workers") workers = std::stoi(value);
    else if (flag == "--queue") queue_limit = std::stoi(value);
    else if (flag == "--io-ms") io_ms = std::stoi(value);
    else if (flag == "--run-seconds") run_seconds = std::stoi(value);
    else if (flag == "--catalog") catalog = value;
    else if (flag == "--index") index = value;
    else if (flag == "--queries") queries = value;
    else if (flag == "--backend") backend = value;
    else if (flag == "--batch") batch_size = std::stoi(value);
    else if (flag == "--batch-wait-us") batch_wait_us = std::stoi(value);
    else throw std::invalid_argument("unknown option " + flag);
  }
  if (port < 1 || port > 65535 || metrics_port < 1 || metrics_port > 65535 || metrics_port == port ||
      workers < 1 || workers > 64 || queue_limit < 1 || queue_limit > 4096 || io_ms < 10 || io_ms > 10000 ||
      run_seconds < 0 || items < 1 || items > 1000000 || dim < 1 || dim > 4096 ||
      batch_size < 1 || batch_size > 64 || batch_wait_us < 0 || batch_wait_us > 10000)
    throw std::invalid_argument("invalid server limits");
  if (backend != "hnsw" && backend != "simd" && backend != "cuda") throw std::invalid_argument("unknown backend");
  if (synthetic ? !catalog.empty() || !queries.empty() || !index.empty() : catalog.empty() || queries.empty())
    throw std::invalid_argument("supply --catalog and --queries, or explicitly opt into --synthetic");
  if (!synthetic && backend == "hnsw" && index.empty()) throw std::invalid_argument("HNSW serving requires --index");

  Engine engine;
  engine.cfg.use_hnsw = backend == "hnsw";
  engine.cfg.kernel = Kernel::Simd;
  engine.cfg.build_threads = 1;
  if (synthetic) engine.init_random(items, 4096, dim, 7);
  else if (!engine.load_fixture(catalog, index, 4096) || !engine.load_queries(queries))
    throw std::runtime_error("artifact loading failed");
  std::unique_ptr<GpuBatcher> batcher;
  if (backend == "cuda") {
    std::string error;
    auto gpu = make_gpu_scorer(engine.cat.aos.data(), engine.cat.n, engine.cat.dim, batch_size, &error);
    if (!gpu) throw std::runtime_error(error);
    batcher = std::make_unique<GpuBatcher>(engine, std::move(gpu), batch_size, static_cast<std::uint32_t>(batch_wait_us));
  }
  const auto listener = listen_tcp(static_cast<std::uint16_t>(port), bind_address.c_str());
  if (listener == net_invalid()) throw std::runtime_error("serving bind failed");
  // Administrative endpoints remain loopback-only, independent of serving bind.
  const auto admin = listen_tcp(static_cast<std::uint16_t>(metrics_port));
  if (admin == net_invalid()) { net_close(listener); throw std::runtime_error("metrics bind failed"); }
  std::signal(SIGINT, stop_signal);
  std::signal(SIGTERM, stop_signal);
  std::atomic<bool> stopping{false};
  Counters counters;
  std::mutex mutex;
  std::condition_variable changed;
  std::deque<Connection> queue;
  std::vector<std::thread> threads;
  for (int worker = 0; worker < workers; ++worker) threads.emplace_back([&] {
    for (;;) {
      Connection connection;
      {
        std::unique_lock lock(mutex);
        changed.wait(lock, [&] { return stopping || !queue.empty(); });
        if (stopping) return;
        connection = queue.front(); queue.pop_front(); --counters.queued;
      }
      ++counters.active;
      counters.connection_queue.observe(now_us() - connection.accepted_us);
      const auto socket = connection.socket;
      auto frame_deadline = connection.accepted_us + static_cast<std::uint64_t>(io_ms) * 1000;
      while (!stopping) {
        std::uint8_t frame[8 + kReqBytes];
        bool clean_eof = false;
        if (!transfer_until(socket, frame, 8, false, frame_deadline, &clean_eof)) {
          if (clean_eof) ++counters.clean_disconnect;
          else ++counters.io_failure;
          break;
        }
        if (!valid_frame_header(frame)) { ++counters.bad; break; }
        if (!transfer_until(socket, frame + 8, kReqBytes, false, frame_deadline)) { ++counters.io_failure; break; }
        Request request;
        if (!decode_request(frame, sizeof(frame), request)) { ++counters.bad; break; }
        ++counters.requests;
        const auto start = now_us();
        Response response;
        response.id = request.id;
        if (request.user_id >= static_cast<unsigned>(engine.n_queries())) response.status = Status::BadRequest;
        else try { response = batcher ? batcher->submit(request).get() : engine.recommend_sync(request); }
        catch (const std::exception&) { response.status = Status::Unavailable; }
        const auto elapsed = now_us() - start;
        counters.duration_us += elapsed;
        counters.request_compute.observe(elapsed);
        if (response.status == Status::Ok && elapsed > request.timeout_us) {
          response.status = Status::Timeout; response.items.clear();
        }
        if (response.status == Status::Ok) ++counters.ok;
        else if (response.status == Status::Timeout) ++counters.timeout;
        else if (response.status == Status::BadRequest) ++counters.bad;
        else ++counters.unavailable;
        auto output = encode_response(response);
        if (!transfer_until(socket, output.data(), output.size(), true, now_us() + static_cast<std::uint64_t>(io_ms) * 1000)) {
          ++counters.io_failure; break;
        }
        frame_deadline = now_us() + static_cast<std::uint64_t>(io_ms) * 1000;
      }
      net_close(socket); --counters.active;
    }
  });
  std::thread monitoring([&] {
    while (!stopping) {
      if (!net_ready(admin, false, 50)) continue;
      const auto socket = accept_tcp(admin);
      if (socket == net_invalid()) continue;
      if (!net_nonblocking(socket)) { net_close(socket); continue; }
      std::string header;
      const auto deadline = now_us() + 250000;
      while (header.size() < 2048 && !header.ends_with("\r\n\r\n")) {
        std::uint8_t byte = 0;
        if (!transfer_until(socket, &byte, 1, false, deadline)) break;
        header += static_cast<char>(byte);
      }
      std::ostringstream body;
      bool found = false;
      if (header.starts_with("GET /healthz ") || header.starts_with("GET /readyz ")) { body << "ok\n"; found = !stopping; }
      else if (header.starts_with("GET /metrics ")) {
        found = true;
        body << "recserve_connections_accepted_total " << counters.accepted << '\n'
             << "recserve_connections_rejected_total " << counters.rejected << '\n'
             << "recserve_connections_active " << counters.active << '\n'
             << "recserve_connections_queued " << counters.queued << '\n'
             << "recserve_requests_total " << counters.requests << '\n'
             << "recserve_responses_ok_total " << counters.ok << '\n'
             << "recserve_timeouts_total " << counters.timeout << '\n'
             << "recserve_bad_requests_total " << counters.bad << '\n'
             << "recserve_unavailable_total " << counters.unavailable << '\n'
             << "recserve_io_failures_total " << counters.io_failure << '\n'
             << "recserve_clean_disconnects_total " << counters.clean_disconnect << '\n'
             << "recserve_compute_duration_microseconds_sum " << counters.duration_us << '\n'
             << "recserve_rss_bytes " << process_rss_bytes() << '\n';
        counters.connection_queue.write(body, "recserve_connection_queue_microseconds");
        counters.request_compute.write(body, "recserve_request_compute_microseconds");
        if (batcher) body << "recserve_gpu_batches_total " << batcher->batches << '\n'
                          << "recserve_gpu_queries_total " << batcher->gpu_queries << '\n'
                          << "recserve_gpu_fallback_total " << batcher->fallback << '\n'
                          << "recserve_gpu_healthy " << batcher->gpu_healthy << '\n'
                          << "recserve_gpu_batch_max " << batcher->max_observed_batch << '\n';
        if (batcher) {
          batcher->queue_time.write(body, "recserve_gpu_queue_microseconds");
          batcher->gpu_wall_time.write(body, "recserve_gpu_wall_microseconds");
          batcher->h2d_time.write(body, "recserve_gpu_h2d_microseconds");
          batcher->device_compute_time.write(body, "recserve_gpu_device_compute_microseconds");
          batcher->d2h_time.write(body, "recserve_gpu_d2h_microseconds");
          body << "recserve_gpu_expired_total " << batcher->expired << '\n'
               << "recserve_gpu_shed_total " << batcher->shed << '\n'
               << "recserve_gpu_first_batch_wall_microseconds " << batcher->first_batch_wall_us << '\n'
               << "recserve_gpu_first_batch_device_microseconds " << batcher->first_batch_device_us << '\n';
        }
      } else body << "not found\n";
      std::string output = std::string("HTTP/1.1 ") + (found ? "200 OK" : "404 Not Found") +
          "\r\nContent-Type: text/plain; version=0.0.4\r\nConnection: close\r\nContent-Length: " +
          std::to_string(body.str().size()) + "\r\n\r\n" + body.str();
      (void)transfer_until(socket, reinterpret_cast<std::uint8_t*>(output.data()), output.size(), true, deadline);
      net_close(socket);
    }
  });
  std::cerr << "ready bind=" << bind_address << ':' << port << " backend=" << backend
            << " workers=" << workers << " queue=" << queue_limit << " synthetic=" << synthetic << '\n';
  const auto start = now_us();
  while (!interrupted && (run_seconds == 0 || now_us() - start < static_cast<std::uint64_t>(run_seconds) * 1000000)) {
    if (!net_ready(listener, false, 50)) continue;
    const auto socket = accept_tcp(listener);
    if (socket == net_invalid()) continue;
    ++counters.accepted;
    std::lock_guard lock(mutex);
    if (queue.size() >= static_cast<std::size_t>(queue_limit) || !net_nonblocking(socket)) {
      ++counters.rejected; net_close(socket); continue;
    }
    queue.push_back({socket, now_us()}); ++counters.queued; changed.notify_one();
  }
  stopping = true;
  net_close(listener);
  {
    std::lock_guard lock(mutex);
    for (const auto& connection : queue) { net_close(connection.socket); ++counters.rejected; }
    queue.clear(); counters.queued = 0;
  }
  changed.notify_all();
  for (auto& thread : threads) thread.join();
  monitoring.join(); net_close(admin);
  std::cerr << "stopped requests=" << counters.requests << " rejected=" << counters.rejected << '\n';
  return 0;
}

int main(int argc, char** argv) {
  try { return run(argc, argv); }
  catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 2; }
}
