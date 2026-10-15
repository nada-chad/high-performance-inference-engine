# High-Performance C++17 ML Inference Engine

A low-latency, high-throughput machine learning inference execution engine written in modern C++17 for Linux systems.

## Key Technical Highlights

- Lock-Free Concurrency: Single-Producer Single-Consumer (SPSC) queue aligned to 64-byte cache lines to prevent false sharing and ensure deterministic thread synchronization.
- Async I/O: Asynchronous event-driven operations prepared via Linux io_uring for high-speed frame ingestion.
- Zero-Copy Memory Architecture: Custom ZeroCopyRingBuffer backed by a single contiguous block allocated via mmap with huge pages (MAP_HUGETLB) and a fallback mechanism to standard 4KB pages.
- Determinism & Jitter Reduction: CPU core pinning via pthread_setaffinity_np to bind threads to dedicated physical cores and mitigate context-switching jitter.
- Direct Engine Integration: Direct memory buffer binding with the ONNX Runtime C++ API (Ort::Value::CreateTensor), bypassing redundant copies (memcpy) between the network layer and the inference engine.

## Architecture & Design

This engine decouples high-speed data ingestion from inference execution using a fixed-capacity ring buffer and a lock-free channel:

1. Network Ingestion (network_thread_io_uring): Frames are acquired using asynchronous I/O and written directly into pre-allocated slots within the huge-page memory arena.
2. Buffer Coordination (ZeroCopyRingBuffer): Employs atomic states (std::memory_order_acquire / release) to safely synchronize writing and reading slots without locks or dynamic allocations (malloc/new).
3. Inference Execution (ai_processing_thread): Worker threads consume slot pointers via the SPSC queue and pass raw pixel pointers straight to ONNX Runtime for sub-millisecond execution.

## Prerequisites & Building

- C++17 compliant compiler (GCC 9+ or Clang 10+)
- Linux kernel 5.1+ (for io_uring and Huge Pages support, with standard page fallback)
- ONNX Runtime C++ SDK
- liburing library

### Build Instructions

```bash
g++ -O3 -march=native -std=c++17 main.cpp -o inference_engine \
     -I$HOME/onnxruntime-linux-x64-1.18.0/include \
     -L$HOME/onnxruntime-linux-x64-1.18.0/lib \
     -lonnxruntime -luring -lpthread
mkdir build && cd build
cmake ..
make -j$(nproc)
