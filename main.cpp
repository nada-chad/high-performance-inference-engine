#include <iostream>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <chrono>
#include <vector>
#include <new>
#include <utility>
#include <cstring>
#include <sys/mman.h>
#include <pthread.h>
#include <sched.h>
#include <immintrin.h>
#include <liburing.h>

#include <onnxruntime_cxx_api.h>

// -----------------------------------------------------------------------------
// 1. STRUCTURES ET ALLIGNEMENT MÉMOIRE (CACHE-LINE FRIENDLY)
// -----------------------------------------------------------------------------
constexpr size_t PAYLOAD_SIZE = 1 * 3 * 224 * 224 * sizeof(float);

struct FrameData {
    uint32_t frame_id;
    uint64_t timestamp_us;
    alignas(64) uint8_t payload[PAYLOAD_SIZE]; // Aligné sur 64 octets (ligne de cache CPU)
};

// Slot de mémoire pré-alloué doté d'une barrière atomique
struct FrameSlot {
    FrameData data;
    // false = Libre (Réseau peut écrire), true = Occupé (IA en cours de lecture)
    alignas(64) std::atomic<bool> in_use{false}; 
};

// -----------------------------------------------------------------------------
// 2. RING BUFFER ZERO-COPY DANS LES HUGE PAGES (MMAP)
// -----------------------------------------------------------------------------
template <size_t Capacity>
class ZeroCopyRingBuffer {
public:
    ZeroCopyRingBuffer() : total_bytes_(sizeof(FrameSlot) * Capacity) {
        // Allocation contiguë dans la mémoire virtuelle Linux via mmap
        void* ptr = mmap(nullptr, total_bytes_,
                         PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
                         -1, 0);

        // Fallback sur des pages classiques de 4 KB si les Huge Pages (2 MB) sont indisponibles
        if (ptr == MAP_FAILED) {
            std::cerr << "[WARN] Fallback sur pages standard (Huge Pages indisponibles)." << std::endl;
            ptr = mmap(nullptr, total_bytes_,
                       PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS,
                       -1, 0);
            if (ptr == MAP_FAILED) {
                throw std::bad_alloc();
            }
        }

        slots_ = static_cast<FrameSlot*>(ptr);

        // Placement new pour initialiser chaque FrameSlot dans le bloc mémoire mmap
        for (size_t i = 0; i < Capacity; ++i) {
            new (&slots_[i]) FrameSlot();
        }
    }

    ~ZeroCopyRingBuffer() {
        if (slots_ && slots_ != MAP_FAILED) {
            for (size_t i = 0; i < Capacity; ++i) {
                slots_[i].~FrameSlot();
            }
            munmap(slots_, total_bytes_);
        }
    }

    ZeroCopyRingBuffer(const ZeroCopyRingBuffer&) = delete;
    ZeroCopyRingBuffer& operator=(const ZeroCopyRingBuffer&) = delete;

    // --- CÔTÉ RÉSEAU (PRODUCTEUR) ---
    FrameSlot* get_slot_for_writing(size_t frame_index) {
        size_t index = frame_index % Capacity;
        FrameSlot* slot = &slots_[index];

        // SÉCURITÉ ANTI-CORRUPTION : Si l'IA n'a pas fini de lire ce slot, on attend.
        while (slot->in_use.load(std::memory_order_acquire)) {
            _mm_pause(); // 
        }

        return slot;
    }

    void publish_frame(FrameSlot* slot) {
        // Marque le slot comme verrouillé pour le passage à l'IA
        slot->in_use.store(true, std::memory_order_release);
    }

    // --- CÔTÉ IA (CONSOMMATEUR) ---
    void release_slot(FrameSlot* slot) {
        // L'IA libère le slot : le réseau peut à nouveau écrire dessus
        slot->in_use.store(false, std::memory_order_release);
    }

private:
    FrameSlot* slots_;
    size_t total_bytes_;
};

// -----------------------------------------------------------------------------
// 3. FILE D'ATTENTE SPSC LOCK-FREE (PASSE-POINTEURS)
// -----------------------------------------------------------------------------
template <typename T, size_t Capacity>
class SPSCQueue {
public:
    SPSCQueue() : head_(0), tail_(0) {}

    bool push(const T& item) {
        const auto current_tail = tail_.load(std::memory_order_relaxed);
        if (current_tail - head_.load(std::memory_order_acquire) == Capacity) {
            return false; // Queue pleine
        }
        buffer_[current_tail % Capacity] = item;
        tail_.store(current_tail + 1, std::memory_order_release);
        return true;
    }

    bool pop(T& item) {
        const auto current_head = head_.load(std::memory_order_relaxed);
        if (current_head == tail_.load(std::memory_order_acquire)) {
            return false; // Queue vide
        }
        item = buffer_[current_head % Capacity];
        _mm_prefetch(reinterpret_cast<const char*>(&buffer_[(current_head + 1) % Capacity]), _MM_HINT_T0);
        head_.store(current_head + 1, std::memory_order_release);
        return true;
    }

private:
    T buffer_[Capacity];
    alignas(64) std::atomic<size_t> head_;
    alignas(64) std::atomic<size_t> tail_;
};

// -----------------------------------------------------------------------------
// 4. INSTANCIATION DES STRUCTURES GLOBALES (PRÉ-ALLOUÉES AU DÉMARRAGE)
// -----------------------------------------------------------------------------
constexpr size_t RING_BUFFER_CAPACITY = 1024;
ZeroCopyRingBuffer<RING_BUFFER_CAPACITY> g_ring_buffer;
SPSCQueue<FrameSlot*, RING_BUFFER_CAPACITY> net_to_ai_queue;

// Ancrage d'un thread sur un cœur CPU physique dédié
void fix_thread_to_core(std::thread& th, int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(th.native_handle(), sizeof(cpu_set_t), &cpuset);
}

// -----------------------------------------------------------------------------
// 5. THREAD RÉSEAU (INGESTION IO_URING + ARENA ZERO-COPY)
// -----------------------------------------------------------------------------
void network_thread_io_uring() {
    struct io_uring ring;
    if (io_uring_queue_init(256, &ring, 0) < 0) {
        std::cerr << "[ERROR] Échec de l'initialisation de io_uring" << std::endl;
        return;
    }

    size_t frame_counter = 0;

    while (true) {
        // 1. Récupération sécurisée du slot dans le Ring Buffer
        FrameSlot* slot = g_ring_buffer.get_slot_for_writing(frame_counter);

        // 2. Écriture des métadonnées
        slot->data.frame_id = static_cast<uint32_t>(++frame_counter);
        auto now = std::chrono::high_resolution_clock::now();
        slot->data.timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
            now.time_since_epoch()).count();

        // SIMULATION RÉSEAU : Écriture directe dans le buffer du slot (Zero-Copy)
        std::memset(slot->data.payload, 0, PAYLOAD_SIZE);

        // 3. Verrouillage du slot pour l'IA
        g_ring_buffer.publish_frame(slot);

        // 4. Transmission de l'adresse 64-bit du slot dans la SPSC Queue
        while (!net_to_ai_queue.push(slot)) {
            _mm_pause();
        }

        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }

    io_uring_queue_exit(&ring);
}

// -----------------------------------------------------------------------------
// 6. THREAD IA (INFÉRENCE ONNX RUNTIME ZERO-COPY)
// -----------------------------------------------------------------------------
void ai_processing_thread() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "PipelineVision");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    Ort::Session session(env, "model.onnx", session_options);
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeDefault);

    std::vector<int64_t> input_node_dims = {1, 3, 224, 224};
    size_t input_tensor_size = 1 * 3 * 224 * 224;

    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};

    FrameSlot* slot = nullptr;
    uint64_t processed_frames = 0;

    while (true) {
        // Dépilage de l'adresse du slot depuis la file d'attente
        if (net_to_ai_queue.pop(slot)) {
            
            // Pointeur brut vers les pixels stockés dans le Ring Buffer (Pas de memcpy)
            float* input_data_ptr = reinterpret_cast<float*>(slot->data.payload);

            // Bind Zero-Copy avec ONNX Runtime : ONNX lit directement en RAM
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, 
                input_data_ptr, 
                input_tensor_size, 
                input_node_dims.data(), 
                input_node_dims.size()
            );

            auto output_tensors = session.Run(
                Ort::RunOptions{nullptr}, 
                input_names, 
                &input_tensor, 
                1, 
                output_names, 
                1
            );

            float* float_results = output_tensors[0].GetTensorMutableData<float>();

            auto now = std::chrono::high_resolution_clock::now();
            auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(
                now.time_since_epoch()).count();

            uint64_t latency = now_us - slot->data.timestamp_us;
            processed_frames++;

            if (processed_frames % 1000 == 0) {
                std::cout << "[IA ONNX Zero-Copy] Frame ID: " << slot->data.frame_id 
                          << " | Latence globale: " << latency << " us" 
                          << " | Output[0]: " << float_results[0] << std::endl;
            }

            // DÉVERROUILLAGE : L'IA signale au réseau qu'elle a fini avec ce slot
            g_ring_buffer.release_slot(slot);

        } else {
            _mm_pause(); // File vide : attente active sans surcharge CPU
        }
    }
}

// -----------------------------------------------------------------------------
// 7. POINT D'ENTRÉE PRINCIPAL
// -----------------------------------------------------------------------------
int main() {
    std::cout << "Démarrage du moteur temps réel (ZeroCopyRingBuffer + ONNX)..." << std::endl;

    std::thread net_thread(network_thread_io_uring);
    std::thread ai_thread(ai_processing_thread);

    // Isolation matérielle : Thread Réseau sur Cœur 0, Thread IA sur Cœur 1
    fix_thread_to_core(net_thread, 0);
    fix_thread_to_core(ai_thread, 1);

    net_thread.join();
    ai_thread.join();

    return 0;
}