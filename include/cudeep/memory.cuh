#pragma once

#include "common.cuh"
#include "error.cuh"

#include <cstddef>
#include <unordered_map>
#include <vector>
#include <mutex>

namespace cudeep {

class MemoryPool {
public:
    static MemoryPool& instance();

    void* allocate(size_t nbytes);
    void deallocate(void* ptr);
    void release_cached();
    size_t allocated_bytes() const { return allocated_bytes_; }
    size_t cached_bytes() const { return cached_bytes_; }

    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;

private:
    MemoryPool() = default;
    ~MemoryPool();

    struct Block {
        void* ptr;
        size_t size;
    };

    std::unordered_map<size_t, std::vector<Block>> free_blocks_;
    std::unordered_map<void*, size_t> allocated_blocks_;
    std::mutex mutex_;
    size_t allocated_bytes_ = 0;
    size_t cached_bytes_ = 0;
};

void* device_malloc(size_t nbytes);
void device_free(void* ptr);
void memcpy_h2d(void* dst, const void* src, size_t nbytes, cudaStream_t stream = nullptr);
void memcpy_d2h(void* dst, const void* src, size_t nbytes, cudaStream_t stream = nullptr);
void memcpy_d2d(void* dst, const void* src, size_t nbytes, cudaStream_t stream = nullptr);

}  // namespace cudeep
