#include "cudeep/memory.cuh"

#include <algorithm>

namespace cudeep {

MemoryPool& MemoryPool::instance() {
    static MemoryPool pool;
    return pool;
}

MemoryPool::~MemoryPool() {
    release_cached();
    for (auto& [ptr, size] : allocated_blocks_) {
        cudaFree(ptr);
    }
}

void* MemoryPool::allocate(size_t nbytes) {
    std::lock_guard<std::mutex> lock(mutex_);

    size_t bin = nbytes;

    auto it = free_blocks_.find(bin);
    if (it != free_blocks_.end() && !it->second.empty()) {
        Block block = it->second.back();
        it->second.pop_back();
        cached_bytes_ -= block.size;
        allocated_blocks_[block.ptr] = block.size;
        allocated_bytes_ += block.size;
        return block.ptr;
    }

    void* ptr = nullptr;
    CUDEEP_CHECK_CUDA(cudaMalloc(&ptr, nbytes));
    allocated_blocks_[ptr] = nbytes;
    allocated_bytes_ += nbytes;
    return ptr;
}

void MemoryPool::deallocate(void* ptr) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = allocated_blocks_.find(ptr);
    if (it == allocated_blocks_.end()) return;

    size_t size = it->second;
    allocated_bytes_ -= size;
    allocated_blocks_.erase(it);

    free_blocks_[size].push_back({ptr, size});
    cached_bytes_ += size;
}

void MemoryPool::release_cached() {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto& [size, blocks] : free_blocks_) {
        for (auto& block : blocks) {
            cudaFree(block.ptr);
        }
    }
    free_blocks_.clear();
    cached_bytes_ = 0;
}

void* device_malloc(size_t nbytes) {
    return MemoryPool::instance().allocate(nbytes);
}

void device_free(void* ptr) {
    MemoryPool::instance().deallocate(ptr);
}

void memcpy_h2d(void* dst, const void* src, size_t nbytes, cudaStream_t stream) {
    if (stream) {
        CUDEEP_CHECK_CUDA(cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyHostToDevice, stream));
    } else {
        CUDEEP_CHECK_CUDA(cudaMemcpy(dst, src, nbytes, cudaMemcpyHostToDevice));
    }
}

void memcpy_d2h(void* dst, const void* src, size_t nbytes, cudaStream_t stream) {
    if (stream) {
        CUDEEP_CHECK_CUDA(cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDeviceToHost, stream));
    } else {
        CUDEEP_CHECK_CUDA(cudaMemcpy(dst, src, nbytes, cudaMemcpyDeviceToHost));
    }
}

void memcpy_d2d(void* dst, const void* src, size_t nbytes, cudaStream_t stream) {
    if (stream) {
        CUDEEP_CHECK_CUDA(cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDeviceToDevice, stream));
    } else {
        CUDEEP_CHECK_CUDA(cudaMemcpy(dst, src, nbytes, cudaMemcpyDeviceToDevice));
    }
}

}  // namespace cudeep
