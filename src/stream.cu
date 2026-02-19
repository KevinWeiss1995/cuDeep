#include "cudeep/stream.cuh"

namespace cudeep {

Stream::Stream() {
    CUDEEP_CHECK_CUDA(cudaStreamCreate(&stream_));
}

Stream::~Stream() {
    if (owns_ && stream_) {
        cudaStreamDestroy(stream_);
    }
}

Stream::Stream(Stream&& other) noexcept
    : stream_(other.stream_), owns_(other.owns_) {
    other.stream_ = nullptr;
    other.owns_ = false;
}

Stream& Stream::operator=(Stream&& other) noexcept {
    if (this != &other) {
        if (owns_ && stream_) cudaStreamDestroy(stream_);
        stream_ = other.stream_;
        owns_ = other.owns_;
        other.stream_ = nullptr;
        other.owns_ = false;
    }
    return *this;
}

void Stream::synchronize() {
    CUDEEP_CHECK_CUDA(cudaStreamSynchronize(stream_));
}

Stream& Stream::default_stream() {
    static Stream s;
    return s;
}

// --- Event ---

Event::Event() {
    CUDEEP_CHECK_CUDA(cudaEventCreate(&event_));
}

Event::~Event() {
    if (event_) {
        cudaEventDestroy(event_);
    }
}

void Event::record(cudaStream_t stream) {
    CUDEEP_CHECK_CUDA(cudaEventRecord(event_, stream));
}

void Event::synchronize() {
    CUDEEP_CHECK_CUDA(cudaEventSynchronize(event_));
}

float Event::elapsed_ms(const Event& start) const {
    float ms = 0.0f;
    CUDEEP_CHECK_CUDA(cudaEventElapsedTime(&ms, start.event_, event_));
    return ms;
}

}  // namespace cudeep
