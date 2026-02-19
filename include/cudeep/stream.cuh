#pragma once

#include "common.cuh"
#include "error.cuh"

namespace cudeep {

class Stream {
public:
    Stream();
    ~Stream();

    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    Stream(Stream&& other) noexcept;
    Stream& operator=(Stream&& other) noexcept;

    cudaStream_t get() const { return stream_; }
    void synchronize();

    static Stream& default_stream();

private:
    cudaStream_t stream_ = nullptr;
    bool owns_ = true;
};

class Event {
public:
    Event();
    ~Event();

    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;

    void record(cudaStream_t stream = nullptr);
    void synchronize();
    float elapsed_ms(const Event& start) const;

    cudaEvent_t get() const { return event_; }

private:
    cudaEvent_t event_ = nullptr;
};

}  // namespace cudeep
