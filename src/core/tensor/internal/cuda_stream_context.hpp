/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <core/export.hpp>
#include <cuda_runtime.h>

namespace lfs::core {

    LFS_CORE_API cudaStream_t getCurrentCUDAStream();
    LFS_CORE_API void setCurrentCUDAStream(cudaStream_t stream);

    /**
     * RAII guard for temporarily setting the current CUDA stream
     * (PyTorch's CUDAStreamGuard pattern)
     *
     * Usage in DataLoader worker:
     *   cudaStream_t worker_stream;
     *   cudaStreamCreate(&worker_stream);
     *   {
     *       CUDAStreamGuard guard(worker_stream);
     *       // All tensor operations in this scope use worker_stream
     *       auto image = load_image();
     *       image = image.to(Device::CUDA);  // Uses worker_stream!
     *       image = preprocess(image);        // Uses worker_stream!
     *   }
     *   // Stream restored to previous value
     */
    class CUDAStreamGuard {
    public:
        explicit CUDAStreamGuard(cudaStream_t stream)
            : prev_stream_(getCurrentCUDAStream()) {
            setCurrentCUDAStream(stream);
        }

        ~CUDAStreamGuard() {
            setCurrentCUDAStream(prev_stream_);
        }

        // Delete copy/move
        CUDAStreamGuard(const CUDAStreamGuard&) = delete;
        CUDAStreamGuard& operator=(const CUDAStreamGuard&) = delete;
        CUDAStreamGuard(CUDAStreamGuard&&) = delete;
        CUDAStreamGuard& operator=(CUDAStreamGuard&&) = delete;

    private:
        cudaStream_t prev_stream_;
    };

    // Resolve explicit stream or fall back to current thread-local stream.
    inline cudaStream_t resolveCUDAStream(cudaStream_t stream = nullptr) {
        return stream ? stream : getCurrentCUDAStream();
    }

    // Ensure producer stream work is visible to consumer stream without global sync.
    inline cudaError_t waitForCUDAStream(cudaStream_t consumer_stream, cudaStream_t producer_stream) {
        if (producer_stream == consumer_stream) {
            return cudaSuccess;
        }

        // Legacy default stream already synchronizes with "blocking" streams.
        // Only inject explicit deps when a non-blocking stream is involved.
        auto stream_is_nonblocking = [](cudaStream_t stream, bool* is_nonblocking) -> cudaError_t {
            if (stream == nullptr) {
                *is_nonblocking = false;
                return cudaSuccess;
            }
            unsigned int flags = 0;
            cudaError_t err = cudaStreamGetFlags(stream, &flags);
            if (err != cudaSuccess) {
                return err;
            }
            *is_nonblocking = (flags & cudaStreamNonBlocking) != 0;
            return cudaSuccess;
        };

        if (consumer_stream == nullptr || producer_stream == nullptr) {
            bool consumer_nonblocking = false;
            bool producer_nonblocking = false;
            cudaError_t err = stream_is_nonblocking(consumer_stream, &consumer_nonblocking);
            if (err != cudaSuccess) {
                return err;
            }
            err = stream_is_nonblocking(producer_stream, &producer_nonblocking);
            if (err != cudaSuccess) {
                return err;
            }

            if (!consumer_nonblocking && !producer_nonblocking) {
                return cudaSuccess;
            }
        }

        struct ThreadLocalEvent {
            cudaEvent_t event = nullptr;
            ~ThreadLocalEvent() {
                if (event) {
                    cudaEventDestroy(event);
                }
            }
        };

        thread_local ThreadLocalEvent tls_event;
        if (!tls_event.event) {
            cudaError_t create_err = cudaEventCreateWithFlags(&tls_event.event, cudaEventDisableTiming);
            if (create_err != cudaSuccess) {
                return create_err;
            }
        }

        cudaError_t err = cudaEventRecord(tls_event.event, producer_stream);
        if (err == cudaSuccess) {
            err = cudaStreamWaitEvent(consumer_stream, tls_event.event, 0);
        }
        return err;
    }

    // Sync only the relevant stream when crossing to host-visible API boundaries.
    inline cudaError_t synchronizeCUDAStream(cudaStream_t stream) {
        return cudaStreamSynchronize(stream);
    }

} // namespace lfs::core
