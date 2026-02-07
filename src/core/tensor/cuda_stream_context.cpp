/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "internal/cuda_stream_context.hpp"

namespace lfs::core {

    static thread_local cudaStream_t current_cuda_stream = nullptr;

    cudaStream_t getCurrentCUDAStream() {
        return current_cuda_stream;
    }

    void setCurrentCUDAStream(cudaStream_t stream) {
        current_cuda_stream = stream;
    }

} // namespace lfs::core
