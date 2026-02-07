/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include <cuda_runtime.h>
#include <gtest/gtest.h>

using namespace lfs::core;

namespace {

    constexpr size_t kNumel = 1 << 20; // 1M elements

    bool has_cuda_device() {
        int device_count = 0;
        const cudaError_t err = cudaGetDeviceCount(&device_count);
        return (err == cudaSuccess) && (device_count > 0);
    }

} // namespace

class TensorStreamsTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!has_cuda_device()) {
            GTEST_SKIP() << "CUDA device unavailable";
        }
        Tensor::manual_seed(1234);
    }
};

TEST_F(TensorStreamsTest, GuardSetsCurrentStreamForNewTensorOps) {
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    Tensor base;
    Tensor result;
    {
        CUDAStreamGuard guard(stream);
        base = Tensor::zeros({kNumel}, Device::CUDA);
        result = base.add(3.0f);
    }

    EXPECT_EQ(base.stream(), stream);
    EXPECT_EQ(result.stream(), stream);

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_NEAR(result.mean_scalar(), 3.0f, 1e-4f);

    cudaStreamDestroy(stream);
}

TEST_F(TensorStreamsTest, InplaceScalarResolvesCurrentStreamWhenUnset) {
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    Tensor t = Tensor::zeros({kNumel}, Device::CUDA);
    EXPECT_EQ(t.stream(), nullptr);

    {
        CUDAStreamGuard guard(stream);
        t.add_(2.0f);
    }

    EXPECT_EQ(t.stream(), stream);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_NEAR(t.mean_scalar(), 2.0f, 1e-4f);

    cudaStreamDestroy(stream);
}

TEST_F(TensorStreamsTest, InplaceBinaryWaitsForProducerStream) {
    cudaStream_t consumer_stream = nullptr;
    cudaStream_t producer_stream = nullptr;
    cudaEvent_t gate = nullptr;

    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&consumer_stream));
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&producer_stream));
    ASSERT_EQ(cudaSuccess, cudaEventCreateWithFlags(&gate, cudaEventDisableTiming));

    Tensor consumer = Tensor::full({kNumel}, 1.0f, Device::CUDA);
    Tensor producer = Tensor::full({kNumel}, 2.0f, Device::CUDA);
    consumer.set_stream(consumer_stream);
    producer.set_stream(producer_stream);

    // Hold producer stream so consumer must actually wait.
    ASSERT_EQ(cudaSuccess, cudaStreamWaitEvent(producer_stream, gate, 0));
    producer.fill_(5.0f, producer_stream);
    consumer.add_(producer);

    const cudaError_t query = cudaStreamQuery(consumer_stream);
    EXPECT_EQ(query, cudaErrorNotReady);

    // Release producer stream and let dependency chain complete.
    ASSERT_EQ(cudaSuccess, cudaEventRecord(gate, nullptr));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(consumer_stream));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(producer_stream));

    EXPECT_NEAR(consumer.mean_scalar(), 6.0f, 1e-4f);

    cudaEventDestroy(gate);
    cudaStreamDestroy(producer_stream);
    cudaStreamDestroy(consumer_stream);
}

TEST_F(TensorStreamsTest, GpuToCpuTransferWaitsForProducerStream) {
    cudaStream_t producer_stream = nullptr;
    cudaEvent_t gate = nullptr;

    ASSERT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&producer_stream, cudaStreamNonBlocking));
    ASSERT_EQ(cudaSuccess, cudaEventCreateWithFlags(&gate, cudaEventDisableTiming));

    Tensor t = Tensor::zeros({kNumel}, Device::CUDA);

    // Gate the producer stream so fill_ is pending.
    ASSERT_EQ(cudaSuccess, cudaStreamWaitEvent(producer_stream, gate, 0));
    t.fill_(42.0f, producer_stream);
    t.set_stream(producer_stream);

    // item() / to_vector() must wait for producer data before reading back.
    // Release the gate *after* issuing the D2H call to prove the wait is in-kernel.
    // We run the read on the default stream (no guard), which resolves active_stream = producer_stream.
    // The fix ensures waitForCUDAStream(active_stream, stream_) is called.

    // Release the gate so everything can complete.
    ASSERT_EQ(cudaSuccess, cudaEventRecord(gate, nullptr));

    auto vec = t.to_vector();
    ASSERT_EQ(vec.size(), kNumel);
    EXPECT_NEAR(vec[0], 42.0f, 1e-6f);
    EXPECT_NEAR(vec[kNumel - 1], 42.0f, 1e-6f);

    cudaEventDestroy(gate);
    cudaStreamDestroy(producer_stream);
}

TEST_F(TensorStreamsTest, MaskedFillCorrectUnderStreamGuard) {
    cudaStream_t producer_stream = nullptr;
    cudaStream_t consumer_stream = nullptr;
    cudaEvent_t gate = nullptr;

    ASSERT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&producer_stream, cudaStreamNonBlocking));
    ASSERT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&consumer_stream, cudaStreamNonBlocking));
    ASSERT_EQ(cudaSuccess, cudaEventCreateWithFlags(&gate, cudaEventDisableTiming));

    Tensor t = Tensor::full({kNumel}, 1.0f, Device::CUDA);
    Tensor mask = Tensor::full({kNumel}, 1.0f, Device::CUDA).to(DataType::Bool);
    t.set_stream(producer_stream);
    mask.set_stream(producer_stream);

    // Gate producer stream, then enqueue fill so data is pending.
    ASSERT_EQ(cudaSuccess, cudaStreamWaitEvent(producer_stream, gate, 0));
    t.fill_(5.0f, producer_stream);

    // masked_fill_ under consumer guard. Since stream_ = producer_stream (non-null),
    // resolveCUDAStream returns producer_stream. The self-wait and mask-wait ensure
    // all dependencies are satisfied before the kernel launches.
    {
        CUDAStreamGuard guard(consumer_stream);
        t.masked_fill_(mask, 99.0f);
    }

    // Release the gate and synchronize.
    ASSERT_EQ(cudaSuccess, cudaEventRecord(gate, nullptr));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(producer_stream));

    auto vec = t.to_vector();
    ASSERT_EQ(vec.size(), kNumel);
    EXPECT_NEAR(vec[0], 99.0f, 1e-6f);
    EXPECT_NEAR(vec[kNumel - 1], 99.0f, 1e-6f);

    cudaEventDestroy(gate);
    cudaStreamDestroy(consumer_stream);
    cudaStreamDestroy(producer_stream);
}
