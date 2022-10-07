/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ENTROPY_CALIBRATOR_HPP
#define ENTROPY_CALIBRATOR_HPP

#include <algorithm>
#include <numeric>
#include <iterator>
#include "NvInfer.h"

//! \class EntropyCalibratorImpl
//!
//! \brief Implements common functionality for Entropy calibrators.
//!
template <typename TBatchStream>
class EntropyCalibratorImpl
{
public:
    EntropyCalibratorImpl(
        TBatchStream stream, int firstBatch, std::string cal_table_name, const char* inputBlobName, bool readCache = true)
        : mStream{stream}
        , mCalibrationTableName(cal_table_name)
        , mInputBlobName(inputBlobName)
        , mReadCache(readCache)
    {
        nvinfer1::Dims4 dims = mStream.getDims();
        mInputCount = std::accumulate(
                dims.d, dims.d + dims.nbDims, 1, std::multiplies<int64_t>());
        cout << "dims.nbDims: " << dims.nbDims << endl;
        for (int i{0}; i < dims.nbDims; ++i) {
            cout << dims.d[i] << ", ";
        }
        cout << endl;

        cudaError_t state;
        state = cudaMalloc(&mDeviceInput, mInputCount * sizeof(float));
        if (state) {
            cout << "allocate memory failed\n"; 
            std::abort();
        }
        cout << "mInputCount: " << mInputCount << endl;
        mStream.reset(firstBatch);
    }

    virtual ~EntropyCalibratorImpl()
    {
        cudaError_t state;
        state = cudaFree(mDeviceInput);
        if (state) {
            cout << "free memory failed\n"; 
            std::abort();
        }
    }

    int getBatchSize() const
    {
        return mStream.getBatchSize();
    }

    bool getBatch(void* bindings[], const char* names[], int nbBindings)
    {
        if (!mStream.next())
        {
            return false;
        }
        cudaError_t state;
        state = cudaMemcpy(mDeviceInput, mStream.getBatch(), mInputCount * sizeof(float), cudaMemcpyHostToDevice);
        if (state) {
            cout << "memory copy to device failed\n"; 
            std::abort();
        }
        assert(!strcmp(names[0], mInputBlobName));
        bindings[0] = mDeviceInput;
        return true;
    }

    const void* readCalibrationCache(size_t& length)
    {
        mCalibrationCache.clear();
        std::ifstream input(mCalibrationTableName, std::ios::binary);
        input >> std::noskipws;
        if (mReadCache && input.good())
        {
            std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(),
                std::back_inserter(mCalibrationCache));
        }
        length = mCalibrationCache.size();
        return length ? mCalibrationCache.data() : nullptr;
    }

    void writeCalibrationCache(const void* cache, size_t length)
    {
        std::ofstream output(mCalibrationTableName, std::ios::binary);
        output.write(reinterpret_cast<const char*>(cache), length);
    }

private:
    TBatchStream mStream;
    size_t mInputCount;
    std::string mCalibrationTableName;
    const char* mInputBlobName;
    bool mReadCache{true};
    void* mDeviceInput{nullptr};
    std::vector<char> mCalibrationCache;
};

//! \class Int8EntropyCalibrator2
//!
//! \brief Implements Entropy calibrator 2.
//!  CalibrationAlgoType is kENTROPY_CALIBRATION_2.
//!
template <typename TBatchStream>
class Int8EntropyCalibrator2 : public nvinfer1::IInt8EntropyCalibrator2
{
public:
    Int8EntropyCalibrator2(
        TBatchStream stream, int firstBatch, const char* networkName, const char* inputBlobName, bool readCache = true)
        : mImpl(stream, firstBatch, networkName, inputBlobName, readCache)
    {
    }

    int getBatchSize() const noexcept override
    {
        return mImpl.getBatchSize();
    }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override
    {
        return mImpl.getBatch(bindings, names, nbBindings);
    }

    const void* readCalibrationCache(size_t& length) noexcept override
    {
        return mImpl.readCalibrationCache(length);
    }

    void writeCalibrationCache(const void* cache, size_t length) noexcept override
    {
        mImpl.writeCalibrationCache(cache, length);
    }

private:
    EntropyCalibratorImpl<TBatchStream> mImpl;
};

#endif // ENTROPY_CALIBRATOR_H
