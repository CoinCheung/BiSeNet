
#ifndef BATCH_STREAM_HPP
#define BATCH_STREAM_HPP


#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <opencv2/opencv.hpp>

#include "NvInfer.h"
#include "read_img.hpp"

using nvinfer1::Dims;
using nvinfer1::Dims3;
using nvinfer1::Dims4;


class IBatchStream
{
public:
    virtual void reset(int firstBatch) = 0;
    virtual bool next() = 0;
    virtual void skip(int skipCount) = 0;
    virtual float* getBatch() = 0;
    virtual int getBatchesRead() const = 0;
    virtual int getBatchSize() const = 0;
    virtual nvinfer1::Dims4 getDims() const = 0;
};


class BatchStream : public IBatchStream
{
public:
    BatchStream(int batchSize, int maxBatches, Dims indim,
            const std::string& dataRoot, 
            const std::string& dataFile)
        : mBatchSize{batchSize}
        , mMaxBatches{maxBatches}
    {
        mDims = Dims3(indim.d[1], indim.d[2], indim.d[3]);

        readDataFile(dataFile, dataRoot);
        mSampleSize = std::accumulate(
                mDims.d, mDims.d + mDims.nbDims, 1, std::multiplies<int64_t>()) * sizeof(float);
        mData.resize(mSampleSize * mBatchSize);
    }

    void reset(int firstBatch) override
    {
        cout << "mBatchCount: " << mBatchCount << endl;
        mBatchCount = firstBatch;
    }

    bool next() override
    {
        if (mBatchCount >= mMaxBatches)
        {
            return false;
        }
        ++mBatchCount;
        return true;
    }

    void skip(int skipCount) override
    {
        mBatchCount += skipCount;
    }

    float* getBatch() override
    {
        int offset = mBatchCount * mBatchSize;
        for (int i{0}; i < mBatchSize; ++i) {
            int ind = offset + i;
            read_data(mPaths[ind], &mData[i * mSampleSize], mDims.d[1], mDims.d[2]);
        }
        return mData.data();
    }

    int getBatchesRead() const override
    {
        return mBatchCount;
    }

    int getBatchSize() const override
    {
        return mBatchSize;
    }

    nvinfer1::Dims4 getDims() const override
    {
        return Dims4{mBatchSize, mDims.d[0], mDims.d[1], mDims.d[2]};
    }

private:
    void readDataFile(const std::string& dataFilePath, const std::string& dataRootPath)
    {
        std::ifstream file(dataFilePath, std::ios::in);
        if (!file.is_open()) {
            cout << "file open failed: " << dataFilePath << endl;
            std::abort();
        }
        std::stringstream ss;
        file >> ss.rdbuf();
        file.close();

        std::string impth;
        int n_imgs = 0;
        while (std::getline(ss, impth)) ++n_imgs;
        ss.clear(); ss.seekg(0, std::ios::beg);
        if (n_imgs <= 0) {
            cout << "ann file is empty, cannot read image paths for int8 calibration: "
                << dataFilePath << endl;
            std::abort();
        }

        mPaths.resize(n_imgs);
        for (int i{0}; i < n_imgs; ++i) {
            std::getline(ss, impth, ',');
            mPaths[i] = dataRootPath + "/" + impth;
            std::getline(ss, impth);
        }
        if (mMaxBatches < 0) {
            mMaxBatches =  n_imgs / mBatchSize - 1;
        }
        if (mMaxBatches <= 0) {
            cout << "must have at least 1 batch for calibration\n";
            std::abort();
        }
        cout << "mMaxBatches = " << mMaxBatches << endl;
    }


    int mBatchSize{0};
    int mBatchCount{0}; 
    int mMaxBatches{0};
    Dims3 mDims{};
    std::vector<string> mPaths;
    std::vector<float> mData;
    int mSampleSize{0};
};


#endif 
