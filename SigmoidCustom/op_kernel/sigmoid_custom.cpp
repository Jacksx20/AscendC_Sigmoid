#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
class KernelSigmoid {
public:
    __aicore__ inline KernelSigmoid() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        //初始化block相关参数
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        //计算每个block处理的数据长度
        this->blockLength = totalLength / GetBlockNum();
        //保存tile数量
        this->tileNum = tileNum;
        //计算每个tile处理的数据长度
        ASSERT(tileNum != 0 && "tile num can not be zero!");
        //计算每个tile的长度
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;
        //设置全局内存地址
        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->blockLength * GetBlockIdx(), 
        this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + this->blockLength * GetBlockIdx(), 
        this->blockLength);
        //初始化pipe和queue
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(half));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(half));
        //初始化临时buffer
        pipe.InitBuffer(tmpBuffer1, this->tileLength * sizeof(half));
        pipe.InitBuffer(tmpBuffer2, this->tileLength * sizeof(half));
        pipe.InitBuffer(tmpBuffer3, this->tileLength * sizeof(half));
        pipe.InitBuffer(tmpBuffer4, this->tileLength * sizeof(half));

    }
    __aicore__ inline void Process()
    {
        //对“loopCount”的定义，注意对Tiling的处理
        int32_t loopCount = this->blockLength / this->tileLength;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        //考生补充算子代码
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        //考生补充算子计算代码
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        //考生补充算子代码
    }

private:
    TPipe pipe;
    //create queue for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    //create queue for output, in this case depth is equal to buffer num
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<half> xGm;
    GlobalTensor<half> yGm;

    //考生补充自定义成员变量

};
extern "C" __global__ __aicore__ void sigmoid_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelSigmoid op;
    //补充init和process函数调用内容
}