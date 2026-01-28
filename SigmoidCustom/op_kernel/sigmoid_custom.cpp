#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
class KernelSigmoid {
public:

    /**
     * @brief 构造一个KernelSigmoid对象
     * 
     * 该函数用于构造一个KernelSigmoid对象，适用于人工智能核心计算。
     */
    __aicore__ inline KernelSigmoid() {}

    /**
     * @brief 初始化函数
     *
     * @param x 输入x的全局内存地址
     * @param y 输出y的全局内存地址
     * @param totalLength 数据总长度
     * @param tileNum tile数量
     *
     * 此函数用于初始化block相关参数，包括每个block处理的数据长度，
     * 每个tile处理的数据长度，设置全局内存地址，并初始化pipe和queue。
     */

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

    /**
     * 处理函数，对数据进行处理。
     * 该函数通过对loopCount的循环，调用CopyIn、Compute和CopyOut函数，对数据进行处理。
     * loopCount: 循环次数，由blockLength和tileLength的商得到。
     */
    __aicore__ inline void Process()
    {
        //对“loopCount”的定义，注意对Tiling的处理
        int32_t loopCount = this->blockLength / this->tileLength;
        for (int32_t i = 0; i < loopCount; i++) {
             // 将数据复制到处理单元
            CopyIn(i);
            // 执行计算操作
            Compute(i);
            // 将结果复制到全局内存
            CopyOut(i);
        }
    }

private:

    /**
     * @brief 复制数据到本地张量并加入队列
     * @param progress 数据复制的起始位置标志
     * @details 此函数用于从全局内存中复制一段数据到本地张量，并将本地张量加入输入队列。
     * @note 本地张量xLocal的数据类型为half，即半精度浮点数。
     */
    __aicore__ inline void CopyIn(int32_t progress)
    {
        // 分配一个本地张量xLocal，数据类型为half
        LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        // 从全局内存xGm中复制数据到本地张量xLocal
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        // 将本地张量xLocal加入输入队列inQueueX
        inQueueX.EnQue(xLocal);
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