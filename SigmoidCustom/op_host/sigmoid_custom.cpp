
#include "sigmoid_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
const uint32_t BLOCK_DIM = 8;
const uint32_t TILE_NUM = 8;

/**
 * @brief TilingFunc 函数用于设置Sigmoid激活函数的分块信息。
 * @param context 分块上下文，用于获取输入形状、设置分块维度、设置分块数量和保存分块数据。
 * @return 返回执行状态，成功返回ge::GRAPH_SUCCESS，失败返回ge::GRAPH_FAILED。
 */
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // 1. Get the shape of input0.
    // 2. Set the block dimension.
    // 3. Set the number of tiles.
    // 4. Save the tiling data.
    // 5. Return the execution status.
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // 设置tiling数据
    SigmoidCustomTilingData tiling;
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalLength(totalLength);
    tiling.set_tileNum(TILE_NUM);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}


namespace ge {

/**
 * @brief 推理形状函数
 *
 * 该函数用于推理输入和输出的形状。
 * 在这个场景中，输出的形状直接设置为输入的形状。
 *
 * @param context 推理形状上下文，用于获取输入和输出的形状
 * @return ge::graphStatus 图形状态，返回GRAPH_SUCCESS表示成功
 */
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}

/**
 * @brief 推理数据类型函数
 * @param context 推理数据类型上下文指针
 * @return 图像处理状态
 * @note 此函数用于推理数据类型，根据输入数据类型设置输出数据类型
 */
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
   // 获取输入数据类型 
const auto inputDataType = context->GetInputDataType(0);
// 设置输出数据类型为输入数据类型
context->SetOutputDataType(0, inputDataType);
// 返回图像处理成功状态
return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class SigmoidCustom : public OpDef {
public:

    /**
     * @brief 构造函数，初始化SigmoidCustom类
     * @param name 操作名称
     * @details 在构造函数中，定义了输入输出参数类型、数据类型、格式等属性，
     * 并设置了推理形状和数据类型函数，最后配置了AI Core相关参数
     */
    explicit SigmoidCustom(const char* name) : OpDef(name)
    {
        // 设置输入参数
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        // 设置输出参数
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        // 设置推理形状和数据类型函数
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        // 配置AI Core相关参数
        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b")
                      .AddConfig("ascend310b");
    }
};

OP_ADD(SigmoidCustom);
}
