# AscendC_Sigmoid
Experiment on Sigmoid Custom Operator for Ascend C Operator Development

Ascend C算子开发之Sigmoid自定义算子的实验

## 环境搭建要求

环境上要有昇腾NPU，且CANN版本为8.0.0.beta1。请开发者自行准备。

典型场景举例（若指导文档中CANN版本号与"8.0.0.beta1"不一致，请自行调整）：

    开发者套件（Atlas200I DK A2，或香橙派）部署方式
    华为云-ModelArts-Notebook 部署方式

## 题目

实现Ascend C算子Sigmoid,算子命名为SigmoidCustom，编写其kernel侧代码、host侧代码，并完成aclnn算子调用测试。
相关算法：

                           sigmoid(x) = 1/(1 + exp(-x))

要求：

1 完成Sigmoid算子kernel侧核函数相关代码补齐。

2 完成Sigmoid算子host侧Tiling结构体成员变量创建，以及Tiling实现函数的补齐。

3 要支持Float16类型输入输出。
