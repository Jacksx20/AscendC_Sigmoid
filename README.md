# AscendC_Sigmoid
Experiment on Sigmoid Custom Operator for Ascend C Operator Development

Ascend C算子开发之Sigmoid自定义算子的实验

## 环境搭建要求

环境上要有昇腾NPU，且CANN版本为8.0.0.beta1。请开发者自行准备。

典型场景举例（若指导文档中CANN版本号与"8.0.0.beta1"不一致，请自行调整）：

开发者套件（Atlas200I DK A2，或香橙派）[部署方式](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/resource/Atlas%20200I%20DK%20A2%E7%AE%97%E5%AD%90%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%AF%BC-Ascendc%E4%B8%AD%E7%BA%A7%E8%AE%A4%E8%AF%81%E4%B8%93%E7%94%A8.docx)

华为云-ModelArts-Notebook [部署方式](https://public-download.obs.cn-east-2.myhuaweicloud.com/%E5%BE%AE%E8%AE%A4%E8%AF%81/%E5%8D%8E%E4%B8%BA%E4%BA%91Ascend%20C%E7%AE%97%E5%AD%90%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%89%8B%E5%86%8C%E5%BE%AE%E8%AE%A4%E8%AF%810611.docx)

## 题目

实现Ascend C算子Sigmoid,算子命名为SigmoidCustom，编写其kernel侧代码、host侧代码，并完成aclnn算子调用测试。
相关算法：

                           sigmoid(x) = 1/(1 + exp(-x))

要求：

1 完成Sigmoid算子kernel侧核函数相关代码补齐。

2 完成Sigmoid算子host侧Tiling结构体成员变量创建，以及Tiling实现函数的补齐。

3 要支持Float16类型输入输出。
