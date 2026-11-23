# 中文多模态新闻伪造基准

本仓库对应论文 **A Chinese Multimodal News Dataset with Multiple Forgery Types: Construction and Benchmarking**，提供基准评测代码及相关说明。

[English](README.md) | [Hugging Face 数据集](https://huggingface.co/datasets/Wangyq112233/chinese-multimodal-news-forgery)

## 数据集概述

数据集包含21,249条中文新闻样本，其中原始新闻10,739条，受控构造的伪造样本10,510条。数据集以新浪新闻平台上来源可追溯的完整新闻报道为基础，涵盖文本篡改、图像篡改、跨事件图文错配和图文联合伪造四种形式。

样本经自动检查与人工复核，并依据实际执行的构造操作标注文本修改状态、整体伪造状态和具体伪造类别。

## 数据集获取

数据集文件单独托管于 **Hugging Face**：

**[中文多模态新闻伪造数据集](https://huggingface.co/datasets/Wangyq112233/chinese-multimodal-news-forgery)**

请在数据集页面提交访问申请，填写所属机构及预期用途。申请由人工审核。

访问获批后，请从Hugging Face获取所需文件，并按照基准代码README中的说明放置数据。

## 基准任务

基准评测设置三项任务：

- 文本篡改识别
- 多模态伪造检测
- 细粒度类别识别

基准实验子集包含5,574条样本，按原报道的发布时间顺序划分为3,474条训练样本、1,050条验证样本和1,050条测试样本。

## 许可说明

代码许可及第三方实现的使用条款见 `LICENSE`、`THIRD_PARTY_NOTICES.md` 和 `LICENSES/`。

数据集使用条件见Hugging Face数据集页面。

## 引用

使用本数据集或基准时，请引用对应论文：

*A Chinese Multimodal News Dataset with Multiple Forgery Types: Construction and Benchmarking*。

完整文献信息将在确定后补充。
