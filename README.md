[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/7lg-zjaN) [![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-2972f46106e565e64193e422d61a12cf1da4916b45550586e14ef0a7c637dd04.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=19493325) \# Assignment 6: Neural Networks and Health Data Analysis\
\# 作业6：神经网络与健康数据分析

## Overview

## 概述

This assignment introduces neural networks and their applications in healthcare data analysis. You'll work through three parts:\
本作业介绍神经网络及其在健康数据分析中的应用。你将完成三个部分：

1.  **Basic Neural Networks**: Implement a simple neural network for EMNIST character recognition
    1.  **基础神经网络**：实现一个用于EMNIST字符识别的简单神经网络
2.  **Convolutional Neural Networks**: Build a CNN for more complex image classification
    2.  **卷积神经网络**：构建一个用于更复杂图像分类的卷积神经网络（CNN）
3.  **Time Series Analysis**: Apply neural networks to ECG signal classification
    3.  **时间序列分析**：将神经网络应用于心电（ECG）信号分类

## Learning Objectives

## 学习目标

-   Implement and train neural networks using TensorFlow/Keras and/or Pytorch
    -   使用TensorFlow/Keras和/或PyTorch实现并训练神经网络
-   Apply CNNs for image classification
    -   应用卷积神经网络进行图像分类
-   Work with time series data using RNNs
    -   使用循环神经网络（RNN）处理时间序列数据
-   Evaluate model performance using appropriate metrics
    -   使用合适的指标评估模型性能
-   Interpret results in a healthcare context
    -   在健康医疗背景下解释结果

## Setup

## 环境配置

1.  **Install Dependencies**:

    1.  **安装依赖**：

    ``` bash
    pip install -r requirements.txt
    ```

2.  **Directory Structure**:

    2.  **目录结构**：

    ```         
    datasci223_assignment6/
    ├── models/              # Saved models
    ├── results/            # Evaluation metrics
    │   ├── part_1/        # Part 1 results
    │   ├── part_2/        # Part 2 results
    │   └── part_3/        # Part 3 results
    ├── logs/              # Training logs
    ├── data/              # Downloaded datasets
    ├── part1_neural_networks_basics.md
    ├── part2_cnn_classification.md
    ├── part3_ecg_analysis.md
    └── requirements.txt
    ```

    datasci223_assignment6/ ├── models/ \# 已保存的模型 ├── results/ \# 评估指标 │ ├── part_1/ \# 第一部分结果 │ ├── part_2/ \# 第二部分结果 │ └── part_3/ \# 第三部分结果 ├── logs/ \# 训练日志 ├── data/ \# 下载的数据集 ├── part1_neural_networks_basics.md ├── part2_cnn_classification.md ├── part3_ecg_analysis.md └── requirements.txt

## Part 1: Neural Networks Basics

## 第一部分：神经网络基础

-   Implement a simple neural network for EMNIST character recognition
    -   实现一个用于EMNIST字符识别的简单神经网络
-   Use dense layers, activation functions, and dropout
    -   使用全连接层、激活函数和Dropout
-   Save model as `models/emnist_classifier.keras`
    -   将模型保存为 `models/emnist_classifier.keras`
-   Save metrics in `results/part_1/emnist_classifier_metrics.txt`
    -   将评估指标保存为 `results/part_1/emnist_classifier_metrics.txt`

Goals:\
目标： - Achieve \> 80% accuracy on test set\
- 测试集准确率达到80%以上 - Minimize overfitting using dropout\
- 通过Dropout最小化过拟合 - Train efficiently with appropriate batch size\
- 使用合适的批量大小高效训练

## Part 2: CNN Classification

## 第二部分：CNN分类

-   Implement a CNN for EMNIST classification
    -   实现一个用于EMNIST分类的卷积神经网络
-   Choose between TensorFlow/Keras or PyTorch
    -   可选择TensorFlow/Keras或PyTorch
-   Save model as:
    -   模型保存为：
        -   TensorFlow: `models/cnn_keras.keras`
            -   TensorFlow: `models/cnn_keras.keras`
        -   PyTorch: `models/cnn_pytorch.pt` and `models/cnn_pytorch_arch.txt`
            -   PyTorch: `models/cnn_pytorch.pt` 和 `models/cnn_pytorch_arch.txt`
-   Save metrics in `results/part_2/cnn_{framework}_metrics.txt`
    -   将评估指标保存为 `results/part_2/cnn_{framework}_metrics.txt`

Goals:\
目标： - Achieve \> 85% accuracy on test set\
- 测试集准确率达到85%以上 - Minimize overfitting using batch normalization and dropout\
- 通过批归一化和Dropout最小化过拟合 - Train efficiently with appropriate batch size and learning rate\
- 使用合适的批量大小和学习率高效训练

## Part 3: ECG Analysis

## 第三部分：心电图（ECG）分析

-   Work with MIT-BIH Arrhythmia Database
    -   使用MIT-BIH心律失常数据库
-   Choose between simple neural network or RNN/LSTM
    -   可选择简单神经网络或RNN/LSTM
-   Save model as `models/ecg_classifier_{model_type}.keras`
    -   将模型保存为 `models/ecg_classifier_{model_type}.keras`
-   Save metrics in `results/part_3/ecg_classifier_{model_type}_metrics.txt`
    -   将评估指标保存为 `results/part_3/ecg_classifier_{model_type}_metrics.txt`

Goals:\
目标： - Achieve \> 75% accuracy on test set\
- 测试集准确率达到75%以上 - Achieve AUC \> 0.80\
- AUC达到0.80以上 - Achieve F1-score \> 0.70\
- F1分数达到0.70以上 - Minimize overfitting using appropriate techniques\
- 使用合适的技术最小化过拟合 - Train efficiently with appropriate batch size\
- 使用合适的批量大小高效训练

## Framework Options

## 框架选项

1.  **TensorFlow/Keras**:

    1.  **TensorFlow/Keras**：

    -   Simpler syntax and more examples
        -   语法更简单，示例更多
    -   Save models as `.keras` files
        -   模型保存为`.keras`文件

2.  **PyTorch**:

    2.  **PyTorch**：

    -   Better for research and customization
        -   更适合研究和自定义
    -   Save models as `.pt` files with architecture in separate `.txt` file
        -   模型保存为`.pt`文件，结构另存为`.txt`文件

## Common Issues and Solutions

## 常见问题与解决方案

1.  **Data Loading**:

    1.  **数据加载**：

    -   Problem: Dataset not found
        -   问题：未找到数据集
    -   Solution: Check directory structure and download scripts
        -   解决方案：检查目录结构和下载脚本

2.  **Model Training**:

    2.  **模型训练**：

    -   Problem: Training instability
        -   问题：训练不稳定
    -   Solution: Use batch normalization, reduce learning rate
        -   解决方案：使用批归一化，降低学习率
    -   Problem: Overfitting
        -   问题：过拟合
    -   Solution: Increase dropout, use data augmentation
        -   解决方案：增加Dropout，使用数据增强

3.  **Evaluation**:

    3.  **评估**：

    -   Problem: Metrics format incorrect
        -   问题：指标格式不正确
    -   Solution: Follow the exact format specified
        -   解决方案：严格按照指定格式
    -   Problem: Model performance below threshold
        -   问题：模型性能低于阈值
    -   Solution: Adjust architecture, hyperparameters
        -   解决方案：调整结构和超参数

## Resources

## 资源

1.  **Documentation**:

    1.  **文档**：

    -   [TensorFlow Guide](https://www.tensorflow.org/guide)
        -   [TensorFlow 指南](https://www.tensorflow.org/guide)
    -   [PyTorch Tutorials](https://pytorch.org/tutorials/)
        -   [PyTorch 教程](https://pytorch.org/tutorials/)
    -   [MIT-BIH Database](https://www.physionet.org/content/mitdb/1.0.0/)
        -   [MIT-BIH 数据库](https://www.physionet.org/content/mitdb/1.0.0/)
    -   [TensorFlow Neural Networks](https://www.tensorflow.org/tutorials)
        -   [TensorFlow 神经网络教程](https://www.tensorflow.org/tutorials)
    -   [PyTorch CNNs](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
        -   [PyTorch CNN 教程](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
    -   [RNN/LSTM Guide](https://www.tensorflow.org/guide/keras/rnn)
        -   [RNN/LSTM 指南](https://www.tensorflow.org/guide/keras/rnn)