## EdgeKE

This is the source code for our paper: **EdgeKE: An On-Demand Deep Learning IoT System for Cognitive Big Data on Industrial Edge Devices**. A brief introduction of this work is as follows:

> Motivated by the prospects of 5G communications and industrial Internet of Things (IoT), recent years have seen the rise of a new computing paradigm, edge computing, which shifts data analytics to network edges that are at the proximity of big data sources. Although Deep Neural Networks (DNNs) have been extensively used in many platforms and scenarios, they are usually both compute and memory intensive, thus difficult to be deployed on resource-limited edge devices and in performance-demanding edge applications. Hence, there is an urgent need for techniques that enable DNN models to fit into edge devices, while ensuring acceptable execution costs and inference accuracy. This paper proposes an on-demand DNN model inference system for industrial edge devices, called Knowledge distillation and Early-exit on Edge (EdgeKE). It focuses on two design knobs: (1) DNN compression based on knowledge distillation, which trains the compact edge models under the supervision of large complex models for improving accuracy and speed; (2) DNN acceleration based on early-exit, which provides flexible choices for satisfying distinct latency or accuracy requirements from edge applications. By extensive evaluations on the CIFAR100 dataset and across three state-of-art edge devices, experimental results demonstrate that EdgeKE significantly outperforms the baseline models in terms of inference latency and memory footprint, while maintaining competitive classification accuracy. Furthermore, EdgeKE is verified to be efficiently adaptive to the application requirements on inference performance. The accuracy loss is within 4.84% under various latency constraints, and the speedup ratio is up to 3.30$\times$ under various accuracy requirements.

> 得益于5G通信与工业物联网（IIoT）的发展前景，近年来兴起了一种新型计算范式——边缘计算，其将数据分析任务转移至更接近大数据源的网络边缘。尽管深度神经网络（DNN）已被广泛应用于众多平台和场景，但其通常对计算和内存资源要求较高，难以部署在资源有限的边缘设备及性能要求严格的边缘应用中。因此，亟需一种能使DNN模型适配边缘设备，同时确保可接受的执行成本与推理精度的技术。本文提出了一种面向工业边缘设备的按需DNN模型推理系统EdgeKE（知识蒸馏与提前退出边缘系统），其核心聚焦两个设计维度：（1）基于知识蒸馏的DNN压缩技术，通过大型复杂模型监督训练紧凑型边缘模型，以提升精度与速度；（2）基于提前退出的DNN加速机制，为满足边缘应用不同延迟或精度需求提供灵活选择。通过在CIFAR100数据集及三种主流边缘设备上的广泛评估，实验结果表明：EdgeKE在推理延迟和内存占用方面显著优于基线模型，同时保持具有竞争力的分类精度。此外，EdgeKE被验证能有效自适应推理性能的应用需求——在不同延迟约束下精度损失控制在4.84%以内，在不同精度要求下加速比最高可达3.30倍。

This work has been published by IEEE Transactions on Industrial Informatics. Click [here](https://ieeexplore.ieee.org/document/9294146/) for our paper.

## Required software

PyTorch

## Citation

    @ARTICLE{9294146,
      author={Fang, Weiwei and Xue, Feng and Ding, Yi and Xiong, Naixue and Leung, Victor C. M.},
      journal={IEEE Transactions on Industrial Informatics}, 
      title={EdgeKE: An On-Demand Deep Learning IoT System for Cognitive Big Data on Industrial Edge Devices}, 
      year={2021},
      volume={17},
      number={9},
      pages={6144-6152},
      doi={10.1109/TII.2020.3044930}
	}


## Contact

Feng Xue (17120431@bjtu.edu.cn)

> Please note that the open source code in this repository was mainly completed by the graduate student author during his master's degree study. Since the author did not continue to engage in scientific research work after graduation, it is difficult to continue to maintain and update these codes. We sincerely apologize that these codes are for reference only.

