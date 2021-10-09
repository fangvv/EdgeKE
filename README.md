## EdgeKE

This is the source code for our paper: **EdgeKE: An On-Demand Deep Learning IoT System for Cognitive Big Data on Industrial Edge Devices**. A brief introduction of this work is as follows:

> Motivated by the prospects of 5G communications and industrial Internet of Things (IoT), recent years have seen the rise of a new computing paradigm, edge computing, which shifts data analytics to network edges that are at the proximity of big data sources. Although Deep Neural Networks (DNNs) have been extensively used in many platforms and scenarios, they are usually both compute and memory intensive, thus difficult to be deployed on resource-limited edge devices and in performance-demanding edge applications. Hence, there is an urgent need for techniques that enable DNN models to fit into edge devices, while ensuring acceptable execution costs and inference accuracy. This paper proposes an on-demand DNN model inference system for industrial edge devices, called Knowledge distillation and Early-exit on Edge (EdgeKE). It focuses on two design knobs: (1) DNN compression based on knowledge distillation, which trains the compact edge models under the supervision of large complex models for improving accuracy and speed; (2) DNN acceleration based on early-exit, which provides flexible choices for satisfying distinct latency or accuracy requirements from edge applications. By extensive evaluations on the CIFAR100 dataset and across three state-of-art edge devices, experimental results demonstrate that EdgeKE significantly outperforms the baseline models in terms of inference latency and memory footprint, while maintaining competitive classification accuracy. Furthermore, EdgeKE is verified to be efficiently adaptive to the application requirements on inference performance. The accuracy loss is within 4.84% under various latency constraints, and the speedup ratio is up to 3.30$\times$ under various accuracy requirements.

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

Weiwei Fang (fangvv@qq.com)

