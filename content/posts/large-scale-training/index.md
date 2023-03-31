---
title: "Resources: Training machine learning models at scale"
date: 2023-03-31
draft: false
---

The ability to train machine learning algorithms at scale has become increasingly important due to the growing size and complexity of datasets and models. Recently this has been enabled more widely by rapid developments in software and hardware techniques and tools. In this article, I outline useful resources for learning how to train machine learning algorithms at large scale. Although some tools were developed for specific deep learning frameworks, they introduce concepts that are generally applicable.


##### "Efficient Training on Multiple GPUs" [[link](https://huggingface.co/docs/transformers/perf_train_gpu_many)]

Describes parallelism methods for large-scale training on multiple machines: data parallelism, tensor parallelism, pipeline parallelism, and Zero Redundancy Optimizer (ZeRO) [[link](https://arxiv.org/pdf/1910.02054.pdf)]. It also includes a general strategy on deciding which technique — or combination of techniques — to use. 

The optimal set of techniques depends on the training setting and available hardware. Nonetheless, ZeRO is generally recommended across scenarios (e.g. when the model can fit in memory or not, in single or multi GPU settings). 

ZeRO reduces the memory consumption of each GPU by partitioning model training states (weights, gradients, and optimizer states) across available devices (this is called “sharding”) in the training hardware. It is however less effective on its own in the presence of slow internode connectivity.
---

##### "How to Train Really Large Models on Many GPUs?" [[link](https://lilianweng.github.io/posts/2021-09-25-train-large/)]

Similarly to the above, it motivates and discusses training parallelism methods, but with additional reference to papers using specific techniques. Parallelism can be combined with optimizing the memory footprint of training and its running speed — the article outlines standard ways: CPU offloading, mixed precision training, gradient checkpointing, compression and memory efficient optimisers.

---

##### "Efficient Training on a Single GPU" [[link](https://huggingface.co/docs/transformers/perf_train_gpu_one)]

This article discusses and demonstrates ways to reduce the memory footprint and speed of training on one GPU, for models that may not fit in memory. This includes: gradient accumulation, gradient checkpointing, mixed precision training, optimizing the batch size, optimizer choice, data loader design, and Microsoft's DeepSpeed ZerO. These constitute useful techniques for optimizing memory usage even if you plan to train on multiple machines because they can enable training at a larger-scale, or with fewer resources.

---

##### "Tensor Parallelism with jax.pjit" [[link](https://irhum.github.io/blog/pjit/)]

If you understand things better when you put numbers to them, this article is for you! It provides excellent demonstrations of different ways of applying tensor parallelism to neural network training, and a tutorial on how to implement it for a 15 billion parameter language model in JAX using `pjit`.

`pjit` is generally flexible and simple to use, however figuring out the optimal sharding dimensions needs careful thought to avoid duplicated tensor copies. This is an active development area for JAX, with new tools to distribute and automatically parallelize array computations being frequently released.

---
##### "Scalable Training of Language Models using JAX pjit and TPUv4" [[link](https://arxiv.org/pdf/2204.06514.pdf)]
This is technical report on Cohere's [[link](https://cohere.ai/)] distributed training framework, which utilizes TPU V4 Pods (a Pod is a group of TPU devices connected by high-speed interconnects) and JAX (specifically pjit) to perform efficient, large-scale, parallel computation. It includes useful practical considerations for training on multiple hosts unlike other articles. One of their key design conclusions is that as long as the model resides on a hardware unit with fast interconnect between accelerators, it's sufficient and even optimal to use tensor and data parallelism only. This is the case with TPUs, but less so for GPUs, especially for architectures prior to the H100 [[link](https://www.nvidia.com/en-gb/data-center/h100/)]. When interconnect is slow, due to the frequent communication needed by Tensor Parallelism, it's often used for parallelism across machines on a single host, and Pipeline Parallelism can be used for parallelism across hosts.

---

##### "Scaling Language Models: Methods, Analysis & Insights from Training Gopher" [[link](https://arxiv.org/pdf/2112.11446.pdf)]

This paper provides a detailed overview of the methods and insights gained from training a 10-280 million parameter transformer model (Gopher) and highlights the importance of using efficient training techniques to achieve high performance on large language models.

Specifically, section “C. Lessons Learned” includes a discussion on training with the adafactor optimiser instead of adam, and mixed-precision training. The Adafactor optimizer [[link](https://arxiv.org/abs/1804.04235)] can reduce the memory footprint of training compared to Adam with minimal code changes. Instead of keeping the rolling average for each element in the weight matrices, Adafactor only stores aggregated information (row- and column-wise sums of the rolling averages) which reduces the footprint considerably. The performance of adafactor however tends to be worse for extremely large models, and more unstable; these can be mitigated by lowering the learning rate and training for longer.
