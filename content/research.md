
##### Adversarially Robust Kernel Smoothing
*JJ Zhu, **C Kouridi**, Y Nemmour, B Schölkopf*  
*25th International Conference on Artificial Intelligence and Statistics (AISTATS) 2022*

We propose the adversarially robust kernel smoothing (ARKS) algorithm, combining kernel smoothing, robust optimization, and adversarial training for robust learning. Our methods are motivated by the convex analysis perspective of distributionally robust optimization based on probability metrics, such as the Wasserstein distance and the maximum mean discrepancy. We adapt the integral operator using supremal convolution in convex analysis to form a novel function majorant used for enforcing robustness. Our method is simple in form and applies to general loss functions and machine learning models. Furthermore, we report experiments with general machine learning models, such as deep neural networks, to demonstrate that ARKS performs competitively with the state-of-the-art methods based on the Wasserstein distance.

[[paper](https://arxiv.org/abs/2102.08474)] [[presentation](https://jj-zhu.github.io/file/oral-arks-aistats-2022.pdf)] [[poster](https://jj-zhu.github.io/file/poster-arks-aistats-2022.pdf)] [[code](https://github.com/christinakouridi/arks)]

---

##### Syntactic language understanding for compositional generalisation in reinforcement learning
***C Kouridi***  
 *Master thesis, University College London, 2020*

Developing artificial agents that act according to natural language instruction remains a key challenge in RL. Existing methods are sample inefficient and generalise poorly to unseen instruction sets and semantic structures. In this thesis, we explore whether a new approach that captures the syntactic structure of instructions can help to overcome these issues. We introduce a babyGIE agent which conditions policy learning on graph-structured representations of instructions, generated through an iterative message passing procedure over a syntactic dependency tree. To test this we evaluate BabyAI’s GRU baselines against our babyGIE algorithm on a set of experiments that extend the BabyAI environment to progressively assess agents’ sample efficiency and capacity for generalisation to unseen colour-shape object pairs and syntactically-varied instructions. Although babyGIE displays greater sample efficiency on simple levels, its performance does not extend to the more complex generalisation tests or tasks. In contrast, the baseline agents appear to exhibit convincing compositional generalisation, but deeper exploration of their behaviour reveals weakness in the face of more diverse linguistic structures. We ultimately argue this exposes underlying limitations in the current set of environments that are tractable to today’s RL agents.

[[thesis](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=QqvI-M4AAAAJ&citation_for_view=QqvI-M4AAAAJ:u5HHmVD_uO8C)] [[code](https://github.com/christinakouridi/babygie)] [[site](https://www.notion.so/Agent-Analysis-678a4693229542868f2d526e132df4cd)]