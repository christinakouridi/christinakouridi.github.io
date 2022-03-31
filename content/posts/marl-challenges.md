---
title: "A brief summary of challenges in Multi-agent RL"
date: 2020-01-02
draft: false
---

Deep reinforcement learning (DRL) algorithms have shown significant success in recent years, surpassing human performance in domains ranging from Atari, Go and no-limit poker [1]. The resemblance of its underlying mechanics to human learning, promises even greater results in real-life applications.

Given that many real-world problems involve environments with a large number of learning agents, a natural extension to DRL is **Multi-Agent Deep Reinforcement Learning (MDRL).** This field of study is concerned with **developing Deep Reinforcement Learning techniques and algorithms that enable a set of autonomous agents to make successful decisions in a shared environment.** Some of its key advantages include the potential of generalisation by approximating the optimal policy and value function, and the ability to scale to problems with high-dimensional state and action spaces. Learning in multi-agent settings is, however, fundamentally more challenging than the single agent case as the agents need to interact with their environment and potentially with each other at the same time.

Multi-agent learning will be a **key stepping stone towards developing systems that interact with humans and other learning algorithms**, therefore it is important to study the challenges inhibiting its progress. **This would make possible interactive systems that will learn directly from their users and continuously adapt to their preferences.** In this article, I provide **a brief summary of these challenges and pinpoint useful reading resources.** They are based on a comprehensive survey on this topic [2], which also serves as an excellent introduction to MDRL. 

#### Challenges

##### A. The moving target problem  
A key challenge in multi-agent learning is **non-stationarity.** Since all agents are learning at the same time, the world is constantly changing from the perspective of any given agent. **In order to succeed, the agents need to take into account the behaviour of other agents and adapt to the joint behaviour accordingly.** Consequentially, there would be variation in the expected return for a state-action pair of a given agent due to its dependence on the **decision policies of the other agents that are continuously updated in the learning process.** Not taking into account this variation would result in inaccurate estimated returns for state-action pairs and therefore good policies at a given time point would not remain so in the future.

This variation **invalidates the stationarity assumption** for establishing the convergence of many single-agent RL algorithms, like Q-learning and Generalised Policy Iteration algorithms. This assumption is known as the Markov property, according to which the reward and current state of an individual agent depend only on the previous state and action taken. In all but very specific cases [3, 4], if the agent ignores this issue and optimises its own policy (usually referred to as independent learning), **the algorithms will likely fail to converge** [5]; each agent would end up entering an endless cycle of adapting to other agents.

A dedicated survey on non-stationarity[19] outlines ways to model it and state-of-the-art approaches to address it.

##### B. Credit assignment  
Team rewards that are not decomposable among agents, give rise to the credit assignment problem. Since the actions of all agents influence the return, it is **difficult for an individual agent to isolate its individual contribution to the team’s success.** For instance, an agent may have chosen the action that maximises reward in a given state, but the resulting return could be negative if all the other agents took exploratory actions. Thus, the agent may falsely adjust its policy to reduce the probability of selecting that action.

A promising approach to address multi-agent credit assignment is Counterfactual Multi-agent Policy Gradients (COMA) [6]. Using this method, **each agent estimates the impact of their action on the return by comparing the estimated reward with a counterfactual baseline.** This baseline is an estimate of what would have happened on average if the agent chose another action. A central critic is responsible for calculating this baseline, by marginalising out a single agent’s action, while keeping the other agents’ actions fixed.

Other approaches include learning opponent representations [7], and alternative neural network architectures and learning approaches that can decompose rewards for smaller group of agents [8, 9].

##### C. Relative overgeneralisation  
A fascinating aspect of multi-agent algorithms is coevolutionary behaviour. Although it is usually desirable, coevolutionary algorithms often lead to poor or middling performance.

One possible failure mode is relative overgeneralisation. It is defined as “the coevolutionary behaviour that occurs when populations in the system are attracted towards areas of the space in which there are many strategies that perform well relative to the interacting partner(s)” [10]. In other words, **agents gravitate towards a robust but sub-optimal joint policy** due to noise induced by the mutual influence of each agent’s exploration strategy on others’ learning updates. When this occurs, agents not only just hill-climb into peaks, but get actively sucked into them, **even with action selection that involves a high degree of randomness** (e.g.  epsilon-greedy action selection).

**Leniency,** is a key concept introduced to prevent this pathology. With lenient learning, agents initially maintain an optimistic disposition to mitigate the noise from transitions resulting in miscoordination, preventing agents from being drawn towards the sub-optimal local optima in the reward search space [2]. Other (value-based) approaches designed to overcome this issue include distributed Q-learning [11] and hysteretic Q-learning [12].

While relative overgeneralisation can occur in both competitive or cooperative environments, it is more problematic for cooperative algorithms. In addition to the studies referenced above, a lot of literature on this topic can be found in the field of cooperative coevolutionary algorithms e.g. [13, 14]. 

##### D. Partial observability  
If the agents have limited access to information of the states pertaining to their environment, they may learn suboptimal policies, i.e. making suboptimal decisions. **Partial observability results in a range of undesirable behaviours.**  For instance, in cooperative environments, one common phenomenon is the “Lazy agent problem” [15]. Learning fails as one of the agents becomes inactive, because when other agent(s) learn a useful policy, it is discouraged from exploration in order not to affect the return.

This type of problem can be modelled using a partially observable Markov decision process (POMDP). [16] includes a good discussion on deep RL models proposed to handle it.

##### E. Transfer learning  
Training a Q-network or generally a deep RL model of one agent usually requires significant computational resources. This problem is exacerbated in MDRL. To reduce the computational demands of training and at the same time improve performance, deep RL models can be pre-trained using transfer learning.

To achieve this, many extensions of DQN have been proposed [16]. However, policy-based or actor-critic methods have not been studied adequately in multi-agent settings [17].

Da Silva and Costa have studied this issue in the context of multi-agent RL in great depth. Their survey [18], provides an elaborate discussion of current lines of research and open questions.

##### F. Global exploration  
Agents trained using RL face a crucial dilemma: taking the action that maximises the immediate reward (exploration) or gathering more information that may prove useful in maximising future rewards (exploitation). This dilemma, known as Exploitation versus Exploration in single-agent settings, has a detrimental effect on performance and has therefore been studied extensively. Like most issues in single-agent RL, it is more challenging to handle in a multi-agent setting.

Most of these problems constitute research directions that are well under way. Nonetheless, **scientific progress in MDRL is not only contingent on theoretical advances but also practical developments.** Practical issues that have long troubled DRL like reproducibility of tests, hyperparameter tuning and the increasing necessity of computational resources, will also need the attention of the multi-agent community.

##### Other great surveys  
- **Agents modelling other agents**.
Albrecht, S. V., and Stone, P. (2018). *Autonomous agents modelling other agents: A comprehensive survey and open problems.* Artificial Intelligence, pages 66–95. Available [here](https://arxiv.org/abs/1709.08071).

- **Game theory**. Nowé, A., Vrancx, P., and Hauwere, Y.M. (2012). *Game theory and multi-agent reinforcement learning, in Reinforcement Learning.* Springer, pp. 441–470. Available [here](https://link.springer.com/chapter/10.1007/978-3-642-27645-3_14).

- **Multi-agent RL**. Busoniu, L. and DeSchutter, B. (2008). *A Comprehensive Survey of Multiagent Reinforcement Learning.* IEEE Transactions on Systems, Man and Cybernetics, Part C (Applications and Reviews) 38 (2) 156–172. Available [here](https://ieeexplore.ieee.org/document/4445757).

- **DRL**. Arulkumaran, K., Deisenroth, M. P., Brundage, and M., Bharath, A. A. (2018). *A Brief Survey of Deep Reinforcement Learning.* Available [here](https://arxiv.org/abs/1708.05866v2).

##### References  
[1] Botvinick, M., Ritter, S., Wang, J. X., Kurth-Nelson, Z., Blundell, C., and Hassabis, D. (2019).  **Reinforcement Learning, Fast and Slow.** Trends in cognitive sciences.

[2] Hernandez-Leal, P., Kartal B., and Taylor, M. E. (2019).  **Survey and Critique of Multiagent Deep Reinforcement Learning.** arXiv:1810.05587.

[3] Arslan, G. and Yuksel, S. (2017). **Decentralized Q-learning for stochastic teams and
games.** IEEE Transactions on Automatic Control, 62 1545–1558.

[4] Yongacoglu, B., Arslan, G. and Yuksel, S. (2019). **Learning team-optimality for decentralized stochastic control and dynamic games.** arXiv:1903.05812.

[5] Zhang, K., Yang, Z., and Baar, T. (2019). **Multi-Agent Reinforcement Learning: A Selective Overview of Theories and Algorithms.** arXiv: 1911.10635.

[6] Foerster, N. J., Farquhar, G., Afouras, T., Nardelli, N., and Whiteson, S. (2017). **Counterfactual Multi-Agent Policy Gradients.** 32nd AAAI Conference on Artificial Intelligence.

[7] Tacchetti, A., Song, F., Mediano, P., Zambaldi, V., Kramar, J., Rabinowitz, N., Graepel, T., Botvinick, M. and Battaglia, P. (2019). **Relational forward models for multi-agent learning. In International Conference on Learning Representations.**

[8] Rashid, T. , Samvelyan, M., de Witt, C. S., Farquhar, G., Foerster, J. N., and Whiteson, S. (2018). **QMIX – Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning.** International Conference on Machine Learning.

[9] Sunehag, P., Lever, G., Gruslys, A., Czarnecki, W.M., Zambaldi, V. F., Jaderberg, M., Lanctot, M., Sonnerat, N., Leibo, J.Z., Tuyls, K., and Graepel. T. (2018). **Value-Decomposition Networks For Cooperative Multi-Agent Learning Based On Team Reward.** Proceedings of 17th International Conference on Autonomous Agents and Multiagent Systems.

[10] Wiegand, R. P. (2003). **An analysis of cooperative coevolutionary algorithms.** Ph.D. Dissertation. George Mason University Virginia.

[11] Lauer, M., Riedmiller, M. (2000). **An algorithm for distributed reinforcement learning in cooperative multi-agent systems.** In Proceedings of the Seventeenth International Conference on Machine Learning.

[12] Matignon, L., Laurent, G., and LeFortPiat, N. (2012). **Independent reinforcement learners in cooperative Markov games: a survey regarding coordination problems.** Knowledge Engineering Review 27 (1) 1–31.

[13] Panait, L. (2006). **The Analysis and Design of Concurrent Learning Algorithms for Cooperative Multiagent Systems.** PhD thesis, George Mason University, Fairfax, Virginia.

[14] Panait, L., Luke, S. and Wiegand, P. (2006). **Biasing coevolutionary search for optimal multiagent behaviors.** IEEE Transactions on Evolutionary Computation. 10(6):629–645.

[15] Sunehag, P., Lever, G., Gruslys, A., Czarnecki, W. M., Zambaldi, V., Jaderberg, M., Lanctot, M., Sonnerat, N., Leibo, J. Z., Tuyls, K., and Graepel, T. (2017). **Value-Decomposition Networks For Cooperative Multi-Agent Learning.** arXiv:1706.05296

[16] Egorov, M. (2016). **Multi-agent deep reinforcement learning.** Stanford University.

[17] Nguyen, T., Nguyen, N, and Nahavandi, S. (2019). **Deep Reinforcement Learning for Multi-Agent Systems: A Review of Challenges, Solutions and Applications.** arXiv: 1812.11794.

[18] Da Silva, F. L., and Costa, A. H. R. (2019). **A Survey on Transfer Learning for Multiagent Reinforcement Learning Systems**. Available [here](https://dl.acm.org/doi/10.1613/jair.1.11396).

[19] Hernandez-Leal, P., Kaisers, M., Baarslag, T., and de Cote, E. M. (2017). **A Survey of Learning in Multiagent Environments: Dealing with Non-Stationarity**. Available [here](https://arxiv.org/pdf/1707.09183.pdf).

