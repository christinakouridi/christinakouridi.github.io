---
title: "Reinforcement Learning papers at NeurIPS 2021"
date: 2021-01-02
draft: false
---

Notes on Reinforcement Learning papers at NeurIPS 2021.

##### 1. Automatic Data Augmentation for Generalisation in Reinforcement Learning [[arXiv](https://arxiv.org/abs/2006.12862), [GitHub](https://github.com/rraileanu/auto-drac)]

**TL;DR**  
Proposes a theoretically motivated way of using data augmentation with actor-critic algorithms, and a practical approach for automatically selecting an effective augmentation to improve generalisation in RL.

**Motivation**  
Recent works have shown data augmentation to be an effective technique for improving sample efficiency and generalisation in RL. However, the authors **cast past applications of data augmentation to RL theoretically unsound due to inaccurate importance sampling estimates**. Additionally, **the choice of data augmentation is arbitrary and fixed**, but different tasks have different biases and hence may need different data transformations.

**Contributions**
1. Firstly, the authors propose **data-regularised actor-critic (DrAC) that applies data augmentation to actor-critic algorithms in a theoretically justified way**. This involves **two novel regularisation terms for the policy and value functions, that ensures they are invariant to transformations induced by various augmentations** (i.e. their values given the augmented observation are constrained to be close to their values given the original observation).
2. Secondly, they propose **three methods for automatically selecting an effective augmentation for an RL task**. The best performing method — UCB-DrAC — finds the best augmentation within a fixed set using a variant of the upper confidence bound (UCB) algorithm (illustrated in Figure 1). The second approach — RL2-DrAC — does so using a meta-learning approach. The third method — Meta-DrAC — directly meta-learns the weights of a convolutional network without access to predefined transformations.

{{< img src="images/blog_neurips_1.png" width="88%">}}

**Results**  
The authors exhibit state-of-the-art generalisation performance on the ProcGen benchmark, outperformance of popular RL methods on four DeepMind Control tasks, and some evidence on DrAC learning policies and representations that better capture task invariances.

**Remarks**
1. **The regularisation terms of DrAC** can be readily added to the objective of any actor-critic algorithm with a discrete stochastic policy (e.g. IMPALA, A3C).
2. Although the experiments make use of environments with observations in raw image format, **DrAC can be adapted to other types of input** (e.g. low dimensional symbolic representations) **by defining an appropriate set of augmentations**.

---

##### 2. Replay-Guided Adversarial Environment Design [[arXiv](https://arxiv.org/pdf/2110.02439.pdf)]

**TL;DR**  
Introduces a class of unsupervised environment design for adaptive RL curricula that unifies prior methods (PAIRED and PLR), and extends them to two algorithms with theoretical guarantees on reaching an equilibrium policy that optimally trades off regret across training levels.

**Motivation**  
**Training deep RL agents on sufficiently diverse and informative variations of environments (termed levels) can improve the generalisability and robustness of the learnt policies to unseen levels.** Therefore, RL agents can benefit from formalised methods that automatically adapt the distribution over environment variations throughout training based on the agents’ learning.

**Background** 
1. This work follows the authors’ prior paper **“Prioritised Level Replay” (PLR)**[[arXiv](https://arxiv.org/pdf/2010.03934.pdf), [GitHub](https://github.com/facebookresearch/level-replay)]. It first **introduces PLR as a practical method to induce adaptive curricula** that improve the sample efficiency and generalisation of RL policies in environments with many tasks / levels. **PLR selectively samples randomly generated training levels weighted by a function of recent temporal-difference (TD) errors experienced on each level (effectively L1 value loss)**. This follow-up work, **extends PLR with theoretical guarantees by partly replacing the L1 value loss prioritisation with a regret prioritisation, as L1 value loss can bias the long-term training behaviour towards high-variance policies**.
2. **Protagonist Antagonist Induced Regret Environment Design (PAIRED)** [[arXiv](https://arxiv.org/pdf/2012.02096.pdf)] is another prominent work that demonstrated the need for adaptive environment curriculum learning. Whereas **PLR prioritises past levels based on their estimated learning potential if replayed, PAIRED actively generates new levels assuming control of a parameterised environment generator**.
3. **PAIRED belongs to a self-supervised RL paradigm called unsupervised environment design (UED)**. In UED, **an environment generator (a teacher) is co-evolved with a student policy that trains on levels actively proposed by the teacher, inducing dynamic curriculum learning**. PAIRED enjoys a useful robustness characterisation of the final student policy in the form of a minimax regret guarantee (optimally trades off regret across training levels), assuming that its underlying teacher-student multi-agent system reaches a Nash equilibrium (NE).

**Contributions**  
1. The authors **introduce a class of UED methods called dual curriculum design (DCD)**. In DCD, a student policy is challenged by a team of two co-evolving teachers (illustrated in Figure 2). One teacher generates new challenging levels dynamically, while the other prioritises existing levels for replay. The authors show that PAIRED and PLR are members of the DCD class, and that all DCD algorithms enjoy certain minimax regret guarantees.

{{< img src="images/blog_neurips_2.png" width="70%">}}

2. They then extend PLR to two algorithms with similar robustness guarantees to that of PAIRED: 
    a) **robust PLR that learns only on trajectories sampled by the PLR teacher** (other teacher is a random level generator)
    b) **Replay-enhanced PAIRED (REPAIRED) which extends PLR to make use of PAIRED as a level generator** instead of the random teacher

**Results**  
- Interestingly, stopping the agent from updating its policy on uncurated levels (i.e. training on less data), improves convergence to NE.
- Robust PLR and REPAIRED outperform the sample-efficiency and zero-shot generalisation of alternative UED methods across a maze domain and a novel car-racing domain.

---

##### 3. On the Expressivity of Markov Reward [[arXiv](https://arxiv.org/pdf/2111.00876.pdf)]

**TL;DR**  
The paper studies the ability of Markov reward to represent tasks defined as a set of acceptable policies, partial ordering over policies, and partial ordering over trajectories.

**Motivation**  
RL is framed by the expressivity of reward as an optimisation signal, also referred to as the reward hypothesis: *“…all of what we mean by goals and purposes can be well thought of as maximisation of the expected value of the cumulative sum of a received scalar signal (reward)”*. Despite being a backdrop assumption, **it lacks grounded theory and formalisation of situations in which Markov rewards are sufficient to express certain tasks**.

**Contribution**
1. The authors focus on three possible notions of task:
    - a set of acceptable policies (SOAP)
    - partial ordering (PO) over policies
    - partial ordering over trajectories (TO)
They show that **for these task notions, there exist task instances that can’t be distinguished or expressed by a single Markov reward function** (Markov in the environment state). **This is often misinterpreted as some tasks not being expressible by a reward function**. A simple example is the task *“always move the same direction”* in a grid world. The SOAP *{“always move right”, “always move left”, ”always move down”, “always move up”}* conveys this task, but no single Markov reward function can make these policies strictly higher in value than all others.
2. They propose polynomial-time algorithms to construct a Markov reward function that allows an agent to optimise tasks of each of these three types, and determine when no such reward function exists.

**Remarks.** An interesting thought is how these theorems would apply to POMDPs, as tasks could be made infinitely expressible by extending state to include history.

---

##### 4. Deep Reinforcement Learning at the Edge of the Statistical Precipice [[arXiv](https://arxiv.org/pdf/2108.13264.pdf), [GitHub](https://github.com/google-research/rliable)]

**TL;DR**  
Statistics protocols for evaluation deep RL algorithms on a suite of tasks, which minimise statistical uncertainty of results consisting of a handful of runs per task.

**Motivation**  
Evaluating deep RL algorithms on research benchmarks **with multiple tasks** requires significant training time. **Most published works compare point estimates of aggregate performance such as mean and median scores across tasks, ignoring the statistical uncertainty arising from the use of a finite number of training runs**. This can result in invalid conclusions, and harder reproducibility.

**Contribution**  
The authors provide practical recommendations for evaluating performance, which treat performance estimates based on a finite number of runs as a random variable:

1. **For measuring uncertainty in aggregate performance, use interval estimates** via stratified bootstrap confidence intervals, as opposed to point estimates
2. **For measuring variability in performance across tasks and runs**, use performance profiles that are robust to outlier runs / tasks
3. For computing an aggregate metric of overall performance, use robust statistics such as:
    - **interquartile mean**: the mean score of the middle 50% of the runs combined across all tasks
    - **optimality gap**: the amount by which the algorithm fails to meet a chosen target minimum score
    - **average probability of improvement**: how likely it is for an algorithm X to outperform algorithm Y on a randomly selected task

Common metrics can be problematic: **mean is easily dominated by performance on a few outlier tasks, while median has high variability and is unaffected by zero scores on nearly half of the tasks**.

{{< img src="images/blog_neurips_4.png" title="Figure 3. Visual representation of recommended statistical evaluation tools" width="100%">}}


**Results**  
- The authors scrutinise performance evaluations of existing algorithms on common benchmarks such as ALE, ProcGen, and the DeepMind Control Suite, **revealing discrepancies in prior comparisons**. 
- Through various case studies, they also demonstrate the deficiencies of commonly reported performance metrics, such as point estimates of median scores on Atari 100k.

**Remarks**  
This is **an important work for supporting reliable evaluation in RL research**. For rigorous statistical comparisons of single-task RL algorithms, check the paper *“A Hitchhiker’s Guide to Statistical Comparisons of Reinforcement Learning algorithms”* [[arXiv](https://arxiv.org/pdf/1904.06979.pdf), [GitHub](https://github.com/ccolas/rl_stats)].

---

##### 5. Why Generalisation in RL is Difficult: Epistemic POMDPs and Implicit Partial Observability [[arXiv](https://arxiv.org/abs/2107.06277)]

**TL;DR**  
Trying to generalise induces a POMDP, even if the problem is an MDP, arising from epistemic uncertainty about the true MDP at test-time. The authors formalise this as “epistemic POMDP”, call for new approaches to solve the generalisation problem, and propose an ensemble-based algorithm to approximately solve it.

**Motivation**  
It has often been observed empirically that generalisation to new test-time contexts is a significant challenge for deep RL agents. Nonetheless, it has remained an open question whether the RL setting itself poses additional challenges to generalisation beyond those found in supervised learning.

**Contribution**  
1. In supervised learning, optimising for performance on the training set (i.e. empirical risk minimisation (ERM)) can result in good generalisation performance in the absence of distribution shifts and with appropriate inductive biases. **The authors show that such ERM approach can be sub-optimal for generalising to new test-time contexts in RL, even when these new contexts are drawn from the training distribution**.
2. **Using a Bayesian RL perspective, they reframe generalisation as the problem of solving a partially observable MDP, which they call “epistemic PODMP”**. The epistemic POMDP highlights that generalising in RL is more difficult than supervised learning due to partial observability induced by epistemic uncertainty. When the agent’s posterior distribution over environments can be calculated, **constructing the epistemic POMDP and running a POMDP-solving algorithm on it will yield a Bayes-optimal policy for maximising test-time performance**.
3. They propose one such algorithm, called LEEP — it learns different policies for sampled environments regularised to make the policies similar, and then combines them into a single policy that is close to optimal.

**Results**  
Policies learned through approximations of the epistemic POMDP obtain better test-time performance on the Procgen benchmark than those learned by standard RL algorithms (shown in Figure 4).

{{< img src="images/blog_neurips_3.png" width="65%">}}

**Remarks**  
**Although LEEP only seems able to optimise a crude approximation to the epistemic POMDP**, it opens avenue for developing algorithms that can handle epistemic uncertainty, and thereby generalise to new contexts better.