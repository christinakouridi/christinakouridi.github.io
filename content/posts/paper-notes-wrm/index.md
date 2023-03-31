---
title: "'Certifying Some Distributional Robustness with Principled Adversarial Training'"
date: 2020-10-27
draft: false
math: true
---

In this post I will provide a brief overview of the paper **[“Certifying Some Distributional Robustness with Principled Adversarial Training”](https://arxiv.org/pdf/1710.10571.pdf)**. It assumes good knowledge of [stochastic optimisation](https://www.youtube.com/watch?v=0MeNygohD6c) and [adversarial robustness](https://adversarial-ml-tutorial.org/). This work is a positive step towards training neural networks that are robust to small perturbations of their inputs, which may stem from adversarial attacks.

A PyTorch implementation of the main algorithm can be found in [my GitHub repo](https://github.com/christinakouridi/arks/blob/master/src/main/methods.py#L166).

##### Contributions

This work makes two key contributions:

1. **Proposes an adversarial procedure to train distributionally robust neural network models**, which is otherwise intractable. This involves augmenting parameter updates with worst-case adversarial distributions within a certain [Wasserstein distance](https://en.wikipedia.org/wiki/Wasserstein_metric) from a nominal distribution created from the training data
2. **For smooth losses and medium levels of required robustness, the training procedure has theoretical guarantees on its computational and statistical performance**; for higher adversarial robustness, it can be used as a heuristic approach. Its major advantages are then simplicity and applicability across many models and machine learning paradigms (e.g. supervised and reinforcement learning).

##### Adversarial training procedure

In stochastic optimisation, the aim is to minimise an expected loss: $\mathbf{E}_{P_0} \left[ \ell(\theta; Z) \right]$ over a parameter $\theta \in \Theta$, for a training distirbution $Z \sim P_0$. In distributionally robust optimisation, the aim is to minimise the following:

$\begin{aligned}
(1) \hspace{0.8cm} \underset{\theta \in \Theta}{\text{minimise }} \hspace{0.1cm} \underset{P \in \mathcal{P}}{\text{sup }} \hspace{0.1cm} \mathbf{E}_{P}[\ell(\theta ; Z)]
\end{aligned}$

where $\mathcal{P}$ is a postulated class of distributions around the data-generating distribution $P_0$. $\mathcal{P}$ influences robustness guarantees and computability. **This work considers the robustness region defined by Wassertedin-based uncertainty sets** $\mathcal{P} = \\{ P: W_c(P, P_0) \leq \rho \\}$, where $\rho$ defines the neighborhood around $P_0$, and $c$ is the “cost” for an adversary to perturbe $z_0$ to $z$ (the authors typically use $c(z, z_0) = || z - z_0||_{p}^{2}$ with $p \geq 1$ and set $p=2$ in their experiments).

**As solving the min-max problem (1) is analytically intractable for deep learning and other complex models for arbitrary $\rho$, the authors reformulate it using a Lagrangian relaxation with a fixed penalty paramter $\gamma \geq 0$**:

$\begin{aligned}
(2a) \hspace{0.8cm} \underset{\theta \in \Theta}{\text{minimise }}  \\{\{ F(\theta):= \underset{P \in \mathcal{P}}{\text{sup }} \{ \mathbf{E}_{P}[\ell(\theta ; Z)]-\gamma W\_{c}\left(P, P\_{0}\right)\}=\mathbf{E}\_{P}[\phi\_{\gamma}(\theta ; Z)]\} \\}
\end{aligned}$

$\begin{aligned}
(2b) \hspace{0.8cm}  \phi_{\gamma}\left(\theta ; z_{0}\right):=\underset{z \in \mathcal{Z}}{\text{sup}} \hspace{0.3cm} {\ell(\theta;z) - \gamma c(z,z_0)}
\end{aligned}$

the usual loss $\ell$ has been replaced by the robust surrogate $\phi_\gamma$, which allows for adversarial perturbations z, modulated by the penalty $\gamma$. As $P_0$ is unknown, the penalty problem (2) is solved with the empirical distribution $\hat{P}_n$:

$\begin{aligned}
(3) \hspace{0.8cm} \underset{\theta \in \Theta}{\text{minimise }} \\{ \{ F\_{n}(\theta):= \underset{P \in \mathcal{P}}{\text{sup }} \{ \mathbf{E}\_{P}[\ell(\theta ; Z)]-\gamma W\_{c}(P, \hat{P}\_{n})\}=\mathbf{E}\_{\hat{P}\_{n}}[\phi_{\gamma}(\theta ; Z)]\} \\}
\end{aligned}$

As we will discuss later on, **the reformulated objective ensures that moderate levels of robustness against adversarial perturbations are achievable at no computational or statistical cost for smooth losses $\ell$**. This utilises the key insight that **for large enough penalty $\gamma$ (by duality, small enough robustness $\rho$), the robust surrogate function $\phi_\gamma = \ell(\theta;z) - \gamma c(z, z_0)$ in (2b) is strongly concave and hence easy to optimise if $\ell(\theta;z)$ is smooth in $z$**. This implies that stochastic gradient methods applied to (2) have similar convergence guarantees as for non-robust methods.

By inspection, we can obtain that for **large $\gamma$, the term $-\gamma c(z, z_0)$ dominates. If $c(z, z_0)$ is designed to be strongly convex, then $-c(z, z_0)$ and $\phi_{\gamma}$ would be strongly concave**. More formally, this key insight relies on the assumptions that the cost $c$ is *1-strong concave* and that the loss $\ell(\theta; \dot)$ is smooth such that there is some $L$ for which $\nabla_{z} \ell(\theta; \dot)$ is *L-Lipschitz*. The former gives a bound for *c*, and the latter along with a taylor series expansion of $\ell(\theta; z')$ around $z’=z$, a bound for $L$. Combining them results in:


$\begin{aligned}
 (4) \hspace{0.8cm} \ell\left(\theta ; z^{\prime}\right)-\gamma c\left(z^{\prime}, z_{0}\right) \leq \\ \ell(\theta ; z)-\gamma c\left(z, z_{0}\right)+\left\langle\nabla_{z}\left(\ell(\theta ; z)-\gamma c\left(z, z_{0}\right)\right), z^{\prime}-z\right\rangle+\frac{L-\gamma}{2}\left|z-z^{\prime}\right|_{2}^{2}
\end{aligned}$

The last term makes use of the property $| \nabla_z \ell | \leq L$, and that $c$ is twice differentiable. For $\gamma \geq L$ i.e. (negative $L - \gamma$), (4) gives us the first-order condition for ($\gamma  - L$)-strong concativity of $z \rightarrow (\ell(\theta; \dot) - \gamma c(z, z_0))$. To reiterate, **when the loss is smooth enough in $z$ and the penalty $\gamma$ is large enough (corresponding to less robustness), computing the surrogate (2b) is a strongly-concave optimisation problem**.

##### Computational guarantees

**Formulation (4) relaxes the requirement for a prescribed amount of robustness $\rho$, and instead focuses on the Lagrangian penalty formulation (3)**. The authors develop a stochastic gradient descent (SGD) procedure to optimise it, motivated by the observation:

$\begin{aligned}
(5a) \hspace{0.8cm} \nabla_{\theta} \phi_{\gamma}\left(\theta ; z_{0}\right)=\nabla_{\theta} \ell\left(\theta ; z^{\star}\left(z_{0}, \theta\right)\right) \hspace{0.2cm}
\end{aligned}$

$\begin{aligned}
(5b) \hspace{0.8cm} z^{\star}\left(z_{0}, \theta\right)=\underset{z \in \mathcal{Z}}{\text{argmax }} \\{\ell(\theta ; z)-\gamma c\left(z, z_{0}\right)\\}
\end{aligned}$

which is met under two assumptions:

- **the cost function $c(z, z_0)$ is continuous and l-strongly convex** (e.g. $|| z - z_0||^{2}_{2}$)
- **the loss** $\ell : \Theta \times Z \rightarrow \mathbb{R}_+$ **satisfies certain Lipschitzian smoothness conditions**

The resulting SGD procedure is demonstrated by Algorithm 1:

{{< figure src="images/blog_wrm_1.png" title="" width="60%">}}

The convergence properties of this algorithm depends on the loss:

- when $\ell$ is convex in $\theta$ and $\gamma$ is large enough (not too much robustness) so that $\phi_\gamma$ is concave for all $(\theta, z_0) \in \theta \times Z$, algorithm 1 is efficiently solvable with convergence rate $\dfrac{1}{\sqrt{T}}$
- when the loss $\ell$ is non-convex in $\theta$, the SGD method can convergence to a stationary point at the same rate as standard smooth non-convex optimisation when $\gamma \geq L_{zz}$ (as shown by theorem 2 in the paper). This theorem also suggests that approximate maximisation of the surrogate objective has limited effects.

**If the loss is not smooth in z, the inner supremum (2b) is NP-hard to compute for non-smooth deep networks**. In practice, distributionally robust optimisation **can easily become tractable for deep learning by replacing ReLUs with sigmoids, ELUs or other smooth activations**.

##### Certificate of robustness and generalisation

*Algorithm 1* provably learns to protect against adversarial perturbations of the form (3) on the training set. The authors also show that such procedures generalize, allowing to prevent attacks on the test set. They are two key results in the corresponding section:

1. an efficiently computable upper bound on the level of robustness for the worst-case population objective $\text{sup}\_{P: W\_{c} ( P, P\_{0} ) \leq \rho} \mathbb{E}\_{P}[\ell(\theta ; Z)]$ for any arbitrary level of robustness $\rho$. This is optimal for $\rho = \hat{\rho}\_n$, the level of robustness achieved for the empirical distribution by solving (3) (this gives parameters $\theta_{WRM})$.

    $\begin{aligned}
    (6) \hspace{0.8cm} \underset{P: W_{c} \left(P, P\_{0}\right) \leq \rho}{\text{sup}} \mathbb{E}\_P [\ell(\theta; Z)] \leq \gamma \rho + \mathbb{E}\_{\hat{P}\_{n}} [\phi\_{\gamma}(\theta ; Z)]+\epsilon_{n}(t)
    \end{aligned}$

2. the adversarial perturbations on the training set generalize: solving the empirical penalty problem (3) guarantees a similar level of robustness as directly solving its population counterpart (2).

##### Bounds on smoothness of neural networks

Since the above guarantees only apply for a loss $x \rightarrow \ell(\theta; (x, y))$ that satisfies $\gamma \geq L_{xx}$, the authors provide conservative upper bounds on the Lipschitz constant $L_{xx}$ of the loss. However, **due to the conservative natural of the bound, choosing $\gamma$ larger than this value — so that the aforementioned theoretical results apply – may not yield to appreciable robustness in practice**.

##### Visualising the benefits of certified robustness

To demonstrate the certified robustness of their WRM approach (short for Wasserstein Risk Minimisation), the authors devise a simple supervise learning task. The underlying model is a small neural network with either all ReLU or ELU activations between layers. It is benchmarked against two common baseline models: ERM (short for Empirical Risk Minimisation) and FGM (Fast Gradient Minimisation).

{{< figure src="images/blog_wrm_2.png" title="" width="70%">}}

Figure 1 shows the classification boundary learnt by each training procedure (separates blue from orange samples). For both activations, **WRM pushes the classification boundaries further outwards than ERM and FGM; intuitively, adversarial examples come from pushing blue points outwards across the boundary**. Additionally, it seems to be less affected by sensitivities in the data than ERM and FGM, **as evident by its more symmetrical shape**. WRM with ELU in particular, yields an axisymmetric classification boundary that hedges against adversarial perturbations in all directions. This demonstrates the certified level of robustness proven in this work.

The authors also demonstrate the certificate of robustness on the worst-case performance for various levels of robustness $\rho$ for the same toy dataset, as well as MNIST:

{{< figure src="images/blog_wrm_3.png" title="" width="70%">}}

<!-- Experimental results can be reproduced using the [official implementation](https://github.com/duchi-lab/certifiable-distributional-robustness) of the paper in TensorFlow. -->

##### Limitations

- **The adversarial training procedure is only tractable for smooth losses** (i.e. the gradient of the loss must not change abruptly). Specifically, for the inner supremum in (3) to be strongly concave, the Lagrangian penalty parameter must satisfy $\gamma \geq L$. $L$ is a problem-dependent smoothness parameter, which is most often unknown and hard to approximate.
- **Convergence for non-convex SGD only applies for small values of robustness $\rho$ and to a limited set of Wasserstein costs**. In practice methods do not outperform baseline models for large adversarial attacks either.
- The upper bound on the level of robustness achieved for the worst-case population objective and generalisation guarantee use a measure of model complexity that can become prohibitively large for neural networks.