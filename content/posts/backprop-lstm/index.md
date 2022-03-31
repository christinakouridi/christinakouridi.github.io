---
title: "Deriving backpropagation equations for an LSTM"
date: 2019-06-19
draft: false
math: true
---

In this post I will derive the backpropagation equations for a LSTM cell in vectorised form. It assumes basic knowledge of LSTMs and backpropagation, which you can refresh at [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) and [A Quick Introduction to Backpropagation](http://arunmallya.github.io/writeups/nn/backprop.html).

#### Derivations  
##### Forward propagation
We will firstly remind ouselves of the forward propagation equations. The nomenclature followed is demonstrated in Figure 1. All equations correspond to one time step.

{{< img src="images/blog_bplstm_1.png" title="Figure 1: Architecture of a LSTM memory cell at timestep t" width="75%">}}

$\begin{aligned}
&h\_{t-1} \in  \mathbb{R}^{n\_{h}}, & \mspace{31mu} x\_{t} \in  \mathbb{R}^{n\_{x}} \\\
&z\_{t}= [h\_{t-1}, x\_{t}] \\\
\end{aligned}$

$\begin{aligned}
&a\_{f}= W\_{f}\cdot z\_{t} + b\_{f},& \mspace{31mu}  f\_{t}= \sigma(a\_{f}) \\\
&a\_{i}= W\_{i}\cdot z\_{t} + b\_{i},& \mspace{40mu}  i\_{t}= \sigma(a\_{i}) \\\
&a\_{o}= W\_{o}\cdot z\_{t} + b\_{o},& \mspace{34mu}  o\_{t}= \sigma(a\_{o})  \\\
&a\_{c}= W\_{c}\cdot z\_{t} + b\_{c},& \mspace{36mu}  \hat{c}\_t=  tanh(a\_{c}) \\\
\end{aligned}$

$\begin{aligned}
&{c}\_t=  i\_{t}\odot \hat{c}\_t + f\_{t}\odot c\_{t-1} \\\
&{h}\_t=  o\_{t}\odot tanh(c\_{t}) \\\
\end{aligned}$

$\begin{aligned}
&v\_{t}= W\_{v}\cdot h\_{t} + b\_{v} \\\
&\hat{y}\_t= softmax(v\_{t})
\end{aligned}$

##### Backward propagation

Backpropagation through a LSTM is not as straightforward as through other common Deep Learning architectures, due to the special way its underlying layers interact. Nonetheless, the approach is largely the same; identifying dependencies and recursively applying the chain rule.

{{< img src="images/blog_bplstm_2.png" title="Figure 2: Backpropagation through a LSTM memory cell" width="80%">}}

Cross-entropy loss with a softmax function are used at the output layer. The standard definition of the derivative of the cross-entropy loss ($\frac{\partial J}{\partial v\_{t}}$) is used directly; a detailed derivation can be found here.

##### Output
$\begin{aligned}
&\frac{\partial J}{\partial v\_{t}} = \hat{y}\_t - y\_{t} \\\
&\frac{\partial J}{\partial W\_{v}} = \frac{\partial J}{\partial v\_{t}} \cdot \frac{\partial v\_{t}}{\partial W\_{v}} \Rightarrow \frac{\partial J}{\partial W\_{v}} = \frac{\partial J}{\partial v\_{t}} \cdot h\_{t}^T \\\
&\frac{\partial J}{\partial b\_{v}} = \frac{\partial J}{\partial v\_{t}} \cdot \frac{\partial v\_{t}}{\partial b\_{v}} \Rightarrow \frac{\partial J}{\partial b\_{v}} = \frac{\partial J}{\partial v\_{t}} \end{aligned}$

##### Hidden state
$\begin{aligned}
&\frac{\partial J}{\partial h\_{t}} = \frac{\partial J}{\partial v\_{t}} \cdot \frac{\partial v\_{t}}{\partial h\_{t}} \Rightarrow \frac{\partial J}{\partial h\_{t}} = W\_{v}^T \cdot \frac{\partial J}{\partial v\_{t}} \\\
&\frac{\partial J}{\partial h\_{t}} += \frac{\partial J}{\partial h\_{next}}
\end{aligned}$

##### Output gate
$\begin{aligned}
&\frac{\partial J}{\partial o\_{t}} = \frac{\partial J}{\partial h\_{t}} \cdot \frac{\partial h\_{t}}{\partial o\_{t}} \Rightarrow \frac{\partial J}{\partial o\_{t}} = \frac{\partial J}{\partial h\_{t}} \odot tanh(c\_{t}) \\\
\end{aligned}$

$\begin{aligned}
&\frac{\partial J}{\partial a\_{o}} = \frac{\partial J}{\partial o\_{t}} \cdot \frac{\partial o\_{t}}{\partial a\_{o}} \Rightarrow \frac{\partial J}{\partial a\_{o}} = \frac{\partial J}{\partial h\_{t}} \odot tanh(c\_{t}) \odot \frac{d(\sigma (a\_{o}))}{da\_{o}}  \\\ 
&\Rightarrow \frac{\partial J}{\partial a\_{o}} = \frac{\partial J}{\partial h\_{t}} \odot tanh(c\_{t}) \odot \sigma (a\_{o})(1- \sigma (a\_{o}))  \\\ 
&\Rightarrow \frac{\partial J}{\partial a\_{o}} = \frac{\partial J}{\partial h\_{t}} \odot tanh(c\_{t}) \odot o\_{t}(1- o\_{t}) \\\
\end{aligned}$

$\begin{aligned}
&\frac{\partial J}{\partial W\_{o}} = \frac{\partial J}{\partial a\_{o}} \cdot \frac{\partial a\_{o}}{\partial W\_{o}}  \Rightarrow \frac{\partial J}{\partial W\_{o}} = \frac{\partial J}{\partial a\_{o}} \cdot z\_{t}^T \\\
&\frac{\partial J}{\partial b\_{o}} = \frac{\partial J}{\partial a\_{o}} \cdot \frac{\partial a\_{o}}{\partial b\_{o}} \Rightarrow \frac{\partial J}{\partial b\_{o}} = \frac{\partial J}{\partial a\_{o}}
\end{aligned}$

##### Cell state
$\begin{aligned}
\frac{\partial J}{\partial c\_{t}} = \frac{\partial J}{\partial h\_{t}} \cdot \frac{\partial h\_{t}}{\partial c\_{t}} \Rightarrow \frac{\partial J}{\partial c\_{t}} = \frac{\partial J}{\partial h\_{t}} \odot o\_{t} \odot (1-tanh(c\_{t})^2) \\\
\end{aligned}$

$\begin{aligned}
\frac{\partial J}{\partial c\_{t}} += \frac{\partial J}{\partial c\_{next}} \\\
\end{aligned}$

$\begin{aligned}
&\frac{\partial J}{\partial \hat{c}\_t} = \frac{\partial J}{\partial c\_{t}} \cdot \frac{\partial c\_{t}}{\partial \hat{c}\_t} \Rightarrow \frac{\partial J}{\partial \hat{c}\_t} = \frac{\partial J}{\partial c\_{t}} \odot i\_{t} \\\
\end{aligned}$

$\begin{aligned}
&\frac{\partial J}{\partial a\_{c}} = \frac{\partial J}{\partial \hat{c}\_t} \cdot \frac{\partial \hat{c}\_t}{\partial a\_{c}} \Rightarrow \frac{\partial J}{\partial a\_{c}} = \frac{\partial J}{\partial c\_{t}} \odot i\_{t} \odot \frac{d(tanh(a\_{c}))}{da\_{c}} \\\
&\Rightarrow \frac{\partial J}{\partial a\_{c}} = \frac{\partial J}{\partial c\_{t}} \odot i\_{t} \odot (1 - tanh(a\_{c})^2) \\\ 
&\Rightarrow \frac{\partial J}{\partial a\_{c}} = \frac{\partial J}{\partial c\_{t}} \odot i\_{t} \odot (1 - \hat{c}\_t^2) \\\
\end{aligned}$

$\begin{aligned}
&\frac{\partial J}{\partial W\_{c}} = \frac{\partial J}{\partial a\_{c}} \cdot \frac{\partial a\_{c}}{\partial W\_{c}} \Rightarrow \frac{\partial J}{\partial W\_{c}} = \frac{\partial J}{\partial a\_{c}} \cdot z\_{t}^T \\\
\end{aligned}$

$\begin{aligned}
&\frac{\partial J}{\partial b\_{c}} = \frac{\partial J}{\partial a\_{c}} \cdot \frac{\partial a\_{c}}{\partial b\_{c}} \Rightarrow \frac{\partial J}{\partial b\_{c}} = \frac{\partial J}{\partial a\_{c}}
\end{aligned}$

##### Input gate
$\begin{aligned}
&\frac{\partial J}{\partial i\_{t}} = \frac{\partial J}{\partial c\_{t}} \cdot \frac{\partial c\_{t}}{\partial i\_{t}} \Rightarrow \frac{\partial J}{\partial i\_{t}} = \frac{\partial J}{\partial c\_{t}} \odot \hat{c}\_t \\\
\end{aligned}$

$\begin{aligned}
&\frac{\partial J}{\partial a\_{i}} = \frac{\partial J}{\partial i\_{t}} \cdot \frac{\partial i\_{t}}{\partial a\_{i}} \Rightarrow \frac{\partial J}{\partial a\_{i}} = \frac{\partial J}{\partial c\_{t}} \odot \hat{c}\_t \odot \frac{d(\sigma (a\_{i}))}{da\_{i}} \\\
&\Rightarrow \frac{\partial J}{\partial a\_{i}} = \frac{\partial J}{\partial c\_{t}} \odot \hat{c}\_t \odot \sigma (a\_{i})(1- \sigma (a\_{i})) \\\ 
&\Rightarrow \frac{\partial J}{\partial a\_{i}} = \frac{\partial J}{\partial c\_{t}} \odot \hat{c}\_t \odot i\_{t}(1- i\_{t}) \\\
\end{aligned}$

$\begin{aligned}
&\frac{\partial J}{\partial W\_{i}} = \frac{\partial J}{\partial a\_{i}} \cdot \frac{\partial a\_{i}}{\partial W\_{i}}  \Rightarrow \frac{\partial J}{\partial W\_{i}} = \frac{\partial J}{\partial a\_{i}} \cdot z\_{t}^T \\\
\end{aligned}$

$\begin{aligned}
\frac{\partial J}{\partial b\_{i}} = \frac{\partial J}{\partial a\_{i}} \cdot \frac{\partial a\_{i}}{\partial b\_{i}} \Rightarrow \frac{\partial J}{\partial b\_{i}} = \frac{\partial J}{\partial a\_{i}}
\end{aligned}$

##### Forget gate
$\begin{aligned}
\frac{\partial J}{\partial f\_{t}} = \frac{\partial J}{\partial c\_{t}} \cdot \frac{\partial c\_{t}}{\partial f\_{t}} \Rightarrow \frac{\partial J}{\partial f\_{t}} = \frac{\partial J}{\partial c\_{t}} \odot c\_{t-1} \\\
\end{aligned}$

$\begin{aligned}
&\frac{\partial J}{\partial a\_{f}} = \frac{\partial J}{\partial f\_{t}} \cdot \frac{\partial f\_{t}}{\partial a\_{f}} \Rightarrow \frac{\partial J}{\partial a\_{f}} = \frac{\partial J}{\partial c\_{t}} \odot c\_{t-1} \odot \frac{d(\sigma (a\_{f}))}{da\_{f}} \\\ 
&\Rightarrow \frac{\partial J}{\partial a\_{f}} = \frac{\partial J}{\partial c\_{t}} \odot c\_{t-1} \odot \sigma (a\_{f})(1- \sigma (a\_{f}) \\\ 
&\Rightarrow \frac{\partial J}{\partial a\_{f}} = \frac{\partial J}{\partial c\_{t}} \odot c\_{t-1} \odot f\_{t}(1- f\_{t}) \\\
\end{aligned}$

$\begin{aligned}
&\frac{\partial J}{\partial W\_{f}} = \frac{\partial J}{\partial a\_{f}} \cdot \frac{\partial a\_{f}}{\partial W\_{f}}  \Rightarrow \frac{\partial J}{\partial W\_{f}} = \frac{\partial J}{\partial a\_{f}} \cdot z\_{t}^T \\\
&\frac{\partial J}{\partial b\_{f}} = \frac{\partial J}{\partial a\_{f}} \cdot \frac{\partial a\_{f}}{\partial b\_{f}} \Rightarrow \frac{\partial J}{\partial b\_{f}} = \frac{\partial J}{\partial a\_{f}}
\end{aligned}$

##### Input
$\begin{aligned}
&\frac{\partial J}{\partial z\_{t}} = \frac{\partial J}{\partial a\_{f}} \cdot \frac{\partial a\_{f}}{\partial z\_{t}} + \frac{\partial J}{\partial a\_{i}} \cdot \frac{\partial a\_{i}}{\partial z\_{t}} + \frac{\partial J}{\partial a\_{o}} \cdot \frac{\partial a\_{o}}{\partial z\_{t}} + \frac{\partial J}{\partial a\_{c}} \cdot \frac{\partial a\_{c}}{\partial z\_{t}}  \\\ 
&\Rightarrow \frac{\partial J}{\partial z\_{t}} =  W\_{f}^T \cdot \frac{\partial J}{\partial a\_{f}} +W\_{i}^T \cdot \frac{\partial J}{\partial a\_{i}} + W\_{o}^T \cdot \frac{\partial J}{\partial a\_{o}} + W\_{c}^T \cdot \frac{\partial J}{\partial a\_{c}} \\\
\end{aligned}$

$\begin{aligned}
&\frac{\partial J}{\partial h\_{t-1}} = \frac{\partial J}{\partial z\_{t}}[:n\_{h}, :] \\\
&\frac{\partial J}{\partial c\_{t-1}} = \frac{\partial J}{\partial c\_{t}} \cdot \frac{\partial c\_{t}}{\partial c\_{t-1}} \Rightarrow \frac{\partial J}{\partial c\_{t-1}} = \frac{\partial J}{\partial c\_{t}} \odot f\_{t}
\end{aligned}$

The above equations for forward propagation and back propagation will be calculated T times (number of time steps) in each training iteration. At the end of each training iteration, the weights will be updated using the accumulated cost gradient with respect to each weight for all time steps. Assuming Stochastic Gradient Descent, the update equations are the following:

$\begin{aligned}
&\frac{\partial J}{\partial W\_{f}} = \sum\limits\_{t}^T \frac{\partial J}{\partial W\_{f}^t}, \mspace{31mu} W\_{f} += \alpha * \frac{\partial J}{\partial W\_{f}} \\\
&\frac{\partial J}{\partial W\_{i}} = \sum\limits\_{t}^T \frac{\partial J}{\partial W\_{i}^t}, \mspace{31mu} W\_{i} += \alpha * \frac{\partial J}{\partial W\_{i}} \\\
&\frac{\partial J}{\partial W\_{o}} = \sum\limits\_{t}^T \frac{\partial J}{\partial W\_{o}^t}, \mspace{31mu} W\_{o} += \alpha * \frac{\partial J}{\partial W\_{o}} \\\
&\frac{\partial J}{\partial W\_{c}} = \sum\limits\_{t}^T \frac{\partial J}{\partial W\_{c}^t}, \mspace{31mu} W\_{c} += \alpha * \frac{\partial J}{\partial W\_{c}} \\\
&\frac{\partial J}{\partial W\_{v}} = \sum\limits\_{t}^T \frac{\partial J}{\partial W\_{v}^t}, \mspace{31mu} W\_{v} += \alpha * \frac{\partial J}{\partial W\_{v}} \\\
\end{aligned}$


In the [next post](/posts/implement-lstm), we will implement the above equations using Numpy and train the resulting LSTM model on real data.