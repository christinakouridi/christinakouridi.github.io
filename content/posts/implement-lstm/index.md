---
title: "Implementing a LSTM from scratch with Numpy"
date: 2019-06-20
draft: false
math: true
---

In this post, we will implement a simple character-level LSTM using Numpy. It is trained in batches with the Adam optimiser and learns basic words after just a few training iterations.

The full code is available on this [GitHub repo](https://github.com/christinakouridi/scratchML/tree/master/LSTM).

{{< figure src="images/blog_implementlstm_1.png" title="Figure 1: Architecture of a LSTM memory cell" width="75%">}}


##### Imports

```python
import numpy as np
import matplotlib.pyplot as plt
```

##### Data preparation
Our dataset is J.K. Rowling’s Harry Potter and the Philosopher’s Stone. I chose this text as the characteristic context and semantic structures present in the abundant dialogue, will help with evaluating the quality of results (also a huge HP fan!).

```python
data = open('HP1.txt').read().lower()
```

We will compute the size of our vocabulary and mapping dictionaries from characters to indices and vice-versa, which will be used for transforming the input data to an appropriate format later on.

```python
chars = set(data)
vocab_size = len(chars)
print('data has %d characters, %d unique' % (len(data), vocab_size))

char_to_idx = {w: i for i,w in enumerate(chars)}
idx_to_char = {i: w for i,w in enumerate(chars)}
```
*data has 442744 characters, 54 unique*


##### Initialisation of model parameters
We will wrap all functions in the LSTM class.

There is not conclusive evidence on which optimiser performs best for this type of architecture. In this implementation, we will use the Adam optimiser, as [general empirical results](https://arxiv.org/abs/1412.6980) demonstrate that it performs favourably compared to other optimisation methods, it converges fast and can effectively navigate local minima by adapting the learning rate for each parameter. The moving averages, $\beta\_{1}$ and $\beta\_{2}$ are initialised as suggested in the original paper. Here’s [a quick explanation](https://ruder.io/optimizing-gradient-descent/index.html#adam) of how Adam works and how it compares to other methods.

After the learning rate, weight initialisation is the second most important setting for LSTMs and other recurrent networks; improper initialisation could slow down the training process to the point of impracticality. We will therefore use the high-performing Xavier initialisation, which involves randomly sampling weights from a distribution $\mathcal{N}(0, \frac{1}{\sqrt{n}})$, where n is the number of neurons in the preceding layer.

The input to the LSTM, z, has dimensions [vocab_size + $n_h$, 1] . Since the LSTM layer wants to output $n_h$ neurons, each weight should be of size [$n_h$, vocab_size + $n_h$] and each bias of size [$n_h$, 1]. Exception is the weight and bias at the output softmax layer ($W_v$, $b_v$). The resulting output will be a probability distribution over all possible characters in the vocabulary, therefore of size [vocab_size, 1], hence $W_v$ should be of size [vocab_size, $n_h$] and bv of size [$n_h$, 1].

```python
class LSTM:
    def __init__(self, char_to_idx, idx_to_char, vocab_size, n_h=100, seq_len=25, 
                          epochs=10, lr=0.01, beta1=0.9, beta2=0.999):
        self.char_to_idx = char_to_idx # characters to indices mapping
        self.idx_to_char = idx_to_char # indices to characters mapping
        self.vocab_size = vocab_size # no. of unique characters in the training data
        self.n_h = n_h # no. of units in the hidden layer
        self.seq_len = seq_len # no. of time steps, also size of mini batch
        self.epochs = epochs # no. of training iterations
        self.lr = lr # learning rate
        self.beta1 = beta1 # 1st momentum parameter
        self.beta2 = beta2 # 2nd momentum parameter
    
        #-----initialise weights and biases-----#
        self.params = {}
        std = (1.0/np.sqrt(self.vocab_size + self.n_h)) # Xavier initialisation
        
        # forget gate
        self.params["Wf"] = np.random.randn(self.n_h, self.n_h + self.vocab_size) * std
        self.params["bf"] = np.ones((self.n_h,1))

        # input gate
        self.params["Wi"] = np.random.randn(self.n_h, self.n_h + self.vocab_size) * std
        self.params["bi"] = np.zeros((self.n_h,1))

        # cell gate
        self.params["Wc"] = np.random.randn(self.n_h, self.n_h + self.vocab_size) * std
        self.params["bc"] = np.zeros((self.n_h,1))

        # output gate
        self.params["Wo"] = np.random.randn(self.n_h, self.n_h + self.vocab_size) * std
        self.params["bo"] = np.zeros((self.n_h ,1))

        # output
        self.params["Wv"] = np.random.randn(self.vocab_size, self.n_h) * \
                                          (1.0/np.sqrt(self.vocab_size))
        self.params["bv"] = np.zeros((self.vocab_size ,1))

        #-----initialise gradients and Adam parameters-----#
        self.grads = {}
        self.adam_params = {}

        for key in self.params:
            self.grads["d"+key] = np.zeros_like(self.params[key])
            self.adam_params["m"+key] = np.zeros_like(self.params[key])
            self.adam_params["v"+key] = np.zeros_like(self.params[key])
            
        self.smooth_loss = -np.log(1.0 / self.vocab_size) * self.seq_len
    return
```

##### Utility functions
Firstly, we will compute the sigmoid activation used at the forget, input and output gate layers, and the softmax activation used at the output layer. Tanh activation is also needed but *numpy.tanh* is used instead.

```python
def sigmoid(self, x):
    return 1 / (1 + np.exp(-x))

LSTM.sigmoid = sigmoid


def softmax(self, x):
    e_x = np.exp(x - np.max(x)) # max(x) subtracted for numerical stability
    return e_x / np.sum(e_x)

LSTM.softmax = softmax
```

Although exploding gradients is not as prevalent for LSTMs as for RNNs, we will limit the gradients to a conservative value using clip_grads. After backpropagating through all LSTM cells, we will reset the gradients using *reset_grads*.

```python
def clip_grads(self):
    for key in self.grads:
        np.clip(self.grads[key], -5, 5, out=self.grads[key])
    return

LSTM.clip_grads = clip_grads


def reset_grads(self):
    for key in self.grads:
        self.grads[key].fill(0)
    return

LSTM.reset_grads = reset_grads
```

The last utility function that we will create is for updating the weights using Adam. Note that the weights are updated using the accumulated gradients for all time steps.

```python
def update_params(self, batch_num):
    for key in self.params:
        self.adam_params["m"+key] = self.adam_params["m"+key] * self.beta1 + \
                                    (1 - self.beta1) * self.grads["d"+key]
        self.adam_params["v"+key] = self.adam_params["v"+key] * self.beta2 + \
                                    (1 - self.beta2) * self.grads["d"+key]**2

        m_correlated = self.adam_params["m" + key] / (1 - self.beta1**batch_num)
        v_correlated = self.adam_params["v" + key] / (1 - self.beta2**batch_num) 
        self.params[key] -= self.lr * m_correlated / (np.sqrt(v_correlated) + 1e-8) 
    return LSTM.update_params = update_params
```

##### Forward propagation for a time-step
We will propagate forwards through each LSTM cell using forward_step. The mathematical form of the forward and backward propagation equations can be found in my [previous post](/posts/backprop-lstm).

A LSTM cell depends on the previous cell’s state (like Neural Networks). forward_step therefore takes as input the previous hidden state ($h_{prev}$) and previous cell state ($c_{prev}$). At the beginning of every training iteration, the previous hidden states are initialised to zero (i.e. at t = -1), but for subsequent time-steps, they correspond to the hidden states at t-1, where t is the current time-step.

```python
def forward_step(self, x, h_prev, c_prev):
    z = np.row_stack((h_prev, x))

    f = self.sigmoid(np.dot(self.params["Wf"], z) + self.params["bf"])
    i = self.sigmoid(np.dot(self.params["Wi"], z) + self.params["bi"])
    c_bar = np.tanh(np.dot(self.params["Wc"], z) + self.params["bc"])

    c = f * c_prev + i * c_bar
    o = self.sigmoid(np.dot(self.params["Wo"], z) + self.params["bo"])
    h = o * np.tanh(c)

    v = np.dot(self.params["Wv"], h) + self.params["bv"]
    y_hat = self.softmax(v)
    return y_hat, v, h, o, c, c_bar, i, f, z

LSTM.forward_step = forward_step
```

##### Backward propagation for a time-step
After forward propagation, we will pass the updated values of the last LSTM cell to backward_step and propagate the gradients backwards to the first LSTM cell.

$dh_{next}$ and $dc_{next}$ are initialised to zero at t = -1, but take the values of $dh_{prev}$ and $dc_{prev}$ that *backward_step* returns in subsequent time steps.
In addition, it is worth clarifying:

1. As weights are shared by all time steps, the weight gradients are accumulated.
2. We are adding $dh_{next}$ to $dh$, because as Figure 1 shows, $h$ is branched in forward propagation in the softmax output layer and the next LSTM cell, where it is concatenated with $x$. Therefore, there are two gradients flowing back. This applies to dc as well.
3. There are four gradients flowing towards the input layer from the gates, therefore $dz$ is the summation of those gradients.

```python
def backward_step(self, y, y_hat, dh_next, dc_next, c_prev, z, f, i, c_bar, c, o, h):
    dv = np.copy(y_hat)
    dv[y] -= 1 # yhat - y

    self.grads["dWv"] += np.dot(dv, h.T)
    self.grads["dbv"] += dv

    dh = np.dot(self.params["Wv"].T, dv)
    dh += dh_next
    
    do = dh * np.tanh(c)
    da_o = do * o*(1-o)
    self.grads["dWo"] += np.dot(da_o, z.T)
    self.grads["dbo"] += da_o

    dc = dh * o * (1-np.tanh(c)**2)
    dc += dc_next

    dc_bar = dc * i
    da_c = dc_bar * (1-c_bar**2)
    self.grads["dWc"] += np.dot(da_c, z.T)
    self.grads["dbc"] += da_c

    di = dc * c_bar
    da_i = di * i*(1-i) 
    self.grads["dWi"] += np.dot(da_i, z.T)
    self.grads["dbi"] += da_i

    df = dc * c_prev
    da_f = df * f*(1-f)
    self.grads["dWf"] += np.dot(da_f, z.T)
    self.grads["dbf"] += da_f

    dz = (np.dot(self.params["Wf"].T, da_f)
         + np.dot(self.params["Wi"].T, da_i)
         + np.dot(self.params["Wc"].T, da_c)
         + np.dot(self.params["Wo"].T, da_o))

    dh_prev = dz[:self.n_h, :]
    dc_prev = f * dc
    return dh_prev, dc_prev

LSTM.backward_step = backward_step
```

##### Forward and backward propagation for all time-steps
The forward and backward propagation steps will be executed within the forward_backward function. Here, we iterate over all time steps and store the results for each time step in dictionaries. In the forward propagation loop, we also accumulate the cross entropy loss.

*forward_backward* exports the cross entropy loss of the training batch, in addition to the hidden and cell states of the last layer which are fed to the first LSTM cell as $h_{prev}$ and prev of the next training batch.

```python
def forward_backward(self, x_batch, y_batch, h_prev, c_prev):
    x, z = {}, {}
    f, i, c_bar, c, o = {}, {}, {}, {}, {}
    y_hat, v, h = {}, {}, {}

    # Values at t= - 1
    h[-1] = h_prev
    c[-1] = c_prev

    loss = 0
    for t in range(self.seq_len): 
        x[t] = np.zeros((self.vocab_size, 1))
        x[t][x_batch[t]] = 1

        y_hat[t], v[t], h[t], o[t], c[t], c_bar[t], i[t], f[t], z[t] = \
        self.forward_step(x[t], h[t-1], c[t-1])

        loss += -np.log(y_hat[t][y_batch[t],0])

    self.reset_grads()

    dh_next = np.zeros_like(h[0])
    dc_next = np.zeros_like(c[0])

    for t in reversed(range(self.seq_len)):
        dh_next, dc_next = self.backward_step(y_batch[t], y_hat[t], dh_next, 
                                              dc_next, c[t-1], z[t], f[t], i[t], 
                                              c_bar[t], c[t], o[t], h[t]) 
    return loss, h[self.seq_len-1], c[self.seq_len-1]
```

LSTM.forward_backward = forward_backward
Sampling character sequences
As training progresses, we will use the sample function to output a sequence of characters from the model, of total length *sample_size*.

```python
def sample(self, h_prev, c_prev, sample_size):
    x = np.zeros((self.vocab_size, 1))
    h = h_prev
    c = c_prev
    sample_string = "" 
    
    for t in range(sample_size):
        y_hat, _, h, _, c, _, _, _, _ = self.forward_step(x, h, c)        
        
        # get a random index within the probability distribution of y_hat(ravel())
        idx = np.random.choice(range(self.vocab_size), p=y_hat.ravel())
        x = np.zeros((self.vocab_size, 1))
        x[idx] = 1
        
        #find the char with the sampled index and concat to the output string
        char = self.idx_to_char[idx]
        sample_string += char
    return sample_string

LSTM.sample = sample
```

##### Training
Next, we define the function to train the model. train takes as input a corpus of text (X) and outputs a list of losses for each training batch (J) as well as the trained parameters.

In order to speed up training, we will train our data in batches. The number of batches (*num_batches*) is given by the total number of characters in the input text (*len(X)*) divided by the number of characters that we want to use in each batch (*seq_len*), which is user-defined. The input text goes through the following processing steps:

1. Firstly, we trim the characters at end of the input text that don’t form a full sequence
2. When we iterate over each training batch, we slice the input text in batches of size *seq_len*
3. We map each character in the input (and output) batch to an index, using *idx_to_char*, effectively converting the input batch to a list of integers

I have mentioned earlier that h_prev and c_prev are set to zero at the beginning of every training iteration. This means that the states for the samples of each batch will be resused as initial states for the samples in the next batch. Keras, the high-level neural networks API, refers to this as “making the RNN stateful”. If each training batch is independent, then $h_{prev}$ and $c_{prev}$ should be reset after training each batch.

```python
def train(self, X, verbose=True):
    J = []  # to store losses

    num_batches = len(X) // self.seq_len
    X_trimmed = X[: num_batches * self.seq_len]  # trim input to have full sequences

    for epoch in range(self.epochs):
        h_prev = np.zeros((self.n_h, 1))
        c_prev = np.zeros((self.n_h, 1))

        for j in range(0, len(X_trimmed) - self.seq_len, self.seq_len):
            # prepare batches
            x_batch = [self.char_to_idx[ch] for ch in X_trimmed[j: j + self.seq_len]]
            y_batch = [self.char_to_idx[ch] for ch in X_trimmed[j + 1: j + self.seq_len + 1]]

            loss, h_prev, c_prev = self.forward_backward(x_batch, y_batch, h_prev, c_prev)

            # smooth out loss and store in list
            self.smooth_loss = self.smooth_loss * 0.999 + loss * 0.001
            J.append(self.smooth_loss)

            # check gradients
            if epoch == 0 and j == 0:
                self.gradient_check(x_batch, y_batch, h_prev, c_prev, num_checks=10, delta=1e-7)

            self.clip_grads()

            batch_num = epoch * self.epochs + j / self.seq_len + 1
            self.update_params(batch_num)

            # print out loss and sample string
            if verbose:
                if j % 400000 == 0:
                    print('Epoch:', epoch, '\tBatch:', j, "-", j + self.seq_len,
                          '\tLoss:', round(self.smooth_loss, 2))
                    s = self.sample(h_prev, c_prev, sample_size=250)
                    print(s, "\n")
    return J, self.params

LSTM.train = train
```

##### Results
Finally, we will run the training process for 5 iterations.

```python
model = LSTM(char_to_idx, idx_to_char, vocab_size, epochs = 5, lr = 0.01)
J, params = model.train(data)
```

<code>

Epoch: 0 	Batch: 0 - 25 	Loss: 99.72  
Jo!~zyy64 f??0xtnqr7"“jol31irp?a*bmm;-efx;vb.;a9:-5l.'7v,“a;xk.x6gx(6si8–mqpj*jq!udfgymvi“9tbp h!h4ken052cihessw-:5\2\74f~s1pt“9'nvx?ysuh0m;,jjn~yu5e48dib3paq"m2z9–56~“7597d 
E!“ qmw0p)gj'-x10341-iq:)yv)-dep.:)sy~2*9aj:6?–j'dd:78hfl4,y60ya-jntp 

Epoch: 0 	Batch: 400000 - 400025 	Loss: 47.69  
ope at cou.... 

"dove the anased, hermione out on ascowkri weing thee blimesace ind y-u her ceuld you haspill, cilleler for, he tour you not ther ele jatt te you bleat purofed coed a, himsint to gonl. sbangout gotagone cos it hould," siree?"in's he. 

Epoch: 1 	Batch: 0 - 25 	Loss: 47.36  
?" the muspe the swertery," harnyound." 

ait sheer in erey -- in you,?r juven a putced of the bain kem the gould the and ande it kiry stheng u he sarmy to a veingl, you hak was fory. 

"wave tay gotnts onte file somione walled to the von! hem ape ea 

Epoch: 1 	Batch: 400000 - 400025 	Loss: 42.05  
antin i marst. 

"sny?" 

"in't woated. 

"" havey to leder and reamy doyed to bopreas slop ay?" har'y and tree tho gosteng bedoning but thap hard anythiont.". the comes beining bucked to there wilss a lifged a diglt beed. i gusted got carst witsheri 

Epoch: 2 	Batch: 0 - 25 	Loss: 42.32  
 on. 

"rexursell been them tungerlly dednow, put one a't seaner. 

herriokge, he ston's glinged sely he mught. "ye,." bac it, he said just gally, you," said, "is ed. the some coired. ic was tookstse," 

and ron," said screchens: a bryaut of he stall 

Epoch: 2 	Batch: 400000 - 400025 	Loss: 39.16  
uldeding what tears from the dooms fronlaim the one aw thr again. in he piond beate, he'piled weellay's fouddout plait," lelt can't a noich arope-ficling hermoling, and hermione, the wlithing to durglevor evevingses agothere and haw and might of them 

Epoch: 3 	Batch: 0 - 25 	Loss: 39.68  
ay, he cain!" said hagrid magalay's low, and our shunds?" tarng, engon't yooh was them. see -- shat." 

it weot sift you mased was ears oworeing us i donn." 

"arwuin upcors avely from maying that one. pury. mer quted would to at the coulding tomk am 

Epoch: 3 	Batch: 400000 - 400025 	Loss: 37.28  
ribla gry toued, we maingh the praying they out harry and simptsor streat, neacly sunding up inta tice of the ortepled at hergion grinned you just franteriogess, blioking, but want a firs-'s they the coods behind hermione aw they goinging toke.. he s 

Epoch: 4 	Batch: 0 - 25 	Loss: 37.92  
ain, askeds to grow." 

"i was a drofey strent for yelle a pupazely unyerseached the steninied, soub, we'll it, but cemsting!" 

peeze, cillod, but would suef freens gundes, forlit brood neder high quidey plyfoid me -- more..., i'me gove, in the purs 

Epoch: 4 	Batch: 400000 - 400025 	Loss: 35.93  
wing. "she --" 

"they dunned. "there, poing's nowed him i poents cuddents wils a the trank of snare-aching was what ,"drown out of the and going. how exven you a stone of his sayt aborghtaney deat who it wecenes to dight," so mach down she cleaved d 
</code>

--- 
Although we have just trained our LSTM for a few iterations, it has learnt to correctly spell words like harry and hermione, and how to start dialogue with quotation marks.

```python
plt.plot([i for i in range(len(J))], J)
plt.xlabel("#training iterations")
plt.ylabel("training loss")
```

{{< figure src="images/blog_implementlstm_2.png" title="" width="50%">}}

##### Improvements
This implementation prioritised simplicity and clarity over performance, thus there are a few enhancements that are worth mentioning:

- **Gradient checking**: To check the backpropagation calculation, we can numerically approximate the gradient at a point and compare it to the model’s backprop gradient. Although this was not discussed in this post, it is included in my implementation on GitHub.
- **Regularisation**: Large recurrent neural networks tend to overfit the training data. Dropout can be used to generalise results better, however it needs to be applied differently than to feedforward neural networks otherwise performance is not enhanced.
- **Gradient clipping**: To prevent exploding gradients, we have clipped their maximum value. This, however, does not improve performance consistently. Rather than clipping each gradient independently, clipping the global norm of the gradient yields more significant improvements.
- **Learning the initial state**: We initialised the initial LSTM states to zero.  Instead of fixing the initial state, we can learn it like any other parameter, which can improve performance and is also recommended by Hinton.