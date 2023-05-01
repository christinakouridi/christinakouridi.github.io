---
title: "Accelerating Python functions with Numba"
date: 2019-12-19
draft: false
---

In this post, I will provide a brief overview of **[Numba](http://numba.pydata.org/), an open-source just-in-time function compiler, which can speed up subsets of your Python code** easily, and with minimal intervention. Unlike other popular JIT compilers (e.g. Cython, pypy) Numba simply requires the addition of a function decorator, with the premise of approaching the speed of C or Fortran. Your source code remains pure Python while Numba handles the compilation at runtime.

##### How does it work?

Numba works best on code that uses **Numpy arrays and functions, as well as loops**. It also supports many of the functions from the math module. A comprehensive list of compatible functions can be found [here](http://numba.pydata.org/numba-doc/0.17.0/reference/pysupported.html). 

The easiest way to use it is through a collection of decorators applied to functions that instruct Numba to compile them (examples later!). When a Numba decorated function is called, **it is compiled to machine code just-in-time for execution**. This enables the entire or subsets of your code to subsequently **run at native machine code speed**.

Numba **generates optimised machine code from Python using the industry-standard LLVM compiler library** (instead of a custom-made compiler, which made Numba possible). It handles all of the details around optimising code and generating machine code. The compilation sequence is as follows [[source](https://www.youtube.com/watch?v=-4tD8kNHdXs)]:

{{< figure src="images/numba.png" title="Figure 1: Numba compilation sequence" width="75%">}}

##### Numba compilation sequence
**A key step in the compilation process, is the conversion of the Python function in consideration to Numba’s intermediary representation**. This process involves **swapping supported functions to implementations provided by Numba, that it can translate fast to machine code**. Python objects are stripped from the provided and inferred data types and are translated into representations with no CPython dependencies. This is then converted into LLVM interpretable code and fed into LLVM’S JIT compiler to get machine code. The code is cached, so that the entire compilation process won’t be repeated next time the function is called.

Numba also offers a range of options for parallelising your code for CPUs and GPUs, often with only minor code changes.

##### Current limitations

- Numba **compiles Python functions, not entire programs** (pypy is great for that).  It also doesn’t support partial compilation of functions – it needs to be able to resolve all data types in the selected function. 
- Presently, Numba is focused on numerical data types, like *int*, *float*, and *complex*. There is very **limited string processing support** and the best results are realised with Numpy arrays.
- Decorating functions that make use of Pandas (or other unsupported data structures) would deteriorate performance. **Pandas is not understood by Numba** and as a result, Numba would simply run this code via the interpreter but with the additional cost of the Numba internal overheads.
- You are better off using Cython for code that interferes with C++, as Numba can’t talk with C++ effectively unless a C wrapper is used.
- **Numba doesn’t generate C/C++ code that can be used for a separate compilation**; it goes directly from Python down to LLVM code.  Cython would be more suitable for this use case,  as it allows inspection of the code in C++ before compilation.

##### A few examples

Below are a few quick demonstrations of how Numba can accelerate your functions. More examples can be found [here](http://numba.pydata.org/).

```python
import numpy as np
import pandas as pd
from sklearn import datasets
from numba import jit, njit, prange
```

load dummy dataset

```python
digits = datasets.load_digits()
X = digits['data']
```

###### Example 1 – numpy function

To enable Numba, simply add the decorator *@njit*.

```python
def func(X):
    Y = np.exp(-X)
    return Y
%timeit func(X)
```

828 µs ± 20.4 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)


```python
@njit
def njit_func(X):
    Y = np.exp(-X)
    return Y
%timeit njit_func(X)
```

710 µs ± 167 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)


###### Example 2 – numpy function and loop

Numba’s prange provides the ability to run loops in parallel, that are scheduled in separate threads (similar to Cython’s prange). Simply replace range with prange.

```python
def func(X):
    for i in range(10000):
        Y = np.exp(-X)
    return Y
%timeit func(X)
```

8.75 s ± 570 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)


```python
@njit
def njit_func(X):
    for i in prange(10000):
        Y = np.exp(-X)
    return Y
%timeit njit_func(X)
```

6.46 s ± 17.2 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)


###### Example 3 – math functions

```python
def hypotenuse(x, y):
    x = abs(x);
    y = abs(y);
    t = min(x, y);
    x = max(x, y);
    t = t / x;
    return x * math.sqrt(1+t*t)

%timeit hypotenuse(5.0, 12.0)
```

674 ns ± 12.2 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)


```python
@njit
def njit_hypotenuse(x, y):
    x = abs(x);
    y = abs(y);
    t = min(x, y);
    x = max(x, y);
    t = t / x;
    return x * math.sqrt(1+t*t)

%timeit njit_hypotenuse(5.0, 12.0)
```


160 ns ± 1.61 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)


##### Remarks

- As mentioned, Numba can’t compile all Python code; **certain functions don’t have a Numba translation, and some data structures can’t be effectively compiled yet (e.g. Pandas)**. When this occurs Numba falls back into a mode,  called “object mode”, which doesn’t do type inference. Unfortunately Numba does not inform the user when this happens.

- In the Numba world, you will also come across the jit decorator. It serves the same purpose as the njit operator, with only different being that jit is falling back to object mode by default, without providing any error warnings if type inference fails. On the other hand, *njit* would raise a warning and break the code. You may also come across its alias, *@jit(nopython=True)*. Let’s see an example.

    ```python
    df = pd.DataFrame(data=[[2,4],[1,3]] , columns=['even', 'odd'])
    @jit
    def bad_example(x):
        return x['even']

    bad_example(df)
    ```

    ```
    0    2  
    1    1  
    Name: even, dtype: int64
    ```

    Although with *@jit* the code runs successfully, *@njit* raises an error.

    ```python
    @njit
    def bad_example(x):
        return x['even']

    bad_example(df)
    ```
    ```
    TypingError                               Traceback (most recent call last)
    <ipython-input-12-63299406f3ac> in <module>()
        3     return x['even']
        4 
    ----> 5 bad_example(df)
        6 
        7 bad_example(df)

    ~/anaconda3/lib/python3.6/site-packages/numba/dispatcher.py in _compile_for_args(self, *args, **kws)
        399                 e.patch_message(msg)
        400 
    --> 401             error_rewrite(e, 'typing')
        402         except errors.UnsupportedError as e:
        403             # Something unsupported is present in the user code, add help info

    ~/anaconda3/lib/python3.6/site-packages/numba/dispatcher.py in error_rewrite(e, issue_type)
        342                 raise e
        343             else:
    --> 344                 reraise(type(e), e, None)
        345 
        346         argtypes = []

    ~/anaconda3/lib/python3.6/site-packages/numba/six.py in reraise(tp, value, tb)
        666             value = tp()
        667         if value.__traceback__ is not tb:
    --> 668             raise value.with_traceback(tb)
        669         raise value
        670 

    TypingError: Failed in nopython mode pipeline (step: nopython frontend)
    non-precise type pyobject
    [1] During: typing of argument at <ipython-input-12-63299406f3ac> (3)

    File "<ipython-input-12-63299406f3ac>", line 3:
    def bad_example(x):
        return x['even']
        ^

    This error may have been caused by the following argument(s):  
    -argument 0: cannot determine Numba type of <class 'pandas.core.frame.DataFrame'>
    ```

    Usually this is not a problem with Numba itself but instead **often caused by the use of unsupported features or an issue in resolving types**. Python/ NumPy features supported by the latest release can be found [here](http://numba.pydata.org/numba-doc/latest/reference/pysupported.html) and [here](http://numba.pydata.org/numba-doc/latest/reference/numpysupported.html). More information on typing errors and how to debug them can be found [here](http://numba.pydata.org/numba-doc/latest/user/troubleshoot.html#my-code-doesn-t-compile).  

    Numba is under active development with lots of exciting functionality in store (e.g. class wrappers). To help its development, consider reporting any new issues at [Numba's GitHub repo](https://github.com/numba/numba/issues/new).