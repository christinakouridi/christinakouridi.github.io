---
title: "My machine learning research toolkit"
date: 2020-10-31
draft: false
---

In this post, I will share the key tools in my machine learning research workflow. My selection criteria included free accessibility to students, ease of adoption, active development, and quality of features.

##### 1. Terminal session organiser - [[Tmux](https://github.com/tmux/tmux/wiki)]

Tmux is a terminal multiplexer; it facilitates running and organising sessions on the terminal. Specifically, it enables alternating between several sessions in one terminal, and restoring their state after detachment (i.e. closing the terminal window does not terminate them). Those sessions can be viewed at the same time by splitting the terminal view.


{{< figure src="images/blog_mltoolkit_1.png" title="Figure 1: example tmux session with split terminal view" width="90%">}}

A common alternative is Linux’s native [Screen](https://linuxize.com/post/how-to-use-linux-screen/), however it is less user friendly. In addition, screen layouts do not automatically persist.

An excellent guide on setting up Tmux can be found [here](https://www.hamvocke.com/blog/a-quick-and-easy-guide-to-tmux/).

---

##### 2. Code development - [[PyCharm Professional](https://www.jetbrains.com/pycharm/)]

PyCharm is an engineering marvel, and students get to enjoy the full range of features that comes with its Professional edition for free. Key features include:

- Git integration with the ability to view and review local tracked changes
- remote development with an integrated ssh terminal
- Docker integration
- [Code With Me](https://www.jetbrains.com/code-with-me/) service for pair programming
- various developer tools, such as built-in terminal, debugger and test runner
- python profiling for speed improvements

{{< figure src="images/blog_mltoolkit_2.png" title="Figure 2: example line profiling in PyCharm" width="90%">}}

---

##### 3. Code organisation - [[GitHub](https://github.com/)]

If you are reading this, you must know that GitHub is THE code platform for version control and collaboration. Beyond these, I have used it extensively to search for code snippets within my code-base — its accuracy has never disappointed me!

My top tips for utilising it effectively: firstly, using tags to pin and document important commits under the releases page will speed up inspecting and reverting to past code; secondly, write elaborate commit messages (in large collaborative projects that may require paragraphs).

Students get access to private repositories, which are ideal for storing work-in-progress research.

---

##### 4. Experiment tracking - [[Weights & Biases](https://wandb.ai/site)]

This is perhaps one of the least known tools on my list, but the one that has improved my productivity the most. Weights & Biases (or W&B in short) can be used to visualise the results of machine learning experiments (e.g. training curves, OpenAI Gym agent trajectories, matplotlib plots), track metrics and model configurations, and perform parameter sweeps (e.g. grid search, random search, bayesian optimisation). Results can be organised in custom groups and projects, along with the code that generated them.

Particularly useful is the ability to plot aggregated results, for example the average training curve across multiple random seeds, along with an indication of variation (e.g. min and max bounds, standard deviation or standard error). Plots are almost paper-ready and fairly customisable, but if you want to add just that little bit of matplotlib magic, the exact data points can be exported to a .csv file.

{{< figure src="images/blog_mltoolkit_3.png" title="Figure 3: example aggregated experimental results in Weights & Biases" width="95%">}}

Integration couldn’t be easier; only a couple of lines of code are needed in a python script. Luckily, it is also compatible with both PyTorch and TensorFlow.

A final attractive aspect of Weights & Biases is its highly responsive support team — they are readily available for questions and feature suggestions through live chat.

---

##### 5. Research/note tracking and more - [[Notion](https://www.notion.so/)]

I have experimented with various note taking apps for research, and life in general. Notion tops my list (with OneNote a close second) due to the power and flexibility of its component system: elements such as images, videos, tables, TeX and code snippets can easily be incorporated on custom pages.

Another neat feature is the ability to host pages without any of the typical configuration (e.g. a [page](https://www.notion.so/Agent-Analysis-678a4693229542868f2d526e132df4cd) that I created to showcase results for a project).

{{< figure src="images/blog_mltoolkit_4.png" title="Figure 4: example Notion page for tracking paper notes" width="95%">}}

Although they don’t provide the range of hand-writing features that OneNote does (e.g. hand-written notes on embedding pdfs, search through hand-written notes), Notion is actively developing these much sought-after capabilities (as evident by the recently announced hand-written to digital text conversion using Apple Pencil).

---

##### 6. Academic writing - [[Overleaf](https://overleaf.com)]

Overleaf is a widely-used LaTeX editor for collaboratively authoring scientific papers. A note of caution — being cloud-based means that you are reliant on internet speed and access; make sure to back-up your work as it is not unheard of for Overleaf to be down for updates close to machine learning conference deadlines.
