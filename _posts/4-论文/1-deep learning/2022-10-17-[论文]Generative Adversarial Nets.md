---
title: 【论文】Generative Adversarial Nets 生成对抗网络
date: 2022-10-17 08:00:00 +/-0800
categories: [论文,深度学习]
tags: [深度学习]     # TAG names should always be lowercase 标记名称应始终为小写
math: true
---


Ian J. Goodfellow∗, Jean Pouget-Abadie†, Mehdi Mirza, Bing Xu, David Warde-Farley,Sherjil Ozair‡, Aaron Courville, Yoshua Bengio§

## Abstract 摘要

We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model $G$ that captures the data distribution, and a discriminative model $D$ that estimates the probability that a sample came from the training data rather than $G$. The training procedure for $G$ is to maximize the probability of $D $making a mistake. This framework corresponds to a minimax two-player game. In the space of arbitrary functions $G$ and $D$, a unique solution exists, with G recovering the training data distribution and $D$ equal to $\frac{1}{2}$ everywhere. In the case where $G$ and $D$ are defined by multilayer perceptrons, the entire system can be trained with backpropagation. There is no need for any Markov chains or unrolled approximate inference networks during either training or generation of samples. Experiments demonstrate the potential of the framework through qualitative and quantitative evaluation of the generated samples.

我们提出了一个通过对抗过程估计生成模型的新框架，其中我们同时训练两个模型：一个生成模型$G$捕捉数据分布，另一个判别模型$D$估计一个样本来自训练数据而不是$G$的概率。G$的训练程序是使D$犯错的概率最大化。这个框架对应于一个最小化的双人游戏。在任意函数$G$和$D$的空间中，存在一个唯一的解，其中 G 恢复训练数据分布，$D$ 等于 $\frac{1}{2}$。 在 $G$ 和 $D$ 由多层感知器定义的情况下，整个系统可以通过反向传播进行训练。 在训练或生成样本期间，不需要任何马尔可夫链或展开的近似推理网络。 实验通过对生成的样本进行定性和定量评估，证明了框架的潜力。

## 1 Introduction 简介

The promise of deep learning is to discover rich, hierarchical models [2] that represent probability distributions over the kinds of data encountered in artificial intelligence applications, such as natural images, audio waveforms containing speech, and symbols in natural language corpora. So far, the most striking successes in deep learning have involved discriminative models, usually those that map a high-dimensional, rich sensory input to a class label [14,20]. These striking successes have primarily been based on the backpropagation and dropout algorithms, using piecewise linear units [17,8,9] which have a particularly well-behaved gradient. Deep *generative* models have had less of an impact, due to the difficulty of approximating many intractable probabilistic computations that arise in maximum likelihood estimation and related strategies, and due to difficulty of leveraging the benefits of piecewise linear units in the generative context. We propose a new generative model estimation procedure that sidesteps these difficulties.$^1$

深度学习的目的是发现丰富的、分层的模型[2]，代表人工智能应用中遇到的各种数据的概率分布，如自然图像、包含语音的音频波形和自然语言语料中的符号。到目前为止，深度学习中最引人注目的成功涉及到判别性模型，通常是那些将高维的、丰富的感官输入映射到一个类别标签的模型[14,20]。这些引人注目的成功主要是基于反向传播和dropout算法，使用具有特别良好梯度的分片线性单元[17,8,9]。深度*生成*模型的影响较小，原因是难以接近最大似然估计和相关策略中出现的许多难以处理的概率计算，以及难以在生成背景下利用分片线性单元的优势。我们提出了一个新的生成模型估计程序，以回避这些困难。

> $^1$All code and hyperparameters available at http://www.github.com/goodfeli/adversarial
> > $^1$所有代码和超参数可在http://www.github.com/goodfeli/adversarial。

In the *proposed adversarial nets* framework, the generative model is pitted against an adversary: a discriminative model that learns to determine whether a sample is from the model distribution or the data distribution. The generative model can be thought of as analogous to a team of counterfeiters, trying to produce fake currency and use it without detection, while the discriminative model is analogous to the police, trying to detect the counterfeit currency. Competition in this game drives both teams to improve their methods until the counterfeits are indistiguishable from the genuine articles.

在*提议的对抗性网络*框架中，生成性模型与对手对立：辨别性模型，它可以学习确定一个样本是来自模型分布还是数据分布。生成式模型可以被认为是类似于一队造假者，试图制造假币并在不被发现的情况下使用，而判别式模型则类似于警察，试图检测假币。这个游戏中的竞争促使两个团队改进他们的方法，直到假币与真币无法区分。

This framework can yield specific training algorithms for many kinds of model and optimization algorithm. In this article, we explore the special case when the generative model generates samples by passing random noise through a multilayer perceptron, and the discriminative model is also a multilayer perceptron. We refer to this special case as *adversarial nets*. In this case, we can train both models using only the highly successful backpropagation and dropout algorithms [16] and sample from the generative model using only forward propagation. No approximate inference or Markov chains are necessary.

该框架可以为多种模型和优化算法生成特定的训练算法。 在本文中，我们探讨了生成模型通过多层感知器传递随机噪声来生成样本的特殊情况，而判别模型也是多层感知器。 我们将这种特殊情况称为*对抗网络*。 在这种情况下，我们可以仅使用非常成功的反向传播和 dropout 算法 [16] 训练这两个模型，并仅使用前向传播从生成模型中采样。 不需要近似推理或马尔可夫链。

## 2 Related work 相关工作

Until recently, most work on deep generative models focused on models that provided a parametric specification of a probability distribution function. The model can then be trained by maximizing the log likelihood. In this family of model, perhaps the most succesful is the deep Boltzmann machine [25]. Such models generally have intractable likelihood functions and therefore require numerous approximations to the likelihood gradient. These difficulties motivated the development of “generative machines”-models that do not explicitly represent the likelihood, yet are able to generate samples from the desired distribution. Generative stochastic networks [4] are an example of a generative machine that can be trained with exact backpropagation rather than the numerous approximations required for Boltzmann machines. This work extends the idea ofa generative machine by eliminating the Markov chains used in generative stochastic networks.

直到最近，大多数关于深度生成模型的工作都集中在提供概率分布函数的参数规范的模型上。 然后可以通过最大化对数似然来训练模型。 在这个模型家族中，也许最成功的是深度玻尔兹曼机[25]。 此类模型通常具有难以处理的似然函数，因此需要对似然梯度进行大量近似。 这些困难推动了“生成机器”的发展——这些模型不能明确表示可能性，但能够从所需的分布中生成样本。 生成随机网络 [4] 是生成机器的一个例子，它可以通过精确的反向传播而不是玻尔兹曼机器所需的大量近似值进行训练。 这项工作通过消除生成随机网络中使用的马尔可夫链来扩展生成机器的想法。

Our work backpropagates derivatives through generative processes by using the observation that

我们的工作通过使用观察结果通过生成过程反向传播导数

$$
\lim_{\sigma \rightarrow 0} \bigtriangledown_x \mathbb{E}_{\epsilon \sim \mathcal{N}(0,\sigma_2I)} f(x + \epsilon) = \bigtriangledown_xf(x).
$$

We were unaware at the time we developed this work that Kingma and Welling [18] and Rezende *et al*. [23] had developed more general stochastic backpropagation rules, allowing one to backprop- agate through Gaussian distributions with finite variance, and to backpropagate to the covariance parameter as well as the mean. These backpropagation rules could allow one to learn the conditional variance of the generator, which we treated as a hyperparameter in this work. Kingma and Welling [18] and Rezende *et al*. [23] use stochastic backpropagation to train variational autoencoders (VAEs). Like generative adversarial networks, variational autoencoders pair a differentiable generator network with a second neural network. Unlike generative adversarial networks, the second network in a VAE is a recognition model that performs approximate inference. GANs require differentiation through the visible units, and thus cannot model discrete data, while VAEs require differentiation through the hidden units, and thus cannot have discrete latent variables. Other VAE-like approaches exist [12,22] but are less closely related to our method.

在我们开展这项工作时，我们并不知道Kingma和Welling[18]以及Rezende *et al*. [23]已经开发了更多的随机反向传播规则，允许人们通过有限方差的高斯分布进行反向传播，并且可以反向传播协方差参数和平均数。这些反向传播规则可以让人们学习生成器的条件方差，我们在这项工作中把它作为一个超参数。Kingma和Welling[18]以及Rezende *et al*. [23]使用随机反向传播来训练变异自动编码器（VAEs）。与生成式对抗网络一样，变异自动编码器将一个可微分生成器网络与第二个神经网络配对。与生成式对抗网络不同，VAE中的第二个网络是一个识别模型，可以进行近似推理。GANs需要通过可见单元进行区分，因此不能对离散数据建模，而VAEs需要通过隐藏单元进行区分，因此不能有离散的潜变量。其他类似VAE的方法也存在[12,22]，但与我们的方法关系不太密切。

Previous work has also taken the approach of using a discriminative criterion to train a generative model [29,13]. These approaches use criteria that are intractable for deep generative models. These methods are difficult even to approximate for deep models because they involve ratios of probabilities which cannot be approximated using variational approximations that lower bound the probability. Noise-contrastive estimation (NCE) [13] involves training a generative model by learning the weights that make the model useful for discriminating data from a fixed noise distribution. Using a previously trained model as the noise distribution allows training a sequence of models of increasing quality. This can be seen as an informal competition mechanism similar in spirit to the formal competition used in the adversarial networks game. The key limitation of NCE is that its “discriminator” is defined by the ratio of the probability densities of the noise distribution and the model distribution, and thus requires the ability to evaluate and backpropagate through both densities.

以前的工作也采用了使用判别标准来训练生成模型的方法[29,13]。这些方法使用深度生成模型难以处理的标准。这些方法甚至对于深度模型也很难近似，因为它们涉及的概率比率不能使用降低概率的变分近似来近似。噪声对比估计 (NCE) [13] 涉及通过学习使模型可用于从固定噪声分布中区分数据的权重来训练生成模型。使用先前训练的模型作为噪声分布允许训练一系列提高质量的模型。这可以看作是一种非正式的竞争机制，在精神上类似于对抗性网络游戏中使用的正式竞争。 NCE 的关键限制是它的“鉴别器”是由噪声分布和模型分布的概率密度之比定义的，因此需要能够通过这两种密度进行评估和反向传播。

Some previous work has used the general concept of having two neural networks compete. The most relevant work is predictability minimization [26]. In predictability minimization, each hidden unit in a neural network is trained to be different from the output of a second network, which predicts the value of that hidden unit given the value of all of the other hidden units. This work differs from predictability minimization in three important ways: 1) in this work, the competition between the networks is the sole training criterion, and is sufficient on its own to train the network. Predictability minimization is only a regularizer that encourages the hidden units of a neural network to be statistically independent while they accomplish some other task; it is not a primary training criterion. 2) The nature of the competition is different. In predictability minimization, two networks’ outputs are compared, with one network trying to make the outputs similar and the other trying to make the outputs different. The output in question is a single scalar. In GANs, one network produces a rich, high dimensional vector that is used as the input to another network, and attempts to choose an input that the other network does not know how to process. 3) The specification of the learning process is different. Predictability minimization is described as an optimization problem with an objective function to be minimized, and learning approaches the minimum of the objective function. GANs are based on a minimax game rather than an optimization problem, and have a value function that one agent seeks to maximize and the other seeks to minimize. The game terminates at a saddle point that is a minimum with respect to one player’s strategy and a maximum with respect to the other player’s strategy.

以前的一些工作使用了让两个神经网络竞争的一般概念。最相关的工作是可预测性最小化[26]。在可预测性最小化中，神经网络中的每个隐藏单元都被训练为不同于第二个网络的输出，第二个网络在给定所有其他隐藏单元的值的情况下预测该隐藏单元的值。这项工作在三个重要方面不同于可预测性最小化：1）在这项工作中，网络之间的竞争是唯一的训练标准，并且其本身就足以训练网络。可预测性最小化只是一个正则化器，它鼓励神经网络的隐藏单元在完成其他任务时在统计上独立；这不是主要的培训标准。 2）比赛性质不同。在可预测性最小化中，比较两个网络的输出，一个网络试图使输出相似，另一个网络试图使输出不同。有问题的输出是单个标量。在 GAN 中，一个网络生成一个丰富的高维向量，用作另一个网络的输入，并尝试选择另一个网络不知道如何处理的输入。 3）学习过程的规范不同。可预测性最小化被描述为具有要最小化的目标函数的优化问题，并且学习接近目标函数的最小值。 GAN 基于极大极小博弈而不是优化问题，并且具有一个价值函数，一个智能体寻求最大化而另一个智能体寻求最小化。游戏在一个鞍点处终止，该鞍点对于一个玩家的策略而言是最小值，而对于另一个玩家的策略而言是最大值。

Generative adversarial networks has been sometimes confused with the related concept of “adversarial examples” [28]. Adversarial examples are examples found by using gradient-based optimization directly on the input to a classification network, in order to find examples that are similar to the data yet misclassified. This is different from the present work because adversarial examples are not a mechanism for training a generative model. Instead, adversarial examples are primarily an analysis tool for showing that neural networks behave in intriguing ways, often confidently classifying two images differently with high confidence even though the difference between them is imperceptible to a human observer. The existence of such adversarial examples does suggest that generative adversarial network training could be inefficient, because they show that it is possible to make modern discriminative networks confidently recognize a class without emulating any of the human-perceptible attributes of that class.

生成对抗网络有时会与“对抗样本”的相关概念混淆 [28]。对抗性示例是通过直接在分类网络的输入上使用基于梯度的优化来找到的示例，以便找到与数据相似但被错误分类的示例。这与目前的工作不同，因为对抗样本不是训练生成模型的机制。相反，对抗性示例主要是一种分析工具，用于显示神经网络以有趣的方式表现，即使人类观察者无法察觉它们之间的差异，它们通常也会自信地以高置信度对两个图像进行不同的分类。这种对抗性示例的存在确实表明生成对抗性网络训练可能效率低下，因为它们表明可以使现代判别网络自信地识别一个类，而无需模仿该类的任何人类可感知属性。

## 3 Adversarial nets 对抗网络

The adversarial modeling framework is most straightforward to apply when the models are both multilayer perceptrons. To learn the generator’s distribution $p_g$ over data $x$, we define a prior on input noise variables $p_z(z)$, then represent a mapping to data space as $G(z;\theta_g)$, where $G$ is a differentiable function represented by a multilayer perceptron with parameters $\theta_g$. We also define a second multilayer perceptron $D(x;\theta_d)$ that outputs a single scalar. $D(x)$ represents the probability that $x$ came from the data rather than $p_g$ . We train $D$ to maximize the probability of assigning the correct label to both training examples and samples from $G$. We simultaneously train $G$ to minimize ${\rm lopg}(1 - D(G(z)))$. In other words, $D$ and $G$ play the following two-player minimax game with value function $V (G,D)$:

当模型都是多层感知器时，对抗性建模框架最容易应用。 为了学习生成器在数据 $x$ 上的分布 $p_g$，我们在输入噪声变量 $p_z(z)$ 上定义先验，然后将到数据空间的映射表示为 $G(z;\theta_g)$，其中 $G $ 是由带有参数 $\theta_g$ 的多层感知器表示的可微函数。 我们还定义了输出单个标量的第二个多层感知器 $D(x;\theta_d)$。 $D(x)$ 表示 $x$ 来自数据而不是 $p_g$ 的概率。 我们训练 $D$ 以最大化将正确标签分配给来自 $G$ 的训练示例和样本的概率。 我们同时训练 $G$ 以最小化 ${\rm lopg}(1 - D(G(z)))$。 换句话说，$D$ 和 $G$ 玩下面的两人最小极大游戏，价值函数 $V (G,D)$：

$$
\min_G \max_D V(D；G) = \mathbb{E}_{x \sim p_{data}}(x)[{\rm log} \ D(x)] + \mathbb{E}_{z \sim p_z(z)}[{\rm log}(1 — D(G(z)))]. \tag{1}
$$

In the next section, we present a theoretical analysis of adversarial nets, essentially showing that the training criterion allows one to recover the data generating distribution as $G$ and $D$ are given enough capacity, *i.e.*, in the non-parametric limit. See Figure 1 for a less formal, more pedagogical explanation of the approach. In practice, we must implement the game using an iterative, numerical approach. Optimizing $D$ to completion in the inner loop of training is computationally prohibitive, and on finite datasets would result in overfitting. Instead, we alternate between $k$ steps of optimizing $D$ and one step of optimizing $G$. This results in $D$ being maintained near its optimal solution, so long as $G$ changes slowly enough. The procedure is formally presented in Algorithm 1.

在下一节中，我们将介绍对抗网络的理论分析，基本上表明训练标准允许人们恢复数据生成分布，因为 $G$ 和 $D$ 被给予足够的容量，即在非参数限制内。 有关该方法的不太正式、更具教学性的解释，请参见图 1。 在实践中，我们必须使用迭代的数值方法来实现游戏。 在训练的内部循环中优化 $D$ 以完成计算是令人望而却步的，并且在有限的数据集上会导致过度拟合。 相反，我们在优化$D$的$k$步骤和优化$G$的步骤之间交替。 这导致 $D$ 保持在其最佳解决方案附近，只要 $G$ 变化足够慢。 该过程在算法 1 中正式介绍。

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Generative-Adversarial-Nets/Algorithm%201.png"/></div>

图 1：通过同时更新判别分布（$D$，蓝色，虚线）来训练生成对抗网络，以便它区分来自数据生成分布（黑色，虚线）的样本 $p_x$ 和生成分布的样本$p_g (G)$（绿色，实线）。较低的水平线是从中采样 $z$ 的域，在这种情况下是均匀的。上面的水平线是 $x$ 域的一部分。向上的箭头显示了映射 $x = G(z)$ 如何将非均匀分布 $p_g$ 施加到转换后的样本上。 $G$ 在高密度区域收缩并在 $p_g$ 的低密度区域扩展。 (a) 考虑一个接近收敛的对抗对：$p_g$ 类似于 $p_{data}$，$D$ 是一个部分准确的分类器。(b) 在算法的内部循环中，$D$ 被训练以区分样本从数据，收敛到 $D^∗(x) =\frac{p_{data}(x)}{p_{data}(x)+p_g(x)}$。 (c) 更新到 $G$ 后，$D$ 的梯度引导 $G(z)$ 流向更有可能被分类为数据的区域。 (d) 经过几个步骤的训练，如果 $G$ 和 $D$ 有足够的容量，它们将达到一个点，因为 $p_g = p_{data}$，它们都无法提高。鉴别器无法区分这两个分布，即 $D(x) = \frac{1}{2}$。

In practice, equation 1 may not provide sufficient gradient for $G$ to learn well. Early in learning, when $G$ is poor, $D$ can reject samples with high confidence because they are clearly different from the training data. In this case, ${\rm log}(1 - D(G(z)))$ saturates. Rather than training $G$ to minimize ${\rm log}(1 一 D(G(z)))$ we can train $G$ to maximize log $D(G(z))$. This objective function results in the same fixed point of the dynamics of $G$ and $D$ but provides much stronger gradients early in learning.

在实践中，等式 1 可能无法为 G 提供足够的梯度来很好地学习。 在学习的早期，当$G$很差时，$D$可以以高置信度拒绝样本，因为它们与训练数据明显不同。 在这种情况下，${\rm log}(1 - D(G(z)))$ 饱和。 与其训练 $G$ 来最小化 ${\rm log}(1 一 D(G(z)))$，我们可以训练 $G$ 来最大化 log $D(G(z))$。 这个目标函数导致 $G$ 和 $D$ 的动态相同的固定点，但在学习早期提供了更强的梯度。

## 4 Theoretical Results

The generator $G$ implicitly defines a probability distribution $p_g$ as the distribution of the samples $G(z)$ obtained when $z \sim p_z$. Therefore, we would like Algorithm 1 to converge to a good estimator of $p_{data}$, if given enough capacity and training time. The results of this section are done in a nonparametric setting, *e.g.* we represent a model with infinite capacity by studying convergence in the space of probability density functions.

We will show in section 4.1 that this minimax game has a global optimum for $p_g = p_{data}$. We will then show in section 4.2 that Algorithm 1 optimizes Eq 1, thus obtaining the desired result.

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Generative-Adversarial-Nets/Fig%201.png"/></div>

Figure 1: Generative adversarial nets are trained by simultaneously updating the discriminative distribution($D$, blue, dashed line) so that it discriminates between samples from the data generating distribution (black,dotted line) $p_x$ from those of the generative distribution $p_g (G)$ (green, solid line). The lower horizontal line is the domain from which $z$ is sampled, in this case uniformly. The horizontal line above is part of the domain of $x$. The upward arrows show how the mapping $x = G(z)$ imposes the non-uniform distribution $p_g$ on transformed samples. $G$ contracts in regions of high density and expands in regions of low density of $p_g$. (a)Consider an adversarial pair near convergence: $p_g$ is similar to $p_{data}$ and $D$ is a partially accurate classifier.(b) In the inner loop of the algorithm $D$ is trained to discriminate samples from data, converging to $D^∗(x) =\frac{p_{data}(x)}{p_{data}(x)+p_g(x)}$. (c) After an update to $G$, gradient of $D$ has guided $G(z)$ to flow to regions that are more likely to be classified as data. (d) After several steps of training, if $G$ and $D$ have enough capacity, they will reach a point at which both cannot improve because $p_g = p_{data}$. The discriminator is unable to differentiate between
the two distributions, i.e. $D(x) = \frac{1}{2}$.

### 4.1 Global Optimality of $p_g = p_{data}$

We first consider the optimal discriminator $D$ for any given generator $G$.

我们首先考虑任何给定生成器 $G$ 的最佳鉴别器 $D$。

**Proposition 1.** *For $G$ fixed, the optimal discriminator $D$ is*

**命题 1.** *对于固定的 $G$，最优鉴别器 $D$ 是*

$$
D^*_G(x) = \frac{p_{data}(x) }{p_{data} (x) + p_g (x)}\tag{2} 
$$

*Proof.* The training criterion for the discriminator $D$, given any generator 
$G$, is to maximize the quantity $V(G, D)$

*证明。* 给定任何生成器 $G$，判别器 $D$ 的训练标准是最大化 $V(G, D)$ 的数量

$$
\begin{equation}
\begin{align*}
V(G,D) &= \int_x p_{data}(x){\rm log}(D(x))dx + \int_zp_z(z){\log}(1-D(g(z)))dz\\
&=\int_x p_{data}(x){\rm log}(D(x))+p_g(x)\log(1-D(x))dx 
\end{align*}
\end{equation}
\tag{3}
$$

For any $(a,b) \isin \mathbb{R}^2 \setminus \{0,0\}$, the function $y \rightarrow a \ {\rm log}(y) + b {\rm log}(1 — y)$ achieves its maximum in $[0,1]$ at $\frac{a}{a+b}$. The discriminator does not need to be defined outside of $Supp(p_{data}) \cup Supp(p_g)$, concluding the proof.

对于任意 $(a,b) \isin \mathbb{R}^2 \setminus \{0,0\}$，函数 $y \rightarrow a \ {\rm log}(y) + b {\rm log }(1 - y)$ 在 $\frac{a}{a+b}$ 处达到 $[0,1]$ 的最大值。 鉴别器不需要在 $Supp(p_{data}) \cup Supp(p_g)$ 之外定义，证明结束。

Note that the training objective for $D$ can be interpreted as maximizing the log-likelihood for estimating the conditional probability $P(Y = y|x)$, where $Y$ indicates whether $x$ comes from $p_{data}$ (with $y = 1$) or from $p_g$ (with $y$ = 0). The minimax game in Eq. 1 can now be reformulated as:

注意$D$的训练目标可以解释为最大化估计条件概率$P(Y=y|x)$的对数可能性，其中$Y$表示$x$是来自$p_{data}$（$y=1$）还是来自$p_g$（$y$=0）。现在，公式1中的最小化博弈可以重新表述为。

$$
\begin{equation}
\begin{align*}
C(G) & = \max_D V(G,D) \\
& = \mathbb{E}_{x \sim p_{data}}[\log D^*_G(x)] +\mathbb{E}_{x \sim z}[\log (1-D^*_G(G(z)))] \\
& = \mathbb{E}_{x \sim p_{data}}[\log D^*_G(x)] +\mathbb{E}_{x \sim z}[\log (1-D^*_G(x))] \\
&= \mathbb{E}_{x \sim p_{data}}[\log \frac{p_{data}(x) }{p_{data} (x) + p_g (x)}] +\mathbb{E}_{x \sim z}[\log \frac{p_{data}(x) }{p_{data} (x) + p_g (x)}]
\end{align*}
\end{equation}
\tag{4}
$$
 
**Theorem 1.** *The global minimum of the virtual training criterion $C(G)$ is achieved if and only if $p_g = p_{data}$. At that point, $C(G)$ achieves the value - $\log 4$.*

*Proof.* For $p_g = p_{data}$, $D^*_G (x) =\frac{1}{2} $, (consider Eq. 2). Hence, by inspecting Eq. 4 at $D^*_G (x) = \frac{1}{2}$, we find $C(G) = \log \frac{1}{2} + \log \frac{1}{2} = - \log 4$. To see that this is the best possible value of $C(G)$, reached only for $p_g = p_{data}$, observe that

$$
\mathbb{E}_{x \sim p_{data}} [— \log 2] + \mathbb{E}_{x \sim p_{g}}[— \log 2] = — \log 4 
$$
and that by subtracting this expression from $C(G) = V (D^*_G,G)$, we obtain:

$$
C(G) = -\log(4) +KL(p_{data}\Vert  \frac{p_{data}+p_g}{2}) + KL(p_g \Vert \frac{p_{data}+p_g}{2}) \tag{5}
$$

where KL is the Kullback-Leibler divergence. We recognize in the previous expression the Jensen- Shannon divergence between the model’s distribution and the data generating process:

$$
C(G) = — \log(4) + 2 \cdot JSD (p_{data} \Vert p_g ) \tag{6}
$$

Since the Jensen-Shannon divergence between two distributions is always non-negative, and zero iff they are equal, we have shown that $C^* =-\log(4)$ is the global minimum of $C(G)$ and that the only solution is  $p_g = p_{data}$ , *i.e.*, the generative model perfectly replicating the data distribution.

### 4.2 Convergence of Algorithm 1

**Proposition 2**. *If $G$ and $D$ have enough capacity, and at each step of Algorithm 1, the discriminator is allowed to reach its optimum given $G$, and $p_g$ is updated so as to improve the criterion*

**提案 2**。 *如果$G$和$D$有足够的容量，在算法1的每一步，给定$G$让判别器达到最优，更新$p_g$以改进判据*

$\mathbb{E}_{x \sim p_{data}} [log D^*_G (x)] + \mathbb{E}_{x \sim p_g} [log(1 - D^*_G (x))]$
  
*then $p_g$ converges to $p_{data}$*

*Proof.* Consider $V (G,D) = U (p_g,D)$ as a function of $p_g$ as done in the above criterion. Note that $U(p_g, D)$ is convex in Pg. The subderivatives of a supremum of convex functions include the derivative of the function at the point where the maximum is attained. In other words, if $f (x) = sup_{\alpha \isin \mathcal{A}} f_{\alpha}(x)$ and $ f_{\alpha}(x)$ is convex in $x$ for every $\alpha$, then ${\partial f}_\beta(x) \isin {\partial f}$ if $\beta = arg \ {\rm sup}_{\alpha \isin \mathcal{A}} f_{\alpha}(x)$. This is equivalent to computing a gradient descent update for $p_g$ at the optimal $D$ given the corresponding $G$. ${\rm sup}_D U(p_g, D)$ is convex in $p_g$ with a unique global optima as proven in Thm 1, therefore with sufficiently small updates of $p_g$, $p_g$ converges to $p_x$, concluding the proof.

*证明。* 将 $V (G,D) = U (p_g,D)$ 视为 $p_g$ 的函数，如上述标准中所做的那样。 注意 $U(p_g, D)$ 在 Pg 中是凸的。 凸函数上确界的子导数包括函数在达到最大值的点的导数。 换句话说，如果 $f (x) = sup_{\alpha \isin \mathcal{A}} f_{\alpha}(x)$ 并且 $ f_{\alpha}(x)$ 在 $x$ 中是凸的 每个 $\alpha$，然后 ${\partial f}_\beta(x) \isin {\partial f}$ 如果 $\beta = arg \ {\rm sup}_{\alpha \isin \mathcal{A} } f_{\alpha}(x)$。 这相当于给定相应的 $G$，在最优 $D$ 处计算 $p_g$ 的梯度下降更新。 ${\rm sup}_D U(p_g, D)$ 在 $p_g$ 中是凸的，具有唯一的全局最优值，如 Thm 1 中所证明的那样，因此对于 $p_g$ 的足够小更新，$p_g$ 收敛到 $p_x$， 结束证明。

In practice, adversarial nets represent a limited family of $p_g$ distributions via the function $G(z; \theta_g)$, and we optimize $\theta_g$ rather than $p_g$ itself, so the proofs do not apply. However, the excellent performance of multilayer perceptrons in practice suggests that they are a reasonable model to use despite their lack of theoretical guarantees.

<div align =center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Generative-Adversarial-Nets/Table%201.png"/></div>

Table 1: Parzen window-based log-likelihood estimates. The reported numbers on MNIST are the mean log-likelihood of samples on test set, with the standard error of the mean computed across examples. On TFD, we computed the standard error across folds of the dataset, with a different $\sigma$ chosen using the validation set of each fold. On TFD, $\sigma$ was cross validated on each fold and mean log-likelihood on each fold were computed. For MNIST we compare against other models of the real-valued (rather than binary) version of dataset.

## 5 Experiments

We trained adversarial nets an a range of datasets including MNIST[21], the Toronto Face Database (TFD) [27], and CIFAR-10 [19]. The generator nets used a mixture of rectifier linear activations [17, 8] and sigmoid activations, while the discriminator net used maxout [9] activations. Dropout [16] was applied in training the discriminator net. While our theoretical framework permits the use of dropout and other noise at intermediate layers of the generator, we used noise as the input to only the bottommost layer of the generator network.

We estimate probability of the test set data under $p_g$ by fitting a Gaussian Parzen window to the samples generated with $G$ and reporting the log-likelihood under this distribution. The parameter of the Gaussians was obtained by cross validation on the validation set. This procedure was introduced in Breuleux $et al$. [7] and used for various generative models for which the exact likelihood is not tractable [24,3,4]. Results are reported in Table 1. This method of estimating the likelihood has somewhat high variance and does not perform well in high dimensional spaces but it is the best method available to our knowledge. Advances in generative models that can sample but not estimate likelihood directly motivate further research into how to evaluate such models. In Figures 2 and 3 we show samples drawn from the generator net after training. While we make no claim that these samples are better than samples generated by existing methods, we believe that these samples are at least competitive with the better generative models in the literature and highlight the potential of the adversarial framework.

## 6 Advantages and disadvantages

This new framework comes with advantages and disadvantages relative to previous modeling frame-works. The disadvantages are primarily that there is no explicit representation of $p_g(x)$, and that $D$ must be synchronized well with $G$ during training (in particular, $G$ must not be trained too much without updating $D$, in order to avoid “the Helvetica scenario” in which $G$ collapses too many values of $\rm z$ to the same value of x to have enough diversity to model $p_{data}$), much as the negative chains of a Boltzmann machine must be kept up to date between learning steps. The advantages are that Markov chains are never needed, only backprop is used to obtain gradients, no inference is needed during learning, and a wide variety of functions can be incorporated into the model. Table 2 summarizes the comparison of generative adversarial nets with other generative modeling approaches.

The aforementioned advantages are primarily computational. Adversarial models may also gain some statistical advantage from the generator network not being updated directly with data examples, but only with gradients flowing through the discriminator. This means that components of the input are not copied directly into the generator’s parameters. Another advantage of adversarial networks is that they can represent very sharp, even degenerate distributions, while methods based on Markov chains require that the distribution be somewhat blurry in order for the chains to be able to mix between modes.

## 7 Conclusions and future work

This framework admits many straightforward extensions: 

<div align =center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Generative-Adversarial-Nets/Fig%202.png"/></div>

Figure 2: Visualization of samples from the model. Rightmost column shows the nearest training example of the neighboring sample, in order to demonstrate that the model has not memorized the training set. Samples are fair random draws, not cherry-picked. Unlike most other visualizations of deep generative models, these images show actual samples from the model distributions, not conditional means given samples of hidden units. Moreover, these samples are uncorrelated because the sampling process does not depend on Markov chain mixing. a) MNIST b) TFD c) CIFAR-10 (fully connected model) d) CIFAR-10 (convolutional discriminator and “deconvolutional” generator)

<div align =center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Generative-Adversarial-Nets/Fig%203.png"/></div>

Figure 3: Digits obtained by linearly interpolating between coordinates in $z$ space of the full model.

1. A *conditional generative* model $p(x | c)$ can be obtained by adding $c$ as input to both $G$ and $D$.
2. *Learned approximate inference* can be performed by training an auxiliary network to predict $z$ given $x$. This is similar to the inference net trained by the wake-sleep algorithm [15] but with the advantage that the inference net may be trained for a fixed generator net after the generator net has finished training.
3. One can approximately model all conditionals $p(x_S | x_{\not S})$ where $S$ is a subset of the indices of $x$ by training a family of conditional models that share parameters. Essentially, one can use adversarial nets to implement a stochastic extension of the deterministic MP-DBM [10].
4. *Semi-supervised learning*: features from the discriminator or inference net could improve performance of classifiers when limited labeled data is available.
5. *Efficiency* improvements: training could be accelerated greatly by devising better methods for coordinating $G$ and $D$ or determining better distributions to sample ${\rm z}$ from during training.

This paper has demonstrated the viability of the adversarial modeling framework, suggesting that these research directions could prove useful.

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Generative-Adversarial-Nets/Table%202.png"/></div>
Table 2: Challenges in generative modeling: a summary of the difficulties encountered by different approaches to deep generative modeling for each of the major operations involving a model.

## Acknowledgments

We would like to acknowledge Patrice Marcotte, Olivier Delalleau, Kyunghyun Cho, Guillaume Alain and Jason Yosinski for helpful discussions. Yann Dauphin shared his Parzen window evaluation code with us. We would like to thank the developers of Pylearn2 [11] and Theano [6, 1], particularly Frédéric Bastien who rushed a Theano feature specifically to benefit this project. Arnaud Bergeron provided much-needed support with LATEX typesetting. We would also like to thank CIFAR, and Canada Research Chairs for funding, and Compute Canada, and Calcul Québec for providing computational resources. Ian Goodfellow is supported by the 2013 Google Fellowship in Deep Learning. Finally, we would like to thank Les Trois Brasseurs for stimulating our creativity.

## References

[1] Bastien, F., Lamblin, P., Pascanu, R., Bergstra, J., Goodfellow, I. J., Bergeron, A., Bouchard, N., and Bengio, Y. (2012). Theano: new features and speed improvements. Deep Learning and Unsupervised Feature Learning NIPS 2012 Workshop.
[2] Bengio, Y. (2009). Learning deep architectures for AI. Now Publishers.
[3] Bengio, Y., Mesnil, G., Dauphin, Y., and Rifai, S. (2013). Better mixing via deep representations. In ICML’13.
[4] Bengio, Y., Thibodeau-Laufer, E., and Yosinski, J. (2014a). Deep generative stochastic networks trainable by backprop. In ICML’14.
[5] Bengio, Y., Thibodeau-Laufer, E., Alain, G., and Yosinski, J. (2014b). Deep generative stochastic networks trainable by backprop. In Proceedings of the 30th International Conference on Machine Learning (ICML’14).
[6] Bergstra, J., Breuleux, O., Bastien, F., Lamblin, P., Pascanu, R., Desjardins, G., Turian, J., Warde-Farley, D., and Bengio, Y. (2010). Theano: a CPU and GPU math expression compiler. In Proceedings of the Python for Scientific Computing Conference (SciPy). Oral Presentation.
[7] Breuleux, O., Bengio, Y., and Vincent, P. (2011). Quickly generating representative samples from an RBM-derived process. Neural Computation,23(8), 2053-2073.
[8] Glorot, X., Bordes, A., and Bengio, Y. (2011). Deep sparse rectifier neural networks. In AISTATS’2011.
[9] Goodfellow, I. J., Warde-Farley, D., Mirza, M., Courville, A., and Bengio, Y. (2013a). Maxout networks. In ICML’2013.
[10] Goodfellow, I. J., Mirza, M., Courville, A., and Bengio, Y. (2013b). Multi-prediction deep Boltzmann machines. In NIPS’2013.
[11] Goodfellow, I. J., Warde-Farley, D., Lamblin, P., Dumoulin, V., Mirza, M., Pascanu, R., Bergstra, J., Bastien, F., and Bengio, Y. (2013c). Pylearn2: a machine learning research library. arXiv preprint arXiv:1308.4214.
[12] Gregor, K., Danihelka, I., Mnih, A., Blundell, C., and Wierstra, D. (2014). Deep autoregressive networks. In ICML’2014.
[13] Gutmann, M. and Hyvarinen, A. (2010). Noise-contrastive estimation: A new estimation principle for unnormalized statistical models. In Proceedings of The Thirteenth International Conference on Artificial Intelligence and Statistics (AISTATS’10).
[14] Hinton, G., Deng, L., Dahl, G. E., Mohamed, A., Jaitly, N., Senior, A., Vanhoucke, V., Nguyen, P., Sainath, T., and Kingsbury, B. (2012a). Deep neural networks for acoustic modeling in speech recognition. IEEE Signal Processing Magazine, 29(6), 82-97.
[15] Hinton, G. E., Dayan, P., Frey, B. J., and Neal, R. M. (1995). The wake-sleep algorithm for unsupervised neural networks. Science, 268, 1558-1161.
[16] Hinton, G. E., Srivastava, N., Krizhevsky, A., Sutskever, I., and Salakhutdinov, R. (2012b). Improving neural networks by preventing co-adaptation of feature detectors. Technical report, arXiv:1207.0580.
[17] Jarrett, K., Kavukcuoglu, K., Ranzato, M., and LeCun, Y. (2009). What is the best multi-stage architecture for object recognition? In Proc. International Conference on Computer Vision (ICCV’09), pages 2146-2153. IEEE.
[18] Kingma, D. P. and Welling, M. (2014). Auto-encoding variational bayes. In Proceedings of the International Conference on Learning Representations (ICLR).
[19] Krizhevsky, A. and Hinton, G. (2009). Learning multiple layers of features from tiny images. Technical report, University of Toronto.
[20] Krizhevsky, A., Sutskever, I., and Hinton, G. (2012). ImageNet classification with deep convolutional neural networks. In NIPS’2012.
[21] LeCun, Y., Bottou, L., Bengio, Y., and Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.
[22] Mnih, A. and Gregor, K. (2014). Neural variational inference and learning in belief networks. Technical report, arXiv preprint arXiv:1402.0030.
[23] Rezende, D. J., Mohamed, S., and Wierstra, D. (2014). Stochastic backpropagation and approximate inference in deep generative models. Technical report, arXiv:1401.4082.
[24] Rifai, S., Bengio, Y., Dauphin, Y., and Vincent, P. (2012). A generative process for sampling contractive auto-encoders. In ICML’12.
[25] Salakhutdinov, R. and Hinton, G. E. (2009). Deep Boltzmann machines. In AISTATS’2009, pages 448455.
[26] Schmidhuber, J. (1992). Learning factorial codes by predictability minimization. Neural Computation, 4(6), 863-879.
[27] Susskind, J., Anderson, A., and Hinton, G. E. (2010). The Toronto face dataset. Technical Report UTML TR 2010-001, U. Toronto.
[28] Szegedy, C., Zaremba, W., Sutskever, I., Bruna, J., Erhan, D., Goodfellow, I. J., and Fergus, R. (2014). Intriguing properties of neural networks. ICLR, abs/1312.6199.
[29] Tu, Z. (2007). Learning generative models via discriminative approaches. In Computer Vision and Pattern Recognition, 2007. CVPR’07. IEEE Conference on, pages 1-8. IEEE.
