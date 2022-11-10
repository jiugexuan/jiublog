---
title: 【论文】Deep Residual Learning for Image Recognition 图像识别中的深度残差学习（Resnet）
date: 2022-10-13 07:00:00 +/-0800
categories: [论文,深度学习]
tags: [深度学习]     # TAG names should always be lowercase 标记名称应始终为小写
math: true
---

## Abstract 摘要

Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth. On the ImageNet dataset we evaluate residual nets with a depth of up to 152 layers—8 deeper than VGG nets [41] but still having lower complexity. An ensemble of these residual nets achieves 3.57% error on the ImageNet test set. This result won the 1st place on the ILSVRC 2015 classification task. We also present analysis on CIFAR-10 with 100 and 1000 layers.

更深层次的神经网络更难训练。我们提出了一个残差学习框架，以简化比以前使用的网络更深的网络培训。我们明确地将层重新定义为参考层输入的学习剩余函数，而不是学习未参考的函数。我们提供了全面的经验证据，表明这些剩余网络更容易优化，并且可以从大幅增加的深度中获得精度。在ImageNet数据集上，我们评估了深度高达152层的残差网络，比VGG网络深8层[41]，但复杂性仍然较低。这些残差网络的集合在ImageNet测试集上达到3.57%的误差。该结果在ILSVRC 2015分类任务中获得第一名。我们还对具有100和1000层的CIFAR-10进行了分析。

The depth of representations is of central importance for many visual recognition tasks. Solely due to our extremely deep representations, we obtain a 28% relative improvement on the COCO object detection dataset. Deep residual nets are foundations of our submissions to ILSVRC & COCO 2015 competitions $^1$, where we also won the 1st places on the tasks of ImageNet detection, ImageNet localization, COCO detection, and COCO segmentation.

特征的深度对于许多视觉识别任务至关重要。仅由于我们的深度表示，我们在COCO对象检测数据集上获得了28%的相对改进。深度残差网是我们提交给ILSVRC和COCO 2015竞赛的基础，在这些竞赛中，我们还赢得了ImageNet检测、ImageNet-定位、COCO检测和COCO分割任务的第一名。

>$^1$ http://image-net.org/challenges/LSVRC/2015/ and http://mscoco.org/dataset/#detections-challenge2015.

## 1. Introduction 导言

Deep convolutional neural networks [22,21] have led to a series of breakthroughs for image classification [21,50,40]. Deep networks naturally integrate low/mid/high- level features [50] and classifiers in an end-to-end multilayer fashion, and the “levels” of features can be enriched by the number of stacked layers (depth). Recent evidence [41, 44] reveals that network depth is of crucial importance, and the leading results [41,44,13,16] on the challenging ImageNet dataset [36] all exploit “very deep” [41] models, with a depth of sixteen [41] to thirty [16]. Many other nontrivial visual recognition tasks [8,12,7,32,27] have also greatly benefited from very deep models.

深度卷积神经网络[22,21]在图像分类方面取得了一系列突破[21,50,40]。深层网络以端到端的多层方式自然地将低/中/高层次特征[50]和分类器集成在一起，并且特征的“层次”可以通过堆叠层的数量（深度）来丰富。最近的证据[41，44]表明，网络深度至关重要，关于富有挑战性的ImageNet数据集[36]的领先结果[41，44，13,16]都利用了“非常深”[41]模型，深度为十六[41]到三十[16]。许多其他非平凡的视觉识别任务[8,12,7,32,27]也从非常深入的模型中受益匪浅。

Driven by the significance of depth, a question arises: Is learning better networks as easy as stacking more layers? An obstacle to answering this question was the notorious problem of vanishing/exploding gradients [1,9], which hamper convergence from the beginning. This problem, however, has been largely addressed by normalized initialization [23, 9, 37, 13] and intermediate normalization layers [16], which enable networks with tens of layers to start con-verging for stochastic gradient descent (SGD) with back-propagation [22].

在深度意义的驱动下，出现了一个问题。学习更好的网络是否就像堆叠更多的层一样容易？回答这个问题的一个障碍是臭名昭著的梯度消失/膨胀问题[1,9]，它从一开始就阻碍了收敛。然而，这个问题在很大程度上已经被归一化初始化[23, 9, 37, 13]和中间归一化层[16]所解决，这使得具有几十层的网络能够开始收敛以实现随机梯度下降（SGD）与反向传播[22]。

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Deep-Residual-Learning-for-Image-Recognition/Fig%201.png"/></div>
Figure 1. Training error (left) and test error (right) on CIFAR-10 with 20-layer and 56-layer “plain” networks. The deeper network has higher training error, and thus test error. Similar phenomena on ImageNet is presented in Fig. 4.
图 1. 具有 20 层和 56 层“普通”网络的 CIFAR-10 上的训练错误（左）和测试错误（右）。 更深的网络有更高的训练误差，因此也有测试误差。 ImageNet 上的类似现象如图 4 所示。

When deeper networks are able to start converging, a degradation problem has been exposed: with the network depth increasing, accuracy gets saturated (which might be unsurprising) and then degrades rapidly. Unexpectedly, such degradation *is not caused by overfitting*, and adding more layers to a suitably deep model leads to higher training error, as reported in [11, 42] and thoroughly verified by our experiments. Fig. 1 shows a typical example.

当更深的网络能够开始收敛时，一个退化的问题已经暴露出来：随着网络深度的增加，准确率会达到饱和（这可能并不令人惊讶），然后迅速退化。出乎意料的是，这种退化并不是由*过拟合引起的*，在一个合适的深度模型上增加更多的层会导致更高的训练误差，这在文献[11, 42]中有所报道，并被我们的实验所彻底验证。图1显示了一个典型的例子

The degradation (of training accuracy) indicates that not all systems are similarly easy to optimize. Let us consider a shallower architecture and its deeper counterpart that adds more layers onto it. There exists a solution by construction to the deeper model: the added layers are identity mapping, and the other layers are copied from the learned shallower model. The existence of this constructed solution indicates that a deeper model should produce no higher training error than its shallower counterpart. But experiments show that our current solvers on hand are unable to find solutions that are comparably good or better than the constructed solution (or unable to do so in feasible time).

训练精度的下降表明，并非所有的系统都同样容易优化。让我们考虑一个较浅的架构和在其上添加更多层的较深的对应物。深层模型存在一个构造上的解决方案：增加的层是身份映射，而其他的层是从学到的浅层模型中复制的。这种构造性解决方案的存在表明，较深的模型应该不会产生比较浅的模型更高的训练误差。但实验表明，我们目前手头的求解器无法找到比构建的解决方案好或更好的解决方案（或者无法在可行的时间内做到这一点）。

>identify mapping x映射x,即输入等于输出，即这些层输入等于输出，什么也不做

In this paper, we address the degradation problem by introducing *a deep residual learning framework*. Instead of hoping each few stacked layers directly fit a desired underlying mapping, we explicitly let these layers fit a residual mapping. Formally, denoting the desired underlying mapping as $\mathcal{H}(\rm x)$, we let the stacked nonlinear layers fit another mapping of $\mathcal{F}(\rm x) := \mathcal{H}(\rm x) — \rm x$. The original mapping is recast into $\mathcal{F}(\rm x) +\rm x$. We hypothesize that it is easier to optimize the residual mapping than to optimize the original, unreferenced mapping. To the extreme, if an identity mapping were optimal, it would be easier to push the residual to zero than to fit an identity mapping by a stack of nonlinear layers.

在本文中，我们通过引入*深度残差学习框架*来解决退化问题。 我们不是希望每个堆叠的层直接适合所需的底层映射，而是明确地让这些层适合残差映射。 形式上，将所需的底层映射表示为 $\mathcal{H}(\rm x)$，我们让堆叠的非线性层拟合 $\mathcal{F}(\rm x) 的另一个映射：= \mathcal{H}( \rm x) — \rm x$。 原始映射被重铸为 $\mathcal{F}(\rm x) +\rm x$。 我们假设优化残差映射比优化原始的、未引用的映射更容易。 极端情况下，如果恒等映射是最优的，则将残差推至零要比通过一堆非线性层拟合恒等映射更容易。

> 新加的层学习真实的值和学到东西之间的残差

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Deep-Residual-Learning-for-Image-Recognition/Fig%202.png"/></div>

Figure 2. Residual learning: a building block. 
图2. 剩余学习：一个构件。

The formulation of  $\mathcal{F}(\rm x) +\rm x$ can be realized by feedforward neural networks with “shortcut connections” (Fig. 2). Shortcut connections [2,34,49] are those skipping one or more layers. In our case, the shortcut connections simply perform *identity* mapping, and their outputs are added to the outputs of the stacked layers (Fig. 2). Identity shortcut connections add neither extra parameter nor computational complexity. The entire network can still be trained end-to-end by SGD with backpropagation, and can be easily implemented using common libraries ($e.g.$, Caffe [19]) without modifying the solvers.

$/mathcal{F}(\rm x) +\rm x$的表述可以通过带有 "捷径连接 "的前馈神经网络来实现（图2）。快捷连接[2,34,49]是指跳过一个或多个层的连接。在我们的例子中，捷径连接只是进行*身份*映射，其输出被添加到堆叠层的输出中（图2）。身份捷径连接既不增加额外的参数，也不增加计算的复杂性。整个网络仍然可以通过SGD与反向传播进行端到端的训练，并且可以在不修改求解器的情况下，使用常见的库（$e.g.$，Caffe[19]）轻松实现。

We present comprehensive experiments on ImageNet [36] to show the degradation problem and evaluate our method. We show that: 1) Our extremely deep residual nets are easy to optimize, but the counterpart “plain” nets (that simply stack layers) exhibit higher training error when the depth increases; 2) Our deep residual nets can easily enjoy accuracy gains from greatly increased depth, producing results substantially better than previous networks.

我们在ImageNet[36]上进行了综合实验，以显示退化问题并评估我们的方法。我们表明。1）我们的极深的残差网很容易优化，但对应的 "普通 "网（简单地堆叠层）在深度增加时表现出更高的训练误差；2）我们的深残差网很容易享受到深度大大增加带来的精度提升，产生的结果大大优于以前的网络。

Similar phenomena are also shown on the CIFAR-10 set [20], suggesting that the optimization difficulties and the effects of our method are not just akin to a particular dataset. We present successfully trained models on this dataset with over 100 layers, and explore models with over 1000 layers.

CIFAR-10 集 [20] 上也显示了类似的现象，这表明我们的方法的优化困难 [的解决] 和效果不仅仅类似于特定的数据集。 我们在这个数据集上展示了超过 100 层的成功训练模型，并探索了超过 1000 层的模型。

On the ImageNet classification dataset [36], we obtain excellent results by extremely deep residual nets. Our 152- layer residual net is the deepest network ever presented on ImageNet, while still having lower complexity than VGG nets [41]. Our ensemble has 3.57% top-5 error on the ImageNet test set, and won the 1st place in the ILSVRC 2015 classification competition. The extremely deep representations also have excellent generalization performance on other recognition tasks, and lead us to further win the 1st places on: ImageNet detection, ImageNet localization, COCO detection, and COCO segmentation in ILSVRC & COCO 2015 competitions. This strong evidence shows that the residual learning principle is generic, and we expect that it is applicable in other vision and non-vision problems.

在 ImageNet 分类数据集 [36] 上，我们通过极深的残差网络获得了出色的结果。 我们的 152 层残差网络是 ImageNet 上有史以来最深的网络，同时仍然比 VGG 网络 [41] 具有更低的复杂性。 我们的集成在 ImageNet 测试集上有 3.57% 的 top-5 错误，并在 ILSVRC 2015 分类竞赛中获得第一名。 极深的表示在其他识别任务上也具有出色的泛化性能，并带领我们在 ILSVRC & COCO 2015 竞赛中进一步赢得了 ImageNet 检测、ImageNet 定位、COCO 检测和 COCO 分割的第一名。 这一强有力的证据表明，残差学习原理是通用的，我们希望它适用于其他视觉和非视觉问题。

## 2. Related Work 相关工作

**Residual Representations**. In image recognition, VLAD [18] is a representation that encodes by the residual vectors with respect to a dictionary, and Fisher Vector [30] can be formulated as a probabilistic version [18] of VLAD. Both of them are powerful shallow representations for image retrieval and classification [4,48]. For vector quantization, encoding residual vectors [17] is shown to be more effective than encoding original vectors.

**剩余陈述**。 在图像识别中，VLAD [18] 是一种由残差向量相对于字典进行编码的表示，Fisher Vector [30] 可以表述为 VLAD 的概率版本 [18]。 它们都是用于图像检索和分类的强大的浅层表示[4,48]。 对于向量量化，编码残差向量[17]被证明比编码原始向量更有效。

In low-level vision and computer graphics, for solving Partial Differential Equations (PDEs), the widely used Multigrid method [3] reformulates the system as subproblems at multiple scales, where each subproblem is responsible for the residual solution between a coarser and a finer scale. An alternative to Multigrid is hierarchical basis preconditioning [45, 46], which relies on variables that represent residual vectors between two scales. It has been shown [3,45,46] that these solvers converge much faster than standard solvers that are unaware of the residual nature of the solutions. These methods suggest that a good reformulation or preconditioning can simplify the optimization.

**Shortcut Connections.** Practices and theories that lead to shortcut connections [2,34,49] have been studied for a long time. An early practice of training multi-layer perceptrons (MLPs) is to add a linear layer connected from the network input to the output [34,49]. In [44,24], a few intermediate layers are directly connected to auxiliary classifiers for addressing vanishing/exploding gradients. The papers of [39,38,31,47] propose methods for centering layer responses, gradients, and propagated errors, implemented by shortcut connections. In [44], an “inception” layer is composed ofa shortcut branch and a few deeper branches.

**快捷连接。** 导致快捷连接的实践和理论 [2,34,49] 已经研究了很长时间。 训练多层感知器 (MLP) 的早期实践是添加一个从网络输入连接到输出的线性层 [34,49]。 在 [44,24] 中，一些中间层直接连接到辅助分类器以解决消失/爆炸梯度。 [39,38,31,47] 的论文提出了通过快捷连接来实现层响应、梯度和传播误差的中心化方法。 在[44]中，一个“inception”层由一个快捷分支和一些更深的分支组成。

Concurrent with our work, “highway networks” [42,43] present shortcut connections with gating functions [15]. These gates are data-dependent and have parameters, in contrast to our identity shortcuts that are parameter-free. When a gated shortcut is “closed” (approaching zero), the layers in highway networks represent non-residual functions. On the contrary, our formulation always learns residual functions; our identity shortcuts are never closed, and all information is always passed through, with additional residual functions to be learned. In addition, highway networks have not demonstrated accuracy gains with extremely increased depth (e.g., over 100 layers).

与我们的工作同时，“highway networks”[42,43] 提供了与门控功能 [15] 的快捷连接。 与我们的无参数身份快捷方式相比，这些门依赖于数据并具有参数。 当门控捷径“关闭”（接近零）时，高速公路网络中的层表示非残差函数。 相反，我们的公式总是学习残差函数； 我们的身份捷径永远不会关闭，所有信息总是通过，还有额外的残差函数需要学习。 此外，高速公路网络没有表现出随着深度的极大增加（例如，超过 100 层）的准确性提高。

## 3. Deep Residual Learning

### 3.1. Residual Learning

Let us consider $\mathcal{H}(\rm x)$ as an underlying mapping to be fit by a few stacked layers (not necessarily the entire net), with $\rm x$ denoting the inputs to the first of these layers. If one hypothesizes that multiple nonlinear layers can asymptotically approximate complicated functions $^2$, then it is equivalent to hypothesize that they can asymptotically approximate the residual functions, $i.e.$, $\mathcal{H}(\rm x) — \rm x$ (assuming that the input and output are of the same dimensions). So rather than expect stacked layers to approximate $\mathcal{H}(\rm x)$, we explicitly let these layers approximate a residual function $\mathcal{F}(\rm x) := \mathcal{H}(\rm x)- \rm x$. The original function thus becomes $\mathcal{F}(\rm x) + \rm x$. Although both forms should be able to asymptotically approximate the desired functions (as hypothesized), the ease of learning might be different.

>$^2$ This hypothesis, however, is still an open question. See [28].

This reformulation is motivated by the counterintuitive phenomena about the degradation problem (Fig. 1, left). As we discussed in the introduction, if the added layers can be constructed as identity mappings, a deeper model should have training error no greater than its shallower counterpart. The degradation problem suggests that the solvers might have difficulties in approximating identity mappings by multiple nonlinear layers. With the residual learning reformulation, if identity mappings are optimal, the solvers may simply drive the weights of the multiple nonlinear layers toward zero to approach identity mappings.

In real cases, it is unlikely that identity mappings are optimal, but our reformulation may help to precondition the problem. If the optimal function is closer to an identity mapping than to a zero mapping, it should be easier for the solver to find the perturbations with reference to an identity mapping, than to learn the function as a new one. We show by experiments (Fig. 7) that the learned residual functions in general have small responses, suggesting that identity map-pings provide reasonable preconditioning.

### 3.2. Identity Mapping by Shortcuts

We adopt residual learning to every few stacked layers. A building block is shown in Fig. 2. Formally, in this paper we consider a building block defined as:

$${\rm y} = \mathcal{F}({\rm x}, \{W_i\}) + \rm x. \tag{1}$$

Here $\rm x$ and $\rm y$ are the input and output vectors of the layers considered. The function $\mathcal{F}({\rm x}, \{W_i\})$ represents the residual mapping to be learned. For the example in Fig. 2 that has two layers, $\mathcal{F}= W_2 \sigma(W_1 {\rm x}) $ in which $\sigma$ denotes ReLU [29] and the biases are omitted for simplifying notations. The operation $\mathcal{F} + \rm x$ is performed by a shortcut connection and element-wise addition. We adopt the second nonlinearity after the addition ($i.e.$, $\sigma({\rm y})$, see Fig. 2).

The shortcut connections in Eqn.(1) introduce neither ex-tra parameter nor computation complexity. This is not only attractive in practice but also important in our comparisons between plain and residual networks. We can fairly compare plain/residual networks that simultaneously have the same number of parameters, depth, width, and computational cost (except for the negligible element-wise addition).

The dimensions of x and F must be equal in Eqn.(1). If this is not the case (e.g., when changing the input/output channels), we can perform a linear projection Ws by the shortcut connections to match the dimensions:

$${\rm y} = F({\rm x}; \{W_i\}) + W_s{\rm x} \tag{2}$$

We can also use a square matrix $W_s$ in Eqn.(1). But we will show by experiments that the identity mapping is sufficient for addressing the degradation problem and is economical, and thus $W_s$ is only used when matching dimensions.

The form of the residual function $\mathcal{F}$ is flexible. Experiments in this paper involve a function $\mathcal{F}$ that has two or three layers (Fig. 5), while more layers are possible. But if $\mathcal{F}$ has only a single layer, Eqn.(1) is similar to a linear layer: ${\rm y} = W_1{\rm x} + {\rm x}$, for which we have not observed advantages.

We also note that although the above notations are about fully-connected layers for simplicity, they are applicable to convolutional layers. The function $F({\rm x}; \{W_i\})$ can represent multiple convolutional layers. The element-wise addition is performed on two feature maps, channel by channel.

### 3.3. Network Architectures

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Deep-Residual-Learning-for-Image-Recognition/Fig%203.png"/></div>

Figure 3. Example network architectures for ImageNet. **Left**: the VGG-19 model [41] (19.6 billion FLOPs) as a reference. **Middle**: a plain network with 34 parameter layers (3.6 billion FLOPs).Right: a residual network with 34 parameter layers (3.6 billion FLOPs). The dotted shortcuts increase dimensions. **Table 1** shows more details and other variants.

We have tested various plain/residual nets, and have ob-served consistent phenomena. To provide instances for dis-cussion, we describe two models for ImageNet as follows.
Plain Network. Our plain baselines (Fig. 3, middle) are mainly inspired by the philosophy of VGG nets [41] (Fig. 3, left). The convolutional layers mostly have 33 filters and follow two simple design rules: (i) for the same output feature map size, the layers have the same number of filters; and (ii) if the feature map size is halved, the number of filters is doubled so as to preserve the time complexity per layer. We perform downsampling directly by convolutional layers that have a stride of 2. The network ends with a global average pooling layer and a 1000-way fully-connected layer with softmax. The total number of weighted layers is 34 in Fig. 3 (middle).

It is worth noticing that our model has fewer filters and lower complexity than VGG nets [41] (Fig. 3, left). Our 34- layer baseline has 3.6 billion FLOPs (multiply-adds), which is only 18% of VGG-19 (19.6 billion FLOPs).

**Residual Network.** Based on the above plain network, we insert shortcut connections (Fig. 3, right) which turn the network into its counterpart residual version. The identity shortcuts (Eqn.(1)) can be directly used when the input and output are of the same dimensions (solid line shortcuts in Fig. 3). When the dimensions increase (dotted line shortcuts in Fig. 3), we consider two options: (A) The shortcut still performs identity mapping, with extra zero entries padded for increasing dimensions. This option introduces no extra parameter; (B) The projection shortcut in Eqn.(2) is used to match dimensions (done by $1 \times 1$ convolutions). For both options, when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2.

**剩余网络。**基于上述普通网络，我们插入快捷连接（图3，右），将网络转换为对应的剩余版本。当输入和输出的尺寸相同时，可以直接使用标识快捷键（等式（1））（图3中的实线快捷键）。当尺寸增加时（图3中的虚线快捷方式），我们考虑两个选项：（A）快捷方式仍然执行标识映射，为增加尺寸填充额外的零条目。此选项不引入额外参数；（B） 等式（2）中的投影快捷键用于匹配尺寸（由$1 \times 1$个卷积完成）。对于这两个选项，当快捷方式穿过两种尺寸的要素地图时，将以2的步幅执行。

### 3.4. Implementation 实现

Our implementation for ImageNet follows the practice in [21,41]. The image is resized with its shorter side randomly sampled in [256; 480] for scale augmentation [41]. A $224 \times 224$ crop is randomly sampled from an image or its horizontal flip, with the per-pixel mean subtracted [21]. The standard color augmentation in [21] is used. We adopt batch normalization (BN) [16] right after each convolution and before activation, following [16]. We initialize the weights as in [13] and train all plain/residual nets from scratch. We use SGD with a mini-batch size of $256$. The learning rate starts from $0.1$ and is divided by 10 when the error plateaus, and the models are trained for up to $60 \times 10^4$ iterations. We use a weight decay of $0.0001$ and a momentum of 0.9. We do not use dropout [14], following the practice in [16].

我们对ImageNet的实现遵循了[21,41]中的做法。图像被调整大小，其较短的一面在[256; 480]中随机取样，用于比例增强[41]。$224 \times 224$的裁剪是从图像或其水平翻转中随机采样的，并减去每像素的平均值[21]。使用[21]中的标准颜色增强。在每次卷积后和激活前，我们都采用批量归一化（BN）[16]，遵循[16]。我们按照[13]的方法初始化权重，并从头开始训练所有的普通/残余网。我们使用SGD，迷你批次大小为$256$。学习率从$0.1$开始，当误差趋于平稳时除以10，模型训练$60 \times 10^4$次迭代。我们使用$0.0001$的权重衰减和0.9的动量。我们不使用dropout[14]，遵循[16]的做法。

In testing, for comparison studies we adopt the standard 10-crop testing [21]. For best results, we adopt the fully-convolutional form as in [41,13], and average the scores at multiple scales (images are resized such that the shorter side is in {224; 256; 384; 480; 640}).

在测试中，对于比较研究，我们采用标准的 10-crop测试 [21]。 为了获得最佳结果，我们采用 [41,13] 中的全卷积形式，并在多个尺度上平均分数（图像被调整大小，使得短边在 {224；256；384；480；640)}。

## 4. Experiments 实验

### 4.1. ImageNet Classification

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Deep-Residual-Learning-for-Image-Recognition/Fig4.png"/></div>

Figure 4. Training on **ImageNet**. Thin curves denote training error, and bold curves denote validation error of the center crops. Left: plain networks of 18 and 34 layers. Right: ResNets of 18 and 34 layers. In this plot, the residual networks have no extra parameter compared to their plain counterparts.

We evaluate our method on the ImageNet 2012 classification dataset [36] that consists of 1000 classes. The models are trained on the 1.28 million training images, and evaluated on the 50k validation images. We also obtain a final result on the 100k test images, reported by the test server. We evaluate both top-1 and top-5 error rates.

**Plain Networks.** We first evaluate 18-layer and 34-layer plain nets. The 34-layer plain net is in Fig. 3 (middle). The 18-layer plain net is of a similar form. See Table 1 for detailed architectures.

**普通网络。** 我们首先评估 18 层和 34 层普通网络。 34 层平面网络如图 3（中）所示。 18层平网也是类似的形式。 有关详细架构，请参见表 1。

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Deep-Residual-Learning-for-Image-Recognition/Table1.png"/></div>
Table 1. Architectures for ImageNet. Building blocks are shown in brackets (see also Fig. 5), with the numbers of blocks stacked. Down-sampling is performed by conv3_1, conv4_1, and conv5_1 with a stride of 2. 
表1. ImageNet的架构。括号中显示的是构件（也见图5），其中的构件数量是叠加的。下采样由conv3_1、conv4_1和conv5_1执行，步长为2。

The results in Table 2 show that the deeper 34-layer plain net has higher validation error than the shallower 18-layer plain net. To reveal the reasons, in Fig. 4 (left) we compare their training/validation errors during the training procedure. We have observed the degradation problem - the 34-layer plain net has higher training error throughout the whole training procedure, even though the solution space of the 18-layer plain network is a subspace of that of the 34-layer one.

表 2 中的结果表明，较深的 34 层普通网络比较浅的 18 层普通网络具有更高的验证误差。 为了揭示原因，在图 4（左）中，我们比较了它们在训练过程中的训练/验证误差。 我们已经观察到退化问题——34 层普通网络在整个训练过程中具有更高的训练误差，尽管 18 层普通网络的解空间是 34 层网络的子空间。

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Deep-Residual-Learning-for-Image-Recognition/Table%202.png"/></div>
Table 2. Top-1 error (%, 10-crop testing) on ImageNet validation. Here the ResNets have no extra parameter compared to their plain counterparts. Fig. 4 shows the training procedures.
表 2. ImageNet 验证的 Top-1 错误（%，10-crop 测试）。 与普通的对应物相比，这里的 ResNets 没有额外的参数。 图 4 显示了训练过程。

We argue that this optimization difficulty is unlikely to be caused by vanishing gradients. These plain networks are trained with BN [16], which ensures forward propagated signals to have non-zero variances. We also verify that the backward propagated gradients exhibit healthy norms with BN. So neither forward nor backward signals vanish. In fact, the 34-layer plain net is still able to achieve competitive accuracy (Table 3), suggesting that the solver works to some extent. We conjecture that the deep plain nets may have exponentially low convergence rates, which impact the reducing of the training error $^3$. The reason for such optimization difficulties will be studied in the future.

>$^3$ We have experimented with more training iterations ($3×$) and still observed the degradation problem, suggesting that this problem cannot be feasibly addressed by simply using more iterations.

**Residual Networks.** Next we evaluate 18-layer and 34- layer residual nets (ResNets). The baseline architectures are the same as the above plain nets, expect that a shortcut connection is added to each pair of $3 \times 3$ filters as in Fig. 3 (right). In the first comparison (Table 2 and Fig. 4 right), we use identity mapping for all shortcuts and zero-padding for increasing dimensions (option A). So they have no extra parameter compared to the plain counterparts.

We have three major observations from Table 2 and Fig. 4. First, the situation is reversed with residual learning - the 34-layer ResNet is better than the 18-layer ResNet (by 2.8%). More importantly, the 34-layer ResNet exhibits considerably lower training error and is generalizable to the validation data. This indicates that the degradation problem is well addressed in this setting and we manage to obtain accuracy gains from increased depth.

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Deep-Residual-Learning-for-Image-Recognition/Table%203.png" height = 320/></div>
Table 3. Error rates (%, **10-crop** testing) on ImageNet validation.
VGG-16 is based on our test. ResNet-50/101/152 are of option B
that only uses projections for increasing dimensions.

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Deep-Residual-Learning-for-Image-Recognition/Table%204.png" height = 250/></div>
Table 4. Error rates (%) of single-model results on the ImageNet validation set (except y reported on the test set).

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Deep-Residual-Learning-for-Image-Recognition/Table%205.png" height = 200/></div>

Table 5. Error rates (%) of **ensembles**. The top-5 error is on the test set of ImageNet and reported by the test server.

Second, compared to its plain counterpart, the 34-layer ResNet reduces the top-1 error by 3.5% (Table 2), resulting from the successfully reduced training error (Fig. 4 right vs. left). This comparison verifies the effectiveness of residual learning on extremely deep systems.

Last, we also note that the 18-layer plain/residual nets are comparably accurate (Table 2), but the 18-layer ResNet converges faster (Fig. 4 right vs. left). When the net is “not overly deep” (18 layers here), the current SGD solver is still able to find good solutions to the plain net. In this case, the ResNet eases the optimization by providing faster convergence at the early stage.

**Identity vs. Projection Shortcuts.** We have shown that parameter-free, identity shortcuts help with training. Next we investigate projection shortcuts (Eqn.(2)). In Table 3 we compare three options: (A) zero-padding shortcuts are used for increasing dimensions, and all shortcuts are parameter-free (the same as Table 2 and Fig. 4 right); (B) projection shortcuts are used for increasing dimensions, and other shortcuts are identity; and (C) all shortcuts are projections.

**身份与投影捷径。**我们已经表明，无参数的身份捷径有助于训练。接下来我们研究投影捷径（公式（2））。在表3中，我们比较了三种选择：(A)零填充捷径用于增加维度，所有捷径都是无参数的（与表2和图4右侧相同）；(B)投影捷径用于增加维度，其他捷径为身份捷径；(C)所有捷径为投影捷径。

Table 3 shows that all three options are considerably bet-ter than the plain counterpart. B is slightly better than A. We argue that this is because the zero-padded dimensions in A indeed have no residual learning. C is marginally better than B, and we attribute this to the extra parameters introduced by many (thirteen) projection shortcuts. But the small differences among A/B/C indicate that projection shortcuts are not essential for addressing the degradation problem. So we do not use option C in the rest of this paper, to reduce mem- ory/time complexity and model sizes. Identity shortcuts are particularly important for not increasing the complexity of the bottleneck architectures that are introduced below.

**Deeper Bottleneck Architectures.** Next we describe our deeper nets for ImageNet. Because of concerns on the training time that we can afford, we modify the building block as a *bottleneck* design $^4$. For each residual function $\mathcal{F}$, we use a stack of 3 layers instead of 2 (Fig. 5). The three layers are $1 \times 1$, $3 \times 3$, and $1 \times 1$ convolutions, where the $1 \times 1$ layers are responsible for reducing and then increasing (restoring) dimensions, leaving the $3 \times 3$ layer a bottleneck with smaller input/output dimensions. Fig. 5 shows an example, where both designs have similar time complexity.

**突破更深的瓶颈架构。**接下来我们将描述我们用于ImageNet的更深的网络。由于对我们能承受的训练时间的关注，我们将构件修改为**突破瓶颈设计$^4$。对于每个残差函数$mathcal{F}$，我们使用3层的堆栈而不是2层（图5）。这三层是1元乘以1元、3元乘以3元和1元乘以1元的卷积，其中1元乘以1元的层负责减少，然后增加（恢复）维度，让3元乘以3元的层成为输入/输出维度较小的瓶颈。图5显示了一个例子，两种设计的时间复杂性相似。

>$^4$Deeper non-bottleneck ResNets (e.g., Fig. 5 left) also gain accuracy from increased depth (as shown on CIFAR-10), but are not as economical as the bottleneck ResNets. So the usage of bottleneck designs is mainly due to practical considerations. We further note that the degradation problem of plain nets is also witnessed for the bottleneck designs.

The parameter-free identity shortcuts are particularly important for the bottleneck architectures. If the identity short-cut in Fig. 5 (right) is replaced with projection, one can show that the time complexity and model size are doubled, as the shortcut is connected to the two high-dimensional ends. So identity shortcuts lead to more efficient models for the bottleneck designs.

**50-layer ResNet:** We replace each 2-layer block in the 34-layer net with this 3-layer bottleneck block, resulting in a 50-layer ResNet (Table 1). We use option B for increasing dimensions. This model has 3.8 billion FLOPs.

**101-layer and 152-layer ResNets:** We construct 101- layer and 152-layer ResNets by using more 3-layer blocks (Table 1). Remarkably, although the depth is significantly increased, the 152-layer ResNet (11.3 billion FLOPs) still has lower complexity than VGG-16/19 nets (15.3/19.6 billion FLOPs).

The 50/101/152-layer ResNets are more accurate than the 34-layer ones by considerable margins (Table 3 and 4). We do not observe the degradation problem and thus enjoy significant accuracy gains from considerably increased depth. The benefits of depth are witnessed for all evaluation metrics (Table 3 and 4).

**Comparisons with State-of-the-art Methods.** In Table 4 we compare with the previous best single-model results. Our baseline 34-layer ResNets have achieved very competitive accuracy. Our 152-layer ResNet has a single-model top-5 validation error of 4.49%. This single-model result outperforms all previous ensemble results (Table 5). We combine six models of different depth to form an ensemble (only with two 152-layer ones at the time of submitting). This leads to 3.57% top-5 error on the test set (Table 5). This entry won the 1st place in ILSVRC 2015.

### 4.2. CIFAR-10 and Analysis

We conducted more studies on the CIFAR-10 dataset [20], which consists of 50k training images and 10k testing images in 10 classes. We present experiments trained on the training set and evaluated on the test set. Our focus is on the behaviors of extremely deep networks, but not on pushing the state-of-the-art results, so we intentionally use simple architectures as follows.

The plain/residual architectures follow the form in Fig. 3 (middle/right). The network inputs are $32 \times 32$ images, with the per-pixel mean subtracted. The first layer is $3 \times 3$ convolutions. Then we use a stack of 6n layers with $3 \times 3$ convolutions on the feature maps of sizes {$32,16,8$} respectively, with $2n$ layers for each feature map size. The numbers of filters are {$16,32,64$} respectively. The subsampling is performed by convolutions with a stride of $2$. The network ends with a global average pooling, a 10-way fully-connected layer, and softmax. There are totally $6n+2$ stacked weighted layers. The following table summarizes the architecture:

|output map size | 32x32 |16x16 | 8x8 |
|---|---|---|---|---|
|# layers |1+2n| 2n| 2n|
|# filters |16 |32 |64|

When shortcut connections are used, they are connected to the pairs of $3 \times 3$ layers (totally $3n$ shortcuts). On this dataset we use identity shortcuts in all cases ($i.e.$, option A),so our residual models have exactly the same depth, width, and number of parameters as the plain counterparts.

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Deep-Residual-Learning-for-Image-Recognition/Table%206.png" height =350/></div>

Table 6. Classification error on the **CIFAR-10** test set. All methods are with data augmentation. For ResNet-110, we run it 5 times and show “best (mean $\pm$ std)” as in [43].

We use a weight decay of 0.0001 and momentum of $0.9$, and adopt the weight initialization in [13] and BN [16] but with no dropout. These models are trained with a minibatch size of 128 on two GPUs. We start with a learning rate of 0.1, divide it by 10 at 32k and 48k iterations, and terminate training at 64k iterations, which is determined on a 45k/5k train/val split. We follow the simple data augmentation in [24] for training: 4 pixels are padded on each side, and a  $32 \times32$ crop is randomly sampled from the padded image or its horizontal flip. For testing, we only evaluate the single view of the original $32 \times32$ image.

We compare $n = \{3, 5,7, 9\}$, leading to $20, 32, 44$, and 56-layer networks. Fig. 6 (left) shows the behaviors of the plain nets. The deep plain nets suffer from increased depth, and exhibit higher training error when going deeper. This phenomenon is similar to that on ImageNet (Fig. 4, left) and on MNIST (see [42]), suggesting that such an optimization difficulty is a fundamental problem.

Fig. 6 (middle) shows the behaviors of ResNets. Also similar to the ImageNet cases (Fig. 4, right), our ResNets manage to overcome the optimization difficulty and demon-strate accuracy gains when the depth increases.

We further explore $n = 18$ that leads to a 110-layer ResNet. In this case, we find that the initial learning rate of 0.1 is slightly too large to start converging $^5$ . So we use 0.01 to warm up the training until the training error is below $80%$ (about 400 iterations), and then go back to 0.1 and continue training. The rest of the learning schedule is as done previously. This 110-layer network converges well (Fig. 6, middle). It has fewer parameters than other deep and thin networks such as FitNet [35] and Highway [42] (Table 6), yet is among the state-of-the-art results (6.43%, Table 6).

>$^5$ With an initial learning rate of 0.1, it starts converging (<90% error) after several epochs, but still reaches similar accuracy

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Deep-Residual-Learning-for-Image-Recognition/Fig%206.png"/></div>

Figure 6. Training on **CIFAR-10**. Dashed lines denote training error, and bold lines denote testing error. **Left**: plain networks. The error of plain-110 is higher than 60% and not displayed. **Middle**: ResNets. **Right**: ResNets with 110 and 1202 layers.

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Deep-Residual-Learning-for-Image-Recognition/Fig%207.png"/></div>

Figure 7. Standard deviations (std) of layer responses on CIFAR-10. The responses are the outputs of each $3×3$ layer, after BN and before nonlinearity. **Top**: the layers are shown in their original order. **Bottom**: the responses are ranked in descending order.

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Deep-Residual-Learning-for-Image-Recognition/Table%207.png"/></div>

Table 7. Object detection mAP (%) on the PASCAL VOC 2007/2012 test sets using **baseline** Faster R-CNN. See also Table 10 and 11 for better results. 

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Deep-Residual-Learning-for-Image-Recognition/Table%208.png"/></div>

Table 8. Object detection mAP (%) on the COCO validation set using **baseline** Faster R-CNN. See also Table 9 for better results.

Analysis of Layer Responses. Fig. 7 shows the standard deviations (std) of the layer responses. The responses are the outputs of each 33 layer, after BN and before other nonlinearity (ReLU/addition). For ResNets, this analysis reveals the response strength of the residual functions. Fig. 7 shows that ResNets have generally smaller responses than their plain counterparts. These results support our basic motivation (Sec.3.1) that the residual functions might be generally closer to zero than the non-residual functions. We also notice that the deeper ResNet has smaller magnitudes of responses, as evidenced by the comparisons among ResNet-20, 56, and 110 in Fig. 7. When there are more layers, an individual layer of ResNets tends to modify the signal less.
Exploring Over 1000 layers. We explore an aggressively deep model of over 1000 layers. We set n = 200 that leads to a 1202-layer network, which is trained as described above. Our method shows no optimization difficulty, and this 103-layer network is able to achieve training error <0.1% (Fig. 6, right). Its test error is still fairly good (7.93%, Table 6).

But there are still open problems on such aggressively deep models. The testing result of this 1202-layer network is worse than that of our 110-layer network, although both have similar training error. We argue that this is because of overfitting. The 1202-layer network may be unnecessarily large (19.4M) for this small dataset. Strong regularization such as maxout [10] or dropout [14] is applied to obtain the best results ([10,25,24,35]) on this dataset. In this paper, we use no maxout/dropout and just simply impose regularization via deep and thin architectures by design, without distracting from the focus on the difficulties of optimization. But combining with stronger regularization may improve results, which we will study in the future.

### 4.3. Object Detection on PASCAL and MS COCO

Our method has good generalization performance on other recognition tasks. Table 7 and 8 show the object detection baseline results on PASCAL VOC 2007 and 2012 [5] and COCO [26]. We adopt Faster R-CNN [32] as the detection method. Here we are interested in the improvements of replacing VGG-16 [41] with ResNet-101. The detection implementation (see appendix) of using both models is the same, so the gains can only be attributed to better networks. Most remarkably, on the challenging COCO dataset we ob-tain a 6.0% increase in COCO’s standard metric (mAP@[.5, .95]), which is a 28% relative improvement. This gain is solely due to the learned representations.

Based on deep residual nets, we won the 1st places in several tracks in ILSVRC & COCO 2015 competitions: Im- ageNet detection, ImageNet localization, COCO detection, and COCO segmentation. The details are in the appendix.

## References

[1] Y. Bengio, P. Simard, and P. Frasconi. Learning long-term dependencies with gradient descent is difficult. IEEE Transactions on Neural Networks, 5⑵:157-166,1994.

[2] C. M. Bishop. Neural networks for pattern recognition. Oxford university press, 1995.

[3] W. L. Briggs, S. F. McCormick, et al. A Multigrid Tutorial. Siam, 2000.

[4] K. Chatfield, V. Lempitsky, A. Vedaldi, and A. Zisserman. The devil is in the details: an evaluation of recent feature encoding methods. In BMVC, 2011.

[5] M. Everingham, L. Van Gool, C. K. Williams, J. Winn, and A. Zis- serman. The Pascal Visual Object Classes (VOC) Challenge. IJCV, pages 303-338, 2010.

[6] S. Gidaris and N. Komodakis. Object detection via a multi-region & semantic segmentation-aware cnn model. In ICCV, 2015.

[7] R. Girshick. Fast R-CNN. In ICCV, 2015.

[8] R. Girshick, J. Donahue, T. Darrell, and J. Malik. Rich feature hierarchies for accurate object detection and semantic segmentation. In CVPR, 2014.

[9] X. Glorot and Y. Bengio. Understanding the difficulty of training deep feedforward neural networks. In AISTATS, 2010.

[10] I. J. Goodfellow, D. Warde-Farley, M. Mirza, A. Courville, and Y. Bengio. Maxout networks. arXiv:1302.4389, 2013.

[11] K. He and J. Sun. Convolutional neural networks at constrained time cost. In CVPR, 2015.

[12] K. He, X. Zhang, S. Ren, and J. Sun. Spatial pyramid pooling in deep convolutional networks for visual recognition. In ECCV, 2014.

[13] K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In ICCV, 2015.

[14] G. E. Hinton, N. Srivastava, A. Krizhevsky, I. Sutskever, and R. R. Salakhutdinov. Improving neural networks by preventing coadaptation of feature detectors. arXiv:1207.0580, 2012.

[15] S. Hochreiter and J. Schmidhuber. Long short-term memory. Neural computation, 9(8):1735-1780, 1997.

[16] S. Ioffe and C. Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In ICML, 2015.

[17] H. Jegou, M. Douze, and C. Schmid. Product quantization for nearest neighbor search. TPAMI, 33, 2011.

[18] H. Jegou, F. Perronnin, M. Douze, J. Sanchez, P. Perez, and C. Schmid. Aggregating local image descriptors into compact codes. TPAMI, 2012.

[19] Y. Jia, E. Shelhamer, J. Donahue, S. Karayev, J. Long, R. Girshick, S. Guadarrama, and T. Darrell. Caffe: Convolutional architecture for fast feature embedding. arXiv:1408.5093, 2014.

[20] A. Krizhevsky. Learning multiple layers of features from tiny images. Tech Report, 2009.

[21] A. Krizhevsky, I. Sutskever, and G. Hinton. Imagenet classification with deep convolutional neural networks. In NIPS, 2012.

[22] Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. Hubbard, and L. D. Jackel. Backpropagation applied to handwritten zip code recognition. Neural computation, 1989.

[23] Y. LeCun, L. Bottou, G. B. Orr, and K.-R. MUller. Efficient backprop. In Neural Networks: Tricks of the Trade, pages 9-50. Springer, 1998.

[24] C.-Y. Lee, S. Xie, P. Gallagher, Z. Zhang, and Z. Tu. Deeply- supervised nets. arXiv:1409.5185, 2014.

[25] M. Lin, Q. Chen, and S. Yan. Network in network. arXiv:1312.4400, 2013.

[26] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollar, and C. L. Zitnick. Microsoft COCO: Common objects in context. In ECCV. 2014.

[27] J. Long, E. Shelhamer, and T. Darrell. Fully convolutional networks for semantic segmentation. In CVPR, 2015.

[28] G. MontUfar, R. Pascanu, K. Cho, and Y. Bengio. On the number of linear regions of deep neural networks. In NIPS, 2014.

[29] V. Nair and G. E. Hinton. Rectified linear units improve restricted boltzmann machines. In ICML, 2010.

[30] F. Perronnin and C. Dance. Fisher kernels on visual vocabularies for image categorization. In CVPR, 2007.

[31] T. Raiko, H. Valpola, and Y. LeCun. Deep learning made easier by linear transformations in perceptrons. In AISTATS, 2012.

[32] S. Ren, K. He, R. Girshick, and J. Sun. Faster R-CNN: Towards real-time object detection with region proposal networks. In NIPS, 2015.

[33] S. Ren, K. He, R. Girshick, X. Zhang, and J. Sun. Object detection networks on convolutional feature maps. arXiv:1504.06066, 2015.

[34] B. D. Ripley. Pattern recognition and neural networks. Cambridge university press, 1996.

[35] A. Romero, N. Ballas, S. E. Kahou, A. Chassang, C. Gatta, and Y. Bengio. Fitnets: Hints for thin deep nets. In ICLR, 2015.

[36] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, et al. Imagenet large scale visual recognition challenge. arXiv:1409.0575, 2014.

[37] A. M. Saxe, J. L. McClelland, and S. Ganguli. Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. arXiv:1312.6120, 2013.

[38] N. N. Schraudolph. Accelerated gradient descent by factor-centering decomposition. Technical report, 1998.

[39] N. N. Schraudolph. Centering neural network gradient factors. In Neural Networks: Tricks of the Trade, pages 207-226. Springer, 1998.

[40] P. Sermanet, D. Eigen, X. Zhang, M. Mathieu, R. Fergus, and Y. Le- Cun. Overfeat: Integrated recognition, localization and detection using convolutional networks. In ICLR, 2014.

[41] K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. In ICLR, 2015.

[42] R. K. Srivastava, K. Greff, and J. Schmidhuber. Highway networks. arXiv:1505.00387, 2015.

[43] R. K. Srivastava, K. Greff, and J. Schmidhuber. Training very deep networks. 1507.06228, 2015.

[44] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Er- han, V. Vanhoucke, and A. Rabinovich. Going deeper with convolutions. In CVPR, 2015.

[45] R. Szeliski. Fast surface interpolation using hierarchical basis functions. TPAMI, 1990.

[46] R. Szeliski. Locally adapted hierarchical basis preconditioning. In SIGGRAPH, 2006.

[47] T. Vatanen, T. Raiko, H. Valpola, and Y. LeCun. Pushing stochastic gradient towards second-order methods-backpropagation learning with transformations in nonlinearities. In Neural Information Processing, 2013.

[48] A. Vedaldi and B. Fulkerson. VLFeat: An open and portable library of computer vision algorithms, 2008.

[49] W. Venables and B. Ripley. Modern applied statistics with s-plus. 1999.

[50] M. D. Zeiler and R. Fergus. Visualizing and understanding convolutional neural networks. In ECCV, 2014.

## A. Object Detection Baselines

In this section we introduce our detection method based on the baseline Faster R-CNN [32] system. The models are initialized by the ImageNet classification models, and then fine-tuned on the object detection data. We have experi-mented with ResNet-50/101 at the time of the ILSVRC & COCO 2015 detection competitions.

Unlike VGG-16 used in [32], our ResNet has no hidden fc layers. We adopt the idea of “Networks on Conv feature maps” (NoC) [33] to address this issue. We compute the full-image shared conv feature maps using those layers whose strides on the image are no greater than 16 pixels (i.e., convl, conv2_ x, conv3_x, and convex, totally 91 conv layers in ResNet-101; Table 1). We consider these layers as analogous to the 13 conv layers in VGG-16, and by doing so, both ResNet and VGG-16 have conv feature maps of the same total stride (16 pixels). These layers are shared by a region proposal network (RPN, generating 300 proposals) [32] and a Fast R-CNN detection network [7]. RoI pooling [7] is performed before conv5_1. On this RoI-pooled feature, all layers of conv5_x and up are adopted for each region, playing the roles of VGG-16’s fc layers. The final classification layer is replaced by two sibling layers (classi-fication and box regression [7]).

For the usage of BN layers, after pre-training, we compute the BN statistics (means and variances) for each layer on the ImageNet training set. Then the BN layers are fixed during fine-tuning for object detection. As such, the BN layers become linear activations with constant offsets and scales, and BN statistics are not updated by fine-tuning. We fix the BN layers mainly for reducing memory consumption in Faster R-CNN training.

**PASCAL VOC**

Following [7, 32], for the PASCAL VOC 2007 test set, we use the 5k trainval images in VOC 2007 and 16k train- val images in VOC 2012 for training (“07+12”). For the PASCAL VOC 2012 test set, we use the 10k trainval+test images in VOC 2007 and 16k trainval images in VOC 2012 for training (“07++12”). The hyper-parameters for training Faster R-CNN are the same as in [32]. Table 7 shows the results. ResNet-101 improves the mAP by >3% over VGG-16. This gain is solely because of the improved features learned by ResNet.

**MS COCO**

The MS COCO dataset [26] involves 80 object categories. We evaluate the PASCAL VOC metric (mAP @ IoU = 0.5) and the standard COCO metric (mAP @ IoU = .5:.05:.95). We use the 80k images on the train set for training and the 40k images on the val set for evaluation. Our detection system for COCO is similar to that for PASCAL VOC. We train the COCO models with an 8-GPU implementation, and thus the RPN step has a mini-batch size of 8 images ($i.e.$, 1 per GPU) and the Fast R-CNN step has a mini-batch size of 16 images. The RPN step and Fast R- CNN step are both trained for 240k iterations with a learning rate of 0.001 and then for 80k iterations with 0.0001.
Table 8 shows the results on the MS COCO validation set. ResNet-101 has a 6% increase of mAP@[.5, .95] over VGG-16, which is a 28% relative improvement, solely con-tributed by the features learned by the better network. Re-markably, the mAP@[.5, .95]’s absolute increase (6.0%) is nearly as big as mAP@.5’s (6.9%). This suggests that a deeper network can improve both recognition and localiza-tion.

## B. Object Detection Improvements

For completeness, we report the improvements made for the competitions. These improvements are based on deep features and thus should benefit from residual learning.

**MS COCO**

Box refinement. Our box refinement partially follows the it-erative localization in [6]. In Faster R-CNN, the final output is a regressed box that is different from its proposal box. So for inference, we pool a new feature from the regressed box and obtain a new classification score and a new regressed box. We combine these 300 new predictions with the original 300 predictions. Non-maximum suppression (NMS) is applied on the union set of predicted boxes using an IoU threshold of 0.3 [8], followed by box voting [6]. Box refinement improves mAP by about 2 points (Table 9).

*Global context.* We combine global context in the Fast R-CNN step. Given the full-image conv feature map, we pool a feature by global Spatial Pyramid Pooling [12] (with a “single-level” pyramid) which can be implemented as “RoI” pooling using the entire image’s bounding box as the RoI. This pooled feature is fed into the post-RoI layers to obtain a global context feature. This global feature is concatenated with the original per-region feature, followed by the sibling classification and box regression layers. This new structure is trained end-to-end. Global context improves mAP@.5 by about 1 point (Table 9).

*Multi-scale testing.* In the above, all results are obtained by single-scale training/testing as in [32], where the image’s shorter side is s = 600 pixels. Multi-scale training/testing has been developed in [12, 7] by selecting a scale from a feature pyramid, and in [33] by using maxout layers. In our current implementation, we have performed multi-scale testing following [33]; we have not performed multi-scale training because of limited time. In addition, we have performed multi-scale testing only for the Fast R-CNN step (but not yet for the RPN step). With a trained model, we compute conv feature maps on an image pyramid, where the image’s shorter sides are $s \in  {200,400, 600, 800, 1000}$.

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Deep-Residual-Learning-for-Image-Recognition/Table%209.png"/></div>
Table 9. Object detection improvements on MS COCO using Faster R-CNN and ResNet-101.

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Deep-Residual-Learning-for-Image-Recognition/Table%2010.png"/></div>
Table 10. Detection results on the PASCAL VOC 2007 test set. The baseline is the Faster 

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Deep-Residual-Learning-for-Image-Recognition/Table%2011.png"/></div>
Table 11. Detection results on the PASCAL VOC 2012 test set (http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?challengeid=11&compid=4). The baseline is the Faster R-CNN system. The system “baseline+++” include box refinement, context, and multi-scale testing in Table 9. 

We select two adjacent scales from the pyramid following [33]. RoI pooling and subsequent layers are performed on the feature maps of these two scales [33], which are merged by maxout as in [33]. Multi-scale testing improves the mAP by over 2 points (Table 9).

*Using validation data.* Next we use the 80k+40k trainval set for training and the 20k test-dev set for evaluation. The test- dev set has no publicly available ground truth and the result is reported by the evaluation server. Under this setting, the results are an mAP@.5 of 55.7% and an mAP@[.5, .95] of 34.9% (Table 9). This is our single-model result.

*Ensemble.* In Faster R-CNN, the system is designed to learn region proposals and also object classifiers, so an ensemble can be used to boost both tasks. We use an ensemble for proposing regions, and the union set of proposals are pro-cessed by an ensemble of per-region classifiers. Table 9 shows our result based on an ensemble of 3 networks. The mAP is 59.0% and 37.4% on the test-dev set. This result won the 1st place in the detection task in COCO 2015.

**PASCAL VOC**

We revisit the PASCAL VOC dataset based on the above model. With the single model on the COCO dataset (55.7% mAP@.5 in Table 9), we fine-tune this model on the PASCAL VOC sets. The improvements of box refinement, context, and multi-scale testing are also adopted. By doing so we achieve 85.6% mAP on PASCAL VOC 2007 (Table 10) and 83.8% on PASCAL VOC 2012 (Table 11) $^6$. The result on PASCAL VOC 2012 is 10 points higher than the previous state-of-the-art result [6].

>$^6$ http://host.robots.ox.ac.uk:8080/anonymous/3OJ4OJ.html,submitted on 2015-11-26.

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Deep-Residual-Learning-for-Image-Recognition/Table%2012.png"/></div>
Table 12. Our results (mAP, %) on the ImageNet detection dataset. Our detection system is Faster R-CNN [32] with the improvements in Table 9, using ResNet-101.

**ImageNet Detection**

The ImageNet Detection (DET) task involves 200 object categories. The accuracy is evaluated by mAP@.5. Our object detection algorithm for ImageNet DET is the same as that for MS COCO in Table 9. The networks are pretrained on the 1000-class ImageNet classification set, and are fine-tuned on the DET data. We split the validation set into two parts (val1/val2) following [8]. We fine-tune the detection models using the DET training set and the val1 set. The val2 set is used for validation. We do not use other ILSVRC 2015 data. Our single model with ResNet-101 has 58.8% mAP and our ensemble of 3 models has 62.1% mAP on the DET test set (Table 12). This result won the 1st place in the ImageNet detection task in ILSVRC 2015, surpassing the second place by **8.5 points** (absolute).

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Deep-Residual-Learning-for-Image-Recognition/Table%2013.png"/></div>

Table 13. Localization error (%) on the ImageNet validation. In the column of “LOC error on GT class” ([41]), the ground truth class is used. In the “testing” column, “1-crop” denotes testing on a center crop of 224224 pixels, “dense” denotes dense (fully convolutional) and multi-scale testing.

## C. ImageNet Localization

The ImageNet Localization (LOC) task [36] requires to classify and localize the objects. Following [40, 41], we assume that the image-level classifiers are first adopted for predicting the class labels of an image, and the localization algorithm only accounts for predicting bounding boxes based on the predicted classes. We adopt the “per-class regression” (PCR) strategy [40, 41], learning a bounding box regressor for each class. We pre-train the networks for Im- ageNet classification and then fine-tune them for localization. We train networks on the provided 1000-class ImageNet training set.

Our localization algorithm is based on the RPN frame-work of [32] with a few modifications. Unlike the way in [32] that is category-agnostic, our RPN for localization is designed in a *per-class* form. This RPN ends with two sibling $1 \times 1$ convolutional layers for binary classification (cls) and box regression (reg), as in [32]. The cls and reg layers are both in a *per-class* from, in contrast to [32]. Specifically, the cls layer has a 1000-d output, and each dimension is binary logistic regression for predicting being or not being an object class; the reg layer has a $1000 \times 4-d$ output consisting of box regressors for 1000 classes. As in [32], our bounding box regression is with reference to multiple translation-invariant “anchor” boxes at each position.

As in our ImageNet classification training (Sec. 3.4), we randomly sample $224 \times 224$ crops for data augmentation. We use a mini-batch size of 256 images for fine-tuning. To avoid negative samples being dominate, 8 anchors are randomly sampled for each image, where the sampled positive and negative anchors have a ratio of 1:1 [32]. For testing, the network is applied on the image fully-convolutionally.
Table 13 compares the localization results. Following [41], we first perform “oracle” testing using the ground truth class as the classification prediction. VGG’s paper [41] reports a center-crop error of 33.1% (Table 13) using ground truth classes. Under the same setting, our RPN method using ResNet-101 net significantly reduces the center-crop error to 13.3%. This comparison demonstrates the excellent performance of our framework. With dense (fully convolu-tional) and multi-scale testing, our ResNet-101 has an error of 11.7% using ground truth classes. Using ResNet-101 for predicting classes (4.6% top-5 classification error, Table 4), the top-5 localization error is 14.4%.

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Deep-Residual-Learning-for-Image-Recognition/Table%2014.png"/></div>

Table 14. Comparisons of localization error (%) on the ImageNet dataset with state-of-the-art methods.

The above results are only based on the proposal network (RPN) in Faster R-CNN [32]. One may use the detection network (Fast R-CNN [7]) in Faster R-CNN to improve the results. But we notice that on this dataset, one image usually contains a single dominate object, and the proposal regions highly overlap with each other and thus have very similar RoI-pooled features. As a result, the image-centric training of Fast R-CNN [7] generates samples of small variations, which may not be desired for stochastic training. Motivated by this, in our current experiment we use the original R- CNN [8] that is RoI-centric, in place of Fast R-CNN.

Our R-CNN implementation is as follows. We apply the per-class RPN trained as above on the training images to predict bounding boxes for the ground truth class. These predicted boxes play a role of class-dependent proposals. For each training image, the highest scored 200 proposals are extracted as training samples to train an R-CNN classifier. The image region is cropped from a proposal, warped to 224224 pixels, and fed into the classification network as in R-CNN [8]. The outputs of this network consist of two sibling fc layers for cls and reg, also in a per-class form. This R-CNN network is fine-tuned on the training set using a mini-batch size of 256 in the RoI-centric fashion. For testing, the RPN generates the highest scored 200 proposals for each predicted class, and the R-CNN network is used to update these proposals’ scores and box positions.

This method reduces the top-5 localization error to 10.6% (Table 13). This is our single-model result on the validation set. Using an ensemble of networks for both classification and localization, we achieve a top-5 localization error of 9.0% on the test set. This number significantly outperforms the ILSVRC 14 results (Table 14), showing a 64% relative reduction of error. *This result won the 1st place in the ImageNet localization task in ILSVRC 2015*.
