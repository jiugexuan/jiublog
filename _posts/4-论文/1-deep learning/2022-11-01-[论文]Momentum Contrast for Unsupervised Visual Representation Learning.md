---
title: 【论文】Momentum Contrast for Unsupervised Visual Representation Learning 通过动量对比学习的方法实现无监督的视觉表征学习

date: 2022-11-01 07:00:00 +/-0800
categories: [论文,深度学习]
tags: [深度学习]     # TAG names should always be lowercase 标记名称应始终为小写
---

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
tex2jax: {
inlineMath: [['$','$'], ['\\(','\\)']],
processEscapes: true
}
});
</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

> 对比学习
> 通过pretext task（代理任务）实现自监督
> 
> instance discrimination：
> 自己图片的仿射变换为正样本，其他图片为负样本
> 模型举例
> - SimCSE：一个句子进行两次forward，使用不同的dropout为正样本，其他句子为负样本
> - CMC：一个物体的不同视角view为正样本，

> 动量
> 加权移动平均
> $y_t = my_{t-1}+(1-m)x_t$
> $y_t:当前时刻的输出。y_{t-1}:上一时刻的输出。x_t：当前时刻的输入$

<div align=center>Kaiming He &nbsp Haoqi Fan &nbsp Yuxin Wu &nbsp Saining Xie &nbspRoss Girshick</div>
<div align=center>Facebook AI Research (FAIR)</div>
<div align=center>Code: https://github.com/facebookresearch/moco </div>

## Abstract 摘要

*We present Momentum Contrast (MoCo) for unsupervised visual representation learning. From a perspective on contrastive learning [29] as dictionary look-up, we build a dynamic dictionary with a queue and a moving-averaged encoder. This enables building a large and consistent dictionary on-the-fly that facilitates contrastive unsupervised learning. MoCo provides competitive results under the common linear protocol on ImageNet classification. More importantly, the representations learned by MoCo transfer well to downstream tasks. MoCo can **outperform** its supervised pre-training counterpart in 7 detection/segmentation tasks on PASCAL VOC, COCO, and other datasets, sometimes surpassing it by large margins. This suggests that the gap between unsupervised and supervised representation learning has been largely closed in many vision tasks.*

*我们提出了用于无监督视觉表征学习的 Momentum Contrast (MoCo)。 从对比学习 [29] 作为字典查询的角度来看，我们构建了一个带有队列【放到队列中样本不需要要进行梯度回传，可以放入大量的负样本】和移动平均编码器【使得字典比较的一致】的动态字典。 这使得能够即时构建一个大型且一致的字典，以促进对比无监督学习。 MoCo 在 ImageNet 分类的通用线性协议【预训练好骨干网络在处理任务时冻住，只调整最后的全连接层，把预训练好模型当特征提取器，来证明其学习表示的特征到底好不好】下提供有竞争力的结果。 更重要的是，MoCo 学习的表示可以很好地迁移到下游任务。 在 PASCAL VOC、COCO 和其他数据集上的 7 个检测/分割任务中，MoCo 可以**胜过**它的监督预训练对手，有时甚至会大大超过它。 这表明在许多视觉任务中，无监督和有监督表示学习之间的差距已基本缩小。*

## 1. Introduction 引言

Unsupervised representation learning is highly successful in natural language processing, e.g., as shown by GPT [50,51] and BERT [12]. But supervised pre-training is still dominant in computer vision, where unsupervised methods generally lag behind. The reason may stem from differences in their respective signal spaces. Language tasks have discrete signal spaces (words, sub-word units, etc.) for building *tokenized dictionaries*, on which unsupervised learning can be based. Computer vision, in contrast, further concerns dictionary building [54,9,5], as the raw signal is in a continuous, high-dimensional space and is not structured for human communication (e.g., unlike words).

无监督表示学习在自然语言处理中非常成功，例如 GPT [50,51] 和 BERT [12] 所示。 但是有监督的预训练在计算机视觉中仍然占主导地位，而无监督的方法通常落后。 原因可能源于它们各自信号空间的差异。 语言任务具有用于构建*标记化字典*【一个单词对应一个特征】的离散信号空间（单词、词根词缀等），无监督学习可以在此基础上进行。 相比之下，计算机视觉进一步关注字典构建 [54,9,5]，因为原始信号处于连续的高维空间中，并且不是为人类交流而结构化的（例如，与单词不同）。

Several recent studies [61,46,36,66,35,56,2] present promising results on unsupervised visual representation learning using approaches related to the contrastive loss [29]. Though driven by various motivations, these methods can be thought of as building *dynamic dictionaries*. The “keys” (tokens) in the dictionary are sampled from data (e.g., images or patches) and are represented by an encoder network. Unsupervised learning trains encoders to perform dictionary look-up: an encoded “query” should be similar to its matching key and dissimilar to others. Learning is formulated as minimizing a contrastive loss [29].

最近的几项研究 [61,46,36,66,35,56,2] 提出了使用与对比损失相关的方法进行无监督视觉表示学习的有希望的结果 [29]。 尽管使用了各种各样的方法，这些方法可以被认为是构建*动态字典*。 字典中的“键”（词元）是从数据（例如图像或patch【块】）中采样的，并由编码器网络表示。 无监督学习训练编码器执行字典查找：编码的“查询”应该与其匹配键相似，而与其他键远离。 学习被表述为最小化对比损失[29]。

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Momentum%20Contrast%20for%20Unsupervised%20Visual%20Representation%20Learning/%E8%A7%A3%E9%87%8A1.png"/></div>

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Momentum%20Contrast%20for%20Unsupervised%20Visual%20Representation%20Learning/Fig%201.png"/></div>

Figure 1. Momentum Contrast (MoCo) trains a visual representation encoder by matching an encoded query $q$ to a dictionary of encoded keys using a contrastive loss. The dictionary keys $\{k_0, k_1, k_2,...\}$ are defined on-the-fly by a set of data samples. The dictionary is built as a queue, with the current mini-batch enqueued and the oldest mini-batch dequeued, decoupling it from the mini-batch size. The keys are encoded by a slowly progressing encoder, driven by a momentum update with the query encoder. This method enables a large and consistent dictionary for learning visual representations.

图 1. Momentum Contrast (MoCo) 通过使用对比损失将编码查询 $q$ 与编码键字典匹配来训练视觉表示编码器。 字典键 $\{k_0, k_1, k_2,...\}$ 由一组数据样本动态定义。 字典被构建为一个队列，当前的小批量入队，最旧的小批量出队，将其与小批量大小解耦。 字典键由缓慢进展的编码器编码，由查询编码器的动量更新驱动。 这种方法可以使用一个大而一致的字典来学习视觉表征。

From this perspective, we hypothesize that it is desirable to build dictionaries that are: (i) large *and* (ii) consistent as they evolve during training. Intuitively, a larger dictionary may better sample the underlying continuous, high-dimensional visual space, while the keys in the dictionary should be represented by the same or similar encoder so that their comparisons to the query are consistent. However, existing methods that use contrastive losses can be limited in one of these two aspects (discussed later in context).

从这个角度来看，我们假设构建的字典是：（i）大的*和*（ii）在训练过程中不断发展的字典。 直观地说，更大的字典可能更好地对底层连续的高维视觉空间进行采样，而字典中的键应该由相同或相似的编码器表示，以便它们与查询的比较是一致的。 但是，使用对比损失的现有方法可能会在这两个方面之一受到限制（稍后在上下文中讨论）。

We present Momentum Contrast (MoCo) as a way of building large and consistent dictionaries for unsupervised learning with a contrastive loss (Figure 1). We maintain the dictionary as a queue of data samples: the encoded representations of the current mini-batch are enqueued, and the oldest are dequeued. The queue decouples the dictionary size from the mini-batch size, allowing it to be large. Moreover, as the dictionary keys come from the preceding several mini-batches, a slowly progressing key encoder, implemented as a momentum-based moving average of the query encoder, is proposed to maintain consistency.

我们提出 Momentum Contrast (MoCo) 作为一种为具有对比损失的无监督学习构建大型且一致的字典的方法（图 1）。 我们将字典维护为数据样本队列：当前小批量的编码表示入队，最旧的出队。 队列将字典大小与小批量大小解耦，允许它很大。 此外，由于字典键来自前面的几个小批量，因此提出了一种缓慢进展的键编码器，实现为查询编码器的基于动量的移动平均值，以保持一致性。

MoCo is a mechanism for building dynamic dictionaries for contrastive learning, and can be used with various pretext tasks. In this paper, we follow a simple instance discrimination task [61,63,2]: a query matches a key if they are encoded views (e.g., different crops) of the same image. Using this pretext task, MoCo shows competitive results under the common protocol of linear classification in the ImageNet dataset [11].

MoCo 是一种为对比学习构建动态词典的机制，可用于各种代理任务。 在本文中，我们遵循一个简单的实例判别任务 [61,63,2]：如果它们是同一图像的不同视图的编码（例如，不同的随机裁剪），则查询匹配同一个键。 使用这个代理任务，MoCo 在 ImageNet 数据集 [11] 中的线性分类通用协议下显示了有竞争力的结果。

A main purpose of unsupervised learning is to pre-train representations (i.e., features) that can be transferred to downstream tasks by fine-tuning. We show that in 7 down-stream tasks related to detection or segmentation, MoCo unsupervised pre-training can *surpass* its ImageNet supervised counterpart, in some cases by nontrivial margins. In these experiments, we explore MoCo pre-trained on ImageNet or on a *one-billion* Instagram image set, demonstrating that MoCo can work well in a more real-world, billionimage scale, and relatively uncurated scenario. These results show that MoCo largely closes the gap between unsupervised and supervised representation learning in many computer vision tasks, and can serve as an alternative to Im- ageNet supervised pre-training in several applications.

无监督学习的一个主要目的是预训练可以通过微调迁移到下游任务的表示（即特征）。 我们表明，在与检测或分割相关的 7 个下游任务中，MoCo 无监督预训练可以*超过*其 ImageNet 监督对应物，在某些情况下，可以大幅超越。 在这些实验中，我们探索了在 ImageNet 或 10 亿张 Instagram 图像集上预训练的 MoCo，证明 MoCo 可以在更真实世界、*十亿*图像规模和相对未经处理的场景中运行良好。 这些结果表明，MoCo 在很大程度上缩小了许多计算机视觉任务中无监督和监督表示学习之间的差距，并且可以在多个应用中作为 ImageNet 监督预训练的替代方案。

## 2. Related Work 相关工作

Unsupervised/self-supervised$^1$  learning methods generally involve two aspects: pretext tasks and loss functions. The term “pretext” implies that the task being solved is not of genuine interest, but is solved only for the true purpose of learning a good data representation. Loss functions can often be investigated independently of pretext tasks. MoCo focuses on the loss function aspect. Next we discuss related studies with respect to these two aspects.

无监督/自监督$^1$学习方法一般涉及两个方面：代理任务和损失函数【目标函数】。 术语“代理”意味着要解决的任务不是真正感兴趣的，而只是为了学习良好数据表示的真正目的而解决。 损失函数通常可以独立于代理任务进行研究。 MoCo 专注于损失函数方面。 接下来我们就这两个方面讨论相关研究。

>$^1$ Self-supervised learning is a form of unsupervised learning. Their distinction is informal in the existing literature. In this paper, we use the more classical term of “unsupervised learning”, in the sense of “not supervised by human-annotated labels”.
>$^1$ 自监督学习是无监督学习的一种形式。 它们的区别在现有文献中是非正式的【即大家混用这两种叫法】。 在本文中，我们使用更经典的术语“无监督学习”，即“不受人工标注的标签监督”。

**Loss functions.** A common way of defining a loss function is to measure the difference between a model’s prediction and a fixed target, such as reconstructing the input pixels (e.g., auto-encoders) by L1 or L2 losses, or classifying the input into pre-defined categories (e.g., eight positions [13], color bins [64]) by cross-entropy or margin-based losses. Other alternatives, as described next, are also possible.

**损失函数。**定义损失函数的常用方法是衡量模型的预测与固定目标之间的差异，例如通过 L1 或 L2 损失重建输入像素（例如，自动编码器），或分类 通过交叉熵或基于边际的损失将输入输入到预定义的类别（例如，八个位置 [13]【将一张图片分成9宫格，给出中间位置的格子，再随机选择一个格子判别这个格子与中间格子位置关系（上、下、左、右、左上、左下、右上、右下）】、颜色箱 [64]）。 如下所述的其他替代方案也是可能的。

Contrastive losses [29] measure the similarities of sample pairs in a representation space. Instead of matching an input to a fixed target, in contrastive loss formulations the target can vary on-the-fly during training and can be defined in terms of the data representation computed by a network [29]. Contrastive learning is at the core of several recent works on unsupervised learning [61,46,36,66,35,56,2], which we elaborate on later in context (Sec. 3.1).

对比损失【对比学习的目标函数】 [29] 测量表示空间中样本对的相似性。 与将输入匹配到固定目标不同，在对比损失公式中，目标可以在训练期间动态变化【由编码器所决定，不同状态的编码器输出的特征不同】，并且可以根据网络计算的数据表示来定义 [29]。 对比学习是最近几项关于无监督学习的工作的核心 [61,46,36,66,35,56,2]，我们稍后会在上下文中详细说明（第 3.1 节）。

Adversarial losses [24] measure the difference between probability distributions. It is a widely successful technique for unsupervised data generation. Adversarial methods for representation learning are explored in [15,16]. There are relations (see [24]) between generative adversarial networks and noise-contrastive estimation (NCE) [28].

对抗性损失【对抗性目标函数】 [24] 衡量概率分布之间的差异。 它是一种广泛成功的无监督数据生成技术。 [15,16] 中探讨了表示学习的对抗性方法。 生成对抗网络和噪声对比估计（NCE）[28]之间存在关系（参见[24]）。

**Pretext tasks.** A wide range of pretext tasks have been proposed. Examples include recovering the input under some corruption, e.g., denoising auto-encoders [58], context autoencoders [48], or cross-channel auto-encoders (colorization) [64, 65]. Some pretext tasks form pseudo-labels by, e.g., transformations of a single (“exemplar”) image [17], patch orderings [13,45], tracking [59] or segmenting objects [47] in videos, or clustering features [3,4].

**代理任务。** 已经提出了广泛的代理任务。 示例包括在某些损失下恢复输入，例如去噪自动编码器 [58]、上下文自动编码器 [48] 或跨通道自动编码器（着色）【给图片上色】[64、65]。 一些代理任务通过例如转换单个（“示例”）图像[17]【给同一张图片做不同的数据增广】、patch排序[13,45]、跟踪[59]或分割视频中的对象【利用视频里的信息】[47]或聚类特征[47]来形成伪标签。 [3,4]。

**Contrastive learning vs. pretext tasks.** Various pretext tasks can be based on some form of contrastive loss functions. The instance discrimination method [61] is related to the exemplar-based task [17] and NCE [28]. The pretext task in contrastive predictive coding (CPC) [46] is a form of context auto-encoding [48], and in contrastive multiview coding (CMC) [56] it is related to colorization [64].

**对比学习与代理任务。**各种代理任务可以基于某种形式的对比损失函数【对比学习的目标函数】。 实例判别方法 [61] 与基于示例的任务 [17] 和 NCE [28] 有关。 对比预测编码 (CPC) [46] 中的代理任务是上下文自动编码 [48] 的一种形式，而在对比多视图编码 (CMC) [56] 中【用一个物体的不同视角来做对比】，它与着色 [64]【给图片上色有关】 有关。

## 3. Method 实验方法

### 3.1. Contrastive Learning as Dictionary Look-up 对比学习作为字典查找

Contrastive learning [29], and its recent developments, can be thought of as training an encoder for a dictionary look-up task, as described next.

对比学习 [29] 及其最近的发展，可以被认为是为字典查找任务训练编码器，如下所述。

Consider an encoded query $q$ and a set of encoded samples $\{k_0,k_1,k_2,...\}$ that are the keys of a dictionary. Assume that there is a single key (denoted as $k_+$) in the dictionary that $q$ matches. A contrastive loss [29] is a function whose value is low when $q$ is similar to its positive key $k_+$ and dissimilar to all other keys (considered negative keys for $q$). With similarity measured by dot product, a form of a contrastive loss function, called InfoNCE [46], is considered in this paper:

考虑一个编码【编码好的】查询 $q$ 和一组编码【编码好的】样本 $\{k_0,k_1,k_2,...\}$，它们是字典的键。 假设在 $q$ 匹配的字典中有一个键（表示为 $k_+$）【互为正样本对】。 对比损失 [29] 是一个函数，其满足当 $q$ 与其正键 $k_+$ 相似并且与所有其他键（考虑为 $q$ 的负键）不同时，该函数的值较低。 通过点积测量相似度，本文考虑了一种称为 InfoNCE [46] 的对比损失函数【对比学习目标函数】：

>NCE:noise contrastive extimation 将输出的样本与抽样的负样本进行交叉熵对比， 减少计算量
>，其作用是将超级多的分类问题变成一个二分类的问题

$$
\mathcal{L}_q = — \log \frac{\exp(q \cdot k_t / \tau)}{\sum^K_{i=0}\exp(q \cdot k_i/ \tau)} \tag{1}
$$

where $\tau$ is a temperature hyper-parameter per [61]. The sum is over one positive and $K$ negative samples. Intuitively, this loss is the log loss ofa ($K+1$)-way softmax-based classifier that tries to classify $q$ as $k_+$. Contrastive loss functions can also be based on other forms [29,59,61,36], such as margin-based losses and variants of NCE losses.

其中 $\tau$ 是每个 [61] 的温度超参数【用于放大分布的峰值，过大，使得目标函数对所有损失函数都一视同仁，导致模型的学习没有轻重。但温度值国小会使得模型关注比较困难的样本，忽略潜在的正样本，导致模型不好收敛或者泛化能力差】。 公式下面的求和是在一个正样本和 $K$ 个负样本上计算的。 直观地说，这个损失是试图将 $q$ 分类为 $k_+$ 的 ($K+1$) 路基于 softmax 的分类器的对数损失。 对比损失函数也可以基于其他形式 [29,59,61,36]，例如基于保证金的损失和 NCE 损失的变体。

The contrastive loss serves as an unsupervised objective function for training the encoder networks that represent the queries and keys [29]. In general, the query representation is $q = f_q(x^q)$ where $f_q$ is an encoder network and $x^q$ is a query sample (likewise, $k = f_k(x^k)$). Their instantiations depend on the specific pretext task. The input $x^q$ and $x^k$ can be images [29,61,63], patches [46], or context consisting a set of patches [46]. The networks $f_q$ and $f_k$ can be identical [29,59,63], partially shared [46,36,2], or different [56].

对比损失作为无监督目标函数，用于训练表示查询和键的编码器网络 [29]。 通常，查询表示为 $q = f_q(x^q)$，其中 $f_q$ 是编码器网络，$x^q$ 是查询样本（同样，$k = f_k(x^k)$）。 它们的实例化取决于特定的代理任务。 输入 $x^q$ 和 $x^k$ 可以是图像 [29,61,63]、patch [46] 或由一组patch组成的上下文 [46]。 网络【即模型或者说是编码器】 $f_q$ 和 $f_k$ 可以是相同的 [29,59,63]、部分共享的 [46,36,2] 或不同的 [56]。

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Momentum%20Contrast%20for%20Unsupervised%20Visual%20Representation%20Learning/Fig%202.png"/></div>

Figure 2. **Conceptual comparison of three contrastive loss mechanisms** (empirical comparisons are in Figure 3 and Table 3). Here we illustrate one pair of query and key. The three mechanisms differ in how the keys are maintained and how the key encoder is updated. **(a)**: The encoders for computing the query and key representations are updated end-to-end by back-propagation (the two encoders can be different). **(b)**: The key representations are sampled from a **memory bank** [61]. **(c): *MoCo*** encodes the new keys on-the-fly by a momentum-updated encoder, and maintains a queue (not illustrated in this figure) of keys.

### 3.2. Momentum Contrast 动量对比

From the above perspective, contrastive learning is a way of building a discrete dictionary on high-dimensional continuous inputs such as images. The dictionary is dynamic in the sense that the keys are randomly sampled, and that the key encoder evolves during training. Our hypothesis is that good features can be learned by a large dictionary that covers a rich set of negative samples, while the encoder for the dictionary keys is kept as consistent as possible despite its evolution. Based on this motivation, we present Momentum Contrast as described next.

从上述角度来看，对比学习是一种在图像等高维连续输入上构建离散字典的方法。 字典是动态的，因为关键字【即key】是随机采样的，并且关键字编码器在训练期间会改变。 我们的假设是，好的特征可以通过覆盖大量负样本的大型字典来学习，而字典键【key】的编码器尽管在进化，但仍尽可能保持一致【防止模型走捷径】。 基于这个动机，我们提出了动量对比，如下所述

**Dictionary as a queue.** At the core of our approach is maintaining the dictionary as a queue of data samples. This allows us to reuse the encoded keys from the immediate preceding mini-batches. The introduction of a queue decouples the dictionary size from the mini-batch size. Our dictionary size can be much larger than a typical mini-batch size, and can be flexibly and independently set as a hyper-parameter.

**字典作为队列。** 我们方法的核心是将字典维护为数据样本队列。 这使我们可以重用之前小批量中的编码key。 队列的引入将字典大小与小批量大小分离。 我们的字典大小可以比典型的 mini-batch 大得多，并且可以灵活独立地设置为超参数。【为了保留原始的学习信息】

The samples in the dictionary are progressively replaced. The current mini-batch is enqueued to the dictionary, and the oldest mini-batch in the queue is removed. The dictionary always represents a sampled subset of all data, while the extra computation of maintaining this dictionary is manageable. Moreover, removing the oldest mini-batch can be beneficial, because its encoded keys are the most outdated and thus the least consistent with the newest ones.

字典中的样本被逐步替换。 当前的 mini-batch 被排入字典，队列中最旧的 mini-batch 被删除。 字典始终代表所有数据的采样子集，而维护该字典的额外计算是可管理【可以被接受，即比较小】的。 此外，删除最旧的 mini-batch 可能是有益的，因为它的编码密钥是最过时的，因此与最新的最不一致。

**Momentum update.** Using a queue can make the dictionary large, but it also makes it intractable to update the key encoder by back-propagation (the gradient should propagate to all samples in the queue). A naive solution is to copy the key encoder $f_k$ from the query encoder $f_q$, ignoring this gradient. But this solution yields poor results in experiments (Sec. 4.1). We hypothesize that such failure is caused by the rapidly changing encoder that reduces the key representations’ consistency. We propose a momentum update to address this issue.

**动量更新。** 使用队列可以使字典变大，但它也使得通过反向传播更新键编码器变得难以处理（梯度应该传播到队列中的所有样本）。 一个简单的解决方案是从查询编码器 $f_q$ 复制key编码器 $f_k$，忽略这个梯度。 但是这个解决方案在实验中产生了很差的结果（第 4.1 节）。 我们假设这种失败是由快速变化的编码器引起的，这降低了key表示的一致性。 我们提出动量更新来解决这个问题。

Formally, denoting the parameters of $f_k$ as $\theta_k$ and those of $f_q$ as $\theta_q$, we update $\theta_k$ by:

形式上，将 $f_k$ 的参数表示为 $\theta_k$，将 $f_q$ 的参数表示为 $\theta_q$，我们通过以下方式更新 $\theta_k$：

$$
\theta_k \rightarrow m\theta_k + (1-m)\theta_q \tag{2}
$$

Here $m \in [0,1)$ is a momentum coefficient. Only the parameters $\theta_q$ are updated by back-propagation. The momentum update in Eqn.(2) makes $\theta_k$ evolve more smoothly than $\theta_q$. As a result, though the keys in the queue are encoded by different encoders (in different mini-batches), the difference among these encoders can be made small. In experiments, a relatively large momentum ($e.g.,$ $m = 0.999$, our default) works much better than a smaller value ($e.g., m = 0.9$), suggesting that a slowly evolving key encoder is a core to making use of a queue.

这里 $m \in [0,1)$ 是动量系数。 只有参数 $\theta_q$ 通过反向传播更新。 Eqn.(2) 中的动量更新使得 $\theta_k$ 的演化比 $\theta_q$ 更平滑。 结果，尽管队列中的键由不同的编码器（在不同的小批量中）编码，但这些编码器之间的差异可以很小。 在实验中，相对较大的动量（$e.g.,$$m = 0.999$，我们的默认值）比较小的值（$e.g.,m = 0.9$）效果更好，这表明如果想充分利用一个队列则缓慢发展的key编码器是核心。

**Relations to previous mechanisms.** MoCo is a general mechanism for using contrastive losses. We compare it with two existing general mechanisms in Figure 2. They exhibit different properties on the dictionary size and consistency.

**与先前机制的关系。** MoCo 是使用对比损失的一般机制。 我们将其与图 2 中的两种现有通用机制进行比较。它们在字典大小和一致性方面表现出不同的属性。

The **end-to-end** update by back-propagation is a natural mechanism ($e.g.,$ [29,46,36,63,2,35], Figure 2a). It uses samples in the current mini-batch as the dictionary, so the keys are consistently encoded (by the same set of encoder parameters). But the dictionary size is coupled with the mini-batch size, limited by the GPU memory size. It is also challenged by large mini-batch optimization [25]. Some recent methods [46,36,2] are based on pretext tasks driven by local positions, where the dictionary size can be made larger by multiple positions. But these pretext tasks may require special network designs such as patchifying the input [46] or customizing the receptive field size [2], which may complicate the transfer of these networks to downstream tasks.

通过反向传播的**端到端**更新是一种自然机制（例如，$ [29,46,36,63,2,35]，图 2a）。 它使用当前小批量中的样本作为字典，因此键被一致地编码（由同一组编码器参数）。 但是字典大小与 mini-batch 大小相结合，受 GPU 内存大小的限制。 它也受到大型小批量优化的挑战[25]。 最近的一些方法 [46,36,2] 基于由本地位置驱动的代理任务，其中字典大小可以通过多个位置变大。 但是这些代理任务可能需要特殊的网络设计，例如修补输入 [46] 或自定义感受野大小 [2]，这可能会使这些网络转移到下游任务变得复杂。

Another mechanism is the **memory bank** approach proposed by [61] (Figure 2b). A memory bank consists of the representations of all samples in the dataset. The dictionary for each mini-batch is randomly sampled from the memory bank with no back-propagation, so it can support a large dictionary size. However, the representation of a sample in the memory bank was updated when it was last seen, so the sampled keys are essentially about the encoders at multiple different steps all over the past epoch and thus are less con-sistent. A momentum update is adopted on the **memory bank **in [61]. Its momentum update is on the representations of the same sample, not the encoder. This momentum update is irrelevant to our method, because MoCo does not keep track of every sample. Moreover, our method is more memory-efficient and can be trained on billion-scale data, which can be intractable for a memory bank.

另一种机制是 [61] 提出的 **memory bank** 方法（图 2b）。 存储库由数据集中所有样本的表示组成。 每个 mini-batch 的字典是从内存库中随机采样的，没有反向传播，因此它可以支持大字典大小。 但是，存储库中样本的表示在上次看到时已更新，因此采样的密钥本质上是关于在过去 epoch 中多个不同步骤的编码器，因此一致性较差。 在[61]中的**记忆库**上采用了动量更新。 它的动量更新是在同一样本的表示上，而不是在编码器上。 这种动量更新与我们的方法无关，因为 MoCo 不会跟踪每个样本。 此外，我们的方法更节省内存，并且可以在十亿规模的数据上进行训练，这对于内存库来说是难以处理的。

Sec. 4 empirically compares these three mechanisms.

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Momentum%20Contrast%20for%20Unsupervised%20Visual%20Representation%20Learning/%E7%AE%97%E6%B3%951.png"/></div>

### 3.3. Pretext Task 代理任务

Contrastive learning can drive a variety of pretext tasks. As the focus of this paper is not on designing a new pretext task, we use a simple one mainly following the instance discrimination task in [61], to which some recent works [63,2] are related.

对比学习可以驱动各种代理任务。 由于本文的重点不是设计一个新的借口任务，我们使用一个简单的主要遵循 [61] 中的实例识别任务，最近的一些工作 [63,2] 与之相关。

Following [61], we consider a query and a key as a positive pair if they originate from the same image, and otherwise as a negative sample pair. Following [63,2], we take two random “views” of the same image under random data augmentation to form a positive pair. The queries and keys are respectively encoded by their encoders, $f_q$ and $f_k$. The encoder can be any convolutional neural network [39].

在[61]之后，如果查询和键来自同一图像，我们将它们视为正对，否则视为负样本对。 在[63,2]之后，我们在随机数据增强下对同一图像进行两个随机“视图”以形成正对。 查询和键分别由它们的编码器 $f_q$ 和 $f_k$ 编码。 编码器可以是任何卷积神经网络[39]。

Algorithm 1 provides the pseudo-code of MoCo for this pretext task. For the current mini-batch, we encode the queries and their corresponding keys, which form the positive sample pairs. The negative samples are from the queue.

**Technical details.** We adopt a ResNet [33] as the encoder, whose last fully-connected layer (after global average pooling) has a fixed-dimensional output (128-D [61]). This output vector is normalized by its L2-norm [61]. This is the representation of the query or key. The temperature $\tau$ in Eqn.(1) is set as 0.07 [61]. The data augmentation setting follows [61]: a $224 \times 224$-pixel crop is taken from a randomly resized image, and then undergoes random color jittering, random horizontal flip, and random grayscale conversion, all available in PyTorch’s torchvision package.

**Shuffling BN.** Our encoders $f_q$ and $f_k$ both have Batch Normalization (BN) [37] as in the standard ResNet [33]. In experiments, we found that using BN prevents the model from learning good representations, as similarly reported in [35] (which avoids using BN). The model appears to “cheat” the pretext task and easily finds a low-loss solution. This is possibly because the intra-batch communication among samples (caused by BN) leaks information.

**洗牌 BN。** 我们的编码器 $f_q$ 和 $f_k$ 都具有批量归一化 (BN) [37]，与标准 ResNet [33] 中的一样。 在实验中，我们发现使用 BN 会阻止模型学习良好的表示，正如 [35] 中类似报道的那样（避免使用 BN）。 该模型似乎“欺骗”了代理任务，并且很容易找到低损失的解决方案。 这可能是因为样本之间的批内通信（由 BN 引起）泄漏了信息。

We resolve this problem by shuffling BN. We train with multiple GPUs and perform BN on the samples independently for each GPU (as done in common practice). For the key encoder $f_k$, we shuffle the sample order in the current mini-batch before distributing it among GPUs (and shuffle back after encoding); the sample order of the mini-batch for the query encoder $f_q$ is not altered. This ensures the batch statistics used to compute a query and its positive key come from two different subsets. This effectively tackles the cheating issue and allows training to benefit from BN.

我们通过改组BN来解决这个问题。 我们使用多个 GPU 进行训练，并为每个 GPU 独立地对样本执行 BN（如通常做法那样）。 对于关键编码器 $f_k$，我们在将当前 mini-batch 中的样本顺序分配到 GPU 之前将其打乱（并在编码后重新打乱）； 查询编码器 $f_q$ 的小批量样本顺序没有改变。 这确保了用于计算查询的批处理统计信息及其正键来自两个不同的子集。 这有效地解决了作弊问题，并使培训受益于 BN。

We use shuffled BN in both our method and its end-to-end ablation counterpart (Figure 2a). It is irrelevant to the memory bank counterpart (Figure 2b), which does not suffer from this issue because the positive keys are from different mini-batches in the past.

我们在我们的方法及其端到端消融对应物中都使用了改组的 BN（图 2a）。 它与对应的内存库无关（图 2b），它不会受到这个问题的影响，因为正键来自过去的不同 mini-batch。

## 4. Experiments 实验

We study unsupervised training performed in:

我们研究在以下方面进行的无监督训练：

***ImageNet-1M* (IN-1M)**: This is the ImageNet [11] training set that has ~1.28 million images in 1000 classes (often called ImageNet-1K; we count the image number instead, as classes are not exploited by unsupervised learning). This dataset is well-balanced in its class distribution, and its images generally contain iconic view of objects.

***ImageNet-1M* (IN-1M)**：这是 ImageNet [11] 训练集，在 1000 个类中有约 128 万张图像（通常称为 ImageNet-1K；我们计算图像数量，因为类是 未被无监督学习利用）。 该数据集的类别分布非常平衡，其图像通常包含对象的标志性视图。

**Instagram-1B (IG-1B)**: Following [44], this is a dataset of ~1 billion (940M) public images from Instagram. The images are from ~1500 hashtags [44] that are related to the ImageNet categories. This dataset is relatively *uncurated* comparing to IN-1M, and has a *long-tailed, unbalanced* distribution of real-world data. This dataset contains both iconic objects and scene-level images.

**Instagram-1B (IG-1B)**：继 [44] 之后，这是一个来自 Instagram 的约 10 亿（9.4 亿）张公共图像的数据集。 这些图像来自与 ImageNet 类别相关的约 1500 个主题标签 [44]。 与 IN-1M 相比，该数据集相对 *uncurated*，并且具有 *长尾、不平衡*的真实数据分布。 该数据集包含标志性对象和场景级图像。

**Training**. We use SGD as our optimizer. The SGD weight decay is 0.0001 and the SGD momentum is 0.9. For IN-1M, we use a mini-batch size of 256 ($N$ in Algorithm 1) in 8 GPUs, and an initial learning rate of 0.03. We train for 200 epochs with the learning rate multiplied by 0.1 at 120 and 160 epochs [61], taking 53 hours training ResNet-50. For IG-1B, we use a mini-batch size of 1024 in 64 GPUs, and a learning rate of 0.12 which is exponentially decayed by $0.9 \times$ after every 62.5k iterations (64M images). We train for 1.25M iterations (~1.4 epochs of IG-1B), taking 6 days for ResNet-50.

**训练**。 我们使用 SGD 作为优化器。 SGD 权重衰减为 0.0001，SGD 动量为 0.9。 对于 IN-1M，我们在 8 个 GPU 中使用 256 的小批量（算法 1 中的 $N$），初始学习率为 0.03。 我们训练了 200 个 epoch，学习率在 120 和 160 个 epoch [61] 时乘以 0.1，用了 53 小时训练 ResNet-50。 对于 IG-1B，我们在 64 个 GPU 中使用 1024 的 mini-batch 大小，学习率为 0.12，在每 62.5k 次迭代（6400 万张图像）后指数衰减 $0.9\times$。 我们训练了 125 万次迭代（IG-1B 的约 1.4 个 epoch），ResNet-50 用了 6 天。

### 4.1. Linear Classification Protocol 线性分类协议

We first verify our method by linear classification on *frozen* features, following a common protocol. In this subsection we perform unsupervised pre-training on IN-1M. Then we freeze the features and train a supervised linear classifier (a fully-connected layer followed by softmax). We train this classifier on the global average pooling features of a ResNet, for 100 epochs. We report 1-crop, top-1 classification accuracy on the ImageNet validation set.

我们首先按照通用协议通过对 *frozen* 特征的线性分类来验证我们的方法。 在本小节中，我们对 IN-1M 进行无监督预训练。 然后我们冻结这些特征并训练一个有监督的线性分类器（一个全连接层，然后是 softmax）。 我们在 ResNet 的全局平均池化特征上训练该分类器 100 个 epoch。 我们在 ImageNet 验证集上报告 1-crop，top-1 分类准确度。

For this classifier, we perform a grid search and find the optimal initial learning rate is 30 and weight decay is 0 (similarly reported in [56]). These hyper-parameters perform consistently well for all ablation entries presented in this subsection. These hyper-parameter values imply that the feature distributions ($e.g.,$ magnitudes) can be substantially different from those of ImageNet supervised training, an issue we will revisit in Sec. 4.2.

对于这个分类器，我们执行网格搜索，发现最佳初始学习率为 30【做对比学习可以将学习率上调到一个比较大的值】，权重衰减为 0（类似地在 [56] 中报道）。 对于本小节中介绍的所有消融条目，这些超参数始终表现良好。 这些超参数值意味着特征分布（例如，幅度）可能与 ImageNet 监督训练的特征分布有很大不同，我们将在 Sec. 中重新讨论这个问题。 4.2.

**Ablation: contrastive loss mechanisms.** We compare the three mechanisms that are illustrated in Figure 2. To focus on the effect of contrastive loss mechanisms, we implement all of them in the same pretext task as described in Sec. 3.3. We also use the same form of InfoNCE as the contrastive loss function, Eqn.(1). As such, the comparison is solely on the three mechanisms.

The results are in Figure 3. Overall, all three mechanisms benefit from a larger $K$. A similar trend has been observed in [61,56] under the memory bank mechanism, while here we show that this trend is more general and can be seen in all mechanisms. These results support our motivation of building a large dictionary.

The **end-to-end** mechanism performs similarly to MoCo when $K$ is small. However, the dictionary size is limited by the mini-batch size due to the end-to-end requirement. Here the largest mini-batch a high-end machine (8 Volta 32GB GPUs) can afford is 1024. More essentially, large mini-batch training is an open problem [25]: we found it necessary to use the linear learning rate scaling rule [25] here, without which the accuracy drops (by ~2% with a 1024 mini-batch). But optimizing with a larger mini-batch is harder [25], and it is questionable whether the trend can be extrapolated into a larger K even if memory is sufficient.

<div align =center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Momentum%20Contrast%20for%20Unsupervised%20Visual%20Representation%20Learning/Fig%203.png"/></div>

Figure 3. **Comparison of three contrastive loss mechanisms** under the ImageNet linear classification protocol. We adopt the same pretext task (Sec. 3.3) and only vary the contrastive loss mechanism (Figure 2). The number of negatives is $K$ in memory bank and MoCo, and is $K—1$ in end-to-end (offset by one because the positive key is in the same mini-batch). The network is ResNet-50.

图 3.**ImageNet 线性分类协议下三种对比损失机制的比较**。 我们采用相同的借口任务（第 3.3 节），仅改变对比损失机制（图 2）。 记忆库和 MoCo 中的负数是 $K$，端到端是 $K-1$（偏移 1，因为正键在同一个 mini-batch 中）。 网络是 ResNet-50。

The **memory bank** [61] mechanism can support a larger dictionary size. But it is 2.6% worse than MoCo. This is inline with our hypothesis: the keys in the memory bank are from very different encoders all over the past epoch and they are not consistent. Note the memory bank result of 58.0% reflects our improved implementation of [61].$^6$

>$^6$ Here 58.0% is with InfoNCE and $K=65536$. We reproduce 54.3%
when using NCE and $K=4096$ (the same as [61]), close to 54.0% in [61].

**Ablation: momentum.** The table below shows ResNet-50 accuracy with different MoCo momentum values (m in Eqn.(2)) used in pre-training ($K = 4096$ here) :

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Momentum%20Contrast%20for%20Unsupervised%20Visual%20Representation%20Learning/table%20-1.png"/></div>

It performs reasonably well when m is in $0.99 \sim 0.9999$,
showing that a slowly progressing ($i.e.,$ relatively large mo-mentum) key encoder is beneficial. When m is too small ($e.g.,$ 0.9), the accuracy drops considerably; at the extreme of *no momentum* ($m$ is 0), the training loss oscillates and fails to converge. These results support our motivation of building a consistent dictionary.

**Comparison with previous results**. Previous unsupervised learning methods can differ substantially in model sizes. For a fair and comprehensive comparison, we report **accuracy vs. #parameters$^3$**  trade-offs. Besides ResNet-50 (R50) [33], we also report its variants that are 2 and 4 wider (more channels), following [38].$^4$  We set $K = 65536$ and $m = 0.999$. Table 1 is the comparison.

>$^3$ Parameters are of the feature extractor: $e.g.,$ we do not count the parameters of $conv_\mathbf{x}$ if $conv_\mathbf{x}$ is not included in linear classification.
>$^4$ Our w$2 \times$ and w$4 \times$ models correspond to the “$\times 8$” and “$\times 16$” cases in [38], because the standard-sized ResNet is referred to as “$\times 4$” in [38].

MoCo with R50 performs competitively and achieves 60.6% accuracy, better than all competitors of similar model sizes (~24M). MoCo benefits from larger models and achieves 68.6% accuracy with R50w$4 \times$.

Notably, we achieve competitive results using a *standard* ResNet-50 and require no specific architecture designs, $e.g.,$ patchified inputs [46, 35], carefully tailored receptive fields [2], or combining two networks [56]. By using an architecture that is not customized for the pretext task, it is easier to transfer features to a variety of visual tasks and make comparisons, studied in the next subsection.

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Momentum%20Contrast%20for%20Unsupervised%20Visual%20Representation%20Learning/Table%201.png"/></div>

Table 1. **Comparison under the linear classification protocol on ImageNet.** The figure visualizes the table. All are reported as unsupervised pre-training on the ImageNet-1M training set, followed by supervised linear classification trained on frozen features, evaluated on the validation set. The parameter counts are those of the feature extractors. We compare with improved reimplementations if available (referenced after the numbers).
Notations: R101*/R170* is ResNet-101/170 with the last residual stage
removed [14,46,35], and R170 is made wider [35]; Rv50 is a reversible net [23], RX50 is ResNeXt-50-32$\times$8d [62].
$^†$: Pre-training uses FastAutoAugment [40] that is supervised by ImageNet labels.

表 1. **ImageNet 上线性分类协议下的比较。** 该图将表格可视化。 所有都报告为在 ImageNet-1M 训练集上进行无监督预训练，然后是在冻结特征上训练的监督线性分类，在验证集上进行评估。 参数计数是特征提取器的那些。 如果可用，我们将与改进的重新实现进行比较（在数字后引用）。
符号：R101*/R170* 是带有最后一个残差阶段的 ResNet-101/170
移除 [14,46,35]，R170 变宽 [35]； Rv50 是可逆网络 [23]，RX50 是 ResNeXt-50-32$\times$8d [62]。
$^†$：预训练使用由 ImageNet 标签监督的 FastAutoAugment [40]。

This paper’s focus is on a mechanism for general contrastive learning; we do not explore orthogonal factors (such as specific pretext tasks) that may further improve accuracy. As an example, “MoCo v2” [8], an extension of a preliminary version of this manuscript, achieves 71.1% accuracy with R50 (up from 60.6%), given small changes on the data augmentation and output projection head [7]. We believe that this additional result shows the generality and robustness of the MoCo framework.

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Momentum%20Contrast%20for%20Unsupervised%20Visual%20Representation%20Learning/Table%202.png"/></div>

Table 2. **Object detection fine-tuned on PASCAL VOC** `trainval07+12`. Evaluation is on `test2007:` $\rm{AP}_{50}$ (default VOC metric), AP (COCO-style), and $\rm{AP}_{75}$, averaged over 5 trials. All are fine-tuned for 24k iterations (~23 epochs). In the brackets are the gaps to the ImageNet supervised pre-training counterpart. In green are the gaps of at least $+0.5$ point.

<div align =center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Momentum%20Contrast%20for%20Unsupervised%20Visual%20Representation%20Learning/Table%203.png"/></div>

Table 3. **Comparison of three contrastive loss mechanisms** on PASCAL VOC object detection, fine-tuned on `trainval07+12` and evaluated on `test2007` (averages over 5 trials). All models are implemented by us (Figure 3), pre-trained on IN-1M, and finetuned using the same settings as in Table 2.

### 4.2. Transferring Features 迁移特征

A main goal of unsupervised learning is to *learn features that are transferrable.* ImageNet supervised pre-training is most influential when serving as the initialization for fine-tuning in downstream tasks ($e.g.,$ [21,20,43,52]). Next we compare MoCo with ImageNet supervised pre-training, transferred to various tasks including PASCAL VOC [18], COCO [42], etc. As prerequisites, we discuss two important issues involved [31]: normalization and schedules.

无监督学习的一个主要目标是*学习可转移的特征。* ImageNet 监督预训练在用作下游任务微调的初始化时最具影响力（例如， [21,20,43,52] ）。 接下来，我们将 MoCo 与 ImageNet 监督预训练进行比较，迁移到各种任务，包括 PASCAL VOC [18]、COCO [42] 等。作为先决条件，我们讨论涉及的两个重要问题 [31]：归一化和调度。

**Normalization.** As noted in Sec. 4.1, features produced by unsupervised pre-training can have different distributions compared with ImageNet supervised pre-training. But a system for a downstream task often has hyper-parameters ($e.g.,$ learning rates) selected for supervised pre-training. To relieve this problem, we adopt *feature normalization* during fine-tuning: we fine-tune with BN that is trained (and synchronized across GPUs [49]), instead of freezing it by an affine layer [33]. We also use BN in the newly initialized layers ($e.g.,$ FPN [41]), which helps calibrate magnitudes.

**归一化。** 如第 2 节所述。 4.1，与ImageNet监督预训练相比，无监督预训练产生的特征可以有不同的分布。 但是用于下游任务的系统通常具有选择用于监督预训练的超参数（例如，学习率）。 为了缓解这个问题，我们在微调期间采用*特征归一化*：我们使用经过训练（并在 GPU 之间同步 [49]）的 BN 进行微调，而不是通过仿射层 [33] 冻结它。 我们还在新初始化的层中使用了 BN（$e.g.,$FPN [41]），这有助于校准幅度【调整值域的大小】。

We perform normalization when fine-tuning supervised and unsupervised pre-training models. MoCo $\underline{use \ the \ same \ hyper{-}parameters} $ as the ImageNet supervised counterpart.

我们在微调监督和无监督预训练模型时执行归一化。 MoCo $\underline{使用同样的超参数}$ 作为 ImageNet 监督的对应物。

**Schedules.** If the fine-tuning schedule is long enough, training detectors from random initialization can be strong baselines, and can match the ImageNet supervised counterpart on COCO [31]. Our goal is to investigate *transferability* of features, so our experiments are on controlled schedules, $e.g.,$ the $1 \times$ (~12 epochs) or $2 \times$ schedules [22] for COCO, in contrast to $6 \times \sim 9 \times$ in [31]. On smaller datasets like VOC, training longer may not catch up [31].

**时间表。**如果微调时间表足够长，从随机初始化训练检测器可以得到强的基线【较好的结果】，并且可以匹配 COCO [31] 上的 ImageNet 监督对应物。 我们的目标是研究特征的*可转移性*，所以我们的实验是在受控的时间表上进行的，如$1\times$（~12 epochs）或 $2\times$ 的 COCO 时间表 [22]，与在 [31] 中$6\ \times \sim 9 \times$相比。 在 VOC 等较小的数据集上，训练时间可能赶不上 [31]。

Nonetheless, in our fine-tuning, MoCo $\underline{uses \ the \ same \ schedule}$ as the ImageNet supervised counterpart, and random initialization results are provided as references.

尽管如此，在我们的微调中，MoCo $\underline{使用同样的计算时间}$ 作为 ImageNet 监督对应物，并提供随机初始化结果作为参考。

Put together, our fine-tuning uses the same setting as the supervised pre-training counterpart. This may place MoCo at a *disadvantage*. Even so, MoCo is competitive. Doing so also makes it feasible to present comparisons on multiple datasets/tasks, without extra hyper-parameter search.

总而言之，我们的微调使用与有监督的预训练对应物相同的设置。 这可能会使 MoCo 处于*劣势*。 即便如此，MoCo 还是很有竞争力的。 这样做还可以在多个数据集/任务上进行比较，而无需额外的超参数搜索。

### 4.2.1 PASCAL VOC Object Detection

**Setup.** The detector is Faster R-CNN [52] with a backbone of R50-dilated-C5 or R50-C4 [32] (details in appendix), with BN tuned, implemented in [60]. We fine-tune all layers end-to-end. The image scale is [480, 800] pixels during training and 800 at inference. The same setup is used for all entries, including the supervised pre-training baseline. We evaluate the default VOC metric of $\rm{AP}_{50}$ ($i.e.,$ IoU threshold is $50\%$) and the more stringent metrics of COCO-style AP and $\rm{AP}_{75}$. Evaluation is on the VOC `test2007` set.

**Ablation: backbones.** Table 2 shows the results fine-tuned on `trainval07+12` (∼16.5k images). For R50-dilatedC5 (Table 2a), MoCo pre-trained on IN-1M is comparable to the supervised pre-training counterpart, and MoCo pre-trained on IG-1B *surpasses* it. For R50-C4 (Table 2b),MoCo with IN-1M or IG-1B is *better* than the supervised counterpart: up to $\mathbf{+0.9}$ $\rm{AP}_{50}$, $\mathbf{+3.7}$ AP, and $\mathbf{+4.9}$ $\rm{AP}_{75}$.

**消融：主干。** 表 2 显示了在“trainval07+12”（~16.5k 图像）上微调的结果。 对于 R50-dilatedC5（表 2a），在 IN-1M 上预训练的 MoCo 与有监督的预训练对应物相当，并且在 IG-1B 上预训练的 MoCo *超过*它。 对于 R50-C4（表 2b），带有 IN-1M 或 IG-1B 的 MoCo 比有监督的对应物*更好*：高达 $\mathbf{+0.9}$ $\rm{AP}_{50}$, $\mathbf{+ 3.7}$ AP 和 $\mathbf{+4.9}$ $\rm{AP}_{75}$。

Interestingly, the transferring accuracy depends on the detector structure. For the C4 backbone, by default used in existing ResNet-based results [14,61,26,66], the advantage of unsupervised pre-training is larger. The relation between pre-training vs. detector structures has been veiled in the past, and should be a factor under consideration.

**Ablation: contrastive loss mechanisms.** We point out that these results are partially because we establish solid detection baselines for contrastive learning. To pin-point the gain that is *solely* contributed by using the MoCo mechanism in contrastive learning, we fine-tune the models pre-trained with the end-to-end or memory bank mechanism, both implemented by us (i.e., the best ones in Figure 3), using the same fine-tuning setting as MoCo.

These competitors perform decently (Table 3). Their AP and AP75 with the C4 backbone are also higher than the ImageNet supervised counterpart’s, c.f . Table 2b, but other metrics are lower. They are worse than MoCo in all metrics. This shows the benefits of MoCo. In addition, how to train these competitors in larger-scale data is an open question, and they may not benefit from IG-1B.

**Comparison with previous results.** Following the competitors, we fine-tune on `trainval2007` (5k images) using the C4 backbone. The comparison is in Table 4.

<div align =center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Momentum%20Contrast%20for%20Unsupervised%20Visual%20Representation%20Learning/Table%204.png"/></div>

Table 4. **Comparison with previous methods on object detection fine-tuned on PASCAL VOC** `trainval2007`. Evaluation is on `test2007`. The ImageNet supervised counterparts are from the respective papers, and are reported as having $the same structure$ as the respective unsupervised pre-training counterparts. All entries are based on the C4 backbone. The models in [14] are R101 v2 [34], and others are R50. The RelPos (relative position) [13] result is the best single-task case in the Multi-task paper [14]. The Jigsaw [45] result is from the ResNet-based implementation in [26]. Our results are with 9k-iteration fine-tuning, averaged over 5 trials. In the brackets are the
gaps to the ImageNet supervised pre-training counterpart. In green are the gaps of at least $+0.5$ point.

For the ${\rm AP}_{50}$ metric, *no* previous method can catch up with its respective supervised pre-training counterpart. MoCo pre-trained on any of IN-1M, IN-14M (full Ima- geNet), YFCC-100M [55], and IG-1B can *outperform* the supervised baseline. Large gains are seen in the more stringent metrics: up to $\mathbf{+5.2}$ AP and $\mathbf{+9.0}$ ${\rm AP}_{75}$. These gains are larger than the gains seen in `trainval07+12` (Table 2b).

### 4.2.2 COCO Object Detection and Segmentation

**Setup.** The model is Mask R-CNN [32] with the FPN [41] or C4 backbone, with BN tuned, implemented in [60]. The image scale is in [640, 800] pixels during training and is 800 at inference. We fine-tune all layers end-to-end. We fine-tune on the `train2017` set (118k images) and evaluate on `val2017`. The schedule is the default $1 \times $ or $2 \times$in [22].

**Results.** Table 5 shows the results on COCO with the FPN (Table 5a, b) and C4 (Table 5c, d) backbones. With the $1 \times$ schedule, all models (including the ImageNet supervised counterparts) are heavily under-trained, as indicated by the ~2 points gaps to the $2 \times $ schedule cases. With the $2 \times $ schedule, MoCo is better than its ImageNet supervised counterpart in all metrics in both backbones.

### 4.2.3 More Downstream Tasks

Table 6 shows more downstream tasks (implementation details in appendix). Overall, MoCo performs competitively with ImageNet supervised pre-training:

*COCO keypoint detection*: supervised pre-training has no clear advantage over random initialization, whereas MoCo outperforms in all metrics.

*COCO dense pose estimation*[1]: MoCo substantially outperforms supervised pre-training, $e.g.,$ by 3.7 points in AP$^{dp}_{75}$, in this highly localization-sensitive task.

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Momentum%20Contrast%20for%20Unsupervised%20Visual%20Representation%20Learning/Table%205.png"/></div>

Table 5. **Object detection and instance segmentation fine-tuned on COCO**: bounding-box AP (AP$^{\rm bb}$) and mask AP (AP$^{\rm mk}$) evaluated on `val2017`. In the brackets are the gaps to the ImageNet supervised pre-training counterpart. In green are the gaps of at least $+0.5$ point.

<div algin =center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Momentum%20Contrast%20for%20Unsupervised%20Visual%20Representation%20Learning/Table%206.png"/></div>

Table 6. **MoCo vs. ImageNet supervised pre-training, fine-tuned on various tasks.** For each task, the same architecture and schedule are used for all entries (see appendix). In the brackets are the gaps to the ImageNet supervised pre-training counterpart. In green are the gaps of at least $+0.5$ point.
†: this entry is with BN frozen, which improves results; see main text.

LVIS v0.5 *instance segmentation* [27]: this task has ~1000 long-tailed distributed categories. Specifically in LVIS for the ImageNet supervised baseline, we find finetuning with frozen BN (24.4 AP$^{\rm mk}$) is better than tunable BN (details in appendix). So we compare MoCo with the better supervised pre-training variant in this task. MoCo with IG-1B surpasses it in all metrics.

*Cityscapes instance segmentation*[10]: MoCo with IG-1B is on par with its supervised pre-training counterpart in AP$^{\rm mk}$, and is higher in AP$^{\rm mk}_{\rm 50}$.

*Semantic segmentation:* On Cityscapes [10], MoCo out-performs its supervised pre-training counterpart by up to 0.9 point. But on VOC semantic segmentation, MoCo is worse by at least 0.8 point, a negative case we have observed.

**Summary.** In sum, MoCo can ***outperform*** its ImageNet supervised pre-training counterpart in 7 detection or segmentation tasks.$^5$  Besides, MoCo is on par on Cityscapes instance segmentation, and lags behind on VOC semantic segmentation; we show another comparable case on iNaturalist [57] in appendix. *Overall, MoCo has largely closed the gap between unsupervised and supervised representation learning in multiple vision tasks.*

**总结。** 总之，MoCo 在 7 个检测或分割任务中可以 ***优于*** 其 ImageNet 监督预训练对手。$^5$ 此外，MoCo 与 Cityscapes 实例分割相当，但落后 关于VOC语义分割； 我们在附录中展示了另一个关于 iNaturalist [57] 的可比较案例。 *总体而言，MoCo 在很大程度上缩小了多视觉任务中无监督和有监督表示学习之间的差距。*

>$^5$ Namely, object detection on VOC/COCO, instance segmentation on COCO/LVIS, keypoint detection on COCO, dense pose on COCO, and semantic segmentation on Cityscapes.

Remarkably, in all these tasks, MoCo pre-trained on IG-1B is consistently better than MoCo pre-trained on IN-1M. This shows that *MoCo can perform well on this large-scale, relatively uncurated dataset.* This represents a scenario towards *real-world* unsupervised learning.

## 5. Discussion and Conclusion 讨论和结论

Our method has shown positive results of unsupervised learning in a variety of computer vision tasks and datasets. A few open questions are worth discussing. MoCo’s improvement from IN-1M to IG-1B is consistently noticeable but relatively small, suggesting that the larger-scale data may not be fully exploited. We hope an advanced pretext task will improve this. Beyond the simple instance discrimination task [61], it is possible to adopt MoCo for pretext tasks like masked auto-encoding, $e.g.,$ in language [12] and in vision [46]. We hope MoCo will be useful with other pretext tasks that involve contrastive learning.

我们的方法在各种计算机视觉任务和数据集中显示了无监督学习的积极结果。一些开放的问题值得讨论。MoCo从IN-1M到IG-1B的改进一直很明显，但相对较小，这表明更大规模的数据可能没有得到充分利用。我们希望一项先进的代理任务将改善这一点。除了简单的实例识别任务[61]之外，还可以采用MoCo作为代理任务，如掩蔽自动编码，例如语言中的[12]和视觉中的[46]。我们希望MoCo对其他涉及对比学习的代理任务有用。

## A. Appendix

### A.1. Implementation: Object detection backbones

The R50-dilated-C5 and R50-C4 backbones are similar to those available in `Detectron2` [60]: (i) R50-dilated- C5: the backbone includes the ResNet conv5 stage with a dilation of 2 and stride 1, followed by a $3 \times 3$ convolution (with BN) that reduces dimension to 512. The box prediction head consists of two hidden fully-connected layers. (ii) *R50-C4*: the backbone ends with the conv$_4$ stage, and the box prediction head consists of the conv$_5$ stage (including global pooling) followed by a BN layer.

### A.2. Implementation: COCO keypoint detection

We use Mask R-CNN (keypoint version) with R50-FPN, implemented in [60], fine-tuned on COCO train2017 and evaluated on val2017. The schedule is $2 \times$.

### A.3. Implementation: COCO dense pose estimation

We use DensePose R-CNN [1] with R50-FPN, implemented in [60], fine-tuned on COCO `train2017` and evaluated on `val2017`. The schedule is $\mathbf{s}1 \times$.

### A.4. Implementation: LVIS instance segmentation

We use Mask R-CNN with R50-FPN, fine-tuned in LVIS [27] `train_v0.5` and evaluated in `val_v0.5`. We follow the baseline in [27] (arXiv v3 Appendix B).

LVIS is a new dataset and model designs on it are to be explored. The following table includes the relevant ablations (all are averages of 5 trials):

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Momentum%20Contrast%20for%20Unsupervised%20Visual%20Representation%20Learning/table%20-2.png"/></div>

A supervised pre-training baseline, end-to-end tuned but with BN frozen, has 24.4 AP$^{\rm mk}$. But tuning BN in this baseline leads to worse results and overfitting (this is unlike on COCO/VOC where tuning BN gives better or comparable accuracy). MoCo has 24.1 AP$^{\rm mk}$ with IN-1M and 24.9 AP$^{\rm mk}$ with IG-1B, both outperforming the supervised pretraining counterpart under the same tunable BN setting. Under the best individual settings, MoCo can still outperform the supervised pre-training case (24.9 vs. 24.4, as reported in Table 6 in Sec 4.2).

### A.5. Implementation: Semantic segmentation

We use an FCN-based [43] structure. The backbone consists of the convolutional layers in R50, and the 33 convolutions in conv5 blocks have dilation 2 and stride 1. This is followed by two extra $3 \times 3$ convolutions of 256 channels, with BN and ReLU, and then a $1 \times 1$ convolution for perpixel classification. The total stride is 16 (FCN-16s [43]). We set dilation $= 6$ in the two extra $3 \times 3$  convolutions, following the large field-of-view design in [6].

Training is with random scaling (by a ratio in [0.5, 2.0]), cropping, and horizontal flipping. The crop size is 513 on VOC and 769 on Cityscapes [6]. Inference is performed on the original image size. We train with mini-batch size 16 and weight decay 0.0001. Learning rate is 0.003 on VOC and is 0.01 on Cityscapes (multiplied by 0.1 at 70th and 90-th percentile of training). For VOC, we train on the `train_aug2 012` set (augmented by [30], 10582 images) for 30k iterations, and evaluate on `val2012`. For Cityscapes, we train on the `train_fine` set (2975 images) for 90k iterations, and evaluate on the `val` set. Results are reported as averages over 5 trials.

### A.6. iNaturalist fine-grained classification

In addition to the detection/segmentation experiments in the main paper, we study fine-grained classification on the iNaturalist 2018 dataset [57]. We fine-tune the pretrained models end-to-end on the `train` set (~437k images, 8142 classes) and evaluate on the `val` set. Training follows the typical ResNet implementation in PyTorch with 100 epochs. Fine-tuning has a learning rate of 0.025 (vs. 0.1 from scratch) decreased by 10 at the 70-th and 90-th percentile of training. The following is the R50 result:

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Momentum%20Contrast%20for%20Unsupervised%20Visual%20Representation%20Learning/Table%20-3.png"/></div>

MoCo is ~4% better than training from random initialization, and is closely comparable with its ImageNet supervised counterpart. This again shows that MoCo unsupervised pre-training is competitive.

### A.7. Fine-tuning in ImageNet

Linear classification on frozen features (Sec. 4.1) is a common protocol of evaluating unsupervised pre-training methods. However, in practice, it is more common to finetune the features end-to-end in a downstream task. For completeness, the following table reports end-to-end finetuning results for the 1000-class ImageNet classification, compared with training from scratch (fine-tuning uses an initial learning rate of 0.03, vs. 0.1 from scratch):

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Momentum%20Contrast%20for%20Unsupervised%20Visual%20Representation%20Learning/Table%20-4.png"/></div>

As here ImageNet is the downstream task, the case of MoCo pre-trained on IN-1M does not represent a real scenario (for reference, we report that its accuracy is 77.0% after fine-tuning). But unsupervised pre-training in the *separate*, unlabeled dataset of IG-1B represents a typical scenario: in this case, MoCo improves by 0.8%.

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Momentum%20Contrast%20for%20Unsupervised%20Visual%20Representation%20Learning/Table%20A1.png"/></div>

Table A.1. Object detection and instance segmentation fine-tuned on COCO: $2× vs. 6× \mathbf{schedule}$. In the brackets are the gaps to the ImageNet supervised pre-training counterpart. In green are the gaps of at least $+0.5$ point.

### A.8. COCO longer fine-tuning

In Table 5 we reported results of the 1$\times$ (~12 epochs) and 2$\times$ schedules on COCO. These schedules were inherited from the original Mask R-CNN paper [32], which could be suboptimal given later advance in the field. In Table A.1, we supplement the results of a 6$\times$ schedule (~72 epochs) [31] and compare with those of the 2$\times$ schedule.

We observe: (i) fine-tuning with ImageNet-supervised pre-training still has improvements (41.9 AP$^{\rm bb}$); (ii) training from scratch largely catches up (41.4 AP$^{\rm bb}$); (iii) the MoCo counterparts improve further ($e.g.,$ to 42.8 AP$^{\rm bb}$) and have larger gaps ($e.g.,$ $+0.9$ AP$^{\rm bb}$ with 6, vs. $+0.5$ AP$^{\rm bb}$ with 2$\times$). Table A.1 and Table 5 suggest that the MoCo pre-trained features can have *larger* advantages than the ImageNet-supervised features when fine-tuning *longer*.

### A.9. Ablation on Shuffling BN

Figure A.1 provides the training curves of MoCo with or without shuffling BN: removing shuffling BN shows obvious overfitting to the pretext task: training accuracy of the pretext task (dash curve) quickly increases to >99.9%, and the kNN-based validation classification accuracy (solid curve) drops soon. This is observed for both the MoCo and end-to-end variants; the memory bank variant implicitly has different statistics for $q$ and $k$, so avoids this issue.

These experiments suggest that without shuffling BN, the sub-batch statistics can serve as a “signature” to tell which sub-batch the positive key is in. Shuffling BN can remove this signature and avoid such cheating.

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Momentum%20Contrast%20for%20Unsupervised%20Visual%20Representation%20Learning/Fig%20A1.png"/></div>

Figure A.1. **Ablation of Shuffling BN.** Dash: training curve of the pretext task, plotted as the accuracy of $(K+1)$-way dictionary lookup. *Solid*: validation curve of a kNN-based monitor [61] (not a linear classifier) on ImageNet classification accuracy. This plot shows the first 80 epochs of training: training longer without shuffling BN overfits more.

## References  参考文献

[1] Riza Alp GUler, Natalia Neverova, and lasonas Kokkinos. DensePose: Dense human pose estimation in the wild. In CVPR, 2018.

[2] Philip Bachman, R Devon Hjelm, and William Buchwalter. Learning representations by maximizing mutual information across views. arXiv:1906.00910, 2019.

[3] Mathilde Caron, Piotr Bojanowski, Armand Joulin, and Matthijs Douze. Deep clustering for unsupervised learning of visual features. In ECCV, 2018.

[4] Mathilde Caron, Piotr Bojanowski, Julien Mairal, and Armand Joulin. Unsupervised pre-training of image features on non-curated data. In ICCV, 2019.

[5] Ken Chatfield, Victor Lempitsky, Andrea Vedaldi, and Andrew Zisserman. The devil is in the details: an evaluation of recent feature encoding methods. In BMVC, 2011.

[6] Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, and Alan L Yuille. DeepLab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected CRFs. TPAMI, 2017.

[7] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. arXiv:2002.05709, 2020.

[8] Xinlei Chen, Haoqi Fan, Ross Girshick, and Kaiming He. Improved baselines with momentum contrastive learning. arXiv:2003.04297, 2020.

[9] Adam Coates and Andrew Ng. The importance of encoding versus training with sparse coding and vector quantization. In ICML, 2011.

[10] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The Cityscapes dataset for semantic urban scene understanding. In CVPR, 2016.

[11] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. ImageNet: A large-scale hierarchical image database. In CVPR, 2009.

[12] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional trans-formers for language understanding. In NAACL, 2019.

[13] Carl Doersch, Abhinav Gupta, and Alexei A Efros. Unsupervised visual representation learning by context prediction. In ICCV, 2015.

[14] Carl Doersch and Andrew Zisserman. Multi-task selfsupervised visual learning. In ICCV, 2017.

[15] Jeff Donahue, Philipp Krahenbuhl, and Trevor Darrell. Ad-versarial feature learning. In ICLR, 2017.

[16] Jeff Donahue and Karen Simonyan. Large scale adversarial representation learning. arXiv:1907.02544, 2019.

[17] Alexey Dosovitskiy, Jost Tobias Springenberg, Martin Ried- miller, and Thomas Brox. Discriminative unsupervised feature learning with convolutional neural networks. In NeurIPS, 2014.

[18] Mark Everingham, Luc Van Gool, Christopher KI Williams, John Winn, and Andrew Zisserman. The Pascal Visual Object Classes (VOC) Challenge. IJCV, 2010.

[19] Spyros Gidaris, Praveer Singh, and Nikos Komodakis. Un-supervised representation learning by predicting image rotations. In ICLR, 2018.

[20] Ross Girshick. Fast R-CNN. In ICCV, 2015.

[21] Ross Girshick, Jeff Donahue, Trevor Darrell, and Jitendra Malik. Rich feature hierarchies for accurate object detection and semantic segmentation. In CVPR, 2014.

[22] Ross Girshick, Ilija Radosavovic, Georgia Gkioxari, Piotr Dollar, and Kaiming He. Detection, 2018.

[23] Aidan N Gomez, Mengye Ren, Raquel Urtasun, and Roger B Grosse. The reversible residual network: Backpropagation without storing activations. In NeurIPS, 2017.

[24] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In NeurIPS, 2014.

[25] Priya Goyal, Piotr Dollar, Ross Girshick, Pieter Noord- huis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, and Kaiming He. Accurate, large minibatch SGD: Training ImageNet in 1 hour. arXiv:1706.02677, 2017.

[26] Priya Goyal, Dhruv Mahajan, Abhinav Gupta, and Ishan Misra. Scaling and benchmarking self-supervised visual representation learning. In ICCV, 2019.

[27] Agrim Gupta, Piotr Dollar, and Ross Girshick. LVIS: A dataset for large vocabulary instance segmentation. In CVPR, 2019.

[28] Michael Gutmann and Aapo Hyvarinen. Noise-contrastive estimation: A new estimation principle for unnormalized sta-tistical models. In AISTATS, 2010.

[29] Raia Hadsell, Sumit Chopra, and Yann LeCun. Dimensionality reduction by learning an invariant mapping. In CVPR, 2006.

[30] Bharath Hariharan, Pablo Arbelaez, Lubomir Bourdev, Subhransu Maji, and Jitendra Malik. Semantic contours from inverse detectors. In ICCV, 2011.

[31] Kaiming He, Ross Girshick, and Piotr Dollar. Rethinking ImageNet pre-training. In ICCV, 2019.

[32] Kaiming He, Georgia Gkioxari, Piotr Dollar, and Ross Gir- shick. Mask R-CNN. In ICCV, 2017.

[33] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, 2016.

[34] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Identity mappings in deep residual networks. In ECCV, 2016.

[35] Olivier J Henaff, Ali Razavi, Carl Doersch, SM Eslami, and Aaron van den Oord. Data-efficient image recognition with contrastive predictive coding. arXiv:1905.09272, 2019. Updated version accessed at <https://openreview.net/pdf?id=rJerHlrYwH.>

[36] R Devon Hjelm, Alex Fedorov, Samuel Lavoie-Marchildon, Karan Grewal, Adam Trischler, and Yoshua Bengio. Learning deep representations by mutual information estimation and maximization. In ICLR, 2019.

[37] Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal co-variate shift. In ICML, 2015.

[38] Alexander Kolesnikov, Xiaohua Zhai, and Lucas Beyer. Re-visiting self-supervised visual representation learning. In CVPR, 2019.

[39] Yann LeCun, Bernhard Boser, John S Denker, Donnie Henderson, Richard E Howard, Wayne Hubbard, and Lawrence D Jackel. Backpropagation applied to handwritten zip code recognition. Neural computation, 1989.

[40] Sungbin Lim, Ildoo Kim, Taesup Kim, Chiheon Kim, and Sungwoong Kim. Fast AutoAugment. arXiv:1905.00397, 2019.

[41] Tsung-Yi Lin, Piotr Dollar, Ross Girshick, Kaiming He, Bharath Hariharan, and Serge Belongie. Feature pyramid networks for object detection. In CVPR, 2017.

[42] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar, and C Lawrence Zitnick. Microsoft COCO: Common objects in context. In ECCV, 2014.

[43] Jonathan Long, Evan Shelhamer, and Trevor Darrell. Fully convolutional networks for semantic segmentation. In CVPR, 2015.

[44] Dhruv Mahajan, Ross Girshick, Vignesh Ramanathan, Kaiming He, Manohar Paluri, Yixuan Li, Ashwin Bharambe, and Laurens van der Maaten. Exploring the limits of weakly supervised pretraining. In ECCV, 2018.

[45] Mehdi Noroozi and Paolo Favaro. Unsupervised learning of visual representations by solving jigsaw puzzles. In ECCV, 2016.

[46] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Rep-resentation learning with contrastive predictive coding. arXiv:1807.03748, 2018.

[47] Deepak Pathak, Ross Girshick, Piotr Dollar, Trevor Darrell, and Bharath Hariharan. Learning features by watching objects move. In CVPR, 2017.

[48] Deepak Pathak, Philipp Krahenbuhl, Jeff Donahue, Trevor Darrell, and Alexei A Efros. Context encoders: Feature learning by inpainting. In CVPR, 2016.

[49] Chao Peng, Tete Xiao, Zeming Li, Yuning Jiang, Xiangyu Zhang, Kai Jia, Gang Yu, and Jian Sun. MegDet: A large mini-batch object detector. In CVPR, 2018.

[50] Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language understanding by generative pre-training. 2018.

[51] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. 2019.

[52] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster R-CNN: Towards real-time object detection with region proposal networks. In NeurIPS, 2015. 

[53] Karen Simonyan and Andrew Zisserman. Very deep convo-lutional networks for large-scale image recognition. In ICLR, 2015.

[54] Josef Sivic and Andrew Zisserman. Video Google: a text retrieval approach to object matching in videos. In ICCV, 2003.

[55] Bart Thomee, David A Shamma, Gerald Friedland, Benjamin Elizalde, Karl Ni, Douglas Poland, Damian Borth, and Li-Jia Li. YFCC100M: The new data in multimedia research. Communications of the ACM, 2016.

[56] Yonglong Tian, Dilip Krishnan, and Phillip Isola. Contrastive multiview coding. arXiv:1906.05849, 2019. Updated version accessed at <https://openreview.net/pdf?id=BkgStySKPB.>

[57] Grant Van Horn, Oisin Mac Aodha, Yang Song, Yin Cui, Chen Sun, Alex Shepard, Hartwig Adam, Pietro Perona, and Serge Belongie. The iNaturalist species classification and detection dataset. In CVPR, 2018.

[58] Pascal Vincent, Hugo Larochelle, Yoshua Bengio, and Pierre-Antoine Manzagol. Extracting and composing robust features with denoising autoencoders. In ICML, 2008.

[59] Xiaolong Wang and Abhinav Gupta. Unsupervised learning of visual representations using videos. In ICCV, 2015.

[60] Yuxin Wu, Alexander Kirillov, Francisco Massa, Wan-Yen Lo, and Ross Girshick. Detectron2. <https://github.com/facebookresearch/detectron2,》 2019.

[61] Zhirong Wu, Yuanjun Xiong, Stella Yu, and Dahua Lin. Un-supervised feature learning via non-parametric instance dis-crimination. In CVPR, 2018. Updated version accessed at: <https://arxiv.org/abs/1805.01978v1.>

[62] Saining Xie, Ross Girshick, Piotr Dollar, Zhuowen Tu, and Kaiming He. Aggregated residual transformations for deep neural networks. In CVPR, 2017.

[63] Mang Ye, Xu Zhang, Pong C Yuen, and Shih-Fu Chang. Un-supervised embedding learning via invariant and spreading instance feature. In CVPR, 2019.

[64] Richard Zhang, Phillip Isola, and Alexei A Efros. Colorful image colorization. In ECCV, 2016.

[65] Richard Zhang, Phillip Isola, and Alexei A Efros. Split-brain autoencoders: Unsupervised learning by cross-channel pre-diction. In CVPR, 2017.

[66] Chengxu Zhuang, Alex Lin Zhai, and Daniel Yamins. Local aggregation for unsupervised learning of visual embeddings. In ICCV, 2019. Additional results accessed from supplemen-tary materials.
