---
title: 【论文】End-to-End Object Detection with Transformers（DETR） 使⽤ Transformer 进⾏端到端⽬标检测
date: 2022-11-26 07:00:00 +/-0800
categories: [论文,深度学习]
tags: [深度学习,论文]     # TAG names should always be lowercase 标记名称应始终为小写
math: true
---

<div align=center>Nicolas Carion*, Francisco Massa*, Gabriel Synnaeve, Nicolas Usunier,</div>
<div align=center>Alexander Kirillov, and Sergey Zagoruyko</div>
<div align=center>Facebook AI</div>

> *Equal contribution 
>
> DETR实现了端到端的目标检测算法，解决了之前神经网络目标检测最后需要极大值抑制操作的问题 
> 
> 极大值抑制（Non-Maximum Suppression，NMS），顾名思义就是抑制不是极大值的元素，可以理解为局部最大搜索。这个局部代表的是一个邻域，邻域有两个参数可变，一是邻域的维数，二是邻域的大小。这里不讨论通用的NMS算法(参考论文《Efficient Non-Maximum Suppression》对1维和2维数据的NMS实现)，而是用于目标检测中提取分数最高的窗口的。例如在行人检测中，滑动窗口经提取特征，经分类器分类识别后，每个窗口都会得到一个分数。但是滑动窗口会导致很多窗口与其他窗口存在包含或者大部分交叉的情况。这时就需要用到NMS来选取那些邻域里分数最高（是行人的概率最大），并且抑制那些分数低的窗口。NMS在计算机视觉领域有着非常重要的应用，如视频目标跟踪、数据挖掘、3D重建、目标识别以及纹理分析等
>
> 不需要anchor,也不需要proposal,利用了Transformer全局建模的能力。自注意可以检测图像边缘

## Abstract 摘要

We present a new method that views object detection as a direct set prediction problem. Our approach streamlines the detection pipeline, effectively removing the need for many hand-designed components like a non-maximum suppression procedure or anchor generation that explicitly encode our prior knowledge about the task. The main ingredients of the new framework, called DEtection TRansformer or DETR, are a set-based global loss that forces unique predictions via bipartite matching, and a transformer encoder-decoder architecture. Given a fixed small set of learned object queries, DETR reasons about the relations of the objects and the global image context to directly output the final set of predictions in parallel. The new model is conceptually simple and does not require a specialized library, unlike many other modern detectors. DETR demonstrates accuracy and runtime performance on par with the well-established and highly-optimized Faster R-CNN baseline on the challenging COCO object detection dataset. Moreover, DETR can be easily generalized to produce panoptic segmentation in a unified manner. We show that it significantly outperforms competitive baselines. Training code and pretrained models are available at <https://github.com/facebookresearch/detr>.

我们提出了一种将对象检测视直接视为集合预测问题的新方法。我们的方法简化了检测流程，有效地消除了对许多手动设计组件的需求，例如非最大抑制程序或锚点生成，这些组件明确编码了我们关于任务的先验知识。称为 DEtection TRAnsformer 或 DETR 的新框架的主要成分是基于集合的全局损失，它通过二分匹配强制进行独特的预测【理想下生成一个框】，以及一个转换器编码器-解码器架构。给定一小组固定的学习对象查询，DETR 推理对象的关系和全局图像上下文以直接并行输出最终的预测集【直接输出预测框】。与许多其他现代检测器不同，新模型在概念上很简单，不需要专门的库。 DETR 在具有挑战性的 COCO 对象检测数据集上展示了与完善且高度优化的 Faster R-CNN 基线相当的准确性和运行时性能。此外，DETR 可以很容易地推广，以统一的方式产生全景分割。【只需要在DETR上加一个分割头就可以实现了】我们表明它明显优于竞争基线。 <https://github.com/facebookresearch/detr> 提供训练代码和预训练模型。

## 1. Introduction 引言

The goal of object detection is to predict a set of bounding boxes and category labels for each object of interest. Modern detectors address this set prediction task in an indirect way, by dening surrogate regression and classification problems on a large set of proposals [37,5], anchors [23], or window centers [53,46]. Their performances are significantly influenced by postprocessing steps to collapse near-duplicate predictions, by the design of the anchor sets and by the heuristics that assign target boxes to anchors [52]. To simplify these pipelines, we propose a direct set prediction approach to bypass the surrogate tasks. This end-to-end philosophy has led to significant advances in complex structured prediction tasks such as machine translation or speech recognition, but not yet in object detection: previous attempts [43,16,4,39] either add other forms of prior knowledge, or have not proven to be competitive with strong baselines on challenging benchmarks. This paper aims to bridge this gap.

对象检测的目标是为每个感兴趣的对象预测一组边界框和类别标签。【集合预测】现代检测器以间接的方式解决这个集合预测任务，通过在大量proposals提案[37,5]、anchor锚点 [23] 或窗口中心 [53,46] 上定义替代的回归和分类问题来实现。它们的性能受到后处理步骤的显着影响【即nsm】，以抑制近乎重复的预测，锚集的设计以及将目标框分配给锚点的启发式方法[52]。为了简化这些流程，我们提出了一种直接集预测方法来绕过代理任务。这种端到端的理念已经在机器翻译或语音识别等复杂的结构化预测任务中取得了重大进展，但在对象检测方面还没有：之前的尝试 [43,16,4,39] 要么添加其他形式的先验知识，或者尚未证明在具有挑战性的基准上具有强大的基线竞争力【没有取得很好的成绩】。本文旨在弥合这一差距。

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/End-to-End%20Object%20Detection%20with%20Transformers(DETR)/Fig%201.png"/></div>

Fig. 1: DETR directly predicts (in parallel) the final set of detections by combining a common CNN with a transformer architecture. During training, bipartite matching uniquely assigns predictions with ground truth boxes. Prediction with no match should yield a `"no object"` ($∅$) class prediction. \
图1：DETR通过将一个普通的CNN与一个Transformer架构相结合，直接预测（并行）最终的检测集合。在训练过程中，双边匹配唯一地将预测与地面真相框进行分配。没有匹配的预测应该产生一个 "无对象"（$∅$）类预测。【第一步CNN抽特征，第二步用Transformer-encoder学习特征，第三步用Transformer-decoder去生成预测框，第四步是用生成的预测框与基准算loss,进行训练，推理的时候在第四步用阈值卡一下置信度，根据置信度输出】

We streamline the training pipeline by viewing object detection as a direct set prediction problem. We adopt an encoder-decoder architecture based on transformers [47], a popular architecture for sequence prediction. The self-attention mechanisms of transformers, which explicitly model all pairwise interactions between elements in a sequence, make these architectures particularly suitable for specific constraints of set prediction such as removing duplicate predictions.

我们通过将物体检测视为一个直接的集合预测问题来简化训练流程。我们采用了基于变换器[47]的编码器-解码器架构，这是一种用于序列预测的流行架构。转换器的自我关注机制，明确地模拟了序列中元素之间的所有成对互动，使这些架构特别适合于集合预测的特定限制，如去除重复的预测。

Our DEtection TRansformer (DETR, see Figure 1) predicts all objects at once, and is trained end-to-end with a set loss function which performs bipartite matching between predicted and ground-truth objects. DETR simplifies the detection pipeline by dropping multiple hand-designed components that encode prior knowledge, like spatial anchors or non-maximal suppression. Unlike most existing detection methods, DETR doesn't require any customized layers, and thus can be reproduced easily in any framework that contains standard CNN and transformer classes.$^1$.

我们的DEtection TRansformer（DETR，见图1）可以一次性预测所有物体，并通过一个集合损失函数进行端到端的训练，该函数在预测物体和地面真实物体之间进行双侧匹配。DETR通过放弃多个手工设计的编码先验知识的组件，如空间锚或非最大抑制，简化了检测流程。与大多数现有的检测方法不同，DETR不需要任何定制层，因此可以在任何包含标准CNN和转化器类的框架中轻松复制。

>$^1$ In our work we use standard implementations of Transformers [47] and ResNet [15]backbones from standard deep learning libraries.

Compared to most previous work on direct set prediction, the main features of DETR are the conjunction of the bipartite matching loss and transformers with (non-autoregressive) parallel decoding [29,12,10,8]. In contrast, previous work focused on autoregressive decoding with RNNs [43,41,30,36,42]. Our matching loss function uniquely assigns a prediction to a ground truth object, and is invariant to a permutation of predicted objects, so we can emit them in parallel.

与以前大多数关于直接集预测的工作相比，DETR的主要特点是结合了双向匹配损失和Transformer与（非自回归）并行解码[29,12,10,8]。相比之下，以前的工作主要是用RNN进行自回归解码[43,41,30,36,42]。我们的匹配损失函数将预测结果唯一地分配给一个地面真实对象，并且对预测对象的排列组合是不变的，因此我们可以并行发射它们。

We evaluate DETR on one of the most popular object detection datasets, COCO [24], against a very competitive Faster R-CNN baseline [37]. Faster R- CNN has undergone many design iterations and its performance was greatly improved since the original publication. Our experiments show that our new model achieves comparable performances. More precisely, DETR demonstrates significantly better performance on large objects, a result likely enabled by the non-local computations of the transformer. It obtains, however, lower performances on small objects. We expect that future work will improve this aspect in the same way the development of FPN [22] did for Faster R-CNN.

我们在最流行的物体检测数据集之一COCO[24]上对DETR进行了评估，与非常有竞争力的Faster R-CNN基线[37]相比较。Faster R-CNN经历了多次设计迭代，其性能自最初发表以来得到了极大的改善。我们的实验表明，我们的新模型取得了相当的性能。更准确地说，DETR在大型物体上表现出明显更好的性能，这一结果可能是由Transformer的非局部计算促成的。然而，它在小物体上获得的性能较低。我们希望未来的工作能够像FPN[22]为Faster R-CNN所做的那样，改善这方面的性能。

Training settings for DETR differ from standard object detectors in multiple ways. The new model requires extra-long training schedule and benefits from auxiliary decoding losses in the transformer. We thoroughly explore what components are crucial for the demonstrated performance.

DETR的训练设置在多个方面与标准物体探测器不同。新模型需要超长的训练时间表，并从Transformer的辅助解码损失中获益。我们彻底探讨了哪些组件对所展示的性能至关重要。

【Deformable DETR 解决了上述的问题】

The design ethos of DETR easily extend to more complex tasks. In our experiments, we show that a simple segmentation head trained on top of a pre-trained DETR outperfoms competitive baselines on Panoptic Segmentation [19], a challenging pixel-level recognition task that has recently gained popularity.

DETR的设计理念很容易扩展到更复杂的任务。在我们的实验中，我们表明在预先训练好的DETR基础上训练的简单分割头在全景分割[19]上优于竞争基线，这是一项具有挑战性的像素级识别任务，最近已经得到了普及。

## 2 Related work 相关工作

Our work build on prior work in several domains: bipartite matching losses for set prediction, encoder-decoder architectures based on the transformer, parallel decoding, and object detection methods.

### 2.1 Set Prediction 集合预测

There is no canonical deep learning model to directly predict sets. The basic set prediction task is multilabel classification (see e.g., [40,33] for references in the context of computer vision) for which the baseline approach, one-vs-rest, does not apply to problems such as detection where there is an underlying structure between elements (i.e., near-identical boxes). The rst difficulty in these tasks is to avoid near-duplicates. Most current detectors use postprocessings such as non-maximal suppression to address this issue, but direct set prediction are postprocessing-free. They need global inference schemes that model interactions between all predicted elements to avoid redundancy. For constant-size set prediction, dense fully connected networks [9] are sufficient but costly. A general approach is to use auto-regressive sequence models such as recurrent neural networks [48]. In all cases, the loss function should be invariant by a permutation of the predictions. The usual solution is to design a loss based on the Hungarian algorithm [20], to nd a bipartite matching between ground-truth and prediction. This enforces permutation-invariance, and guarantees that each target element has a unique match. We follow the bipartite matching loss approach. In contrast to most prior work however, we step away from autoregressive models and use transformers with parallel decoding, which we describe below.

没有典型的深度学习模型可以直接预测集合。基本的集合预测任务是多标签分类（例如，见[40,33]在计算机视觉背景下的参考文献），其中基线方法 one-vs-rest 不适用于检测等问题元素之间的底层结构（即几乎相同的盒子）。这些任务的第一个困难是避免近似重复。大多数当前检测器使用非最大抑制等后处理来解决此问题，但直接集合预测是无后处理的。他们需要全局推理方案来模拟所有预测元素之间的交互以避免冗余。对于恒定大小的集合预测，密集的全连接网络 [9] 就足够了，但成本很高。一种通用的方法是使用自回归序列模型，例如递归神经网络 [48]。在所有情况下，损失函数都应该是预测排列不变的。通常的解决方案是基于匈牙利算法 [20] 设计一个损失，以找到 ground-truth 和预测之间的二分匹配。这强制执行排列不变性，并保证每个目标元素都具有唯一的匹配项。我们遵循二分匹配损失方法。然而，与大多数先前的工作相比，我们远离自回归模型并使用具有并行解码的转换器，我们将在下面进行描述。

### 2.2 Transformers and Parallel Decoding

Transformers were introduced by Vaswani et al . [47] as a new attention-based building block for machine translation. Attention mechanisms [2] are neural network layers that aggregate information from the entire input sequence. Transformers introduced self-attention layers, which, similarly to Non-Local Neural Networks [49], scan through each element of a sequence and update it by aggregating information from the whole sequence. One of the main advantages of attention-based models is their global computations and perfect memory, which makes them more suitable than RNNs on long sequences. Transformers are now replacing RNNs in many problems in natural language processing, speech processing and computer vision [8,27,45,34,31].

Transformers是由Vaswani等人[47]引入的，作为机器翻译的一个新的基于注意力的构建块。注意机制[2]是神经网络层，从整个输入序列中聚合信息。Transformers引入了自我注意层，与非局部神经网络[49]类似，它扫描序列的每个元素，并通过聚合整个序列的信息来更新它。基于注意力的模型的主要优势之一是它们的全局计算和完美的记忆，这使得它们在长序列上比RNNs更适合。目前，在自然语言处理、语音处理和计算机视觉的许多问题上，Transformers正在取代RNNs[8,27,45,34,31]。

Transformers were rst used in auto-regressive models, following early sequence-to-sequence models [44], generating output tokens one by one. However, the prohibitive inference cost (proportional to output length, and hard to batch) lead to the development of parallel sequence generation, in the domains of audio [29], machine translation [12,10], word representation learning [8], and more recently speech recognition [6]. We also combine transformers and parallel decoding for their suitable trade-off between computational cost and the ability to perform the global computations required for set prediction.

Tranformers最早用于自动回归模型，遵循早期的序列到序列模型[44]，逐一生成输出标记。然而，令人望而却步的推理成本（与输出长度成正比，且难以批量化）导致了并行序列生成的发展，在音频[29]、机器翻译[12,10]、单词表示学习[8]以及最近的语音识别[6]等领域。我们还将Transformer和并行解码结合起来，因为它们在计算成本和执行集合预测所需的全局计算的能力之间有合适的权衡。

### 2.3 Object detection 目标检测

Most modern object detection methods make predictions relative to some initial guesses. Two-stage detectors [37,5] predict boxes w.r.t. proposals, whereas single-stage methods make predictions w.r.t. anchors [23] or a grid of possible object centers [53,46]. Recent work [52] demonstrate that the final performance of these systems heavily depends on the exact way these initial guesses are set. In our model we are able to remove this hand-crafted process and streamline the detection process by directly predicting the set of detections with absolute box prediction w.r.t. the input image rather than an anchor.

大多数现代物体检测方法是相对于一些初始猜测进行预测的。两阶段检测器[37,5]以提议proposals为依据预测方框，而单阶段方法则以锚点[23]或可能的物体中心的网格[53,46]为依据进行预测。最近的工作[52]表明，这些系统的最终性能在很大程度上取决于这些初始猜测的确切设置方式。在我们的模型中，我们能够消除这种手工制作的过程，并通过对输入图像而非锚点的绝对箱体预测来直接预测检测集合，从而精简检测过程。

**Set-based loss.** Several object detectors [9,25,35] used the bipartite matching loss. However, in these early deep learning models, the relation between different prediction was modeled with convolutional or fully-connected layers only and a hand-designed NMS post-processing can improve their performance. More recent detectors [37,23,53] use non-unique assignment rules between ground truth and predictions together with an NMS.

**基于集合的目标函数。**几个物体检测器[9,25,35]使用了双点匹配损失。然而，在这些早期的深度学习模型中，不同预测之间的关系只用卷积层或全连接层来建模，手工设计的NMS后处理可以提高其性能。最近的检测器[37,23,53]与NMS一起使用基础事实和预测之间的非唯一分配规则。

Learnable NMS methods [16,4] and relation networks [17] explicitly model relations between different predictions with attention. Using direct set losses, they do not require any post-processing steps. However, these methods employ additional hand-crafted context features like proposal box coordinates to model relations between detections efficiently, while we look for solutions that reduce the prior knowledge encoded in the model.

可学习的NMS方法[16,4]和关系网络[17]明确地对不同预测之间的关系进行了关注建模。使用直接集合损失，它们不需要任何后处理步骤。然而，这些方法采用了额外的手工制作的上下文特征【场景特征】，如提案箱坐标，以有效地模拟检测之间的关系，而我们寻找的解决方案是减少模型中编码的先验知识。

**Recurrent detectors.** Closest to our approach are end-to-end set predictions for object detection [43] and instance segmentation [41,30,36,42]. Similarly to us, they use bipartite-matching losses with encoder-decoder architectures based on CNN activations to directly produce a set of bounding boxes. These approaches, however, were only evaluated on small datasets and not against modern baselines. In particular, they are based on autoregressive models (more precisely RNNs), so they do not leverage the recent transformers with parallel decoding.

**循环检测器。**与我们的方法最接近的是用于物体检测[43]和实例分割[41,30,36,42]的端到端集合预测。与我们类似，他们使用基于CNN激活的编码器-解码器架构的双字节匹配损失，直接产生一组边界框。然而，这些方法只在小型数据集上进行了评估，并没有针对现代基线。特别是，它们是基于自回归模型（更确切地说，是RNN），所以它们没有利用最近的具有并行解码的转换器。

## 3 The DETR model DETR 模型

Two ingredients are essential for direct set predictions in detection: (1) a set prediction loss that forces unique matching between predicted and ground truth boxes; (2) an architecture that predicts (in a single pass) a set of objects and models their relation. We describe our architecture in detail in Figure 2.

在检测中，有两个因素对于直接的集合预测是必不可少的：（1）一个集合预测目标函数，强制预测和地面真实箱之间的唯一匹配；（2）DETR架构，预测（在一个通道中）一组对象并对它们的关系进行建模。我们在图2中详细描述了我们的架构。

### 3.1 Object detection set prediction loss 基于集合的目标函数

DETR infers a fixed-size set of $N$ predictions, in a single pass through the decoder, where $N$ is set to be significantly larger than the typical number of objects in an image. One of the main difficulties of training is to score predicted objects (class, position, size) with respect to the ground truth. Our loss produces an optimal bipartite matching between predicted and ground truth objects, and then optimize object-specific (bounding box) losses.

DETR推断出一个固定大小的$N$预测集，只需通过一次解码器，其中$N$被设定为明显大于图像中的典型物体数量【输出的框比要预测的物体多很多】。训练的主要困难之一是对预测的对象（类别、位置、大小）进行评分。我们的损失在预测对象和地面真相对象之间产生一个最佳的二分图匹配，然后优化特定对象（边界盒）的损失。

Let us denote by $y$ the ground truth set of objects, and $\hat{y}=\{\hat{y}_i\}^N_{i=1}$ the set of $N$ predictions. Assuming $N$ is larger than the number of objects in the image, we consider $y$ also as a set of size $N$ padded with $∅$ (no object). To nd a bipartite matching between these two sets we search for a permutation of $N$ elements $\sigma \in \mathfrak{S}_N$ with the lowest cost:

让我们用 $y$ 表示对象的基本事实集，用 $\hat{y} = \{\hat{y}_i\}^N_{i=1}$ 表示 $N$ 预测集。 假设 $N$ 大于图像中的对象数量，我们将 $y$ 也视为用 $∅$（无对象）填充的大小为 $N$ 的集合。 为了找到这两个集合之间的二分匹配，我们搜索具有最低成本的 $N$ 个元素 $\sigma \in \mathfrak{S}_N$ 的排列：

$$
\hat{\sigma}= \argmin_{\sigma \in \mathfrak{S}_N} \sum^N_i \mathcal{L}(y_i,\hat{y}_{\sigma(i)}),\tag{1}
$$

where $\mathcal{L}(y_i,\hat{y}_{\sigma(i)})$ is a pair-wise matching cost between ground truth $y_i$ and a prediction with index $\sigma(i)$. This optimal assignment is computed efficiently with the Hungarian algorithm, following prior work (e.g . [43]).

其中 $\mathcal{L}(y_i,\hat{y}_{\sigma(i)})$ 是真实值 $y_i$ 和索引为 $\sigma(i)$ 的预测之间的成对匹配成本 . 根据先前的工作（例如 [43]），使用匈牙利算法可以有效地计算出这种最优分配。

The matching cost takes into account both the class prediction and the similarity of predicted and ground truth boxes. Each element i of the ground truth set can be seen as a $y_i = (c_i,b_i)$ where ci is the target class label (which may be $∅$) and $b_i \in [0,1]^4$ is a vector that defines ground truth box center coordinates and its height and width relative to the image size. For the prediction with index $\sigma(i)$ we define probability of class $c_i$ as $\hat{p}\_{\sigma(i)}(c\_i)$ and the predicted box as $\hat{b}\_{\sigma(i)}$. With these notations we define $\mathcal{L}(y\_i,\hat{y}\_{\sigma(i)})$ as $-{1}\_{\{c_i \neq ∅\}}\hat{p}\_{\sigma(i)}+ 1_{\{c_i \neq ∅\}}\mathcal{L}\_{\rm box} (b_i,\hat{b}\_{\hat{\sigma}(i)})$.

匹配成本考虑了类别预测以及预测框和地面真值框的相似性。 ground truth 集合的每个元素 i 都可以看作是 $y_i = (c_i,b_i)$ 其中 ci 是目标类标签（可能是 $∅$）和 $b_i \in [0,1]^4$ 是一个向量，它定义了地面实况框中心坐标及其相对于图像大小的高度和宽度。 对于索引为 $\sigma(i)$ 的预测，我们将类别 $c_i$ 的概率定义为 $\hat{p}_{\sigma(i)}(c_i)$ 并将预测框定义为 $\hat{b} _{\sigma (i)}$。 使用这些符号，我们将 $\mathcal{L}(y_i,\hat{y}\_{\sigma(i)})$ 定义为 $-{1}\_{\{c_i \neq ∅\}}\hat{p }\_{\sigma(i)}+ 1_{\{c_i \neq ∅\}}\mathcal{L}_{\rm box} (b_i,\hat{b}\_{\hat{\sigma}(i )})$。【前面一项是分类对不对，后面一项是框准不准】

This procedure of finding matching plays the same role as the heuristic assignment rules used to match proposal [37] or anchors [22] to ground truth objects in modern detectors. The main difference is that we need to find one-to-one matching for direct set prediction without duplicates.

这种寻找匹配的程序与现代检测器中用于将提议[37]或锚[22]与地面真实对象相匹配的启发式分配规则的作用相同。主要的区别是，我们需要找到一对一的匹配，以实现没有重复的直接集合预测。

The second step is to compute the loss function, the Hungarian loss for all pairs matched in the previous step. We define the loss similarly to the losses of common object detectors, i.e. a linear combination of a negative log-likelihood for class prediction and a box loss defined later:

第二步是计算损失函数，即上一步中匹配的所有配对的匈牙利损失。我们对损失的定义与普通物体检测器的损失类似，即类别预测的负对数可能性和后面定义的盒式损失的线性组合。

$$
\mathcal{L}_{Hungarian}(y,\hat{y})=\sum_{i=1}^N[-\log \hat{p}_{\hat{\sigma}(i)}(c_i) + \mathfrak{1}_{\{c_i \neq ∅ \}}\mathcal{L}_{\rm box} (b_i,\hat{b}_{\hat{\sigma}}(i))]；\tag{2}
$$

【作者改进了上述算法，为了两个损失函数大小差不多，前面去掉了log,后面用加上了generalize iou loss去掉框大小对目标函数的影响】

$$
\mathcal{L}_{Hungarian}(y,\hat{y})=\sum_{i=1}^N[- \hat{p}_{\hat{\sigma}(i)}(c_i) + \mathfrak{1}_{\{c_i \neq ∅ \}}\mathcal{L}_{\rm box} (b_i,\hat{b}_{\hat{\sigma}}(i))]；\tag{2}
$$

where $\hat{\sigma}$ is the optimal assignment computed in the first step (1). In practice, we down-weight the log-probability term when $c_i = ∅$ by a factor 10 to account for class imbalance. This is analogous to how Faster R-CNN training procedure balances positive/negative proposals by subsampling [37]. Notice that the matching cost between an object and ∅ doesn't depend on the prediction, which means that in that case the cost is a constant. In the matching cost we use probabilities $\hat{p}\_{\hat{\sigma}(i)}(c_i)$ instead of log-probabilities. This makes the class prediction term commensurable to  $\mathcal{L}\_{\rm box} (\cdot,\cdot )$ (described below), and we observed better empirical performances.

其中$\hat{\sigma}$是在第一步（1）中计算的最佳分配。在实践中，当$c_i=∅$时，我们将对数概率项的权重降低了10倍，以考虑到类的不平衡性。这类似于Faster R-CNN训练程序如何通过子抽样来平衡正/负的提议[37]。请注意，一个物体和∅之间的匹配成本并不取决于预测，这意味着在这种情况下，成本是一个常数。在匹配成本中，我们使用概率$\hat{p}\_{\hat{\sigma}(i)}(c_i)$而不是对数概率。这使得类预测项可以与$\mathcal{L}\_{\rm box} (\cdot,\cdot )$相称。（如下所述），我们观察到更好的经验表现。

**Bounding box loss.** The second part of the matching cost and the Hungarian loss is $\mathcal{L}\_{\rm box} (\cdot)$ that scores the bounding boxes. Unlike many detectors that do box predictions as a $\Delta$ w.r.t. some initial guesses, we make box predictions directly. While such approach simplify the implementation it poses an issue with relative scaling of the loss. The most commonly-used $\ell_1$ loss will have different scales for small and large boxes even if their relative errors are similar. To mitigate this issue we use a linear combination of the $\ell_1$ loss and the generalized IoU loss [38] $\mathcal{L}_{iou}(\cdot)$ that is scale-invariant. Overall, our box loss is $\mathcal{L}\_{\rm box}(b\_{\sigma(i)},\hat{b}_i)$ defined as $\lambda\_{\rm iou}\mathcal{L}_{\rm iou}(b\_{\sigma(i)},\hat{b}_i) + \lambda\_{\rm L1} \Vert b_{\sigma(i)} -\hat{b}_i \Vert_1$ where $\lambda\_{\rm iou},\lambda\_{\rm L1} \in \mathbb{R}$ are hyperparameters.These two losses are normalized by the number of objects inside the batch.

**边界框损失。** 匹配成本和匈牙利损失的第二部分是 $\mathcal{L}\_{\rm box} (\cdot)$，对边界盒进行评分。与许多检测器不同的是，我们直接进行盒子的预测，即对一些初始猜测的$\Delta$进行预测。虽然这种方法简化了实施，但却带来了损失的相对比例问题。最常用的$ell_1$损失对于小盒子和大盒子会有不同的比例，即使它们的相对误差是相似的。为了缓解这个问题，我们使用了$ell_1$损失和广义IoU损失的线性组合[38]$\mathcal{L}_{iou}(\cdot)$，它是尺度不变的。总的来说，我们的箱体损失是$\mathcal{L}\_{\rm box}(b\_{\sigma(i)},\hat{b}_i)$，定义为$\lambda\_{\rm iou}\mathcal{L}\_{\rm iou}(b\_{\sigma(i)},\hat{b}_i) + \lambda\_{\rm L1} \Vert b\_{\sigma(i)} -\hat{b}_i \Vert_1$ 其中$\lambda\_{\rm iou},\lambda\_{\rm L1} \in \mathbb{R}$. 这两个损失被批处理中的对象数量标准化。

### 3.2 DETR architecture

The overall DETR architecture is surprisingly simple and depicted in Figure 2. It contains three main components, which we describe below: a CNN backbone to extract a compact feature representation, an encoder-decoder transformer, and a simple feed forward network (FFN) that makes the final detection prediction.

Unlike many modern detectors, DETR can be implemented in any deep learning framework that provides a common CNN backbone and a transformer architecture implementation with just a few hundred lines. Inference code for DETR can be implemented in less than 50 lines in PyTorch [32]. We hope that the simplicity of our method will attract new researchers to the detection community.

**Backbone.** Starting from the initial image $x_{\rm img} \in \mathbb{R}^{3 \times H_0 \times W_0}$ (with $3$ color channels$^2$), a conventional CNN backbone generates a lower-resolution activation map $f \in \mathbb{R}^{C \times H \times W}$. Typical values we use are $C = 2048$ and $H,W =\frac{H_0}{32},\frac{W_0}{32}$.

>$^2$ The input images are batched together, applying 0-padding adequately to ensure they all have the same dimensions $(H_0, W_0)$ as the largest image of the batch.

**Transformer encoder.** First, a $1 \times 1$ convolution reduces the channel dimension of the high-level activation map $f$ from $C$ to a smaller dimension $d.$ creating a new feature map $z_0 \in \mathbb{R}^{d \times H \times W}$. The encoder expects a sequence as input, hence we collapse the spatial dimensions of $z_0$ into one dimension, resulting in a $d \times HW$ feature map. Each encoder layer has a standard architecture and consists of a multi-head self-attention module and a feed forward network (FFN). Since the transformer architecture is permutation-invariant, we supplement it with fixed positional encodings [31,3] that are added to the input of each attention layer. We defer to the supplementary material the detailed definition of the architecture, which follows the one described in [47].

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/End-to-End%20Object%20Detection%20with%20Transformers(DETR)/Fig%202.png"/></div>

Fig. 2: DETR uses a conventional CNN backbone to learn a 2D representation of an input image. The model flattens it and supplements it with a positional encoding before passing it into a transformer encoder. A transformer decoder then takes as input a small fixed number of learned positional embeddings, which we call *object queries*, and additionally attends to the encoder output. We pass each output embedding of the decoder to a shared feed forward network (FFN) that predicts either a detection (class and bounding box) or a `“no object”` class.

图2：DETR使用一个传统的CNN骨架来学习输入图像的二维表示。该模型对其进行扁平化处理，并在将其传递给转化器编码器之前用位置编码对其进行补充。然后，转化器解码器将学习到的少量固定的位置嵌入作为输入，我们称之为*物体查询*，并额外关注编码器的输出。我们将解码器的每个输出嵌入传递给一个共享的前馈网络（FFN），该网络预测一个检测（类别和边界框）或一个 "无对象 "类别。【
1. Decoder中object queries除第一层外都会做自注意操作，是为了消除冗余的框，他们相互通信之后可以知道大家要做什么框
2. Decoder之后加了很多auxiliary loss目标函数，让模型更快收敛】

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/End-to-End%20Object%20Detection%20with%20Transformers(DETR)/Fig%202-3.png"/></div>

**Transformer decoder.** The decoder follows the standard architecture of the transformer, transforming $N$ embeddings of size $d$ using multi-headed self- and encoder-decoder attention mechanisms. The difference with the original transformer is that our model decodes the $N$ objects in parallel at each decoder layer, while Vaswani et al. [47] use an autoregressive model that predicts the output sequence one element at a time. We refer the reader unfamiliar with the concepts to the supplementary material. Since the decoder is also permutation-invariant, the $N$ input embeddings must be different to produce different results. These input embeddings are learnt positional encodings that we refer to as object queries, and similarly to the encoder, we add them to the input of each attention layer. The $N$ object queries are transformed into an output embedding by the decoder. They are then independently decoded into box coordinates and class labels by a feed forward network (described in the next subsection), resulting $N$ final predictions. Using self- and encoder-decoder attention over these embeddings, the model globally reasons about all objects together using pair-wise relations between them, while being able to use the whole image as context.

**Prediction feed-forward networks (FFNs).** The final prediction is computed by a 3-layer perceptron with ReLU activation function and hidden dimension $d$, and a linear projection layer. The FFN predicts the normalized center coordinates, height and width of the box w.r.t. the input image, and the linear layer predicts the class label using a softmax function. Since we predict a fixed-size set of $N$ bounding boxes, where $N$ is usually much larger than the actual number of objects of interest in an image, an additional special class label $∅$ is used to represent that no object is detected within a slot. This class plays a similar role to the “background” class in the standard object detection approaches.

**Auxiliary decoding losses.** We found helpful to use auxiliary losses [1] in decoder during training, especially to help the model output the correct number of objects of each class. We add prediction FFNs and Hungarian loss after each decoder layer. All predictions FFNs share their parameters. We use an additional shared layer-norm to normalize the input to the prediction FFNs from different decoder layers.

## 4 Experiments 实验

We show that DETR achieves competitive results compared to Faster R-CNN in quantitative evaluation on COCO. Then, we provide a detailed ablation study of the architecture and loss, with insights and qualitative results. Finally, to show that DETR is a versatile and extensible model, we present results on panoptic segmentation, training only a small extension on a fixed DETR model. We provide code and pretrained models to reproduce our experiments at <https://github.com/facebookresearch/detr>.

**Dataset.** We perform experiments on COCO 2017 detection and panoptic segmentation datasets [24,18], containing 118k training images and 5k validation images. Each image is annotated with bounding boxes and panoptic segmentation. There are 7 instances per image on average, up to 63 instances in a single image in training set, ranging from small to large on the same images. If not specied, we report  $\text{AP}$ as bbox  $\text{AP}$, the integral metric over multiple thresholds. For comparison with Faster R-CNN we report validation  $\text{AP}$ at the last training epoch, for ablations we report median over validation results from the last 10 epochs.

**Technical details.** We train DETR with AdamW [26] setting the initial transformer's learning rate to $10^{-4}$, the backbone's to $10^{-5}$, and weight decay to $10^{-4}$. All transformer weights are initialized with Xavier init [11], and the backbone is with ImageNet-pretrained ResNet model [15] from torchvision with frozen batchnorm layers. We report results with two different backbones: a ResNet-50 and a ResNet-101. The corresponding models are called respectively DETR and DETR-R101. Following [21], we also increase the feature resolution by adding a dilation to the last stage of the backbone and removing a stride from the rst convolution of this stage. The corresponding models are called respectively DETR-DC5 and DETR-DC5-R101 (dilated C5 stage). This modification increases the resolution by a factor of two, thus improving performance for small objects, at the cost of a $16 \times$ higher cost in the self-attentions of the encoder, leading to an overall $2 \times$ increase in computational cost. A full comparison of FLOPs of these models and Faster R-CNN is given in Table 1.

We use scale augmentation, resizing the input images such that the shortest side is at least 480 and at most 800 pixels while the longest at most 1333 [50]. To help learning global relationships through the self-attention of the encoder, we also apply random crop augmentations during training, improving the performance by approximately 1  $\text{AP}$. Specically, a train image is cropped with probability 0.5 to a random rectangular patch which is then resized again to 800-1333. The transformer is trained with default dropout of 0.1. At inference time, some slots predict empty class. To optimize for  $\text{AP}$, we override the prediction of these slots with the second highest scoring class, using the corresponding confidence. This improves  $\text{AP}$ by 2 points compared to filtering out empty slots. Other training hyperparameters can be found in section A.4. For our ablation experiments we use training schedule of 300 epochs with a learning rate drop by a factor of 10 after 200 epochs, where a single epoch is a pass over all training images once. Training the baseline model for 300 epochs on 16 V100 GPUs takes 3 days, with 4 images per GPU (hence a total batch size of 64). For the longer schedule used to compare with Faster R-CNN we train for 500 epochs with learning rate drop after 400 epochs. This schedule adds 1.5  $\text{AP}$ compared to the shorter schedule.

Table 1: Comparison with Faster R-CNN with a ResNet-50 and ResNet-101 backbones on the COCO validation set. The top section shows results for Faster R-CNN models in Detectron2 [50], the middle section shows results for Faster R-CNN models with GIoU [38], random crops train-time augmentation, and the long 9x training schedule. DETR models achieve comparable results to heavily tuned Faster R-CNN baselines, having lower ${\rm AP}\_{\rm L}$ but greatly improved ${\rm AP}\_{\rm L}$ . We use torchscript Faster R-CNN and DETR models to measure FLOPS and FPS. Results without R101 in the name correspond to ResNet-50. \
表1：与Faster R-CNN的ResNet-50和ResNet-101骨干模型在COCO验证集上的比较。上部分显示了Detectron2[50]中Faster R-CNN模型的结果，中间部分显示了Faster R-CNN模型与GIoU[38]、随机作物训练时间增强和长9倍训练计划的结果。DETR模型取得了与重度调整的Faster R-CNN基线相当的结果，具有较低的${\rm AP}\_{\rm L}$，但大大改善了${\rm AP}\_{\rm L}$。我们使用torchscript Faster R-CNN和DETR模型来测量FLOPS和FPS。名称中没有R101的结果对应于ResNet-50。\\(β_m\\)

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/End-to-End%20Object%20Detection%20with%20Transformers(DETR)/Table%201.png"/></div>

### 4.1 Comparison with Faster R-CNN

Table 2: Effect of encoder size. Each row corresponds to a model with varied number of encoder layers and fixed number of decoder layers. Performance gradually improves with more encoder layers. \
表2：编码器大小的影响。每一行都对应于一个具有不同数量编码器层和固定数量解码器层的模型。性能随着编码器层数的增加而逐渐提高。

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/End-to-End%20Object%20Detection%20with%20Transformers(DETR)/Table%202.png"/></div>

Transformers are typically trained with Adam or Adagrad optimizers with very long training schedules and dropout, and this is true for DETR as well. Faster R-CNN, however, is trained with SGD with minimal data augmentation and we are not aware of successful applications of Adam or dropout. Despite these differences we attempt to make a Faster R-CNN baseline stronger. To align it with DETR, we add generalized IoU [38] to the box loss, the same random crop augmentation and long training known to improve results [13]. Results are presented in Table 1. In the top section we show Faster R-CNN results from Detectron2 Model Zoo [50] for models trained with the 3x schedule. In the middle section we show results (with a \+") for the same models but trained with the $9 \times$ schedule (109 epochs) and the described enhancements, which in total adds 1-2 $\text{AP}$. In the last section of Table 1 we show the results for multiple DETR models. To be comparable in the number of parameters we choose a model with 6 transformer and 6 decoder layers of width 256 with 8 attention heads. Like Faster R-CNN with FPN this model has 41.3M parameters, out of which 23.5M are in ResNet-50, and 17.8M are in the transformer. Even though both Faster R-CNN and DETR are still likely to further improve with longer training, we can conclude that DETR can be competitive with Faster R-CNN with the same number of parameters, achieving $42 \text{AP}$ on the COCO val subset. The way DETR achieves this is by improving  $\text{AP}\_{\text{L}} (+7.8)$, however note that the model is still lagging behind in $\text{AP}\_{\text{S}} (-5.5)$. DETR-DC5 with the same number of parameters and similar FLOP count has higher  $\text{AP}$, but is still significantly behind in  $\text{AP}\_\text{S}$ too. Faster R-CNN and DETR with ResNet-101 backbone show comparable results as well.

### 4.2 Ablations

Attention mechanisms in the transformer decoder are the key components which model relations between feature representations of different detections. In our ablation analysis, we explore how other components of our architecture and loss influence the final performance. For the study we choose ResNet-50-based DETR model with 6 encoder, 6 decoder layers and width 256. The model has 41.3M parameters, achieves 40.6 and 42.0  $\text{AP}$ on short and long schedules respectively, and runs at 28 FPS, similarly to Faster R-CNN-FPN with the same backbone.

**Number of encoder layers.** We evaluate the importance of global image-level self-attention by changing the number of encoder layers (Table 2). Without encoder layers, overall  $\text{AP}$ drops by 3.9 points, with a more significant drop of 6.0  $\text{AP}$ on large objects. We hypothesize that, by using global scene reasoning, the encoder is important for disentangling objects. In Figure 3, we visualize the attention maps of the last encoder layer of a trained model, focusing on a few points in the image. The encoder seems to separate instances already, which likely simplifies object extraction and localization for the decoder.

**Number of decoder layers.** We apply auxiliary losses after each decoding layer (see Section 3.2), hence, the prediction FFNs are trained by design to predict objects out of the outputs of every decoder layer. We analyze the importance of each decoder layer by evaluating the objects that would be predicted at each stage of the decoding (Fig. 4). Both  $\text{AP}$ and  $\text{AP}_{50}$ improve after every layer, totalling into a very significant +8.2/9.5  $\text{AP}$ improvement between the rst and the last layer. With its set-based loss, DETR does not need NMS by design. To verify this we run a standard NMS procedure with default parameters [50] for the outputs after each decoder. NMS improves performance for the predictions from the rst decoder. This can be explained by the fact that a single decoding layer of the transformer is not able to compute any cross-correlations between the output elements, and thus it is prone to making multiple predictions for the same object. In the second and subsequent layers, the self-attention mechanism over the activations allows the model to inhibit duplicate predictions. We observe that the improvement brought by NMS diminishes as depth increases. At the last layers, we observe a small loss in  $\text{AP}$ as NMS incorrectly removes true positive predictions.

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/End-to-End%20Object%20Detection%20with%20Transformers(DETR)/Fig%203.png"/></div>

Fig. 3: Encoder self-attention for a set of reference points. The encoder is able to separate individual instances. Predictions are made with baseline DETR model on a validation set image. \
图3：编码器对一组参考点的自我关注。编码器能够分离单个实例。用基线DETR模型对验证集图像进行预测。

Similarly to visualizing encoder attention, we visualize decoder attentions in Fig. 6, coloring attention maps for each predicted object in different colors. We observe that decoder attention is fairly local, meaning that it mostly attends to object extremities such as heads or legs. We hypothesise that after the encoder has separated instances via global attention, the decoder only needs to attend to the extremities to extract the class and object boundaries.

**Importance of FFN.** FFN inside tranformers can be seen as $1  \times 1$ convolutional layers, making encoder similar to attention augmented convolutional networks [3]. We attempt to remove it completely leaving only attention in the transformer layers. By reducing the number of network parameters from 41.3M to 28.7M, leaving only 10.8M in the transformer, performance drops by 2.3  $\text{AP}$, we thus conclude that FFN are important for achieving good results.

**Importance of positional encodings.** There are two kinds of positional encodings in our model: spatial positional encodings and output positional encodings (object queries). We experiment with various combinations of xed and learned encodings, results can be found in table 3. Output positional encodings are required and cannot be removed, so we experiment with either passing them once at decoder input or adding to queries at every decoder attention layer. In the rst experiment we completely remove spatial positional encodings and pass output positional encodings at input and, interestingly, the model still achieves more than 32  $\text{AP}$, losing 7.8  $\text{AP}$ to the baseline. Then, we pass fixed sine spatial positional encodings and the output encodings at input once, as in the original transformer [47], and nd that this leads to 1.4  $\text{AP}$ drop compared to passing the positional encodings directly in attention. Learned spatial encodings passed to the attentions give similar results. Surprisingly, we nd that not passing any spatial encodings in the encoder only leads to a minor  $\text{AP}$ drop of 1.3  $\text{AP}$. When we pass the encodings to the attentions, they are shared across all layers, and the output encodings (object queries) are always learned.

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/End-to-End%20Object%20Detection%20with%20Transformers(DETR)/Fig%204.png"/></div>

Fig. 4: $\text{AP}$ and $\text{AP}\_{50}$ performance after each decoder layer. A single long schedule baseline model is evaluated. DETR does not need NMS by design, which is validated by this figure. NMS lowers $\text{AP}$ in the final layers, removing TP predictions, but improves $\text{AP}$ in the first decoder layers, removing double predictions, as there is no communication in the first layer, and slightly improves $\text{AP}\_{50}$.

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/End-to-End%20Object%20Detection%20with%20Transformers(DETR)/Fig%205.png"/></div>

Fig. 5: Out of distribution generalization for rare classes.Even though no image in the training set has more than 13 giraffes, DETR has no difficulty generalizing to 24 and more instances of the same class.

Given these ablations, we conclude that transformer components: the global self-attention in encoder, FFN, multiple decoder layers, and positional encodings, all significantly contribute to the final object detection performance.

**Loss ablations.** To evaluate the importance of dierent components of the matching cost and the loss, we train several models turning them on and o. There are three components to the loss: classification loss, $\ell_1$ bounding box distance loss, and GIoU [38] loss. The classification loss is essential for training and cannot be turned o, so we train a model without bounding box distance loss, and a model without the GIoU loss, and compare with baseline, trained with all three losses. Results are presented in table 4. GIoU loss on its own accounts for most of the model performance, losing only 0.7 AP to the baseline with combined losses. Using $\ell_1$ without GIoU shows poor results. We only studied simple ablations of dierent losses (using the same weighting every time), but other means of combining them may achieve dierent results.

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/End-to-End%20Object%20Detection%20with%20Transformers(DETR)/Fig%206.png"/></div>

Fig. 6: Visualizing decoder attention for every predicted object (images from COCO val set). Predictions are made with DETR-DC5 model. Attention scores are coded with different colors for different objects. Decoder typically attends to object extremities, such as legs and heads. Best viewed in color. \
图6：可视化解码器对每个预测对象的关注（图像来自COCO值集）。预测是用DETR-DC5模型进行的。不同物体的注意分数用不同的颜色来表示。解码器通常关注物体的四肢，如腿和头。最好以彩色观看。【encoder学全局特征，decoder学边缘特征】

Table 3: Results for dierent positional encodings compared to the baseline (last row), which has fixed sine pos. encodings passed at every attention layer in both the encoder and the decoder. Learned embeddings are shared between all layers. Not using spatial positional encodings leads to a significant drop in AP. Interestingly, passing them in decoder only leads to a minor AP drop. All these models use learned output positional encodings.

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/End-to-End%20Object%20Detection%20with%20Transformers(DETR)/Table%203.png"/></div>

Table 4: Effect of loss components on AP. We train two models turning off $\ell_1$ loss, and GIoU loss, and observe that $\ell_1$ gives poor results on its own, but when combined with GIoU improves $\text{AP}_\text{M}$ and ${\text{AP}}_{\text{L}}$. Our baseline (last row) combines both losses.

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/End-to-End%20Object%20Detection%20with%20Transformers(DETR)/Table%204.png"/></div>

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/End-to-End%20Object%20Detection%20with%20Transformers(DETR)/Fig%207.png"/></div>

Fig. 7: Visualization of all box predictions on all images from COCO 2017 val set for 20 out of total $N = 100$ prediction slots in DETR decoder. Each box prediction is represented as a point with the coordinates of its center in the 1-by-1 square normalized by each image size. The points are color-coded so that green color corresponds to small boxes, red to large horizontal boxes and blue to large vertical boxes. We observe that each slot learns to specialize on certain areas and box sizes with several operating modes. We note that almost all slots have a mode of predicting large image-wide boxes that are common in COCO dataset. \
图 7：DETR 解码器中总共 $N = 100$ 个预测框中的 20 个来自 COCO 2017 val 集的所有图像的所有框预测的可视化。 每个框预测都表示为一个点，其中心坐标在由每个图像大小归一化的 1×1 正方形中。 这些点是用颜色编码的，绿色对应小方框，红色对应大水平方框，蓝色对应大竖直方框。 我们观察到每个插槽都学会了专注于具有多种操作模式的特定区域和盒子尺寸。 我们注意到几乎所有框都有一种预测大型图像宽框的模式，这在 COCO 数据集中很常见。【展示object query学习了什么 】

### 4.3 Analysis

**Decoder output slot analysis** In Fig. 7 we visualize the boxes predicted by different slots for all images in COCO 2017 val set. DETR learns different specialization for each query slot. We observe that each slot has several modes of operation focusing on different areas and box sizes. In particular, all slots have the mode for predicting image-wide boxes (visible as the red dots aligned in the middle of the plot). We hypothesize that this is related to the distribution of objects in COCO.

**Generalization to unseen numbers of instances.** Some classes in COCO are not well represented with many instances of the same class in the same image. For example, there is no image with more than 13 giraffes in the training set. We create a synthetic image $^3$ to verify the generalization ability of DETR (see Figure 5). Our model is able to nd all 24 giraffes on the image which is clearly out of distribution. This experiment confirms that there is no strong class-specialization in each object query.

>$^3$ Base picture credit: <https://www.piqsels.com/en/public-domain-photo-jzlwu>

### 4.4 DETR for panoptic segmentation

Panoptic segmentation [19] has recently attracted a lot of attention from the computer vision community. Similarly to the extension of Faster R-CNN [37] to Mask R-CNN [14], DETR can be naturally extended by adding a mask head on top of the decoder outputs. In this section we demonstrate that such a head can be used to produce panoptic segmentation [19] by treating stu and thing classes in a unied way. We perform our experiments on the panoptic annotations of the COCO dataset that has 53 stu categories in addition to 80 things categories.

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/End-to-End%20Object%20Detection%20with%20Transformers(DETR)/Fig%208.png"/></div>

Fig. 8: Illustration of the panoptic head. A binary mask is generated in parallel for each detected object, then the masks are merged using pixel-wise argmax.

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/End-to-End%20Object%20Detection%20with%20Transformers(DETR)/Fig%209.png"/></div>

Fig. 9: Qualitative results for panoptic segmentation generated by DETR-R101. DETR produces aligned mask predictions in a unified manner for things and stuff.

We train DETR to predict boxes around both stu and things classes on COCO, using the same recipe. Predicting boxes is required for the training to be possible, since the Hungarian matching is computed using distances between boxes. We also add a mask head which predicts a binary mask for each of the predicted boxes, see Figure 8. It takes as input the output of transformer decoder for each object and computes multi-head (with M heads) attention scores of this embedding over the output of the encoder, generating M attention heatmaps per object in a small resolution. To make the final prediction and increase the resolution, an FPN-like architecture is used. We describe the architecture in more details in the supplement. The final resolution of the masks has stride 4 and each mask is supervised independently using the DICE/F-1 loss [28] and Focal loss [23].

The mask head can be trained either jointly, or in a two steps process, where we train DETR for boxes only, then freeze all the weights and train only the mask head for 25 epochs. Experimentally, these two approaches give similar results, we report results using the latter method since it results in a shorter total wall-clock time training.

Table 5: Comparison with the state-of-the-art methods UPSNet [51] and Panoptic FPN [18] on the COCO val dataset We retrained PanopticFPN with the same data- augmentation as DETR, on a 18x schedule for fair comparison. UPSNet uses the 1x schedule, UPSNet-M is the version with multiscale test-time augmentations.

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/End-to-End%20Object%20Detection%20with%20Transformers(DETR)/Table%205.png"/></div>

To predict the final panoptic segmentation we simply use an argmax over the mask scores at each pixel, and assign the corresponding categories to the resulting masks. This procedure guarantees that the final masks have no overlaps and, therefore, DETR does not require a heuristic [19] that is often used to align dierent masks.

**Training details.** We train DETR, DETR-DC5 and DETR-R101 models following the recipe for bounding box detection to predict boxes around stu and things classes in COCO dataset. The new mask head is trained for 25 epochs (see supplementary for details). During inference we rst lter out the detection with a condence below 85%, then compute the per-pixel argmax to determine in which mask each pixel belongs. We then collapse dierent mask predictions of the same stu category in one, and lter the empty ones (less than 4 pixels).

**Main results.** Qualitative results are shown in Figure 9. In table 5 we compare our unied panoptic segmenation approach with several established methods that treat things and stu dierently. We report the Panoptic Quality (PQ) and the break-down on things (PQth) and stu (PQst). We also report the mask AP (computed on the things classes), before any panoptic post-treatment (in our case, before taking the pixel-wise argmax). We show that DETR outperforms published results on COCO-val 2017, as well as our strong PanopticFPN baseline (trained with same data-augmentation as DETR, for fair comparison). The result break-down shows that DETR is especially dominant on stu classes, and we hypothesize that the global reasoning allowed by the encoder attention is the key element to this result. For things class, despite a severe deficit of up to 8 mAP compared to the baselines on the mask AP computation, DETR obtains competitive PQth . We also evaluated our method on the test set of the COCO dataset, and obtained 46 PQ. We hope that our approach will inspire the exploration of fully unied models for panoptic segmentation in future work.

## 5 Conclusion 结论

We presented DETR, a new design for ob ject detection systems based on transformers and bipartite matching loss for direct set prediction. The approach achieves comparable results to an optimized Faster R-CNN baseline on the challenging COCO dataset. DETR is straightforward to implement and has a exible architecture that is easily extensible to panoptic segmentation, with competitive results. In addition, it achieves signicantly better performance on large ob jects than Faster R-CNN, likely thanks to the processing of global information performed by the self-attention.

我们介绍了 DETR，一种基于 transformers 和二分匹配损失的目标检测系统的新设计，用于直接集合预测。 该方法在具有挑战性的 COCO 数据集上取得了与优化的 Faster R-CNN 基线相当的果。 DETR 易于实施，并且具有灵活的体系结构，可以轻松扩展到全景分割，并具有有竞争力的结果。 此外，它在大型对象上的性能明显优于 Faster R-CNN，这可能要归功于自我注意执行的全局信息处理。

This new design for detectors also comes with new challenges, in particular regarding training, optimization and performances on small objects. Current detectors required several years of improvements to cope with similar issues, and we expect future work to successfully address them for DETR.

这种新的检测器设计也带来了新的挑战，特别是在小物体的训练、优化和性能方面。 目前的检测器需要几年的改进才能应对类似的问题，我们希望未来的工作能够成功解决 DETR 的这些问题。

【后续工作：

- Omni DETR (arXiv:2203.16089)
- Up DETR(arXiv:2011.09094)
- pnp DETR(arXiV:07036)
- SMAC DETR
- Deformable DETR(arXiv:2010.04159)
- DAB DETR(arXiv:2201.12329)
- SAM DETR
- DN DETR(arXiv:2203.01305)
- OW DETR(arXiv:2112.01513)
- OV DETR
】

## 6 Acknowledgements

We thank Sainbayar Sukhbaatar, Piotr Bojanowski, Natalia Neverova, David Lopez-Paz, Guillaume Lample, Danielle Rothermel, Kaiming He, Ross Girshick, Xinlei Chen and the whole Facebook AI Research Paris team for discussions and advices without which this work would not be possible.

## References

1. Al-Rfou, R., Choe, D., Constant, N., Guo, M., Jones, L.: Character-level language modeling with deeper self-attention. In: AAAI Conference on Articial Intelligence (2019)
2. Bahdanau, D., Cho, K., Bengio, Y.: Neural machine translation by jointly learning to align and translate. In: ICLR (2015)
3. Bello, I., Zoph, B., Vaswani, A., Shlens, J., Le, Q.V.: Attention augmented convolutional networks. In: ICCV (2019)
4. Bodla, N., Singh, B., Chellappa, R., Davis, L.S.: Soft-NMS improving object detection with one line of code. In: ICCV (2017)
5. Cai, Z., Vasconcelos, N.: Cascade R-CNN: High quality object detection and instance segmentation. PAMI (2019)
6. Chan, W., Saharia, C., Hinton, G., Norouzi, M., Jaitly, N.: Imputer: Sequence modelling via imputation and dynamic programming. arXiv:2002.08926 (2020)
7. Cordonnier, J.B., Loukas, A., Jaggi, M.: On the relationship between self-attention and convolutional layers. In: ICLR (2020)
8. Devlin, J., Chang, M.W., Lee, K., Toutanova, K.: BERT: Pre-training of deep bidirectional transformers for language understanding. In: NAACL-HLT (2019)
9. Erhan, D., Szegedy, C., Toshev, A., Anguelov, D.: Scalable object detection using deep neural networks. In: CVPR (2014)
10. Ghazvininejad, M., Levy, O., Liu, Y., Zettlemoyer, L.: Mask-predict: Parallel decoding of conditional masked language models. arXiv:1904.09324 (2019)
11. Glorot, X., Bengio, Y.: Understanding the diculty of training deep feedforward neural networks. In: AISTATS (2010)
12. Gu, J., Bradbury, J., Xiong, C., Li, V.O., Socher, R.: Non-autoregressive neural machine translation. In: ICLR (2018)
13. He, K., Girshick, R., Dollar, P.: Rethinking imagenet pre-training. In: ICCV (2019)
14. He, K., Gkioxari, G., Dollar, P., Girshick, R.B.: Mask R-CNN. In: ICCV (2017)
15. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: CVPR (2016)
16. Hosang, J.H., Benenson, R., Schiele, B.: Learning non-maximum suppression. In: CVPR (2017)
17. Hu, H., Gu, J., Zhang, Z., Dai, J., Wei, Y.: Relation networks for object detection. In: CVPR (2018)
18. Kirillov, A., Girshick, R., He, K., Dollar, P.: Panoptic feature pyramid networks. In: CVPR (2019)
19. Kirillov, A., He, K., Girshick, R., Rother, C., Dollar, P.: Panoptic segmentation. In: CVPR (2019)
20. Kuhn, H.W.: The hungarian method for the assignment problem (1955)
21. Li, Y., Qi, H., Dai, J., Ji, X., Wei, Y.: Fully convolutional instance-aware semantic segmentation. In: CVPR (2017)
22. Lin, T.Y., Dollar, P., Girshick, R., He, K., Hariharan, B., Belongie, S.: Feature pyramid networks for object detection. In: CVPR (2017)
23. Lin, T.Y., Goyal, P., Girshick, R.B., He, K., Dollar, P.: Focal loss for dense object detection. In: ICCV (2017)
24. Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollar, P., Zitnick, C.L.: Microsoft COCO: Common objects in context. In: ECCV (2014)
25. Liu, W., Anguelov, D., Erhan, D., Szegedy, C., Reed, S.E., Fu, C.Y., Berg, A.C.: Ssd: Single shot multibox detector. In: ECCV (2016)
26. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. In: ICLR (2017)
27. Luscher, C., Beck, E., Irie, K., Kitza, M., Michel, W., Zeyer, A., Schluter, R., Ney, H.: Rwth asr systems for librispeech: Hybrid vs attention - w/o data augmentation. arXiv:1905.03072 (2019)
28. Milletari, F., Navab, N., Ahmadi, S.A.: V-net: Fully convolutional neural networks for volumetric medical image segmentation. In: 3DV (2016)
29. Oord, A.v.d., Li, Y., Babuschkin, I., Simonyan, K., Vinyals, O., Kavukcuoglu, K., Driessche, G.v.d., Lockhart, E., Cobo, L.C., Stimberg, F., et al.: Parallel wavenet: Fast high-delity speech synthesis. arXiv:1711.10433 (2017)
30. Park, E., Berg, A.C.: Learning to decompose for object detection and instance segmentation. arXiv:1511.06449 (2015)
31. Parmar, N., Vaswani, A., Uszkoreit, J., Kaiser, L., Shazeer, N., Ku, A., Tran, D.: Image transformer. In: ICML (2018)
32. Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., Chintala, S.: Pytorch: An imperative style, high-performance deep learning library. In: NeurIPS (2019)
33. Pineda, L., Salvador, A., Drozdzal, M., Romero, A.: Elucidating image-to-set prediction: An analysis of models, losses and datasets. arXiv:1904.05709 (2019)
34. Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I.: Language models are unsupervised multitask learners (2019)
35. Redmon, J., Divvala, S., Girshick, R., Farhadi, A.: You only look once: Unied, real-time object detection. In: CVPR (2016)
36. Ren, M., Zemel, R.S.: End-to-end instance segmentation with recurrent attention. In: CVPR (2017)
37. Ren, S., He, K., Girshick, R.B., Sun, J.: Faster R-CNN: Towards real-time object detection with region proposal networks. PAMI (2015)
38. Rezatoghi, H., Tsoi, N., Gwak, J., Sadeghian, A., Reid, I., Savarese, S.: Generalized intersection over union. In: CVPR (2019)
39. Rezatoghi, S.H., Kaskman, R., Motlagh, F.T., Shi, Q., Cremers, D., Leal-Taixe, L., Reid, I.: Deep perm-set net: Learn to predict sets with unknown permutation and cardinality using deep neural networks. arXiv:1805.00613 (2018)
40. Rezatoghi, S.H., Milan, A., Abbasnejad, E., Dick, A., Reid, I., Kaskman, R., Cremers, D., Leal-Taix, l.: Deepsetnet: Predicting sets with deep neural networks. In: ICCV (2017)
41. Romera-Paredes, B., Torr, P.H.S.: Recurrent instance segmentation. In: ECCV (2015)
42. Salvador, A., Bellver, M., Baradad, M., Marques, F., Torres, J., Giro, X.: Recurrent neural networks for semantic instance segmentation. arXiv:1712.00617 (2017)
43. Stewart, R.J., Andriluka, M., Ng, A.Y.: End-to-end people detection in crowded scenes. In: CVPR (2015)
44. Sutskever, I., Vinyals, O., Le, Q.V.: Sequence to sequence learning with neural networks. In: NeurIPS (2014)
45. Synnaeve, G., Xu, Q., Kahn, J., Grave, E., Likhomanenko, T., Pratap, V., Sri- ram, A., Liptchinsky, V., Collobert, R.: End-to-end ASR: from supervised to semisupervised learning with modern architectures. arXiv:1911.08460 (2019)
46. Tian, Z., Shen, C., Chen, H., He, T.: FCOS: Fully convolutional one-stage object detection. In: ICCV (2019)
47. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., Polosukhin, I.: Attention is all you need. In: NeurIPS (2017)
48. Vinyals, O., Bengio, S., Kudlur, M.: Order matters: Sequence to sequence for sets. In: ICLR (2016)
49. Wang, X., Girshick, R.B., Gupta, A., He, K.: Non-local neural networks. In: CVPR (2018)
50. Wu, Y., Kirillov, A., Massa, F., Lo, W.Y., Girshick, R.: Detectron2. <https://github.com/facebookresearch/detectron2> (2019)
51. Xiong, Y., Liao, R., Zhao, H., Hu, R., Bai, M., Yumer, E., Urtasun, R.: Upsnet: A unied panoptic segmentation network. In: CVPR (2019)
52. Zhang, S., Chi, C., Yao, Y., Lei, Z., Li, S.Z.: Bridging the gap between anchor-based and anchor-free detection via adaptive training sample selection. arXiv:1912.02424 (2019)
53. Zhou, X., Wang, D., Kr•henbUhl, P.: Objects as points. arXiv:1904.07850 (2019)

## A Appendix

### A.1 Preliminaries: Multi-head attention layers

Since our model is based on the Transformer architecture, we remind here the general form of attention mechanisms we use for exhaustivity. The attention mechanism follows [47], except for the details of positional encodings (see Equation 8) that follows [7].

**Multi-head** The general form of multi-head attention with M heads of dimension d is a function with the following signature (using $d' = \frac{d}{M}$, and giving matrix/tensors sizes in underbrace)

$$
\text{mh-attn} :  \underbrace{X_{\rm q}}_{d \times N_{\rm q}}, \underbrace{X_{\rm kv}}_{d \times N_{\rm kv}}, \underbrace{T}_{M \times 3 \times d' \times d},\underbrace{L}_{d \times d} \mapsto \underbrace{\tilde{X}_{\rm q}}_{d \times N_{\rm q}} \tag{3}
$$

where $X_{\rm q}$ is the *query sequence* of length $N_q$, $X_{\rm kv}$ is the *key-value sequence of* length $N_{\rm kv}$ (with the same number of channels $d$ for simplicity of exposition), $T$ is the weight tensor to compute the so-called query, key and value embeddings, and $L$ is a projection matrix. The output is the same size as the query sequence. To x the vocabulary before giving details, multi-head self-attention (mh-s-attn) is the special case $X_{\rm q} = X_{\rm kv}$, i.e.

$$
\text{mh-s-attn}(X; T; L) = \text{mh-attn}(X,X,T,L). \tag{4}
$$

The multi-head attention is simply the concatenation of $M$ single attention heads followed by a pro jection with $L$. The common practice [47] is to use residual connections, dropout and layer normalization. In other words, denoting $\tilde{X}_{\rm q} = \text{mh-attn}(X_{\rm q},X_{\rm kv}, T,L)$ and $\bar{\bar{X}}^{(q)}$ the concatenation of attention heads, we have

$$
X_q' = [\text{attn}(X_{\rm q},X_{\rm kv}, T_1);...; \text{attn}(X_{\rm q},X_{\rm kv}, T_M )] \tag{5}
$$
$$
\tilde{X}_{\rm q} = \text{layernorm}( X_{\rm q} + \text{dropout}(LX_q' )) \tag{6}
$$
where $[;]$ denotes concatenation on the channel axis.

**Single head** An attention head with weight tensor T 0 2 R3d0d , denoted by attn(Xq; Xkv; T0), depends on additional positional encoding Pq 2 RdNq and Pkv 2 RdNkv . It starts by computing so-called query, key and value embeddings after adding the query and key positional encodings [7]:

$$
[Q;K;V] = [T'_1(X_{\rm q} + P_{\rm q});T'_2(X_{\rm kv} + P_{\rm kv});T'_3X_{\rm kv}] \tag{7}
$$

where $T'$ is the concatenation of $T'_1,T'_2; T'_3$. The *attention weights* are then computed based on the softmax of dot products between queries and keys, so that each element of the query sequence attends to all elements of the key-value sequence ($i$ is a query index and $j$ a key-value index):

$$
\alpha_{i,j}=\frac{e^{\frac{1}{\sqrt{d'}}Q^T_iK_j}}{Z_i} \ \text{where} \ Z_i=\sum_{j=1}^{N_{\rm kv}}{e^{\frac{1}{\sqrt{d'}}Q^T_iK_j}} \tag{8}
$$

In our case, the positional encodings may be learnt or fixed, but are shared across all attention layers for a given query/key-value sequence, so we do not explicitly write them as parameters of the attention. We give more details on their exact value when describing the encoder and the decoder. The final output is the aggregation of values weighted by attention weights: The $i$-th row is given by ${\text{attn}_i(X_{\rm q},X_{\rm kv},T')=\sum_{j=1}^{N_{\rm kv}}a_{i,j}V_j}$ .

**Feed-forward network (FFN) layers** The original transformer alternates multi-head attention and so-called FFN layers [47], which are eectively multilayer $1\times1$ convolutions, which have Md input and output channels in our case. The FFN we consider is composed of two-layers of $1\times1$ convolutions with ReLU activations. There is also a residual connection/dropout/layernorm after the two layers, similarly to equation 6.

### A.2 Losses

For completeness, we present in detail the losses used in our approach. All losses are normalized by the number of objects inside the batch. Extra care must be taken for distributed training: since each GPU receives a sub-batch, it is not sucient to normalize by the number of objects in the local batch, since in general the sub-batches are not balanced across GPUs. Instead, it is important to normalize by the total number of ob jects in all sub-batches.

**Box loss** Similarly to [41,36], we use a soft version of Intersection over Union in our loss, together with a $\ell_1$ loss on $\hat{b}$:

$$
\mathcal{L}_{\rm box}(b_{\sigma(i)},\hat{b}_i)=\lambda_{\rm iou}\mathcal{L}_{\rm iou}(b_{\sigma(i)},\hat{b}_i) + \lambda_{\rm L1} \Vert b_{\sigma(i)} -\hat{b}_i \Vert_1\tag{9}
$$

where $\lambda_{iou} ,\lambda_{\rm L1} \in \mathbb{R}$ are hyperparameters and $\mathcal{L}_{\rm iou}(\cdot)$ is the generalized IoU [38]:

$$
\mathcal{L}_{\rm iou}(b_{\sigma(i)},\hat{b}_i) = 1 - (\frac{|b_{\sigma(i)} \cap \hat{b}_i|}{|b_{\sigma(i)} \cup \hat{b}_i|} - \frac{|{B(b_{\sigma(i)},\hat{b}_i)}\backslash {b_{\sigma(i)} \cup \hat{b}_i}|}{|B(b_{\sigma(i)},\hat{b}_i)|}). \tag{11}
$$

$\vert.\vert$ means “area”, and the union and intersection of box coordinates are used as shorthands for the boxes themselves. The areas of unions or intersections are computed by min/max of the linear functions of ${b\_{\sigma(i)}}$ and ${\hat{b}\_{i}}$, which makes the loss suciently well-behaved for stochastic gradients. ${B(b\_{\sigma(i)},\hat{b}\_i)}$ means the largest box containing ${b\_{\sigma(i)},\hat{b}\_i}$ (the areas involving $B$ are also computed based on min/max of linear functions of the box coordinates).

**DICE/F-1 loss** [28] The DICE coecient is closely related to the Intersection over Union. If we denote by m the raw mask logits prediction of the model, and $m$ the binary target mask, the loss is dened as:

$$
\mathcal{L}_{\rm DICE}(m,\hat{m}) = 1 -\frac{2m\sigma(\hat{m})+1}{\sigma(\hat{m})+m+1} \tag{11}
$$

where $\sigma$ is the sigmoid function. This loss is normalized by the number of ob jects.

### A.3 Detailed architecture

The detailed description of the transformer used in DETR, with positional encodings passed at every attention layer, is given in Fig. 10. Image features from the CNN backbone are passed through the transformer encoder, together with spatial positional encoding that are added to queries and keys at every multihead self-attention layer. Then, the decoder receives queries (initially set to zero), output positional encoding (object queries), and encoder memory, and produces the nal set of predicted class labels and bounding boxes through multiple multihead self-attention and decoder-encoder attention. The rst self-attention layer in the rst decoder layer can be skipped.

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/End-to-End%20Object%20Detection%20with%20Transformers(DETR)/Fig%2010.png"/></div>

Fig. 10: Architecture of DETR's transformer. Please, see Section A.3 for details.

**Computational complexity** Every self-attention in the encoder has complexity $\mathcal{O}(d^2HW +d(HW)^2): \mathcal{O}(d'd)$ is the cost of computing a single query/key/value embeddings (and $Md' = d$), while $\mathcal{O}(d'(HW)^2)$ is the cost of computing the attention weights for one head. Other computations are negligible. In the decoder, each self-attention is in $\mathcal{O}(d^2N +dN^2)$, and cross-attention between encoder and decoder is in $\mathcal{O}(d^2 (N + HW) + dNHW)$, which is much lower than the encoder since $N \ll HW$ in practice.

**FLOPS computation** Given that the FLOPS for Faster R-CNN depends on the number of proposals in the image, we report the average number of FLOPS for the rst 100 images in the COCO 2017 validation set. We compute the FLOPS with the tool `flop_count_operators` from Detectron2 [50]. We use it without modications for Detectron2 models, and extend it to take batch matrix multiply (`bmm`) into account for DETR models.

### A.4 Training hyperparameters

We train DETR using AdamW [26] with improved weight decay handling, set to $10^{-4}$. We also apply gradient clipping, with a maximal gradient norm of 0.1. The backbone and the transformers are treated slightly dierently, we now discuss the details for both.

**Backbone** ImageNet pretrained backbone ResNet-50 is imported from Torchvision, discarding the last classication layer. Backbone batch normalization weights and statistics are frozen during training, following widely adopted practice in object detection. We fine-tune the backbone using learning rate of $10^{-5}$. We observe that having the backbone learning rate roughly an order of magnitude smaller than the rest of the network is important to stabilize training, especially in the first few epochs.

**Transformer** We train the transformer with a learning rate of $10^{-4}$. Additive dropout of 0.1 is applied after every multi-head attention and FFN before layer normalization. The weights are randomly initialized with Xavier initialization.

**Losses** We use linear combination of $\ell_1$ and GloU losses for bounding box regression with $\lambda_{\rm L1} = 5$ and $\lambda_{\rm iou} = 2$ weights respectively. All models were trained with $N = 100$ decoder query slots.

**Baseline** Our enhanced Faster-RCNN+ baselines use GIoU [38] loss along with the standard $\ell_1$ loss for bounding box regression. We performed a grid search to find the best weights for the losses and the final models use only GIoU loss with weights 20 and 1 for box and proposal regression tasks respectively. For the baselines we adopt the same data augmentation as used in DETR and train it with $9 \times$ schedule (approximately 109 epochs). All other settings are identical to the same models in the Detectron2 model zoo [50].

**Spatial positional encoding** Encoder activations are associated with corresponding spatial positions of image features. In our model we use a fixed absolute encoding to represent these spatial positions. We adopt a generalization of the original Transformer [47] encoding to the 2D case [31]. Specifically, for both spatial coordinates of each embedding we independently use $\frac{d}{2}$ sine and cosine functions with dierent frequencies. We then concatenate them to get the final $d$ channel positional encoding.

### A.5 Additional results

Some extra qualitative results for the panoptic prediction of the DETR-R101 model are shown in Fig.11.

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/End-to-End%20Object%20Detection%20with%20Transformers(DETR)/Fig%2011.png"/></div>

Fig. 11: Comparison of panoptic predictions. From left to right: Ground truth, PanopticFPN with ResNet 101, DETR with ResNet 101

**Increasing the number of instances** By design, DETR cannot predict more objects than it has query slots, i.e. 100 in our experiments. In this section, we analyze the behavior of DETR when approaching this limit. We select a canonical square image of a given class, repeat it on a 1010 grid, and compute the percentage of instances that are missed by the model. To test the model with less than 100 instances, we randomly mask some of the cells. This ensures that the absolute size of the objects is the same no matter how many are visible. To account for the randomness in the masking, we repeat the experiment 100 times with dierent masks. The results are shown in Fig.12. The behavior is similar across classes, and while the model detects all instances when up to 50 are visible, it then starts saturating and misses more and more instances. Notably, when the image contains all 100 instances, the model only detects 30 on average, which is less than if the image contains only 50 instances that are all detected. The counter-intuitive behavior of the model is likely because the images and the detections are far from the training distribution.

Note that this test is a test of generalization out-of-distribution by design, since there are very few example images with a lot of instances of a single class. It is dicult to disentangle, from the experiment, two types of out-of-domain generalization: the image itself vs the number of object per class. But since few to no COCO images contain only a lot of objects of the same class, this type of experiment represents our best eort to understand whether query objects overt the label and position distribution of the dataset. Overall, the experiments suggests that the model does not overt on these distributions since it yields near-perfect detections up to 50 objects. 

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/End-to-End%20Object%20Detection%20with%20Transformers(DETR)/Fig%2012.png"/></div>

Fig. 12: Analysis of the number of instances of various classes missed by DETR depending on how many are present in the image. We report the mean and the standard deviation. As the number of instances gets close to 100, DETR starts saturating and misses more and more objects

### A.6 PyTorch inference code

To demonstrate the simplicity of the approach, we include inference code with PyTorch and Torchvision libraries in Listing 1. The code runs with Python 3.6+, PyTorch 1.4 and Torchvision 0.5. Note that it does not support batching, hence it is suitable only for inference or training with DistributedDataParallel with one image per GPU. Also note that for clarity, this code uses learnt positional encodings in the encoder instead of fixed, and positional encodings are added to the input only instead of at each transformer layer. Making these changes requires going beyond PyTorch implementation of transformers, which hampers readability. The entire code to reproduce the experiments will be made available before the conference. 

```python
import torch
from torch import nn
from torchvision.models import resnet50

class DETR(nn.Module):
    def __init__(self, num_classes, hidden_dim, nheads, num_encoder_layers, num_decoder_layers): super().__init__()
    # We take only convolutional layers from ResNet-50 model
    self.backbone = nn.Sequential(*list(resnet50(pretrained=True).children())[:-2])
    self.conv = nn.Conv2d(2048, hidden_dim, 1)
    self.transformer = nn.Transformer(hidden_dim, nheads,num_encoder_layers, num_decoder_layers)
    self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
    self.linear_bbox = nn.Linear(hidden_dim, 4)
    self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))
    self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
    self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

def forward(self, inputs):
    x = self.backbone(inputs)
    h = self.conv(x)
    H, W = h.shape[-2:]
    pos = torch.cat([
        self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
        self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
    ], dim=-1).flatten(0, 1).unsqueeze(1)
    h = self.transformer(pos + h.flatten(2).permute(2, 0, 1), self.query_pos.unsqueeze(1))
    return self.linear_class(h), self.linear_bbox(h).sigmoid()

detr = DETR(num_classes=91, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6) 
detr.eval()
inputs = torch.randn(1, 3, 800, 1200) 
logits, bboxes = detr(inputs)
```

Listing 1: DETR PyTorch inference code. For clarity it uses learnt positional encodings in the encoder instead of fixed, and positional encodings are added to the input only instead of at each transformer layer. Making these changes requires going beyond PyTorch implementation of transformers, which hampers readability. The entire code to reproduce the experiments will be made available before the conference.
