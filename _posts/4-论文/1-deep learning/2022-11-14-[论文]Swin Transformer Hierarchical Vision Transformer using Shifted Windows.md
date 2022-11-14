---
title: 【论文】Swin Transformer: Hierarchical Vision Transformer using Shifted Windows Swin Transformer:使用移动窗口的乘积式的Vision Transformer（SWIN）
date: 2022-11-14 07:00:00 +/-0800
categories: [论文,深度学习]
tags: [深度学习,论文]     # TAG names should always be lowercase 标记名称应始终为小写
math: true
---

<div align=center>Ze Liu†* &nbsp Yutong Lin†* &nbsp Yue Cao* &nbsp  Han Hu*‡ &nbsp Yixuan Wei†</div>
<div align=center>Zheng Zhang Stephen Lin Baining Guo</div>
<div align = center>Microsoft Research Asia</div>
<div align=center>{v-zeliu1,v-yutlin,yuecao,hanhu,v-yixwe,zhez,stevelin,bainguog}@microsoft.com</div>

>*Equal contribution. †Interns at MSRA. ‡Contact person

## Abstract 摘要

*This paper presents a new vision Transformer, called Swin Transformer, that capably serves as a general-purpose backbone for computer vision. Challenges in adapting Transformer from language to vision arise from differences between the two domains, such as large variations in the scale of visual entities and the high resolution of pixels in images compared to words in text. To address these differences, we propose a hierarchical Transformer whose representation is computed with **S**hifted **win**dows. The shifted windowing scheme brings greater efficiency by limiting self-attention computation to non-overlapping local windows while also allowing for cross-window connection. This hierarchical architecture has the flexibility to model at various scales and has linear computational complexity with respect to image size. These qualities of Swin Transformer make it compatible with a broad range of vision tasks, including image classification (87.3 top-1 accuracy on ImageNet-1K) and dense prediction tasks such as object detection (58.7 box AP and 51.1 mask AP on COCO test- dev) and semantic segmentation (53.5 mIoU on ADE20K val). Its performance surpasses the previous state-of-the-art by a large margin of +2.7 box AP and +2.6 mask AP on COCO, and +3.2 mIoU on ADE20K, demonstrating the potential of Transformer-based models as vision backbones. The hierarchical design and the shifted window approach also prove beneficial for all-MLP architectures. The code and models are publicly available at <https://github.com/microsoft/Swin-Transformer>.*

*本文介绍了一种新的Vision Transformer，称为 Swin Transformer，它能够作为计算机视觉的通用骨干网络。将 Transformer 从语言适应到视觉的挑战来自两个领域之间的差异，例如视觉实体的规模变化很大【尺度上的问题，同一张图中同一个物体可能尺寸不同】，以及与文本中的单词相比，图像中像素的高分辨率【导致输入太大了】。为了解决这些差异，我们提出了一个分层 Transformer，其特征是用 **S**hifted **win**dows 计算的。移动窗口方案通过将 self-attention 计算限制在不重叠的本地窗口上，同时还允许跨窗口连接，从而带来更高的效率。这种分层架构具有在各种尺度上建模的灵活性，并且具有相对于图像大小的线性计算复杂度。 Swin Transformer 的这些特性使其与广泛的视觉任务兼容，包括图像分类（ImageNet-1K 上 87.3 的 top-1 准确率）和密集预测任务，例如对象检测（COCO 测试上的 58.7 box AP 和 51.1 mask AP） dev）和语义分割（ADE20K val 为 53.5 mIoU）。它的性能大大超过了之前的 state-of-the-art，在 COCO 上 +2.7 box AP 和 +2.6 mask AP，在 ADE20K 上 +3.2 mIoU，展示了基于 Transformer 的模型作为视觉骨干的潜力。分层设计和移位窗口方法也证明对全 MLP 架构有益。代码和模型可在 <https://github.com/microsoft/Swin-Transformer> 上公开获得。*

## 1. Introduction 引言

Modeling in computer vision has long been dominated by convolutional neural networks (CNNs). Beginning with AlexNet [39] and its revolutionary performance on the ImageNet image classification challenge, CNN architectures have evolved to become increasingly powerful through greater scale [30,76], more extensive connections [34], and more sophisticated forms of convolution [70,18,84]. With CNNs serving as backbone networks for a variety of vision tasks, these architectural advances have led to performance improvements that have broadly lifted the entire field.

长期以来，计算机视觉建模一直由卷积神经网络 (CNN) 主导。 从 AlexNet [39] 及其在 ImageNet 图像分类挑战中的革命性表现开始，CNN 架构通过更大的规模 [30,76]、更广泛的连接 [34] 和更复杂的卷积形式 [70, 18,84]。 随着 CNN 作为各种视觉任务的骨干网络，这些架构上的进步带来了性能改进，从而广泛提升了整个领域。

<div align =center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Swin%20Transformer%3A%20Hierarchical%20Vision%20Transformer%20using%20Shifted%20Windows/Fig%201.png"/></div>

Figure 1. (a) The proposed Swin Transformer builds hierarchical feature maps by merging image patches (shown in gray) in deeper layers and has linear computation complexity to input image size due to computation of self-attention only within each local window (shown in red). It can thus serve as a general-purpose backbone for both image classification and dense recognition tasks. (b) In contrast, previous vision Transformers [20] produce feature maps of a single low resolution and have quadratic computation complexity to input image size due to computation of self-attention globally.
图 1. (a) 所提出的 Swin Transformer 通过在更深的层中合并图像块（以灰色显示）来构建分层特征图，并且由于仅在每个局部窗口内计算自注意力（如图所示），因此对输入图像大小具有线性计算复杂度 （以红色显示））。 因此，它可以作为图像分类和密集识别任务的通用主干。 (b) 相比之下，以前的视觉 Transformer [20] 生成单个低分辨率的特征图【单一尺寸，低分辨率】，并且由于全局自注意力的计算，输入图像大小具有平方计算复杂度。【之前的Transformer不一定能够很好抓取层级特征】

On the other hand, the evolution of network architectures in natural language processing (NLP) has taken a different path, where the prevalent architecture today is instead the Transformer [64]. Designed for sequence modeling and transduction tasks, the Transformer is notable for its use of attention to model long-range dependencies in the data. Its tremendous success in the language domain has led re-searchers to investigate its adaptation to computer vision, where it has recently demonstrated promising results on cer-tain tasks, specifically image classification [20] and joint vision-language modeling [47].

另一方面，自然语言处理 (NLP) 中网络架构的演变采取了不同的路径，如今流行的架构是 Transformer [64]。 Transformer 专为序列建模和转导任务而设计，以其使用注意力来建模数据中的长期依赖关系而著称。 它在语言领域的巨大成功促使研究人员研究它对计算机视觉的适应性，最近它在某些任务上展示了有希望的结果，特别是图像分类 [20] 和联合视觉语言建模 [47]。

In this paper, we seek to expand the applicability of Transformer such that it can serve as a general-purpose backbone for computer vision, as it does for NLP and as CNNs do in vision. We observe that significant challenges in transferring its high performance in the language domain to the visual domain can be explained by differences between the two modalities. One of these differences involves scale. Unlike the word tokens that serve as the basic elements of processing in language Transformers, visual elements can vary substantially in scale, a problem that receives attention in tasks such as object detection [42,53,54]. In existing Transformer-based models [64,20], tokens are all of a fixed scale, a property unsuitable for these vision applications. Another difference is the much higher resolution of pixels in images compared to words in passages of text. There exist many vision tasks such as semantic segmentation that require dense prediction at the pixel level, and this would be intractable for Transformer on high-resolution images, as the computational complexity of its self-attention is quadratic to image size. To overcome these issues, we propose a general-purpose Transformer backbone, called Swin Transformer, which constructs hierarchical feature maps and has linear computational complexity to image size. As illustrated in Figure 1(a), Swin Transformer constructs a hierarchical rep-resentation by starting from small-sized patches (outlined in gray) and gradually merging neighboring patches in deeper Transformer layers. With these hierarchical feature maps, the Swin Transformer model can conveniently leverage advanced techniques for dense prediction such as feature pyramid networks (FPN) [42] or U-Net [51]. The linear computational complexity is achieved by computing self-attention locally within non-overlapping windows that partition an image (outlined in red). The number of patches in each window is fixed, and thus the complexity becomes linear to image size. These merits make Swin Transformer suitable as a general-purpose backbone for various vision tasks, in contrast to previous Transformer based architectures [20] which produce feature maps of a single resolution and have quadratic complexity.

在本文中，我们寻求扩展 Transformer 的适用性，使其可以作为计算机视觉的通用主干，就像它对 NLP 和 CNN 在视觉中所做的那样。我们观察到，将其在语言领域的高性能转移到视觉领域的重大挑战可以通过两种模式之间的差异来解释。这些差异之一涉及规模。与作为语言 Transformers 中处理的基本元素的词标记不同，视觉元素在规模上可能有很大差异，这是在对象检测等任务中受到关注的问题 [42,53,54]。在现有的基于 Transformer 的模型 [64,20] 中，令牌都是固定比例的，这是不适合这些视觉应用的属性。另一个区别是与文本段落中的单词相比，图像中像素的分辨率要高得多。存在许多视觉任务，例如语义分割，需要在像素级别进行密集预测，这对于 Transformer 在高分辨率图像上的处理来说是很困难的，因为其自注意力的计算复杂度与图像大小成二次方。为了克服这些问题，我们提出了一个通用的 Transformer 主干，称为 Swin Transformer，它构建分层特征图，并且对图像大小具有线性计算复杂度。如图 1(a) 所示，Swin Transformer 通过从小块（灰色轮廓）开始并逐渐合并更深的 Transformer 层中的相邻块来构建分层表示。借助这些分层特征图，Swin Transformer 模型可以方便地利用高级技术进行密集预测，例如特征金字塔网络 (FPN) [42] 或 U-Net [51]。线性计算复杂度是通过在分割图像的非重叠窗口中本地计算自注意力来实现的（红色轮廓）。每个窗口中的块数量是固定的，因此复杂度与图像大小成线性关系。这些优点使 Swin Transformer 适合作为各种视觉任务的通用主干，与以前基于 Transformer 的架构 [20] 形成对比，后者产生单一分辨率的特征图并具有平方复杂度。

A key design element of Swin Transformer is its shift of the window partition between consecutive self-attention layers, as illustrated in Figure 2. The shifted windows bridge the windows of the preceding layer, providing connections among them that significantly enhance modeling power (see Table 4). This strategy is also efficient in regards to real-world latency: all *query* patches within a window share the same *key* set$^1$ , which facilitates memory access in hardware. In contrast, earlier sliding window based self-attention approaches [33,50] suffer from low latency on general hardware due to different key sets for different *query* pixels$^2$. Our experiments show that the proposed *shifted window* approach has much lower latency than the sliding window method, yet is similar in modeling power (see Tables 5 and 6). The shifted window approach also proves beneficial for all-MLP architectures [61].

Swin Transformer 的一个关键设计元素是它在连续的 self-attention 层之间移动窗口分区，如图 2 所示。移动的窗口桥接前一层的窗口，提供它们之间的连接，从而显着增强建模能力（见表 4）。 这种策略在实际延迟方面也很有效：一个窗口内的所有 *query* 块共享相同的 *key* set$^1$ ，这有助于硬件中的内存访问。 相比之下，早期基于滑动窗口的自注意力方法 [33,50] 由于不同 *query* 像素$^2$ 的不同键集，在通用硬件上存在低延迟。 我们的实验表明，所提出的 *shifted window* 方法的延迟比滑动窗口方法低得多，但建模能力相似（见表 5 和表 6）。 移位窗口方法也证明对全 MLP 架构有益[61]。

>$^1$ The query and key are projection vectors in a self-attention layer.
>$^2$ While there are efficient methods to implement a sliding-window
based convolution layer on general hardware, thanks to its shared kernel weights across a feature map, it is difficult for a sliding-window based
self-attention layer to have efficient memory access in practice.

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Swin%20Transformer%3A%20Hierarchical%20Vision%20Transformer%20using%20Shifted%20Windows/Fig%202.png"/></div>

Figure 2. An illustration of the *shifted window* approach for computing self-attention in the proposed Swin Transformer architecture. In layer $l$ (left), a regular window partitioning scheme is adopted, and self-attention is computed within each window. In the next layer $l + 1$ (right), the window partitioning is shifted, resulting in new windows. The self-attention computation in the new windows crosses the boundaries of the previous windows in layer l, providing connections among them.
图 2. 在提议的 Swin Transformer 架构中计算自注意力的 *shifted window* 方法的图示。 在层$l$（左）中，采用常规窗口划分方案，并在每个窗口内计算自注意力。 在下一层 $l + 1$（右）中，窗口分区发生了变化，从而产生了新窗口。 新窗口中的自注意力计算跨越了第 l 层中先前窗口的边界，提供了它们之间的连接。【每个灰色的框是一个小的计算单元，规格是$4 \times 4$,每个红色的框是一个中型的计算单元，规格是$70 \times 7$,】

The proposed Swin Transformer achieves strong performance on the recognition tasks of image classification, object detection and semantic segmentation. It outperforms the ViT / DeiT [20,63] and ResNe(X)t models [30,70] significantly with similar latency on the three tasks. Its 58.7 box AP and 51.1 mask AP on the COCO test-dev set surpass the previous state-of-the-art results by +2.7 box AP (Copy-paste [26] without external data) and +2.6 mask AP (DetectoRS [46]). On ADE20K semantic segmentation, it obtains 53.5 mIoU on the val set, an improvement of +3.2 mIoU over the previous state-of-the-art (SETR [81]). It also achieves a top-1 accuracy of 87.3% on ImageNet-1K image classification.

所提出的 Swin Transformer 在图像分类、目标检测和语义分割的识别任务上取得了强大的性能。 它显着优于 ViT / DeiT [20,63] 和 ResNe(X)t 模型 [30,70]，在三个任务上具有相似的延迟。 它在 COCO 测试开发集上的 58.7 box AP 和 51.1 mask AP 通过 +2.7 box AP（没有外部数据的复制粘贴 [26]）和 +2.6 mask AP（DetectoRS [ 46]）。 在 ADE20K 语义分割上，它在 val 集上获得了 53.5 mIoU，比之前的最新技术（SETR [81]）提高了 +3.2 mIoU。 它还在 ImageNet-1K 图像分类上实现了 87.3% 的 top-1 准确率。

It is our belief that a unified architecture across computer vision and natural language processing could benefit both fields, since it would facilitate joint modeling of visual and textual signals and the modeling knowledge from both domains can be more deeply shared. We hope that Swin Transformer’s strong performance on various vision problems can drive this belief deeper in the community and encourage unified modeling of vision and language signals.

我们相信，跨计算机视觉和自然语言处理的统一架构可以使这两个领域受益，因为它将促进视觉和文本信号的联合建模，并且可以更深入地共享来自两个领域的建模知识。 我们希望 Swin Transformer 在各种视觉问题上的出色表现能够在社区中更深入地推动这种信念，并鼓励视觉和语言信号的统一建模。【作者的展望】

## 2. Related Work 相关工作

**CNN and variants** CNNs serve as the standard network model throughout computer vision. While the CNN has existed for several decades [40], it was not until the introduction of AlexNet [39] that the CNN took off and became mainstream. Since then, deeper and more effective convolutional neural architectures have been proposed to further propel the deep learning wave in computer vision, e.g., VGG [52], GoogleNet [57], ResNet [30], DenseNet [34], weights across a feature map, it is difficult for a sliding-window based self-attention layer to have efficient memory access in practice.

HRNet [65], and EfficientNet [58]. In addition to these architectural advances, there has also been much work on improving individual convolution layers, such as depthwise convolution [70] and deformable convolution [18, 84]. While the CNN and its variants are still the primary backbone architectures for computer vision applications, we highlight the strong potential of Transformer-like architectures for unified modeling between vision and language. Our work achieves strong performance on several basic visual recognition tasks, and we hope it will contribute to a modeling shift.

**Self-attention based backbone architectures** Also inspired by the success of self-attention layers and Transformer architectures in the NLP field, some works employ self-attention layers to replace some or all of the spatial convolution layers in the popular ResNet [33,50,80]. In these works, the self-attention is computed within a local window of each pixel to expedite optimization [33], and they achieve slightly better accuracy/FLOPs trade-offs than the counterpart ResNet architecture. However, their costly memory access causes their actual latency to be significantly larger than that of the convolutional networks [33]. Instead of using sliding windows, we propose to shift windows between consecutive layers, which allows for a more efficient implementation in general hardware.

**Self-attention/Transformers to complement CNNs** Another line of work is to augment a standard CNN architecture with self-attention layers or Transformers. The self-attention layers can complement backbones [67,7,3,71,23,74,55] or head networks [32,27] by providing the capability to encode distant dependencies or heterogeneous interactions. More recently, the encoder-decoder design in Transformer has been applied for the object detection and instance segmentation tasks [8,13,85,56]. Our work explores the adaptation of Transformers for basic visual feature extraction and is complementary to these works.

**Transformer based vision backbones** Most related to our work is the Vision Transformer (ViT) [20] and its follow-ups [63,72,15,28,66]. The pioneering work of ViT directly applies a Transformer architecture on nonoverlapping medium-sized image patches for image classification. It achieves an impressive speed-accuracy tradeoff on image classification compared to convolutional networks. While ViT requires large-scale training datasets ($i.e.,$ JFT-300M) to perform well, DeiT [63] introduces several training strategies that allow ViT to also be effective using the smaller ImageNet-1K dataset. The results of ViT on image classification are encouraging, but its architecture is unsuitable for use as a general-purpose backbone network on dense vision tasks or when the input image resolution is high, due to its low-resolution feature maps and the quadratic increase in complexity with image size. There are a few works applying ViT models to the dense vision tasks of object detection and semantic segmentation by direct upsampling or deconvolution but with relatively lower performance [2,81]. Concurrent to our work are some that modify the ViT architecture [72,15,28] for better image classification. Empirically, we find our Swin Transformer architecture to achieve the best speed-accuracy trade-off among these methods on image classification, even though our work focuses on general-purpose performance rather than specifically on classification. Another concurrent work [66] explores a similar line of thinking to build multi-resolution feature maps on Transformers. Its complexity is still quadratic to image size, while ours is linear and also operates locally which has proven beneficial in modeling the high correlation in visual signals [36,25,41]. Our approach is both efficient and effective, achieving state-of-the-art accuracy on both COCO object detection and ADE20K semantic segmentation.

## 3. Method 方法

### 3.1. Overall Architecture 整体流程

An overview of the Swin Transformer architecture is pre-sented in Figure 3, which illustrates the tiny version (SwinT). It first splits an input RGB image into non-overlapping patches by a patch splitting module, like ViT. Each patch is treated as a “token” and its feature is set as a concatenation of the raw pixel RGB values. In our implementation, we use a patch size of $4 \times 4$ and thus the feature dimension of each patch is $4 \times 4 \times 3 = 48$. A linear embedding layer is applied on this raw-valued feature to project it to an arbitrary dimension (denoted as $C$).

图 3 显示了 Swin Transformer 架构的概述，该图说明了微型版本 (SwinT)。 它首先通过像 ViT 这样的块分割模块将输入的 RGB 图像分割成不重叠的块。 每个块都被视为一个“令牌”，其特征被设置为原始像素 RGB 值的串联。 在我们的实现中，我们使用 $4\times 4$ 的块大小，因此每个块的特征维度是 $4\times 4\times 3 = 48$。 在这个原始值特征上应用线性嵌入层，以将其投影到任意维度（表示为 $C$）。

Several Transformer blocks with modified self-attention computation (*Swin Transformer blocks*) are applied on these patch tokens. The Transformer blocks maintain the number of tokens ($\frac{H}{4} \times \frac{W}{4}$)，and together with the linear embedding are referred to as “Stage 1”.

在这些块词元上应用了几个具有改进的自注意力计算的 Transformer 块（*Swin Transformer 块*）。 Transformer 块保持令牌的数量（$\frac{H}{4} \times \frac{W}{4}$），并与线性嵌入一起被称为“阶段 1”。

To produce a hierarchical representation, the number of tokens is reduced by patch merging layers as the network gets deeper. The first patch merging layer concatenates the features of each group of $2 \times 2$ neighboring patches, and applies a linear layer on the $4C$-dimensional concatenated features. This reduces the number of tokens by a multiple of $2 \times 2 = 4$ ($2 \times$ downsampling of resolution), and the output dimension is set to $2C$. Swin Transformer blocks are applied afterwards for feature transformation, with the resolution kept at $\frac{H}{8} \times \frac{W}{8}$. This first block of patch merging and feature transformation is denoted as “Stage 2”. The procedure is repeated twice, as “Stage 3” and “Stage 4”, with output resolutions of $\frac{H}{16} \times \frac{W}{16}$ and $\frac{H}{32} \times \frac{W}{32}$, respectively. These stages jointly produce a hierarchical representation,with the same feature map resolutions as those of typical convolutional networks, e.g., VGG [52] and ResNet [30]. As a result, the proposed architecture can conveniently replace the backbone networks in existing methods for various vision tasks.

为了产生分层表示，随着网络变得更深，通过块合并层减少令牌的数量。第一个块合并层将每组$2×2$相邻块的特征连接起来，并在$4C$-维连接特征上应用一个线性层。这将令牌的数量减少了 $2 \times 2 = 4$ 的倍数（$2 \times$ 分辨率下采样），并且输出维度设置为 $2C$。之后应用 Swin Transformer 块进行特征转换，分辨率保持在 $\frac{H}{8}\times \frac{W}{8}$。第一个块合并和特征转换被称为“阶段 2”。该过程重复两次，分别为“Stage 3”和“Stage 4”，输出分辨率为 $\frac{H}{16}\times \frac{W}{16}$ 和 $\frac{H}{32 } \times \frac{W}{32}$，分别。这些阶段共同产生分层表示，具有与典型卷积网络相同的特征图分辨率，例如 VGG [52] 和 ResNet [30]。因此，所提出的架构可以方便地替换现有方法中用于各种视觉任务的骨干网络。

<div align =center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Swin%20Transformer%3A%20Hierarchical%20Vision%20Transformer%20using%20Shifted%20Windows/Fig%203.png"/></div>

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Highly%20accurate%20protein%20structure%20prediction/Fig%203-1.png"/></div>

Figure 3. (a) The architecture of a Swin Transformer (Swin-T); (b) two successive Swin Transformer Blocks (notation presented with Eq. (3)). W-MSA and SW-MSA are multi-head self attention modules with regular and shifted windowing configurations, respectively.
图 3. (a) Swin Transformer (Swin-T) 的架构； (b) 两个连续的 Swin 变压器块（用公式 (3) 表示的符号）。 W-MSA 和 SW-MSA 是多头自注意力模块，分别具有常规和移位窗口配置。【像披着Transformer皮的卷积神经网络】

> patch partition:大小是4 × 4  

<div align =center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Swin%20Transformer%3A%20Hierarchical%20Vision%20Transformer%20using%20Shifted%20Windows/swiftWindows.png"/></div>

**Swin Transformer block** Swin Transformer is built by replacing the standard multi-head self attention (MSA) module in a Transformer block by a module based on shifted windows (described in Section 3.2), with other layers kept the same. As illustrated in Figure 3(b), a Swin Transformer block consists of a shifted window based MSA module, followed by a 2-layer MLP with GELU nonlinearity in between. A LayerNorm (LN) layer is applied before each MSA module and each MLP, and a residual connection is applied after each module.

**Swin Transformer 模块** Swin Transformer 是通过将 Transformer 模块中的标准多头自注意力 (MSA) 模块替换为基于移位窗口的模块（在第 3.2 节中描述）而构建的，其他层保持不变。 如图 3(b) 所示，一个 Swin Transformer 模块由一个基于移动窗口的 MSA 模块组成，然后是一个 2 层 MLP，其间具有 GELU 非线性。 在每个 MSA 模块和每个 MLP 之前应用一个 LayerNorm (LN) 层，在每个模块之后应用一个残差连接。

### 3.2. Shifted Window based Self-Attention 基于移动窗口的自注意力

The standard Transformer architecture [64] and its adaptation for image classification [20] both conduct global self-attention, where the relationships between a token and all other tokens are computed. The global computation leads to quadratic complexity with respect to the number of tokens, making it unsuitable for many vision problems requiring an immense set of tokens for dense prediction or to represent a high-resolution image.

标准的 Transformer 架构 [64] 及其对图像分类的适应 [20] 都进行全局自我注意，其中计算了一个词元和所有其他词元之间的关系。 全局计算导致词元数量的平方复杂度，使其不适用于许多需要大量词元集进行密集预测【图像语义分割的目标是将图像的每个像素所属类别进行标注。因为是预测图像中的每个像素，这个任务通常被称为密集预测（dense prediction）。】或表示高分辨率图像的视觉问题。

Self-attention in non-overlapped windows For efficient modeling, we propose to compute self-attention within local windows. The windows are arranged to evenly partition the image in a non-overlapping manner. Supposing each window contains $M \times M$ patches, the computational complexity of a global MSA module and a window based one on an image of $h \times w$ patches are $^3$:

非重叠窗口中的自注意力为了有效建模，我们建议在局部窗口内计算自注意力。 窗口被布置成以不重叠的方式均匀地划分图像。 假设每个窗口包含 $M \times M$个块，一个全局 MSA 模块和一个基于 $h \times w$ 个补丁图像的窗口的计算复杂度为 $^3$：

>$^3$ We omit SoftMax computation in determining complexity

$$
\begin{align*}
& \Omega({\rm MSA}) = 4hwC^2 + 2(hw)^2C, \tag{1} \\
&\Omega{\rm (W-MSA)} = 4hwC^2 + 2M^2hwC, \tag{2}
\end{align*}
$$

where the former is quadratic to patch number $hw$, and the latter is linear when $M$ is fixed (set to $7$ by default). Global self-attention computation is generally unaffordable for a large $hw$, while the window based self-attention is scalable.

其中前者是patch编号 $hw$ 的二次方，而后者在 $M$ 固定时是线性的（默认设置为 $7$）。 全局 self-attention 计算对于大的 $hw$ 来说通常是负担不起的，而基于窗口的 self-attention 是可扩展的。

<div align =center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Swin%20Transformer%3A%20Hierarchical%20Vision%20Transformer%20using%20Shifted%20Windows/multiC.png"/></div>

**Shifted window partitioning in successive blocks** The window-based self-attention module lacks connections across windows, which limits its modeling power. To introduce cross-window connections while maintaining the efficient computation of non-overlapping windows, we propose a shifted window partitioning approach which alternates be-tween two partitioning configurations in consecutive Swin Transformer blocks.

**连续块中的移动窗口分区**基于窗口的自我关注模块缺少跨窗口的连接，这限制了其建模能力。为了在保持非重叠窗口的有效计算的同时引入跨窗口连接，我们提出了一种移位窗口划分方法，该方法在连续Swin Transformer块中的两个划分配置之间交替。

As illustrated in Figure 2, the first module uses a regular window partitioning strategy which starts from the top-left pixel, and the $8 \times 8$ feature map is evenly partitioned into $2 \times 2$ windows of size $4 \times 4 (M = 4)$. Then, the next module adopts a windowing configuration that is shifted from that of the preceding layer, by displacing the windows by $(\lfloor \frac{M}{2} \rfloor  ,\lfloor \frac{M}{2} \rfloor)$ pixels from the regularly partitioned windows.

With the shifted window partitioning approach, consecutive Swin Transformer blocks are computed as

$$
\begin{align*}
& \hat{\mathbf{z}}^l = \mathbf{W \textrm{-} MSA} (\mathbf{LN} (\mathbf{z}^{l—1})) + \mathbf{z}^{l—1}, \\
& \mathbf{z}^l = \mathbf{MLP} (\mathbf{LN} (\hat{\mathbf{z}}^l)) + \hat{\mathbf{z}}^l,\\
& \hat{\mathbf{z}}^{l+1} = \mathbf{SW \textrm{-} MSA }(\mathbf{LN} (\mathbf{z}^l)) + \mathbf{z}^l,\\
& \mathbf{z}^{l+1} = \mathbf{MLP} (\mathbf{LN}  (\hat{\mathbf{z}}^{l+1})) + \hat{\mathbf{z}}^{l+1}, \tag{3}
\end{align*}
$$

where $\hat{\mathbf{z}}^l$ and $\mathbf{z}^l$ denote the output features of the $\rm{(S)W\textrm{-}MSA}$ module and the MLP module for block $l$, respectively;$\rm{W\textrm{-}MSA}$ and $\rm SW\textrm{-}MSA$ denote window based multi-head self-attention using regular and shifted window partitioning configurations, respectively.

<div align =center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Swin%20Transformer%3A%20Hierarchical%20Vision%20Transformer%20using%20Shifted%20Windows/Fig%204.png"/></div>

Figure 4. Illustration of an efficient batch computation approach for self-attention in shifted window partitioning.
图4.移位窗口分区中用于自我关注的高效批计算方法的说明。

>通过掩码实现对应区域的自注意

<div align =center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Swin%20Transformer%3A%20Hierarchical%20Vision%20Transformer%20using%20Shifted%20Windows/mask.png"/></div>

<div align =center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Swin%20Transformer%3A%20Hierarchical%20Vision%20Transformer%20using%20Shifted%20Windows/mask2.png"/></div>

The shifted window partitioning approach introduces connections between neighboring non-overlapping windows in the previous layer and is found to be effective in image classification, object detection, and semantic segmentation, as shown in Table 4.

**Efficient batch computation for shifted configuration** An issue with shifted window partitioning is that it will result in more windows, from $\lceil \frac{h}{M} \rceil \times \lceil \frac{w}{M} \rceil$ to $(\lceil \frac{w}{M} \rceil +1)$ in the shifted configuration, and some of the windows will be smaller than $M \times M^4$ . A naive solution is to pad the smaller windows to a size of $M \times M$ and mask out the padded values when computing attention. When the number of windows in regular partitioning is small, $e.g. 2 \times 2$, the increased computation with this naive solution is considerable ($2 \times 2 \rightarrow 3 \times 3$, which is 2.25 times greater). Here, we propose a *more efficient batch computation approach* by cyclic-shifting toward the top-left direction, as il-lustrated in Figure 4. After this shift, a batched window may be composed of several sub-windows that are not adjacent in the feature map, so a masking mechanism is employed to limit self-attention computation to within each sub-window. With the cyclic-shift, the number of batched windows remains the same as that of regular window partitioning, and thus is also efficient. The low latency of this approach is shown in Table 5.

**移位配置的有效批量计算**移位窗口分区的一个问题是，它将导致更多的窗口，从移位配置中的$\lceil \frac{h}{M} \rceil \times \lceil \frac{w}{M} \rceil$到 $(\lceil \frac{w}{M} \rceil +1)$。一个简单的解决方案是将较小的窗口填充到$M \times M$的大小，并在计算注意力时屏蔽填充的值。当常规分区中的窗口数量很小时，例如$2\times 2$，使用这种原始解决方案增加的计算量相当大（$2 \times 2 \rightarrow 3\times 3$，是2.25倍）。这里，我们通过向左上方向循环移位，提出了一种*更有效的批量计算方法*，如图4所示。在这种移位之后，一个批量窗口可能由几个在特征图中不相邻的子窗口组成，因此采用了一种掩蔽机制，将自关注计算限制在每个子窗口内。通过循环移位，批处理窗口的数量与常规窗口分区的数量相同，因此也是有效的。这种方法的低延迟如表5所示。

>$^4$ 4To make the window size $(M, M)$ divisible by the feature map size of $(h, w)$, bottom-right padding is employed on the feature map if needed.

**Relative position bias** In computing self-attention, we follow [49,1,32,33] by including a relative position bias $B \in \mathbb{R}^{M^2 \times M^2}$ to each head in computing similarity:

$$
{\rm Attention}(Q, K, V) = {\rm SoftMax}(QK^T/\sqrt{d} + B) V, \tag{4}
$$
where $Q,K,V \in \mathbb{R}^{M^2 \times d} $ are the *query, key* and value matrices; $d$ is the *query/key* dimension, and $M^2$ is the number of patches in a window. Since the relative position along each axis lies in the range $[-M + 1, M — 1]$, we parameterize a smaller-sized bias matrix $\hat{B} \in \mathbb{R}^{(2M-1)\times(2M -1)}$, and values in $B$ are taken from $\hat{B}$.

We observe significant improvements over counterparts without this bias term or that use absolute position embedding, as shown in Table 4. Further adding absolute position embedding to the input as in [20] drops performance slightly, thus it is not adopted in our implementation.

The learnt relative position bias in pre-training can be also used to initialize a model for fine-tuning with a different window size through bi-cubic interpolation [20, 63].

>提高窗口的计算效率：
>1. mask掩码:移动窗口后的大小和数量发生改变，见图四 
>2. 用相对的位置编码

### 3.3. Architecture Variants

We build our base model, called Swin-B, to have of model size and computation complexity similar to ViT- B/DeiT-B. We also introduce Swin-T, Swin-S and Swin-L, which are versions of about $0.25 \times$, $0.5 \times$ and $ 2 \times$ the model size and computational complexity, respectively. Note that the complexity of Swin-T and Swin-S are similar to those of ResNet-50 (DeiT-S) and ResNet-101, respectively. The window size is set to $M = 7$ by default. The query dimension of each head is $d = 32,$ and the expansion layer of each MLP is $\alpha = 4$, for all experiments. The architecture hyper-parameters of these model variants are:

我们构建了我们的基础模型，称为 Swin-B，具有类似于 ViT-B/DeiT-B 的模型大小和计算复杂度。 我们还介绍了 Swin-T、Swin-S 和 Swin-L，它们分别是模型大小和计算复杂度约为 0.25 美元、0.5 美元和 2 倍的版本。 请注意，Swin-T 和 Swin-S 的复杂度分别类似于 ResNet-50 (DeiT-S) 和 ResNet-101。 默认情况下，窗口大小设置为 $M = 7$。 对于所有实验，每个 head 的查询维度为 $d = 32,$，每个 MLP 的扩展层为 $\alpha = 4$。 这些模型变体的架构超参数是：

- Swin-T: $C = 96$, layer numbers $ =\{2, 2, 6, 2\}$
- Swin-S: $C = 96$, layer numbers $ =\{2, 2, 18, 2\}$
- Swin-B: $C = 128$, layer numbers $ =\{2, 2, 18, 2\}$
- Swin-L: $C = 192$, layer numbers $ =\{2, 2, 18, 2\}$

where $C$ is the channel number of the hidden layers in the first stage. The model size, theoretical computational complexity (FLOPs), and throughput of the model variants for ImageNet image classification are listed in Table 1.

其中$C$是第一阶段隐藏层的通道号。 表 1 列出了 ImageNet 图像分类模型变体的模型大小、理论计算复杂度 (FLOP) 和吞吐量。

## 4. Experiments 实验

We conduct experiments on ImageNet-1K image classification [19], COCO object detection [43], and ADE20K semantic segmentation [83]. In the following, we first compare the proposed Swin Transformer architecture with the previous state-of-the-arts on the three tasks. Then, we ablate the important design elements of Swin Transformer.

### 4.1. Image Classification on ImageNet-1K

Settings For image classification, we benchmark the proposed Swin Transformer on ImageNet-1K [19], which contains 1.28M training images and 50K validation images from 1,000 classes. The top-1 accuracy on a single crop is reported. We consider two training settings:

- *Regular ImageNet-1K training.* This setting mostly follows [63]. We employ an AdamW [37] optimizer for 300 epochs using a cosine decay learning rate scheduler and 20 epochs of linear warm-up. A batch size of 1024, an initial learning rate of 0.001, and a weight decay of 0.05 are used. We include most of the augmentation and regularization strategies of [63] in training, except for repeated augmentation [31] and EMA [45], which do not enhance performance. Note that this is contrary to [63] where repeated augmentation is crucial to stabilize the training of ViT.
- *常规 ImageNet-1K 训练。*此设置主要遵循 [63]。 我们使用了一个 AdamW [37] 优化器，使用余弦衰减学习率调度器和 20 个线性热身的 epoch 进行 300 个 epoch。 使用 1024 的批量大小、0.001 的初始学习率和 0.05 的权重衰减。 我们在训练中包含了 [63] 的大部分增强和正则化策略，除了重复增强 [31] 和 EMA [45]，它们不会提高性能。 请注意，这与 [63] 中的重复增强对于稳定 ViT 的训练至关重要。
- *Pre-training on ImageNet-22K and fine-tuning on ImageNet-1K.* We also pre-train on the larger ImageNet-22K dataset, which contains 14.2 million images and 22K classes. We employ an AdamW optimizer for 90 epochs using a linear decay learning rate scheduler with a 5-epoch linear warm-up. A batch size of 4096, an initial learning rate of 0.001, and a weight decay of 0.01 are used. In ImageNet-1K fine-tuning, we train the models for 30 epochs with a batch size of 1024, a constant learning rate of $10^{-5}$, and a weight decay of $10^{-8}$.
- *在 ImageNet-22K 上进行预训练并在 ImageNet-1K 上进行微调。*我们还在更大的 ImageNet-22K 数据集上进行预训练，该数据集包含 1420 万张图像和 22K 类。 我们使用具有 5 个 epoch 线性预热的线性衰减学习率调度器使用 AdamW 优化器 90 个 epoch。 使用 4096 的批大小、0.001 的初始学习率和 0.01 的权重衰减。 在 ImageNet-1K 微调中，我们训练模型 30 个 epoch，批量大小为 1024，恒定学习率为 $10^{-5}$，权重衰减为 $10^{-8}$。

**Results with regular ImageNet-1K** training Table 1(a) presents comparisons to other backbones, including both Transformer-based and ConvNet-based, using regular ImageNet-1K training.

Compared to the previous state-of-the-art Transformer-based architecture, i.e. DeiT [63], Swin Transformers no-ticeably surpass the counterpart DeiT architectures with similar complexities: +1.5% for Swin-T (81.3%) over DeiT-S (79.8%) using $224^2$ input, and +1.5%/1.4% for Swin-B (83.3%/84.5%) over DeiT-B (81.8%/83.1%) using $224^2/384^2$ input, respectively.

Compared with the state-of-the-art ConvNets, $i.e.$ RegNet [48] and EfficientNet [58], the Swin Transformer achieves a slightly better speed-accuracy trade-off. Noting that while RegNet [48] and EfficientNet [58] are obtained via a thorough architecture search, the proposed Swin Transformer is adapted from the standard Transformer and has strong potential for further improvement.

**Results with ImageNet-22K pre-training** We also pre-train the larger-capacity Swin-B and Swin-L on ImageNet- 22K. Results fine-tuned on ImageNet-1K image classification are shown in Table 1(b). For Swin-B, the ImageNet-22K pre-training brings $1.8\% \sim 1.9\%$ gains over training on ImageNet-1K from scratch. Compared with the previous best results for ImageNet-22K pre-training, our models achieve significantly better speed-accuracy trade-offs: Swin-B obtains 86.4% top-1 accuracy, which is 2.4% higher than that of ViT with similar inference throughput (84.7 vs. 85.9 images/sec) and slightly lower FLOPs (47.0G vs. 55.4G). The larger Swin-L model achieves 87.3% top-1 accuracy, +0.9% better than that of the Swin-B model.

<div align =center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Swin%20Transformer%3A%20Hierarchical%20Vision%20Transformer%20using%20Shifted%20Windows/Table%201.png"/></div>

Table 1. Comparison of different backbones on ImageNet-1K classification. Throughput is measured using the GitHub repository of [68] and a V100 GPU, following [63].

### 4.2. Object Detection on COCO

**Settings** Object detection and instance segmentation experiments are conducted on COCO 2017, which contains 118K training, 5K validation and 20K test-dev images. An ablation study is performed using the validation set, and a system-level comparison is reported on test-dev. For the ablation study, we consider four typical object detection frameworks: Cascade Mask R-CNN [29, 6], ATSS [79], RepPoints v2 [12], and Sparse RCNN [56] in mmdetection [10]. For these four frameworks, we utilize the same settings: multi-scale training [8, 56] (resizing the input such that the shorter side is between 480 and 800 while the longer side is at most 1333), AdamW [44] optimizer (initial learning rate of 0.0001, weight decay of 0.05, and batch size of 16), and 3x schedule (36 epochs). For system-level comparison, we adopt an improved HTC [9] (denoted as HTC++) with instaboost [22], stronger multi-scale training [7], 6x schedule (72 epochs), soft-NMS [5], and ImageNet-22K pre-trained model as initialization.

<div align =center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Swin%20Transformer%3A%20Hierarchical%20Vision%20Transformer%20using%20Shifted%20Windows/Table%202.png"/></div>

Table 2. Results on COCO object detection and instance segmentation. † denotes that additional decovolution layers are used to produce hierarchical feature maps. * indicates multi-scale testing.

We compare our Swin Transformer to standard ConvNets, i.e. ResNe(X)t, and previous Transformer networks, e.g. DeiT. The comparisons are conducted by changing only the backbones with other settings unchanged. Note that while Swin Transformer and ResNe(X)t are directly applicable to all the above frameworks because of their hierarchical feature maps, DeiT only produces a single resolution of feature maps and cannot be directly applied. For fair comparison, we follow [81] to construct hierarchical feature maps for DeiT using deconvolution layers.

**Comparison to ResNe(X)t** Table 2(a) lists the results of Swin-T and ResNet-50 on the four object detection frameworks. Our Swin-T architecture brings consistent $+3.4 \sim4.2$ box AP gains over ResNet-50, with slightly larger model size, FLOPs and latency.

<div align =center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Swin%20Transformer%3A%20Hierarchical%20Vision%20Transformer%20using%20Shifted%20Windows/Table%203.png"/></div>

Table 3. Results of semantic segmentation on the ADE20K val and test set. † indicates additional deconvolution layers are used to produce hierarchical feature maps. ‡ indicates that the model is pre-trained on ImageNet-22K.

Table 2(b) compares Swin Transformer and ResNe(X)t under different model capacity using Cascade Mask R-CNN. Swin Transformer achieves a high detection accuracy of 51.9 box AP and 45.0 mask AP, which are significant gains of +3.6 box AP and +3.3 mask AP over ResNeXt101-64x4d, which has similar model size, FLOPs and latency. On a higher baseline of 52.3 box AP and 46.0 mask AP using an improved HTC framework, the gains by Swin Transformer are also high, at +4.1 box AP and +3.1 mask AP (see Table 2(c)). Regarding inference speed, while ResNe(X)t is built by highly optimized Cudnn functions, our architecture is implemented with built-in PyTorch functions that are not all well-optimized. A thorough kernel optimization is beyond the scope of this paper.

**Comparison to DeiT** The performance of DeiT-S using the Cascade Mask R-CNN framework is shown in Table 2(b). The results of Swin-T are +2.5 box AP and +2.3 mask AP higher than DeiT-S with similar model size (86M vs. 80M) and significantly higher inference speed (15.3 FPS vs. 10.4 FPS). The lower inference speed of DeiT is mainly due to its quadratic complexity to input image size.

**Comparison to previous state-of-the-art** Table 2(c) compares our best results with those of previous state-of- the-art models. Our best model achieves 58.7 box AP and 51.1 mask AP on COCO test-dev, surpassing the previous best results by +2.7 box AP (Copy-paste [26] without external data) and +2.6 mask AP (DetectoRS [46]).

### 4.3. Semantic Segmentation on ADE20K

Settings ADE20K [83] is a widely-used semantic segmentation dataset, covering a broad range of 150 semantic categories. It has 25K images in total, with 20K for training, 2K for validation, and another 3K for testing. We utilize UperNet [69] in mmseg [16] as our base framework for its high efficiency. More details are presented in the Appendix.

<div align =center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Swin%20Transformer%3A%20Hierarchical%20Vision%20Transformer%20using%20Shifted%20Windows/Table%204.png"/></div>

Table 4. Ablation study on the shifted windows approach and different position embedding methods on three benchmarks, using the Swin-T architecture. w/o shifting: all self-attention modules adopt regular window partitioning, without shifting; abs. pos.: absolute position embedding term of ViT; rel. pos.: the default settings with an additional relative position bias term (see Eq. (4)); app.: the first scaled dot-product term in Eq. (4).

**Results** Table 3 lists the mIoU, model size (#param), FLOPs and FPS for different method/backbone pairs. From these results, it can be seen that Swin-S is +5.3 mIoU higher (49.3 vs. 44.0) than DeiT-S with similar computation cost. It is also +4.4 mIoU higher than ResNet-101, and +2.4 mIoU higher than ResNeSt-101 [78]. Our Swin-L model with ImageNet-22K pre-training achieves 53.5 mIoU on the val set, surpassing the previous best model by +3.2 mIoU (50.3 mIoU by SETR [81] which has a larger model size).

### 4.4. Ablation Study

In this section, we ablate important design elements in the proposed Swin Transformer, using ImageNet-1K image classification, Cascade Mask R-CNN on COCO object detection, and UperNet on ADE20K semantic segmentation.

**Shifted windows** Ablations of the *shifted window* approach on the three tasks are reported in Table 4. Swin-T with the shifted window partitioning outperforms the counterpart built on a single window partitioning at each stage by +1.1% top-1 accuracy on ImageNet-1K, +2.8 box AP/+2.2 mask AP on COCO, and +2.8 mIoU on ADE20K. The results indicate the effectiveness of using shifted windows to build connections among windows in the preceding layers. The latency overhead by *shifted window* is also small, as shown in Table 5.

**Relative position bias** Table 4 shows comparisons of different position embedding approaches. Swin-T with relative position bias yields +1.2%/+0.8% top-1 accuracy on ImageNet-1K, +1.3/+1.5 box AP and +1.1/+1.3 mask AP on COCO, and +2.3/+2.9 mIoU on ADE20K in relation to those without position encoding and with absolute position embedding, respectively, indicating the effectiveness of the relative position bias. Also note that while the inclusion of absolute position embedding improves image classification accuracy (+0.4%), it harms object detection and semantic segmentation (-0.2 box/mask AP on COCO and -0.6 mIoU on ADE20K).

<div align =center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Swin%20Transformer%3A%20Hierarchical%20Vision%20Transformer%20using%20Shifted%20Windows/Table%205.png"/></div>

Table 5. Real speed of different self-attention computation methods and implementations on a V100 GPU.

While the recent ViT/DeiT models abandon translation invariance in image classification even though it has long been shown to be crucial for visual modeling, we find that inductive bias that encourages certain translation invariance is still preferable for general-purpose visual modeling, particularly for the dense prediction tasks of object detection and semantic segmentation.

**Different self-attention methods** The real speed of different self-attention computation methods and implementations are compared in Table 5. Our cyclic implementation is more hardware efficient than naive padding, particularly for deeper stages. Overall, it brings a 13%, 18% and 18% speed-up on Swin-T, Swin-S and Swin-B, respectively.

The self-attention modules built on the proposed shifted window approach are $40.8\times/2.5\times $, $20.2\times/2.5\times$,$9.3\times/2.1\times$, and $7.6\times/1.8\times$ more efficient than those of sliding windows in naive/kernel implementations on four network stages, respectively. Overall, the Swin Transformer architectures built on *shifted windows* are 4.1/1.5, 4.0/1.5, 3.6/1.5 times faster than variants built on *sliding windows* for Swin-T, Swin-S, and Swin-B, respectively. Table 6 compares their accuracy on the three tasks, showing that they are similarly accurate in visual modeling.

Compared to Performer [14], which is one of the fastest Transformer architectures (see [60]), the proposed *shifted window* based self-attention computation and the overall Swin Transformer architectures are slightly faster (see Table 5), while achieving +2.3% top-1 accuracy compared to Performer on ImageNet-1K using Swin-T (see Table 6).

## 5. Conclusion 结论

This paper presents Swin Transformer, a new vision Transformer which produces a hierarchical feature representation and has linear computational complexity with respect to input image size. Swin Transformer achieves the state-of-the-art performance on COCO object detection and ADE20K semantic segmentation, significantly surpassing previous best methods. We hope that Swin Transformer’s strong performance on various vision problems will encourage unified modeling of vision and language signals.

本文介绍了 Swin Transformer，这是一种新的视觉 Transformer，它产生分层特征表示，并且相对于输入图像大小具有线性计算复杂度。 Swin Transformer 在 COCO 对象检测和 ADE20K 语义分割方面实现了最先进的性能，显着超越了以前的最佳方法。 我们希望 Swin Transformer 在各种视觉问题上的强大表现将鼓励视觉和语言信号的统一建模。

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Swin%20Transformer%3A%20Hierarchical%20Vision%20Transformer%20using%20Shifted%20Windows/Table%206.png"/></div>

Table 6. Accuracy of Swin Transformer using different methods for self-attention computation on three benchmarks.

As a key element of Swin Transformer, the *shifted window* based self-attention is shown to be effective and efficient on vision problems, and we look forward to investigating its use in natural language processing as well.

作为 Swin Transformer 的一个关键元素，基于 *shifted window* 的自注意在视觉问题上被证明是有效和高效的，我们也期待研究其在自然语言处理中的应用。

## Acknowledgement 致谢

We thank many colleagues at Microsoft for their help, in particular, Li Dong and Furu Wei for useful discussions; Bin Xiao, Lu Yuan and Lei Zhang for help on datasets.

### A1. Detailed Architectures

The detailed architecture specifications are shown in Table 7, where an input image size of $224 \times 224$ is assumed for all architectures. “Concat $n \times n$” indicates a concatenation of n n neighboring features in a patch. This operation results in a downsampling of the feature map by a rate of n. “96-d” denotes a linear layer with an output dimension of 96. “win. sz. $7 \times 7$” indicates a multi-head self-attention module with window size of $7 \times 7$.

## A2. Detailed Experimental Settings

### A2.1. Image classification on ImageNet-1K

The image classification is performed by applying a global average pooling layer on the output feature map of the last stage, followed by a linear classifier. We find this strategy to be as accurate as using an additional `class` token as in ViT [20] and DeiT [63]. In evaluation, the top-1 accuracy using a single crop is reported.

**Regular ImageNet-1K** training The training settings mostly follow [63]. For all model variants, we adopt a default input image resolution of $224^2$ . For other resolutions such as 3842, we fine-tune the models trained at $224^2$ resolution, instead of training from scratch, to reduce GPU consumption.

When training from scratch with a $224^2$ input, we employ an AdamW [37] optimizer for 300 epochs using a cosine decay learning rate scheduler with 20 epochs of linear warm-up. A batch size of 1024, an initial learning rate of 0.001, a weight decay of 0.05, and gradient clipping with a max norm of 1 are used. We include most of the augmentation and regularization strategies of [63] in training, including RandAugment [17], Mixup [77], Cutmix [75], random erasing [82] and stochastic depth [35], but not repeated augmentation [31] and Exponential Moving Average (EMA) [45] which do not enhance performance. Note that this is contrary to [63] where repeated augmentation is crucial to stabilize the training of ViT. An increasing degree of stochastic depth augmentation is employed for larger models, i.e. $0.2,0.3,0.5$ for Swin-T, Swin-S, and Swin-B, respectively.

For fine-tuning on input with larger resolution, we employ an adamW [37] optimizer for 30 epochs with a constant learning rate of $10^{-5}$, weight decay of $10^{-8}$, and the same data augmentation and regularizations as the first stage except for setting the stochastic depth ratio to 0.1.

ImageNet-22K pre-training We also pre-train on the larger ImageNet-22K dataset, which contains 14.2 million images and 22K classes. The training is done in two stages. For the first stage with $224^2$ input, we employ an AdamW optimizer for 90 epochs using a linear decay learning rate scheduler with a 5-epoch linear warm-up. A batch size of 4096, an initial learning rate of 0.001, and a weight decay of 0.01 are used. In the second stage of ImageNet-1K finetuning with $224^2/384^2$ input, we train the models for 30 epochs with a batch size of 1024, a constant learning rate of $10^{—5}$, and a weight decay of $10^{—8}$.

### A2.2. Object detection on COCO

For an ablation study, we consider four typical object detection frameworks: Cascade Mask R-CNN [29, 6], ATSS [79], RepPoints v2 [12], and Sparse RCNN [56] in mmdetection [10]. For these four frameworks, we utilize the same settings: multi-scale training [8, 56] (resizing the input such that the shorter side is between 480 and 800 while the longer side is at most 1333), AdamW [44] optimizer (initial learning rate of 0.0001, weight decay of 0.05, and batch size of 16), and 3x schedule (36 epochs with the learning rate decayed by $10 \times$ at epochs 27 and 33).

For system-level comparison, we adopt an improved HTC [9] (denoted as HTC++) with instaboost [22], stronger multi-scale training [7] (resizing the input such that the shorter side is between 400 and 1400 while the longer side is at most 1600), 6x schedule (72 epochs with the learning rate decayed at epochs 63 and 69 by a factor of 0.1), soft-NMS [5], and an extra global self-attention layer appended at the output of last stage and ImageNet-22K pre-trained model as initialization. We adopt stochastic depth with ratio of 0.2 for all Swin Transformer models.

### A2.3. Semantic segmentation on ADE20K

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Swin%20Transformer%3A%20Hierarchical%20Vision%20Transformer%20using%20Shifted%20Windows/Table%207.png"/></div>

Table 7. Detailed architecture specifications.

ADE20K [83] is a widely-used semantic segmentation dataset, covering a broad range of 150 semantic categories. It has 25K images in total, with 20K for training, 2K for val-idation, and another 3K for testing. We utilize UperNet [69] in mmsegmentation [16] as our base framework for its high efficiency.

In training, we employ the AdamW [44] optimizer with an initial learning rate of $6 x 10^{-5}$, a weight decay of 0.01, a scheduler that uses linear learning rate decay, and a linear warmup of 1,500 iterations. Models are trained on 8 GPUs with 2 images per GPU for 160K iterations. For augmentations, we adopt the default setting in mmsegmentation of random horizontal flipping, random re-scaling within ratio range [0.5, 2.0] and random photometric distortion. Stochastic depth with ratio of 0.2 is applied for all Swin Transformer models. Swin-T, Swin-S are trained on the standard setting as the previous approaches with an input of $512\times 512$. Swin-B and Swin-L with ‡ indicate that these two models are pre-trained on ImageNet-22K, and trained with the input of $640 \times 640$.

In inference, a multi-scale test using resolutions that are $[0.5, 0.75, 1.0, 1.25, 1.5, 1.75]\times$ of that in training is employed. When reporting test scores, both the training images and validation images are used for training, following common practice [71].

## A3. More Experiments

### A3.1. Image classification with different input size

Table 8 lists the performance of Swin Transformers with different input image sizes from $224^2$ to $384^2$ . In general, a larger input resolution leads to better top-1 accuracy but with slower inference speed.

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Swin%20Transformer%3A%20Hierarchical%20Vision%20Transformer%20using%20Shifted%20Windows/Table%208.png"/></div>

Table 8. Swin Transformers with different input image size on ImageNet-1K classification.

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Swin%20Transformer%3A%20Hierarchical%20Vision%20Transformer%20using%20Shifted%20Windows/Table%209.png"/></div>

Table 9. Comparison of the SGD and AdamW optimizers for ResNe(X)t backbones on COCO object detection using the Cascade Mask R-CNN framework.

### A3.2. Different Optimizers for ResNe(X)t on COCO

Table 9 compares the AdamW and SGD optimizers of the ResNe(X)t backbones on COCO object detection. The Cascade Mask R-CNN framework is used in this comparison. While SGD is used as a default optimizer for Cascade Mask R-CNN framework, we generally observe improved accuracy by replacing it with an AdamW optimizer, particularly for smaller backbones. We thus use AdamW for ResNe(X)t backbones when compared to the proposed Swin Transformer architectures.

### A3.3. Swin MLP-Mixer

We apply the proposed hierarchical design and the shifted window approach to the MLP-Mixer architectures [61], referred to as Swin-Mixer. Table 10 shows the performance of Swin-Mixer compared to the original MLP- Mixer architectures MLP-Mixer [61] and a follow-up approach, ResMLP [61]. Swin-Mixer performs significantly better than MLP-Mixer (81.3% vs. 76.4%) using slightly smaller computation budget (10.4G vs. 12.7G). It also has better speed accuracy trade-off compared to ResMLP [62]. These results indicate the proposed hierarchical design and the shifted window approach are generalizable.

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Swin%20Transformer%3A%20Hierarchical%20Vision%20Transformer%20using%20Shifted%20Windows/Table%2010.png"/></div>

Table 10. Performance of Swin MLP-Mixer on ImageNet-IK classification. $D$ indictes the number of channels per head. Throughput is measured using the GitHub repository of [68] and a V100 GPU, following [63].

## References

[1] Hangbo Bao, Li Dong, Furu Wei, Wenhui Wang, Nan Yang, Xiaodong Liu, Yu Wang, Jianfeng Gao, Songhao Piao, Ming Zhou, et al. Unilmv2: Pseudo-masked language models for unified language model pre-training. In International Con-ference on Machine Learning, pages 642-652. PMLR, 2020. 5

[2] Josh Beal, Eric Kim, Eric Tzeng, Dong Huk Park, Andrew Zhai, and Dmitry Kislyuk. Toward transformer-based object detection. arXiv preprint arXiv:2012.09958, 2020. 3

[3] Irwan Bello, Barret Zoph, Ashish Vaswani, Jonathon Shlens, and Quoc V. Le. Attention augmented convolutional networks, 2020. 3

[4] Alexey Bochkovskiy, Chien-Yao Wang, and HongYuan Mark Liao. Yolov4: Optimal speed and accuracy of object detection. arXiv preprint arXiv:2004.10934, 2020. 7

[5] Navaneeth Bodla, Bharat Singh, Rama Chellappa, and Larry S. Davis. Soft-nms - improving object detection with one line of code. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), Oct 2017. 6, 9

[6] Zhaowei Cai and Nuno Vasconcelos. Cascade r-cnn: Delving into high quality object detection. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 6154-6162, 2018. 6, 9

[7] Yue Cao, Jiarui Xu, Stephen Lin, Fangyun Wei, and Han Hu. Gcnet: Non-local networks meet squeeze-excitation networks and beyond. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops, Oct 2019. 3, 6, 7, 9

[8] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to- end object detection with transformers. In European Conference on Computer Vision, pages 213-229. Springer, 2020. 3, 6, 9

[9] Kai Chen, Jiangmiao Pang, Jiaqi Wang, Yu Xiong, Xiaox- iao Li, Shuyang Sun, Wansen Feng, Ziwei Liu, Jianping Shi, Wanli Ouyang, et al. Hybrid task cascade for instance segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 49744983, 2019. 6,9

[10] Kai Chen, Jiaqi Wang, Jiangmiao Pang, Yuhang Cao, Yu Xiong, Xiaoxiao Li, Shuyang Sun, Wansen Feng, Ziwei Liu, Jiarui Xu, et al. Mmdetection: Open mmlab detection toolbox and benchmark. arXiv preprint arXiv:1906.07155, 2019. 6, 9

[11] Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, and Hartwig Adam. Encoder-decoder with atrous separable convolution for semantic image segmentation. In Proceedings of the European conference on computer vision (ECCV), pages 801-818, 2018. 7

[12] Yihong Chen, Zheng Zhang, Yue Cao, Liwei Wang, Stephen Lin, and Han Hu. Reppoints v2: Verification meets regression for object detection. In NeurIPS, 2020. 6, 7, 9

[13] Cheng Chi, Fangyun Wei, and Han Hu. Relationnet++: Bridging visual representations for object detection via transformer decoder. In NeurIPS, 2020. 3, 7

[14] Krzysztof Marcin Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sar- los, Peter Hawkins, Jared Quincy Davis, Afroz Mohiuddin, Lukasz Kaiser, David Benjamin Belanger, Lucy J Colwell, and Adrian Weller. Rethinking attention with performers. In International Conference on Learning Representations, 2021. 8, 9

[15] Xiangxiang Chu, Bo Zhang, Zhi Tian, Xiaolin Wei, and Huaxia Xia. Do we really need explicit position encodings for vision transformers? arXiv preprint arXiv:2102.10882, 2021. 3

[16] MMSegmentation Contributors. MMSegmentation: Openmmlab semantic segmentation toolbox and benchmark. <https://github.com/open-mmlab/>
mmsegmentation, 2020. 8, 10

[17] Ekin D Cubuk, Barret Zoph, Jonathon Shlens, and Quoc V Le. Randaugment: Practical automated data augmentation with a reduced search space. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pages 702-703, 2020. 9

[18] Jifeng Dai, Haozhi Qi, Yuwen Xiong, Yi Li, Guodong Zhang, Han Hu, and Yichen Wei. Deformable convolutional networks. In Proceedings of the IEEE International Conference on Computer Vision, pages 764-773, 2017. 1, 3

[19] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248-255. Ieee, 2009. 5

[20] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021. 1, 2, 3, 4, 5, 6, 9

[21] Xianzhi Du, Tsung-Yi Lin, Pengchong Jin, Golnaz Ghiasi, Mingxing Tan, Yin Cui, Quoc V Le, and Xiaodan Song. Spinenet: Learning scale-permuted backbone for recognition and localization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11592-11601, 2020. 7

[22] Hao-Shu Fang, Jianhua Sun, Runzhong Wang, Minghao Gou, Yong-Lu Li, and Cewu Lu. Instaboost: Boosting instance segmentation via probability map guided copypasting. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 682-691, 2019. 6, 9

[23] Jun Fu, Jing Liu, Haijie Tian, Yong Li, Yongjun Bao, Zhi- wei Fang, and Hanqing Lu. Dual attention network for scene segmentation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 31463154, 2019. 3, 7

[24] Jun Fu, Jing Liu, Yuhang Wang, Yong Li, Yongjun Bao, Jin- hui Tang, and Hanqing Lu. Adaptive context network for scene parsing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6748-6757, 2019. 7

[25] Kunihiko Fukushima. Cognitron: A self-organizing multilayered neural network. Biological cybernetics, 20(3):121- 136, 1975. 3

[26] Golnaz Ghiasi, Yin Cui, Aravind Srinivas, Rui Qian, Tsung- Yi Lin, Ekin D Cubuk, Quoc V Le, and Barret Zoph. Simple copy-paste is a strong data augmentation method for instance segmentation. arXiv preprint arXiv:2012.07177, 2020. 2, 7

[27] Jiayuan Gu, Han Hu, Liwei Wang, Yichen Wei, and Jifeng Dai. Learning region features for object detection. In Proceedings of the European Conference on Computer Vision (ECCV), 2018. 3

[28] Kai Han, An Xiao, Enhua Wu, Jianyuan Guo, Chunjing Xu, and Yunhe Wang. Transformer in transformer. arXiv preprint arXiv:2103.00112, 2021. 3

[29] Kaiming He, Georgia Gkioxari, Piotr Dollar, and Ross Gir- shick. Mask r-cnn. In Proceedings of the IEEE international conference on computer vision, pages 2961-2969, 2017. 6, 9

[30] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770-778, 2016. 1, 2, 4

[31] Elad Hoffer, Tal Ben-Nun, Itay Hubara, Niv Giladi, Torsten Hoefler, and Daniel Soudry. Augment your batch: Improving generalization through instance repetition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8129-8138, 2020. 6, 9

[32] Han Hu, Jiayuan Gu, Zheng Zhang, Jifeng Dai, and Yichen Wei. Relation networks for object detection. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 3588-3597, 2018. 3, 5

[33] Han Hu, Zheng Zhang, Zhenda Xie, and Stephen Lin. Local relation networks for image recognition. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 3464-3473, October 2019. 2, 3, 5

[34] Gao Huang, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q Weinberger. Densely connected convolutional networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4700-4708, 2017. 1, 2

[35] Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, and Kilian Q Weinberger. Deep networks with stochastic depth. In European conference on computer vision, pages 646-661. Springer, 2016. 9

[36] David H Hubel and Torsten N Wiesel. Receptive fields, binocular interaction and functional architecture in the cat’s visual cortex. The Journal of physiology, 160(1):106-154, 1962. 3

[37] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014. 5, 9

[38] Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Joan Puigcerver, Jessica Yung, Sylvain Gelly, and Neil Houlsby. Big transfer (bit): General visual representation learning. arXiv preprint arXiv:1912.11370, 6(2):8, 2019. 6

[39] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural net-works. In Advances in neural information processing systems, pages 1097-1105, 2012. 1, 2

[40] Yann LeCun, Leon Bottou, Yoshua Bengio, Patrick Haffner, et al. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278-2324, 1998. 2

[41] Yann LeCun, Patrick Haffner, Leon Bottou, and Yoshua Ben- gio. Object recognition with gradient-based learning. In Shape, contour and grouping in computer vision, pages 319345. Springer, 1999. 3

[42] Tsung-Yi Lin, Piotr Dollar, Ross Girshick, Kaiming He, Bharath Hariharan, and Serge Belongie. Feature pyramid networks for object detection. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), July 2017. 2

[43] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pages 740-755. Springer, 2014. 5

[44] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. 6, 9, 10

[45] Boris T Polyak and Anatoli B Juditsky. Acceleration of stochastic approximation by averaging. SIAM journal on control and optimization, 30(4):838-855, 1992. 6, 9

[46] Siyuan Qiao, Liang-Chieh Chen, and Alan Yuille. Detectors: Detecting objects with recursive feature pyramid and switchable atrous convolution. arXiv preprint arXiv:2006.02334, 2020. 2, 7

[47] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision, 2021. 1

[48] Ilija Radosavovic, Raj Prateek Kosaraju, Ross Girshick, Kaiming He, and Piotr Dollar. Designing network design spaces. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10428- 10436, 2020. 6

[49] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21(140):1-67, 2020. 5

[50] Prajit Ramachandran, Niki Parmar, Ashish Vaswani, Irwan Bello, Anselm Levskaya, and Jon Shlens. Stand-alone selfattention in vision models. In Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019. 2, 3

[51] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U- net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention, pages 234-241. Springer, 2015. 2

[52] K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. In International Conference on Learning Representations, May 2015. 2, 4

[53] Bharat Singh and Larry S Davis. An analysis of scale invariance in object detection snip. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3578-3587, 2018. 2

[54] Bharat Singh, Mahyar Najibi, and Larry S Davis. Sniper: Efficient multi-scale training. In Advances in Neural Information Processing Systems, volume 31. Curran Associates, Inc., 2018. 2

[55] Aravind Srinivas, Tsung-Yi Lin, Niki Parmar, Jonathon Shlens, Pieter Abbeel, and Ashish Vaswani. Bottleneck transformers for visual recognition. arXiv preprint arXiv:2101.11605, 2021. 3

[56] Peize Sun, Rufeng Zhang, Yi Jiang, Tao Kong, Chen- feng Xu, Wei Zhan, Masayoshi Tomizuka, Lei Li, Zehuan Yuan, Changhu Wang, et al. Sparse r-cnn: End-to-end object detection with learnable proposals. arXiv preprint arXiv:2011.12450, 2020. 3, 6, 9

[57] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. Going deeper with convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1-9, 2015. 2

[58] Mingxing Tan and Quoc Le. Efficientnet: Rethinking model scaling for convolutional neural networks. In International Conference on Machine Learning, pages 6105-6114. PMLR, 2019. 3, 6

[59] Mingxing Tan, Ruoming Pang, and Quoc V Le. Efficientdet: Scalable and efficient object detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10781-10790, 2020. 7

[60] Yi Tay, Mostafa Dehghani, Samira Abnar, Yikang Shen, Dara Bahri, Philip Pham, Jinfeng Rao, Liu Yang, Sebastian Ruder, and Donald Metzler. Long range arena : A benchmark for efficient transformers. In International Conference on Learning Representations, 2021. 8

[61] Ilya Tolstikhin, Neil Houlsby, Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Thomas Unterthiner, Jessica Yung, Andreas Steiner, Daniel Keysers, Jakob Uszkoreit, Mario Lucic, and Alexey Dosovitskiy. Mlp-mixer: An all-mlp architecture for vision, 2021. 2, 10, 11

[62] Hugo Touvron, Piotr Bojanowski, Mathilde Caron, Matthieu Cord, Alaaeldin El-Nouby, Edouard Grave, Gautier Izac- ard, Armand Joulin, Gabriel Synnaeve, Jakob Verbeek, and Herve Jegou. Resmlp: Feedforward networks for image classification with data-efficient training, 2021. 11

[63] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Herve Jegou. Training data-efficient image transformers & distillation through at-tention. arXiv preprint arXiv:2012.12877, 2020. 2, 3, 5, 6, 9, 11

[64] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszko- reit, Llion Jones, Aidan N Gomez,匕ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, pages 5998-6008, 2017. 1, 2, 4

[65] Jingdong Wang, Ke Sun, Tianheng Cheng, Borui Jiang, Chaorui Deng, Yang Zhao, Dong Liu, Yadong Mu, Mingkui Tan, Xinggang Wang, et al. Deep high-resolution representation learning for visual recognition. IEEE transactions on pattern analysis and machine intelligence, 2020. 3

[66] Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, and Ling Shao. Pyramid vision transformer: A versatile backbone for dense prediction without convolutions. arXiv preprint arXiv:2102.12122, 2021. 3

[67] Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaim- ing He. Non-local neural networks. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2018, 2018. 3

[68] Ross Wightman. Pytorch image models. <https://github.com/rwightman/> pytorch-image-models, 2019. 6, 11

[69] Tete Xiao, Yingcheng Liu, Bolei Zhou, Yuning Jiang, and Jian Sun. Unified perceptual parsing for scene understanding. In Proceedings of the European Conference on Computer Vision (ECCV), pages 418-434, 2018. 7, 8, 10

[70] Saining Xie, Ross Girshick, Piotr Dollar, Zhuowen Tu, and Kaiming He. Aggregated residual transformations for deep neural networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 14921500, 2017. 1,2,3

[71] Minghao Yin, Zhuliang Yao, Yue Cao, Xiu Li, Zheng Zhang, Stephen Lin, and Han Hu. Disentangled non-local neural networks. In Proceedings of the European conference on computer vision (ECCV), 2020. 3, 7, 10

[72] Li Yuan, Yunpeng Chen, Tao Wang, Weihao Yu, Yujun Shi, Francis EH Tay, Jiashi Feng, and Shuicheng Yan. Tokens- to-token vit: Training vision transformers from scratch on imagenet. arXiv preprint arXiv:2101.11986, 2021. 3

[73] Yuhui Yuan, Xilin Chen, and Jingdong Wang. Object- contextual representations for semantic segmentation. In 
16th European Conference Computer Vision (ECCV 2020), August 2020. 7

[74] Yuhui Yuan and Jingdong Wang. Ocnet: Object context net-work for scene parsing. arXiv preprint arXiv:1809.00916, 2018. 3

[75] Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, and Youngjoon Yoo. Cutmix: Regularization strategy to train strong classifiers with localizable features. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6023-6032, 2019. 9

[76] Sergey Zagoruyko and Nikos Komodakis. Wide residual net-works. In BMVC, 2016. 1

[77] Hongyi Zhang, Moustapha Cisse, Yann N Dauphin, and David Lopez-Paz. mixup: Beyond empirical risk minimization. arXiv preprint arXiv:1710.09412, 2017. 9

[78] Hang Zhang, Chongruo Wu, Zhongyue Zhang, Yi Zhu, Zhi Zhang, Haibin Lin, Yue Sun, Tong He, Jonas Mueller, R Manmatha, et al. Resnest: Split-attention networks. arXiv preprint arXiv:2004.08955, 2020. 7, 8

[79] Shifeng Zhang, Cheng Chi, Yongqiang Yao, Zhen Lei, and Stan Z Li. Bridging the gap between anchor-based and anchor-free detection via adaptive training sample selection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9759-9768, 2020. 6, 9

[80] Hengshuang Zhao, Jiaya Jia, and Vladlen Koltun. Exploring self-attention for image recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10076-10085, 2020. 3

[81] Sixiao Zheng, Jiachen Lu, Hengshuang Zhao, Xiatian Zhu, Zekun Luo, Yabiao Wang, Yanwei Fu, Jianfeng Feng, Tao Xiang, Philip HS Torr, et al. Rethinking semantic segmentation from a sequence-to-sequence perspective with transformers. arXiv preprint arXiv:2012.15840, 2020. 2, 3, 7, 8

[82] Zhun Zhong, Liang Zheng, Guoliang Kang, Shaozi Li, and Yi Yang. Random erasing data augmentation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 13001-13008, 2020. 9

[83] Bolei Zhou, Hang Zhao, Xavier Puig, Tete Xiao, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Semantic understanding of scenes through the ade20k dataset. International Journal on Computer Vision, 2018. 5, 7, 10

[84] Xizhou Zhu, Han Hu, Stephen Lin, and Jifeng Dai. Deformable convnets v2: More deformable, better results. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 9308-9316, 2019. 1, 3

[85] Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, and Jifeng Dai. Deformable fdetrg: Deformable transformers for end-to-end object detection. In International Conference on Learning Representations, 2021. 3
