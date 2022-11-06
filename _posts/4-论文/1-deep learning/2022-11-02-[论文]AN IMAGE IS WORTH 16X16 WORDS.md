---
title: 【论文】AN IMAGE IS WORTH 16 × 16 WORDS:TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE 一幅图像值$16 \times 16$个字：用于大规模图像识别的TRANSFORMERS
date: 2022-11-02 08:00:00 +/-0800
categories: [论文,深度学习]
tags: [深度学习]     # TAG names should always be lowercase 标记名称应始终为小写
---

<script>
MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']]
  },
  svg: {
    fontCache: 'global'
  }
};
</script>
<script type="text/javascript" id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js">
</script>

> 把图片以$16 × 16$大小的方格进行裁切

<div align = center>
Alexey Dosovitskiy∗,†, Lucas Beyer*, Alexander Kolesnikov*, Dirk Weissenborn*,
Xiaohua Zhai*, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer,
Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby;∗,†
</div>
<div align = center>
*equal technical contribution, †equal advising
</div>
</div>
<div align = center>Google Research, Brain Team
</div>
<div align = center>{adosovitskiy, neilhoulsby}@google.com</div>

## ABSTRACT 摘要

While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited. In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure in place. We show that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks. When pre-trained on large amounts of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train.$^1$

虽然 Transformer 架构已成为自然语言处理任务的事实标准，但其在计算机视觉中的应用仍然有限。 在视觉上，注意力要么与卷积网络结合使用，要么用于替换卷积网络的某些组件，同时保持其整体结构不变。 我们表明，这种对 CNN 的依赖是不必要的，直接应用于图像块序列的纯变换器可以在图像分类任务上表现得非常好。 当对大量数据进行预训练并迁移到多个中型或小型图像识别基准（ImageNet、CIFAR-100、VTAB 等）时，与与最先进的卷积网络相比，Vision Transformer（ViT）获得了出色的结果，同时需要更少的计算资源来训练。$^1$

>$^1$ Fine-tuning code and pre-trained models are available at https://github.com/google-research/vision_transformer

# 1 INTRODUCTION 引言

Self-attention-based architectures, in particular Transformers (Vaswani et al., 2017), have become the model of choice in natural language processing (NLP). The dominant approach is to pre-train on a large text corpus and then fine-tune on a smaller task-specific dataset (Devlin et al., 2019). Thanks to Transformers’ computational efficiency and scalability, it has become possible to train models of unprecedented size, with over 100B parameters (Brown et al., 2020; Lepikhin et al., 2020). With the models and datasets growing, there is still no sign of saturating performance.

基于自注意的架构，特别是Transformers（Vaswani等人，2017），已经成为自然语言处理（NLP）的首选模型。占主导地位的方法是在大型文本语料库上进行预训练，然后在较小的特定任务数据集上进行微调（Devlin等人，2019）。由于Transformer的计算效率和可扩展性，已经有可能训练出规模空前的模型，参数超过100B（Brown等人，2020；Lepikhin等人，2020）。随着模型和数据集的增长，仍然没有性能饱和的迹象。

In computer vision, however, convolutional architectures remain dominant (LeCun et al., 1989; Krizhevsky et al., 2012; He et al., 2016). Inspired by NLP successes, multiple works try combining CNN-like architectures with self-attention (Wang et al., 2018; Carion et al., 2020), some replacing the convolutions entirely (Ramachandran et al., 2019; Wang et al., 2020a). The latter models, while theoretically efficient, have not yet been scaled effectively on modern hardware accelerators due to the use of specialized attention patterns. Therefore, in large-scale image recognition, classic ResNet- like architectures are still state of the art (Mahajan et al., 2018; Xie et al., 2020; Kolesnikov et al., 2020).

然而，在计算机视觉中，卷积架构仍然占主导地位（LeCun 等人，1989；Krizhevsky 等人，2012；He 等人，2016）。 受 NLP 成功的启发，多项工作尝试将类似 CNN 的架构与自我注意相结合（Wang 等人，2018 年；Carion 等人，2020 年），其中一些完全取代了卷积（Ramachandran 等人，2019 年；Wang 等人）。 , 2020a)。 后一种模型虽然理论上有效，但由于使用了专门的注意力模式，尚未在现代硬件加速器上有效地扩展。 因此，在大规模图像识别中，经典的类 ResNet 架构仍然是最先进的（Mahajan 等人，2018；Xie 等人，2020；Kolesnikov 等人，2020）。

Inspired by the Transformer scaling successes in NLP, we experiment with applying a standard Transformer directly to images, with the fewest possible modifications. To do so, we split an image into patches and provide the sequence of linear embeddings of these patches as an input to a Transformer. Image patches are treated the same way as tokens (words) in an NLP application. We train the model on image classification in supervised fashion.

受 NLP 中 Transformer 可扩展性成功的启发，我们尝试将标准 Transformer 直接应用于图像，并尽可能减少修改。 为此，我们将图像拆分为块，并提供这些块的线性嵌入序列作为 Transformer 的输入。 图像块的处理方式与 NLP 应用程序中的标记（单词）相同。 我们以监督方式训练模型进行图像分类。

When trained on mid-sized datasets such as ImageNet without strong regularization, these models yield modest accuracies of a few percentage points below ResNets of comparable size. This seemingly discouraging outcome may be expected: Transformers lack some of the inductive biases inherent to CNNs, such as translation equivariance and locality, and therefore do not generalize well when trained on insufficient amounts of data.

当在没有强正则化[强约束性]的 ImageNet 等中型数据集上进行训练时，这些模型产生的准确度比同等大小的 ResNet 低几个百分点。 这种看似令人沮丧的结果可能是意料之中的：Transformers 缺乏 CNN 固有的一些归纳偏置 [指一种先验的知识或者说指一种预设的假设]，例如平移等效性和局部性，因此在数据量不足的情况下训练时不能很好地泛化。

>卷积神经网络的两个假设：
> 1. locality 图片相邻的区域有相邻的特征
> 2. translation equivariance 平移同变性 $f(g(x))=g(f(x))$，即平移不影响卷积操作。

However, the picture changes if the models are trained on larger datasets (14M-300M images). We find that large scale training trumps inductive bias. Our Vision Transformer (ViT) attains excellent results when pre-trained at sufficient scale and transferred to tasks with fewer datapoints. When pre-trained on the public ImageNet-21k dataset or the in-house JFT-300M dataset, ViT approaches or beats state of the art on multiple image recognition benchmarks. In particular, the best model reaches the accuracy of $88.55\%$ on ImageNet, $90.72\%$ on ImageNet-ReaL, $94.55\%$ on CIFAR-100, and $77.63\%$ on the VTAB suite of 19 tasks.

但是，如果模型在更大的数据集（14M-300M 图像）上训练，情况就会发生变化。 我们发现大规模训练胜过归纳偏置。 我们的 Vision Transformer (ViT) 在以足够的规模进行预训练并迁移到下游的任务时获得了出色的结果。 当在公共 ImageNet-21k 数据集或内部 JFT-300M 数据集上进行预训练时，ViT 在多个图像识别基准上接近或超过了最先进的水平。 特别是，最好的模型在 ImageNet 上达到了 $88.55\%$的准确度，在 ImageNet-ReaL 上达到了  $90.72\%$ ，在 CIFAR-100 上达到了 $94.55\%$，在 19 个任务的 VTAB 套件上达到了 $77.63\%$ 。

## 2 RELATED WORK 相关工作

Transformers were proposed by Vaswani et al. (2017) for machine translation, and have since become the state of the art method in many NLP tasks. Large Transformer-based models are often pre-trained on large corpora and then fine-tuned for the task at hand: BERT (Devlin et al., 2019) uses a denoising self-supervised pre-training task, while the GPT line of work uses language modeling as its pre-training task (Radford et al., 2018; 2019; Brown et al., 2020).

Transformer是由 Vaswani 等人提出的。 （2017）用于机器翻译，并已成为许多 NLP 任务中最先进的方法。 基于大型 Transformer 的模型通常在大型语料库上进行预训练，然后针对手头的任务进行微调：BERT (Devlin et al., 2019) 使用去噪自监督预训练任务，而 GPT 工作线 使用语言建模作为其预训练任务（Radford 等人，2018；2019；Brown 等人，2020）[已有句子预测下一个词]。

Naive application of self-attention to images would require that each pixel attends to every other pixel. With quadratic cost in the number of pixels, this does not scale to realistic input sizes. Thus, to apply Transformers in the context of image processing, several approximations have been tried in the past. Parmar et al. (2018) applied the self-attention only in local neighborhoods for each query pixel instead of globally. Such local multi-head dot-product self attention blocks can completely replace convolutions (Hu et al., 2019; Ramachandran et al., 2019; Zhao et al., 2020). In a different line of work, Sparse Transformers (Child et al., 2019) employ scalable approximations to global selfattention in order to be applicable to images. An alternative way to scale attention is to apply it in blocks of varying sizes (Weissenborn et al., 2019), in the extreme case only along individual axes (Ho et al., 2019; Wang et al., 2020a). Many of these specialized attention architectures demonstrate promising results on computer vision tasks, but require complex engineering to be implemented efficiently on hardware accelerators.

将自注意力简单地应用于图像需要每个像素都关注其他每个像素。由于像素数量的二次方，这不能扩展到实际的输入大小。因此，为了在图像处理的上下文中应用 Transformer，过去曾尝试过几种近似方法。帕尔马等人。 （2018）仅在每个查询像素的局部邻域中应用自我注意，而不是全局。这种局部多头点积自注意力块可以完全替代卷积（Hu et al., 2019; Ramachandran et al., 2019; Zhao et al., 2020）。在另一项工作中，Sparse Transformers (Child et al., 2019) 对全局自注意力采用可扩展的近似值，以便适用于图像。扩展注意力的另一种方法是将其应用于不同大小的块（Weissenborn 等人，2019 年），在极端情况下仅沿单个轴应用（Ho 等人[在横轴上做自注意力，然后再纵轴上做自注意力]，2019 年；Wang 等人，2020a）。许多这些专门的注意力架构在计算机视觉任务上展示了不错的结果，但需要复杂的工程才能在硬件加速器上有效实施。

Most related to ours is the model of Cordonnier et al. (2020), which extracts patches of size $2 \times 2$ from the input image and applies full self-attention on top. This model is very similar to ViT, but our work goes further to demonstrate that large scale pre-training makes vanilla transformers competitive with (or even better than) state-of-the-art CNNs. Moreover, Cordonnier et al. (2020) use a small patch size of $2 \times 2$ pixels, which makes the model applicable only to small-resolution images, while we handle medium-resolution images as well.

与我们最相关的是 Cordonnier 等人的模型。 (2020)，它从输入图像中提取大小为 $2\times 2$ 的补丁，并在顶部应用完全自注意力。 该模型与 ViT 非常相似，但我们的工作进一步证明了大规模的预训练使标准的 Transformer 可以与（甚至优于）最先进的 CNN 竞争。 此外，Cordonnier 等人。 (2020) 使用 2 x 2 像素的小块大小，这使得该模型仅适用于小分辨率图像，然而我们也能处理中等分辨率图像。

There has also been a lot of interest in combining convolutional neural networks (CNNs) with forms of self-attention, e.g. by augmenting feature maps for image classification (Bello et al., 2019) or by further processing the output ofa CNN using self-attention, e.g. for object detection (Hu et al., 2018; Carion et al., 2020), video processing (Wang et al., 2018; Sun et al., 2019), image classification (Wu et al., 2020), unsupervised object discovery (Locatello et al., 2020), or unified text-vision tasks (Chen et al., 2020c; Lu et al., 2019; Li et al., 2019).

人们对将卷积神经网络（CNN）与各种形式的自注意相结合也很感兴趣，例如，通过增强图像分类的特征图（Bello等人，2019），或通过使用自注意进一步处理CNN的输出，例如，用于物体检测（Hu等人。2018；Carion等人，2020）、视频处理（Wang等人，2018；Sun等人，2019）、图像分类（Wu等人，2020）、无监督物体发现（Locatello等人，2020）或统一的文本视觉任务（Chen等人，2020c；Lu等人，2019；Li等人，2019）。

Another recent related model is image GPT (iGPT) (Chen et al., 2020a), which applies Transformers to image pixels after reducing image resolution and color space. The model is trained in an unsupervised fashion as a generative model, and the resulting representation can then be fine-tuned or probed linearly for classification performance, achieving a maximal accuracy of $72\%$ on ImageNet.

另一个最近的相关模型是image GPT（iGPT）（Chen等人，2020a），它在降低图像分辨率和色彩空间后对图像像素应用Transformer。该模型以无监督的方式作为生成模型进行训练，然后可以对产生的表征进行微调或对分类性能进行线性探测，在ImageNet上取得了$72\%$ 的最高准确率。

Our work adds to the increasing collection of papers that explore image recognition at larger scales than the standard ImageNet dataset. The use of additional data sources allows to achieve state-of- the-art results on standard benchmarks (Mahajan et al., 2018; Touvron et al., 2019; Xie et al., 2020). Moreover, Sun et al. (2017) study how CNN performance scales with dataset size, and Kolesnikov et al. (2020); Djolonga et al. (2020) perform an empirical exploration of CNN transfer learning from large scale datasets such as ImageNet-21k and JFT-300M. We focus on these two latter datasets as well, but train Transformers instead of ResNet-based models used in prior works.

我们的工作增加了越来越多的论文，这些论文探索了比标准 ImageNet 数据集更大规模的图像识别。 使用额外的数据源可以在标准基准上获得最先进的结果（Mahajan 等人，2018；Touvron 等人，2019；Xie 等人，2020）。 此外，孙等人。 (2017) 研究 CNN 性能如何随数据集大小扩展，以及 Kolesnikov 等人。 （2020）； Djolonga 等人。 (2020) 对 ImageNet-21k 和 JFT-300M 等大规模数据集的 CNN 迁移学习进行了实证探索。 我们也关注后两个数据集，但训练 Transformer 而不是之前工作中使用的基于 ResNet 的模型。

## 3 METHOD 实验方法

In model design we follow the original Transformer (Vaswani et al., 2017) as closely as possible. An advantage of this intentionally simple setup is that scalable NLP Transformer architectures - and their efficient implementations - can be used almost out of the box.

在模型设计中，我们尽可能地遵循原始的 Transformer (Vaswani et al., 2017)。 这种有意简单设置的优点是可扩展的 NLP Transformer 架构及其高效实现几乎可以开箱即用。

### 3.1 VISION TRANSFORMER (VIT)

<div align =center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/AN%20IMAGE%20IS%20WORTH%2016%24%5Ctimes%2416%20WORDS%3ATRANSFORMERS%20FOR%20IMAGE%20RECOGNITION%20AT%20SCALE/Fig%201.png"/></div>

Figure 1: Model overview. We split an image into fixed-size patches, linearly embed each of them, add position embeddings, and feed the resulting sequence of vectors to a standard Transformer encoder. In order to perform classification, we use the standard approach of adding an extra learnable “classification token” to the sequence. The illustration of the Transformer encoder was inspired by Vaswani et al. (2017).
图 1：模型概述。 我们将图像分割成固定大小的块，线性嵌入每个块，添加位置嵌入，并将生成的向量序列馈送到标准的 Transformer 编码器。 为了执行分类，我们使用向序列添加额外可学习的“分类标记”的标准方法。 Transformer 编码器的插图受到 Vaswani 等人的启发。 （2017）。

An overview of the model is depicted in Figure 1. The standard Transformer receives as input a 1D sequence of token embeddings. To handle 2D images, we reshape the image $\mathbf{x} \isin \mathbb{R}^{H \times W \times C}$ into a sequence of flattened 2D patches $\mathbf{x}_p \isin \mathbb{R}^{N \times (P^2 \cdot C)}$, where $(H,W)$ is the resolution of the original image, $C$ is the number of channels, $(P,P)$ is the resolution of each image patch, and $N = HW/P^2$ is the resulting number of patches, which also serves as the effective input sequence length for the Transformer. The Transformer uses constant latent vector size $D$ through all of its layers, so we flatten the patches and map to $D$ dimensions with a trainable linear projection (Eq. 1). We refer to the output of this projection as the patch embeddings.

该模型的概述如图 1 所示。标准 Transformer 接收一维词元嵌入序列作为输入。 为了处理 2D 图像，我们将图像 $\mathbf{x} \isin \mathbb{R}^{H \times W \times C}$ 重塑为一系列扁平的 2D patch $\mathbf{x}_p \isin \mathbb{R}^{N \times (P^2 \cdot C)}$，其中$(H,W)$是原始图像的分辨率，$C$是通道数，$(P,P )$ 是每个图像块的分辨率，$N = HW/P^2$ 是生成的块数，它也作为 Transformer 的有效输入序列长度。 Transformer 在其所有层中使用恒定的潜在向量大小 $D$，因此我们将补丁展平并使用可训练的线性投[全连接层]映射到 $D$ 维度（方程式 1）。 我们将此投影的输出称为patch嵌入。[D的维度为$16 \times 16 \times 3$]

Similar to BERT’s `[class]` token, we prepend a learnable embedding to the sequence of embedded patches ($\mathbf {z^0_0 = x_{class}}$), whose state at the output of the Transformer encoder $ (\mathbf{z}^0_L)$ serves as the image representation $\mathbf y$ (Eq. 4). Both during pre-training and fine-tuning, a classification head is attached to ${\mathbf z}^0_L$. The classification head is implemented by a MLP with one hidden layer at pre-training time and by a single linear layer at fine-tuning time.

与 BERT 的 `[class]` 标记类似，我们在嵌入patch序列 ($\mathbf {z^0_0 = x_{class}}$) 之前添加一个可学习的嵌入，其状态在 Transformer 编码器 $ (\ mathbf{z}^0_L)$ 用作图像表示 $\mathbf y$ (Eq. 4)。 在预训练和微调期间，分类头都附加到 ${\mathbf z}^0_L$。 分类头由 MLP 实现，在预训练时具有一个隐藏层，在微调时由单个线性层实现。

Position embeddings are added to the patch embeddings to retain positional information. We use standard learnable 1D position embeddings, since we have not observed significant performance gains from using more advanced 2D-aware position embeddings (Appendix D.4). The resulting sequence of embedding vectors serves as input to the encoder.

位置嵌入被添加到patch嵌入中以保留位置信息。 我们使用标准的可学习 1D 位置嵌入，因为我们没有观察到使用更高级的 2D 感知位置嵌入带来的显着性能提升（附录 D.4）。 生成的嵌入向量序列用作编码的输入。

The Transformer encoder (Vaswani et al., 2017) consists of alternating layers of multiheaded self-attention (MSA, see Appendix A) and MLP blocks (Eq. 2, 3). Layernorm (LN) is applied before every block, and residual connections after every block (Wang et al., 2019; Baevski & Auli, 2019).

Transformer 编码器（Vaswani 等人，2017）由多头自注意力（MSA，见附录 A）和 MLP 块（等式 2、3）的交替层组成。 在每个块之前应用 Layernorm (LN)，在每个块之后应用残差连接 (Wang et al., 2019; Baevski & Auli, 2019)。

The MLP contains two layers with a GELU non-linearity.

MLP 包含两个具有 GELU 非线性的层。

$$
\begin{align}
{\mathbf z_0} & = [\mathbf{ x_{class}};{\mathbf x}^1_p{\mathbf E};{\mathbf x}^2_p{\mathbf E};...;{\mathbf x}^N_p{\mathbf E}] + {\mathbf E}_{pos}, & {\mathbf E} \isin \mathbb{R}^{(P^2 \cdot C) \times D}, {\mathbf E}_{pos} \isin \mathbb{R}^{(N+1) \times D}\\
{\mathbf z}'_{\ell} &= {\rm MSA}({\rm LN}(\mathbf {z}_{\ell -1})) + \rm {z}_{\ell -1},&\ell = 1 ... L \\
{\mathbf z}_{\ell} &= {\rm MLP}({\rm LN}(\mathbf {z}'_{\ell})) + \mathbf{z}_{\ell -1},&\ell = 1 ... L \\
{\mathbf y} &= {\rm LN}( {\rm z}^0_L)
\end{align}
$$

**Inductive bias.** We note that Vision Transformer has much less image-specific inductive bias than CNNs. In CNNs, locality, two-dimensional neighborhood structure, and translation equivariance are baked into each layer throughout the whole model. In ViT, only MLP layers are local and transla- tionally equivariant, while the self-attention layers are global. The two-dimensional neighborhood structure is used very sparingly: in the beginning of the model by cutting the image into patches and at fine-tuning time for adjusting the position embeddings for images of different resolution (as described below). Other than that, the position embeddings at initialization time carry no information about the 2D positions of the patches and all spatial relations between the patches have to be learned from scratch.

**归纳偏置。** 我们注意到 Vision Transformer 的图像特定归纳偏置比 CNN 少得多。 在 CNN 中，局部性、二维邻域结构和平移等变性被烘焙到整个模型的每一层中。 在 ViT 中，只有 MLP 层是局部的和平移等变性的，而自注意力层是全局的。 二维邻域结构的使用非常谨慎：在模型开始时，通过将图像切割成块，并在微调时调整不同分辨率图像的位置嵌入（如下所述）。 除此之外，初始化时的位置嵌入不携带有关补丁的 2D 位置的信息，并且必须从头开始学习patch之间的所有空间关系。[空间位置信息]

**Hybrid Architecture.** As an alternative to raw image patches, the input sequence can be formed from feature maps of a CNN (LeCun et al., 1989). In this hybrid model, the patch embedding projection $\mathbf{E}$ (Eq. 1) is applied to patches extracted from a CNN feature map. As a special case, the patches can have spatial size 1x1, which means that the input sequence is obtained by simply flattening the spatial dimensions of the feature map and projecting to the Transformer dimension. The classification input embedding and position embeddings are added as described above.

**混合架构。** 作为原始图像块的替代方案，输入序列可以由 CNN 的特征图形成（LeCun 等人，1989 年）。 在这个混合模型中，patch嵌入投影 $\mathbf{E}$ (Eq. 1) 应用于从 CNN 特征图中提取的补丁。 作为一种特殊情况，patch 可以具有 1x1 的空间大小，这意味着输入序列是通过简单地将特征图的空间维度展平并投影到 Transformer 维度来获得的。 如上所述添加分类输入嵌入和位置嵌入。

### 3.2 FINE-TUNING AND HIGHER RESOLUTION 微调和更高的分辨率

Typically, we pre-train ViT on large datasets, and fine-tune to (smaller) downstream tasks. For this, we remove the pre-trained prediction head and attach a zero-initialized $D \times K$ feedforward layer, where $K$ is the number of downstream classes. It is often beneficial to fine-tune at higher resolution than pre-training (Touvron et al., 2019; Kolesnikov et al., 2020). When feeding images of higher resolution, we keep the patch size the same, which results in a larger effective sequence length. The Vision Transformer can handle arbitrary sequence lengths (up to memory constraints), however, the pre-trained position embeddings may no longer be meaningful. We therefore perform 2D interpolation of the pre-trained position embeddings, according to their location in the original image. Note that this resolution adjustment and patch extraction are the only points at which an inductive bias about the 2D structure of the images is manually injected into the Vision Transformer.

通常，我们在大型数据集上预训练 ViT，并微调到（较小的）下游任务。 为此，我们移除了预训练的预测头并附加了一个零初始化的 $D\times K$ 前馈层，其中 $K$ 是下游类的数量。 与预训练相比，以更高的分辨率进行微调通常是有益的（Touvron 等人，2019；Kolesnikov 等人，2020）。 当提供更高分辨率的图像时，我们保持patch大小相同，从而产生更大的有效序列长度。 Vision Transformer 可以处理任意序列长度（直至内存限制），但是，预训练的位置嵌入可能不再有意义。 因此，我们根据它们在原始图像中的位置对预训练的位置嵌入进行 2D 插值。 请注意，这种分辨率调整和patch提取是将有关图像 2D 结构的归纳偏置手动注入Vision Transformer的唯一点。

## 4 EXPERIMENTS 实验

We evaluate the representation learning capabilities of ResNet, Vision Transformer (ViT), and the hybrid. To understand the data requirements of each model, we pre-train on datasets of varying size and evaluate many benchmark tasks. When considering the computational cost of pre-training the model, ViT performs very favourably, attaining state of the art on most recognition benchmarks at a lower pre-training cost. Lastly, we perform a small experiment using self-supervision, and show that self-supervised ViT holds promise for the future.

我们评估了 ResNet、Vision Transformer (ViT) 和混合的表示学习能力。 为了了解每个模型的数据要求，我们对不同大小的数据集进行预训练并评估许多基准任务。 在考虑预训练模型的计算成本时，ViT 表现非常出色，以较低的预训练成本在大多数识别基准上达到了最先进的水平。 最后，我们使用自监督进行了一个小型实验，并表明自监督的 ViT 对未来充满希望。

### 4.1 SETUP

**Datasets.** To explore model scalability, we use the ILSVRC-2012 ImageNet dataset with 1k classes and 1.3M images (we refer to it as ImageNet in what follows), its superset ImageNet-21k with 21k classes and 14M images (Deng et al., 2009), and JFT (Sun et al., 2017) with 18k classes and 303M high-resolution images. We de-duplicate the pre-training datasets w.r.t. the test sets of the downstream tasks following Kolesnikov et al. (2020). We transfer the models trained on these dataset to several benchmark tasks: ImageNet on the original validation labels and the cleaned-up ReaL labels (Beyer et al., 2020), CIFAR-10/100 (Krizhevsky, 2009), Oxford-IIIT Pets (Parkhi et al., 2012), and Oxford Flowers-102 (Nilsback & Zisserman, 2008). For these datasets, pre-processing follows Kolesnikov et al. (2020).

**数据集。**为了探索模型的可扩展性，我们使用具有 1k 类和 1.3M 图像的 ILSVRC-2012 ImageNet 数据集（以下我们将其称为 ImageNet），它的超集 ImageNet-21k 具有 21k 类和 14M 图像（ Deng et al., 2009) 和 JFT (Sun et al., 2017) 具有 18k 类和 303M 高分辨率图像。 我们对预训练数据集进行去重。 Kolesnikov 等人的下游任务的测试集。 （2020 年）。 我们将在这些数据集上训练的模型转移到几个基准任务：原始验证标签上的 ImageNet 和清理后的 RealL 标签（Beyer 等人，2020）、CIFAR-10/100（Krizhevsky，2009）、Oxford-IIIT Pets (Parkhi et al., 2012) 和 Oxford Flowers-102 (Nilsback & Zisserman, 2008)。 对于这些数据集，预处理遵循 Kolesnikov 等人。 （2020 年）。

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/AN%20IMAGE%20IS%20WORTH%2016%24%5Ctimes%2416%20WORDS%3ATRANSFORMERS%20FOR%20IMAGE%20RECOGNITION%20AT%20SCALE/Table%201.png"/></div>

Table 1: Details of Vision Transformer model variants.

We also evaluate on the 19-task VTAB classification suite (Zhai et al., 2019b). VTAB evaluates low-data transfer to diverse tasks, using 1 000 training examples per task. The tasks are divided into three groups: *Natural* - tasks like the above, Pets, CIFAR, etc. *Specialized* - medical and satellite imagery, and Structured - tasks that require geometric understanding like localization.

**Model Variants.** We base ViT configurations on those used for BERT (Devlin et al., 2019), as summarized in Table 1. The “Base” and “Large” models are directly adopted from BERT and we add the larger “Huge” model. In what follows we use brief notation to indicate the model size and the input patch size: for instance, ViT-L/16 means the “Large” variant with $16 \times 16$ input patch size. Note that the Transformer’s sequence length is inversely proportional to the square of the patch size, thus models with smaller patch size are computationally more expensive.

**模型变体。** 我们将 ViT 配置基于用于 BERT 的配置（Devlin 等人，2019 年），如表 1 所示。“基本”和“大型”模型直接来自 BERT，我们添加了较大的模型 “巨大”模型。 在下文中，我们使用简短的符号来表示模型大小和输入patch大小：例如，ViT-L/16 表示具有 $16 \times 16$ 输入patch大小的“大”变体。 请注意，Transformer 的序列长度与块大小的平方成反比，因此块大小较小的模型计算成本更高。

For the baseline CNNs, we use ResNet (He et al., 2016), but replace the Batch Normalization layers (Ioffe & Szegedy, 2015) with Group Normalization (Wu & He, 2018), and used standardized convolutions (Qiao et al., 2019). These modifications improve transfer (Kolesnikov et al., 2020), and we denote the modified model “ResNet (BiT)”. For the hybrids, we feed the intermediate feature maps into ViT with patch size of one “pixel”. To experiment with different sequence lengths, we either (i) take the output of stage 4 of a regular ResNet50 or (ii) remove stage 4, place the same number of layers in stage 3 (keeping the total number of layers), and take the output of this extended stage 3. Option (ii) results in a 4x longer sequence length, and a more expensive ViT model.

**Training & Fine-tuning.** We train all models, including ResNets, using Adam (Kingma & Ba, 2015) with $\beta_1 = 0.9$, $\beta_2 = 0.999$, a batch size of 4096 and apply a high weight decay of $0.1$, which we found to be useful for transfer of all models (Appendix D.1 shows that, in contrast to common practices, Adam works slightly better than SGD for ResNets in our setting). We use a linear learning rate warmup and decay, see Appendix B.1 for details. For fine-tuning we use SGD with momentum, batch size 512, for all models, see Appendix B.1.1. For ImageNet results in Table 2, we fine-tuned at higher resolution: 512 for ViT-L/16 and 518 for ViT-H/14, and also used Polyak & Juditsky (1992) averaging with a factor of $0.9999$ (Ramachandran et al., 2019; Wang et al., 2020b).

**Metrics.** We report results on downstream datasets either through few-shot or fine-tuning accuracy. Fine-tuning accuracies capture the performance of each model after fine-tuning it on the respective dataset. Few-shot accuracies are obtained by solving a regularized least-squares regression problem that maps the (frozen) representation of a subset of training images to $\{-1, 1\}^K$ target vectors. This formulation allows us to recover the exact solution in closed form. Though we mainly focus on fine-tuning performance, we sometimes use linear few-shot accuracies for fast on-the-fly evaluation where fine-tuning would be too costly.

### 4.2 COMPARISON TO STATE OF THE ART

We first compare our largest models - ViT-H/14 and ViT-L/16 - to state-of-the-art CNNs from the literature. The first comparison point is Big Transfer (BiT) (Kolesnikov et al., 2020), which performs supervised transfer learning with large ResNets. The second is Noisy Student (Xie et al., 2020), which is a large EfficientNet trained using semi-supervised learning on ImageNet and JFT- 300M with the labels removed. Currently, Noisy Student is the state of the art on ImageNet and BiT-L on the other datasets reported here. All models were trained on TPUv3 hardware, and we report the number of TPUv3-core-days taken to pre-train each of them, that is, the number of TPU v3 cores (2 per chip) used for training multiplied by the training time in days.

Table 2 shows the results. The smaller ViT-L/16 model pre-trained on JFT-300M outperforms BiT-L (which is pre-trained on the same dataset) on all tasks, while requiring substantially less computational resources to train. The larger model, ViT-H/14, further improves the performance, especially on the more challenging datasets - ImageNet, CIFAR-100, and the VTAB suite. Interestingly, this model still took substantially less compute to pre-train than prior state of the art. However, we note that pre-training efficiency may be affected not only by the architecture choice, but also other pa-rameters, such as training schedule, optimizer, weight decay, etc. We provide a controlled study of performance vs. compute for different architectures in Section 4.4. Finally, the ViT-L/16 model pre-trained on the public ImageNet-21k dataset performs well on most datasets too, while taking fewer resources to pre-train: it could be trained using a standard cloud TPUv3 with 8 cores in approximately 30 days.

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/AN%20IMAGE%20IS%20WORTH%2016%24%5Ctimes%2416%20WORDS%3ATRANSFORMERS%20FOR%20IMAGE%20RECOGNITION%20AT%20SCALE/Table%202.png"/></div>

Table 2: Comparison with state of the art on popular image classification benchmarks. We report mean and standard deviation of the accuracies, averaged over three fine-tuning runs. Vision Transformer models pre-trained on the JFT-300M dataset outperform ResNet-based baselines on all datasets, while taking substantially less computational resources to pre-train. ViT pre-trained on the smaller public ImageNet-21k dataset performs well too. Slightly improved 88:5% result reported in Touvron et al. (2020).

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/AN%20IMAGE%20IS%20WORTH%2016%24%5Ctimes%2416%20WORDS%3ATRANSFORMERS%20FOR%20IMAGE%20RECOGNITION%20AT%20SCALE/Fig2.png"/></div>

Figure 2: Breakdown of VTAB performance in *Natural*, *Specialized*, and *Structured* task groups.

Figure 2 decomposes the VTAB tasks into their respective groups, and compares to previous SOTA methods on this benchmark: BiT, VIVI - a ResNet co-trained on ImageNet and Youtube (Tschannen et al., 2020), and S4L — supervised plus semi-supervised learning on ImageNet (Zhai et al., 2019a). ViT-H/14 outperforms BiT-R152x4, and other methods, on the *Natural* and *Structured* tasks. On the Specialized the performance of the top two models is similar.

### 4.3 PRE-TRAINING DATA REQUIREMENTS

The Vision Transformer performs well when pre-trained on a large JFT-300M dataset. With fewer inductive biases for vision than ResNets, how crucial is the dataset size? We perform two series of experiments.

First, we pre-train ViT models on datasets of increasing size: ImageNet, ImageNet-21k, and JFT- 300M. To boost the performance on the smaller datasets, we optimize three basic regularization parameters - weight decay, dropout, and label smoothing. Figure 3 shows the results after finetuning to ImageNet (results on other datasets are shown in Table 5)$^2$ . When pre-trained on the smallest dataset, ImageNet, ViT-Large models underperform compared to ViT-Base models, despite (moderate) regularization. With ImageNet-21k pre-training, their performances are similar. Only with JFT-300M, do we see the full benefit of larger models. Figure 3 also shows the performance region spanned by BiT models of different sizes. The BiT CNNs outperform ViT on ImageNet, but with the larger datasets, ViT overtakes.

>$^2$ Note that the ImageNet pre-trained models are also fine-tuned, but again on ImageNet. This is because the resolution increase during fine-tuning improves the performance.

Second, we train our models on random subsets of 9M, 30M, and 90M as well as the full JFT- 300M dataset. We do not perform additional regularization on the smaller subsets and use the same hyper-parameters for all settings. This way, we assess the intrinsic model properties, and not the effect of regularization. We do, however, use early-stopping, and report the best validation accuracy achieved during training. To save compute, we report few-shot linear accuracy instead of full fine-tuning accuracy. Figure 4 contains the results. Vision Transformers overfit more than ResNets with comparable computational cost on smaller datasets. For example, ViT-B/32 is slightly faster than ResNet50; it performs much worse on the 9M subset, but better on 90M+ subsets. The same is true for ResNet152x2 and ViT-L/16. This result reinforces the intuition that the convolutional inductive bias is useful for smaller datasets, but for larger ones, learning the relevant patterns directly from data is sufficient, even beneficial.

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/AN%20IMAGE%20IS%20WORTH%2016%24%5Ctimes%2416%20WORDS%3ATRANSFORMERS%20FOR%20IMAGE%20RECOGNITION%20AT%20SCALE/Fig%203.png"/></div>

Figure 3: Transfer to ImageNet. While large ViT models perform worse than BiT ResNets (shaded area) when pre-trained on small datasets, they shine when pre-trained on larger datasets. Similarly, larger ViT variants overtake smaller ones as the dataset grows.
图 3：迁移到 ImageNet。 虽然大型 ViT 模型在小型数据集上进行预训练时的性能比 BiT ResNets（阴影区域）差，但在大型数据集上进行预训练时它们表现出色。 同样，随着数据集的增长，较大的 ViT 变体会超过较小的变体

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/AN%20IMAGE%20IS%20WORTH%2016%24%5Ctimes%2416%20WORDS%3ATRANSFORMERS%20FOR%20IMAGE%20RECOGNITION%20AT%20SCALE/Fig%204.png"/></div>

Figure 4: Linear few-shot evaluation on ImageNet versus pre-training size. ResNets perform better with smaller pre-training datasets but plateau sooner than ViT, which performs better with larger pre-training. ViT-b is ViT-B with all hidden dimensions halved.
图 4：ImageNet 上的线性少样本[Vit当作特征提取器，而不是做fine-tune.直接将输出做一个逻辑回归]评估与预训练大小。 ResNets 在较小的预训练数据集上表现更好，但比 ViT 更快地达到稳定状态，而 ViT 在较大的预训练中表现更好。 ViT-b 是所有隐藏维度减半的 ViT-B。

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/AN%20IMAGE%20IS%20WORTH%2016%24%5Ctimes%2416%20WORDS%3ATRANSFORMERS%20FOR%20IMAGE%20RECOGNITION%20AT%20SCALE/Fig%205.png"/></div>

Figure 5: Performance versus pre-training compute for different architectures: Vision Transformers,ResNets, and hybrids. Vision Transformers generally outperform ResNets with the same computational budget. Hybrids improve upon pure Transformers for smaller model sizes, but the gap vanishes for larger models.
图 5：不同架构的性能与预训练计算：视觉转换器、ResNet 和混合。 在计算预算相同的情况下，Vision Transformers 通常优于 ResNet。 对于较小的模型尺寸，Hybrid 改进了纯 Transformer，但对于较大的模型，差距消失了。

Overall, the few-shot results on ImageNet (Figure 4), as well as the low-data results on VTAB (Table 2) seem promising for very low-data transfer. Further analysis of few-shot properties of ViT is an exciting direction of future work.

### 4.4 SCALING S TUDY

We perform a controlled scaling study of different models by evaluating transfer performance from JFT-300M. In this setting data size does not bottleneck the models’ performances, and we assess performance versus pre-training cost of each model. The model set includes: 7 ResNets, R50x1, R50x2 R101x1, R152x1, R152x2, pre-trained for 7 epochs, plus R152x2 and R200x3 pre-trained for 14 epochs; 6 Vision Transformers, ViT-B/32, B/16, L/32, L/16, pre-trained for 7 epochs, plus L/16 and H/14 pre-trained for 14 epochs; and 5 hybrids, R50+ViT-B/32, B/16, L/32, L/16 pretrained for 7 epochs, plus R50+ViT-L/16 pre-trained for 14 epochs (for hybrids, the number at the end of the model name stands not for the patch size, but for the total dowsampling ratio in the ResNet backbone).

Figure 5 contains the transfer performance versus total pre-training compute (see Appendix D.5 for details on computational costs). Detailed results per model are provided in Table 6 in the Appendix. A few patterns can be observed. First, Vision Transformers dominate ResNets on the performance/compute trade-off. ViT uses approximately $2 — 4 \times$ less compute to attain the same performance (average over 5 datasets). Second, hybrids slightly outperform ViT at small compu-tational budgets, but the difference vanishes for larger models. This result is somewhat surprising, since one might expect convolutional local feature processing to assist ViT at any size. Third, Vision Transformers appear not to saturate within the range tried, motivating future scaling efforts.

### 4.5 INSPECTING VISION TRANSFORMER 研究VISION TRANSFORMER

To begin to understand how the Vision Transformer processes image data, we analyze its internal representations. The first layer of the Vision Transformer linearly projects the flattened patches into a lower-dimensional space (Eq. 1). Figure 7 (left) shows the top principal components of the the learned embedding filters. The components resemble plausible basis functions for a low-dimensional representation of the fine structure within each patch.

为了开始了解 Vision Transformer 如何处理图像数据，我们分析了它的内部表示。 Vision Transformer 的第一层将展平的补丁线性投影到低维空间（方程式 1）[patch经过MLP后的结果，即E]。 图 7（左）显示了学习的嵌入过滤器的顶部主成分。 这些组件类似于合理的基函数，用于每个patch内精细结构的低维表示。

After the projection, a learned position embedding is added to the patch representations. Figure 7 (center) shows that the model learns to encode distance within the image in the similarity of position embeddings, i.e. closer patches tend to have more similar position embeddings. Further, the row-column structure appears; patches in the same row/column have similar embeddings. Finally, a sinusoidal structure is sometimes apparent for larger grids (Appendix D). That the position embeddings learn to represent 2D image topology explains why hand-crafted 2D-aware embedding variants do not yield improvements (Appendix D.4).

在投影之后，将学习的位置嵌入添加到patch表示中。 图 7（中）显示该模型学习在位置嵌入的相似性中对图像内的距离进行编码，即更接近的块往往具有更相似的位置嵌入。 进一步，出现了行列结构； 同一行/列中的patch具有相似的嵌入。 最后，对于较大的网格，有时会出现正弦结构（附录 D）。 位置嵌入学习表示 2D 图像拓扑解释了为什么手工制作的 2D 感知嵌入变体不会产生改进（附录 D.4）。

We find that some heads attend to most of the image already in the lowest layers, showing that the ability to integrate information globally is indeed used by the model. Other attention heads have consistently small attention distances in the low layers. This highly localized attention is less pronounced in hybrid models that apply a ResNet before the Transformer (Figure 7, right), suggesting that it may serve a similar function as early convolutional layers in CNNs. Further, the attention distance increases with network depth. Globally, we find that the model attends to image regions that are semantically relevant for classification (Figure 6).

我们发现，有些头在最低层已经注意到了大部分的图像，这表明全局信息整合的能力确实被模型所利用。其他注意头在低层的注意距离一直很小。这种高度本地化的注意力在Transformer之前应用ResNet的混合模型中不太明显（图7，右），表明它可能起到与CNN中早期卷积层类似的功能。此外，注意力距离随着网络深度的增加而增加。总的来说，我们发现该模型关注的是与分类有语义关系的图像区域（图6）。

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/AN%20IMAGE%20IS%20WORTH%2016%24%5Ctimes%2416%20WORDS%3ATRANSFORMERS%20FOR%20IMAGE%20RECOGNITION%20AT%20SCALE/Fig%206.png"/></div>

Figure 6: Representative examples of attention from the output token to the input space. See Appendix D.7 for details

### 4.6 SELF-SUPERVISION 自监督

Transformers show impressive performance on NLP tasks. However, much of their success stems not only from their excellent scalability but also from large scale self-supervised pre-training (Devlin et al., 2019; Radford et al., 2018). We also perform a preliminary exploration on masked patch prediction for self-supervision, mimicking the masked language modeling task used in BERT. With self-supervised pre-training, our smaller ViT-B/16 model achieves 79.9% accuracy on ImageNet, a significant improvement of 2% to training from scratch, but still 4% behind supervised pre-training. Appendix B.1.2 contains further details. We leave exploration of contrastive pre-training (Chen et al., 2020b; He et al., 2020; Bachman et al., 2019; Henaff et al., 2020) to future work.

Transformers 在 NLP 任务上表现出令人印象深刻的表现。 然而，他们的大部分成功不仅源于其出色的可扩展性，还源于大规模的自我监督预训练（Devlin et al., 2019; Radford et al., 2018）。 我们还对用于自我监督的掩码patch预测进行了初步探索，模仿了 BERT 中使用的掩码语言建模任务。 通过自我监督预训练，我们较小的 ViT-B/16 模型在 ImageNet 上实现了 79.9% 的准确率，比从头开始训练显着提高了 2%，但仍落后于监督预训练 4%。 附录 B.1.2 包含更多细节。 我们将对比预训练的探索（Chen 等人，2020b；He 等人，2020；Bachman 等人，2019；Henaff 等人，2020）留给未来的工作。

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/AN%20IMAGE%20IS%20WORTH%2016%24%5Ctimes%2416%20WORDS%3ATRANSFORMERS%20FOR%20IMAGE%20RECOGNITION%20AT%20SCALE/Fig%207.png"/></div>

Figure 7: **Left**: Filters of the initial linear embedding of RGB values of ViT-L/32. **Center**: Similarity of position embeddings of ViT-L/32. Tiles show the cosine similarity between the position embedding of the patch with the indicated row and column and the position embeddings of all other patches. **Right**: Size of attended area by head and network depth. Each dot shows the mean attention distance across images for one of 16 heads at one layer. See Appendix D.7 for details

## 5 CONCLUSION 结论

We have explored the direct application of Transformers to image recognition. Unlike prior works using self-attention in computer vision, we do not introduce image-specific inductive biases into the architecture apart from the initial patch extraction step. Instead, we interpret an image as a sequence of patches and process it by a standard Transformer encoder as used in NLP. This simple, yet scalable, strategy works surprisingly well when coupled with pre-training on large datasets. Thus, Vision Transformer matches or exceeds the state of the art on many image classification datasets, whilst being relatively cheap to pre-train.

我们已经探索了Transformer在图像识别中的直接应用。 与在计算机视觉中使用自注意力的先前工作不同，除了初始patch[图像块]提取步骤之外，我们不会将特定于图像的归纳偏置引入架构中。 相反，我们将图像解释为一个序列的patch，并通过 NLP 中使用的标准 Transformer 编码器对其进行处理。 这种简单但可扩展的策略在与大型数据集的预训练相结合时效果出奇地好。 因此，Vision Transformer 在许多图像分类数据集上匹配或超过了现有技术，同时预训练相对便宜。

While these initial results are encouraging, many challenges remain. One is to apply ViT to other computer vision tasks, such as detection and segmentation. Our results, coupled with those in Carion et al. (2020), indicate the promise of this approach. Another challenge is to continue exploring self-supervised pre-training methods. Our initial experiments show improvement from self-supervised pre-training, but there is still large gap between self-supervised and large-scale supervised pretraining. Finally, further scaling of ViT would likely lead to improved performance.

尽管这些初步结果令人鼓舞，但仍然存在许多挑战。 一种是将 ViT 应用于其他计算机视觉任务，例如检测[新模型Vit-FRCNN]和分割[新模型SERT]。 我们的结果，再加上 Carion 等人的结果。[DETR]（2020），表明这种方法的效果。 另一个挑战是继续探索自我监督的预训练方法。 我们最初的实验显示自监督预训练的改进，但自监督和大规模监督预训练之间仍然存在很大差距。 最后，进一步扩展 ViT 可能会提高性能。 [扩展性 swin Transformer][后续模型 ViT-G]

## ACKNOWLEDGEMENTS 致谢

The work was performed in Berlin, Zurich, and Amsterdam. We thank many colleagues at Google for their help, in particular Andreas Steiner for crucial help with the infrastructure and the opensource release of the code; Joan Puigcerver and Maxim Neumann for help with the large-scale training infrastructure; Dmitry Lepikhin, Aravindh Mahendran, Daniel Keysers, Mario LuCiC, Noam Shazeer, Ashish Vaswani, and Colin Raffel for useful discussions.

## REFERENCES

Samira Abnar and Willem Zuidema. Quantifying attention flow in transformers. In ACL, 2020.

Philip Bachman, R Devon Hjelm, and William Buchwalter. Learning representations by maximizing mutual information across views. In NeurIPS, 2019.

Alexei Baevski and Michael Auli. Adaptive input representations for neural language modeling. In ICLR, 2019.

I. Bello, B. Zoph, Q. Le, A. Vaswani, and J. Shlens. Attention augmented convolutional networks. In ICCV, 2019.

Lucas Beyer, Olivier J. Henaff, Alexander Kolesnikov, Xiaohua Zhai, and Aaron van den Oord. Are we done with imagenet? arXiv, 2020.

Tom B Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. arXiv, 2020.

Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. In ECCV, 2020.

Mark Chen, Alec Radford, Rewon Child, Jeff Wu, and Heewoo Jun. Generative pretraining from pixels. In ICML, 2020a.

Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey E. Hinton. A simple framework for contrastive learning of visual representations. In ICML, 2020b.

Yen-Chun Chen, Linjie Li, Licheng Yu, Ahmed El Kholy, Faisal Ahmed, Zhe Gan, Yu Cheng, and Jingjing Liu. UNITER: UNiversal Image-TExt Representation Learning. In ECCV, 2020c.

Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating long sequences with sparse transformers. arXiv, 2019.

Jean-Baptiste Cordonnier, Andreas Loukas, and Martin Jaggi. On the relationship between self-attention and convolutional layers. In ICLR, 2020.

J. Deng, W. Dong, R. Socher, L. Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, 2009.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In NAACL, 2019.

Josip Djolonga, Jessica Yung, Michael Tschannen, Rob Romijnders, Lucas Beyer, Alexander Kolesnikov, Joan Puigcerver, Matthias Minderer, Alexander D’Amour, Dan Moldovan, Sylvan Gelly, Neil Houlsby, Xiaohua Zhai, and Mario Lucic. On robustness and transferability of convo-lutional neural networks. arXiv, 2020.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recog-nition. In CVPR, 2016.

Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In CVPR, 2020.

Jonathan Ho, Nal Kalchbrenner, Dirk Weissenborn, and Tim Salimans. Axial attention in multidi-mensional transformers. arXiv, 2019.

Han Hu, Jiayuan Gu, Zheng Zhang, Jifeng Dai, and Yichen Wei. Relation networks for object detection. In CVPR, 2018.

Han Hu, Zheng Zhang, Zhenda Xie, and Stephen Lin. Local relation networks for image recognition. In ICCV, 2019.

Zilong Huang, Xinggang `Wang, Yunchao Wei, Lichao Huang, Humphrey Shi, Wenyu Liu, and Thomas S. Huang. Ccnet: Criss-cross attention for semantic segmentation. In ICCV, 2020.

Olivier J. Henaff, Aravind Srinivas, Jeffrey De Fauw, Ali Razavi, Carl Doersch, S. M. Ali Eslami, and Aaron van den Oord. Data-efficient image recognition with contrastive predictive coding. In ICML, 2020.

Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. 2015.

Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR, 2015.

Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Joan Puigcerver, Jessica Yung, Sylvain Gelly, and Neil Houlsby. Big transfer (BiT): General visual representation learning. In ECCV, 2020.

Alex Krizhevsky. Learning multiple layers of features from tiny images. Technical report, 2009.

Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. Imagenet classification with deep convo-lutional neural networks. In NIPS, 2012.

Y. LeCun, B. Boser, J. Denker, D. Henderson, R. Howard, W. Hubbard, and L. Jackel. Backpropa- gation applied to handwritten zip code recognition. Neural Computation, 1:541-551, 1989.

Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. Gshard: Scaling giant models with conditional computation and automatic sharding. arXiv, 2020.

Liunian Harold Li, Mark Yatskar, Da Yin, Cho-Jui Hsieh, and Kai-Wei Chang. VisualBERT: A Simple and Performant Baseline for Vision and Language. In Arxiv, 2019.

Francesco Locatello, Dirk Weissenborn, Thomas Unterthiner, Aravindh Mahendran, Georg Heigold, Jakob Uszkoreit, Alexey Dosovitskiy, and Thomas Kipf. Object-centric learning with slot attention. arXiv, 2020.

Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. ViLBERT: Pretraining Task-Agnostic Visi- olinguistic Representations for Vision-and-Language Tasks. In NeurIPS. 2019.

Dhruv Mahajan, Ross Girshick, Vignesh Ramanathan, Kaiming He, Manohar Paluri, Yixuan Li, Ashwin Bharambe, and Laurens van der Maaten. Exploring the limits of weakly supervised pretraining. In ECCV, 2018.

M. Nilsback and A. Zisserman. Automated flower classification over a large number of classes. In ICVGIP, 2008.

Omkar M. Parkhi, Andrea Vedaldi, Andrew Zisserman, and C. V. Jawahar. Cats and dogs. In CVPR, 2012.

Niki Parmar, Ashish Vaswani, Jakob Uszkoreit, Lukasz Kaiser, Noam Shazeer, Alexander Ku, and Dustin Tran. Image transformer. In ICML, 2018.

B. T. Polyak and A. B. Juditsky. Acceleration of stochastic approximation by averaging. SIAM Journal on Control and Optimization, 30(4):838-855, 1992. doi: 10.1137/0330046. URL https://doi.org/10.1137/0330046.

Siyuan Qiao, Huiyu Wang, Chenxi Liu, Wei Shen, and Alan Yuille. Weight standardization. arXiv preprint arXiv:1903.10520, 2019.

Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language under-standing with unsupervised learning. Technical Report, 2018.

Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. Technical Report, 2019.

Prajit Ramachandran, Niki Parmar, Ashish Vaswani, Irwan Bello, Anselm Levskaya, and Jon Shlens. Stand-alone self-attention in vision models. In NeurIPS, 2019.

Chen Sun, Abhinav Shrivastava, Saurabh Singh, and Abhinav Gupta. Revisiting unreasonable ef-fectiveness of data in deep learning era. In ICCV, 2017.

Chen Sun, Austin Myers, Carl Vondrick, Kevin Murphy, and Cordelia Schmid. Videobert: A joint model for video and language representation learning. In ICCV, 2019.

Hugo Touvron, Andrea Vedaldi, Matthijs Douze, and Herve Jegou. Fixing the train-test resolution discrepancy. In NeurIPS. 2019.

Hugo Touvron, Andrea Vedaldi, Matthijs Douze, and Herve Jegou. Fixing the train-test resolution discrepancy: Fixefficientnet. arXiv preprint arXiv:2003.08237, 2020.

Michael Tschannen, Josip Djolonga, Marvin Ritter, Aravindh Mahendran, Neil Houlsby, Sylvain Gelly, and Mario Lucic. Self-supervised learning of video-induced visual invariances. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2020.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Eukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NIPS, 2017.

Huiyu Wang, Yukun Zhu, Bradley Green, Hartwig Adam, Alan Yuille, and Liang-Chieh Chen. Axial-deeplab: Stand-alone axial-attention for panoptic segmentation. In ECCV, 2020a.

Huiyu Wang, Yukun Zhu, Bradley Green, Hartwig Adam, Alan Yuille, and Liang-Chieh Chen. Axial-deeplab: Stand-alone axial-attention for panoptic segmentation. arXiv preprint arXiv:2003.07853, 2020b.

Qiang Wang, Bei Li, Tong Xiao, Jingbo Zhu, Changliang Li, Derek F. Wong, and Lidia S. Chao. Learning deep transformer models for machine translation. In ACL, 2019.

Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He. Non-local neural networks. In CVPR, 2018.

Dirk Weissenborn, Oscar Tackstrtom, and Jakob Uszkoreit. Scaling autoregressive video models. In ICLR, 2019.

Bichen Wu, Chenfeng Xu, Xiaoliang Dai, Alvin Wan, Peizhao Zhang, Masayoshi Tomizuka, Kurt Keutzer, and Peter Vajda. Visual transformers: Token-based image representation and processing for computer vision. arxiv, 2020.

Yuxin Wu and Kaiming He. Group normalization. In ECCV, 2018.

Qizhe Xie, Minh-Thang Luong, Eduard Hovy, and Quoc V. Le. Self-training with noisy student improves imagenet classification. In CVPR, 2020.

Xiaohua Zhai, Avital Oliver, Alexander Kolesnikov, and Lucas Beyer. S4L: Self-Supervised Semi-Supervised Learning. In ICCV, 2019a.

Xiaohua Zhai, Joan Puigcerver, Alexander Kolesnikov, Pierre Ruyssen, Carlos Riquelme, Mario Lucic, Josip Djolonga, Andre Susano Pinto, Maxim Neumann, Alexey Dosovitskiy, et al. A large-scale study of representation learning with the visual task adaptation benchmark. arXiv preprint arXiv:1910.04867, 2019b.

Hengshuang Zhao, Jiaya Jia, and Vladlen Koltun. Exploring self-attention for image recognition. In CVPR, 2020. 

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/AN%20IMAGE%20IS%20WORTH%2016%24%5Ctimes%2416%20WORDS%3ATRANSFORMERS%20FOR%20IMAGE%20RECOGNITION%20AT%20SCALE/Table%203.png"/></div>

Table 3: Hyperparameters for training. All models are trained with a batch size of 4096 and learning rate warmup of 10k steps. For ImageNet we found it beneficial to additionally apply gradient clipping at global norm 1. Training resolution is 224.

## APPENDIX

## A MULTIHEAD SELF-ATTENTION

Standard $\mathbf{qkv}$ self-attention (SA, Vaswani et al. (2017)) is a popular building block for neural archi-tectures. For each element in an input sequence ${\rm z} \isin \mathbb{R}^{N \times D}$ , we compute a weighted sum over all values $\mathbf{v}$ in the sequence. The attention weights $A_{ij}$ are based on the pairwise similarity between two elements of the sequence and their respective query $\mathbf{q}^i$ and key $\mathbf{k}^j$ representations.

$$
\begin{align}
[\mathbf{q,k,v}] &= \mathbf{zU}_{qkv}  &\mathbf{U}_{qkv} &\isin \mathbb{R}^{D \times 3D_h}, \\
A &= {\rm softmax}(\mathbf{qk}^\top/\sqrt{D_h}) &A &\isin \mathbb{R}^{N \times N} \\
SA(\mathbf{z}) &= A\mathbf{v}
\end{align}
$$

Multihead self-attention (MSA) is an extension of SA in which we run $k$ self-attention operations, called “heads”, in parallel, and project their concatenated outputs. To keep compute and number of parameters constant when changing $k, D_h$ (Eq. 5) is typically set to $D/k$.

$$
{\rm MSA}(\mathbf{z}) = [{\rm SA}_1(z);{\rm SA}_2(z);...;{\rm SA}_k(z)]\mathbf{U}_{msa} \qquad \mathbf{U}_{msa} \isin \mathbb{R}^{k \cdot D_h \times D} \tag{8}
$$

## B EXPERIMENT DETAILS

### B.1 TRAINING

Table 3 summarizes our training setups for our different models. We found strong regularization to be key when training models from scratch on ImageNet. Dropout, when used, is applied after every dense layer except for the the qkv-projections and directly after adding positional- to patch embeddings. Hybrid models are trained with the exact setup as their ViT counterparts. Finally, all training is done on resolution 224.

#### B.1.1 FINE-TUNING

We fine-tune all ViT models using SGD with a momentum of 0.9. We run a small grid search over learning rates, see learning rate ranges in Table 4. To do so, we use small sub-splits from the training set (10% for Pets and Flowers, 2% for CIFAR, 1% ImageNet) as development set and train on the remaining data. For final results we train on the entire training set and evaluate on the respective test data. For fine-tuning ResNets and hybrid models we use the exact same setup, with the only exception of ImageNet where we add another value 0:06 to the learning rate sweep. Additionally,for ResNets we also run the setup of Kolesnikov et al. (2020) and select the best results across this run and our sweep. Finally, if not mentioned otherwise, all fine-tuning experiments run at 384 resolution (running fine-tuning at different resolution than training is common practice (Kolesnikov et al., 2020)).

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/AN%20IMAGE%20IS%20WORTH%2016%24%5Ctimes%2416%20WORDS%3ATRANSFORMERS%20FOR%20IMAGE%20RECOGNITION%20AT%20SCALE/Table%204.png"/></div>
Table 4: Hyperparameters for fine-tuning. All models are fine-tuned with cosine learning rate decay, a batch size of 512, no weight decay, and grad clipping at global norm 1. If not mentioned otherwise, fine-tuning resolution is 384.

When transferring ViT models to another dataset, we remove the whole head (two linear layers) and replace it by a single, zero-initialized linear layer outputting the number of classes required by the target dataset. We found this to be a little more robust than simply re-initializing the very last layer.

For VTAB we follow the protocol in Kolesnikov et al. (2020), and use the same hyperparameter setting for all tasks. We use a learning rate of 0:01 and train for 2500 steps (Tab. 4). We chose this setting by running a small sweep over two learning rates and two schedules, and selecting the setting with the highest VTAB score on the 200-example validation sets. We follow the pre-processing used in Kolesnikov et al. (2020), except that we do not use task-specific input resolutions. Instead we find that Vision Transformer benefits most from a high resolution ($384  \times 384$) for all tasks.

#### B.1.2 SELF-SUPERVISION

We employ the masked patch prediction objective for preliminary self-supervision experiments. To do so we corrupt 50% of patch embeddings by either replacing their embeddings with a learnable `[mask]` embedding (80%), a random other patch embedding (10%) or just keeping them as is (10%). This setup is very similar to the one used for language by Devlin et al. (2019). Finally, we predict the 3-bit, mean color (i.e., 512 colors in total) of every corrupted patch using their respective patch representations.

We trained our self-supervised model for 1M steps (ca. $14$ epochs) with batch size $4096$ on JFT. We use Adam, with a base learning rate of $2 \cdot 10^{-4}$, warmup of 10k steps and cosine learning rate decay. As prediction targets for pretraining we tried the following settings: 1) predicting only the mean, 3bit color (i.e., 1 prediction of $512$ colors), 2) predicting a $4 \times 4$ downsized version of the $16 \times 16$ patch with 3bit colors in parallel (i.e., 16 predictions of $512$ colors), 3) regression on the full patch using L2 (i.e., 256 regressions on the 3 RGB channels). Surprisingly, we found that all worked quite well, though L2 was slightly worse. We report final results only for option 1) because it has shown best few-shot performance. We also experimented with $15\%$ corruption rate as used by Devlin et al. (2019) but results were also slightly worse on our few-shot metrics.

Lastly, we would like to remark that our instantiation of masked patch prediction doesn’t require such an enormous amount of pretraining nor a large dataset such as JFT in order to lead to similar performance gains on ImageNet classification. That is, we observed diminishing returns on downstream performance after 100k pretraining steps, and see similar gains when pretraining on ImageNet.

## C ADDITIONAL RESULTS

We report detailed results corresponding to the figures presented in the paper. Table 5 corresponds to Figure 3 from the paper and shows transfer performance of different ViT models pre-trained on datasets of increasing size: ImageNet, ImageNet-21k, and JFT-300M. Table 6 corresponds to Figure 5 from the paper and shows the transfer performance of ViT, ResNet, and hybrid models of varying size, as well as the estimated computational cost of their pre-training.

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/AN%20IMAGE%20IS%20WORTH%2016%24%5Ctimes%2416%20WORDS%3ATRANSFORMERS%20FOR%20IMAGE%20RECOGNITION%20AT%20SCALE/Table5.png"/></div>

Table 5: Top1 accuracy (in %) of Vision Transformer on various datasets when pre-trained on Im- ageNet, ImageNet-21k or JFT300M. These values correspond to Figure 3 in the main text. Models are fine-tuned at 384 resolution. Note that the ImageNet results are computed without additional techniques (Polyak averaging and 512 resolution images) used to achieve results in Table 2.

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/AN%20IMAGE%20IS%20WORTH%2016%24%5Ctimes%2416%20WORDS%3ATRANSFORMERS%20FOR%20IMAGE%20RECOGNITION%20AT%20SCALE/Table6.png"/></div>

Table 6: Detailed results of model scaling experiments. These correspond to .

## D ADDITIONAL ANALYSES

### D.1 SGD VS. ADAM FOR RESNETS

ResNets are typically trained with SGD and our use of Adam as optimizer is quite unconventional. Here we show the experiments that motivated this choice. Namely, we compare the fine-tuning performance of two ResNets - 50x1 and 152x2 - pre-trained on JFT with SGD and Adam. For SGD, we use the hyperparameters recommended by Kolesnikov et al. (2020). Results are presented in Table 7. Adam pre-training outperforms SGD pre-training on most datasets and on average. This justifies the choice of Adam as the optimizer used to pre-train ResNets on JFT. Note that the absolute numbers are lower than those reported by Kolesnikov et al. (2020), since we pre-train only for 7 epochs, not 30.

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/AN%20IMAGE%20IS%20WORTH%2016%24%5Ctimes%2416%20WORDS%3ATRANSFORMERS%20FOR%20IMAGE%20RECOGNITION%20AT%20SCALE/Table7.png"/></div>
Table 7: Fine-tuning ResNet models pre-trained with Adam and SGD.

### D.2 TRANSFORMER SHAPE

We ran ablations on scaling different dimensions of the Transformer architecture to find out which are best suited for scaling to very large models. Figure 8 shows 5-shot performance on ImageNet for different configurations. All configurations are based on a ViT model with 8 layers, $D = 1024$, $D_{MLP} = 2048$ and a patch size of 32, the intersection of all lines. We can see that scaling the depth results in the biggest improvements which are clearly visible up until 64 layers. However, diminishing returns are already visible after 16 layers. Interestingly, scaling the width of the network seems to result in the smallest changes. Decreasing the patch size and thus increasing the effective sequence length shows surprisingly robust improvements without introducing parameters. These findings suggest that compute might be a better predictor of performance than the number of parameters, and that scaling should emphasize depth over width if any. Overall, we find that scaling all dimensions proportionally results in robust improvements.

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/AN%20IMAGE%20IS%20WORTH%2016%24%5Ctimes%2416%20WORDS%3ATRANSFORMERS%20FOR%20IMAGE%20RECOGNITION%20AT%20SCALE/Fig%208.png"/></div>

Figure 8: Scaling different model dimensions of the Vision Transformer.

### D.3 HEAD TYPE AND CLASS TOKEN

In order to stay as close as possible to the original Transformer model, we made use ofan additional `[class]` token, which is taken as image representation. The output of this token is then transformed into a class prediction via a small multi-layer perceptron (MLP) with tanh as non-linearity in the single hidden layer.

为了尽可能接近原始 Transformer 模型，我们使用了一个额外的 `[class]` 标记，作为图像表示。 然后通过一个小型多层感知器 (MLP) 将这个令牌的输出转换为一个类别预测，其中 tanh 作为单个隐藏层中的非线性。

This design is inherited from the Transformer model for text, and we use it throughout the main paper. An initial attempt at using only image-patch embeddings, globally average-pooling (GAP) them, followed by a linear classifier—just like ResNet’s final feature map—performed very poorly. However, we found that this is neither due to the extra token, nor to the GAP operation. Instead,the difference in performance is fully explained by the requirement for a different learning-rate, see Figure 9.

这种设计继承自文本的 Transformer 模型，我们在整篇论文中都使用它。 最初尝试仅使用图像patch嵌入、全局平均池化（GAP）它们，然后是线性分类器——就像 ResNet 的最终特征图一样——表现非常糟糕。 但是，我们发现这既不是由于额外的词元，也不是由于 GAP 操作。 相反，性能差异完全可以通过对不同学习率的要求来解释，参见图 9。

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/AN%20IMAGE%20IS%20WORTH%2016%24%5Ctimes%2416%20WORDS%3ATRANSFORMERS%20FOR%20IMAGE%20RECOGNITION%20AT%20SCALE/Fig%209.png"/></div>

Figure 9: Comparison of class-token and global average pooling classifiers. Both work similarly well, but require different learning-rates.

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/AN%20IMAGE%20IS%20WORTH%2016%24%5Ctimes%2416%20WORDS%3ATRANSFORMERS%20FOR%20IMAGE%20RECOGNITION%20AT%20SCALE/Table8.png"/></div>

Table 8: Results of the ablation study on positional embeddings with ViT-B/16 model evaluated on ImageNet 5-shot linear.

### D.4 POSITIONAL EMBEDDING 位置

We ran ablations on different ways of encoding spatial information using positional embedding. We tried the following cases:

我们使用位置嵌入对编码空间信息的不同方式进行了消融。 我们尝试了以下案例：

- Providing no positional information: Considering the inputs as a *bag of patches*.
- 不提供位置信息：将输入视为*补丁包*。
- 1-dimensional positional embedding: Considering the inputs as a sequence of patches in the raster order (default across all other experiments in this paper).
- 一维位置嵌入：将输入视为光栅顺序的补丁序列（本文中所有其他实验的默认值）。
- 2-dimensional positional embedding: Considering the inputs as a grid of patches in two dimensions. In this case, two sets of embeddings are learned, each for one of the axes, $X$-embedding, and $Y$-embedding, each with size $D/2$. Then, based on the coordinate on the path in the input, we concatenate the $X$ and $Y$ embedding to get the final positional embedding for that patch.
- 二维位置嵌入：将输入视为二维的patch网格。 在这种情况下，学习了两组嵌入，每组用于一个轴，$X$-embedding 和 $Y$-embedding，每组的大小为 $D/2$。 然后，基于输入中路径上的坐标，我们连接 $X$ 和 $Y$ 嵌入以获得该补丁的最终位置嵌入。
- Relative positional embeddings: Considering the relative distance between patches to encode the spatial information as instead of their absolute position. To do so, we use 1dimensional Relative Attention, in which we define the relative distance all possible pairs of patches. Thus, for every given pair (one as query, and the other as key/value in the attention mechanism), we have an offset $p_q$ —— $p_k$, where each offset is associated with an embedding. Then, we simply run extra attention, where we use the original query (the content of query), but use relative positional embeddings as keys. We then use the logits from the relative attention as a bias term and add it to the logits of the main attention (content-based attention) before applying the softmax.
- 相对位置嵌入：考虑patch之间的相对距离来编码空间信息，而不是它们的绝对位置。 为此，我们使用一维相对注意力，其中我们定义了所有可能的补丁对的相对距离。 因此，对于每个给定的对（一个作为查询，另一个作为注意力机制中的键/值），我们有一个偏移量 $p_q$ —— $p_k$，其中每个偏移量都与一个嵌入相关联。 然后，我们简单地运行额外的注意力，我们使用原始查询（查询的内容），但使用相对位置嵌入作为键。 然后，我们使用相对注意力的 logits 作为偏置项，并在应用 softmax 之前将其添加到主要注意力（基于内容的注意力）的 logits。

In addition to different ways of encoding spatial information, we also tried different ways of in-corporating this information in our model. For the 1-dimensional and 2-dimensional positional embeddings, we tried three different cases: (1) add positional embeddings to the inputs right after the stem of them model and before feeding the inputs to the Transformer encoder (default across all other experiments in this paper); (2) learn and add positional embeddings to the inputs at the beginning of each layer; (3) add a learned positional embeddings to the inputs at the beginning of each layer (shared between layers).

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/AN%20IMAGE%20IS%20WORTH%2016%24%5Ctimes%2416%20WORDS%3ATRANSFORMERS%20FOR%20IMAGE%20RECOGNITION%20AT%20SCALE/Fig%2010.png"/></div>

Figure 10: Position embeddings of models trained with different hyperparameters.

Table 8 summarizes the results from this ablation study on a ViT-B/16 model. As we can see, while there is a large gap between the performances of the model with no positional embedding and models with positional embedding, there is little to no difference between different ways of encoding positional information. We speculate that since our Transformer encoder operates on patch-level inputs, as opposed to pixel-level, the differences in how to encode spatial information is less important. More precisely, in patch-level inputs, the spatial dimensions are much smaller than the original pixel-level inputs, e.g., $14  \times 14$ as opposed to $224 \times 224$, and learning to represent the spatial relations in this resolution is equally easy for these different positional encoding strategies. Even so, the specific pattern of position embedding similarity learned by the network depends on the training hyperparameters (Figure 10).

表 8 总结了对 ViT-B/16 模型的消融研究的结果。 正如我们所看到的，虽然没有位置嵌入的模型和有位置嵌入的模型的性能之间存在很大差距，但编码位置信息的不同方式之间几乎没有差异。 我们推测，由于我们的 Transformer 编码器在patch级输入上运行，而不是像素级，因此如何编码空间信息的差异不太重要。 更准确地说，在patch级输入中，空间维度比原始像素级输入要小得多，例如，$14\times 14$ 而不是 $224\times 224$，学习在这个分辨率下表示空间关系是 对于这些不同的位置编码策略同样容易。 即便如此，网络学习到的位置嵌入相似性的具体模式取决于训练超参数（图 10）。

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/AN%20IMAGE%20IS%20WORTH%2016%24%5Ctimes%2416%20WORDS%3ATRANSFORMERS%20FOR%20IMAGE%20RECOGNITION%20AT%20SCALE/Fig%2011.png"/></div>

Figure 11: Size of attended area by head and network depth. Attention distance was computed for 128 example images by averaging the distance between the query pixel and all other pixels, weighted by the attention weight. Each dot shows the mean attention distance across images for one of 16 heads at one layer. Image width is 224 pixels.

### D.5 EMPIRICAL COMPUTATIONAL COSTS

We are also interested in real-world speed of the architectures on our hardware, which is not always well predicted by theoretical FLOPs due to details like lane widths and cache sizes. For this purpose,

we perform timing of inference speed for the main models of interest, on a TPUv3 accelerator; the difference between inference and backprop speed is a constant model-independent factor.

Figure 12 (left) shows how many images one core can handle per second, across various input sizes. Every single point refers to the peak performance measured across a wide range of batch-sizes. As can be seen, the theoretical bi-quadratic scaling of ViT with image size only barely starts happening for the largest models at the largest resolutions.

Another quantity of interest is the largest batch-size each model can fit onto a core, larger being better for scaling to large datasets. Figure 12 (right) shows this quantity for the same set of models. This shows that large ViT models have a clear advantage in terms of memory-efficiency over ResNet models.

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/AN%20IMAGE%20IS%20WORTH%2016%24%5Ctimes%2416%20WORDS%3ATRANSFORMERS%20FOR%20IMAGE%20RECOGNITION%20AT%20SCALE/Fig%2012.png"/></div>

Figure 12: **Left**: Real wall-clock timings of various architectures across input sizes. ViT models have speed comparable to similar ResNets. **Right**: Largest per-core batch-size fitting on device with various architectures across input sizes. ViT models are clearly more memory-efficient.

### D.6 AXIAL ATTENTION

Axial Attention (Huang et al., 2020; Ho et al., 2019) is a simple, yet effective technique to run self-attention on large inputs that are organized as multidimensional tensors. The general idea of axial attention is to perform multiple attention operations, each along a single axis of the input tensor, instead of applying 1-dimensional attention to the flattened version of the input. In axial attention, each attention mixes information along a particular axis, while keeping information along the other axes independent. Along this line, Wang et al. (2020b) proposed the AxialResNet model in which all the convolutions with kernel size $3 \times 3$ in a ResNet50 are replaced by axial self-attention, i.e. a row and column attention, augmented by relative positional encoding. We have implemented AxialResNet as a baseline model$^3$.

>$^3$ Our implementation is based on the open-sourced PyTorch implementation in https://github.com/csrhddlam/axial-deeplab. In our experiments, we reproduced the scores reported in (Wang et al.,2020b) in terms of accuracy, however, our implementation, similar to the open-source implementation, is very slow on TPUs. Therefore, we were not able to use it for extensive large-scale experiments. These may be unlocked by a carefully optimized implementation.

Moreover, we have modified ViT to process inputs in the 2-dimensional shape, instead of a 1-dimensional sequence of patches, and incorporate Axial Transformer blocks, in which instead of a self-attention followed by an MLP, we have a a row-self-attention plus an MLP followed by a column-self-attention plus an MLP.

Figure 13, present the performance of Axial ResNet, Axial-ViT-B/32 and Axial-ViT-B/16 on Ima- geNet 5shot linear, when pretrained on JFT dataset, verses the pretraining compute, both in terms of number of FLOPs and inference time (example per seconds). As we can see, both Axial-ViT-B/32 and Axial-ViT-B/16 do better than their ViT-B counterpart in terms of performance, but it comes at the cost of more compute. This is because in Axial-ViT models, each Transformer block with global self-attention is replaced by two Axial Transformer blocks, one with row and one with column self-attention and although the sequence length that self-attention operates on is smaller in axial case, there is a extra MLP per Axial-ViT block. For the AxialResNet, although it looks reasonable in terms of accuracy/compute trade-off (Figure 13, left), the naive implementation is extremely slow on TPUs (Figure 13, right).

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/AN%20IMAGE%20IS%20WORTH%2016%24%5Ctimes%2416%20WORDS%3ATRANSFORMERS%20FOR%20IMAGE%20RECOGNITION%20AT%20SCALE/Fig%2013.png"/></div>

Figure 13: Performance of Axial-Attention based models, in terms of top-1 accuracy on ImageNet 5-shot linear, versus their speed in terms of number of FLOPs (**left**) and inference time (**left**).

### D.7 ATTENTION DISTANCE

To understand how ViT uses self-attention to integrate information across the image, we analyzed the average distance spanned by attention weights at different layers (Figure 11). This “attention distance” is analogous to receptive field size in CNNs. Average attention distance is highly variable across heads in lower layers, with some heads attending to much of the image, while others attend to small regions at or near the query location. As depth increases, attention distance increases for all heads. In the second half of the network, most heads attend widely across tokens.

### D.8 ATTENTION MAPS

To compute maps of the attention from the output token to the input space (Figures 6 and 14), we used Attention Rollout (Abnar & Zuidema, 2020). Briefly, we averaged attention weights of ViT- L/16 across all heads and then recursively multiplied the weight matrices of all layers. This accounts for the mixing of attention across tokens through all layers.

### D.9 OBJECTNET RESULTS

We also evaluate our flagship ViT-H/14 model on the ObjectNet benchmark following the evaluation setup in Kolesnikov et al. (2020), resulting in 82.1% top-5 accuracy and 61.7% top-1 accuracy.

### D.10 VTAB BREAKDOWN

Table 9 shows the scores attained on each of the VTAB-1k tasks. 

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/AN%20IMAGE%20IS%20WORTH%2016%24%5Ctimes%2416%20WORDS%3ATRANSFORMERS%20FOR%20IMAGE%20RECOGNITION%20AT%20SCALE/AN%20IMAGE%20IS%20WORTH%2016X16%20WORDS.png"/></div>

Figure 14: Further example attention maps as in Figure 6 (random selection). 

<div align = center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/AN%20IMAGE%20IS%20WORTH%2016%24%5Ctimes%2416%20WORDS%3ATRANSFORMERS%20FOR%20IMAGE%20RECOGNITION%20AT%20SCALE/Table%209.png"/></div>