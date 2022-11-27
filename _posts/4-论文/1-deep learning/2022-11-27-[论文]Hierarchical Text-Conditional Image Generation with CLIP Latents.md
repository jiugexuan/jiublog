---
title: 【论文】Hierarchical Text-Conditional Image Generation with CLIP Latents(DELIE 2) 使用 CLIP 特征的层级式的文本条件图像生成
date: 2022-11-27 07:00:00 +/-0800
categories: [论文,深度学习]
tags: [深度学习,论文]     # TAG names should always be lowercase 标记名称应始终为小写
math: true
mermaid: true
---


|Aditya Ramesh*| Prafulla Dhariwal* |Alex Nichol*| Casey Chu *| Mark Chen
| :---: |:---: | :---: | :---: | :---: | :---: |
| OpenAI | OpenAI | OpenAI| OpenAI | OpenAI |
|aramesh@openai.com |prafulla@openai.com | alex@openai.com | casey@openai.com | mark@openai.com |

>∗Equal contribution

```mermaid
graph TD
subgraph 发展趋势 
1[DALIE 01/21]-->2[CogView 05/21]-->3[NiiWA 11/21]-->4[GLIDE 12/21]
-->5[ERNZE ViLG 12/21]-->6[DALIE 2 04/12]-->7[Cog View2 04/22]-->8[CogVideo 05/12]-->9[Imagen 05/22]
end

10>支持中文生成图像]---2
11>支持中文生成图像还能生成比较短的视频]---3
12>百度的模型,100亿参数]---5
13>google的模型,模型更为简单]---9
```

## Abstract 摘要

Contrastive models like CLIP have been shown to learn robust representations of images that capture both semantics and style. To leverage these representations for image generation, we propose a two-stage model: a prior that generates a CLIP image embedding given a text caption, and a decoder that generates an image conditioned on the image embedding. We show that explicitly generating image representations improves image diversity with minimal loss in photorealism and caption similarity. Our decoders conditioned on image representations can also produce variations of an image that preserve both its semantics and style, while varying the non-essential details absent from the image representation. Moreover, the joint embedding space of CLIP enables language-guided image manipulations in a zero-shot fashion. We use diffusion models for the decoder and experiment with both autoregressive and diffusion models for the prior, finding that the latter are computationally more efficient and produce higher-quality samples.

CLIP 等对比模型已被证明可以学习捕获语义和风格的图像的稳健表示。为了利用这些表示来生成图像，我们提出了一个两阶段模型：一个在给定文本标题的情况下生成 CLIP 图像嵌入的先验（prior）模型，以及一个以图像特征为条件生成图像的解码器。我们发现，显式生成图像特征可以提高图像多样性，同时将真实感和文本匹配性的损失降到最低。【图像又逼真又多样】我们以图像特征为条件的解码器还可以生成图像的变体，同时保留其语义和风格，同时改变图像表示中不存在的非必要细节。此外，CLIP 的联合特征空间能够以零样本的方式进行语言引导的图像操作。我们对解码器使用扩散模型，并对先验（prior）模型使用自回归模型和扩散模型进行实验，发现后者在计算上更高效并产生更高质量的样本。

```mermaid
graph LR
1[text]-->2[Text Embedding]--prior-->3[Image Embedding]-->4[Image]

```

## 1 Introduction 引言

Recent progress in computer vision has been driven by scaling models on large datasets of captioned images collected from the internet [10,44,60,39,31,16]. Within this framework, CLIP [39] has emerged as a successful representation learner for images. CLIP embeddings have a number of desirable properties: they are robust to image distribution shift, have impressive zero-shot capabilities, and have been fine-tuned to achieve state-of-the-art results on a wide variety of vision and language tasks [45]. Concurrently, diffusion models [46,48,25] have emerged as a promising generative modeling framework, pushing the state-of-the-art on image and video generation tasks [11,26,24]. To achieve best results, diffusion models leverage a guidance technique [11,24] which improves sample fidelity (for images, photorealism) at the cost of sample diversity.

计算机视觉的最新进展是由从互联网收集的带标题的图像的大型数据集上的扩展模型所驱动的 [10,44,60,39,31,16]。 在这个框架内，CLIP [39] 已经成为一个成功的图像表示学习器。 CLIP 嵌入具有许多理想的特性：它们对图像分布偏移具有鲁棒性，具有令人印象深刻的零样本能力，并且经过微调以在各种视觉和语言任务上实现最先进的结果 [45]。 同时，扩散模型 [46,48,25] 已成为一种很有前途的生成建模框架，推动了图像和视频生成任务的最新技术发展 [11,26,24]。为了获得最佳结果，扩散模型利用引导技术 [11,24]，以牺牲样本多样性为代价提高样本保真度（对于图像，照片级真实感）。

>扩散模型的发展历程
```mermaid
graph LR
subgraph 主线 
1[DDPM]-->2[improve DDPM]-->3[Diffusion Models beats GAN]-->4[Glide]-->5[DALIE2]
end

6[classfier free guidance]-->4

```

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Hierarchical%20Text-Conditional%20Image%20Generation%20with%20CLIP%20Latents/Fig%201.png"/></div>

Figure 1: Selected $1024 \times 1024$ samples from a production version of our model.

图 1：从我们模型的生产版本中选择 $1024 \times 1024$ 样本。

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Hierarchical%20Text-Conditional%20Image%20Generation%20with%20CLIP%20Latents/Fig%202.png"/></div>

Figure 2: A high-level overview of unCLIP. Above the dotted line, we depict the CLIP training process, through which we learn a joint representation space for text and images. Below the dotted line, we depict our text-to-image generation process: a CLIP text embedding is first fed to an autoregressive or diffusion prior to produce an image embedding, and then this embedding is used to condition a diffusion decoder which produces a final image. Note that the CLIP model is frozen during training of the prior and decoder. \
图 2：unCLIP 的高级概述。 在虚线上方，我们描绘了 CLIP 训练过程，通过它我们学习了文本和图像的联合表示空间。 在虚线下方，我们描述了文本到图像的生成过程：在生成图像特征之前，首先将 CLIP 文本特征输入到自回归或扩散，然后该特征用于调节扩散解码器，从而产生最终的 图片。 请注意，CLIP 模型在先验和解码器的训练过程中被冻结。【用CLIP生成的图像特征当基准去监督prior模型的学习】

In this work, we combine these two approaches for the problem of text-conditional image generation. We first train a diffusion decoder to invert the CLIP image encoder. Our inverter is non-deterministic, and can produce multiple images corresponding to a given image embedding. The presence of an encoder and its approximate inverse (the decoder) allows for capabilities beyond text-to-image translation. As in GAN inversion [62,55], encoding and decoding an input image produces semantically similar output images (Figure 3). We can also interpolate between input images by inverting interpolations of their image embeddings (Figure 4). However, one notable advantage of using the CLIP latent space is the ability to semantically modify images by moving in the direction of any encoded text vector (Figure 5), whereas discovering these directions in GAN latent space involves luck and diligent manual examination. Furthermore, encoding and decoding images also provides us with a tool for observing which features of the image are recognized or disregarded by CLIP.

在这项工作中，我们结合这两种方法来解决文本条件图像生成问题。我们首先训练一个扩散解码器来反转 CLIP 图像编码器。我们的反相器是非确定性的，可以产生与给定图像嵌入相对应的多个图像。编码器及其近似逆（解码器）的存在允许超越文本到图像翻译的能力。与 GAN 反转 [62,55] 一样，对输入图像进行编码和解码会产生语义相似的输出图像（图 3）。我们还可以通过反转图像嵌入的插值来在输入图像之间进行插值（图 4）。然而，使用 CLIP 潜在空间的一个显着优势是能够通过沿任何编码文本向量的方向移动来语义修改图像（图 5），而在 GAN 潜在空间中发现这些方向需要运气和勤奋的人工检查。此外，编码和解码图像还为我们提供了一种工具，用于观察图像的哪些特征被 CLIP 识别或忽略。

To obtain a full generative model of images, we combine the CLIP image embedding decoder with a prior model, which generates possible CLIP image embeddings from a given text caption. We compare our text-to-image system with other systems such as DALL-E [40] and GLIDE [35], finding that our samples are comparable in quality to GLIDE, but with greater diversity in our generations. We also develop methods for training diffusion priors in latent space, and show that they achieve comparable performance to autoregressive priors, while being more compute-efficient. We refer to our full text-conditional image generation stack as unCLIP, since it generates images by inverting the CLIP image encoder.

为了获得完整的图像生成模型，我们将 CLIP 图像嵌入解码器与先验模型相结合，后者从给定的文本标题生成可能的 CLIP 图像嵌入。 我们将我们的文本到图像系统与 DALL-E [40] 和 GLIDE [35] 等其他系统进行比较，发现我们的样本在质量上与 GLIDE 相当，但在我们这一代人中具有更大的多样性。 我们还开发了在潜在空间中训练扩散先验的方法，并表明它们实现了与自回归先验相当的性能，同时计算效率更高。 我们将全文条件图像生成堆栈称为 unCLIP，因为它通过反转 CLIP 图像编码器来生成图像。

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Hierarchical%20Text-Conditional%20Image%20Generation%20with%20CLIP%20Latents/GAN2.png"/></div>

>GAN介绍：
> 缺点：
> 
> - 难训练
> - 多样性差(多样性来自随机噪声)
> - 隐式生成，不知道图像的概率分布

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Hierarchical%20Text-Conditional%20Image%20Generation%20with%20CLIP%20Latents/VAE2.png"/></div>

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Hierarchical%20Text-Conditional%20Image%20Generation%20with%20CLIP%20Latents/VAE4.png"/></div>

>VAE介绍
>Auto-encoder: encoder和decoder，希望输出x'接近输入x
>denoising AE: 加入噪音$x_c$
>Variational AE: 生成一个高斯分布，z是prior，x'是likelihood
>VQ-VAE: vector quantized，离散化处理分布，用codebook代替，类似于聚类中心; fq是quantized features
>DALL-E:图像文本对，过VQVAE，文本特征和图像特征concat；推理时自回归

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Hierarchical%20Text-Conditional%20Image%20Generation%20with%20CLIP%20Latents/Diffusion.png"/></div>

>Diffusion介绍
>前向扩散：对图片加T次正态分布的噪音
>reverse diffusion: 反向扩散
>U-Net: encoder, decoder, 前后大小一致

```mermaid
graph LR
subgraph 主线 
1[DDPM]-->2[improve DDPM]-->3[Diffusion Models beats GAN]-->4[Glide]-->5[DALIE2]
end

6[classfier free guidance]-->4

```

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Hierarchical%20Text-Conditional%20Image%20Generation%20with%20CLIP%20Latents/classifier%20.png"/></div>

>Classifier介绍

<div align=center></div>

## 2 Method 方法

Our training dataset consists of pairs $(x,y)$ of images $x$ and their corresponding captions $y$. Given an image $x$, let $z_i$ and $z_t$ be its CLIP image and text embeddings, respectively. We design our generative stack to produce images from captions using two components:

我们的训练数据集由一对$(x,y)$的图像$x$和它们相应的标题$y$组成。给定一个图像$x$，让$z_i$和$z_t$分别为其CLIP图像和文本特征。我们设计了我们的生成堆栈，使用两个组件从标题中产生图像。

- A *prior* $P(z_i|y)$ that produces CLIP image embeddings $z_i$ conditioned on captions $y$. \
一个*先验的*$P(z_i|y)$，产生CLIP图像特征$z_i$，以标题$y$为条件。
- A *decoder* $P(x|z_i,y)$ that produces images $x$ conditioned on CLIP image embeddings $z_i$ (and optionally text captions $y$). \
一个*解码器*$P(x|z_i,y)$，产生以CLIP图像特征$z_i$为条件的图像$x$（和可选的文本标题$y$）

The decoder allows us to invert images given their CLIP image embeddings, while the prior allows us to learn a generative model of the image embeddings themselves. Stacking these two components yields a generative model $P(x|y)$ of images $x$ given captions $y$:

解码器允许我们根据CLIP图像特征来反转图像，而先验（prior）允许我们学习图像特征本身的生成模型。将这两个部分叠加起来，就会产生一个图像$x$的生成模型$P(x|y)$，并给出标题$y$。

$$
P(x|y) = P(x,z_i|y) = P(x|z_i,y)P (z_i|y).
$$

The first equality holds because $z_i$ is a deterministic function of $x$. The second equality holds because of the chain rule. Thus, we can sample from the true conditional distribution $P(x|y)$ by first sampling $z_i$ using the prior, and then sampling $x$ using the decoder. In the following sections, we describe our decoder and prior stacks. For training details and hyperparameters, refer to Appendix C.

第一个等式成立是因为 $z_i$ 是 $x$ 的确定性函数（一对一的关系）。 由于链式法则，第二个等式成立。 因此，我们可以通过首先使用先验对 $z_i$ 进行采样，然后使用解码器对 $x$ 进行采样，从真实条件分布 $P(x|y)$ 进行采样。 在以下部分中，我们将描述我们的解码器和先前的堆栈。 有关训练详细信息和超参数，请参阅附录 C。【通过概率证明两阶段设计是有证据的】

### 2.1 Decoder 解码器

We use diffusion models [25,48] to produce images conditioned on CLIP image embeddings (and optionally text captions). Specifically, we modify the architecture described in Nichol et al. (2021) by projecting and adding CLIP embeddings to the existing timestep embedding, and by projecting CLIP embeddings into four extra tokens of context that are concatenated to the sequence of outputs from the GLIDE text encoder. We retained the text conditioning pathway present in the original GLIDE model, hypothesizing that it could allow the diffusion model to learn aspects of natural language that CLIP fails to capture (e.g. variable binding), but find that it offers little help in this regard (Section 7).

我们使用扩散模型 [25,48] 来生成以 CLIP 图像特征（以及可选的文本说明）为条件的图像。 具体来说，我们修改了 Nichol 等人中描述的架构。 (2021) 通过将 CLIP 特征投影和添加到现有的时间步长嵌入，并将 CLIP 嵌入投影到四个额外的上下文标记中，这些标记连接到 GLIDE 文本编码器的输出序列。 我们保留了原始 GLIDE 模型中存在的文本条件通路，假设它可以让扩散模型学习 CLIP 无法捕获的自然语言方面（例如变量绑定），但发现它在这方面提供的帮助很小（在第7部分).

While we can sample from the conditional distribution of the decoder directly, past work using diffusion models shows using guidance on the conditioning information [11,24,35] improves sample quality a lot. We enable classifier-free guidance [24] by randomly setting the CLIP embeddings to zero (or a learned embedding) 10% of the time, and randomly dropping the text caption 50% of the time during training.

虽然我们可以直接从解码器的条件分布中采样，但过去使用扩散模型的工作表明，使用对条件信息的指导 [11,24,35] 可以大大提高样本质量。 我们通过在 10% 的时间内将 CLIP 特征随机设置为零（或学习特征）并在训练期间在 50% 的时间内随机删除文本标题来启用无分类器指导 [24]。

To generate high resolution images, we train two diffusion upsampler models [34,43]: one to upsample images from $64 \times 64$ to $256 \times 256$ resolution, and another to further upsample those to $1024  \times 1024$ resolution. To improve the robustness of our upsamplers, we slightly corrupt the conditioning images during training. For the first upsampling stage, we use gaussian blur [43], and for the second, we use a more diverse BSR degradation [42,59]. To reduce training compute and improve numerical stability, we follow Rombach et al. [42] and train on random crops of images that are one-fourth the target size. We use only spatial convolutions in the model (i.e., no attention layers) and at inference time directly apply the model at the target resolution, observing that it readily generalizes to the higher resolution. We found no benefit from conditioning the upsamplers on the caption, and use unconditional ADMNets [11] with no guidance.

为了生成高分辨率图像，我们训练了两个扩散上采样器模型 [34,43]：一个将图像从 $64 \times 64$ 分辨率上采样到 $256 \times 256$ 分辨率，另一个将这些图像进一步上采样到 $1024 \times 1024$ 分辨率。 为了提高我们的上采样器的稳健性，我们在训练期间稍微破坏了条件图像。 对于第一个上采样阶段，我们使用高斯模糊 [43]，对于第二个，我们使用更多样化的 BSR 退化 [42,59]。 为了减少训练计算并提高数值稳定性，我们遵循 Rombach 等人。 [42] 并训练随机裁剪的图像，这些图像是目标大小的四分之一。 我们在模型中仅使用空间卷积（即没有注意力层），并在推理时直接在目标分辨率下应用模型，观察到它很容易推广到更高分辨率。 我们发现在标题上调节上采样器没有任何好处，并且在没有指导的情况下使用无条件 ADMNets [11]。

### 2.2 Prior

While a decoder can invert CLIP image embeddings $z_i$ to produce images $x$, we need a prior model that produces $z_i$ from captions $y$ to enable image generations from text captions. We explore two different model classes for the prior model:

虽然解码器可以反转 CLIP 图像特征 $z_i$ 以生成图像 $x$，但我们需要一个先验模型从标题 $y$ 生成 $z_i$ 以启用从文本标题生成图像。 我们为先前的模型探索了两个不同的模型类：

- *Autoregressive* (AR) prior: the CLIP image embedding $z_i$ is converted into a sequence of discrete codes and predicted autoregressively conditioned on the caption $y$. \
- *自回归* (AR) 先验：嵌入 $z_i$ 的 CLIP 图像被转换为一系列离散代码，并根据标题 $y$ 进行自回归预测。
- *Diffusion* prior: The continuous vector $z_i$ is directly modelled using a Gaussian diffusion model conditioned on the caption $y$. \
*扩散*先验：连续向量 $z_i$ 直接使用以标题 $y$ 为条件的高斯扩散模型建模。

In addition to the caption, we can condition the prior on the CLIP text embedding $z_t$ since it is a deterministic function of the caption. To improve sample quality we also enable sampling using classifier-free guidance for both the AR and diffusion prior, by randomly dropping this text conditioning information 10% of the time during training.

除了标题之外，我们还可以在 CLIP 文本特征 $z_t$ 上调节先验，因为它是标题的确定性函数。 为了提高样本质量，我们还通过在训练期间 10% 的时间内随机删除此文本条件信息，来启用对 AR 和扩散先验使用无分类器指导的采样。

To train and sample from the AR prior more efficiently, we first reduce the dimensionality of the CLIP image embeddings $z_i$ by applying Principal Component Analysis (PCA) [37]. In particular, we find that the rank of the CLIP representation space is drastically reduced when training CLIP with SAM [15] while slightly improving evaluation metrics. We are able to preserve nearly all of the information$^2$ by retaining only 319 principal components out of the original 1,024. After applying PCA, we order the principal components by decreasing eigenvalue magnitude, quantize each of the 319 dimensions into 1,024 discrete buckets, and predict the resulting sequence using a Transformer [53] model with a causal attention mask. This results in a threefold reduction in the number of tokens predicted during inference, and improves training stability.

为了更有效地从 AR 先验中训练和采样，我们首先通过应用主成分分析 (PCA) [37] 来降低 CLIP 图像嵌入 $z_i$ 的维数。 特别是，我们发现当使用 SAM [15] 训练 CLIP 时，CLIP 表示空间的等级会大大降低，同时略微提高评估指标。 通过仅保留原始 1,024 个主成分中的 319 个，我们能够保留几乎所有信息$^2$。 应用 PCA 后，我们通过降低特征值大小对主成分进行排序，将 319 个维度中的每一个量化为 1,024 个离散数据块，并使用具有因果注意掩码的 Transformer [53] 模型预测结果序列。 这导致推理期间预测的标记数量减少了三倍，并提高了训练稳定性。

>$^2$ I.e., less than 1% average mean-squared error in reconstructing the image representations.

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Hierarchical%20Text-Conditional%20Image%20Generation%20with%20CLIP%20Latents/Fig%203.png"/></div>

Figure 3: Variations of an input image by encoding with CLIP and then decoding with a diffusion model. The variations preserve both semantic information like presence of a clock in the painting and the overlapping strokes in the logo, as well as stylistic elements like the surrealism in the painting and the color gradients in the logo, while varying the non-essential details.

We condition the AR prior on the text caption and the CLIP text embedding by encoding them as a prefix to the sequence. Additionally, we prepend a token indicating the (quantized) dot product between the text embedding and image embedding, $z_i \cdot z_t$ . This allows us to condition the model on a higher dot product, since higher text-image dot products correspond to captions which better describe the image. In practice, we find it beneficial to sample the dot product from the top half of the distribution$^3$.

我们通过将文本标题和 CLIP 文本特征编码为序列的前缀来调节 AR 先验。 此外，我们在前面加上一个标记，表示文本特征和图像特征之间的（量化的）点积 $z_i \cdot z_t$ 。 这允许我们在更高的点积上调整模型，因为更高的文本图像点积对应于更好地描述图像的标题。 在实践中，我们发现从分布 $^3$ 的上半部分采样点积是有益的。

>$^3$We swept over percentiles 50%, 70%, 85%, 95% and found 50% to be optimal in all experiments.

For the diffusion prior, we train a decoder-only Transformer with a causal attention mask on a sequence consisting of, in order: the encoded text, the CLIP text embedding, an embedding for the diffusion timestep, the noised CLIP image embedding, and a final embedding whose output from the Transformer is used to predict the unnoised CLIP image embedding. We choose not to condition the diffusion prior on $z_i \cdot z_t$ like in the AR prior; instead, we improve quality during sampling time by generating two samples of $z_i$ and selecting the one with a higher dot product with $z_t$ . Instead of using the $\epsilon$-prediction formulation from Ho et al. [25], we find it better to train our model to predict the unnoised $z_i$ directly, and use a mean-squared error loss on this prediction:

对于扩散先验，我们在一个序列上训练一个带有因果注意掩码的解码器 Transformer，顺序包括：编码文本、CLIP 文本特征、扩散时间步长的特征、噪声 CLIP 图像特征，以及最终嵌入——其来自 Transformer 的输出用于预测无噪声 CLIP 图像特征【Transformer自己的特征如[CLS]】。 我们选择不像 AR 先验那样在 $z_i \cdot z_t$ 上先验扩散； 相反，我们通过生成 $z_i$ 的两个样本并选择与 $z_t$ 具有更高点积的样本来提高采样时间的质量。 而不是使用 Ho 等人的 $\epsilon$ 预测公式。 [25]，我们发现最好训练我们的模型直接预测无噪声 $z_i$，并在此预测上使用均方误差损失：【与之前的不同】

>Parti:用自回归打败了imagen和DELIE2

$$
L_{\rm Prior} = \mathbb{E}_{t〜[1,T],z_i^{(t)}〜q_t}[\Vert f_\theta(z^{(t)}_i,t,y)-z_i \Vert^2]
$$

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Hierarchical%20Text-Conditional%20Image%20Generation%20with%20CLIP%20Latents/Fig%204.png"/></div>

Figure 4: Variations between two images by interpolating their CLIP image embedding and then decoding with a diffusion model. We fix the decoder seed across each row. The intermediate variations naturally blend the content and style from both input images.

## 3 Image Manipulations

Our approach allows us to encode any given image $x$ into a bipartite latent representation $(z_i,x_T)$ that is sufficient for the decoder to produce an accurate reconstruction. The latent $z_i$ describes the aspects of the image that are recognized by CLIP, while the latent $x_T$ encodes all of the residual information necessary for the decoder to reconstruct $x$. The former is obtained by simply encoding the image with the CLIP image encoder. The latter is obtained by applying DDIM inversion (Appendix F in [11]) to $x$ using the decoder, while conditioning on $z_i$ . We describe three different kinds of manipulations that are enabled by this bipartite representation.

## 3.1 Variations

Given an image $x$, we can produce related images that share the same essential content but vary in other apects, such as shape and orientation (Figure 3). To do this, we apply the decoder to the bipartite representation $(z_i, x_T)$ using DDIM with $η> 0$ for sampling. With $η= 0$, the decoder becomes deterministic and will reconstruct the given image $x$. Larger values of introduce stochasticity into successive sampling steps, resulting in variations that are perceptually “centered” around the original image $x$. As increases, these variations tell us what information was captured in the CLIP image embedding (and thus is preserved across samples), and what was lost (and thus changes across the samples).

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Hierarchical%20Text-Conditional%20Image%20Generation%20with%20CLIP%20Latents/Fig%205.png"/></div>

Figure 5: Text diffs applied to images by interpolating between their CLIP image embeddings and a normalised difference of the CLIP text embeddings produced from the two descriptions. We also perform DDIM inversion to perfectly reconstruct the input image in the first column, and fix the decoder DDIM noise across each row.

### 3.2 Interpolations

It is also possible to blend two images $x_1$ and $x_2$ for variations (Figure 4), traversing all of the concepts in CLIP’s embedding space that occur between them. To do this, we rotate between their CLIP embeddings $z_{i_1}$ and $z_{i_2}$ using spherical interpolation, yielding intermediate CLIP representations $z_{i_\theta} = {\rm slerp}(z_{i_1} ,z_{i_2},\theta)$ as $\theta$ is varied from 0 to 1. There are two options for producing the intermediate DDIM latents along the trajectory. The first option involves interpolating between their DDIM inverted latents $x_{T_1}$ and $x_{T_2}$ (by setting $x_{T_\theta} = {\rm slerp}(x_{T_1},x_{T_2},\theta))$, which yields a single trajectory whose endpoints reconstruct $x_1$ and $x_2$. The second option involves fixing the DDIM latent to a randomly-sampled value for all interpolates in the trajectory. This results in an infinite number of trajectories between $x_1$ and $x_2$, though the endpoints of these trajectories will generally no longer coincide with the original images. We use this approach in Figure 4.

### 3.3 Text Diffs

A key advantage of using CLIP compared to other models for image representations is that it embeds images and text to the same latent space, thus allowing us to apply language-guided image manipulations (i.e., text diffs), which we show in Figure 5. To modify the image to reflect a new text description $y$, we first obtain its CLIP text embedding $z_t$ , as well as the CLIP text embedding $z_{t_0}$ of a caption describing the current image$^4$. We then compute a *text diff* vector $z_d = norm(z_t — z_{t_0})$ from these by taking their difference and normalizing. Now, we can rotate between the image CLIP embedding $z_i$ and the text diff vector $z_d$ using spherical interpolation, yielding intermediate CLIP representations $z ={\rm slerp}(z_i,z_d,\theta)$, where $\theta$ is increased linearly from 0 to a maximum value that is typically in [0.25, 0.50]. We produce the final outputs by decoding the interpolates $z_\theta$, fixing the base DDIM noise to $x_T$ throughout the entire trajectory.

> $^4$ Instead of a description of the current image, we also experimented with using a dummy caption like “a photo” for the baseline, or removing it altogether. These also worked well.

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Hierarchical%20Text-Conditional%20Image%20Generation%20with%20CLIP%20Latents/Fig%206.png"/></div>

Figure 6: Variations of images featuring typographic attacks [20] paired with the CLIP model’s predicted probabilities across three labels. Surprisingly, the decoder still recovers Granny Smith apples even when the predicted probability for this label is near 0%. We also find that our CLIP model is slightly less susceptible to the “pizza” attack than the models investigated in [20].

## 4 Probing the CLIP Latent Space

Our decoder model provides a unique opportunity to explore CLIP latent space by allowing us to directly visualize what the CLIP image encoder is seeing. As an example use case, we can revisit cases where CLIP makes incorrect predictions, such as typographic attacks [20]. In these adversarial images, a piece of text is overlayed on top of an object, which causes CLIP to predict the object described by the text rather than the object depicted in the image. This piece of text essentially hides the original object in terms of output probabilities. In Figure 6, we show an example of this attack from [20], wherein an apple can be misclassified as an iPod. Surprisingly, we find that our decoder still generates pictures of apples with high probability even though the predicted probability of “Granny Smith” is near zero. Even more notable, the model never produces pictures of iPods, despite the very high relative predicted probability of this caption.

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Hierarchical%20Text-Conditional%20Image%20Generation%20with%20CLIP%20Latents/Fig%207.png"/></div>

Figure 7: Visualization of reconstructions of CLIP latents from progressively more PCA dimensions (20, 30, 40, 80, 120, 160, 200, 320 dimensions), with the original source image on the far right. The lower dimensions preserve coarse-grained semantic information, whereas the higher dimensions encode finer-grained details about the exact form of the objects in the scene.

PCA reconstructions offer another tool for probing the structure of the CLIP latent space. In Figure 7, we take the CLIP image embeddings of a handful of source images and reconstruct them with progressively more PCA dimensions, and then visualize the reconstructed image embeddings using our decoder with DDIM on a fixed seed. This allows us to see what semantic information the different dimensions encode. We observe that the early PCA dimensions preserve coarse-grained semantic information such as what types of objects are in the scene, whereas the later PCA dimensions encode finer-grained detail such as the shapes and exact form of the objects. For example, in the first scene, the earlier dimensions seem to encode that there is food and perhaps a container present, whereas the later dimensions encode tomatoes and a bottle specifically. Figure 7 also serves as a visualization of what the AR prior is modeling, since the AR prior is trained to explicitly predict these principal components in this order.

## 5 Text-to-Image Generation

### 5.1 Importance of the Prior

Although we train a prior to generate CLIP image embeddings from captions, the prior is not strictly necessary for caption-to-image generation. For instance, our decoder can condition on both CLIP image embeddings and captions, but the CLIP image embedding is dropped 5% of the time during training in order to enable classifier-free guidance. Therefore, at sampling time, we can condition on only the caption, although this underperforms a model trained fully in this way (this model is GLIDE, and we do a thorough comparison with GLIDE in Sections 5.2 and 5.3). Another possibility is to feed the decoder the CLIP text embedding as if it were an image embedding, as previously observed [61, 54]. The first two rows of Figure 8 depicts samples obtained in these two ways; the third row depicts samples obtained with a prior. Conditioning the decoder on just the caption is clearly worst, but conditioning on text embeddings zero-shot does produce reasonable results. Building on this observation, another approach would be to train the decoder to condition on CLIP text embeddings [9] instead of CLIP image embeddings (although we would lose the capabilities mentioned in Section 4).

To quantify the effectiveness of these alternate approaches, we train two models: a small decoder conditioned on CLIP text embeddings, and a small unCLIP stack (diffusion prior and decoder). We then compare samples from the text-embedding decoder, samples from the unCLIP stack, and samples obtained from feeding text embeddings to the unCLIP decoder zero-shot, sweeping across guidance scales for all models. We find that these approaches respectively score FIDs of 9.16, 7.99, and 16.55 on a test set, suggesting the unCLIP approach is best. We also run human evaluations comparing the first two settings, sweeping over sampling hyperparameters for each using our human evaluation proxy model (Appendix A). We find that humans prefer the full unCLIP stack 57.0% ± 3.1% of the time for photorealism and 53.1% ± 3.1% of the time for caption
similarity.

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Hierarchical%20Text-Conditional%20Image%20Generation%20with%20CLIP%20Latents/Fig%208.png"/></div>

Figure 8: Samples using different conditioning signals for the same decoder. In the first row, we pass the text caption to the decoder, and pass a zero vector for the CLIP embedding. In the second row, we pass both the text caption and the CLIP text embedding of the caption. In the third row, we pass the text and a CLIP image embedding generated by an autoregressive prior for the given caption. Note that this decoder is only trained to do the text-to-image generation task (without the CLIP image representation) 5% of the time.

Given the importance of the prior, it is worth evaluating different approaches for training it. We compare both the AR and diffusion priors throughout our experiments. In all cases (Sections 5.2, 5.4, and 5.5), we find that the diffusion prior outperforms the AR prior for comparable model size and reduced training compute.

### 5.2 Human Evaluations

We observe in Figure 1 that unCLIP is capable of synthesizing complex, realistic images. While we can compare sample quality to past models using FID, it is not always aligned with human judgment. To better gauge the generation capabilities of our system, we conduct systematic human evaluations comparing unCLIP to GLIDE for photorealism, caption similarity, and sample diversity.

We follow the protocol of Ramesh et al., Nichol et al. [40, 35] for the first two evaluations: for photorealism, users are presented with pairs of images and must choose which looks more photorealistic; for caption similarity, users are additionally prompted with a caption, and must choose which image better matches the caption. In both evaluations, there is a third “Not sure” option. For diversity, we propose a new evaluation protocol in which humans are presented with two 4 4 grids of samples and must choose which is more diverse (with a third option, “Not sure”). For this evaluation, we produce sample grids using 1,000 captions from the MS-COCO validation set, and always compare sample grids for the same caption. Before running human comparisons, we swept over sampling hyperparameters for each model using a CLIP linear probe trained to be a proxy for human photorealism evaluations (Appendix A). These hyperparameters are fixed across all three types of evaluation.

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Hierarchical%20Text-Conditional%20Image%20Generation%20with%20CLIP%20Latents/Fig%209.png"/></div>

Figure 9: Samples when increasing guidance scale for both unCLIP and GLIDE, using the prompt, “A green vase filled with red roses sitting on top of table.” For unCLIP, we fix the latent vectors sampled from the prior, and only vary the guidance scale of the decoder. For both models, we fix the diffusion noise seed for each column. Samples from unCLIP improve in quality (more realistic lighting and shadows) but do not change in content as we increase guidance scale, preserving semantic diversity even at high decoder guidance scales.

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Hierarchical%20Text-Conditional%20Image%20Generation%20with%20CLIP%20Latents/Table%201.png"/></div>

Table 1: Human evaluations comparing unCLIP to GLIDE. We compare to both the AR and diffusion prior for unCLIP. Reported figures are 95% confidence intervals of the probability that the unCLIP model specified by the row beats GLIDE. Sampling hyperparameters for all models were swept to optimize an automated proxy for human photorealism evaluations.

We present our results in Table 1. In general, the diffusion prior performs better than the AR prior in pairwise comparisons against GLIDE. We find that humans still slightly prefer GLIDE to unCLIP in terms of photorealism, but the gap is very small. Even with similar photorealism, unCLIP is strongly preferred over GLIDE in terms of diversity, highlighting one of its benefits.

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Hierarchical%20Text-Conditional%20Image%20Generation%20with%20CLIP%20Latents/Fig%2010.png"/></div>

Figure 10: When comparing unCLIP (with our best sampling settings) to various settings of guidance scale for GLIDE, unCLIP was preferred by human evaluators on at least one axis among photorealism, caption similarity, and diversity for each comparison. At the higher guidance scales used to generate photorealistic images, unCLIP yields greater diversity for comparable photorealism and caption similarity.

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Hierarchical%20Text-Conditional%20Image%20Generation%20with%20CLIP%20Latents/Fig%2011.png"/></div>

Figure 11: FID versus guidance scale for unCLIP and GLIDE. For the unCLIP priors, we swept over sampling hyperparameters and fixed to the settings with the best minimum FID.

### 5.3 Improved Diversity-Fidelity Trade-off with Guidance

Compared to GLIDE, we qualitatively observe that unCLIP is able to generate more diverse images while leveraging the guidance technique to improve sample quality. To understand why, consider Figure 9 where we increase guidance scale for both GLIDE and unCLIP. For GLIDE, the semantics (camera angle, color, size) converge as we increase guidance scale, whereas for unCLIP the semantic information of the scene is frozen in the CLIP image embedding and therefore does not collapse when guiding the decoder.

In Section 5.2, we observed that unCLIP achieves similar photorealism as GLIDE while maintaining more diversity, but that its caption matching capabilities were slightly worse. It is natural to ask whether GLIDE’s guidance scale can be lowered to obtain the same diversity level as unCLIP while maintaining better caption matching. In Figure 10, we conduct a more careful study of this question by performing human evaluations across several GLIDE guidance scales. We find that GLIDE at guidance scale 2.0 is very close to the photorealism and caption similarity of unCLIP, while still producing less diverse samples.

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Hierarchical%20Text-Conditional%20Image%20Generation%20with%20CLIP%20Latents/Table%202.png"/></div>

Table 2: Comparison of FID on MS-COCO $256 \times 256$. We use guidance scale 1.25 for the decoder for both the AR and diffusion prior, and achieve the best results using the diffusion prior. \
表 2：MS-COCO 上 FID 的比较 $256 \times 256$。 我们对 AR 和扩散先验的解码器使用 guidance scale 1.25，并使用扩散先验获得最佳结果。

Finally, in Figure 11 we compute MS-COCO zero-shot FID [23] while sweeping over guidance scale for both unCLIP and GLIDE, finding that guidance hurts the FID of unCLIP much less so than for GLIDE. In this evaluation, we fix the guidance scale of the unCLIP prior and only vary the guidance scale of the decoder. This is another indication that guidance hurts the diversity of GLIDE much more than unCLIP, since FID heavily penalizes non-diverse generations.

### 5.4 Comparison on MS-COCO

In the text-conditional image generation literature, it has become standard practice to evaluate FID on the MS-COCO [28] validation set. We present results on this benchmark in Table 2. Like GLIDE and DALL-E, unCLIP is not directly trained on the MS-COCO training set, but can still generalize to the validation set zero-shot. We find that, compared to these other zero-shot models, unCLIP achieves a new state-of-the-art FID of 10.39 when sampling with the diffusion prior. In Figure 12, we visually compare unCLIP to various recent text-conditional image generation models on several captions from MS-COCO. We find that, like the other methods, unCLIP produces realistic scenes that capture the text prompts.

### 5.5 Aesthetic Quality Comparison

We additionally perform automated aesthetic quality evaluations comparing unCLIP to GLIDE. Our goal with this evaluation is to assess how well each model produces artistic illustrations and photographs. To this end, we generated 512 “artistic” captions using GPT-3 [4] by prompting it with captions for existing artwork (both real and AI generated). Next, we trained a CLIP linear probe to predict human aesthetic judgments using the AVA dataset [33] (Appendix A). For each model and set of sampling hyperparameters, we produce four images for each prompt, and report the mean predicted aesthetic judgment over the full batch of 2048 images.

In Figure 13, we present results on our aesthetic quality evaluation. We find that guidance improves aesthetic quality for both GLIDE and unCLIP. For unCLIP, we only guide the decoder (we found that guiding the prior hurt results). We also plot the aesthetic quality against Recall$^5$ , since guidance typically induces a trade-off between fidelity and diversity. Interestingly, we find that guiding unCLIP does not decrease Recall while still improving aesthetic quality according to this metric.

>$^5$ Recall is computed with respect to the training dataset.

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Hierarchical%20Text-Conditional%20Image%20Generation%20with%20CLIP%20Latents/Fig%2012.png"/></div>

Figure 12: Random image samples on MS—coco prompts.

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Hierarchical%20Text-Conditional%20Image%20Generation%20with%20CLIP%20Latents/Fig%2013.png"/></div>

Figure 13: Aesthetic quality evaluations comparing GLIDE and unCLIP using 512 auto-generated artistic prompts. We find that both models benefit from guidance, but unCLIP does not sacrifice recall for aesthetic quality.

## 6 Related Work 相关工作

Synthetic image generation is a well studied problem, and most popular techniques for unconditional image generation have also been applied to the text-conditional setting. Many previous works have trained GANs [21] on publicly available image captioning datasets to produce text-conditional image samples [56,63,49,58,57]. Other works have adapted the VQ-VAE approach [52] to text-conditional image generation by training autoregressive transformers on sequences of text tokens followed by image tokens [40,12,1]. Finally, some works have applied diffusion models to the problem, training either continuous [35] or discrete [22] diffusion models with auxiliary text encoders to handle textual input.

Previous works have leveraged hierarchical generative processes to create high-quality synthetic images. Razavi et al. [41] trains a multi-layer discrete autoencoder, allowing them to first sample coarse-grained latent codes and then use this as conditioning information when sampling higher-resolution latent codes. Child, Vahdat and Kautz [5,50] generate images using VAEs with a hierarchy of latent codes that increase progressively with resolution. Concurrently with our work, Gafni et al. [17] conditions a generative image model on segmentation masks, allowing for a generative process that first samples a semantic map of an image and then conditions the generated image on this information.

The computational benefits of using diffusion to model a latent space has been noted by previous works. Preechakul et al. [38] propose an autoencoder framework where diffusion models are used to render latent variables as images, and a second diffusion model is used to generate these latents (similar to our diffusion prior). Vahdat et al. [51] use a score-based model for the latent space of a VAE, while Rombach et al. [42] use diffusion models on the latents obtained from a VQGAN [14] like autoencoder.

Since its release, CLIP [39] has been used extensively to steer generative image models towards text prompts. Galatolo et al., Patashnik et al., Murdock, Gal et al. [19, 36, 32, 18] guide GANs using gradients from a CLIP model. For diffusion models, Dhariwal and Nichol [11] introduced classifier guidance as a way to use gradients from a classifier trained on noised images to steer the model towards higher quality generations. Nichol et al. [35] train a CLIP model on noised images and guide a text-conditional diffusion model, while Crowson, Crowson [7, 8] use an unnoised CLIP model to guide unconditional or class-conditional diffusion models. Ho and Salimans [24] introduced classifier-free guidance and showed that one can perform guidance  implictly from the predictions of the model with and without the conditioning information, thus removing the need for a classifier. Nichol et al. [35] showed classifier-free guidance works more favorably than CLIP guidance for text conditional image generation.

Several previous works have trained generative image models that are directly conditioned on CLIP embeddings. Zhou et al. [61] condition GAN models on randomly perturbed CLIP image embeddings, finding that these models can generalize to CLIP text embeddings to produce text-conditional images. Crowson [9] trained diffusion models conditioned on CLIP text embeddings, allowing for direct text-conditional image generation. Wang et al. [54] train an autoregressive generative model conditioned on CLIP image embeddings, finding that it generalizes to CLIP text embeddings well enough to allow for text-conditional image synthesis.

Bordes et al. [3] train diffusion models conditioned on image representations from contrastive models. While the diffusion models themselves cannot generate images unconditionally, the authors experimented with a simple approach for two-stage image generation by employing Kernel Density Estimation to sample image representations. By feeding these generated representations to the diffusion model, they can generate images end-to-end in a way similar to our proposed technique. However, our work differs from this in two ways: first, we use multimodal contrastive representations rather than image-only representations; second, we employ much more powerful generative models for the first stage of the generation hierarchy, and these generative models are conditioned on text.

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Hierarchical%20Text-Conditional%20Image%20Generation%20with%20CLIP%20Latents/Fig%2014.png"/></div>

Figure 14: Samples from unCLIP and GLIDE for the prompt “a red cube on top of a blue cube”.

## 7 Limitations and Risks 局限和风险

Although conditioning image generation on CLIP embeddings improves diversity, this choice does come with certain limitations. In particular, unCLIP is worse at binding attributes to objects than a corresponding GLIDE model. In Figure 14, we find that unCLIP struggles more than GLIDE with a prompt where it must bind two separate objects (cubes) to two separate attributes (colors). We hypothesize that this occurs because the CLIP embedding itself does not explicitly bind attributes to objects, and find that reconstructions from the decoder often mix up attributes and objects, as shown in Figure 15. A similar and likely related issue is that unCLIP struggles at producing coherent text, as illustrated in Figure 16; it is possible that the CLIP embedding does not precisely encode spelling information of rendered text. This issue is likely made worse because the BPE encoding we use obscures the spelling of the words in a caption from the model, so the model needs to have independently seen each token written out in the training images in order to learn to render it.

虽然在 CLIP 特征上调节图像生成可以提高多样性，但这种选择确实有一定的局限性。特别是，unCLIP 在将属性绑定到对象方面比相应的 GLIDE 模型更差。在图 14 中，我们发现 unCLIP 比 GLIDE 更费力地提示它必须将两个单独的对象（立方体）绑定到两个单独的属性（颜色）。我们假设发生这种情况是因为 CLIP 嵌入本身没有明确地将属性绑定到对象，并且发现来自解码器的重建经常混淆属性和对象，如图 15 所示。一个类似且可能相关的问题是 unCLIP 在生成时遇到困难连贯的文本，如图 16 所示； CLIP 嵌入可能没有精确编码渲染文本的拼写信息。这个问题可能会变得更糟，因为我们使用的 BPE 编码从模型中模糊了标题中单词的拼写，因此模型需要独立地看到训练图像中写出的每个标记，以便学习渲染它。

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Hierarchical%20Text-Conditional%20Image%20Generation%20with%20CLIP%20Latents/Fig%2015.png"/></div>

Figure 15: Reconstructions from the decoder for difficult binding problems. We find that the reconstructions mix up objects and attributes. In the first two examples, the model mixes up the color of two objects. In the rightmost example, the model does not reliably reconstruct the relative size of two objects.

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Hierarchical%20Text-Conditional%20Image%20Generation%20with%20CLIP%20Latents/Fig%2016.png"/></div>

Figure 16: Samples from unCLIP for the prompt, “A sign that says deep learning.”

We also note that our stack still has a hard time producing details in complex scenes (Figure 17). We hypothesize that this is a limitation of our decoder hierarchy producing an image at a base resolution of 64 64 and then upsampling it. Training our unCLIP decoder at a higher base resolution should be able to alleviate this, at the cost of additional training and inference compute.

我们还注意到，我们的堆栈仍然很难在复杂场景中生成细节（图 17）。 我们假设这是我们的解码器层次结构的局限性，它以 64×64 的基本分辨率生成图像，然后对其进行上采样。 以更高的基本分辨率训练我们的 unCLIP 解码器应该能够缓解这种情况，但代价是额外的训练和推理计算。

As discussed in the GLIDE paper, image generation models carry risks related to deceptive and otherwise harmful content. unCLIP’s performance improvements also raise the risk profile over GLIDE. As the technology matures, it leaves fewer traces and indicators that outputs are AI-generated, making it easier to mistake generated images for authentic ones and vice versa. More research is also needed on how the change in architecture changes how the model learns biases in training data.

正如 GLIDE 论文中所讨论的，图像生成模型具有与欺骗性和其他有害内容相关的风险。 unCLIP 的性能改进也提高了 GLIDE 的风险预测。 随着技术的成熟，它留下的痕迹和指示器会减少输出是由 AI 生成的，从而更容易将生成的图像误认为是真实图像，反之亦然。 还需要更多的研究来了解架构的变化如何改变模型学习训练数据中的偏差的方式。

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Hierarchical%20Text-Conditional%20Image%20Generation%20with%20CLIP%20Latents/Fig%2017.png"/></div>

Figure 17: unCLIP samples show low levels of detail for some complex scenes.

The risks of these models should be assessed in relation to the particular deployment context, which includes training data, guardrails in place, the deployment space, and who will have access. A preliminary analysis of these issues in the context of the DALL·E 2 Preview platform (the first deployment of an unCLIP model), can be found in Mishkin et al. [30].

这些模型的风险应该根据特定的部署环境进行评估，包括训练数据、适当的护栏、部署空间以及谁将有权访问。 在 DALL·E 2 预览平台（unCLIP 模型的首次部署）背景下对这些问题的初步分析可以在 Mishkin 等人的文章中找到。 [30]。

## 8 Acknowledgements 致谢

We’d like to thank Jong Wook Kim, Hyeonwoo Noh, Alec Radford, Pranav Shyam, and Ilya Sutskever for helpful discussions and contributions to our work. We’d also like to thank Yunxin Jiao for creating several figures used in the paper. We are grateful to the Acceleration and Supercomputing teams at OpenAI for their work on software and hardware infrastructure this project used.

## References 参考文献

[1] Armen Aghajanyan, Bernie Huang, Candace Ross, Vladimir Karpukhin, Hu Xu, Naman Goyal, Dmytro Okhonko, Mandar Joshi, Gargi Ghosh, Mike Lewis, and Luke Zettlemoyer. CM3: A Causal Masked Multimodal Model of the Internet. arXiv:2201.07520, 2022.

[2] Fan Bao, Chongxuan Li, Jun Zhu, and Bo Zhang. Analytic-DPM: an Analytic Estimate of the Optimal Reverse Variance in Diffusion Probabilistic Models. CoRR, abs/2201.06503, 2022. URL <https://arxiv.org/abs/2201.06503>.

[3] Florian Bordes, Randall Balestriero, and Pascal Vincent. High Fidelity Visualization of What Your Self-Supervised Representation Knows About. arXiv:2112.09164, 2021.

[4] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language Models are Few-Shot Learners. arXiv:2005.14165, 2020.

[5] Rewon Child. Very Deep VAEs Generalize Autoregressive Models and Can Outperform Them on Images. arXiv:2011.10650, 2021.

[6] Katherine Crowson. AVA Linear Probe. <https://twitter.com/RiversHaveWings/status/1472346186728173568?s=20&t=T-HRr3Gw5HRGjQaMDtRe3A>, 2021.

[7] Katherine Crowson. CLIP guided diffusion HQ 256x256. <https://colab.research.google.com/drive/12a_Wrfi2_gwwAuN3VvMTwVMz9TfqctNj>, 2021.

[8] Katherine Crowson. CLIP Guided Diffusion 512x512, Secondary Model Method. <https://twitter.com/RiversHaveWings/status/1462859669454536711>, 2021.

[9] Katherine Crowson. v-diffusion. <https://github.com/crowsonkb/v-diffusion-pytorch>, 2021.

[10] Karan Desai and Justin Johnson. VirTex: Learning Visual Representations from Textual Annotations. arXiv:2006.06666, 2020.

[11] Prafulla Dhariwal and Alex Nichol. Diffusion Models Beat GANs on Image Synthesis. arXiv:2105.05233, 2021.

[12] Ming Ding, Zhuoyi Yang, Wenyi Hong, Wendi Zheng, Chang Zhou, Da Yin, Junyang Lin, Xu Zou, Zhou Shao, Hongxia Yang, and Jie Tang. CogView: Mastering Text-to-Image Generation via Transformers. arXiv:2105.13290, 2021.

[13] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. arXiv:2010.11929, 2020.

[14] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming Transformers for High-Resolution Image Synthesis. arXiv:2012.09841, 2020.

[15] Pierre Foret, Ariel Kleiner, Hossein Mobahi, and Behnam Neyshabur. Sharpness-Aware Minimization for Efficiently Improving Generalization. arXiv:2010.01412, 2020.

[16] Andreas Furst, Elisabeth Rumetshofer, Viet Thuong Tran, Hubert Ramsauer, Fei Tang, Johannes Lehner, D P Kreil, Michael K Kopp, Gunter Klambauer, Angela Bitto-Nemling, and Sepp Hochreiter. CLOOB: Modern Hopfield Networks with InfoLOOB Outperform CLIP, 2022. URL <https://openreview.net/forum?id=qw674L9PfQE>.

[17] Oran Gafni, Adam Polyak, Oron Ashual, Shelly Sheynin, Devi Parikh, and Yaniv Taigman. Make-A- Scene: Scene-Based Text-to-Image Generation with Human Priors. arXiv:2203.13131, 2022.

[18] Rinon Gal, Or Patashnik, Haggai Maron, Gal Chechik, and Daniel Cohen-Or. StyleGAN-NADA: CLIP-Guided Domain Adaptation of Image Generators. arXiv:2108.00946, 2021.

[19] Federico A. Galatolo, Mario G. C. A. Cimino, and Gigliola Vaglini. Generating images from caption and vice versa via CLIP-Guided Generative Latent Space Search. arXiv:2102.01645, 2021.

[20] Gabriel Goh, Nick Cammarata t, Chelsea Voss t, Shan Carter, Michael Petrov, Ludwig Schubert, Alec Radford, and Chris Olah. Multimodal Neurons in Artificial Neural Networks. Distill, 2021. doi: 10.23915/distill.00030. <https://distill.pub/2021/multimodal-neurons>.

[21] Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative Adversarial Networks. arXiv:1406.2661, 2014.

[22] Shuyang Gu, Dong Chen, Jianmin Bao, Fang Wen, Bo Zhang, Dongdong Chen, Lu Yuan, and Baining Guo. Vector Quantized Diffusion Model for Text-to-Image Synthesis. arXiv:2111.14822, 2021.

[23] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium. Advances in Neural Information Processing Systems 30 (NIPS 2017), 2017.

[24] Jonathan Ho and Tim Salimans. Classifier-Free Diffusion Guidance. In NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications, 2021. URL <https://openreview.net/forum?id=qw8AKxfYbI>.

[25] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising Diffusion Probabilistic Models. arXiv:2006.11239, 2020.

[26] Jonathan Ho, Chitwan Saharia, William Chan, David J. Fleet, Mohammad Norouzi, and Tim Salimans. Cascaded Diffusion Models for High Fidelity Image Generation. arXiv:2106.15282, 2021.

[27] Diederik P. Kingma and Jimmy Ba. Adam: A Method for Stochastic Optimization. arXiv:1412.6980, 2014.

[28] Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, and Piotr Dolldr. Microsoft COCO: Common Objects in Context. arXiv:1405.0312, 2014.

[29] Ilya Loshchilov and Frank Hutter. Decoupled Weight Decay Regularization. arXiv:1711.05101, 2017.

[30] Pamela Mishkin, Lama Ahmad, Miles Brundage, Gretchen Krueger, and Girish Sastry. DALLE 2 Preview - Risks and Limitations. 2022. URL <https://github.com/openai/dalle-2-preview/blob/main/system-card.md>.

[31] Norman Mu, Alexander Kirillov, David Wagner, and Saining Xie. SLIP: Self-supervision meets Language-Image Pre-training. arXiv:2112.12750, 2021.

[32] Ryan Murdock. The Big Sleep. <https://twitter.com/advadnoun/status/1351038053033406468>, 2021.

[33] Naila Murray, Luca Marchesotti, and Florent Perronnin. AVA: A large-scale database for aesthetic visual analysis. In 2012 IEEE Conference on Computer Vision and Pattern Recognition, pages 2408-2415, 2012. doi: 10.1109/CVPR.2012.6247954.

[34] Alex Nichol and Prafulla Dhariwal. Improved Denoising Diffusion Probabilistic Models. arXiv:2102.09672, 2021.

[35] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models. arXiv:2112.10741, 2021.

[36] Or Patashnik, Zongze Wu, Eli Shechtman, Daniel Cohen-Or, and Dani Lischinski. StyleCLIP: Text- Driven Manipulation of StyleGAN Imagery. arXiv:2103.17249, 2021.

[37] Karl Pearson. LIII. On lines and planes of closest fit to systems of points in space, November 1901. URL <https://doi.org/10.1080/14786440109462720>.

[38] Konpat Preechakul, Nattanat Chatthee, Suttisak Wizadwongsa, and Supasorn Suwajanakorn. Diffusion Autoencoders: Toward a Meaningful and Decodable Representation. arXiv:2111.15640, 2021.

[39] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning Transferable Visual Models From Natural Language Supervision. arXiv:2103.00020, 2021.

[40] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-Shot Text-to-Image Generation. arXiv:2102.12092, 2021.

[41] Ali Razavi, Aaron van den Oord, and Oriol Vinyals. Generating Diverse High-Fidelity Images with VQ-VAE-2. arXiv:1906.00446, 2019.

[42] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High- Resolution Image Synthesis with Latent Diffusion Models. arXiv:2112.10752, 2021.

[43] Chitwan Saharia, Jonathan Ho, William Chan, Tim Salimans, David J. Fleet, and Mohammad Norouzi. Image Super-Resolution via Iterative Refinement. arXiv:arXiv:2104.07636, 2021.

[44] Mert Bulent Sariyildiz, Julien Perez, and Diane Larlus. Learning Visual Representations with Caption Annotations. arXiv:2008.01392, 2020.

[45] Sheng Shen, Liunian Harold Li, Hao Tan, Mohit Bansal, Anna Rohrbach, Kai-Wei Chang, Zhewei Yao, and Kurt Keutzer. How Much Can CLIP Benefit Vision-and-Language Tasks? arXiv:2107.06383, 2021.

[46] Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep Unsupervised Learning using Nonequilibrium Thermodynamics. arXiv:1503.03585, 2015.

[47] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising Diffusion Implicit Models. arXiv:2010.02502, 2020.

[48] Yang Song and Stefano Ermon. Improved Techniques for Training Score-Based Generative Models. arXiv:2006.09011, 2020.

[49] Ming Tao, Hao Tang, Songsong Wu, Nicu Sebe, Xiao-Yuan Jing, Fei Wu, and Bingkun Bao. DF-GAN: Deep Fusion Generative Adversarial Networks for Text-to-Image Synthesis. arXiv:2008.05865, 2020.

[50] Arash Vahdat and Jan Kautz. NVAE: A Deep Hierarchical Variational Autoencoder. arXiv:2007.03898, 2020.

[51] Arash Vahdat, Karsten Kreis, and Jan Kautz. Score-based Generative Modeling in Latent Space. In Neural Information Processing Systems (NeurIPS), 2021.

[52] Aaron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu. Neural Discrete Representation Learning. arXiv:1711.00937, 2017.

[53] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention Is All You Need. arXiv:1706.03762, 2017.

[54] Zihao Wang, Wei Liu, Qian He, Xinglong Wu, and Zili Yi. CLIP-GEN: Language-Free Training of a Text-to-Image Generator with CLIP. arXiv:2203.00386, 2022.

[55] Weihao Xia, Yulun Zhang, Yujiu Yang, Jing-Hao Xue, Bolei Zhou, and Ming-Hsuan Yang. GAN Inversion: A Survey. arXiv:2101.05278, 2021.

[56] Tao Xu, Pengchuan Zhang, Qiuyuan Huang, Han Zhang, Zhe Gan, Xiaolei Huang, and Xiaodong He. AttnGAN: Fine-Grained Text to Image Generation with Attentional Generative Adversarial Networks. arXiv:1711.10485, 2017.

[57] Hui Ye, Xiulong Yang, Martin Takac, Rajshekhar Sunderraman, and Shihao Ji. Improving Text-to-Image Synthesis Using Contrastive Learning. arXiv:2107.02423, 2021.

[58] Han Zhang, Jing Yu Koh, Jason Baldridge, Honglak Lee, and Yinfei Yang. Cross-Modal Contrastive Learning for Text-to-Image Generation. arXiv:2101.04702, 2021.

[59] Kai Zhang, Jingyun Liang, Luc Van Gool, and Radu Timofte. Designing a Practical Degradation Model for Deep Blind Image Super-Resolution. 2021 IEEE/CVF International Conference on Computer Vision (ICCV), Oct 2021. doi: 10.1109/iccv48922.2021.00475. URL <http://dx.doi.org/10.1109/ICCV48922.2021.00475>.

[60] Yuhao Zhang, Hang Jiang, Yasuhide Miura, Christopher D. Manning, and Curtis P. Langlotz. Contrastive Learning of Medical Visual Representations from Paired Images and Text. arXiv:2010.00747, 2020.

[61] Yufan Zhou, Ruiyi Zhang, Changyou Chen, Chunyuan Li, Chris Tensmeyer, Tong Yu, Jiuxiang Gu, Jinhui Xu, and Tong Sun. LAFITE: Towards Language-Free Training for Text-to-Image Generation. arXiv:2111.13792, 2021.

[62] Jun-Yan Zhu, Philipp Krahenbuhl, Eli Shechtman, and Alexei A. Efros. Generative Visual Manipulation on the Natural Image Manifold. arXiv:1609.03552, 2016.

[63] Minfeng Zhu, Pingbo Pan, Wei Chen, and Yi Yang. DM-GAN: Dynamic Memory Generative Adversarial Networks for Text-to-Image Synthesis. arXiv:1904.01310, 2019.

## A Linear Probes for Evaluations

For our evaluations, we leverage two new linear probes on top of a CLIP ViT-L/14 [13] model. To automate aesthetic quality evaluations, we follow the procedure used by Crowson [6], training a linear regression model on images and mean ratings from the AVA dataset [33]. To reduce the cost of hyperparameter sweeps before conducting human evaluations, we train a logistic regression model to predict win probabilities between pairs of images. To train this model, we used 15,000 pairwise image comparisons gathered from all of our previous human evaluations. For each comparison i, we computed CLIP image embeddings xi and yi for the two images in the pair. We then trained a linear model $f(x)$ such that $1/(1 + \exp (f (x_i) — f (y_i)))$ approximates the probability that a human prefers the image for yi . This can be reduced to a logistic regression problem with inputs equal to $y_i - x_i$.

## B Error Bars for Human Evaluation

When computing error bars for human evaluations, we use the normal approximation interval with $p = 0.95$. We expect the normal approximation to be accurate for such a large sample size of $n = 1000$.

## C Training Details

The unCLIP models used for the experiments in this paper were trained with the hyperparameters described below, unless otherwise noted. We additionally trained a production version of unCLIP using similarly sized models but with modified architectures and trained for longer; we include changes to accommodate product and safety requirements (e.g. inpainting, preventing unwanted memorization), and train on a larger dataset that is filtered for aesthetic quality and safety. We report model and training hyperparameters for the paper models in Table 3. All models were trained using Adam [27] with corrected weight decay [29] and momentum $\beta_1 = 0.9$.

Our CLIP model uses a ViT-H/16 [13] image encoder that consumes $256 \times 256$ resolution images, and has width 1280 with 32 Transformer [53] blocks. The text encoder also follows the architecture described in Radford et al. [39]: it is a Transformer [53] with a causal attention mask, with width 1024 and 24 Transformer blocks. Both models are trained with learning rate $3 \times 10^{-4}$ and SAM [15] with $ρ = 0.1$, where the perturbations are applied independently by the replicas, each of which uses batch size 64. The remaining hyperparameters are the same as those reported in Radford et al. [39].

When training the encoder, we sample from the CLIP [39] and DALL-E [40] datasets (approximately 650M images in total) with equal probability. When training the decoder, upsamplers, and prior, we use only the DALL-E dataset [40] (approximately 250M images). Incorporating the noisier CLIP dataset while training the generative stack negatively impacted sample quality in our initial evaluations.

Our decoder architecture is the 3.5 billion parameter GLIDE model, with the same architecture and diffusion hyperparameters as in Nichol et al. [35]. We train with learned sigma and sample with 250 strided sampling steps as in Nichol and Dhariwal [34].

We use the ADMNet architecture [11] for the upsamplers. In the first upsampling stage, we use a cosine noising schedule, 320 channels and a depth of 3 resblocks per resolution inside the ADMNet. We also apply gaussian blur (kernel size 3, sigma 0.6) as described in Saharia et al. [43]. In the second upsampling stage, we use a linear noising schedule, 192 channels, a depth of 2 resblocks per resolution, and train with the BSR degradation from Rombach et al. [42]. Neither upsampler uses attention. To reduce inference time, we use DDIM [47] and manually tune the number of steps, with 27 steps for $256 \times 256$ model, and 15 steps for the $1024 \times 1024$ model.

For the AR prior, we use a Transformer text encoder with width 2048 and 24 blocks and a decoder with a causal attention mask, width 1664, and 24 blocks. For the diffusion prior, we use a Transformer with width 2048 and 24 blocks, and sample with Analytic DPM [2] with 64 strided sampling steps. To reuse hyperparameters tuned for diffusion noise schedules on images from Dhariwal and Nichol [11], we scale the CLIP embedding inputs by 17.2 to match the empirical variance of RGB pixel values of ImageNet images scaled to $[-1,1]$.

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Hierarchical%20Text-Conditional%20Image%20Generation%20with%20CLIP%20Latents/Table%203.png"/></div>

Table 3: Hyperparameters for the models 

## D Random samples

In Figures 18, 19 and 20 we show random samples from our production model for some of the prompts from Figure 1.

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Hierarchical%20Text-Conditional%20Image%20Generation%20with%20CLIP%20Latents/Fig%2018.png"/></div>

Figure 18: Random samples from unCLIP for prompt “Vibrant portrait painting of Salvador Dali with a robotic half face” 

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Hierarchical%20Text-Conditional%20Image%20Generation%20with%20CLIP%20Latents/Fig%2019.png"/></div>

Figure 19: Random samples from unCLIP for prompt “A close up of a handpalm with leaves growing from it.” 

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Hierarchical%20Text-Conditional%20Image%20Generation%20with%20CLIP%20Latents/Fig%2020.png"/></div>

Figure 20: Random samples from unCLIP for prompt “A teddybear on a skateboard in Times Square.”
