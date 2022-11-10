---
title: 【《nature》】Advancing mathematics by guiding human intuition with AI 通过用人工智能引导人类直觉来推进数学的发展
date: 2022-11-07 07:00:00 +/-0800
categories: [论文,深度学习]
tags: [深度学习,《nature》]     # TAG names should always be lowercase 标记名称应始终为小写
math: true
---

Alex Davies1 $^{1✉}$, Petar Veličković$^1$, Lars Buesing$^1$, Sam Blackwell$^1$, Daniel Zheng$^1$, Nenad Tomasev$^1$, Richard Tanburn$^1$, Peter Battaglia$^1$, Charles Blundell$^1$, András Juhász$^2$,
Marc Lackenby$^2$, Geordie Williamson$^3$, Demis Hassabis$^2$ & Pushmeet Kohli$^{1✉}$

>$^1$ DeepMind, London, UK. $^2$University of Oxford, Oxford, UK.$^3$University of Sydney, Sydney, New South Wales, Australia. $^✉$e-mail: adavies@deepmind.com;pushmeet@deepmind.com

*The practice of mathematics involves discovering patterns and using these to formulate and prove conjectures, resulting in theorems. Since the 1960s, mathematicians have used computers to assist in the discovery of patterns and formulation of conjectures[1], most famously in the Birch and Swinnerton-Dyer conjecture[2], a Millennium Prize Problem[3]. Here we provide examples of new fundamental results in pure mathematics that have been discovered with the assistance of machine learning—demonstrating a method by which machine learning can aid mathematicians in discovering new conjectures and theorems. We propose a process of using machine learning to discover potential patterns and relations between mathematical objects, understanding them with attribution techniques and using these observations to guide intuition and propose conjectures. We outline this machine-learning-guided framework and demonstrate its successful application to current research questions in distinct areas of pure mathematics, in each case showing how it led to meaningful mathematical contributions on important open problems: a new connection between the algebraic and geometric structure of knots, and a candidate algorithm predicted by the combinatorial invariance conjecture for symmetric groups[4]. Our work may serve as a model for collaboration between the fields of mathematics and artificial intelligence (AI) that can achieve surprising results by leveraging the respective strengths of mathematicians and machine learning.*

*数学实践涉及发现模式并使用这些模式来制定和证明猜想，从而产生定理。自 1960 年代以来，数学家一直使用计算机来帮助发现猜想的模式和公式[1]，最著名的是 Birch 和 Swinnerton-Dyer 猜想[2]，这是一个千禧年大奖 [3]【美国克雷数学研究所提出的七个公开的问题】。在这里，我们提供了在机器学习的帮助下发现的纯数学新基本结果的例子——展示了一种机器学习可以帮助数学家发现新猜想和定理的方法。我们提出了一个使用机器学习来发现潜在模式和数学对象之间关系的过程，通过归因技术来理解它们，并使用这些观察来指导直觉并提出猜想。我们概述了这个机器学习引导的框架，并展示了它在纯数学不同领域的当前研究问题中的成功应用，在每种情况下都展示了它如何导致对重要的开放问题做出有意义的数学贡献：一个新的关于结的代数和几何结构之间的关系，以及由对称组的组合不变猜想而诞生的预测的候选算法[4]。我们的工作可以作为数学和人工智能 (AI) 领域之间合作的模型，通过利用数学家和机器学习的各自优势，可以取得令人惊讶的结果。*

One of the central drivers of mathematical progress is the discovery of patterns and formulation of useful conjectures: statements that are suspected to be true but have not been proven to hold in all cases. Mathematicians have always used data to help in this process—from the early hand-calculated prime tables used by Gauss and others that led to the prime number theorem[5], to modern computer-generated data[1,5] in cases such as the Birch and Swinnerton-Dyer conjecture[2]. The introduction of computers to generate data and test conjectures afforded mathematicians a new understanding of problems that were previously inaccessible[6], but while computational techniques have become consistently useful in other parts of the mathematical process[7,8], artificial intelligence (AI) systems have not yet established a similar place. Prior systems for generating conjectures have either contributed genuinely useful research conjectures[9] via methods that do not easily generalize to other mathematical areas[10], or have demonstrated novel, general methods for finding conjectures[11] that have not yet yielded mathematically valuable results.

数学进步的核心驱动力之一是发现模式和提出有用的猜想：所谓的猜想即可能是真，但并未在所有情况下都都被证明成立的一些命题。数学家一直使用数据来帮助这个过程——从早期的高斯和其他人使用的手工计算素数表来推导素数定理[5]，到现代计算机生成的数据[1,5]，例如Birch 和 Swinnerton-Dyer 猜想[2]。引入计算机来生成数据和测试猜想使数学家对以前无法解决的问题有了新的理解[6]，但是虽然计算技术在数学过程的其他部分变得始终有用[7,8]，但人工智能 (AI) 系统已经还没有建立类似的地方。用于生成猜想的先前系统要么贡献了真正有用的研究猜想 [9]但不容易推广到其他数学领域 [10]，要么或者已经证明了寻找猜想的新的一般方法[11]，但尚未产生有数学价值的结果。【找到了方法但结果不好，或者结果很好但是方法的一般性不足】

AI, in particular the field of machine learning[12-14], offers a collection of techniques that can effectively detect patterns in data and has increasingly demonstrated utility in scientific disciplines[15]. In mathematics, it has been shown that AI can be used as a valuable tool by finding counterexamples to existing conjectures[16], accelerating calculations[17], generating symbolic solutions[18] and detecting the existence of structure in mathematical objects[19]. In this work, we demonstrate that AI can also be used to assist in the discovery of theorems and conjectures at the forefront of mathematical research. This extends work using supervised learning to find patterns[20-24] by focusing on enabling mathematicians to understand the learned functions and derive useful mathematical insight. We propose a framework for augmenting the standard mathematician’s toolkit with powerful pattern recognition and interpretation methods from machine learning and demonstrate its value and generality by showing how it led us to two fundamental new discoveries, one in topology and another in representation theory. Our contribution shows how mature machine learning methodologies can be adapted and integrated into existing mathematical workflows to achieve novel results.

人工智能，特别是机器学习领域[12-14]，提供了一个可以有效检测数据模式的技术集合，并在科学学科中越来越显示出实用性[15]。在数学领域，已经证明人工智能可以通过寻找现有猜想的反例[16]、加速计算[17]、生成符号解[18]和检测数学对象中存在的结构[19]，作为一种有价值的工具。在这项工作中，我们证明人工智能也可以用来协助发现数学研究前沿的定理和猜想。这扩展了使用监督学习来寻找模式的工作[20-24]，重点是使数学家能够理解所学的函数并得出有用的数学见解。我们提出了一个框架，用机器学习中强大的模式识别和解释方法来增强标准数学家的工具包，并通过展示它如何使我们获得两个基本的新发现，一个是拓扑学，另一个是表示理论，来证明其价值和通用性。我们的贡献显示了成熟的机器学习方法是如何被调整并整合到现有的数学工作流程中以获得新的结果。

## Guiding mathematical intuition with AI

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Advancing%20mathematics%20by%20guiding%20human/Fig%201.png"/></div>

**Fig. 1 | Flowchart of the framework.** The process helps guide a mathematician’s intuition about a hypothesized function $f$, by training a machine learning model to estimate that function over a particular distribution of data $P_Z$. The insights from the accuracy of the learned function $\hat{f}$ and attribution techniques applied to it can aid in the understanding of the problem and the construction of a closed-form  $f'$. The process is iterative and interactive, rather than a single series of steps.\
**图。 1 | 框架流程图。** 该过程通过训练机器学习模型来估计特定数据分布 $P_Z$ 上的函数，从而帮助指导数学家对假设函数 $f$ 的直觉。 从学习函数 $\hat{f}$ 的准确性和应用于它的归因技术的见解可以帮助理解问题和构建封闭形式的 $f'$。 该过程是迭代和交互式的，而不是一系列步骤。

A mathematician’s intuition plays an enormously important role in mathematical discovery—“It is only with a combination of both rigorous formalism and good intuition that one can tackle complex mathematical problems”[25]. The following framework, illustrated in Fig. 1, describes a general method by which mathematicians can use tools from machine learning to guide their intuitions concerning complex mathematical objects, verifying their hypotheses about the existence of relationships and helping them understand those relationships. We propose that this is a natural and empirically productive way that these  well-understood techniques in statistics and machine learning can be used as part of a mathematician’s work.

数学家的直觉在数学发现中起着极其重要的作用--"只有将严格的形式主义和良好的直觉相结合，才能解决复杂的数学问题"[25]。上面的框架，如图1所示，描述了一种一般的方法，数学家可以利用机器学习的工具来指导他们关于复杂数学对象的直觉，验证他们关于存在关系的假设，并帮助他们理解这些关系。我们提出，这是一种自然的、富有经验的方式，这些在统计学和机器学习中被充分理解的技术可以作为数学家工作的一部分来使用。

Concretely, it helps guide a mathematician’s intuition about the relationship between two mathematical objects $X(z)$ and $Y(z)$ associated with $z$ by identifying a function $\hat{f}$ such that $\hat{f} (X(z)) \approx Y(z)$ and analysing it to allow the mathematician to understand properties of the relationship. As an illustrative example: let $z$ be convex polyhedra, $X(z) \isin \mathbb{Z}^2 \times \mathbb{R}^2$ be the number of vertices and edges of $z$, as well as the volume and surface area, and $Y(z) \isin \mathbb{Z}$ be the number of faces of  $z$.Euler’s formula states that there is an exact relationship between $X(z)$ and $Y(z)$ in this case:$X(z) \cdot (-1,1,0,0) + 2= Y(z)$. In this simple example, among many other ways, the relationship could be rediscovered by the traditional methods of data-driven conjecture generation[1]. However, for $X(z)$ and $Y(z)$ in higher-dimensional spaces, or of more complex types, such as graphs, and for more complicated, nonlinear $\hat{f}$, this approach is either less useful or entirely infeasible.

具体来说，它有助于引导数学家对两个数学对象$X(z)$和$Y(z)$之间的关系的直觉，通过确定一个函数$\hat{f}$，使得$\hat{f}(X(z))$ 逼近$Y(z)$，并对其进行分析，使数学家能够理解这种关系的属性。举个例子：让$z$是凸多面体，$X(z)\isin \mathbb{Z}^2 \times \mathbb{R}^2$是$z$的顶点和边的数量，以及体积和表面积，$Y(z) \isin \mathbb{Z}$是$z$的面的数量。 欧拉公式指出，在这种情况下，$X(z)$和$Y(z)$之间存在着精确的关系：$X(z)\cdot(-1,1,0,0)+2=Y(z)$。在这个简单的例子中，在许多其他方式中，这种关系可以通过数据驱动猜想生成的传统方法重新发现[1]。然而，对于高维空间中的$X(z)$和$Y(z)$，或更复杂的类型，如图，以及更复杂的非线性$\hat{f}$，这种方法要么不太有用，要么完全不可行。

The framework helps guide the intuition of mathematicians in two ways: by verifying the hypothesized existence of structure/patterns in mathematical objects through the use of supervised machine learning; and by helping in the understanding of these patterns through the use of attribution techniques.

该框架以两种方式帮助指导数学家的直觉：通过使用监督机器学习验证数学对象中假设存在的结构/模式；以及通过使用归因技术【进行特征选择】帮助理解这些模式。

In the supervised learning stage, the mathematician proposes a hypothesis that there exists a relationship between $X(z)$ and $Y(z)$. By generating a dataset of $X(z)$ and $Y(z)$ pairs, we can use supervised learning to train a function $\hat{f}$ that predicts $Y(z)$, using only $X(z)$ as input. The key contributions of machine learning in this regression process are the broad set of possible nonlinear functions that can be learned given a sufficient amount of data. If $\hat{f}$ is more accurate than would be expected by chance, it indicates that there may be such a relationship to explore. If so, attribution techniques can help in the understanding of the learned function $\hat{f}$ sufficiently for the mathematician to conjecture a candidate $f'$. Attribution techniques can be used to understand which aspects of f are relevant for predictions of $Y(z)$. For example, many attribution techniques aim to quantify which component of $X(z)$ the function $\hat{f}$ is sensitive to. The attribution technique we use in our work, gradient saliency, does this by calculating the derivative of outputs of $\hat{f}$, with respect to the inputs. This allows a mathematician to identify and prioritize aspects of the problem that are most likely to be relevant for the relationship. This iterative process might need to be repeated several times before a viable conjecture is settled on. In this process, the mathematician can guide the choice of conjectures to those that not just fit the data but also seem interesting, plausibly true and, ideally, suggestive of a proof strategy.

在监督学习阶段，数学家提出了一个假设，即 $X(z)$ 和 $Y(z)$ 之间存在关系。通过生成 $X(z)$ 和 $Y(z)$ 对的数据集，我们可以使用监督学习来训练预测 $Y(z)$ 的函数 $\hat{f}$，仅使用 $X( z)$ 作为输入。机器学习在这个回归过程中的主要贡献是可以在给定足够数量的数据的情况下学习广泛的可能非线性函数。如果 $\hat{f}$ 比随机的更准确，则表明可能存在需要探索的这种关系。如果是这样，归因技术可以帮助理解学习函数 $\hat{f}$，足以让数学家推测候选 $f'$。归因技术可用于了解 f 的哪些方面与 $Y(z)$ 的预测相关。例如，许多归因技术旨在量化函数 $\hat{f}$ 对 $X(z)$ 的哪个分量敏感。我们在工作中使用的归因技术是梯度显着性，通过计算 $\hat{f}$ 的输出相对于输入的导数来做到这一点。【计算输入关于输出的梯度，梯度大的这个特征重要，梯度小的这个特征不重要】【一个特征重要的话，对输出的结果影响较大】这允许数学家识别和优先考虑问题中最可能与关系相关的方面。在确定一个可行的猜想之前，这个迭代过程可能需要重复几次。在这个过程中，数学家可以将猜想的选择引导到那些不仅适合数据而且看起来很有趣、看似正确并且理想情况下暗示证明策略的猜想。

Conceptually, this framework provides a ‘test bed for intuition’— quickly verifying whether an intuition about the relationship between two quantities may be worth pursuing and, if so, guidance as to how they may be related. We have used the above framework to help mathematicians to obtain impactful mathematical results in two cases—discovering and proving one of the first relationships between algebraic and geometric invariants in knot theory and conjecturing a resolution to the combinatorial invariance conjecture for symmetric groups[4], a well-known conjecture in representation theory. In each area, we demonstrate how the framework has successfully helped guide the mathematician to achieve the result. In each of these cases, the necessary models can be trained within several hours on a machine with a single graphics processing unit.

从概念上讲，这个框架提供了一个“直觉测试平台”——快速验证关于两个量之间关系的直觉是否值得追求，如果是，则指导它们如何相关。 我们已经使用上述框架来帮助数学家在两种情况下获得有影响力的数学结果——发现并证明纽结理论中代数和几何不变量之间的第一个关系，以及推测对称群组合不变性猜想的解决方案[4]， 表示论中的一个著名猜想。 在每个领域，我们都展示了该框架如何成功地帮助指导数学家取得成果。 在每种情况下，都可以在具有单个图形处理单元的机器上在几个小时内训练必要的模型。

## Topology 拓扑学

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Advancing%20mathematics%20by%20guiding%20human/Fig%202.png"/></div>

**Fig. 2 | Examples of invariants for three hyperbolic knots.** We hypothesized that there was a previously undiscovered relationship between the geometric and algebraic invariants. \
**图.2 | 三个双曲结的不变量示例。** 我们假设几何和代数不变量之间存在先前未被发现的关系

Low-dimensional topology is an active and influential area of mathematics. Knots, which are simple closed curves in $\mathbb{R}^3$, are one of the key objects that are studied, and some of the subject’s main goals are to classify them, to understand their properties and to establish connections with other fields. One of the principal ways that this is carried out is through invariants, which are algebraic, geometric or numerical quantities that are the same for any two equivalent knots. These invariants are derived in many different ways, but we focus on two of the main categories: hyperbolic invariants and algebraic invariants. These two types of invariants are derived from quite different mathematical disciplines, and so it is of considerable interest to establish connections between them. Some examples of these invariants for small knots are shown in Fig. 2. A notable example of a conjectured connection is the volume conjecture[26], which proposes that the hyperbolic volume of a knot (a geometric invariant) should be encoded within the asymptotic behaviour of its coloured Jones polynomials (which are algebraic invariants).

低维拓扑是一个活跃且有影响力的数学领域。结是 $\mathbb{R}^3$ 中的简单闭合曲线，是研究的关键对象之一，一些主题的主要目标是对它们进行分类，了解它们的属性并与其他对象【其他领域的物体】建立联系字段。研究该领域的主要方式之一是通过不变量，它们可以是来自代数、几何或数值量，对于任何两个等效节点这些不变量都是相同的。这些不变量以许多不同的方式派生，但我们主要关注两个主要类别：双曲线不变量和代数不变量。这两种类型的不变量源自完全不同的数学学科，因此在它们之间建立联系具有相当大的意义。小结的这些不变量的一些示例如图 2 所示。一个著名的猜想关于这个方面的研究是体积猜想[26]，它提出结的双曲线体积（几何不变量）可以由渐近线内编码其彩色琼斯多项式（代数不变量）的得出。

Our hypothesis was that there exists an undiscovered relationship between the hyperbolic and algebraic invariants of a knot. A supervised learning model was able to detect the existence of a pattern between a large set of geometric invariants and the signature $\sigma(K)$, which is known to encode important information about a knot K, but was not previously known to be related to the hyperbolic geometry. The most relevant features identified by the attribution technique, shown in Fig. 3a, were three invariants of the cusp geometry, with the relationship visualized partly in Fig. 3b. Training a second model with $X(z)$ consisting of only these measurements achieved a very similar accuracy, suggesting that they are a sufficient set of features to capture almost all of the effect of the geometry on the signature. These three invariants were the real and imaginary parts of the meridional translation $\mu$ and the longitudinal translation $\lambda$ There is a nonlinear, multivariate relationship between these quantities and the signature. Having been guided to focus on these invariants, we discovered that this relationship is best understood by means of a new quantity, which is linearly related to the signature.

我们的假设是，在一个结的双曲和代数不变量之间存在着一种未被发现的关系。一个监督学习模型能够检测出一大组几何不变量和签名$sigma(K)$之间存在着联系【signature $\sigma(K)$与几何很多的特征存在联系】，已知它编码了一个结K的重要信息，但以前并不知道它与双曲几何有关。图3a所示的归因技术确定的最相关的特征是尖峰几何的三个不变量，其关系在图3b中部分地被可视化。用仅由这些测量值组成的$X(z)$【只用这三个特征】训练第二个模型，取得了非常相似的准确性，这表明它们是一套足够的特征，几乎可以捕捉到几何形状对签名的所有影响。这三个不变量是经向平移$mu$的实部和虚部以及经向平移$lambda$ 这些数量和特征之间存在着非线性、多变量的关系。在被引导关注这些不变量后，我们发现这种关系最好通过一个新的量来理解，这个量与签名有线性关系。

We introduce the 'natural slope’, defined to be slope $(K) = {\rm Re}(\lambda/\mu)$, where Re denotes the real part. It has the following geometric interpretation. One can realize the meridian curve as a geodesic $\gamma$ on the Euclidean torus. If one fires off a geodesic $\gamma^⊥$ from this orthogonally, it will eventually return and hit y at some point. In doing so, it will have travelled along a longitude minus some multiple of the meridian. This multiple is the natural slope. It need not be an integer, because the endpoint of $\gamma^⊥$ might not be the same as its starting point. Our initial conjecture relating natural slope and signature was as follows.

我们引入“自然斜率”，定义为斜率 $(K) = {\rm Re}(\lambda/\mu)$，其中 Re 表示实部。 它具有以下几何解释。 可以将子午线曲线理解为欧几里得环面上的测地线$\gamma$。 如果一个人由此正交发射测地线 $\gamma^⊥$，它最终会返回并在某个点击中 y。 在这样做时，它将沿着经度减去一些子午线。 这个倍数是自然坡度。 它不必是整数，因为 $\gamma^⊥$ 的终点可能与其起点不同。 我们最初关于自然坡度和特征的猜想如下。

Conjecture: There exist constants $c_1$ and $c_2$ such that, for every hyperbolic knot $K$,

猜想：存在常数 $c_1$ 和 $c_2$ 使得对于每个双曲结 $K$，

$$
| 2 \sigma(K) - {\rm slope}(K) |< {\rm c_1vol}(K) + c_2 \tag{1}
$$

While this conjecture was supported by an analysis of several large datasets sampled from different distributions, we were able to construct counterexamples using braids of a specific form. Subsequently, we were able to establish a relationship between ${\rm slope}(K)$, signature $\sigma(K)$, volume ${\rm vol}(K)$ and one of the next most salient geometric invariants, the injectivity radius ${\rm inj}(K)$ (ref.[27]).

虽然对从不同分布采样的几个大型数据集的分析支持了这一猜想，但我们能够使用特定形式的辫子构建反例【出现了反例】。 随后，我们能够在 ${\rm 斜率}(K)$、签名 $\sigma(K)$、体积 ${\rm vol}(K)$ 和下一个最显着的几何不变量之一之间建立关系 , 内射性半径 ${\rm inj}(K)$ (ref.[27])。

Theorem: There exists a constant $c$ such that, for any hyperbolic knot $K$,

$$
|2 \sigma (K)-{\rm slope}( K )|   \leq {\rm cvol}(K) {\rm inj}( K )^{-3} \tag{2}
$$

It turns out that the injectivity radius tends not to get very small, even for knots of large volume. Hence, the term ${\rm inj}( K )^{-3}$ tends not to get very large in practice. However, it would clearly be desirable to have a theorem that avoided the dependence on ${\rm inj}( K )^{-3}$, and we give such a result that instead relies on short geodesics, another of the most salient features, in the Supplementary Information. Further details and a full proof of the above theorem are available in ref.[27]. Across the datasets we generated, we can place a lower bound of $c \geq 0.23392$, and it would be reasonable to conjecture that $c$ is at most 0.3, which gives a tight relationship in the regions in which we have calculated.

事实证明，即使对于大体积的结，注入性半径也不会变得非常小。因此，术语${\rm inj}( K )^{-3}$在实践中往往不会变得非常大。然而，显然最好能有一个定理来避免对${\rm inj}( K )^{-3}$的依赖，我们在补充资料中给出了这样一个结果，它依赖于短测地线，这是另一个最突出的特征。上述定理的进一步细节和完整证明见参考文献[27]。在我们生成的整个数据集中，我们可以把$c\geq 0.23392$作为下限，有理由猜测$c$最多为0.3，这在我们计算的区域中给出了一个紧密的关系。

The above theorem is one of the first results that connect the algebraic and geometric invariants of knots and has various interesting applications. It directly implies that the signature controls the non-hyperbolic Dehn surgeries on the knot and that the natural slope controls the genus of surfaces in $\mathbb{R}_+^4$ whose boundary is the knot. We expect that this newly discovered relationship between natural slope and signature will have many other applications in low-dimensional topology. It is surprising that a simple yet profound connection such as this has been overlooked in an area that has been extensively studied.

上述定理是连接结的代数和几何不变量的首批结果之一，并有各种有趣的应用。它直接意味着签名控制着结上的非双曲Dehn surgeries，自然斜率控制着结的边界为$mathbb{R}_+^4$的曲面的属。我们期望这个新发现的自然斜率和签名之间的关系将在低维拓扑学中有许多其他应用。令人惊讶的是，在一个已经被广泛研究的领域中，像这样一个简单而深刻的联系竟然被忽略了。【解释结果为什么是这个样子】

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Advancing%20mathematics%20by%20guiding%20human/Fig%203.png"/></div>

**Fig. 3 | Knot theory attribution. a,** Attribution values for each of the input $X(z)$. The features with high values are those that the learned function is most sensitive to and are probably relevant for further exploration. The 95% confidence interval error bars are across 10 retrainings of the model.**b**, Example visualization of relevant features—the real part of the meridional translation against signature, coloured by the longitudinal translation \
**a,**每个输入$X(z)$的归属值。值高的特征是那些学习函数最敏感的特征，可能与进一步探索有关。95%的置信区间误差条是在模型的10次再训练中出现的。**b**, 相关特征的可视化示例--经向平移的真实部分与特征的对比，由经向平移的颜色来显示。

## Representation theory 表示论/表征论

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Advancing%20mathematics%20by%20guiding%20human/Fig%204.png"/></div>

**Fig. 4 | Two example dataset elements, one from $S_5$ and one from $S_6$.** The combinatorial invariance conjecture states that the KL polynomial of a pair of permutations should be computable from their unlabelled Bruhat interval, but no such function was previously known. \
图4 **两个数据集元素的例子，一个来自$S_5$，一个来自$S_6$。**组合不变性猜想指出，一对排列组合的KL多项式应该可以从其未标记的Bruhat区间计算出来，但以前没有这样的函数。

Representation theory is the theory of linear symmetry. The building blocks of all representations are the irreducible ones, and understanding them is one of the most important goals of representation theory. Irreducible representations generalize the fundamental frequencies of Fourier analysis[28]. In several important examples, the structure of irreducible representations is governed by Kazhdan-Lusztig (KL) polynomials, which have deep connections to combinatorics, algebraic geometry and singularity theory. KL polynomials are polynomials attached to pairs of elements in symmetric groups (or more generally, pairs of elements in Coxeter groups). The combinatorial invariance conjecture is a fascinating open conjecture concerning KL polynomials that has stood for 40 years, with only partial progress[29]. It states that the KL polynomial of two elements in a symmetric group SN can be calculated from their unlabelled Bruhat interval[30], a directed graph. One barrier to progress in understanding the relationship between these objects is that the Bruhat intervals for non-trivial KL polynomials (those that are not equal to 1) are very large graphs that are difficult to develop intuition about. Some examples of small Bruhat intervals and their KL polynomials are shown in Fig. 4.

表征理论是关于线性对称的理论。所有表征的构件都是不可还原的，理解它们是表征理论最重要的目标之一。不可还原的表征概括了傅里叶分析的基频[28]。在几个重要的例子中，不可还原表征的结构受Kazhdan-Lusztig (KL)多项式的支配，它与组合学、代数几何和奇点理论有着深刻的联系。KL多项式是连接到对称群中的元素对上的多项式（或者更普遍的是连接到Coxeter群中的元素对上的多项式）。组合不变性猜想是关于KL多项式的一个迷人的开放性猜想，已经存在了40年，只有部分进展[29]。它指出，对称群SN中两个元素的KL多项式可以从它们的无标记的Bruhat区间[30]，即一个有向图中计算出来。在理解这些对象之间的关系方面取得进展的一个障碍是，非三阶KL多项式（不等于1的那些）的Bruhat区间是非常大的图，很难形成直观的认识。一些小Bruhat区间及其KL多项式的例子显示在图4中。

We took the conjecture as our initial hypothesis, and found that a supervised learning model was able to predict the KL polynomial from the Bruhat interval with reasonably high accuracy. By experimenting on the way in which we input the Bruhat interval to the network, it became apparent that some choices of graphs and features were particularly conducive to accurate predictions. In particular, we found that a subgraph inspired by prior work[31] may be sufficient to calculate the KL polynomial, and this was supported by a much more accurate estimated function.

我们把这个猜想作为我们的初始假设，并发现一个监督学习模型能够以相当高的精度预测布鲁哈特区间的KL多项式。通过试验我们向网络输入Bruhat区间的方式，我们发现一些图形和特征的选择特别有利于准确预测。特别是，我们发现受先前工作[31]启发的子图可能足以计算出KL多项式，而且这得到了更准确的估计函数的支持。

Further structural evidence was found by calculating salient subgraphs that attribution techniques determined were most relevant and analysing the edge distribution in these graphs compared to the original graphs. In Fig. 5a, we aggregate the relative frequency of the edges in the salient subgraphs by the reflection that they represent. It shows that extremal reflections (those of the form $(0, i)$ or $(i ,N-1)$ for $S_N$) appear more commonly in salient subgraphs than one would expect, at the expense of simple reflections (those of the form $(i, i +1)$), which is confirmed over many retrainings of the model in Fig. 5b. This is notable because the edge labels are not given to the network and are not recoverable from the unlabelled Bruhat interval. From the definition of KL polynomials, it is intuitive that the distinction between simple and non-simple reflections is relevant for calculating it; however, it was not initially obvious why extremal reflections would be overrepresented in salient subgraphs. Considering this observation led us to the discovery that there is a natural decomposition of an interval into two parts—a hypercube induced by one set of extremal edges and a graph isomorphic to an interval in $S_{N-1}$.

通过计算归因技术确定为最相关的突出子图，并分析这些图与原始图的边缘分布，可以发现进一步的结构证据。在图5a中，我们通过它们所代表的反射来汇总突出的子图中的边缘的相对频率。它表明极端反射（那些形式为$(0, i)$或$(i ,N-1)$的$S_N$）在突出子图中出现的频率比人们预期的要高，而牺牲了简单反射（那些形式为$(i, i +1)$的反射），这在图5b的模型的多次重新训练中得到了证实。这是值得注意的，因为边缘标签没有给网络，也不能从无标签的Bruhat区间恢复。从KL多项式的定义来看，简单反射和非简单反射之间的区别与计算它有关是很直观的；然而，最初并不明显，为什么极端反射会在突出的子图中被过度代表。考虑到这一观察，我们发现一个区间有一个自然的分解，即由一组极值边诱导的超立方体和一个与$S_{N-1}$中的区间同构的图。

<div align=center><img src="https://raw.githubusercontent.com/jiugexuan/image-repository/main/Papers/Advancing%20mathematics%20by%20guiding%20human/Fig%205.png"/></div>

**Fig. 5 | Representation theory attribution.a,** An example heatmap of the percentage increase in reflections present in the salient subgraphs compared with the average across intervals in the dataset when predicting $q^4$. **b**, The percentage of observed edges of each type in the salient subgraph for 10 retrainings of the model compared to 10 bootstrapped samples of the same size from the dataset. The error bars are 95% confidence intervals, and the significance level shown was determined using a two-sided two-sample t-test.$^*p$ < 0.05; **** p < 0.0001. **c**, Illustration for the interval $021435-240513 \in S_6$  of the interesting substructures that were discovered through the iterative process of hypothesis, supervised learning and attribution. The subgraph inspired by previous work[31] is highlighted in red, the hypercube in green and the decomposition component isomporphic to an interval in $S_{N-1}$ in blue.

The importance of these two structures, illustrated in Fig. 5c, led to a proof that the KL polynomial can be computed directly from the hypercube and $S_{N-1}$ components through a beautiful formula that is summarized in the Supplementary Information. A further detailed treatment of the mathematical results is given in ref.[32].

Theorem: Every Bruhat interval admits a canonical hypercube decom-position along its extremal reflections, from which the KL polynomial is directly computable.

该定理。每一个布鲁哈特区间都有一个沿其极值反射的典型超立方体解构位置，从中可以直接计算出KL多项式。

Remarkably, further tests suggested that all hypercube decompositions correctly determine the KL polynomial. This has been computationally verified for all of the $\sim 3 \times 10^6$ intervals in the symmetric groups up to $S_7$ and more than $1.3 \times 10^5$ non-isomorphic intervals sampled from the symmetric groups $S_8$ and $S_9$.

值得注意的是，进一步的测试表明，所有超立方体分解都能正确确定KL多项式。这一点对于对称组中所有$S_7以下的$3\times 10^6$区间以及从对称组$S_8和$S_9采样的超过1.3\times 10^5$的非同构区间都得到了计算验证。

Conjecture: The KL polynomial of an unlabelled Bruhat interval can be calculated using the previous formula with any hypercube decomposition.

猜想。无标签的Bruhat区间的KL多项式可以用前面的公式计算出任何超立方体分解。

This conjectured solution, if proven true, would settle the combinatorial invariance conjecture for symmetric groups. This is a promising direction as not only is the conjecture empirically verified up to quite large examples, but it also has a particularly nice form that suggests potential avenues for attacking the conjecture. This case demonstrates how non-trivial insights about the behaviour of large mathematical objects can be obtained from trained models, such that new structure can be discovered.

这个猜想的解决方案，如果被证明是真的，将解决对称群的组合不变性猜想。这是一个很有希望的方向，因为这个猜想不仅在相当大的例子中得到了经验上的验证，而且它还有一个特别好的形式，表明了攻击这个猜想的潜在途径。这个案例展示了如何从训练有素的模型中获得关于大型数学对象行为的非微观见解，从而发现新结构。

## Conclusion

In this work we have demonstrated a framework for mathematicians to use machine learning that has led to mathematical insight across two distinct disciplines: one of the first connections between the algebraic and geometric structure of knots and a proposed resolution to a long-standing open conjecture in representation theory. Rather than use machine learning to directly generate conjectures, we focus on helping guide the highly tuned intuition of expert mathematicians, yielding results that are both interesting and deep. It is clear that intuition plays an important role in elite performance in many human pursuits. For example, it is critical for top Go players and the success of AlphaGo (ref.[33]) came in part from its ability to use machine learning to learn elements of play that humans perform intuitively. It is similarly seen as critical for top mathematicians—Ramanujan was dubbed the Prince of Intuition34 and it has inspired reflections by famous mathematicians on its place in their field[35,36]. As mathematics is a very different, more cooperative endeavour than Go, the role of AI in assisting intuition is far more natural. Here we show that there is indeed fruitful space to assist mathematicians in this aspect of their work.

在这项工作中，我们展示了一个数学家使用机器学习的框架，该框架导致了跨越两个不同学科的数学洞察力：结的代数和几何结构之间的最早联系之一，以及对表示理论中一个长期存在的公开猜想的拟议解决方案。我们没有使用机器学习来直接产生猜想，而是专注于帮助指导专家数学家的高度调整的直觉，产生了既有趣又深刻的结果。很明显，直觉在许多人类追求的精英表现中起着重要作用。例如，它对顶级围棋选手至关重要，AlphaGo（参考文献[33]）的成功部分来自于它利用机器学习学习人类凭直觉下棋的元素的能力。同样，它也被视为顶级数学家的关键--拉马努扬被称为直觉王子34，它激发了著名数学家对其在其领域中的地位的思考[35,36]。由于数学是一种非常不同的，比围棋更有合作性的工作，人工智能在协助直觉方面的作用要自然得多。在此，我们表明，在协助数学家进行这方面的工作方面，确实存在着富有成效的空间。

Our case studies demonstrate how a foundational connection in a well-studied and mathematically interesting area can go unnoticed, and how the framework allows mathematicians to better understand the behaviour of objects that are too large for them to otherwise observe patterns in. There are limitations to where this framework will be useful—it requires the ability to generate large datasets of the representations of objects and for the patterns to be detectable in examples that are calculable. Further, in some domains the functions of interest may be difficult to learn in this paradigm. However, we believe there are many areas that could benefit from our methodology. More broadly, it is our hope that this framework is an effective mechanism to allow for the introduction of machine learning into mathematicians’work, and encourage further collaboration between the two fields.

我们的案例研究表明，在一个被充分研究的、数学上有趣的领域中，一个基础性的联系是如何被忽视的，以及这个框架是如何让数学家更好地理解那些大到他们无法观察到的物体的行为模式。这个框架在哪些方面是有用的--它需要有能力生成对象表征的大型数据集【局限性：需要大量的数据】，并且在可计算的例子中可以发现模式。此外，在某些领域，感兴趣的功能可能很难在这种模式下学习。【有些问题很难用机器学习表示】然而，我们相信有许多领域可以从我们的方法中受益。更广泛地说，我们希望这个框架是一个有效的机制，可以将机器学习引入到数学家的工作中，并鼓励这两个领域的进一步合作。

>【应该使用auto ml工具】
>【当遇到新问题的时候应该关心问题的本身，关系问题数据怎么来，而不是机器学习的模型，在已知的问题上提升机器学习的提升则需要更关心模型，关心算法的提升】

## Online content

Any methods, additional references, Nature Research reporting summaries, source data, extended data, supplementary information,acknowledgements, peer review information; details of author contributions and competing interests; and statements of data and code availability are available at <https://doi.org/10.1038/s41586-021-04086-x.>

1. Borwein, J. & Bailey, D. Mathematics by Experiment (CRC, 2008).
2. Birch, B. J. & Swinnerton-Dyer, H. P. F. Notes on elliptic curves. II. J. Reine Angew. Math.**1965**, 79-108 (1965).
3. Carlson, J. et al. The Millennium Prize Problems (American Mathematical Soc., 2006).
4. Brenti, F. Kazhdan-Lusztig polynomials: history, problems, and combinatorial invariance. Semin. Lothar. Combin. **49**, B49b (2002).
5. Hoche, R. Nicomachi Geraseni Pythagorei Introductionis Arithmeticae Libri 2 (In aedibus BG Teubneri, 1866).
6. Khovanov, M. Patterns in knot cohomology, I. Exp. Math. 12, 365-374 (2003).
7. Appel, K. I. & Haken, W. Every Planar Map Is Four Colorable Vol. 98 (American Mathematical Soc., 1989).
8. Scholze, P. Half a year of the Liquid Tensor Experiment: amazing developments Xena <https://xenaproject.wordpress.com/2021/06/05/half-a-year-of-the-liquid-tensor- experiment-amazing-developments/> (2021).
9. Fajtlowicz, S. in Annals of Discrete Mathematics Vol. 38 113-118 (Elsevier, 1988).
10. Larson, C. E. in DIMACS Series in Discrete Mathematics and Theoretical Computer Science Vol. 69 (eds Fajtlowicz, S. et al.) 297-318 (AMS & DIMACS, 2005).
11. Raayoni, G. et al. Generating conjectures on fundamental constants with the Ramanujan machine. Nature 590, 67-73 (2021).
12. MacKay, D. J. C. Information Theory, Inference and Learning Algorithms (Cambridge Univ. Press, 2003).
13. Bishop, C. M. Pattern Recognition and Machine Learning (Springer, 2006).
14. LeCun, Y., Bengio, Y. & Hinton, G. Deep learning. Nature 521, 436-444 (2015).
15. Raghu, M. & Schmidt, E. A survey of deep learning for scientific discovery. Preprint at <https://arxiv.org/abs/2003.11755> (2020).
16. Wagner, A. Z. Constructions in combinatorics via neural networks. Preprint at <https://arxiv.org/abs/2104.14516> (2021).
17. Peifer, D., Stillman, M. & Halpern-Leistner, D. Learning selection strategies in Buchberger’s algorithm. Preprint at <https://arxiv.org/abs/2005.01917> (2020).
18. Lample, G. & Charton, F. Deep learning for symbolic mathematics. Preprint at <https://arxiv.org/abs/1912.01412> (2019).
19. He, Y.-H. Machine-learning mathematical structures. Preprint at <https://arxiv.org/ abs/2101.06317> (2021).
20. Carifio, J., Halverson, J., Krioukov, D. & Nelson, B. D. Machine learning in the string landscape. J. High Energy Phys. 2017, 157 (2017).
21. Heal, K., Kulkarni, A. & Sertoz, E. C. Deep learning Gauss-Manin connections. Preprint at <https://arxiv.org/abs/2007.13786?> (2020).
22. Hughes, M. C. A neural network approach to predicting and computing knot invariants. Preprint at <https://arxiv.org/abs/1610.05744> (2016).
23. Levitt, J. S. F., Hajij, M. & Sazdanovic, R. Big data approaches to knot theory: understanding the structure of the Jones polynomial. Preprint at <https://arxiv.org/abs/1912.10086?> (2019).
24. Jejjala, V., Kar, A. & Parrikar, O. Deep learning the hyperbolic volume of a knot. Phys. Lett. B 799, 135033 (2019).
25. Tao, T. There’s more to mathematics than rigour and proofs Blog <https://terrytao.wordpress.com/career-advice/theres-more-to-mathematics-than-rigour-and-proofs/> (2016).
26. Kashaev, R. M. The hyperbolic volume of knots from the quantum dilogarithm. Lett. Math. Phys. 39, 269-275 (1997).
27. Davies, A., Lackenby, M., Juhasz, A. & Tomasev, N. The signature and cusp geometry of hyperbolic knots. Preprint at arxiv.org (in the press).
28. Curtis, C. W. & Reiner, I. Representation Theory of Finite Groups and Associative Algebras Vol. 356 (American Mathematical Soc., 1966).
29. Brenti, F., Caselli, F. & Marietti, M. Special matchings and Kazhdan-Lusztig polynomials. Adv. Math. 202, 555-601 (2006).
30. Verma, D.-N. Structure of certain induced representations of complex semisimple Lie algebras. Bull. Am. Math. Soc. 74, 160-166 (1968).
31. Braden, T. & MacPherson, R. From moment graphs to intersection cohomology.Math. Ann. 321, 533-551 (2001).
32. Blundell, C., Buesing, L., Davies, A., Velickovic, P. & Williamson, G. Towards combinatorial invariance for Kazhdan-Lusztig polynomials. Preprint at arxiv.org (in the press).
33. Silver, D. et al. Mastering the game of Go with deep neural networks and tree search. Nature 529, 484-489 (2016).
34. Kanigel, R. The Man Who Knew Infinity: a Life of the Genius Ramanujan (Simon and Schuster, 2016).
35. Poincare, H. The Value of Science： Essential Writings of Henri Poincare (Modern Library, 1907). 36. Hadamard, J. The Mathematician’s Mind (Princeton Univ. Press, 1997).

>**Publisher’s note** Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.
>**Open Access** This article is licensed under a Creative Commons Attribution 4.0 International License, which permits use, sharing, adaptation, distribution and reproduction in any medium or format, as long as you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons license, and indicate if changes were made. The images or other third party material in this article are included in the article’s Creative Commons license, unless indicated otherwise in a credit line to the material. If material is not included in the article’s Creative Commons license and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder. To view a copy of this license, visit <http://creativecommons.org/licenses/by/4.0/.>
© The Author(s) 2021

## Methods

### Framework

**Supervised learning.** In the supervised learning stage, the mathematician proposes a hypothesis that there exists a relationship between $X(z)$ and $Y(z)$. In this work we assume that there is no known function mapping from $X(z)$ to $Y(z)$, which in turn implies that $X$ is not invertible (otherwise there would exist a known function $Y \circ X^{-1}$). While there may still be value to this process when the function is known, we leave this for future work. To test the hypothesis that $X$ and $Y$ are related, we generate a dataset of $X(z)$, $Y(z)$ pairs, where $z$ is sampled from a distribution $P_Z$. The results of the subsequent stages will hold true only for the distribution $P_Z$, and not the whole space $Z$. Initially, sensible choices for $P_Z$ would be, for example, uniformly over the first $N$ items for $Z$ with a notion of ordering, or uniformly at random where possible. In subsequent iterations, $P_Z$ may be chosen to understand the behaviour on different parts of the space $Z$ (for example, regions of $Z$ that may be more likely to provide counterexamples to a particular hypothesis). To first test whether a relation between $X(z)$ and $Y(z)$ can be found, we use supervised learning to train a function $\hat{f}$ that approximately maps $X(z)$ to $Y(z)$. In this work we use neural networks as the supervised learning method, in part because they can be easily adapted to many different types of X and Y and knowledge of any inherent geometry (in terms of invariances and symmetries) of the input domain X can be incorporated into the architecture of the network[37]. We consider a relationship between $X(z)$ and $Y(z)$ to be found if the accuracy of the learned function $\hat{f}$ is statistically above chance on further samples from $P_Z$ on which the model was not trained. The converse is not true; namely, if the model cannot predict the relationship better than chance, it may mean that a pattern exists, but is sufficiently complicated that it cannot be captured by the given model and training procedure. If it does indeed exist, this can give a mathematician confidence to pursue a particular line of enquiry in a problem that may otherwise be only speculative.

**Attribution techniques.** If a relationship is found, the attribution stage is to probe the learned function $\hat{f}$ with attribution techniques to further understand the nature of the relationship. These techniques attempt to explain what features or structures are relevant to the predictions made by $\hat{f}$, which can be used to understand what parts of the problem are relevant to explore further. There are many attribution techniques in the body of literature on machine learning and statistics, including stepwise forward feature selection[38], feature occlusion and attention weights[39]. In this work we use gradient-based techniques[40], broadly similar to sensitivity analysis in classical statistics and sometimes referred to as saliency maps. These techniques attribute importance to the elements of $X(z)$, by calculating how much $\hat{f}$ changes in predictions of $Y(z)$ given small changes in $X(z)$. We believe these are a particularly useful class of attribution techniques as they are conceptually simple, flexible and easy to calculate with machine learning libraries that support automatic differentiation[41-43]. Information extracted via attribution techniques can then be useful to guide the next steps of mathematical reasoning, such as conjecturing closed-form candidates $f'$, altering the sampling distribution $P_Z$ or generating new hypotheses about the object of interest $z$, as shown in Fig. 1. This can then lead to an improved or corrected version of the conjectured relationship between these quantities.

## Topology

**Problem framing.** Not all knots admit a hyperbolic geometry; however, most do, and all knots can be constructed from hyperbolic and torus knots using satellite operations[44]. In this work we focus only on hyperbolic knots. We characterize the hyperbolic structure of the knot complement by a number of easily computable invariants. These invariants do not fully define the hyperbolic structure, but they are representative of the most commonly interesting properties of the geometry. Our initial general hypothesis was that the hyperbolic invariants would be predictive of algebraic invariants. The specific hypothesis we investigated was that the geometry is predictive of the signature. The signature is an ideal candidate as it is a well-understood and common invariant, it is easy to calculate for large knots and it is an integer, which makes the prediction task particularly straightforward (compared to, for example, a polynomial).

**Data generation.** We generated a number of datasets from different distributions $P_Z$ on the set of knots using the SnapPy software package[45], as follows.

1. All knots up to 16 crossings ($\sim 1.7 \times 10^6$ knots), taken from the Regina census[46].
2. Random knot diagrams of 80 crossings generated by SnapPy’s random_link function ($\sim 10^6$ knots). As random knot generation can po-tentially lead to duplicates, we calculate a large number of invariants for each knot diagram and remove any samples that have identical invariants to a previous sample, as they are likely to represent that same knot with very high probability.
3. Knots obtained as the closures of certain braids. Unlike the previous two datasets, the knots that were produced here are not, in any sense, generic. Instead, they were specifically constructed to disprove Conjecture 1. The braids that we used were 4-braids ($n = 11,756$), 5-braids ($n = 13,217$) and 6-braids ($n = 10,897$). In terms of the standard generators $\sigma_i$ for these braid groups, the braids were chosen to be $(\sigma^{n_1}_{i_1}\sigma^{n_2}_{i_2}...\sigma^{n_K}_{i_K})^N$. The integers $I_J$ were chosen uniformly at random for the appropriate braid group. The powers $n_j$ were chosen uniformly at random in the ranges [-10, -3] and [3, 10]. The final power N was chosen uniformly between 1 and 10. The quantity $\sum |n_i|$ was restricted to be at most 15 for 5-braids and 6-braids and 12 for 4-braids, and the total number of crossings $N\sum |n_i|$ was restricted to lie in the range between 10 and 60. The rationale for these restrictions was to ensure a rich set of examples that were small enough to avoid an excessive number of failures in the invariant computations.

For the above datasets, we computed a number of algebraic and geo-metric knot invariants. Different datasets involved computing different subsets of these, depending on their role in forming and examining the main conjecture. Each of the datasets contains a subset of the following list of invariants: signature, slope, volume, meridional translation, longitudinal translation, injectivity radius, positivity, Chern-Simons invariant, symmetry group, hyperbolic torsion, hyperbolic adjoint torsion, invariant trace field, normal boundary slopes and length spectrum including the linking numbers of the short geodesics.

The computation of the canonical triangulation of randomly generated knots fails in SnapPy in our data generation process in between 0.6% and 1.7% of the cases, across datasets. The computation of the injectivity radius fails between 2.8% of the time on smaller knots up to 7.8% of the time on datasets of knots with a higher number of crossings. On knots up to 16 crossings from the Regina dataset, the injectivity radius computation failed in 5.2% of the cases. Occasional failures can occur in most of the invariant computations, in which case the computations continue for the knot in question for the remaining invariants in the requested set. Additionally, as the computational complexity of some invariants is high, operations can time out if they take more than 5min for an invariant. This is a flexible bound and ultimately a trade-off that we have used only for the invariants that were not critical for our analysis, to avoid biasing the results.

**Data encoding.** The following encoding scheme was used for converting the different types of features into real valued inputs for the network: reals directly encoded; complex numbers as two reals corresponding to the real and imaginary parts; categoricals as one-hot vectors.

All features are normalized by subtracting the mean and dividing by the variance. For simplicity, in Fig. 3a, the salience values of categoricals 
are aggregated by taking the maximum value of the saliencies of their encoded features.

**Model and training procedure.** The model architecture used for the experiments was a fully connected, feed-forward neural network, with hidden unit sizes [300,300,300] and sigmoid activations. The task was framed as a multi-class classification problem, with the distinct values of the signature as classes, cross-entropy loss as an optimizable loss function and test classification accuracy as a metric of performance. It is trained for a fixed number of steps using a standard optimizer (Adam). All settings were chosen as a priori reasonable values and did not need to be optimized.

**Process.** First, to assess whether there may be a relationship between the geometry and algebra of a knot, we trained a feed-forward neural network to predict the signature from measurements of the geometry on a dataset of randomly sampled knots. The model was able to achieve an accuracy of 78% on a held-out test set, with no errors larger than $±2$. This is substantially higher than chance (a baseline accuracy of 25%), which gave us strong confidence that a relationship may exist.

To understand how this prediction is being made by the network, we used gradient-based attribution to determine which measurements of the geometry are most relevant to the signature. We do this using a simple sensitivity measure $\mathbf{r}_i$ that averages the gradient of the loss $L$ with respect to a given input feature xi over all of the examples $\mathbf{x}_i$ in a dataset $\mathcal{X}$:

$$
\mathbf{r}_i=\frac{1}{|\mathcal{X}|}\sum\nolimits_{x \in \mathcal{X}}|\frac{\partial L}{\partial {\mathbf{X}_i}}| \tag{3}
$$

This quantity for each input feature is shown in Fig.3a, where we can determine that the relevant measurements of the geometry appear to be what is known as the cusp shape: the meridional translation, which we will denote $\mu$, and the longitudinal translation, which we will denote $\lambda$ This was confirmed by training a new model to predict the signature from only these three measurements, which was able to achieve the same level of performance as the original model.

To confirm that the slope is a sufficient aspect of the geometry to focus on, we trained a model to predict the signature from the slope alone. Visual inspection of the slope and signature in Extended Data Fig. 1a, b shows a clear linear trend, and training a linear model on this data results in a test accuracy of 78%, which is equivalent to the predictive power of the original model. This implies that the slope linearly captures all of the information about the signature that the original model had extracted from the geometry.

**Evaluation.** The confidence intervals on the feature saliencies were calculated by retraining the model 10 times with a different train/test split and a different random seed initializing both the network weights and training procedure.

## Representation theory

**Data generation.** For our main dataset we consider the symmetric groups up to $S_9$. The first symmetric group that contains a non-trivial Bruhat interval whose KL polynomial is not simply $1$ is $S_5$, and the largest interval in $S_9$ contains $9! \approx 3.6 \times 10^5$ nodes, which starts to pose computational issues when used as inputs to networks. The number of intervals in a symmetric group $S_N$ is $O(N!^2)$, which results in many billions of intervals in $S_9$. The distribution of coefficients of the KL polynomials uniformly across intervals is very imbalanced, as higher coefficients are especially rare and associated with unknown complex structure. To adjust for this and simplify the learning problem, we take advantage of equivalence classes of Bruhat intervals that eliminate many redundant small polynomials[47]. This has the added benefit of reducing the number of intervals per symmetric group(for example, to $\sim 2.9$ million intervals in $S_9$). We further reduce the dataset by including a single interval for each distinct KL polynomial for all graphs with the same number of nodes, resulting in $24,322$ non-isomorphic graphs for $S_9$. We split the intervals randomly into train/test partitions at 80%/20%.

**Data encoding.** The Bruhat interval of a pair of permutations is a partially ordered set of the elements of the group, and it can be represented as a directed acyclic graph where each node is labelled by a permutation, and each edge is labelled by a reflection. We add two features at each node representing the in-degree and out-degree of that node.

**Model and training procedure.** For modelling the Bruhat intervals, we used a particular GraphNet architecture called a message-passing neural network (MPNN)[48]. The design of the model architecture (in terms of activation functions and directionality) was motivated by the algorithms for computing KL polynomials from labelled Bruhat intervals. While labelled Bruhat intervals contain privileged information, these algorithms hinted at the kind of computation that may be useful for computing KL polynomial coefficients. Accordingly, we designed our MPNN to algorithmically align to this computation[49]. The model is bi-directional, with a hidden layer width of 128, four propagation steps and skip connections. We treat the prediction of each coefficient of the KL polynomial as a separate classification problem.

**Process.** First, to gain confidence that the conjecture is correct, we trained a model to predict coefficients of the KL polynomial from the unlabelled Bruhat interval. We were able to do so across the different coefficients with reasonable accuracy (Extended Data Table 1) giving some evidence that a general function may exist, as a four-step MPNN is a relatively simple function class. We trained a GraphNet model on the basis of a newly hypothesized representation and could achieve significantly better performance, lending evidence that it is a sufficient and helpful representation to understand the KL polynomial.

To understand how the predictions were being made by the learned function $\hat{f}$, we used gradient-based attribution to define a salient subgraph $S_G$ for each example interval $G$, induced by a subset of nodes in that interval, where $L$ is the loss and $x_v$ is the feature for vertex $v$:

$$
S_G =\{v \in G \ \big\vert | \frac{\partial L}{\partial {\mathcal{X}_v}}|> C_k \}\tag{4}
$$
We then aggregated the edges by their edge type (each is a reflection) and compared the frequency of their occurrence to the overall dataset. The effect on extremal edges was present in the salient subgraphs for predictions of the higher-order terms $(q^3, q^4)$, which are the more complicated and less well-understood terms.

**Evaluation.** The threshold $C_k$ for salient nodes was chosen a priori as the 99th percentile of attribution values across the dataset, although the results are present for different values of $C_k$ in the range [95, 99.5]. In Fig. 5a, we visualize a measure of edge attribution for a particular snapshot of a trained model for expository purposes. This view will change across time and random seeds, but we can confirm that the pattern remains by looking at aggregate statistics over many runs of training the model, as in Fig. 5b. In this diagram, the two-sample two-sided t-test statistics are as follows—simple edges: $t= 25.7, P=4.0 \times 10^{-10}$； extremal edges: $t = -13.8, P=1.1 \times 10^{-7}$； other edges: $t= -3.2, P = 0.01$. These significance results are robust to different settings of the hyper-parameters of the model.

## Code availability

Interactive notebooks to regenerate the results for both knot theory and representation theory have been made available for download at <https://github.com/deepmind.>

## Data availability

The generated datasets used in the experiments have been made available for download at <https://github.com/deepmind.>

37. Bronstein, M. M., Bruna, J., Cohen, T. & Velickovic, P. Geometric deep learning： grids, groups, graphs, geodesics, and gauges. Preprint at <https://arxiv.org/abs/2104.13478> (2021).
38. Efroymson, M. A. in Mathematical Methods for Digital Computers 191-203 (John Wiley, 1960).
39. Xu, K. et al. Show, attend and tell: neural image caption generation with visual attention. In Proc. International Conference on Machine Learning 2048-2057 (PMLR, 2015).
40. Sundararajan, M., Taly, A. & Yan, Q. Axiomatic attribution for deep networks.
In Proc. International Conference on Machine Learning 3319-3328 (PMLR, 2017).
41. Bradbury, J. et al. JAX: composable transformations of Python+NumPy programs (2018); <https://github.com/google/jax>
42. Martin A. B. A. D. I. et al. TensorFlow： large-scale machine learning on heterogeneous systems (2015); <https://doi.org/10.5281/zenodo.4724125.>
43. Paszke, A. et al. in Advances in Neural Information Processing Systems 32
(eds Wallach, H. et al.) 8024-8035 (Curran Associates, 2019).
44. Thurston, W. P. Three dimensional manifolds, Kleinian groups and hyperbolic geometry. Bull. Am. Math. Soc 6, 357-381 (1982).
45. Culler, M., Dunfield, N. M., Goerner, M. & Weeks, J. R. SnapPy, a computer program for studying the geometry and topology of 3-manifolds (2020); <http://snappy.computop.org.>
46. Burton, B. A. The next 350 million knots. In Proc. 36th International Symposium on Computational Geometry (SoCG2020) (Schloss Dagstuhl-Leibniz-Zentrum fur Informatik, 2020).
47. Warrington, G. S. Equivalence classes for the ^-coefficient of Kazhdan-Lusztig polynomials in Sn. Exp. Math. 20, 457-466 (2011).
48. Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O. & Dahl, G. E. Neural message passing for quantum chemistry. Preprint at <https://arxiv.org/abs/1704.01212> (2017).
49. Velickovic, P., Ying, R., Padovano, M., Hadsell, R. & Blundell, C. Neural execution of graph algorithms. Preprint at <https://arxiv.org/abs/1910.10593> (2019).

**Acknowledgements** We thank J.Ellenberg, S.Mohamed, O.Vinyals, A.Gaunt, A. Fawzi and D.Saxton for advice and comments on early drafts； J.Vonk for contemporary supporting work； X.Glorot and M. Overlan for insight and assistance； and A.Pierce, N.Lambert, G. Holland, R.Ahamed and C.Meyer for assistance coordinating the research. This research was funded by DeepMind.

**Author contributions** A.D., D.H. and P.K. conceived of the project. A.D., A.J. and M.L. discovered the knot theory results, with D.Z. and N.T. running additional experiments. A.D., P.V. and G.W. discovered the representation theory results, with P.V. designing the model, L.B. running additional experiments, and C.B. providing advice and ideas. S.B. and R.T. provided additional support, experiments and infrastructure. A.D., D.H. and P.K. directed and managed the project. A.D. and P.V. wrote the paper with help and feedback from P.B., C.B., M.L., A.J., G.W., P.K. and D.H.

**Competing interests** The authors declare no competing interests.

**Additional information** Supplementary information The online version contains supplementary material available at <https://doi.org/10.1038/s41586-021-04086-x>.

**Correspondence and requests for materials** should be addressed to Alex Davies or Pushmeet Kohli.

**Peer review information** Nature thanks Sanjeev Arora, Christian Stump and the other, anonymous, reviewer(s) for their contribution to the peer review of this work. Peer reviewer reports are available.

**Reprints and permissions information** is available at <http://www.nature.com/reprints.>