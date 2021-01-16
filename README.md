## Summary for 2019-12-20, created on 2021-01-16


<details><summary><b>secml: A Python Library for Secure and Explainable Machine Learning</b>
<a href="https://arxiv.org/abs/1912.10013">arxiv:1912.10013</a>
&#x1F4C8; 50 <br>
<p>Marco Melis, Ambra Demontis, Maura Pintor, Angelo Sotgiu, Battista Biggio</p></summary>
<p>

**Abstract:** We present secml, an open-source Python library for secure and explainable machine learning. It implements the most popular attacks against machine learning, including not only test-time evasion attacks to generate adversarial examples against deep neural networks, but also training-time poisoning attacks against support vector machines and many other algorithms. These attacks enable evaluating the security of learning algorithms and of the corresponding defenses under both white-box and black-box threat models. To this end, secml provides built-in functions to compute security evaluation curves, showing how quickly classification performance decreases against increasing adversarial perturbations of the input data. secml also includes explainability methods to help understand why adversarial attacks succeed against a given model, by visualizing the most influential features and training prototypes contributing to each decision. It is distributed under the Apache License 2.0, and hosted at https://gitlab.com/secml/secml.

</p>
</details>

<details><summary><b>Mastering Complex Control in MOBA Games with Deep Reinforcement Learning</b>
<a href="https://arxiv.org/abs/1912.09729">arxiv:1912.09729</a>
&#x1F4C8; 43 <br>
<p>Deheng Ye, Zhao Liu, Mingfei Sun, Bei Shi, Peilin Zhao, Hao Wu, Hongsheng Yu, Shaojie Yang, Xipeng Wu, Qingwei Guo, Qiaobo Chen, Yinyuting Yin, Hao Zhang, Tengfei Shi, Liang Wang, Qiang Fu, Wei Yang, Lanxiao Huang</p></summary>
<p>

**Abstract:** We study the reinforcement learning problem of complex action control in the Multi-player Online Battle Arena (MOBA) 1v1 games. This problem involves far more complicated state and action spaces than those of traditional 1v1 games, such as Go and Atari series, which makes it very difficult to search any policies with human-level performance. In this paper, we present a deep reinforcement learning framework to tackle this problem from the perspectives of both system and algorithm. Our system is of low coupling and high scalability, which enables efficient explorations at large scale. Our algorithm includes several novel strategies, including control dependency decoupling, action mask, target attention, and dual-clip PPO, with which our proposed actor-critic network can be effectively trained in our system. Tested on the MOBA game Honor of Kings, our AI agent, called Tencent Solo, can defeat top professional human players in full 1v1 games.

</p>
</details>

<details><summary><b>Measuring Compositional Generalization: A Comprehensive Method on Realistic Data</b>
<a href="https://arxiv.org/abs/1912.09713">arxiv:1912.09713</a>
&#x1F4C8; 36 <br>
<p>Daniel Keysers, Nathanael Schärli, Nathan Scales, Hylke Buisman, Daniel Furrer, Sergii Kashubin, Nikola Momchev, Danila Sinopalnikov, Lukasz Stafiniak, Tibor Tihon, Dmitry Tsarkov, Xiao Wang, Marc van Zee, Olivier Bousquet</p></summary>
<p>

**Abstract:** State-of-the-art machine learning methods exhibit limited compositional generalization. At the same time, there is a lack of realistic benchmarks that comprehensively measure this ability, which makes it challenging to find and evaluate improvements. We introduce a novel method to systematically construct such benchmarks by maximizing compound divergence while guaranteeing a small atom divergence between train and test sets, and we quantitatively compare this method to other approaches for creating compositional generalization benchmarks. We present a large and realistic natural language question answering dataset that is constructed according to this method, and we use it to analyze the compositional generalization ability of three machine learning architectures. We find that they fail to generalize compositionally and that there is a surprisingly strong negative correlation between compound divergence and accuracy. We also demonstrate how our method can be used to create new compositionality benchmarks on top of the existing SCAN dataset, which confirms these findings.

</p>
</details>

<details><summary><b>Are Transformers universal approximators of sequence-to-sequence functions?</b>
<a href="https://arxiv.org/abs/1912.10077">arxiv:1912.10077</a>
&#x1F4C8; 31 <br>
<p>Chulhee Yun, Srinadh Bhojanapalli, Ankit Singh Rawat, Sashank J. Reddi, Sanjiv Kumar</p></summary>
<p>

**Abstract:** Despite the widespread adoption of Transformer models for NLP tasks, the expressive power of these models is not well-understood. In this paper, we establish that Transformer models are universal approximators of continuous permutation equivariant sequence-to-sequence functions with compact support, which is quite surprising given the amount of shared parameters in these models. Furthermore, using positional encodings, we circumvent the restriction of permutation equivariance, and show that Transformer models can universally approximate arbitrary continuous sequence-to-sequence functions on a compact domain. Interestingly, our proof techniques clearly highlight the different roles of the self-attention and the feed-forward layers in Transformers. In particular, we prove that fixed width self-attention layers can compute contextual mappings of the input sequences, playing a key role in the universal approximation property of Transformers. Based on this insight from our analysis, we consider other simpler alternatives to self-attention layers and empirically evaluate them.

</p>
</details>

<details><summary><b>A Hierarchical Model for Data-to-Text Generation</b>
<a href="https://arxiv.org/abs/1912.10011">arxiv:1912.10011</a>
&#x1F4C8; 26 <br>
<p>Clément Rebuffel, Laure Soulier, Geoffrey Scoutheeten, Patrick Gallinari</p></summary>
<p>

**Abstract:** Transcribing structured data into natural language descriptions has emerged as a challenging task, referred to as "data-to-text". These structures generally regroup multiple elements, as well as their attributes. Most attempts rely on translation encoder-decoder methods which linearize elements into a sequence. This however loses most of the structure contained in the data. In this work, we propose to overpass this limitation with a hierarchical model that encodes the data-structure at the element-level and the structure level. Evaluations on RotoWire show the effectiveness of our model w.r.t. qualitative and quantitative metrics.

</p>
</details>

<details><summary><b>Recommendations and User Agency: The Reachability of Collaboratively-Filtered Information</b>
<a href="https://arxiv.org/abs/1912.10068">arxiv:1912.10068</a>
&#x1F4C8; 25 <br>
<p>Sarah Dean, Sarah Rich, Benjamin Recht</p></summary>
<p>

**Abstract:** Recommender systems often rely on models which are trained to maximize accuracy in predicting user preferences. When the systems are deployed, these models determine the availability of content and information to different users. The gap between these objectives gives rise to a potential for unintended consequences, contributing to phenomena such as filter bubbles and polarization. In this work, we consider directly the information availability problem through the lens of user recourse. Using ideas of reachability, we propose a computationally efficient audit for top-$N$ linear recommender models. Furthermore, we describe the relationship between model complexity and the effort necessary for users to exert control over their recommendations. We use this insight to provide a novel perspective on the user cold-start problem. Finally, we demonstrate these concepts with an empirical investigation of a state-of-the-art model trained on a widely used movie ratings dataset.

</p>
</details>

<details><summary><b>A Fair Comparison of Graph Neural Networks for Graph Classification</b>
<a href="https://arxiv.org/abs/1912.09893">arxiv:1912.09893</a>
&#x1F4C8; 25 <br>
<p>Federico Errica, Marco Podda, Davide Bacciu, Alessio Micheli</p></summary>
<p>

**Abstract:** Experimental reproducibility and replicability are critical topics in machine learning. Authors have often raised concerns about their lack in scientific publications to improve the quality of the field. Recently, the graph representation learning field has attracted the attention of a wide research community, which resulted in a large stream of works. As such, several Graph Neural Network models have been developed to effectively tackle graph classification. However, experimental procedures often lack rigorousness and are hardly reproducible. Motivated by this, we provide an overview of common practices that should be avoided to fairly compare with the state of the art. To counter this troubling trend, we ran more than 47000 experiments in a controlled and uniform framework to re-evaluate five popular models across nine common benchmarks. Moreover, by comparing GNNs with structure-agnostic baselines we provide convincing evidence that, on some datasets, structural information has not been exploited yet. We believe that this work can contribute to the development of the graph learning field, by providing a much needed grounding for rigorous evaluations of graph classification models.

</p>
</details>

<details><summary><b>Taxonomy and Evaluation of Structured Compression of Convolutional Neural Networks</b>
<a href="https://arxiv.org/abs/1912.09802">arxiv:1912.09802</a>
&#x1F4C8; 25 <br>
<p>Andrey Kuzmin, Markus Nagel, Saurabh Pitre, Sandeep Pendyam, Tijmen Blankevoort, Max Welling</p></summary>
<p>

**Abstract:** The success of deep neural networks in many real-world applications is leading to new challenges in building more efficient architectures. One effective way of making networks more efficient is neural network compression. We provide an overview of existing neural network compression methods that can be used to make neural networks more efficient by changing the architecture of the network. First, we introduce a new way to categorize all published compression methods, based on the amount of data and compute needed to make the methods work in practice. These are three 'levels of compression solutions'. Second, we provide a taxonomy of tensor factorization based and probabilistic compression methods. Finally, we perform an extensive evaluation of different compression techniques from the literature for models trained on ImageNet. We show that SVD and probabilistic compression or pruning methods are complementary and give the best results of all the considered methods. We also provide practical ways to combine them.

</p>
</details>

<details><summary><b>Meta-Graph: Few Shot Link Prediction via Meta Learning</b>
<a href="https://arxiv.org/abs/1912.09867">arxiv:1912.09867</a>
&#x1F4C8; 17 <br>
<p>Avishek Joey Bose, Ankit Jain, Piero Molino, William L. Hamilton</p></summary>
<p>

**Abstract:** We consider the task of few shot link prediction on graphs. The goal is to learn from a distribution over graphs so that a model is able to quickly infer missing edges in a new graph after a small amount of training. We show that current link prediction methods are generally ill-equipped to handle this task. They cannot effectively transfer learned knowledge from one graph to another and are unable to effectively learn from sparse samples of edges. To address this challenge, we introduce a new gradient-based meta learning framework, Meta-Graph. Our framework leverages higher-order gradients along with a learned graph signature function that conditionally generates a graph neural network initialization. Using a novel set of few shot link prediction benchmarks, we show that Meta-Graph can learn to quickly adapt to a new graph using only a small sample of true edges, enabling not only fast adaptation but also improved results at convergence.

</p>
</details>

<details><summary><b>Probability Calibration for Knowledge Graph Embedding Models</b>
<a href="https://arxiv.org/abs/1912.10000">arxiv:1912.10000</a>
&#x1F4C8; 16 <br>
<p>Pedro Tabacof, Luca Costabello</p></summary>
<p>

**Abstract:** Knowledge graph embedding research has overlooked the problem of probability calibration. We show popular embedding models are indeed uncalibrated. That means probability estimates associated to predicted triples are unreliable. We present a novel method to calibrate a model when ground truth negatives are not available, which is the usual case in knowledge graphs. We propose to use Platt scaling and isotonic regression alongside our method. Experiments on three datasets with ground truth negatives show our contribution leads to well-calibrated models when compared to the gold standard of using negatives. We get significantly better results than the uncalibrated models from all calibration methods. We show isotonic regression offers the best the performance overall, not without trade-offs. We also show that calibrated models reach state-of-the-art accuracy without the need to define relation-specific decision thresholds.

</p>
</details>

<details><summary><b>Certified Robustness for Top-k Predictions against Adversarial Perturbations via Randomized Smoothing</b>
<a href="https://arxiv.org/abs/1912.09899">arxiv:1912.09899</a>
&#x1F4C8; 10 <br>
<p>Jinyuan Jia, Xiaoyu Cao, Binghui Wang, Neil Zhenqiang Gong</p></summary>
<p>

**Abstract:** It is well-known that classifiers are vulnerable to adversarial perturbations. To defend against adversarial perturbations, various certified robustness results have been derived. However, existing certified robustnesses are limited to top-1 predictions. In many real-world applications, top-$k$ predictions are more relevant. In this work, we aim to derive certified robustness for top-$k$ predictions. In particular, our certified robustness is based on randomized smoothing, which turns any classifier to a new classifier via adding noise to an input example. We adopt randomized smoothing because it is scalable to large-scale neural networks and applicable to any classifier. We derive a tight robustness in $\ell_2$ norm for top-$k$ predictions when using randomized smoothing with Gaussian noise. We find that generalizing the certified robustness from top-1 to top-$k$ predictions faces significant technical challenges. We also empirically evaluate our method on CIFAR10 and ImageNet. For example, our method can obtain an ImageNet classifier with a certified top-5 accuracy of 62.8\% when the $\ell_2$-norms of the adversarial perturbations are less than 0.5 (=127/255). Our code is publicly available at: \url{https://github.com/jjy1994/Certify_Topk}.

</p>
</details>

<details><summary><b>Explainability and Adversarial Robustness for RNNs</b>
<a href="https://arxiv.org/abs/1912.09855">arxiv:1912.09855</a>
&#x1F4C8; 10 <br>
<p>Alexander Hartl, Maximilian Bachl, Joachim Fabini, Tanja Zseby</p></summary>
<p>

**Abstract:** Recurrent Neural Networks (RNNs) yield attractive properties for constructing Intrusion Detection Systems (IDSs) for network data. With the rise of ubiquitous Machine Learning (ML) systems, malicious actors have been catching up quickly to find new ways to exploit ML vulnerabilities for profit. Recently developed adversarial ML techniques focus on computer vision and their applicability to network traffic is not straightforward: Network packets expose fewer features than an image, are sequential and impose several constraints on their features.
  We show that despite these completely different characteristics, adversarial samples can be generated reliably for RNNs. To understand a classifier's potential for misclassification, we extend existing explainability techniques and propose new ones, suitable particularly for sequential data. Applying them shows that already the first packets of a communication flow are of crucial importance and are likely to be targeted by attackers. Feature importance methods show that even relatively unimportant features can be effectively abused to generate adversarial samples. Since traditional evaluation metrics such as accuracy are not sufficient for quantifying the adversarial threat, we propose the Adversarial Robustness Score (ARS) for comparing IDSs, capturing a common notion of adversarial robustness, and show that an adversarial training procedure can significantly and successfully reduce the attack surface.

</p>
</details>

<details><summary><b>TentacleNet: A Pseudo-Ensemble Template for Accurate Binary Convolutional Neural Networks</b>
<a href="https://arxiv.org/abs/1912.10103">arxiv:1912.10103</a>
&#x1F4C8; 9 <br>
<p>Luca Mocerino, Andrea Calimera</p></summary>
<p>

**Abstract:** Binarization is an attractive strategy for implementing lightweight Deep Convolutional Neural Networks (CNNs). Despite the unquestionable savings offered, memory footprint above all, it may induce an excessive accuracy loss that prevents a widespread use. This work elaborates on this aspect introducing TentacleNet, a new template designed to improve the predictive performance of binarized CNNs via parallelization. Inspired by the ensemble learning theory, it consists of a compact topology that is end-to-end trainable and organized to minimize memory utilization. Experimental results collected over three realistic benchmarks show TentacleNet fills the gap left by classical binary models, ensuring substantial memory savings w.r.t. state-of-the-art binary ensemble methods.

</p>
</details>

<details><summary><b>Adversarial Representation Active Learning</b>
<a href="https://arxiv.org/abs/1912.09720">arxiv:1912.09720</a>
&#x1F4C8; 9 <br>
<p>Ali Mottaghi, Serena Yeung</p></summary>
<p>

**Abstract:** Active learning aims to develop label-efficient algorithms by querying the most informative samples to be labeled by an oracle. The design of efficient training methods that require fewer labels is an important research direction that allows more effective use of computational and human resources for labeling and training deep neural networks. In this work, we demonstrate how we can use recent advances in deep generative models, to outperform the state-of-the-art in achieving the highest classification accuracy using as few labels as possible. Unlike previous approaches, our approach uses not only labeled images to train the classifier but also unlabeled images and generated images for co-training the whole model. Our experiments show that the proposed method significantly outperforms existing approaches in active learning on a wide range of datasets (MNIST, CIFAR-10, SVHN, CelebA, and ImageNet).

</p>
</details>

<details><summary><b>End-to-end Named Entity Recognition and Relation Extraction using Pre-trained Language Models</b>
<a href="https://arxiv.org/abs/1912.13415">arxiv:1912.13415</a>
&#x1F4C8; 8 <br>
<p>John Giorgi, Xindi Wang, Nicola Sahar, Won Young Shin, Gary D. Bader, Bo Wang</p></summary>
<p>

**Abstract:** Named entity recognition (NER) and relation extraction (RE) are two important tasks in information extraction and retrieval (IE \& IR). Recent work has demonstrated that it is beneficial to learn these tasks jointly, which avoids the propagation of error inherent in pipeline-based systems and improves performance. However, state-of-the-art joint models typically rely on external natural language processing (NLP) tools, such as dependency parsers, limiting their usefulness to domains (e.g. news) where those tools perform well. The few neural, end-to-end models that have been proposed are trained almost completely from scratch. In this paper, we propose a neural, end-to-end model for jointly extracting entities and their relations which does not rely on external NLP tools and which integrates a large, pre-trained language model. Because the bulk of our model's parameters are pre-trained and we eschew recurrence for self-attention, our model is fast to train. On 5 datasets across 3 domains, our model matches or exceeds state-of-the-art performance, sometimes by a large margin.

</p>
</details>

<details><summary><b>Probabilistic Safety Constraints for Learned High Relative Degree System Dynamics</b>
<a href="https://arxiv.org/abs/1912.10116">arxiv:1912.10116</a>
&#x1F4C8; 8 <br>
<p>Mohammad Javad Khojasteh, Vikas Dhiman, Massimo Franceschetti, Nikolay Atanasov</p></summary>
<p>

**Abstract:** This paper focuses on learning a model of system dynamics online while satisfying safety constraints.Our motivation is to avoid offline system identification or hand-specified dynamics models and allowa system to safely and autonomously estimate and adapt its own model during online operation.Given streaming observations of the system state, we use Bayesian learning to obtain a distributionover the system dynamics. In turn, the distribution is used to optimize the system behavior andensure safety with high probability, by specifying a chance constraint over a control barrier function.

</p>
</details>

<details><summary><b>Landscape Connectivity and Dropout Stability of SGD Solutions for Over-parameterized Neural Networks</b>
<a href="https://arxiv.org/abs/1912.10095">arxiv:1912.10095</a>
&#x1F4C8; 8 <br>
<p>Alexander Shevchenko, Marco Mondelli</p></summary>
<p>

**Abstract:** The optimization of multilayer neural networks typically leads to a solution with zero training error, yet the landscape can exhibit spurious local minima and the minima can be disconnected. In this paper, we shed light on this phenomenon: we show that the combination of stochastic gradient descent (SGD) and over-parameterization makes the landscape of multilayer neural networks approximately connected and thus more favorable to optimization. More specifically, we prove that SGD solutions are connected via a piecewise linear path, and the increase in loss along this path vanishes as the number of neurons grows large. This result is a consequence of the fact that the parameters found by SGD are increasingly dropout stable as the network becomes wider. We show that, if we remove part of the neurons (and suitably rescale the remaining ones), the change in loss is independent of the total number of neurons, and it depends only on how many neurons are left. Our results exhibit a mild dependence on the input dimension: they are dimension-free for two-layer networks and depend linearly on the dimension for multilayer networks. We validate our theoretical findings with numerical experiments for different architectures and classification tasks.

</p>
</details>

<details><summary><b>HiLLoC: Lossless Image Compression with Hierarchical Latent Variable Models</b>
<a href="https://arxiv.org/abs/1912.09953">arxiv:1912.09953</a>
&#x1F4C8; 8 <br>
<p>James Townsend, Thomas Bird, Julius Kunze, David Barber</p></summary>
<p>

**Abstract:** We make the following striking observation: fully convolutional VAE models trained on 32x32 ImageNet can generalize well, not just to 64x64 but also to far larger photographs, with no changes to the model. We use this property, applying fully convolutional models to lossless compression, demonstrating a method to scale the VAE-based 'Bits-Back with ANS' algorithm for lossless compression to large color photographs, and achieving state of the art for compression of full size ImageNet images. We release Craystack, an open source library for convenient prototyping of lossless compression using probabilistic models, along with full implementations of all of our compression results.

</p>
</details>

<details><summary><b>Second-order Information in First-order Optimization Methods</b>
<a href="https://arxiv.org/abs/1912.09926">arxiv:1912.09926</a>
&#x1F4C8; 8 <br>
<p>Yuzheng Hu, Licong Lin, Shange Tang</p></summary>
<p>

**Abstract:** In this paper, we try to uncover the second-order essence of several first-order optimization methods. For Nesterov Accelerated Gradient, we rigorously prove that the algorithm makes use of the difference between past and current gradients, thus approximates the Hessian and accelerates the training. For adaptive methods, we related Adam and Adagrad to a powerful technique in computation statistics---Natural Gradient Descent. These adaptive methods can in fact be treated as relaxations of NGD with only a slight difference lying in the square root of the denominator in the update rules. Skeptical about the effect of such difference, we design a new algorithm---AdaSqrt, which removes the square root in the denominator and scales the learning rate by sqrt(T). Surprisingly, our new algorithm is comparable to various first-order methods(such as SGD and Adam) on MNIST and even beats Adam on CIFAR-10! This phenomenon casts doubt on the convention view that the square root is crucial and training without it will lead to terrible performance. As far as we have concerned, so long as the algorithm tries to explore second or even higher information of the loss surface, then proper scaling of the learning rate alone will guarantee fast training and good generalization performance. To the best of our knowledge, this is the first paper that seriously considers the necessity of square root among all adaptive methods. We believe that our work can shed light on the importance of higher-order information and inspire the design of more powerful algorithms in the future.

</p>
</details>

<details><summary><b>Dependable Neural Networks for Safety Critical Tasks</b>
<a href="https://arxiv.org/abs/1912.09902">arxiv:1912.09902</a>
&#x1F4C8; 8 <br>
<p>Molly O'Brien, William Goble, Greg Hager, Julia Bukowski</p></summary>
<p>

**Abstract:** Neural Networks are being integrated into safety critical systems, e.g., perception systems for autonomous vehicles, which require trained networks to perform safely in novel scenarios. It is challenging to verify neural networks because their decisions are not explainable, they cannot be exhaustively tested, and finite test samples cannot capture the variation across all operating conditions. Existing work seeks to train models robust to new scenarios via domain adaptation, style transfer, or few-shot learning. But these techniques fail to predict how a trained model will perform when the operating conditions differ from the testing conditions. We propose a metric, Machine Learning (ML) Dependability, that measures the network's probability of success in specified operating conditions which need not be the testing conditions. In addition, we propose the metrics Task Undependability and Harmful Undependability to distinguish network failures by their consequences. We evaluate the performance of a Neural Network agent trained using Reinforcement Learning in a simulated robot manipulation task. Our results demonstrate that we can accurately predict the ML Dependability, Task Undependability, and Harmful Undependability for operating conditions that are significantly different from the testing conditions. Finally, we design a Safety Function, using harmful failures identified during testing, that reduces harmful failures, in one example, by a factor of 700 while maintaining a high probability of success.

</p>
</details>

<details><summary><b>A Survey on Distributed Machine Learning</b>
<a href="https://arxiv.org/abs/1912.09789">arxiv:1912.09789</a>
&#x1F4C8; 8 <br>
<p>Joost Verbraeken, Matthijs Wolting, Jonathan Katzy, Jeroen Kloppenburg, Tim Verbelen, Jan S. Rellermeyer</p></summary>
<p>

**Abstract:** The demand for artificial intelligence has grown significantly over the last decade and this growth has been fueled by advances in machine learning techniques and the ability to leverage hardware acceleration. However, in order to increase the quality of predictions and render machine learning solutions feasible for more complex applications, a substantial amount of training data is required. Although small machine learning models can be trained with modest amounts of data, the input for training larger models such as neural networks grows exponentially with the number of parameters. Since the demand for processing training data has outpaced the increase in computation power of computing machinery, there is a need for distributing the machine learning workload across multiple machines, and turning the centralized into a distributed system. These distributed systems present new challenges, first and foremost the efficient parallelization of the training process and the creation of a coherent model. This article provides an extensive overview of the current state-of-the-art in the field by outlining the challenges and opportunities of distributed machine learning over conventional (centralized) machine learning, discussing the techniques used for distributed machine learning, and providing an overview of the systems that are available.

</p>
</details>

<details><summary><b>Triple Generative Adversarial Networks</b>
<a href="https://arxiv.org/abs/1912.09784">arxiv:1912.09784</a>
&#x1F4C8; 8 <br>
<p>Chongxuan Li, Kun Xu, Jiashuo Liu, Jun Zhu, Bo Zhang</p></summary>
<p>

**Abstract:** We propose a unified game-theoretical framework to perform classification and conditional image generation given limited supervision. It is formulated as a three-player minimax game consisting of a generator, a classifier and a discriminator, and therefore is referred to as Triple Generative Adversarial Network (Triple-GAN). The generator and the classifier characterize the conditional distributions between images and labels to perform conditional generation and classification, respectively. The discriminator solely focuses on identifying fake image-label pairs. Under a nonparametric assumption, we prove the unique equilibrium of the game is that the distributions characterized by the generator and the classifier converge to the data distribution. As a byproduct of the three-player mechanism, Triple-GAN is flexible to incorporate different semi-supervised classifiers and GAN architectures. We evaluate Triple-GAN in two challenging settings, namely, semi-supervised learning and the extreme low data regime. In both settings, Triple-GAN can achieve excellent classification results and generate meaningful samples in a specific class simultaneously. In particular, using a commonly adopted 13-layer CNN classifier, Triple-GAN outperforms extensive semi-supervised learning methods substantially on more than 10 benchmarks no matter data augmentation is applied or not.

</p>
</details>

<details><summary><b>An adaptive simulated annealing EM algorithm for inference on non-homogeneous hidden Markov models</b>
<a href="https://arxiv.org/abs/1912.09733">arxiv:1912.09733</a>
&#x1F4C8; 8 <br>
<p>Aliaksandr Hubin</p></summary>
<p>

**Abstract:** Non-homogeneous hidden Markov models (NHHMM) are a subclass of dependent mixture models used for semi-supervised learning, where both transition probabilities between the latent states and mean parameter of the probability distribution of the responses (for a given state) depend on the set of $p$ covariates. A priori we do not know which (and how) covariates influence the transition probabilities and the mean parameters. This induces a complex combinatorial optimization problem for model selection with $4^p$ potential configurations. To address the problem, in this article we propose an adaptive (A) simulated annealing (SA) expectation maximization (EM) algorithm (ASA-EM) for joint optimization of models and their parameters with respect to a criterion of interest.

</p>
</details>

<details><summary><b>Jacobian Adversarially Regularized Networks for Robustness</b>
<a href="https://arxiv.org/abs/1912.10185">arxiv:1912.10185</a>
&#x1F4C8; 7 <br>
<p>Alvin Chan, Yi Tay, Yew Soon Ong, Jie Fu</p></summary>
<p>

**Abstract:** Adversarial examples are crafted with imperceptible perturbations with the intent to fool neural networks. Against such attacks, adversarial training and its variants stand as the strongest defense to date. Previous studies have pointed out that robust models that have undergone adversarial training tend to produce more salient and interpretable Jacobian matrices than their non-robust counterparts. A natural question is whether a model trained with an objective to produce salient Jacobian can result in better robustness. This paper answers this question with affirmative empirical results. We propose Jacobian Adversarially Regularized Networks (JARN) as a method to optimize the saliency of a classifier's Jacobian by adversarially regularizing the model's Jacobian to resemble natural training images. Image classifiers trained with JARN show improved robust accuracy compared to standard models on the MNIST, SVHN and CIFAR-10 datasets, uncovering a new angle to boost robustness without using adversarial training examples.

</p>
</details>

<details><summary><b>Black Box Recursive Translations for Molecular Optimization</b>
<a href="https://arxiv.org/abs/1912.10156">arxiv:1912.10156</a>
&#x1F4C8; 7 <br>
<p>Farhan Damani, Vishnu Sresht, Stephen Ra</p></summary>
<p>

**Abstract:** Machine learning algorithms for generating molecular structures offer a promising new approach to drug discovery. We cast molecular optimization as a translation problem, where the goal is to map an input compound to a target compound with improved biochemical properties. Remarkably, we observe that when generated molecules are iteratively fed back into the translator, molecular compound attributes improve with each step. We show that this finding is invariant to the choice of translation model, making this a "black box" algorithm. We call this method Black Box Recursive Translation (BBRT), a new inference method for molecular property optimization. This simple, powerful technique operates strictly on the inputs and outputs of any translation model. We obtain new state-of-the-art results for molecular property optimization tasks using our simple drop-in replacement with well-known sequence and graph-based models. Our method provides a significant boost in performance relative to its non-recursive peers with just a simple "for" loop. Further, BBRT is highly interpretable, allowing users to map the evolution of newly discovered compounds from known starting points.

</p>
</details>

<details><summary><b>Measuring Dataset Granularity</b>
<a href="https://arxiv.org/abs/1912.10154">arxiv:1912.10154</a>
&#x1F4C8; 7 <br>
<p>Yin Cui, Zeqi Gu, Dhruv Mahajan, Laurens van der Maaten, Serge Belongie, Ser-Nam Lim</p></summary>
<p>

**Abstract:** Despite the increasing visibility of fine-grained recognition in our field, "fine-grained'' has thus far lacked a precise definition. In this work, building upon clustering theory, we pursue a framework for measuring dataset granularity. We argue that dataset granularity should depend not only on the data samples and their labels, but also on the distance function we choose. We propose an axiomatic framework to capture desired properties for a dataset granularity measure and provide examples of measures that satisfy these properties. We assess each measure via experiments on datasets with hierarchical labels of varying granularity. When measuring granularity in commonly used datasets with our measure, we find that certain datasets that are widely considered fine-grained in fact contain subsets of considerable size that are substantially more coarse-grained than datasets generally regarded as coarse-grained. We also investigate the interplay between dataset granularity with a variety of factors and find that fine-grained datasets are more difficult to learn from, more difficult to transfer to, more difficult to perform few-shot learning with, and more vulnerable to adversarial attacks.

</p>
</details>

<details><summary><b>A Generalizable Method for Automated Quality Control of Functional Neuroimaging Datasets</b>
<a href="https://arxiv.org/abs/1912.10127">arxiv:1912.10127</a>
&#x1F4C8; 7 <br>
<p>Matthew Kollada, Qingzhu Gao, Monika S Mellem, Tathagata Banerjee, William J Martin</p></summary>
<p>

**Abstract:** Over the last twenty five years, advances in the collection and analysis of fMRI data have enabled new insights into the brain basis of human health and disease. Individual behavioral variation can now be visualized at a neural level as patterns of connectivity among brain regions. Functional brain imaging is enhancing our understanding of clinical psychiatric disorders by revealing ties between regional and network abnormalities and psychiatric symptoms. Initial success in this arena has recently motivated collection of larger datasets which are needed to leverage fMRI to generate brain-based biomarkers to support development of precision medicines. Despite methodological advances and enhanced computational power, evaluating the quality of fMRI scans remains a critical step in the analytical framework. Before analysis can be performed, expert reviewers visually inspect raw scans and preprocessed derivatives to determine viability of the data. This Quality Control (QC) process is labor intensive, and the inability to automate at large scale has proven to be a limiting factor in clinical neuroscience fMRI research. We present a novel method for automating the QC of fMRI scans. We train machine learning classifiers using features derived from brain MR images to predict the "quality" of those images, based on the ground truth of an expert's opinion. We emphasize the importance of these classifiers' ability to generalize their predictions across data from different studies. To address this, we propose a novel approach entitled "FMRI preprocessing Log mining for Automated, Generalizable Quality Control" (FLAG-QC), in which features derived from mining runtime logs are used to train the classifier. We show that classifiers trained on FLAG-QC features perform much better (AUC=0.79) than previously proposed feature sets (AUC=0.56) when testing their ability to generalize across studies.

</p>
</details>

<details><summary><b>Learning for Safety-Critical Control with Control Barrier Functions</b>
<a href="https://arxiv.org/abs/1912.10099">arxiv:1912.10099</a>
&#x1F4C8; 7 <br>
<p>Andrew Taylor, Andrew Singletary, Yisong Yue, Aaron Ames</p></summary>
<p>

**Abstract:** Modern nonlinear control theory seeks to endow systems with properties of stability and safety, and have been deployed successfully in multiple domains. Despite this success, model uncertainty remains a significant challenge in synthesizing safe controllers, leading to degradation in the properties provided by the controllers. This paper develops a machine learning framework utilizing Control Barrier Functions (CBFs) to reduce model uncertainty as it impact the safe behavior of a system. This approach iteratively collects data and updates a controller, ultimately achieving safe behavior. We validate this method in simulation and experimentally on a Segway platform.

</p>
</details>

<details><summary><b>EAST: Encoding-Aware Sparse Training for Deep Memory Compression of ConvNets</b>
<a href="https://arxiv.org/abs/1912.10087">arxiv:1912.10087</a>
&#x1F4C8; 7 <br>
<p>Matteo Grimaldi, Valentino Peluso, Andrea Calimera</p></summary>
<p>

**Abstract:** The implementation of Deep Convolutional Neural Networks (ConvNets) on tiny end-nodes with limited non-volatile memory space calls for smart compression strategies capable of shrinking the footprint yet preserving predictive accuracy. There exist a number of strategies for this purpose, from those that play with the topology of the model or the arithmetic precision, e.g. pruning and quantization, to those that operate a model agnostic compression, e.g. weight encoding. The tighter the memory constraint, the higher the probability that these techniques alone cannot meet the requirement, hence more awareness and cooperation across different optimizations become mandatory. This work addresses the issue by introducing EAST, Encoding-Aware Sparse Training, a novel memory-constrained training procedure that leads quantized ConvNets towards deep memory compression. EAST implements an adaptive group pruning designed to maximize the compression rate of the weight encoding scheme (the LZ4 algorithm in this work). If compared to existing methods, EAST meets the memory constraint with lower sparsity, hence ensuring higher accuracy. Results conducted on a state-of-the-art ConvNet (ResNet-9) deployed on a low-power microcontroller (ARM Cortex-M4) validate the proposal.

</p>
</details>

<details><summary><b>Analysis of Video Feature Learning in Two-Stream CNNs on the Example of Zebrafish Swim Bout Classification</b>
<a href="https://arxiv.org/abs/1912.09857">arxiv:1912.09857</a>
&#x1F4C8; 7 <br>
<p>Bennet Breier, Arno Onken</p></summary>
<p>

**Abstract:** Semmelhack et al. (2014) have achieved high classification accuracy in distinguishing swim bouts of zebrafish using a Support Vector Machine (SVM). Convolutional Neural Networks (CNNs) have reached superior performance in various image recognition tasks over SVMs, but these powerful networks remain a black box. Reaching better transparency helps to build trust in their classifications and makes learned features interpretable to experts. Using a recently developed technique called Deep Taylor Decomposition, we generated heatmaps to highlight input regions of high relevance for predictions. We find that our CNN makes predictions by analyzing the steadiness of the tail's trunk, which markedly differs from the manually extracted features used by Semmelhack et al. (2014). We further uncovered that the network paid attention to experimental artifacts. Removing these artifacts ensured the validity of predictions. After correction, our best CNN beats the SVM by 6.12%, achieving a classification accuracy of 96.32%. Our work thus demonstrates the utility of AI explainability for CNNs.

</p>
</details>

<details><summary><b>An Artificial Intelligence approach to Shadow Rating</b>
<a href="https://arxiv.org/abs/1912.09764">arxiv:1912.09764</a>
&#x1F4C8; 7 <br>
<p>Angela Rita Provenzano, Daniele Trifirò, Nicola Jean, Giacomo Le Pera, Maurizio Spadaccino, Luca Massaron, Claudio Nordio</p></summary>
<p>

**Abstract:** We analyse the effectiveness of modern deep learning techniques in predicting credit ratings over a universe of thousands of global corporate entities obligations when compared to most popular, traditional machine-learning approaches such as linear models and tree-based classifiers. Our results show a adequate accuracy over different rating classes when applying categorical embeddings to artificial neural networks (ANN) architectures.

</p>
</details>

<details><summary><b>Robust Data Preprocessing for Machine-Learning-Based Disk Failure Prediction in Cloud Production Environments</b>
<a href="https://arxiv.org/abs/1912.09722">arxiv:1912.09722</a>
&#x1F4C8; 7 <br>
<p>Shujie Han, Jun Wu, Erci Xu, Cheng He, Patrick P. C. Lee, Yi Qiang, Qixing Zheng, Tao Huang, Zixi Huang, Rui Li</p></summary>
<p>

**Abstract:** To provide proactive fault tolerance for modern cloud data centers, extensive studies have proposed machine learning (ML) approaches to predict imminent disk failures for early remedy and evaluated their approaches directly on public datasets (e.g., Backblaze SMART logs). However, in real-world production environments, the data quality is imperfect (e.g., inaccurate labeling, missing data samples, and complex failure types), thereby degrading the prediction accuracy. We present RODMAN, a robust data preprocessing pipeline that refines data samples before feeding them into ML models. We start with a large-scale trace-driven study of over three million disks from Alibaba Cloud's data centers, and motivate the practical challenges in ML-based disk failure prediction. We then design RODMAN with three data preprocessing echniques, namely failure-type filtering, spline-based data filling, and automated pre-failure backtracking, that are applicable for general ML models. Evaluation on both the Alibaba and Backblaze datasets shows that RODMAN improves the prediction accuracy compared to without data preprocessing under various settings.

</p>
</details>

<details><summary><b>Deep Curvature Suite</b>
<a href="https://arxiv.org/abs/1912.09656">arxiv:1912.09656</a>
&#x1F4C8; 7 <br>
<p>Diego Granziol, Xingchen Wan, Timur Garipov</p></summary>
<p>

**Abstract:** We present MLRG Deep Curvature suite, a PyTorch-based, open-source package for analysis and visualisation of neural network curvature and loss landscape. Despite of providing rich information into properties of neural network and useful for a various designed tasks, curvature information is still not made sufficient use for various reasons, and our method aims to bridge this gap. We present a primer, including its main practical desiderata and common misconceptions, of \textit{Lanczos algorithm}, the theoretical backbone of our package, and present a series of examples based on synthetic toy examples and realistic modern neural networks tested on CIFAR datasets, and show the superiority of our package against existing competing approaches for the similar purposes.

</p>
</details>

<details><summary><b>A Voice Interactive Multilingual Student Support System using IBM Watson</b>
<a href="https://arxiv.org/abs/2001.00471">arxiv:2001.00471</a>
&#x1F4C8; 6 <br>
<p>Kennedy Ralston, Yuhao Chen, Haruna Isah, Farhana Zulkernine</p></summary>
<p>

**Abstract:** Systems powered by artificial intelligence are being developed to be more user-friendly by communicating with users in a progressively human-like conversational way. Chatbots, also known as dialogue systems, interactive conversational agents, or virtual agents are an example of such systems used in a wide variety of applications ranging from customer support in the business domain to companionship in the healthcare sector. It is becoming increasingly important to develop chatbots that can best respond to the personalized needs of their users so that they can be as helpful to the user as possible in a real human way. This paper investigates and compares three popular existing chatbots API offerings and then propose and develop a voice interactive and multilingual chatbot that can effectively respond to users mood, tone, and language using IBM Watson Assistant, Tone Analyzer, and Language Translator. The chatbot was evaluated using a use case that was targeted at responding to users needs regarding exam stress based on university students survey data generated using Google Forms. The results of measuring the chatbot effectiveness at analyzing responses regarding exam stress indicate that the chatbot responding appropriately to the user queries regarding how they are feeling about exams 76.5%. The chatbot could also be adapted for use in other application areas such as student info-centers, government kiosks, and mental health support systems.

</p>
</details>

<details><summary><b>"The Squawk Bot": Joint Learning of Time Series and Text Data Modalities for Automated Financial Information Filtering</b>
<a href="https://arxiv.org/abs/1912.10858">arxiv:1912.10858</a>
&#x1F4C8; 6 <br>
<p>Xuan-Hong Dang, Syed Yousaf Shah, Petros Zerfos</p></summary>
<p>

**Abstract:** Multimodal analysis that uses numerical time series and textual corpora as input data sources is becoming a promising approach, especially in the financial industry. However, the main focus of such analysis has been on achieving high prediction accuracy while little effort has been spent on the important task of understanding the association between the two data modalities. Performance on the time series hence receives little explanation though human-understandable textual information is available. In this work, we address the problem of given a numerical time series, and a general corpus of textual stories collected in the same period of the time series, the task is to timely discover a succinct set of textual stories associated with that time series. Towards this goal, we propose a novel multi-modal neural model called MSIN that jointly learns both numerical time series and categorical text articles in order to unearth the association between them. Through multiple steps of data interrelation between the two data modalities, MSIN learns to focus on a small subset of text articles that best align with the performance in the time series. This succinct set is timely discovered and presented as recommended documents, acting as automated information filtering, for the given time series. We empirically evaluate the performance of our model on discovering relevant news articles for two stock time series from Apple and Google companies, along with the daily news articles collected from the Thomson Reuters over a period of seven consecutive years. The experimental results demonstrate that MSIN achieves up to 84.9% and 87.2% in recalling the ground truth articles respectively to the two examined time series, far more superior to state-of-the-art algorithms that rely on conventional attention mechanism in deep learning.

</p>
</details>

<details><summary><b>Random CapsNet Forest Model for Imbalanced Malware Type Classification Task</b>
<a href="https://arxiv.org/abs/1912.10836">arxiv:1912.10836</a>
&#x1F4C8; 6 <br>
<p>Aykut Çayır, Uğur Ünal, Hasan Dağ</p></summary>
<p>

**Abstract:** Behavior of a malware varies with respect to malware types. Therefore,knowing type of a malware affects strategies of system protection softwares. Many malware type classification models empowered by machine and deep learning achieve superior accuracies to predict malware types.Machine learning based models need to do heavy feature engineering and feature engineering is dominantly effecting performance of models.On the other hand, deep learning based models require less feature engineering than machine learning based models. However, traditional deep learning architectures and components cause very complex and data sensitive models. Capsule network architecture minimizes this complexity and data sensitivity unlike classical convolutional neural network architectures. This paper proposes an ensemble capsule network model based on bootstrap aggregating technique. The proposed method are tested on two malware datasets, whose the-state-of-the-art results are well-known.

</p>
</details>

<details><summary><b>CDPA: Common and Distinctive Pattern Analysis between High-dimensional Datasets</b>
<a href="https://arxiv.org/abs/1912.09989">arxiv:1912.09989</a>
&#x1F4C8; 6 <br>
<p>Hai Shu, Zhe Qu</p></summary>
<p>

**Abstract:** A representative model in integrative analysis of two high-dimensional correlated datasets is to decompose each data matrix into a low-rank common matrix generated by latent factors shared across datasets, a low-rank distinctive matrix corresponding to each dataset, and an additive noise matrix. Existing decomposition methods claim that their common matrices capture the common pattern of the two datasets. However, their so-called common pattern only denotes the common latent factors but ignores the common information between the two coefficient matrices of these latent factors. We propose a novel method, called the common and distinctive pattern analysis (CDPA), which appropriately defines the two patterns by further incorporating the common and distinctive information of the coefficient matrices. A consistent estimation approach is developed for high-dimensional settings, and shows reasonably good finite-sample performance in simulations. The superiority of CDPA over state-of-the-art methods is corroborated in both simulated data and two real-data examples from the Human Connectome Project and The Cancer Genome Atlas. A Python package implementing the CDPA method is available at https://github.com/shu-hai/CDPA.

</p>
</details>

<details><summary><b>When Explanations Lie: Why Many Modified BP Attributions Fail</b>
<a href="https://arxiv.org/abs/1912.09818">arxiv:1912.09818</a>
&#x1F4C8; 6 <br>
<p>Leon Sixt, Maximilian Granz, Tim Landgraf</p></summary>
<p>

**Abstract:** Attribution methods aim to explain a neural network's prediction by highlighting the most relevant image areas. A popular approach is to backpropagate (BP) a custom relevance score using modified rules, rather than the gradient. We analyze an extensive set of modified BP methods: Deep Taylor Decomposition, Layer-wise Relevance Propagation (LRP), Excitation BP, PatternAttribution, DeepLIFT, Deconv, RectGrad, and Guided BP. We find empirically that the explanations of all mentioned methods, except for DeepLIFT, are independent of the parameters of later layers. We provide theoretical insights for this surprising behavior and also analyze why DeepLIFT does not suffer from this limitation. Empirically, we measure how information of later layers is ignored by using our new metric, cosine similarity convergence (CSC). The paper provides a framework to assess the faithfulness of new and existing modified BP methods theoretically and empirically. For code see: https://github.com/berleon/when-explanations-lie

</p>
</details>

<details><summary><b>Distributed Online Optimization with Long-Term Constraints</b>
<a href="https://arxiv.org/abs/1912.09705">arxiv:1912.09705</a>
&#x1F4C8; 6 <br>
<p>Deming Yuan, Alexandre Proutiere, Guodong Shi</p></summary>
<p>

**Abstract:** We consider distributed online convex optimization problems, where the distributed system consists of various computing units connected through a time-varying communication graph. In each time step, each computing unit selects a constrained vector, experiences a loss equal to an arbitrary convex function evaluated at this vector, and may communicate to its neighbors in the graph. The objective is to minimize the system-wide loss accumulated over time. We propose a decentralized algorithm with regret and cumulative constraint violation in $\mathcal{O}(T^{\max\{c,1-c\} })$ and $\mathcal{O}(T^{1-c/2})$, respectively, for any $c\in (0,1)$, where $T$ is the time horizon. When the loss functions are strongly convex, we establish improved regret and constraint violation upper bounds in $\mathcal{O}(\log(T))$ and $\mathcal{O}(\sqrt{T\log(T)})$. These regret scalings match those obtained by state-of-the-art algorithms and fundamental limits in the corresponding centralized online optimization problem (for both convex and strongly convex loss functions). In the case of bandit feedback, the proposed algorithms achieve a regret and constraint violation in $\mathcal{O}(T^{\max\{c,1-c/3 \} })$ and $\mathcal{O}(T^{1-c/2})$ for any $c\in (0,1)$. We numerically illustrate the performance of our algorithms for the particular case of distributed online regularized linear regression problems.

</p>
</details>

<details><summary><b>AdaBits: Neural Network Quantization with Adaptive Bit-Widths</b>
<a href="https://arxiv.org/abs/1912.09666">arxiv:1912.09666</a>
&#x1F4C8; 6 <br>
<p>Qing Jin, Linjie Yang, Zhenyu Liao</p></summary>
<p>

**Abstract:** Deep neural networks with adaptive configurations have gained increasing attention due to the instant and flexible deployment of these models on platforms with different resource budgets. In this paper, we investigate a novel option to achieve this goal by enabling adaptive bit-widths of weights and activations in the model. We first examine the benefits and challenges of training quantized model with adaptive bit-widths, and then experiment with several approaches including direct adaptation, progressive training and joint training. We discover that joint training is able to produce comparable performance on the adaptive model as individual models. We further propose a new technique named Switchable Clipping Level (S-CL) to further improve quantized models at the lowest bit-width. With our proposed techniques applied on a bunch of models including MobileNet-V1/V2 and ResNet-50, we demonstrate that bit-width of weights and activations is a new option for adaptively executable deep neural networks, offering a distinct opportunity for improved accuracy-efficiency trade-off as well as instant adaptation according to the platform constraints in real-world applications.

</p>
</details>

<details><summary><b>Emergence of functional and structural properties of the head direction system by optimization of recurrent neural networks</b>
<a href="https://arxiv.org/abs/1912.10189">arxiv:1912.10189</a>
&#x1F4C8; 5 <br>
<p>Christopher J. Cueva, Peter Y. Wang, Matthew Chin, Xue-Xin Wei</p></summary>
<p>

**Abstract:** Recent work suggests goal-driven training of neural networks can be used to model neural activity in the brain. While response properties of neurons in artificial neural networks bear similarities to those in the brain, the network architectures are often constrained to be different. Here we ask if a neural network can recover both neural representations and, if the architecture is unconstrained and optimized, the anatomical properties of neural circuits. We demonstrate this in a system where the connectivity and the functional organization have been characterized, namely, the head direction circuits of the rodent and fruit fly. We trained recurrent neural networks (RNNs) to estimate head direction through integration of angular velocity. We found that the two distinct classes of neurons observed in the head direction system, the Compass neurons and the Shifter neurons, emerged naturally in artificial neural networks as a result of training. Furthermore, connectivity analysis and in-silico neurophysiology revealed structural and mechanistic similarities between artificial networks and the head direction system. Overall, our results show that optimization of RNNs in a goal-driven task can recapitulate the structure and function of biological circuits, suggesting that artificial neural networks can be used to study the brain at the level of both neural activity and anatomical organization.

</p>
</details>

<details><summary><b>Optimizing Collision Avoidance in Dense Airspace using Deep Reinforcement Learning</b>
<a href="https://arxiv.org/abs/1912.10146">arxiv:1912.10146</a>
&#x1F4C8; 5 <br>
<p>Sheng Li, Maxim Egorov, Mykel Kochenderfer</p></summary>
<p>

**Abstract:** New methodologies will be needed to ensure the airspace remains safe and efficient as traffic densities rise to accommodate new unmanned operations. This paper explores how unmanned free-flight traffic may operate in dense airspace. We develop and analyze autonomous collision avoidance systems for aircraft operating in dense airspace where traditional collision avoidance systems fail. We propose a metric for quantifying the decision burden on a collision avoidance system as well as a metric for measuring the impact of the collision avoidance system on airspace. We use deep reinforcement learning to compute corrections for an existing collision avoidance approach to account for dense airspace. The results show that a corrected collision avoidance system can operate more efficiently than traditional methods in dense airspace while maintaining high levels of safety.

</p>
</details>

<details><summary><b>Destruction of Image Steganography using Generative Adversarial Networks</b>
<a href="https://arxiv.org/abs/1912.10070">arxiv:1912.10070</a>
&#x1F4C8; 5 <br>
<p>Isaac Corley, Jonathan Lwowski, Justin Hoffman</p></summary>
<p>

**Abstract:** Digital image steganalysis, or the detection of image steganography, has been studied in depth for years and is driven by Advanced Persistent Threat (APT) groups', such as APT37 Reaper, utilization of steganographic techniques to transmit additional malware to perform further post-exploitation activity on a compromised host. However, many steganalysis algorithms are constrained to work with only a subset of all possible images in the wild or are known to produce a high false positive rate. This results in blocking any suspected image being an unreasonable policy. A more feasible policy is to filter suspicious images prior to reception by the host machine. However, how does one optimally filter specifically to obfuscate or remove image steganography while avoiding degradation of visual image quality in the case that detection of the image was a false positive? We propose the Deep Digital Steganography Purifier (DDSP), a Generative Adversarial Network (GAN) which is optimized to destroy steganographic content without compromising the perceptual quality of the original image. As verified by experimental results, our model is capable of providing a high rate of destruction of steganographic image content while maintaining a high visual quality in comparison to other state-of-the-art filtering methods. Additionally, we test the transfer learning capability of generalizing to to obfuscate real malware payloads embedded into different image file formats and types using an unseen steganographic algorithm and prove that our model can in fact be deployed to provide adequate results.

</p>
</details>

<details><summary><b>Progressive transfer learning for low frequency data prediction in full waveform inversion</b>
<a href="https://arxiv.org/abs/1912.09944">arxiv:1912.09944</a>
&#x1F4C8; 5 <br>
<p>Wenyi Hu, Yuchen Jin, Xuqing Wu, Jiefu Chen</p></summary>
<p>

**Abstract:** For the purpose of effective suppression of the cycle-skipping phenomenon in full waveform inversion (FWI), we developed a Deep Neural Network (DNN) approach to predict the absent low-frequency components by exploiting the implicit relation connecting the low-frequency and high-frequency data through the subsurface geological and geophysical properties. In order to solve this challenging nonlinear regression problem, two novel strategies were proposed to design the DNN architecture and the learning workflow: 1) Dual Data Feed; 2) Progressive Transfer Learning. With the Dual Data Feed structure, both the high-frequency data and the corresponding Beat Tone data are fed into the DNN to relieve the burden of feature extraction, thus reducing the network complexity and the training cost. The second strategy, Progressive Transfer Learning, enables us to unbiasedly train the DNN using a single training dataset. Unlike most established deep learning approaches where the training datasets are fixed, within the framework of the Progressive Transfer Learning, the training dataset evolves in an iterative manner while gradually absorbing the subsurface information retrieved by the physics-based inversion module, progressively enhancing the prediction accuracy of the DNN and propelling the FWI process out of the local minima. The Progressive Transfer Learning, alternatingly updating the training velocity model and the DNN parameters in a complementary fashion toward convergence, saves us from being overwhelmed by the otherwise tremendous amount of training data, and avoids the underfitting and biased sampling issues. The numerical experiments validated that, without any a priori geological information, the low-frequency data predicted by the Progressive Transfer Learning are sufficiently accurate for an FWI engine to produce reliable subsurface velocity models free of cycle-skipping-induced artifacts.

</p>
</details>

<details><summary><b>Prediction of Physical Load Level by Machine Learning Analysis of Heart Activity after Exercises</b>
<a href="https://arxiv.org/abs/1912.09848">arxiv:1912.09848</a>
&#x1F4C8; 5 <br>
<p>Peng Gang, Wei Zeng, Yuri Gordienko, Oleksandr Rokovyi, Oleg Alienin, Sergii Stirenko</p></summary>
<p>

**Abstract:** The assessment of energy expenditure in real life is of great importance for monitoring the current physical state of people, especially in work, sport, elderly care, health care, and everyday life even. This work reports about application of some machine learning methods (linear regression, linear discriminant analysis, k-nearest neighbors, decision tree, random forest, Gaussian naive Bayes, support-vector machine) for monitoring energy expenditures in athletes. The classification problem was to predict the known level of the in-exercise loads (in three categories by calories) by the heart rate activity features measured during the short period of time (1 minute only) after training, i.e by features of the post-exercise load. The results obtained shown that the post-exercise heart activity features preserve the information of the in-exercise training loads and allow us to predict their actual in-exercise levels. The best performance can be obtained by the random forest classifier with all 8 heart rate features (micro-averaged area under curve value AUCmicro = 0.87 and macro-averaged one AUCmacro = 0.88) and the k-nearest neighbors classifier with 4 most important heart rate features (AUCmicro = 0.91 and AUCmacro = 0.89). The limitations and perspectives of the ML methods used are outlined, and some practical advices are proposed as to their improvement and implementation for the better prediction of in-exercise energy expenditures.

</p>
</details>

<details><summary><b>Background Hardly Matters: Understanding Personality Attribution in Deep Residual Networks</b>
<a href="https://arxiv.org/abs/1912.09831">arxiv:1912.09831</a>
&#x1F4C8; 5 <br>
<p>Gabriëlle Ras, Ron Dotsch, Luca Ambrogioni, Umut Güçlü, Marcel A. J. van Gerven</p></summary>
<p>

**Abstract:** Perceived personality traits attributed to an individual do not have to correspond to their actual personality traits and may be determined in part by the context in which one encounters a person. These apparent traits determine, to a large extent, how other people will behave towards them. Deep neural networks are increasingly being used to perform automated personality attribution (e.g., job interviews). It is important that we understand the driving factors behind the predictions, in humans and in deep neural networks. This paper explicitly studies the effect of the image background on apparent personality prediction while addressing two important confounds present in existing literature; overlapping data splits and including facial information in the background. Surprisingly, we found no evidence that background information improves model predictions for apparent personality traits. In fact, when background is explicitly added to the input, a decrease in performance was measured across all models.

</p>
</details>

<details><summary><b>Community detection in node-attributed social networks: a survey</b>
<a href="https://arxiv.org/abs/1912.09816">arxiv:1912.09816</a>
&#x1F4C8; 5 <br>
<p>Petr Chunaev</p></summary>
<p>

**Abstract:** Community detection is a fundamental problem in social network analysis consisting in unsupervised dividing social actors (nodes in a social graph) with certain social connections (edges in a social graph) into densely knitted and highly related groups with each group well separated from the others. Classical approaches for community detection usually deal only with network structure and ignore features of its nodes (called node attributes), although many real-world social networks provide additional actors' information such as interests. It is believed that the attributes may clarify and enrich the knowledge about the actors and give sense to the communities. This belief has motivated the progress in developing community detection methods that use both the structure and the attributes of network (i.e. deal with a node-attributed graph) to yield more informative and qualitative results.
  During the last decade many such methods based on different ideas have appeared. Although there exist partial overviews of them, a recent survey is a necessity as the growing number of the methods may cause repetitions in methodology and uncertainty in practice.
  In this paper we aim at describing and clarifying the overall situation in the field of community detection in node-attributed social networks. Namely, we perform an exhaustive search of known methods and propose a classification of them based on when and how structure and attributes are fused. We not only give a description of each class but also provide general technical ideas behind each method in the class. Furthermore, we pay attention to available information which methods outperform others and which datasets and quality measures are used for their evaluation. Basing on the information collected, we make conclusions on the current state of the field and disclose several problems that seem important to be resolved in future.

</p>
</details>

<details><summary><b>Shareable Representations for Search Query Understanding</b>
<a href="https://arxiv.org/abs/2001.04345">arxiv:2001.04345</a>
&#x1F4C8; 4 <br>
<p>Mukul Kumar, Youna Hu, Will Headden, Rahul Goutam, Heran Lin, Bing Yin</p></summary>
<p>

**Abstract:** Understanding search queries is critical for shopping search engines to deliver a satisfying customer experience. Popular shopping search engines receive billions of unique queries yearly, each of which can depict any of hundreds of user preferences or intents. In order to get the right results to customers it must be known queries like "inexpensive prom dresses" are intended to not only surface results of a certain product type but also products with a low price. Referred to as query intents, examples also include preferences for author, brand, age group, or simply a need for customer service. Recent works such as BERT have demonstrated the success of a large transformer encoder architecture with language model pre-training on a variety of NLP tasks. We adapt such an architecture to learn intents for search queries and describe methods to account for the noisiness and sparseness of search query data. We also describe cost effective ways of hosting transformer encoder models in context with low latency requirements. With the right domain-specific training we can build a shareable deep learning model whose internal representation can be reused for a variety of query understanding tasks including query intent identification. Model sharing allows for fewer large models needed to be served at inference time and provides a platform to quickly build and roll out new search query classifiers.

</p>
</details>

<details><summary><b>Locality and compositionality in zero-shot learning</b>
<a href="https://arxiv.org/abs/1912.12179">arxiv:1912.12179</a>
&#x1F4C8; 4 <br>
<p>Tristan Sylvain, Linda Petrini, Devon Hjelm</p></summary>
<p>

**Abstract:** In this work we study locality and compositionality in the context of learning representations for Zero Shot Learning (ZSL). In order to well-isolate the importance of these properties in learned representations, we impose the additional constraint that, differently from most recent work in ZSL, no pre-training on different datasets (e.g. ImageNet) is performed. The results of our experiments show how locality, in terms of small parts of the input, and compositionality, i.e. how well can the learned representations be expressed as a function of a smaller vocabulary, are both deeply related to generalization and motivate the focus on more local-aware models in future research directions for representation learning.

</p>
</details>

<details><summary><b>Dissecting Ethereum Blockchain Analytics: What We Learn from Topology and Geometry of Ethereum Graph</b>
<a href="https://arxiv.org/abs/1912.10105">arxiv:1912.10105</a>
&#x1F4C8; 4 <br>
<p>Yitao Li, Umar Islambekov, Cuneyt Akcora, Ekaterina Smirnova, Yulia R. Gel, Murat Kantarcioglu</p></summary>
<p>

**Abstract:** Blockchain technology and, in particular, blockchain-based cryptocurrencies offer us information that has never been seen before in the financial world. In contrast to fiat currencies, all transactions of crypto-currencies and crypto-tokens are permanently recorded on distributed ledgers and are publicly available. As a result, this allows us to construct a transaction graph and to assess not only its organization but to glean relationships between transaction graph properties and crypto price dynamics. The ultimate goal of this paper is to facilitate our understanding on horizons and limitations of what can be learned on crypto-tokens from local topology and geometry of the Ethereum transaction network whose even global network properties remain scarcely explored. By introducing novel tools based on topological data analysis and functional data depth into Blockchain Data Analytics, we show that Ethereum network (one of the most popular blockchains for creating new crypto-tokens) can provide critical insights on price strikes of crypto-tokens that are otherwise largely inaccessible with conventional data sources and traditional analytic methods.

</p>
</details>

<details><summary><b>Chart Auto-Encoders for Manifold Structured Data</b>
<a href="https://arxiv.org/abs/1912.10094">arxiv:1912.10094</a>
&#x1F4C8; 4 <br>
<p>Stefan Schonsheck, Jie Chen, Rongjie Lai</p></summary>
<p>

**Abstract:** Deep generative models have made tremendous advances in image and signal representation learning and generation. These models employ the full Euclidean space or a bounded subset as the latent space, whose flat geometry, however, is often too simplistic to meaningfully reflect the manifold structure of the data. In this work, we advocate the use of a multi-chart latent space for better data representation. Inspired by differential geometry, we propose a \textbf{Chart Auto-Encoder (CAE)} and prove a universal approximation theorem on its representation capability. We show that the training data size and the network size scale exponentially in approximation error with an exponent depending on the intrinsic dimension of the data manifold. CAE admits desirable manifold properties that auto-encoders with a flat latent space fail to obey, predominantly proximity of data. We conduct extensive experimentation with synthetic and real-life examples to demonstrate that CAE provides reconstruction with high fidelity, preserves proximity in the latent space, and generates new data remaining near the manifold. These experiments show that CAE is advantageous over existing auto-encoders and variants by preserving the topology of the data manifold as well as its geometry.

</p>
</details>

<details><summary><b>SCR-Apriori for Mining `Sets of Contrasting Rules'</b>
<a href="https://arxiv.org/abs/1912.09817">arxiv:1912.09817</a>
&#x1F4C8; 4 <br>
<p>Marharyta Aleksandrova, Oleg Chertov</p></summary>
<p>

**Abstract:** In this paper, we propose an efficient algorithm for mining novel `Set of Contrasting Rules'-pattern (SCR-pattern), which consists of several association rules. This pattern is of high interest due to the guaranteed quality of the rules forming it and its ability to discover useful knowledge. However, SCR-pattern has no efficient mining algorithm. We propose SCR-Apriori algorithm, which results in the same set of SCR-patterns as the state-of-the-art approache, but is less computationally expensive. We also show experimentally that by incorporating the knowledge about the pattern structure into Apriori algorithm, SCR-Apriori can significantly prune the search space of frequent itemsets to be analysed.

</p>
</details>

<details><summary><b>What do Asian Religions Have in Common? An Unsupervised Text Analytics Exploration</b>
<a href="https://arxiv.org/abs/1912.10847">arxiv:1912.10847</a>
&#x1F4C8; 3 <br>
<p>Preeti Sah, Ernest Fokoué</p></summary>
<p>

**Abstract:** The main source of various religious teachings is their sacred texts which vary from religion to religion based on different factors like the geographical location or time of the birth of a particular religion. Despite these differences, there could be similarities between the sacred texts based on what lessons it teaches to its followers. This paper attempts to find the similarity using text mining techniques. The corpus consisting of Asian (Tao Te Ching, Buddhism, Yogasutra, Upanishad) and non-Asian (four Bible texts) is used to explore findings of similarity measures like Euclidean, Manhattan, Jaccard and Cosine on raw Document Term Frequency [DTM], normalized DTM which reveals similarity based on word usage. The performance of Supervised learning algorithms like K-Nearest Neighbor [KNN], Support Vector Machine [SVM] and Random Forest is measured based on its accuracy to predict correct scared text for any given chapter in the corpus. The K-means clustering visualizations on Euclidean distances of raw DTM reveals that there exists a pattern of similarity among these sacred texts with Upanishads and Tao Te Ching is the most similar text in the corpus.

</p>
</details>

<details><summary><b>Regularized Operating Envelope with Interpretability and Implementability Constraints</b>
<a href="https://arxiv.org/abs/1912.10158">arxiv:1912.10158</a>
&#x1F4C8; 3 <br>
<p>Qiyao Wang, Haiyan Wang, Chetan Gupta, Susumu Serita</p></summary>
<p>

**Abstract:** Operating envelope is an important concept in industrial operations. Accurate identification for operating envelope can be extremely beneficial to stakeholders as it provides a set of operational parameters that optimizes some key performance indicators (KPI) such as product quality, operational safety, equipment efficiency, environmental impact, etc. Given the importance, data-driven approaches for computing the operating envelope are gaining popularity. These approaches typically use classifiers such as support vector machines, to set the operating envelope by learning the boundary in the operational parameter spaces between the manually assigned `large KPI' and `small KPI' groups. One challenge to these approaches is that the assignment to these groups is often ad-hoc and hence arbitrary. However, a bigger challenge with these approaches is that they don't take into account two key features that are needed to operationalize operating envelopes: (i) interpretability of the envelope by the operator and (ii) implementability of the envelope from a practical standpoint. In this work, we propose a new definition for operating envelope which directly targets the expected magnitude of KPI (i.e., no need to arbitrarily bin the data instances into groups) and accounts for the interpretability and the implementability. We then propose a regularized `GA + penalty' algorithm that outputs an envelope where the user can tradeoff between bias and variance. The validity of our proposed algorithm is demonstrated by two sets of simulation studies and an application to a real-world challenge in the mining processes of a flotation plant.

</p>
</details>

<details><summary><b>Generating Robust Supervision for Learning-Based Visual Navigation Using Hamilton-Jacobi Reachability</b>
<a href="https://arxiv.org/abs/1912.10120">arxiv:1912.10120</a>
&#x1F4C8; 3 <br>
<p>Anjian Li, Somil Bansal, Georgios Giovanis, Varun Tolani, Claire Tomlin, Mo Chen</p></summary>
<p>

**Abstract:** In Bansal et al. (2019), a novel visual navigation framework that combines learning-based and model-based approaches has been proposed. Specifically, a Convolutional Neural Network (CNN) predicts a waypoint that is used by the dynamics model for planning and tracking a trajectory to the waypoint. However, the CNN inevitably makes prediction errors which often lead to collisions in cluttered and tight spaces. In this paper, we present a novel Hamilton-Jacobi (HJ) reachability-based method to generate supervision for the CNN for waypoint prediction in an unseen environment. By modeling CNN prediction error as "disturbances" in robot's dynamics, our generated waypoints are robust to these disturbances, and consequently to the prediction errors. Moreover, using globally optimal HJ reachability analysis leads to predicting waypoints that are time-efficient and avoid greedy behavior. Through simulations and hardware experiments, we demonstrate the advantages of the proposed approach on navigating through cluttered, narrow indoor environments.

</p>
</details>

<details><summary><b>Sum-Product Network Decompilation</b>
<a href="https://arxiv.org/abs/1912.10092">arxiv:1912.10092</a>
&#x1F4C8; 3 <br>
<p>Cory J. Butz, Jhonatan S. Oliveira, Robert Peharz</p></summary>
<p>

**Abstract:** There exists a dichotomy between classical probabilistic graphical models, such as Bayesian networks (BNs), and modern tractable models, such as sum-product networks (SPNs). The former generally have intractable inference, but provide a high level of interpretability, while the latter admits a wide range of tractable inference routines, but are typically harder to interpret. Due to this dichotomy, tools to convert between BNs and SPNs are desirable. While one direction -- compiling BNs into SPNs -- is well discussed in Darwiche's seminal work on arithmetic circuit compilation, the converse direction -- decompiling SPNs into BNs -- has received surprisingly little attention.
  In this paper, we fill this gap by proposing SPN2BN, an algorithm that decompiles an SPN into a BN. SPN2BN has several salient features when compared to the only other two works decompiling SPNs. Most significantly, the BNs returned by SPN2BN are minimal independence-maps that are more parsimonious with respect to the introduction of latent variables. Secondly, the output BN produced by SPN2BN can be precisely characterized with respect to a compiled BN. More specifically, a certain set of directed edges will be added to the input BN, giving what we will call the moral-closure. Lastly, it is established that our compilation-decompilation process is idempotent. This has practical significance as it limits the size of the decompiled SPN.

</p>
</details>

<details><summary><b>Dynamic Prediction of ICU Mortality Risk Using Domain Adaptation</b>
<a href="https://arxiv.org/abs/1912.10080">arxiv:1912.10080</a>
&#x1F4C8; 3 <br>
<p>Tiago Alves, Alberto Laender, Adriano Veloso, Nivio Ziviani</p></summary>
<p>

**Abstract:** Early recognition of risky trajectories during an Intensive Care Unit (ICU) stay is one of the key steps towards improving patient survival. Learning trajectories from physiological signals continuously measured during an ICU stay requires learning time-series features that are robust and discriminative across diverse patient populations. Patients within different ICU populations (referred here as domains) vary by age, conditions and interventions. Thus, mortality prediction models using patient data from a particular ICU population may perform suboptimally in other populations because the features used to train such models have different distributions across the groups. In this paper, we explore domain adaptation strategies in order to learn mortality prediction models that extract and transfer complex temporal features from multivariate time-series ICU data. Features are extracted in a way that the state of the patient in a certain time depends on the previous state. This enables dynamic predictions and creates a mortality risk space that describes the risk of a patient at a particular time. Experiments based on cross-ICU populations reveals that our model outperforms all considered baselines. Gains in terms of AUC range from 4% to 8% for early predictions when compared with a recent state-of-the-art representative for ICU mortality prediction. In particular, models for the Cardiac ICU population achieve AUC numbers as high as 0.88, showing excellent clinical utility for early mortality prediction. Finally, we present an explanation of factors contributing to the possible ICU outcomes, so that our models can be used to complement clinical reasoning.

</p>
</details>

<details><summary><b>Learning Deep Attribution Priors Based On Prior Knowledge</b>
<a href="https://arxiv.org/abs/1912.10065">arxiv:1912.10065</a>
&#x1F4C8; 3 <br>
<p>Ethan Weinberger, Joseph Janizek, Su-In Lee</p></summary>
<p>

**Abstract:** Feature attribution methods, which explain an individual prediction made by a model as a sum of attributions for each input feature, are an essential tool for understanding the behavior of complex deep learning models. However, ensuring that models produce meaningful explanations, rather than ones that rely on noise, is not straightforward. Exacerbating this problem is the fact that attribution methods do not provide insight as to why features are assigned their attribution values, leading to explanations that are difficult to interpret. In real-world problems we often have sets of additional information for each feature that are predictive of that feature's importance to the task at hand. Here, we propose the deep attribution prior (DAPr) framework to exploit such information to overcome the limitations of attribution methods. Our framework jointly learns a relationship between prior information and feature importance, as well as biases models to have explanations that rely on features predicted to be important. We find that our framework both results in networks that generalize better to out of sample data and admits new methods for interpreting model behavior.

</p>
</details>

<details><summary><b>Shear Stress Distribution Prediction in Symmetric Compound Channels Using Data Mining and Machine Learning Models</b>
<a href="https://arxiv.org/abs/2001.01558">arxiv:2001.01558</a>
&#x1F4C8; 2 <br>
<p>Zohreh Sheikh Khozani, Khabat Khosravi, Mohammadamin Torabi, Amir Mosavi, Bahram Rezaei, Timon Rabczuk</p></summary>
<p>

**Abstract:** Shear stress distribution prediction in open channels is of utmost importance in hydraulic structural engineering as it directly affects the design of stable channels. In this study, at first, a series of experimental tests were conducted to assess the shear stress distribution in prismatic compound channels. The shear stress values around the whole wetted perimeter were measured in the compound channel with different floodplain widths also in different flow depths in subcritical and supercritical conditions. A set of, data mining and machine learning models including Random Forest (RF), M5P, Random Committee (RC), KStar and Additive Regression Model (AR) implemented on attained data to predict the shear stress distribution in the compound channel. Results indicated among these five models, RF method indicated the most precise results with the highest R2 value of 0.9. Finally, the most powerful data mining method which studied in this research (RF) compared with two well-known analytical models of Shiono and Knight Method (SKM) and Shannon method to acquire the proposed model functioning in predicting the shear stress distribution. The results showed that the RF model has the best prediction performance compared to SKM and Shannon models.

</p>
</details>

<details><summary><b>Predictive Coding for Boosting Deep Reinforcement Learning with Sparse Rewards</b>
<a href="https://arxiv.org/abs/1912.13414">arxiv:1912.13414</a>
&#x1F4C8; 2 <br>
<p>Xingyu Lu, Stas Tiomkin, Pieter Abbeel</p></summary>
<p>

**Abstract:** While recent progress in deep reinforcement learning has enabled robots to learn complex behaviors, tasks with long horizons and sparse rewards remain an ongoing challenge. In this work, we propose an effective reward shaping method through predictive coding to tackle sparse reward problems. By learning predictive representations offline and using these representations for reward shaping, we gain access to reward signals that understand the structure and dynamics of the environment. In particular, our method achieves better learning by providing reward signals that 1) understand environment dynamics 2) emphasize on features most useful for learning 3) resist noise in learned representations through reward accumulation. We demonstrate the usefulness of this approach in different domains ranging from robotic manipulation to navigation, and we show that reward signals produced through predictive coding are as effective for learning as hand-crafted rewards.

</p>
</details>

<details><summary><b>Unsupervised Few-shot Learning via Self-supervised Training</b>
<a href="https://arxiv.org/abs/1912.12178">arxiv:1912.12178</a>
&#x1F4C8; 2 <br>
<p>Zilong Ji, Xiaolong Zou, Tiejun Huang, Si Wu</p></summary>
<p>

**Abstract:** Learning from limited exemplars (few-shot learning) is a fundamental, unsolved problem that has been laboriously explored in the machine learning community. However, current few-shot learners are mostly supervised and rely heavily on a large amount of labeled examples. Unsupervised learning is a more natural procedure for cognitive mammals and has produced promising results in many machine learning tasks. In the current study, we develop a method to learn an unsupervised few-shot learner via self-supervised training (UFLST), which can effectively generalize to novel but related classes. The proposed model consists of two alternate processes, progressive clustering and episodic training. The former generates pseudo-labeled training examples for constructing episodic tasks; and the later trains the few-shot learner using the generated episodic tasks which further optimizes the feature representations of data. The two processes facilitate with each other, and eventually produce a high quality few-shot learner. Using the benchmark dataset Omniglot and Mini-ImageNet, we show that our model outperforms other unsupervised few-shot learning methods. Using the benchmark dataset Market1501, we further demonstrate the feasibility of our model to a real-world application on person re-identification.

</p>
</details>

<details><summary><b>Assessing Data Quality of Annotations with Krippendorff Alpha For Applications in Computer Vision</b>
<a href="https://arxiv.org/abs/1912.10107">arxiv:1912.10107</a>
&#x1F4C8; 2 <br>
<p>Joseph Nassar, Viveca Pavon-Harr, Marc Bosch, Ian McCulloh</p></summary>
<p>

**Abstract:** Current supervised deep learning frameworks rely on annotated data for modeling the underlying data distribution of a given task. In particular for computer vision algorithms powered by deep learning, the quality of annotated data is the most critical factor in achieving the desired algorithm performance. Data annotation is, typically, a manual process where the annotator follows guidelines and operates in a best-guess manner. Labeling criteria among annotators can show discrepancies in labeling results. This may impact the algorithm inference performance. Given the popularity and widespread use of deep learning among computer vision, more and more custom datasets are needed to train neural networks to tackle different kinds of tasks. Unfortunately, there is no full understanding of the factors that affect annotated data quality, and how it translates into algorithm performance. In this paper we studied this problem for object detection and recognition.We conducted several data annotation experiments to measure inter annotator agreement and consistency, as well as how the selection of ground truth impacts the perceived algorithm performance.We propose a methodology to monitor the quality of annotations during the labeling of images and how it can be used to measure performance. We also show that neglecting to monitor the annotation process can result in significant loss in algorithm precision. Through these experiments, we observe that knowledge of the labeling process, training data, and ground truth data used for algorithm evaluation are fundamental components to accurately assess trustworthiness of an AI system.

</p>
</details>

<details><summary><b>(Newtonian) Space-Time Algebra</b>
<a href="https://arxiv.org/abs/2001.04242">arxiv:2001.04242</a>
&#x1F4C8; 1 <br>
<p>James E. Smith</p></summary>
<p>

**Abstract:** The space-time (s-t) algebra provides a mathematical model for communication and computation using values encoded as events in discretized linear (Newtonian) time. Consequently, the input-output behavior of s-t algebra and implemented functions are consistent with the flow of time. The s-t algebra and functions are formally defined. A network design framework for s-t functions is described, and the design of temporal neural networks, a form of spiking neural networks, is discussed as an extended case study. Finally, the relationship with Allen's interval algebra is briefly discussed.

</p>
</details>

<details><summary><b>Bayesian machine learning for Boltzmann machine in quantum-enhanced feature spaces</b>
<a href="https://arxiv.org/abs/1912.10857">arxiv:1912.10857</a>
&#x1F4C8; 1 <br>
<p>Yusen Wu, Chao-hua Yu, Sujuan Qin, Qiaoyan Wen, Fei Gao</p></summary>
<p>

**Abstract:** Bayesian learning is ubiquitous for implementing classification and regression tasks, however, it is accompanied by computationally intractable limitations when the feature spaces become extremely large. Aiming to solve this problem, we develop a quantum bayesian learning framework of the restricted Boltzmann machine in the quantum-enhanced feature spaces. Our framework provides the encoding phase to map the real data and Boltzmann weight onto the quantum feature spaces and the training phase to learn an optimal inference function. Specifically, the training phase provides a physical quantity to measure the posterior distribution in quantum feature spaces, and this measure is utilized to design the quantum maximum a posterior (QMAP) algorithm and the quantum predictive distribution estimator (QPDE). It is shown that both quantum algorithms achieve exponential speed-up over their classical counterparts. Furthermore, it is interesting to note that our framework can figure out the classical bayesian learning tasks, i.e. processing the classical data and outputting corresponding classical labels. And a simulation, which is performed on an open-source software framework for quantum computing, illustrates that our algorithms show almost the same classification performance compared to their classical counterparts. Noting that the proposed quantum algorithms utilize the shallow circuit, our work is expected to be implemented on the noisy intermediate-scale quantum (NISQ) devices, and is one of the promising candidates to achieve quantum supremacy.

</p>
</details>

<details><summary><b>Big Data Approaches to Knot Theory: Understanding the Structure of the Jones Polynomial</b>
<a href="https://arxiv.org/abs/1912.10086">arxiv:1912.10086</a>
&#x1F4C8; 1 <br>
<p>Jesse S F Levitt, Mustafa Hajij, Radmila Sazdanovic</p></summary>
<p>

**Abstract:** We examine the structure and dimensionality of the Jones polynomial using manifold learning techniques. Our data set consists of more than 10 million knots up to 17 crossings and two other special families up to 2001 crossings. We introduce and describe a method for using filtrations to analyze infinite data sets where representative sampling is impossible or impractical, an essential requirement for working with knots and the data from knot invariants. In particular, this method provides a new approach for analyzing knot invariants using Principal Component Analysis. Using this approach on the Jones polynomial data we find that it can be viewed as an approximately 3 dimensional manifold, that this description is surprisingly stable with respect to the filtration by the crossing number, and that the results suggest further structures to be examined and understood.

</p>
</details>

<details><summary><b>Morphy: A Datamorphic Software Test Automation Tool</b>
<a href="https://arxiv.org/abs/1912.09881">arxiv:1912.09881</a>
&#x1F4C8; 1 <br>
<p>Hong Zhu, Ian Bayley, Dongmei Liu, Xiaoyu Zheng</p></summary>
<p>

**Abstract:** This paper presents an automated tool called Morphy for datamorphic testing. It classifies software test artefacts into test entities and test morphisms, which are mappings on testing entities. In addition to datamorphisms, metamorphisms and seed test case makers, Morphy also employs a set of other test morphisms including test case metrics and filters, test set metrics and filters, test result analysers and test executers to realise test automation. In particular, basic testing activities can be automated by invoking test morphisms. Test strategies can be realised as complex combinations of test morphisms. Test processes can be automated by recording, editing and playing test scripts that invoke test morphisms and strategies. Three types of test strategies have been implemented in Morphy: datamorphism combination strategies, cluster border exploration strategies and strategies for test set optimisation via genetic algorithms. This paper focuses on the datamorphism combination strategies by giving their definitions and implementation algorithms. The paper also illustrates their uses for testing both traditional software and AI applications with three case studies.

</p>
</details>

<details><summary><b>A Paraconsistent ASP-like Language with Tractable Model Generation</b>
<a href="https://arxiv.org/abs/1912.09715">arxiv:1912.09715</a>
&#x1F4C8; 1 <br>
<p>Andrzej Szalas</p></summary>
<p>

**Abstract:** Answer Set Programming (ASP) is nowadays a dominant rule-based knowledge representation tool. Though existing ASP variants enjoy efficient implementations, generating an answer set remains intractable. The goal of this research is to define a new \asp-like rule language, 4SP, with tractable model generation. The language combines ideas of ASP and a paraconsistent rule language 4QL. Though 4SP shares the syntax of \asp and for each program all its answer sets are among 4SP models, the new language differs from ASP in its logical foundations, the intended methodology of its use and complexity of computing models.
  As we show in the paper, 4QL can be seen as a paraconsistent counterpart of ASP programs stratified with respect to default negation. Although model generation of well-supported models for 4QL programs is tractable, dropping stratification makes both 4QL and ASP intractable. To retain tractability while allowing non-stratified programs, in 4SP we introduce trial expressions interlacing programs with hypotheses as to the truth values of default negations. This allows us to develop a~model generation algorithm with deterministic polynomial time complexity.
  We also show relationships among 4SP, ASP and 4QL.

</p>
</details>

<details><summary><b>CORE: Automating Review Recommendation for Code Changes</b>
<a href="https://arxiv.org/abs/1912.09652">arxiv:1912.09652</a>
&#x1F4C8; 1 <br>
<p>JingKai Siow, Cuiyun Gao, Lingling Fan, Sen Chen, Yang Liu</p></summary>
<p>

**Abstract:** Code review is a common process that is used by developers, in which a reviewer provides useful comments or points out defects in the submitted source code changes via pull request. Code review has been widely used for both industry and open-source projects due to its capacity in early defect identification, project maintenance, and code improvement. With rapid updates on project developments, code review becomes a non-trivial and labor-intensive task for reviewers. Thus, an automated code review engine can be beneficial and useful for project development in practice. Although there exist prior studies on automating the code review process by adopting static analysis tools or deep learning techniques, they often require external sources such as partial or full source code for accurate review suggestion. In this paper, we aim at automating the code review process only based on code changes and the corresponding reviews but with better performance. The hinge of accurate code review suggestion is to learn good representations for both code changes and reviews. To achieve this with limited source, we design a multi-level embedding (i.e., word embedding and character embedding) approach to represent the semantics provided by code changes and reviews. The embeddings are then well trained through a proposed attentional deep learning model, as a whole named CORE. We evaluate the effectiveness of CORE on code changes and reviews collected from 19 popular Java projects hosted on Github. Experimental results show that our model CORE can achieve significantly better performance than the state-of-the-art model (DeepMem), with an increase of 131.03% in terms of Recall@10 and 150.69% in terms of Mean Reciprocal Rank. Qualitative general word analysis among project developers also demonstrates the performance of CORE in automating code review.

</p>
</details>

<details><summary><b>A vector-contraction inequality for Rademacher complexities using $p$-stable variables</b>
<a href="https://arxiv.org/abs/1912.10136">arxiv:1912.10136</a>
&#x1F4C8; 0 <br>
<p>Oscar Zatarain-Vera</p></summary>
<p>

**Abstract:** Andreas Maurer in the paper "A vector-contraction inequality for Rademacher complexities'' extended the contraction inequality for Rademacher averages to Lipschitz functions with vector-valued domains; He did it replacing the Rademacher variables in the bounding expression by arbitrary idd symmetric and sub-gaussian variables. We will see how to extend this work when we replace sub-gaussian variables by $p$-stable variables for $1<p<2$.

</p>
</details>

<details><summary><b>Teaching robots to perceive time -- A reinforcement learning approach (Extended version)</b>
<a href="https://arxiv.org/abs/1912.10113">arxiv:1912.10113</a>
&#x1F4C8; 0 <br>
<p>Inês Lourenço, Bo Wahlberg, Rodrigo Ventura</p></summary>
<p>

**Abstract:** Time perception is the phenomenological experience of time by an individual. In this paper, we study how to replicate neural mechanisms involved in time perception, allowing robots to take a step towards temporal cognition. Our framework follows a twofold biologically inspired approach. The first step consists of estimating the passage of time from sensor measurements, since environmental stimuli influence the perception of time. Sensor data is modeled as Gaussian processes that represent the second-order statistics of the natural environment. The estimated elapsed time between two events is computed from the maximum likelihood estimate of the joint distribution of the data collected between them. Moreover, exactly how time is encoded in the brain remains unknown, but there is strong evidence of the involvement of dopaminergic neurons in timing mechanisms. Since their phasic activity has a similar behavior to the reward prediction error of temporal-difference learning models, the latter are used to replicate this behavior. The second step of this approach consists therefore of applying the agent's estimate of the elapsed time in a reinforcement learning problem, where a feature representation called Microstimuli is used. We validate our framework by applying it to an experiment that was originally conducted with mice, and conclude that a robot using this framework is able to reproduce the timing mechanisms of the animal's brain.

</p>
</details>

<details><summary><b>SensAI+Expanse Adaptation on Human Behaviour Towards Emotional Valence Prediction</b>
<a href="https://arxiv.org/abs/1912.10084">arxiv:1912.10084</a>
&#x1F4C8; 0 <br>
<p>Nuno A. C. Henriques, Helder Coelho, Leonel Garcia-Marques</p></summary>
<p>

**Abstract:** An agent, artificial or human, must be continuously adjusting its behaviour in order to thrive in a more or less demanding environment. An artificial agent with the ability to predict human emotional valence in a geospatial and temporal context requires proper adaptation to its mobile device environment with resource consumption strict restrictions (e.g., power from battery). The developed distributed system includes a mobile device embodied agent (SensAI) plus Cloud-expanded (Expanse) cognition and memory resources. The system is designed with several adaptive mechanisms in a best effort for the agent to cope with its interacting humans and to be resilient on collecting data for machine learning towards prediction. These mechanisms encompass homeostatic-like adjustments such as auto recovering from an unexpected failure in the mobile device, forgetting repeated data to save local memory, adjusting actions to a proper moment (e.g., notify only when human is interacting), and the Expanse complementary learning algorithms' parameters with auto adjustments. Regarding emotional valence prediction performance, results from a comparison study between state-of-the-art algorithms revealed Extreme Gradient Boosting on average the best model for prediction with efficient energy use, and explainable using feature importance inspection. Therefore, this work contributes with a smartphone sensing-based system, distributed in the Cloud, robust to unexpected behaviours from humans and the environment, able to predict emotional valence states with very good performance.

</p>
</details>

<details><summary><b>Online and Offline Deep Learning Strategies For Channel Estimation and Hybrid Beamforming in Multi-Carrier mm-Wave Massive MIMO Systems</b>
<a href="https://arxiv.org/abs/1912.10036">arxiv:1912.10036</a>
&#x1F4C8; 0 <br>
<p>Ahmet M. Elbir, Kumar Vijay Mishra, M. R. Bhavani Shankar, Björn Ottersten</p></summary>
<p>

**Abstract:** Hybrid analog and digital beamforming transceivers are instrumental in addressing the challenge of expensive hardware and high training overheads in the next generation millimeter-wave (mm-Wave) massive MIMO (multiple-input multiple-output) systems. However, lack of fully digital beamforming in hybrid architectures and short coherence times at mm-Wave impose additional constraints on the channel estimation. Prior works on addressing these challenges have focused largely on narrowband channels wherein optimization-based or greedy algorithms were employed to derive hybrid beamformers. In this paper, we introduce a deep learning (DL) approach for joint channel estimation and hybrid beamforming for frequency-selective, wideband mm-Wave systems. In particular, we consider a massive MIMO Orthogonal Frequency Division Multiplexing (MIMO-OFDM) system and propose three different DL frameworks comprising convolutional neural networks (CNNs), which accept the received pilot signal as input and yield the hybrid beamformers at the output. We also introduce both offline and online prediction schemes for channel estimation and hybrid beamforming. Numerical experiments demonstrate that, compared to the current state-of-the-art optimization and DL methods, our approach provides higher spectral efficiency, lesser computational cost, and higher tolerance against the deviations in the received pilot data, corrupted channel matrix, and propagation environment.

</p>
</details>

<details><summary><b>Lightweight and Unobtrusive Data Obfuscation at IoT Edge for Remote Inference</b>
<a href="https://arxiv.org/abs/1912.09859">arxiv:1912.09859</a>
&#x1F4C8; 0 <br>
<p>Dixing Xu, Mengyao Zheng, Linshan Jiang, Chaojie Gu, Rui Tan, Peng Cheng</p></summary>
<p>

**Abstract:** Executing deep neural networks for inference on the server-class or cloud backend based on data generated at the edge of Internet of Things is desirable due primarily to the limited compute power of edge devices and the need to protect the confidentiality of the inference neural networks. However, such a remote inference scheme incurs concerns regarding the privacy of the inference data transmitted by the edge devices to the curious backend. This paper presents a lightweight and unobtrusive approach to obfuscate the inference data at the edge devices. It is lightweight in that the edge device only needs to execute a small-scale neural network; it is unobtrusive in that the edge device does not need to indicate whether obfuscation is applied. Extensive evaluation by three case studies of free spoken digit recognition, handwritten digit recognition, and American sign language recognition shows that our approach effectively protects the confidentiality of the raw forms of the inference data while effectively preserving the backend's inference accuracy.

</p>
</details>

<details><summary><b>Adversarial symmetric GANs: bridging adversarial samples and adversarial networks</b>
<a href="https://arxiv.org/abs/1912.09670">arxiv:1912.09670</a>
&#x1F4C8; 0 <br>
<p>Faqiang Liu, Mingkun Xu, Guoqi Li, Jing Pei, Luping Shi, Rong Zhao</p></summary>
<p>

**Abstract:** Generative adversarial networks have achieved remarkable performance on various tasks but suffer from training instability. Despite many training strategies proposed to improve training stability, this issue remains as a challenge. In this paper, we investigate the training instability from the perspective of adversarial samples and reveal that adversarial training on fake samples is implemented in vanilla GANs, but adversarial training on real samples has long been overlooked. Consequently, the discriminator is extremely vulnerable to adversarial perturbation and the gradient given by the discriminator contains non-informative adversarial noises, which hinders the generator from catching the pattern of real samples. Here, we develop adversarial symmetric GANs (AS-GANs) that incorporate adversarial training of the discriminator on real samples into vanilla GANs, making adversarial training symmetrical. The discriminator is therefore more robust and provides more informative gradient with less adversarial noise, thereby stabilizing training and accelerating convergence. The effectiveness of the AS-GANs is verified on image generation on CIFAR-10 , CelebA, and LSUN with varied network architectures. Not only the training is more stabilized, but the FID scores of generated samples are consistently improved by a large margin compared to the baseline. The bridging of adversarial samples and adversarial networks provides a new approach to further develop adversarial networks.

</p>
</details>


[Next Page](2019/2019-12/2019-12-19.md)
