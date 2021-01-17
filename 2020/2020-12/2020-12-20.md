## Summary for 2020-12-20, created on 2021-01-16


<details><summary><b>Visual Speech Enhancement Without A Real Visual Stream</b>
<a href="https://arxiv.org/abs/2012.10852">arxiv:2012.10852</a>
&#x1F4C8; 495 <br>
<p>Sindhu B Hegde, K R Prajwal, Rudrabha Mukhopadhyay, Vinay Namboodiri, C. V. Jawahar</p></summary>
<p>

**Abstract:** In this work, we re-think the task of speech enhancement in unconstrained real-world environments. Current state-of-the-art methods use only the audio stream and are limited in their performance in a wide range of real-world noises. Recent works using lip movements as additional cues improve the quality of generated speech over "audio-only" methods. But, these methods cannot be used for several applications where the visual stream is unreliable or completely absent. We propose a new paradigm for speech enhancement by exploiting recent breakthroughs in speech-driven lip synthesis. Using one such model as a teacher network, we train a robust student network to produce accurate lip movements that mask away the noise, thus acting as a "visual noise filter". The intelligibility of the speech enhanced by our pseudo-lip approach is comparable (< 3% difference) to the case of using real lips. This implies that we can exploit the advantages of using lip movements even in the absence of a real video stream. We rigorously evaluate our model using quantitative metrics as well as human evaluations. Additional ablation studies and a demo video on our website containing qualitative comparisons and results clearly illustrate the effectiveness of our approach. We provide a demo video which clearly illustrates the effectiveness of our proposed approach on our website: \url{http://cvit.iiit.ac.in/research/projects/cvit-projects/visual-speech-enhancement-without-a-real-visual-stream}. The code and models are also released for future research: \url{https://github.com/Sindhu-Hegde/pseudo-visual-speech-denoising}.

</p>
</details>

<details><summary><b>Monte-Carlo Graph Search for AlphaZero</b>
<a href="https://arxiv.org/abs/2012.11045">arxiv:2012.11045</a>
&#x1F4C8; 227 <br>
<p>Johannes Czech, Patrick Korus, Kristian Kersting</p></summary>
<p>

**Abstract:** The AlphaZero algorithm has been successfully applied in a range of discrete domains, most notably board games. It utilizes a neural network, that learns a value and policy function to guide the exploration in a Monte-Carlo Tree Search. Although many search improvements have been proposed for Monte-Carlo Tree Search in the past, most of them refer to an older variant of the Upper Confidence bounds for Trees algorithm that does not use a policy for planning. We introduce a new, improved search algorithm for AlphaZero which generalizes the search tree to a directed acyclic graph. This enables information flow across different subtrees and greatly reduces memory consumption. Along with Monte-Carlo Graph Search, we propose a number of further extensions, such as the inclusion of Epsilon-greedy exploration, a revised terminal solver and the integration of domain knowledge as constraints. In our evaluations, we use the CrazyAra engine on chess and crazyhouse as examples to show that these changes bring significant improvements to AlphaZero.

</p>
</details>

<details><summary><b>Recent advances in deep learning theory</b>
<a href="https://arxiv.org/abs/2012.10931">arxiv:2012.10931</a>
&#x1F4C8; 67 <br>
<p>Fengxiang He, Dacheng Tao</p></summary>
<p>

**Abstract:** Deep learning is usually described as an experiment-driven field under continuous criticizes of lacking theoretical foundations. This problem has been partially fixed by a large volume of literature which has so far not been well organized. This paper reviews and organizes the recent advances in deep learning theory. The literature is categorized in six groups: (1) complexity and capacity-based approaches for analyzing the generalizability of deep learning; (2) stochastic differential equations and their dynamic systems for modelling stochastic gradient descent and its variants, which characterize the optimization and generalization of deep learning, partially inspired by Bayesian inference; (3) the geometrical structures of the loss landscape that drives the trajectories of the dynamic systems; (4) the roles of over-parameterization of deep neural networks from both positive and negative perspectives; (5) theoretical foundations of several special structures in network architectures; and (6) the increasingly intensive concerns in ethics and security and their relationships with generalizability.

</p>
</details>

<details><summary><b>LieTransformer: Equivariant self-attention for Lie Groups</b>
<a href="https://arxiv.org/abs/2012.10885">arxiv:2012.10885</a>
&#x1F4C8; 50 <br>
<p>Michael Hutchinson, Charline Le Lan, Sheheryar Zaidi, Emilien Dupont, Yee Whye Teh, Hyunjik Kim</p></summary>
<p>

**Abstract:** Group equivariant neural networks are used as building blocks of group invariant neural networks, which have been shown to improve generalisation performance and data efficiency through principled parameter sharing. Such works have mostly focused on group equivariant convolutions, building on the result that group equivariant linear maps are necessarily convolutions. In this work, we extend the scope of the literature to non-linear neural network modules, namely self-attention, that is emerging as a prominent building block of deep learning models. We propose the LieTransformer, an architecture composed of LieSelfAttention layers that are equivariant to arbitrary Lie groups and their discrete subgroups. We demonstrate the generality of our approach by showing experimental results that are competitive to baseline methods on a wide range of tasks: shape counting on point clouds, molecular property regression and modelling particle trajectories under Hamiltonian dynamics.

</p>
</details>

<details><summary><b>Learning to Localize Using a LiDAR Intensity Map</b>
<a href="https://arxiv.org/abs/2012.10902">arxiv:2012.10902</a>
&#x1F4C8; 9 <br>
<p>Ioan Andrei Bârsan, Shenlong Wang, Andrei Pokrovsky, Raquel Urtasun</p></summary>
<p>

**Abstract:** In this paper we propose a real-time, calibration-agnostic and effective localization system for self-driving cars. Our method learns to embed the online LiDAR sweeps and intensity map into a joint deep embedding space. Localization is then conducted through an efficient convolutional matching between the embeddings. Our full system can operate in real-time at 15Hz while achieving centimeter level accuracy across different LiDAR sensors and environments. Our experiments illustrate the performance of the proposed approach over a large-scale dataset consisting of over 4000km of driving.

</p>
</details>

<details><summary><b>Learning to Localize Through Compressed Binary Maps</b>
<a href="https://arxiv.org/abs/2012.10942">arxiv:2012.10942</a>
&#x1F4C8; 8 <br>
<p>Xinkai Wei, Ioan Andrei Bârsan, Shenlong Wang, Julieta Martinez, Raquel Urtasun</p></summary>
<p>

**Abstract:** One of the main difficulties of scaling current localization systems to large environments is the on-board storage required for the maps. In this paper we propose to learn to compress the map representation such that it is optimal for the localization task. As a consequence, higher compression rates can be achieved without loss of localization accuracy when compared to standard coding schemes that optimize for reconstruction, thus ignoring the end task. Our experiments show that it is possible to learn a task-specific compression which reduces storage requirements by two orders of magnitude over general-purpose codecs such as WebP without sacrificing performance.

</p>
</details>

<details><summary><b>Fusion of CNNs and statistical indicators to improve image classification</b>
<a href="https://arxiv.org/abs/2012.11049">arxiv:2012.11049</a>
&#x1F4C8; 7 <br>
<p>Javier Huertas-Tato, Alejandro Martín, Julian Fierrez, David Camacho</p></summary>
<p>

**Abstract:** Convolutional Networks have dominated the field of computer vision for the last ten years, exhibiting extremely powerful feature extraction capabilities and outstanding classification performance. The main strategy to prolong this trend relies on further upscaling networks in size. However, costs increase rapidly while performance improvements may be marginal. We hypothesise that adding heterogeneous sources of information may be more cost-effective to a CNN than building a bigger network. In this paper, an ensemble method is proposed for accurate image classification, fusing automatically detected features through Convolutional Neural Network architectures with a set of manually defined statistical indicators. Through a combination of the predictions of a CNN and a secondary classifier trained on statistical features, better classification performance can be cheaply achieved. We test multiple learning algorithms and CNN architectures on a diverse number of datasets to validate our proposal, making public all our code and data via GitHub. According to our results, the inclusion of additional indicators and an ensemble classification approach helps to increase the performance in 8 of 9 datasets, with a remarkable increase of more than 10% precision in two of them.

</p>
</details>

<details><summary><b>High-Fidelity Neural Human Motion Transfer from Monocular Video</b>
<a href="https://arxiv.org/abs/2012.10974">arxiv:2012.10974</a>
&#x1F4C8; 6 <br>
<p>Moritz Kappel, Vladislav Golyanik, Mohamed Elgharib, Jann-Ole Henningson, Hans-Peter Seidel, Susana Castillo, Christian Theobalt, Marcus Magnor</p></summary>
<p>

**Abstract:** Video-based human motion transfer creates video animations of humans following a source motion. Current methods show remarkable results for tightly-clad subjects. However, the lack of temporally consistent handling of plausible clothing dynamics, including fine and high-frequency details, significantly limits the attainable visual quality. We address these limitations for the first time in the literature and present a new framework which performs high-fidelity and temporally-consistent human motion transfer with natural pose-dependent non-rigid deformations, for several types of loose garments. In contrast to the previous techniques, we perform image generation in three subsequent stages, synthesizing human shape, structure, and appearance. Given a monocular RGB video of an actor, we train a stack of recurrent deep neural networks that generate these intermediate representations from 2D poses and their temporal derivatives. Splitting the difficult motion transfer problem into subtasks that are aware of the temporal motion context helps us to synthesize results with plausible dynamics and pose-dependent detail. It also allows artistic control of results by manipulation of individual framework stages. In the experimental results, we significantly outperform the state-of-the-art in terms of video realism. Our code and data will be made publicly available.

</p>
</details>

<details><summary><b>DeepKeyGen: A Deep Learning-based Stream Cipher Generator for Medical Image Encryption and Decryption</b>
<a href="https://arxiv.org/abs/2012.11097">arxiv:2012.11097</a>
&#x1F4C8; 5 <br>
<p>Yi Ding, Fuyuan Tan, Zhen Qin, Mingsheng Cao, Kim-Kwang Raymond Choo, Zhiguang Qin</p></summary>
<p>

**Abstract:** The need for medical image encryption is increasingly pronounced, for example to safeguard the privacy of the patients' medical imaging data. In this paper, a novel deep learning-based key generation network (DeepKeyGen) is proposed as a stream cipher generator to generate the private key, which can then be used for encrypting and decrypting of medical images. In DeepKeyGen, the generative adversarial network (GAN) is adopted as the learning network to generate the private key. Furthermore, the transformation domain (that represents the "style" of the private key to be generated) is designed to guide the learning network to realize the private key generation process. The goal of DeepKeyGen is to learn the mapping relationship of how to transfer the initial image to the private key. We evaluate DeepKeyGen using three datasets, namely: the Montgomery County chest X-ray dataset, the Ultrasonic Brachial Plexus dataset, and the BraTS18 dataset. The evaluation findings and security analysis show that the proposed key generation network can achieve a high-level security in generating the private key.

</p>
</details>

<details><summary><b>Can Everybody Sign Now? Exploring Sign Language Video Generation from 2D Poses</b>
<a href="https://arxiv.org/abs/2012.10941">arxiv:2012.10941</a>
&#x1F4C8; 5 <br>
<p>Lucas Ventura, Amanda Duarte, Xavier Giro-i-Nieto</p></summary>
<p>

**Abstract:** Recent work have addressed the generation of human poses represented by 2D/3D coordinates of human joints for sign language. We use the state of the art in Deep Learning for motion transfer and evaluate them on How2Sign, an American Sign Language dataset, to generate videos of signers performing sign language given a 2D pose skeleton. We evaluate the generated videos quantitatively and qualitatively showing that the current models are not enough to generated adequate videos for Sign Language due to lack of detail in hands.

</p>
</details>

<details><summary><b>A Graph Reasoning Network for Multi-turn Response Selection via Customized Pre-training</b>
<a href="https://arxiv.org/abs/2012.11099">arxiv:2012.11099</a>
&#x1F4C8; 4 <br>
<p>Yongkang Liu, Shi Feng, Daling Wang, Kaisong Song, Feiliang Ren, Yifei Zhang</p></summary>
<p>

**Abstract:** We investigate response selection for multi-turn conversation in retrieval-based chatbots. Existing studies pay more attention to the matching between utterances and responses by calculating the matching score based on learned features, leading to insufficient model reasoning ability. In this paper, we propose a graph-reasoning network (GRN) to address the problem. GRN first conducts pre-training based on ALBERT using next utterance prediction and utterance order prediction tasks specifically devised for response selection. These two customized pre-training tasks can endow our model with the ability of capturing semantical and chronological dependency between utterances. We then fine-tune the model on an integrated network with sequence reasoning and graph reasoning structures. The sequence reasoning module conducts inference based on the highly summarized context vector of utterance-response pairs from the global perspective. The graph reasoning module conducts the reasoning on the utterance-level graph neural network from the local perspective. Experiments on two conversational reasoning datasets show that our model can dramatically outperform the strong baseline methods and can achieve performance which is close to human-level.

</p>
</details>

<details><summary><b>PPGN: Phrase-Guided Proposal Generation Network For Referring Expression Comprehension</b>
<a href="https://arxiv.org/abs/2012.10890">arxiv:2012.10890</a>
&#x1F4C8; 4 <br>
<p>Chao Yang, Guoqing Wang, Dongsheng Li, Huawei Shen, Su Feng, Bin Jiang</p></summary>
<p>

**Abstract:** Reference expression comprehension (REC) aims to find the location that the phrase refer to in a given image. Proposal generation and proposal representation are two effective techniques in many two-stage REC methods. However, most of the existing works only focus on proposal representation and neglect the importance of proposal generation. As a result, the low-quality proposals generated by these methods become the performance bottleneck in REC tasks. In this paper, we reconsider the problem of proposal generation, and propose a novel phrase-guided proposal generation network (PPGN). The main implementation principle of PPGN is refining visual features with text and generate proposals through regression. Experiments show that our method is effective and achieve SOTA performance in benchmark datasets.

</p>
</details>

<details><summary><b>Unfolded Algorithms for Deep Phase Retrieval</b>
<a href="https://arxiv.org/abs/2012.11102">arxiv:2012.11102</a>
&#x1F4C8; 3 <br>
<p>Naveed Naimipour, Shahin Khobahi, Mojtaba Soltanalian</p></summary>
<p>

**Abstract:** Exploring the idea of phase retrieval has been intriguing researchers for decades, due to its appearance in a wide range of applications. The task of a phase retrieval algorithm is typically to recover a signal from linear phaseless measurements. In this paper, we approach the problem by proposing a hybrid model-based data-driven deep architecture, referred to as Unfolded Phase Retrieval (UPR), that exhibits significant potential in improving the performance of state-of-the art data-driven and model-based phase retrieval algorithms. The proposed method benefits from versatility and interpretability of well-established model-based algorithms, while simultaneously benefiting from the expressive power of deep neural networks. In particular, our proposed model-based deep architecture is applied to the conventional phase retrieval problem (via the incremental reshaped Wirtinger flow algorithm) and the sparse phase retrieval problem (via the sparse truncated amplitude flow algorithm), showing immense promise in both cases. Furthermore, we consider a joint design of the sensing matrix and the signal processing algorithm and utilize the deep unfolding technique in the process. Our numerical results illustrate the effectiveness of such hybrid model-based and data-driven frameworks and showcase the untapped potential of data-aided methodologies to enhance the existing phase retrieval algorithms.

</p>
</details>

<details><summary><b>Parameter Identification for Digital Fabrication: A Gaussian Process Learning Approach</b>
<a href="https://arxiv.org/abs/2012.11022">arxiv:2012.11022</a>
&#x1F4C8; 3 <br>
<p>Yvonne R. Stürz, Mohammad Khosravi, Roy S. Smith</p></summary>
<p>

**Abstract:** Tensioned cable nets can be used as supporting structures for the efficient construction of lightweight building elements, such as thin concrete shell structures. To guarantee important mechanical properties of the latter, the tolerances on deviations of the tensioned cable net geometry from the desired target form are very tight. Therefore, the form needs to be readjusted on the construction site. In order to employ model-based optimization techniques, the precise identification of important uncertain model parameters of the cable net system is required. This paper proposes the use of Gaussian process regression to learn the function that maps the cable net geometry to the uncertain parameters. In contrast to previously proposed methods, this approach requires only a single form measurement for the identification of the cable net model parameters. This is beneficial since measurements of the cable net form on the construction site are very expensive. For the training of the Gaussian processes, simulated data is efficiently computed via convex programming. The effectiveness of the proposed method and the impact of the precise identification of the parameters on the form of the cable net are demonstrated in numerical experiments on a quarter-scale prototype of a roof structure.

</p>
</details>

<details><summary><b>Post-hoc Uncertainty Calibration for Domain Drift Scenarios</b>
<a href="https://arxiv.org/abs/2012.10988">arxiv:2012.10988</a>
&#x1F4C8; 3 <br>
<p>Christian Tomani, Sebastian Gruber, Muhammed Ebrar Erdem, Daniel Cremers, Florian Buettner</p></summary>
<p>

**Abstract:** We address the problem of uncertainty calibration. While standard deep neural networks typically yield uncalibrated predictions, calibrated confidence scores that are representative of the true likelihood of a prediction can be achieved using post-hoc calibration methods. However, to date the focus of these approaches has been on in-domain calibration. Our contribution is two-fold. First, we show that existing post-hoc calibration methods yield highly over-confident predictions under domain shift. Second, we introduce a simple strategy where perturbations are applied to samples in the validation set before performing the post-hoc calibration step. In extensive experiments, we demonstrate that this perturbation step results in substantially better calibration under domain shift on a wide range of architectures and modelling tasks.

</p>
</details>

<details><summary><b>Towards Trustworthy Predictions from Deep Neural Networks with Fast Adversarial Calibration</b>
<a href="https://arxiv.org/abs/2012.10923">arxiv:2012.10923</a>
&#x1F4C8; 3 <br>
<p>Christian Tomani, Florian Buettner</p></summary>
<p>

**Abstract:** To facilitate a wide-spread acceptance of AI systems guiding decision making in real-world applications, trustworthiness of deployed models is key. That is, it is crucial for predictive models to be uncertainty-aware and yield well-calibrated (and thus trustworthy) predictions for both in-domain samples as well as under domain shift. Recent efforts to account for predictive uncertainty include post-processing steps for trained neural networks, Bayesian neural networks as well as alternative non-Bayesian approaches such as ensemble approaches and evidential deep learning. Here, we propose an efficient yet general modelling approach for obtaining well-calibrated, trustworthy probabilities for samples obtained after a domain shift. We introduce a new training strategy combining an entropy-encouraging loss term with an adversarial calibration loss term and demonstrate that this results in well-calibrated and technically trustworthy predictions for a wide range of domain drifts. We comprehensively evaluate previously proposed approaches on different data modalities, a large range of data sets including sequence data, network architectures and perturbation strategies. We observe that our modelling approach substantially outperforms existing state-of-the-art approaches, yielding well-calibrated predictions under domain drift.

</p>
</details>

<details><summary><b>Memory Approximate Message Passing</b>
<a href="https://arxiv.org/abs/2012.10861">arxiv:2012.10861</a>
&#x1F4C8; 3 <br>
<p>Lei Liu, Shunqi Huang, Brian M. Kurkoski</p></summary>
<p>

**Abstract:** Approximate message passing (AMP) is a low-cost iterative parameter-estimation technique for certain high-dimensional linear systems with non-Gaussian distributions. However, AMP only applies to the independent identically distributed (IID) transform matrices, but may become unreliable for other matrix ensembles, especially for ill-conditioned ones. To handle this difficulty, orthogonal/vector AMP (OAMP/VAMP) was proposed for general unitarily-invariant matrices, including IID matrices and partial orthogonal matrices. However, the Bayes-optimal OAMP/VAMP requires high-complexity linear minimum mean square error (MMSE) estimator. This limits the application of OAMP/VAMP to large-scale systems.
  To solve the disadvantages of AMP and OAMP/VAMP, this paper proposes a low-complexity memory AMP (MAMP) for unitarily-invariant matrices. MAMP is consisted of an orthogonal non-linear estimator (NLE) for denoising (same as OAMP/VAMP), and an orthogonal long-memory matched filter (MF) for interference suppression. Orthogonal principle is used to guarantee the asymptotic Gaussianity of estimation errors in MAMP. A state evolution is derived to asymptotically characterize the performance of MAMP. The relaxation parameters and damping vector in MAMP are analytically optimized based on the state evolution to guarantee and improve the convergence. We show that MAMP has comparable complexity to AMP. Furthermore, we prove that for all unitarily-invariant matrices, the optimized MAMP converges to the high-complexity OAMP/VAMP, and thus is Bayes-optimal if it has a unique fixed point. Finally, simulations are provided to verify the validity and accuracy of the theoretical results.

</p>
</details>

<details><summary><b>Color Channel Perturbation Attacks for Fooling Convolutional Neural Networks and A Defense Against Such Attacks</b>
<a href="https://arxiv.org/abs/2012.14456">arxiv:2012.14456</a>
&#x1F4C8; 2 <br>
<p>Jayendra Kantipudi, Shiv Ram Dubey, Soumendu Chakraborty</p></summary>
<p>

**Abstract:** The Convolutional Neural Networks (CNNs) have emerged as a very powerful data dependent hierarchical feature extraction method. It is widely used in several computer vision problems. The CNNs learn the important visual features from training samples automatically. It is observed that the network overfits the training samples very easily. Several regularization methods have been proposed to avoid the overfitting. In spite of this, the network is sensitive to the color distribution within the images which is ignored by the existing approaches. In this paper, we discover the color robustness problem of CNN by proposing a Color Channel Perturbation (CCP) attack to fool the CNNs. In CCP attack new images are generated with new channels created by combining the original channels with the stochastic weights. Experiments were carried out over widely used CIFAR10, Caltech256 and TinyImageNet datasets in the image classification framework. The VGG, ResNet and DenseNet models are used to test the impact of the proposed attack. It is observed that the performance of the CNNs degrades drastically under the proposed CCP attack. Result show the effect of the proposed simple CCP attack over the robustness of the CNN trained model. The results are also compared with existing CNN fooling approaches to evaluate the accuracy drop. We also propose a primary defense mechanism to this problem by augmenting the training dataset with the proposed CCP attack. The state-of-the-art performance using the proposed solution in terms of the CNN robustness under CCP attack is observed in the experiments. The code is made publicly available at \url{https://github.com/jayendrakantipudi/Color-Channel-Perturbation-Attack}.

</p>
</details>

<details><summary><b>TSEQPREDICTOR: Spatiotemporal Extreme Earthquakes Forecasting for Southern California</b>
<a href="https://arxiv.org/abs/2012.14336">arxiv:2012.14336</a>
&#x1F4C8; 2 <br>
<p>Bo Feng, Geoffrey C. Fox</p></summary>
<p>

**Abstract:** Seismology from the past few decades has utilized the most advanced technologies and equipment to monitor seismic events globally. However, forecasting disasters like earthquakes is still an underdeveloped topic from the history. Recent researches in spatiotemporal forecasting have revealed some possibilities of successful predictions, which becomes an important topic in many scientific research fields. Most studies of them have many successful applications of using deep neural networks. In the geoscience study, earthquake prediction is one of the world's most challenging problems, about which cutting edge deep learning technologies may help to discover some useful patterns. In this project, we propose a joint deep learning modeling method for earthquake forecasting, namely TSEQPREDICTOR. In TSEQPREDICTOR, we use comprehensive deep learning technologies with domain knowledge in seismology and exploit the prediction problem using encoder-decoder and temporal convolutional neural networks. Comparing to some state-of-art recurrent neural networks, our experiments show our method is promising in terms of predicting major shocks for earthquakes in Southern California.

</p>
</details>

<details><summary><b>Complexity of zigzag sampling algorithm for strongly log-concave distributions</b>
<a href="https://arxiv.org/abs/2012.11094">arxiv:2012.11094</a>
&#x1F4C8; 2 <br>
<p>Jianfeng Lu, Lihan Wang</p></summary>
<p>

**Abstract:** We study the computational complexity of zigzag sampling algorithm for strongly log-concave distributions. The zigzag process has the advantage of not requiring time discretization for implementation, and that each proposed bouncing event requires only one evaluation of partial derivative of the potential, while its convergence rate is dimension independent. Using these properties, we prove that the zigzag sampling algorithm achieves $\varepsilon$ error in chi-square divergence with a computational cost equivalent to $O\bigl(κ^2 d^\frac{1}{2}(\log\frac{1}{\varepsilon})^{\frac{3}{2}}\bigr)$ gradient evaluations in the regime $κ\ll \frac{d}{\log d}$ under a warm start assumption, where $κ$ is the condition number and $d$ is the dimension.

</p>
</details>

<details><summary><b>On Relating 'Why?' and 'Why Not?' Explanations</b>
<a href="https://arxiv.org/abs/2012.11067">arxiv:2012.11067</a>
&#x1F4C8; 2 <br>
<p>Alexey Ignatiev, Nina Narodytska, Nicholas Asher, Joao Marques-Silva</p></summary>
<p>

**Abstract:** Explanations of Machine Learning (ML) models often address a 'Why?' question. Such explanations can be related with selecting feature-value pairs which are sufficient for the prediction. Recent work has investigated explanations that address a 'Why Not?' question, i.e. finding a change of feature values that guarantee a change of prediction. Given their goals, these two forms of explaining predictions of ML models appear to be mostly unrelated. However, this paper demonstrates otherwise, and establishes a rigorous formal relationship between 'Why?' and 'Why Not?' explanations. Concretely, the paper proves that, for any given instance, 'Why?' explanations are minimal hitting sets of 'Why Not?' explanations and vice-versa. Furthermore, the paper devises novel algorithms for extracting and enumerating both forms of explanations.

</p>
</details>

<details><summary><b>Fairness, Welfare, and Equity in Personalized Pricing</b>
<a href="https://arxiv.org/abs/2012.11066">arxiv:2012.11066</a>
&#x1F4C8; 2 <br>
<p>Nathan Kallus, Angela Zhou</p></summary>
<p>

**Abstract:** We study the interplay of fairness, welfare, and equity considerations in personalized pricing based on customer features. Sellers are increasingly able to conduct price personalization based on predictive modeling of demand conditional on covariates: setting customized interest rates, targeted discounts of consumer goods, and personalized subsidies of scarce resources with positive externalities like vaccines and bed nets. These different application areas may lead to different concerns around fairness, welfare, and equity on different objectives: price burdens on consumers, price envy, firm revenue, access to a good, equal access, and distributional consequences when the good in question further impacts downstream outcomes of interest. We conduct a comprehensive literature review in order to disentangle these different normative considerations and propose a taxonomy of different objectives with mathematical definitions. We focus on observational metrics that do not assume access to an underlying valuation distribution which is either unobserved due to binary feedback or ill-defined due to overriding behavioral concerns regarding interpreting revealed preferences. In the setting of personalized pricing for the provision of goods with positive benefits, we discuss how price optimization may provide unambiguous benefit by achieving a "triple bottom line": personalized pricing enables expanding access, which in turn may lead to gains in welfare due to heterogeneous utility, and improve revenue or budget utilization. We empirically demonstrate the potential benefits of personalized pricing in two settings: pricing subsidies for an elective vaccine, and the effects of personalized interest rates on downstream outcomes in microcredit.

</p>
</details>

<details><summary><b>Bayesian Semi-supervised Crowdsourcing</b>
<a href="https://arxiv.org/abs/2012.11048">arxiv:2012.11048</a>
&#x1F4C8; 2 <br>
<p>Panagiotis A. Traganitis, Georgios B. Giannakis</p></summary>
<p>

**Abstract:** Crowdsourcing has emerged as a powerful paradigm for efficiently labeling large datasets and performing various learning tasks, by leveraging crowds of human annotators. When additional information is available about the data, semi-supervised crowdsourcing approaches that enhance the aggregation of labels from human annotators are well motivated. This work deals with semi-supervised crowdsourced classification, under two regimes of semi-supervision: a) label constraints, that provide ground-truth labels for a subset of data; and b) potentially easier to obtain instance-level constraints, that indicate relationships between pairs of data. Bayesian algorithms based on variational inference are developed for each regime, and their quantifiably improved performance, compared to unsupervised crowdsourcing, is analytically and empirically validated on several crowdsourcing datasets.

</p>
</details>

<details><summary><b>Privacy Analysis of Online Learning Algorithms via Contraction Coefficients</b>
<a href="https://arxiv.org/abs/2012.11035">arxiv:2012.11035</a>
&#x1F4C8; 2 <br>
<p>Shahab Asoodeh, Mario Diaz, Flavio P. Calmon</p></summary>
<p>

**Abstract:** We propose an information-theoretic technique for analyzing privacy guarantees of online algorithms. Specifically, we demonstrate that differential privacy guarantees of iterative algorithms can be determined by a direct application of contraction coefficients derived from strong data processing inequalities for $f$-divergences. Our technique relies on generalizing the Dobrushin's contraction coefficient for total variation distance to an $f$-divergence known as $E_γ$-divergence. $E_γ$-divergence, in turn, is equivalent to approximate differential privacy. As an example, we apply our technique to derive the differential privacy parameters of gradient descent. Moreover, we also show that this framework can be tailored to batch learning algorithms that can be implemented with one pass over the training dataset.

</p>
</details>

<details><summary><b>DISCO: Dynamic and Invariant Sensitive Channel Obfuscation for deep neural networks</b>
<a href="https://arxiv.org/abs/2012.11025">arxiv:2012.11025</a>
&#x1F4C8; 2 <br>
<p>Abhishek Singh, Ayush Chopra, Vivek Sharma, Ethan Garza, Emily Zhang, Praneeth Vepakomma, Ramesh Raskar</p></summary>
<p>

**Abstract:** Recent deep learning models have shown remarkable performance in image classification. While these deep learning systems are getting closer to practical deployment, the common assumption made about data is that it does not carry any sensitive information. This assumption may not hold for many practical cases, especially in the domain where an individual's personal information is involved, like healthcare and facial recognition systems. We posit that selectively removing features in this latent space can protect the sensitive information and provide a better privacy-utility trade-off. Consequently, we propose DISCO which learns a dynamic and data driven pruning filter to selectively obfuscate sensitive information in the feature space. We propose diverse attack schemes for sensitive inputs \& attributes and demonstrate the effectiveness of DISCO against state-of-the-art methods through quantitative and qualitative evaluation. Finally, we also release an evaluation benchmark dataset of 1 million sensitive representations to encourage rigorous exploration of novel attack schemes.

</p>
</details>

<details><summary><b>Biased Models Have Biased Explanations</b>
<a href="https://arxiv.org/abs/2012.10986">arxiv:2012.10986</a>
&#x1F4C8; 2 <br>
<p>Aditya Jain, Manish Ravula, Joydeep Ghosh</p></summary>
<p>

**Abstract:** We study fairness in Machine Learning (FairML) through the lens of attribute-based explanations generated for machine learning models. Our hypothesis is: Biased Models have Biased Explanations. To establish that, we first translate existing statistical notions of group fairness and define these notions in terms of explanations given by the model. Then, we propose a novel way of detecting (un)fairness for any black box model. We further look at post-processing techniques for fairness and reason how explanations can be used to make a bias mitigation technique more individually fair. We also introduce a novel post-processing mitigation technique which increases individual fairness in recourse while maintaining group level fairness.

</p>
</details>

<details><summary><b>Learning Halfspaces With Membership Queries</b>
<a href="https://arxiv.org/abs/2012.10985">arxiv:2012.10985</a>
&#x1F4C8; 2 <br>
<p>Ori Kelner</p></summary>
<p>

**Abstract:** Active learning is a subfield of machine learning, in which the learning algorithm is allowed to choose the data from which it learns. In some cases, it has been shown that active learning can yield an exponential gain in the number of samples the algorithm needs to see, in order to reach generalization error $\leq ε$. In this work we study the problem of learning halfspaces with membership queries. In the membership query scenario, we allow the learning algorithm to ask for the label of every sample in the input space. We suggest a new algorithm for this problem, and prove it achieves a near optimal label complexity in some cases. We also show that the algorithm works well in practice, and significantly outperforms uncertainty sampling.

</p>
</details>

<details><summary><b>Auto-Encoded Reservoir Computing for Turbulence Learning</b>
<a href="https://arxiv.org/abs/2012.10968">arxiv:2012.10968</a>
&#x1F4C8; 2 <br>
<p>Nguyen Anh Khoa Doan, Wolfgang Polifke, Luca Magri</p></summary>
<p>

**Abstract:** We present an Auto-Encoded Reservoir-Computing (AE-RC) approach to learn the dynamics of a 2D turbulent flow. The AE-RC consists of a Convolutional Autoencoder, which discovers an efficient manifold representation of the flow state, and an Echo State Network, which learns the time evolution of the flow in the manifold. The AE-RC is able to both learn the time-accurate dynamics of the turbulent flow and predict its first-order statistical moments. The AE-RC approach opens up new possibilities for the spatio-temporal prediction of turbulent flows with machine learning.

</p>
</details>

<details><summary><b>Recent Developments in Detection of Central Serous Retinopathy through Imaging and Artificial Intelligence Techniques A Review</b>
<a href="https://arxiv.org/abs/2012.10961">arxiv:2012.10961</a>
&#x1F4C8; 2 <br>
<p>Syed Ale Hassan, Shahzad Akbar, Amjad Rehman, Tanzila Saba, Rashid Abbasi</p></summary>
<p>

**Abstract:** The Central Serous Retinopathy (CSR) is a major significant disease responsible for causing blindness and vision loss among numerous people across the globe. This disease is also known as the Central Serous Chorioretinopathy (CSC) occurs due to the accumulation of watery fluids behind the retina. The detection of CSR at an early stage allows taking preventive measures to avert any impairment to the human eye. Traditionally, several manual detection methods were developed for observing CSR, but they were proven to be inaccurate, unreliable, and time-consuming. Consequently, the research community embarked on seeking automated solutions for CSR detection. With the advent of modern technology in the 21st century, Artificial Intelligence (AI) techniques are immensely popular in numerous research fields including the automated CSR detection. This paper offers a comprehensive review of various advanced technologies and researches, contributing to the automated CSR detection in this scenario. Additionally, it discusses the benefits and limitations of many classical imaging methods ranging from Optical Coherence Tomography (OCT) and the Fundus imaging, to more recent approaches like AI based Machine/Deep Learning techniques. Study primary objective is to analyze and compare many Artificial Intelligence (AI) algorithms that have efficiently achieved automated CSR detection using OCT imaging. Furthermore, it describes various retinal datasets and strategies proposed for CSR assessment and accuracy. Finally, it is concluded that the most recent Deep Learning (DL) classifiers are performing accurate, fast, and reliable detection of CSR.

</p>
</details>

<details><summary><b>SPlit: An Optimal Method for Data Splitting</b>
<a href="https://arxiv.org/abs/2012.10945">arxiv:2012.10945</a>
&#x1F4C8; 2 <br>
<p>V. Roshan Joseph, Akhil Vakayil</p></summary>
<p>

**Abstract:** In this article we propose an optimal method referred to as SPlit for splitting a dataset into training and testing sets. SPlit is based on the method of Support Points (SP), which was initially developed for finding the optimal representative points of a continuous distribution. We adapt SP for subsampling from a dataset using a sequential nearest neighbor algorithm. We also extend SP to deal with categorical variables so that SPlit can be applied to both regression and classification problems. The implementation of SPlit on real datasets shows substantial improvement in the worst-case testing performance for several modeling methods compared to the commonly used random splitting procedure.

</p>
</details>

<details><summary><b>Automated Clustering of High-dimensional Data with a Feature Weighted Mean Shift Algorithm</b>
<a href="https://arxiv.org/abs/2012.10929">arxiv:2012.10929</a>
&#x1F4C8; 2 <br>
<p>Saptarshi Chakraborty, Debolina Paul, Swagatam Das</p></summary>
<p>

**Abstract:** Mean shift is a simple interactive procedure that gradually shifts data points towards the mode which denotes the highest density of data points in the region. Mean shift algorithms have been effectively used for data denoising, mode seeking, and finding the number of clusters in a dataset in an automated fashion. However, the merits of mean shift quickly fade away as the data dimensions increase and only a handful of features contain useful information about the cluster structure of the data. We propose a simple yet elegant feature-weighted variant of mean shift to efficiently learn the feature importance and thus, extending the merits of mean shift to high-dimensional data. The resulting algorithm not only outperforms the conventional mean shift clustering procedure but also preserves its computational simplicity. In addition, the proposed method comes with rigorous theoretical convergence guarantees and a convergence rate of at least a cubic order. The efficacy of our proposal is thoroughly assessed through experimental comparison against baseline and state-of-the-art clustering methods on synthetic as well as real-world datasets.

</p>
</details>

<details><summary><b>Adaptive Bi-directional Attention: Exploring Multi-Granularity Representations for Machine Reading Comprehension</b>
<a href="https://arxiv.org/abs/2012.10877">arxiv:2012.10877</a>
&#x1F4C8; 2 <br>
<p>Nuo Chen, Fenglin Liu, Chenyu You, Peilin Zhou, Yuexian Zou</p></summary>
<p>

**Abstract:** Recently, the attention-enhanced multi-layer encoder, such as Transformer, has been extensively studied in Machine Reading Comprehension (MRC). To predict the answer, it is common practice to employ a predictor to draw information only from the final encoder layer which generates the \textit{coarse-grained} representations of the source sequences, i.e., passage and question. Previous studies have shown that the representation of source sequence becomes more \textit{coarse-grained} from \textit{fine-grained} as the encoding layer increases. It is generally believed that with the growing number of layers in deep neural networks, the encoding process will gather relevant information for each location increasingly, resulting in more \textit{coarse-grained} representations, which adds the likelihood of similarity to other locations (referring to homogeneity). Such a phenomenon will mislead the model to make wrong judgments so as to degrade the performance. To this end, we propose a novel approach called Adaptive Bidirectional Attention, which adaptively exploits the source representations of different levels to the predictor. Experimental results on the benchmark dataset, SQuAD 2.0 demonstrate the effectiveness of our approach, and the results are better than the previous state-of-the-art model by 2.5$\%$ EM and 2.3$\%$ F1 scores.

</p>
</details>

<details><summary><b>Reinforcement Learning-based Product Delivery Frequency Control</b>
<a href="https://arxiv.org/abs/2012.10858">arxiv:2012.10858</a>
&#x1F4C8; 2 <br>
<p>Yang Liu, Zhengxing Chen, Kittipat Virochsiri, Juan Wang, Jiahao Wu, Feng Liang</p></summary>
<p>

**Abstract:** Frequency control is an important problem in modern recommender systems. It dictates the delivery frequency of recommendations to maintain product quality and efficiency. For example, the frequency of delivering promotional notifications impacts daily metrics as well as the infrastructure resource consumption (e.g. CPU and memory usage). There remain open questions on what objective we should optimize to represent business values in the long term best, and how we should balance between daily metrics and resource consumption in a dynamically fluctuating environment. We propose a personalized methodology for the frequency control problem, which combines long-term value optimization using reinforcement learning (RL) with a robust volume control technique we termed "Effective Factor". We demonstrate statistically significant improvement in daily metrics and resource efficiency by our method in several notification applications at a scale of billions of users. To our best knowledge, our study represents the first deep RL application on the frequency control problem at such an industrial scale.

</p>
</details>

<details><summary><b>eTREE: Learning Tree-structured Embeddings</b>
<a href="https://arxiv.org/abs/2012.10853">arxiv:2012.10853</a>
&#x1F4C8; 2 <br>
<p>Faisal M. Almutairi, Yunlong Wang, Dong Wang, Emily Zhao, Nicholas D. Sidiropoulos</p></summary>
<p>

**Abstract:** Matrix factorization (MF) plays an important role in a wide range of machine learning and data mining models. MF is commonly used to obtain item embeddings and feature representations due to its ability to capture correlations and higher-order statistical dependencies across dimensions. In many applications, the categories of items exhibit a hierarchical tree structure. For instance, human diseases can be divided into coarse categories, e.g., bacterial, and viral. These categories can be further divided into finer categories, e.g., viral infections can be respiratory, gastrointestinal, and exanthematous viral diseases. In e-commerce, products, movies, books, etc., are grouped into hierarchical categories, e.g., clothing items are divided by gender, then by type (formal, casual, etc.). While the tree structure and the categories of the different items may be known in some applications, they have to be learned together with the embeddings in many others. In this work, we propose eTREE, a model that incorporates the (usually ignored) tree structure to enhance the quality of the embeddings. We leverage the special uniqueness properties of Nonnegative MF (NMF) to prove identifiability of eTREE. The proposed model not only exploits the tree structure prior, but also learns the hierarchical clustering in an unsupervised data-driven fashion. We derive an efficient algorithmic solution and a scalable implementation of eTREE that exploits parallel computing, computation caching, and warm start strategies. We showcase the effectiveness of eTREE on real data from various application domains: healthcare, recommender systems, and education. We also demonstrate the meaningfulness of the tree obtained from eTREE by means of domain experts interpretation.

</p>
</details>

<details><summary><b>Resting-state EEG sex classification using selected brain connectivity representation</b>
<a href="https://arxiv.org/abs/2012.11105">arxiv:2012.11105</a>
&#x1F4C8; 1 <br>
<p>Jean Li, Jeremiah D. Deng, Divya Adhia, Dirk de Ridder</p></summary>
<p>

**Abstract:** Effective analysis of EEG signals for potential clinical applications remains a challenging task. So far, the analysis and conditioning of EEG have largely remained sex-neutral. This paper employs a machine learning approach to explore the evidence of sex effects on EEG signals, and confirms the generality of these effects by achieving successful sex prediction of resting-state EEG signals. We have found that the brain connectivity represented by the coherence between certain sensor channels are good predictors of sex.

</p>
</details>

<details><summary><b>To Talk or to Work: Energy Efficient Federated Learning over Mobile Devices via the Weight Quantization and 5G Transmission Co-Design</b>
<a href="https://arxiv.org/abs/2012.11070">arxiv:2012.11070</a>
&#x1F4C8; 1 <br>
<p>Rui Chen, Liang Li, Kaiping Xue, Chi Zhang, Lingjia Liu, Miao Pan</p></summary>
<p>

**Abstract:** Federated learning (FL) is a new paradigm for large-scale learning tasks across mobile devices. However, practical FL deployment over resource constrained mobile devices confronts multiple challenges. For example, it is not clear how to establish an effective wireless network architecture to support FL over mobile devices. Besides, as modern machine learning models are more and more complex, the local on-device training/intermediate model update in FL is becoming too power hungry/radio resource intensive for mobile devices to afford. To address those challenges, in this paper, we try to bridge another recent surging technology, 5G, with FL, and develop a wireless transmission and weight quantization co-design for energy efficient FL over heterogeneous 5G mobile devices. Briefly, the 5G featured high data rate helps to relieve the severe communication concern, and the multi-access edge computing (MEC) in 5G provides a perfect network architecture to support FL. Under MEC architecture, we develop flexible weight quantization schemes to facilitate the on-device local training over heterogeneous 5G mobile devices. Observed the fact that the energy consumption of local computing is comparable to that of the model updates via 5G transmissions, we formulate the energy efficient FL problem into a mixed-integer programming problem to elaborately determine the quantization strategies and allocate the wireless bandwidth for heterogeneous 5G mobile devices. The goal is to minimize the overall FL energy consumption (computing + 5G transmissions) over 5G mobile devices while guaranteeing learning performance and training latency. Generalized Benders' Decomposition is applied to develop feasible solutions and extensive simulations are conducted to verify the effectiveness of the proposed scheme.

</p>
</details>

<details><summary><b>Domain-adaptive Fall Detection Using Deep Adversarial Training</b>
<a href="https://arxiv.org/abs/2012.10911">arxiv:2012.10911</a>
&#x1F4C8; 1 <br>
<p>Kai-Chun Liu, Michael Chan, Chia-Yeh Hsieh, Hsiang-Yun Huang, Chia-Tai Chan, Yu Tsao</p></summary>
<p>

**Abstract:** Fall detection (FD) systems are important assistive technologies for healthcare that can detect emergency fall events and alert caregivers. However, it is not easy to obtain large-scale annotated fall events with various specifications of sensors or sensor positions, during the implementation of accurate FD systems. Moreover, the knowledge obtained through machine learning has been restricted to tasks in the same domain. The mismatch between different domains might hinder the performance of FD systems. Cross-domain knowledge transfer is very beneficial for machine-learning based FD systems to train a reliable FD model with well-labeled data in new environments. In this study, we propose domain-adaptive fall detection (DAFD) using deep adversarial training (DAT) to tackle cross-domain problems, such as cross-position and cross-configuration. The proposed DAFD can transfer knowledge from the source domain to the target domain by minimizing the domain discrepancy to avoid mismatch problems. The experimental results show that the average F1score improvement when using DAFD ranges from 1.5% to 7% in the cross-position scenario, and from 3.5% to 12% in the cross-configuration scenario, compared to using the conventional FD model without domain adaptation training. The results demonstrate that the proposed DAFD successfully helps to deal with cross-domain problems and to achieve better detection performance.

</p>
</details>

<details><summary><b>Towards Fair Personalization by Avoiding Feedback Loops</b>
<a href="https://arxiv.org/abs/2012.12862">arxiv:2012.12862</a>
&#x1F4C8; 0 <br>
<p>Gökhan Çapan, Özge Bozal, İlker Gündoğdu, Ali Taylan Cemgil</p></summary>
<p>

**Abstract:** Self-reinforcing feedback loops are both cause and effect of over and/or under-presentation of some content in interactive recommender systems. This leads to erroneous user preference estimates, namely, overestimation of over-presented content while violating the right to be presented of each alternative, contrary of which we define as a fair system. We consider two models that explicitly incorporate, or ignore the systematic and limited exposure to alternatives. By simulations, we demonstrate that ignoring the systematic presentations overestimates promoted options and underestimates censored alternatives. Simply conditioning on the limited exposure is a remedy for these biases.

</p>
</details>


[Next Page](2020/2020-12/2020-12-19.md)
