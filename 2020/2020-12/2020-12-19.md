## Summary for 2020-12-19, created on 2021-01-15


<details><summary><b>Sample Complexity of Adversarially Robust Linear Classification on Separated Data</b>
<a href="https://arxiv.org/abs/2012.10794">arxiv:2012.10794</a>
&#x1F4C8; 4 <br>
<p>Robi Bhattacharjee, Somesh Jha, Kamalika Chaudhuri</p></summary>
<p>

**Abstract:** We consider the sample complexity of learning with adversarial robustness. Most prior theoretical results for this problem have considered a setting where different classes in the data are close together or overlapping. Motivated by some real applications, we consider, in contrast, the well-separated case where there exists a classifier with perfect accuracy and robustness, and show that the sample complexity narrates an entirely different story. Specifically, for linear classifiers, we show a large class of well-separated distributions where the expected robust loss of any algorithm is at least $Ω(\frac{d}{n})$, whereas the max margin algorithm has expected standard loss $O(\frac{1}{n})$. This shows a gap in the standard and robust losses that cannot be obtained via prior techniques. Additionally, we present an algorithm that, given an instance where the robustness radius is much smaller than the gap between the classes, gives a solution with expected robust loss is $O(\frac{1}{n})$. This shows that for very well-separated data, convergence rates of $O(\frac{1}{n})$ are achievable, which is not the case otherwise. Our results apply to robustness measured in any $\ell_p$ norm with $p > 1$ (including $p = \infty$).

</p>
</details>

<details><summary><b>Uncertainty-Aware Policy Optimization: A Robust, Adaptive Trust Region Approach</b>
<a href="https://arxiv.org/abs/2012.10791">arxiv:2012.10791</a>
&#x1F4C8; 4 <br>
<p>James Queeney, Ioannis Ch. Paschalidis, Christos G. Cassandras</p></summary>
<p>

**Abstract:** In order for reinforcement learning techniques to be useful in real-world decision making processes, they must be able to produce robust performance from limited data. Deep policy optimization methods have achieved impressive results on complex tasks, but their real-world adoption remains limited because they often require significant amounts of data to succeed. When combined with small sample sizes, these methods can result in unstable learning due to their reliance on high-dimensional sample-based estimates. In this work, we develop techniques to control the uncertainty introduced by these estimates. We leverage these techniques to propose a deep policy optimization approach designed to produce stable performance even when data is scarce. The resulting algorithm, Uncertainty-Aware Trust Region Policy Optimization, generates robust policy updates that adapt to the level of uncertainty present throughout the learning process.

</p>
</details>

<details><summary><b>On (Emergent) Systematic Generalisation and Compositionality in Visual Referential Games with Straight-Through Gumbel-Softmax Estimator</b>
<a href="https://arxiv.org/abs/2012.10776">arxiv:2012.10776</a>
&#x1F4C8; 4 <br>
<p>Kevin Denamganaï, James Alfred Walker</p></summary>
<p>

**Abstract:** The drivers of compositionality in artificial languages that emerge when two (or more) agents play a non-visual referential game has been previously investigated using approaches based on the REINFORCE algorithm and the (Neural) Iterated Learning Model. Following the more recent introduction of the \textit{Straight-Through Gumbel-Softmax} (ST-GS) approach, this paper investigates to what extent the drivers of compositionality identified so far in the field apply in the ST-GS context and to what extent do they translate into (emergent) systematic generalisation abilities, when playing a visual referential game. Compositionality and the generalisation abilities of the emergent languages are assessed using topographic similarity and zero-shot compositional tests. Firstly, we provide evidence that the test-train split strategy significantly impacts the zero-shot compositional tests when dealing with visual stimuli, whilst it does not when dealing with symbolic ones. Secondly, empirical evidence shows that using the ST-GS approach with small batch sizes and an overcomplete communication channel improves compositionality in the emerging languages. Nevertheless, while shown robust with symbolic stimuli, the effect of the batch size is not so clear-cut when dealing with visual stimuli. Our results also show that not all overcomplete communication channels are created equal. Indeed, while increasing the maximum sentence length is found to be beneficial to further both compositionality and generalisation abilities, increasing the vocabulary size is found detrimental. Finally, a lack of correlation between the language compositionality at training-time and the agents' generalisation abilities is observed in the context of discriminative referential games with visual stimuli. This is similar to previous observations in the field using the generative variant with symbolic stimuli.

</p>
</details>

<details><summary><b>Fundamental Limits and Tradeoffs in Invariant Representation Learning</b>
<a href="https://arxiv.org/abs/2012.10713">arxiv:2012.10713</a>
&#x1F4C8; 4 <br>
<p>Han Zhao, Chen Dan, Bryon Aragam, Tommi S. Jaakkola, Geoffrey J. Gordon, Pradeep Ravikumar</p></summary>
<p>

**Abstract:** Many machine learning applications involve learning representations that achieve two competing goals: To maximize information or accuracy with respect to a subset of features (e.g.\ for prediction) while simultaneously maximizing invariance or independence with respect to another, potentially overlapping, subset of features (e.g.\ for fairness, privacy, etc). Typical examples include privacy-preserving learning, domain adaptation, and algorithmic fairness, just to name a few. In fact, all of the above problems admit a common minimax game-theoretic formulation, whose equilibrium represents a fundamental tradeoff between accuracy and invariance. Despite its abundant applications in the aforementioned domains, theoretical understanding on the limits and tradeoffs of invariant representations is severely lacking.
  In this paper, we provide an information-theoretic analysis of this general and important problem under both classification and regression settings. In both cases, we analyze the inherent tradeoffs between accuracy and invariance by providing a geometric characterization of the feasible region in the information plane, where we connect the geometric properties of this feasible region to the fundamental limitations of the tradeoff problem. In the regression setting, we also derive a tight lower bound on the Lagrangian objective that quantifies the tradeoff between accuracy and invariance. This lower bound leads to a better understanding of the tradeoff via the spectral properties of the joint distribution. In both cases, our results shed new light on this fundamental problem by providing insights on the interplay between accuracy and invariance. These results deepen our understanding of this fundamental problem and may be useful in guiding the design of adversarial representation learning algorithms.

</p>
</details>

<details><summary><b>Quantum reinforcement learning in continuous action space</b>
<a href="https://arxiv.org/abs/2012.10711">arxiv:2012.10711</a>
&#x1F4C8; 4 <br>
<p>Shaojun Wu, Shan Jin, Dingding Wen, Xiaoting Wang</p></summary>
<p>

**Abstract:** Quantum mechanics has the potential to speedup machine learning algorithms, including reinforcement learning(RL). Previous works have shown that quantum algorithms can efficiently solve RL problems in discrete action space, but could become intractable in continuous domain, suffering notably from the curse of dimensionality due to discretization. In this work, we propose an alternative quantum circuit design that can solve RL problems in continuous action space without the dimensionality problem. Specifically, we propose a quantum version of the Deep Deterministic Policy Gradient method constructed from quantum neural networks, with the potential advantage of obtaining an exponential speedup in gate complexity for each iteration. As applications, we demonstrate that quantum control tasks, including the eigenvalue problem and quantum state generation, can be formulated as sequential decision problems and solved by our method.

</p>
</details>

<details><summary><b>Multi-Decoder Attention Model with Embedding Glimpse for Solving Vehicle Routing Problems</b>
<a href="https://arxiv.org/abs/2012.10638">arxiv:2012.10638</a>
&#x1F4C8; 4 <br>
<p>Liang Xin, Wen Song, Zhiguang Cao, Jie Zhang</p></summary>
<p>

**Abstract:** We present a novel deep reinforcement learning method to learn construction heuristics for vehicle routing problems. In specific, we propose a Multi-Decoder Attention Model (MDAM) to train multiple diverse policies, which effectively increases the chance of finding good solutions compared with existing methods that train only one policy. A customized beam search strategy is designed to fully exploit the diversity of MDAM. In addition, we propose an Embedding Glimpse layer in MDAM based on the recursive nature of construction, which can improve the quality of each policy by providing more informative embeddings. Extensive experiments on six different routing problems show that our method significantly outperforms the state-of-the-art deep learning based models.

</p>
</details>

<details><summary><b>Analyzing the Performance of Graph Neural Networks with Pipe Parallelism</b>
<a href="https://arxiv.org/abs/2012.10840">arxiv:2012.10840</a>
&#x1F4C8; 3 <br>
<p>Matthew T. Dearing,  Xiaoyan,  Wang</p></summary>
<p>

**Abstract:** Many interesting datasets ubiquitous in machine learning and deep learning can be described via graphs. As the scale and complexity of graph-structured datasets increase, such as in expansive social networks, protein folding, chemical interaction networks, and material phase transitions, improving the efficiency of the machine learning techniques applied to these is crucial. In this study, we focus on Graph Neural Networks (GNN), which have found great success in tasks such as node or edge classification and link prediction. However, standard GNN models have scaling limits due to necessary recursive calculations performed through dense graph relationships that lead to memory and runtime bottlenecks. While new approaches for processing larger networks are needed to advance graph techniques, and several have been proposed, we study how GNNs could be parallelized using existing tools and frameworks that are already known to be successful in the deep learning community. In particular, we investigate applying pipeline parallelism to GNN models with GPipe, introduced by Google in 2018.

</p>
</details>

<details><summary><b>Forming Human-Robot Cooperation for Tasks with General Goal using Evolutionary Value Learning</b>
<a href="https://arxiv.org/abs/2012.10773">arxiv:2012.10773</a>
&#x1F4C8; 3 <br>
<p>Lingfeng Tao, Michael Bowman, Jiucai Zhang, Xiaoli Zhang</p></summary>
<p>

**Abstract:** In human-robot cooperation, the robot cooperates with the human to accomplish the task together. Existing approaches assume the human has a specific goal during the cooperation, and the robot infers and acts toward it. However, in real-world environments, a human usually only has a general goal (e.g., general direction or area in motion planning) at the beginning of the cooperation which needs to be clarified to a specific goal (e.g., an exact position) during cooperation. The specification process is interactive and dynamic, which depends on the environment and the behavior of the partners. The robot that does not consider the goal specification process may cause frustration to the human partner, elongate the time to come to an agreement, and compromise or fail team performance. We present Evolutionary Value Learning (EVL) approach which uses a State-based Multivariate Bayesian Inference method to model the dynamics of goal specification process in HRC, and an Evolutionary Value Updating method to actively enhance the process of goal specification and cooperation formation. This enables the robot to simultaneously help the human to specify the goal and learn a cooperative policy in a Reinforcement Learning manner. In experiments with real human subjects, the robot equipped with EVL outperforms existing methods with faster goal specification processes and better team performance.

</p>
</details>

<details><summary><b>An Information-Theoretic Framework for Unifying Active Learning Problems</b>
<a href="https://arxiv.org/abs/2012.10695">arxiv:2012.10695</a>
&#x1F4C8; 3 <br>
<p>Quoc Phong Nguyen, Bryan Kian Hsiang Low, Patrick Jaillet</p></summary>
<p>

**Abstract:** This paper presents an information-theoretic framework for unifying active learning problems: level set estimation (LSE), Bayesian optimization (BO), and their generalized variant. We first introduce a novel active learning criterion that subsumes an existing LSE algorithm and achieves state-of-the-art performance in LSE problems with a continuous input domain. Then, by exploiting the relationship between LSE and BO, we design a competitive information-theoretic acquisition function for BO that has interesting connections to upper confidence bound and max-value entropy search (MES). The latter connection reveals a drawback of MES which has important implications on not only MES but also on other MES-based acquisition functions. Finally, our unifying information-theoretic framework can be applied to solve a generalized problem of LSE and BO involving multiple level sets in a data-efficient manner. We empirically evaluate the performance of our proposed algorithms using synthetic benchmark functions, a real-world dataset, and in hyperparameter tuning of machine learning models.

</p>
</details>

<details><summary><b>Bayesian unsupervised learning reveals hidden structure in concentrated electrolytes</b>
<a href="https://arxiv.org/abs/2012.10694">arxiv:2012.10694</a>
&#x1F4C8; 3 <br>
<p>Penelope Jones, Fabian Coupette, Andreas Härtel, Alpha A. Lee</p></summary>
<p>

**Abstract:** Electrolytes play an important role in a plethora of applications ranging from energy storage to biomaterials. Notwithstanding this, the structure of concentrated electrolytes remains enigmatic. Many theoretical approaches attempt to model the concentrated electrolytes by introducing the idea of ion pairs, with ions either being tightly `paired' with a counter-ion, or `free' to screen charge. In this study we reframe the problem into the language of computational statistics, and test the null hypothesis that all ions share the same local environment. Applying the framework to molecular dynamics simulations, we show that this null hypothesis is not supported by data. Our statistical technique suggests the presence of distinct local ionic environments; surprisingly, these differences arise in like charge correlations rather than unlike charge attraction. The resulting fraction of particles in non-aggregated environments shows a universal scaling behaviour across different background dielectric constants and ionic concentrations.

</p>
</details>

<details><summary><b>Deep Reinforcement Learning for Joint Spectrum and Power Allocation in Cellular Networks</b>
<a href="https://arxiv.org/abs/2012.10682">arxiv:2012.10682</a>
&#x1F4C8; 3 <br>
<p>Yasar Sinan Nasir, Dongning Guo</p></summary>
<p>

**Abstract:** A wireless network operator typically divides the radio spectrum it possesses into a number of subbands. In a cellular network those subbands are then reused in many cells. To mitigate co-channel interference, a joint spectrum and power allocation problem is often formulated to maximize a sum-rate objective. The best known algorithms for solving such problems generally require instantaneous global channel state information and a centralized optimizer. In fact those algorithms have not been implemented in practice in large networks with time-varying subbands. Deep reinforcement learning algorithms are promising tools for solving complex resource management problems. A major challenge here is that spectrum allocation involves discrete subband selection, whereas power allocation involves continuous variables. In this paper, a learning framework is proposed to optimize both discrete and continuous decision variables. Specifically, two separate deep reinforcement learning algorithms are designed to be executed and trained simultaneously to maximize a joint objective. Simulation results show that the proposed scheme outperforms both the state-of-the-art fractional programming algorithm and a previous solution based on deep reinforcement learning.

</p>
</details>

<details><summary><b>Towards Coarse and Fine-grained Multi-Graph Multi-Label Learning</b>
<a href="https://arxiv.org/abs/2012.10650">arxiv:2012.10650</a>
&#x1F4C8; 3 <br>
<p>Yejiang Wang, Yuhai Zhao, Zhengkui Wang, Chengqi Zhang</p></summary>
<p>

**Abstract:** Multi-graph multi-label learning (\textsc{Mgml}) is a supervised learning framework, which aims to learn a multi-label classifier from a set of labeled bags each containing a number of graphs. Prior techniques on the \textsc{Mgml} are developed based on transfering graphs into instances and focus on learning the unseen labels only at the bag level. In this paper, we propose a \textit{coarse} and \textit{fine-grained} Multi-graph Multi-label (cfMGML) learning framework which directly builds the learning model over the graphs and empowers the label prediction at both the \textit{coarse} (aka. bag) level and \textit{fine-grained} (aka. graph in each bag) level. In particular, given a set of labeled multi-graph bags, we design the scoring functions at both graph and bag levels to model the relevance between the label and data using specific graph kernels. Meanwhile, we propose a thresholding rank-loss objective function to rank the labels for the graphs and bags and minimize the hamming-loss simultaneously at one-step, which aims to addresses the error accumulation issue in traditional rank-loss algorithms. To tackle the non-convex optimization problem, we further develop an effective sub-gradient descent algorithm to handle high-dimensional space computation required in cfMGML. Experiments over various real-world datasets demonstrate cfMGML achieves superior performance than the state-of-arts algorithms.

</p>
</details>

<details><summary><b>GLISTER: Generalization based Data Subset Selection for Efficient and Robust Learning</b>
<a href="https://arxiv.org/abs/2012.10630">arxiv:2012.10630</a>
&#x1F4C8; 3 <br>
<p>Krishnateja Killamsetty, Durga Sivasubramanian, Ganesh Ramakrishnan, Rishabh Iyer</p></summary>
<p>

**Abstract:** Large scale machine learning and deep models are extremely data-hungry. Unfortunately, obtaining large amounts of labeled data is expensive, and training state-of-the-art models (with hyperparameter tuning) requires significant computing resources and time. Secondly, real-world data is noisy and imbalanced. As a result, several recent papers try to make the training process more efficient and robust. However, most existing work either focuses on robustness or efficiency, but not both. In this work, we introduce Glister, a GeneraLIzation based data Subset selecTion for Efficient and Robust learning framework. We formulate Glister as a mixed discrete-continuous bi-level optimization problem to select a subset of the training data, which maximizes the log-likelihood on a held-out validation set. Next, we propose an iterative online algorithm Glister-Online, which performs data selection iteratively along with the parameter updates and can be applied to any loss-based learning algorithm. We then show that for a rich class of loss functions including cross-entropy, hinge-loss, squared-loss, and logistic-loss, the inner discrete data selection is an instance of (weakly) submodular optimization, and we analyze conditions for which Glister-Online reduces the validation loss and converges. Finally, we propose Glister-Active, an extension to batch active learning, and we empirically demonstrate the performance of Glister on a wide range of tasks including, (a) data selection to reduce training time, (b) robust learning under label noise and imbalance settings, and (c) batch-active learning with several deep and shallow models. We show that our framework improves upon state of the art both in efficiency and accuracy (in cases (a) and (c)) and is more efficient compared to other state-of-the-art robust learning algorithms in case (b).

</p>
</details>

<details><summary><b>Scalable and Provably Accurate Algorithms for Differentially Private Distributed Decision Tree Learning</b>
<a href="https://arxiv.org/abs/2012.10602">arxiv:2012.10602</a>
&#x1F4C8; 3 <br>
<p>Kaiwen Wang, Travis Dick, Maria-Florina Balcan</p></summary>
<p>

**Abstract:** This paper introduces the first provably accurate algorithms for differentially private, top-down decision tree learning in the distributed setting (Balcan et al., 2012). We propose DP-TopDown, a general privacy preserving decision tree learning algorithm, and present two distributed implementations. Our first method NoisyCounts naturally extends the single machine algorithm by using the Laplace mechanism. Our second method LocalRNM significantly reduces communication and added noise by performing local optimization at each data holder. We provide the first utility guarantees for differentially private top-down decision tree learning in both the single machine and distributed settings. These guarantees show that the error of the privately-learned decision tree quickly goes to zero provided that the dataset is sufficiently large. Our extensive experiments on real datasets illustrate the trade-offs of privacy, accuracy and generalization when learning private decision trees in the distributed setting.

</p>
</details>

<details><summary><b>A hybrid deep-learning approach for complex biochemical named entity recognition</b>
<a href="https://arxiv.org/abs/2012.10824">arxiv:2012.10824</a>
&#x1F4C8; 2 <br>
<p>Jian Liu, Lei Gao, Sujie Guo, Rui Ding, Xin Huang, Long Ye, Qinghua Meng, Asef Nazari, Dhananjay Thiruvady</p></summary>
<p>

**Abstract:** Named entity recognition (NER) of chemicals and drugs is a critical domain of information extraction in biochemical research. NER provides support for text mining in biochemical reactions, including entity relation extraction, attribute extraction, and metabolic response relationship extraction. However, the existence of complex naming characteristics in the biomedical field, such as polysemy and special characters, make the NER task very challenging. Here, we propose a hybrid deep learning approach to improve the recognition accuracy of NER. Specifically, our approach applies the Bidirectional Encoder Representations from Transformers (BERT) model to extract the underlying features of the text, learns a representation of the context of the text through Bi-directional Long Short-Term Memory (BILSTM), and incorporates the multi-head attention (MHATT) mechanism to extract chapter-level features. In this approach, the MHATT mechanism aims to improve the recognition accuracy of abbreviations to efficiently deal with the problem of inconsistency in full-text labels. Moreover, conditional random field (CRF) is used to label sequence tags because this probabilistic method does not need strict independence assumptions and can accommodate arbitrary context information. The experimental evaluation on a publicly-available dataset shows that the proposed hybrid approach achieves the best recognition performance; in particular, it substantially improves performance in recognizing abbreviations, polysemes, and low-frequency entities, compared with the state-of-the-art approaches. For instance, compared with the recognition accuracies for low-frequency entities produced by the BILSTM-CRF algorithm, those produced by the hybrid approach on two entity datasets (MULTIPLE and IDENTIFIER) have been increased by 80% and 21.69%, respectively.

</p>
</details>

<details><summary><b>On the Power of Localized Perceptron for Label-Optimal Learning of Halfspaces with Adversarial Noise</b>
<a href="https://arxiv.org/abs/2012.10793">arxiv:2012.10793</a>
&#x1F4C8; 2 <br>
<p>Jie Shen</p></summary>
<p>

**Abstract:** We study {\em online} active learning of homogeneous $s$-sparse halfspaces in $\mathbb{R}^d$ with adversarial noise \cite{kearns1992toward}, where the overall probability of a noisy label is constrained to be at most $ν$ and the marginal distribution over unlabeled data is unchanged. Our main contribution is a state-of-the-art online active learning algorithm that achieves near-optimal attribute efficiency, label and sample complexity under mild distributional assumptions. In particular, under the conditions that the marginal distribution is isotropic log-concave and $ν= Ω(ε)$, where $ε\in (0, 1)$ is the target error rate, we show that our algorithm PAC learns the underlying halfspace in polynomial time with near-optimal label complexity bound of $\tilde{O}\big(s \cdot polylog(d, \frac{1}ε)\big)$ and sample complexity bound of $\tilde{O}\big(\frac{s}ε \cdot polylog(d)\big)$. Prior to this work, existing online algorithms designed for tolerating the adversarial noise are either subject to label complexity polynomial in $d$ or $\frac{1}ε$, or work under the restrictive uniform marginal distribution. As an immediate corollary of our main result, we show that under the more challenging agnostic model \cite{kearns1992toward} where no assumption is made on the noise rate, our active learner achieves an error rate of $O(OPT) + ε$ with the same running time and label and sample complexity, where $OPT$ is the best possible error rate achievable by any homogeneous $s$-sparse halfspace. Our algorithm builds upon the celebrated Perceptron while leveraging novel localized sampling and semi-random gradient update to tolerate the adversarial noise. We believe that our algorithmic design and analysis are of independent interest, and may shed light on learning halfspaces with broader noise models.

</p>
</details>

<details><summary><b>Constructing and Evaluating an Explainable Model for COVID-19 Diagnosis from Chest X-rays</b>
<a href="https://arxiv.org/abs/2012.10787">arxiv:2012.10787</a>
&#x1F4C8; 2 <br>
<p>Rishab Khincha, Soundarya Krishnan, Krishnan Guru-Murthy, Tirtharaj Dash, Lovekesh Vig, Ashwin Srinivasan</p></summary>
<p>

**Abstract:** In this paper, our focus is on constructing models to assist a clinician in the diagnosis of COVID-19 patients in situations where it is easier and cheaper to obtain X-ray data than to obtain high-quality images like those from CT scans. Deep neural networks have repeatedly been shown to be capable of constructing highly predictive models for disease detection directly from image data. However, their use in assisting clinicians has repeatedly hit a stumbling block due to their black-box nature. Some of this difficulty can be alleviated if predictions were accompanied by explanations expressed in clinically relevant terms. In this paper, deep neural networks are used to extract domain-specific features(morphological features like ground-glass opacity and disease indications like pneumonia) directly from the image data. Predictions about these features are then used to construct a symbolic model (a decision tree) for the diagnosis of COVID-19 from chest X-rays, accompanied with two kinds of explanations: visual (saliency maps, derived from the neural stage), and textual (logical descriptions, derived from the symbolic stage). A radiologist rates the usefulness of the visual and textual explanations. Our results demonstrate that neural models can be employed usefully in identifying domain-specific features from low-level image data; that textual explanations in terms of clinically relevant features may be useful; and that visual explanations will need to be clinically meaningful to be useful.

</p>
</details>

<details><summary><b>Augmentation Inside the Network</b>
<a href="https://arxiv.org/abs/2012.10769">arxiv:2012.10769</a>
&#x1F4C8; 2 <br>
<p>Maciej Sypetkowski, Jakub Jasiulewicz, Zbigniew Wojna</p></summary>
<p>

**Abstract:** In this paper, we present augmentation inside the network, a method that simulates data augmentation techniques for computer vision problems on intermediate features of a convolutional neural network. We perform these transformations, changing the data flow through the network, and sharing common computations when it is possible. Our method allows us to obtain smoother speed-accuracy trade-off adjustment and achieves better results than using standard test-time augmentation (TTA) techniques. Additionally, our approach can improve model performance even further when coupled with test-time augmentation. We validate our method on the ImageNet-2012 and CIFAR-100 datasets for image classification. We propose a modification that is 30% faster than the flip test-time augmentation and achieves the same results for CIFAR-100.

</p>
</details>

<details><summary><b>(Decision and regression) tree ensemble based kernels for regression and classification</b>
<a href="https://arxiv.org/abs/2012.10737">arxiv:2012.10737</a>
&#x1F4C8; 2 <br>
<p>Dai Feng, Richard Baumgartner</p></summary>
<p>

**Abstract:** Tree based ensembles such as Breiman's random forest (RF) and Gradient Boosted Trees (GBT) can be interpreted as implicit kernel generators, where the ensuing proximity matrix represents the data-driven tree ensemble kernel. Kernel perspective on the RF has been used to develop a principled framework for theoretical investigation of its statistical properties. Recently, it has been shown that the kernel interpretation is germane to other tree-based ensembles e.g. GBTs. However, practical utility of the links between kernels and the tree ensembles has not been widely explored and systematically evaluated.
  Focus of our work is investigation of the interplay between kernel methods and the tree based ensembles including the RF and GBT. We elucidate the performance and properties of the RF and GBT based kernels in a comprehensive simulation study comprising of continuous and binary targets. We show that for continuous targets, the RF/GBT kernels are competitive to their respective ensembles in higher dimensional scenarios, particularly in cases with larger number of noisy features. For the binary target, the RF/GBT kernels and their respective ensembles exhibit comparable performance. We provide the results from real life data sets for regression and classification to show how these insights may be leveraged in practice. Overall, our results support the tree ensemble based kernels as a valuable addition to the practitioner's toolbox.
  Finally, we discuss extensions of the tree ensemble based kernels for survival targets, interpretable prototype and landmarking classification and regression. We outline future line of research for kernels furnished by Bayesian counterparts of the frequentist tree ensembles.

</p>
</details>

<details><summary><b>Model-Based Actor-Critic with Chance Constraint for Stochastic System</b>
<a href="https://arxiv.org/abs/2012.10716">arxiv:2012.10716</a>
&#x1F4C8; 2 <br>
<p>Baiyu Peng, Yao Mu, Yang Guan, Shengbo Eben Li, Yuming Yin, Jianyu Chen</p></summary>
<p>

**Abstract:** Safety constraints are essential for reinforcement learning (RL) applied in real-world situations. Chance constraints are suitable to represent the safety requirements in stochastic systems. Most existing RL methods with chance constraints have a low convergence rate, and only learn a conservative policy. In this paper, we propose a model-based chance constrained actor-critic (CCAC) algorithm which can efficiently learn a safe and non-conservative policy. Different from existing methods that optimize a conservative lower bound, CCAC directly solves the original chance constrained problems, where the objective function and safe probability is simultaneously optimized with adaptive weights. In order to improve the convergence rate, CCAC utilizes the gradient of dynamic model to accelerate policy optimization. The effectiveness of CCAC is demonstrated by an aggressive car-following task. Experiments indicate that compared with previous methods, CCAC improves the performance by 57.6% while guaranteeing safety, with a five times faster convergence rate.

</p>
</details>

<details><summary><b>Unsupervised Scale-Invariant Multispectral Shape Matching</b>
<a href="https://arxiv.org/abs/2012.10685">arxiv:2012.10685</a>
&#x1F4C8; 2 <br>
<p>Idan Pazi, Dvir Ginzburg, Dan Raviv</p></summary>
<p>

**Abstract:** Alignment between non-rigid stretchable structures is one of the hardest tasks in computer vision, as the invariant properties are hard to define on one hand, and on the other hand no labelled data exists for real datasets. We present unsupervised neural network architecture based upon the spectrum of scale-invariant geometry. We build ontop the functional maps architecture, but show that learning local features, as done until now, is not enough once the isometric assumption breaks but can be solved using scale-invariant geometry. Our method is agnostic to local-scale deformations and shows superior performance for matching shapes from different domains when compared to existing spectral state-of-the-art solutions.

</p>
</details>

<details><summary><b>Dense Multiscale Feature Fusion Pyramid Networks for Object Detection in UAV-Captured Images</b>
<a href="https://arxiv.org/abs/2012.10643">arxiv:2012.10643</a>
&#x1F4C8; 2 <br>
<p>Yingjie Liu</p></summary>
<p>

**Abstract:** Although much significant progress has been made in the research field of object detection with deep learning, there still exists a challenging task for the objects with small size, which is notably pronounced in UAV-captured images. Addressing these issues, it is a critical need to explore the feature extraction methods that can extract more sufficient feature information of small objects. In this paper, we propose a novel method called Dense Multiscale Feature Fusion Pyramid Networks(DMFFPN), which is aimed at obtaining rich features as much as possible, improving the information propagation and reuse. Specifically, the dense connection is designed to fully utilize the representation from the different convolutional layers. Furthermore, cascade architecture is applied in the second stage to enhance the localization capability. Experiments on the drone-based datasets named VisDrone-DET suggest a competitive performance of our method.

</p>
</details>

<details><summary><b>AWA: Adversarial Website Adaptation</b>
<a href="https://arxiv.org/abs/2012.10832">arxiv:2012.10832</a>
&#x1F4C8; 1 <br>
<p>Amir Mahdi Sadeghzadeh, Behrad Tajali, Rasool Jalili</p></summary>
<p>

**Abstract:** One of the most important obligations of privacy-enhancing technologies is to bring confidentiality and privacy to users' browsing activities on the Internet. The website fingerprinting attack enables a local passive eavesdropper to predict the target user's browsing activities even she uses anonymous technologies, such as VPNs, IPsec, and Tor. Recently, the growth of deep learning empowers adversaries to conduct the website fingerprinting attack with higher accuracy. In this paper, we propose a new defense against website fingerprinting attack using adversarial deep learning approaches called Adversarial Website Adaptation (AWA). AWA creates a transformer set in each run so that each website has a unique transformer. Each transformer generates adversarial traces to evade the adversary's classifier. AWA has two versions, including Universal AWA (UAWA) and Non-Universal AWA (NUAWA). Unlike NUAWA, there is no need to access the entire trace of a website in order to generate an adversarial trace in UAWA. We accommodate secret random elements in the training phase of transformers in order for AWA to generate various sets of transformers in each run. We run AWA several times and create multiple sets of transformers. If an adversary and a target user select different sets of transformers, the accuracy of adversary's classifier is almost 19.52\% and 31.94\% with almost 22.28\% and 26.28\% bandwidth overhead in UAWA and NUAWA, respectively. If a more powerful adversary generates adversarial traces through multiple sets of transformers and trains a classifier on them, the accuracy of adversary's classifier is almost 49.10\% and 25.93\% with almost 62.52\% and 64.33\% bandwidth overhead in UAWA and NUAW, respectively.

</p>
</details>

<details><summary><b>Achieving Reliable Causal Inference with Data-Mined Variables: A Random Forest Approach to the Measurement Error Problem</b>
<a href="https://arxiv.org/abs/2012.10790">arxiv:2012.10790</a>
&#x1F4C8; 1 <br>
<p>Mochen Yang, Edward McFowland III, Gordon Burtch, Gediminas Adomavicius</p></summary>
<p>

**Abstract:** Combining machine learning with econometric analysis is becoming increasingly prevalent in both research and practice. A common empirical strategy involves the application of predictive modeling techniques to 'mine' variables of interest from available data, followed by the inclusion of those variables into an econometric framework, with the objective of estimating causal effects. Recent work highlights that, because the predictions from machine learning models are inevitably imperfect, econometric analyses based on the predicted variables are likely to suffer from bias due to measurement error. We propose a novel approach to mitigate these biases, leveraging the ensemble learning technique known as the random forest. We propose employing random forest not just for prediction, but also for generating instrumental variables to address the measurement error embedded in the prediction. The random forest algorithm performs best when comprised of a set of trees that are individually accurate in their predictions, yet which also make 'different' mistakes, i.e., have weakly correlated prediction errors. A key observation is that these properties are closely related to the relevance and exclusion requirements of valid instrumental variables. We design a data-driven procedure to select tuples of individual trees from a random forest, in which one tree serves as the endogenous covariate and the other trees serve as its instruments. Simulation experiments demonstrate the efficacy of the proposed approach in mitigating estimation biases and its superior performance over three alternative methods for bias correction.

</p>
</details>

<details><summary><b>Top-$k$ Ranking Bayesian Optimization</b>
<a href="https://arxiv.org/abs/2012.10688">arxiv:2012.10688</a>
&#x1F4C8; 0 <br>
<p>Quoc Phong Nguyen, Sebastian Tay, Bryan Kian Hsiang Low, Patrick Jaillet</p></summary>
<p>

**Abstract:** This paper presents a novel approach to top-$k$ ranking Bayesian optimization (top-$k$ ranking BO) which is a practical and significant generalization of preferential BO to handle top-$k$ ranking and tie/indifference observations. We first design a surrogate model that is not only capable of catering to the above observations, but is also supported by a classic random utility model. Another equally important contribution is the introduction of the first information-theoretic acquisition function in BO with preferential observation called multinomial predictive entropy search (MPES) which is flexible in handling these observations and optimized for all inputs of a query jointly. MPES possesses superior performance compared with existing acquisition functions that select the inputs of a query one at a time greedily. We empirically evaluate the performance of MPES using several synthetic benchmark functions, CIFAR-$10$ dataset, and SUSHI preference dataset.

</p>
</details>


[Next Page](2020/2020-12/2020-12-18.md)
