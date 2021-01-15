## Summary for 2020-12-18, created on 2021-01-14


<details><summary><b>When Machine Learning Meets Quantum Computers: A Case Study</b>
<a href="https://arxiv.org/abs/2012.10360">arxiv:2012.10360</a>
&#x1F4C8; 154 <br>
<p>Weiwen Jiang, Jinjun Xiong, Yiyu Shi</p></summary>
<p>

**Abstract:** Along with the development of AI democratization, the machine learning approach, in particular neural networks, has been applied to wide-range applications. In different application scenarios, the neural network will be accelerated on the tailored computing platform. The acceleration of neural networks on classical computing platforms, such as CPU, GPU, FPGA, ASIC, has been widely studied; however, when the scale of the application consistently grows up, the memory bottleneck becomes obvious, widely known as memory-wall. In response to such a challenge, advanced quantum computing, which can represent 2^N states with N quantum bits (qubits), is regarded as a promising solution. It is imminent to know how to design the quantum circuit for accelerating neural networks. Most recently, there are initial works studying how to map neural networks to actual quantum processors. To better understand the state-of-the-art design and inspire new design methodology, this paper carries out a case study to demonstrate an end-to-end implementation. On the neural network side, we employ the multilayer perceptron to complete image classification tasks using the standard and widely used MNIST dataset. On the quantum computing side, we target IBM Quantum processors, which can be programmed and simulated by using IBM Qiskit. This work targets the acceleration of the inference phase of a trained neural network on the quantum processor. Along with the case study, we will demonstrate the typical procedure for mapping neural networks to quantum circuits.

</p>
</details>

<details><summary><b>Identifying the latent space geometry of network models through analysis of curvature</b>
<a href="https://arxiv.org/abs/2012.10559">arxiv:2012.10559</a>
&#x1F4C8; 111 <br>
<p>Shane Lubold, Arun G. Chandrasekhar, Tyler H. McCormick</p></summary>
<p>

**Abstract:** Statistically modeling networks, across numerous disciplines and contexts, is fundamentally challenging because of (often high-order) dependence between connections. A common approach assigns each person in the graph to a position on a low-dimensional manifold. Distance between individuals in this (latent) space is inversely proportional to the likelihood of forming a connection. The choice of the latent geometry (the manifold class, dimension, and curvature) has consequential impacts on the substantive conclusions of the model. More positive curvature in the manifold, for example, encourages more and tighter communities; negative curvature induces repulsion among nodes. Currently, however, the choice of the latent geometry is an a priori modeling assumption and there is limited guidance about how to make these choices in a data-driven way. In this work, we present a method to consistently estimate the manifold type, dimension, and curvature from an empirically relevant class of latent spaces: simply connected, complete Riemannian manifolds of constant curvature. Our core insight comes by representing the graph as a noisy distance matrix based on the ties between cliques. Leveraging results from statistical geometry, we develop hypothesis tests to determine whether the observed distances could plausibly be embedded isometrically in each of the candidate geometries. We explore the accuracy of our approach with simulations and then apply our approach to data-sets from economics and sociology as well as neuroscience.

</p>
</details>

<details><summary><b>HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection</b>
<a href="https://arxiv.org/abs/2012.10289">arxiv:2012.10289</a>
&#x1F4C8; 13 <br>
<p>Binny Mathew, Punyajoy Saha, Seid Muhie Yimam, Chris Biemann, Pawan Goyal, Animesh Mukherjee</p></summary>
<p>

**Abstract:** Hate speech is a challenging issue plaguing the online social media. While better models for hate speech detection are continuously being developed, there is little research on the bias and interpretability aspects of hate speech. In this paper, we introduce HateXplain, the first benchmark hate speech dataset covering multiple aspects of the issue. Each post in our dataset is annotated from three different perspectives: the basic, commonly used 3-class classification (i.e., hate, offensive or normal), the target community (i.e., the community that has been the victim of hate speech/offensive speech in the post), and the rationales, i.e., the portions of the post on which their labelling decision (as hate, offensive or normal) is based. We utilize existing state-of-the-art models and observe that even models that perform very well in classification do not score high on explainability metrics like model plausibility and faithfulness. We also observe that models, which utilize the human rationales for training, perform better in reducing unintended bias towards target communities. We have made our code and dataset public at https://github.com/punyajoy/HateXplain

</p>
</details>

<details><summary><b>Rebuilding Trust in Active Learning with Actionable Metrics</b>
<a href="https://arxiv.org/abs/2012.11365">arxiv:2012.11365</a>
&#x1F4C8; 10 <br>
<p>Alexandre Abraham, Léo Dreyfus-Schmidt</p></summary>
<p>

**Abstract:** Active Learning (AL) is an active domain of research, but is seldom used in the industry despite the pressing needs. This is in part due to a misalignment of objectives, while research strives at getting the best results on selected datasets, the industry wants guarantees that Active Learning will perform consistently and at least better than random labeling. The very one-off nature of Active Learning makes it crucial to understand how strategy selection can be carried out and what drives poor performance (lack of exploration, selection of samples that are too hard to classify, ...).
  To help rebuild trust of industrial practitioners in Active Learning, we present various actionable metrics. Through extensive experiments on reference datasets such as CIFAR100, Fashion-MNIST, and 20Newsgroups, we show that those metrics brings interpretability to AL strategies that can be leveraged by the practitioner.

</p>
</details>

<details><summary><b>On Modality Bias in the TVQA Dataset</b>
<a href="https://arxiv.org/abs/2012.10210">arxiv:2012.10210</a>
&#x1F4C8; 9 <br>
<p>Thomas Winterbottom, Sarah Xiao, Alistair McLean, Noura Al Moubayed</p></summary>
<p>

**Abstract:** TVQA is a large scale video question answering (video-QA) dataset based on popular TV shows. The questions were specifically designed to require "both vision and language understanding to answer". In this work, we demonstrate an inherent bias in the dataset towards the textual subtitle modality. We infer said bias both directly and indirectly, notably finding that models trained with subtitles learn, on-average, to suppress video feature contribution. Our results demonstrate that models trained on only the visual information can answer ~45% of the questions, while using only the subtitles achieves ~68%. We find that a bilinear pooling based joint representation of modalities damages model performance by 9% implying a reliance on modality specific information. We also show that TVQA fails to benefit from the RUBi modality bias reduction technique popularised in VQA. By simply improving text processing using BERT embeddings with the simple model first proposed for TVQA, we achieve state-of-the-art results (72.13%) compared to the highly complex STAGE model (70.50%). We recommend a multimodal evaluation framework that can highlight biases in models and isolate visual and textual reliant subsets of data. Using this framework we propose subsets of TVQA that respond exclusively to either or both modalities in order to facilitate multimodal modelling as TVQA originally intended.

</p>
</details>

<details><summary><b>Minimax Active Learning</b>
<a href="https://arxiv.org/abs/2012.10467">arxiv:2012.10467</a>
&#x1F4C8; 6 <br>
<p>Sayna Ebrahimi, William Gan, Kamyar Salahi, Trevor Darrell</p></summary>
<p>

**Abstract:** Active learning aims to develop label-efficient algorithms by querying the most representative samples to be labeled by a human annotator. Current active learning techniques either rely on model uncertainty to select the most uncertain samples or use clustering or reconstruction to choose the most diverse set of unlabeled examples. While uncertainty-based strategies are susceptible to outliers, solely relying on sample diversity does not capture the information available on the main task. In this work, we develop a semi-supervised minimax entropy-based active learning algorithm that leverages both uncertainty and diversity in an adversarial manner. Our model consists of an entropy minimizing feature encoding network followed by an entropy maximizing classification layer. This minimax formulation reduces the distribution gap between the labeled/unlabeled data, while a discriminator is simultaneously trained to distinguish the labeled/unlabeled data. The highest entropy samples from the classifier that the discriminator predicts as unlabeled are selected for labeling. We extensively evaluate our method on various image classification and semantic segmentation benchmark datasets and show superior performance over the state-of-the-art methods.

</p>
</details>

<details><summary><b>Understood in Translation, Transformers for Domain Understanding</b>
<a href="https://arxiv.org/abs/2012.10271">arxiv:2012.10271</a>
&#x1F4C8; 6 <br>
<p>Dimitrios Christofidellis, Matteo Manica, Leonidas Georgopoulos, Hans Vandierendonck</p></summary>
<p>

**Abstract:** Knowledge acquisition is the essential first step of any Knowledge Graph (KG) application. This knowledge can be extracted from a given corpus (KG generation process) or specified from an existing KG (KG specification process). Focusing on domain specific solutions, knowledge acquisition is a labor intensive task usually orchestrated and supervised by subject matter experts. Specifically, the domain of interest is usually manually defined and then the needed generation or extraction tools are utilized to produce the KG. Herein, we propose a supervised machine learning method, based on Transformers, for domain definition of a corpus. We argue why such automated definition of the domain's structure is beneficial both in terms of construction time and quality of the generated graph. The proposed method is extensively validated on three public datasets (WebNLG, NYT and DocRED) by comparing it with two reference methods based on CNNs and RNNs models. The evaluation shows the efficiency of our model in this task. Focusing on scientific document understanding, we present a new health domain dataset based on publications extracted from PubMed and we successfully utilize our method on this. Lastly, we demonstrate how this work lays the foundation for fully automated and unsupervised KG generation.

</p>
</details>

<details><summary><b>Affirmative Algorithms: The Legal Grounds for Fairness as Awareness</b>
<a href="https://arxiv.org/abs/2012.14285">arxiv:2012.14285</a>
&#x1F4C8; 5 <br>
<p>Daniel E. Ho, Alice Xiang</p></summary>
<p>

**Abstract:** While there has been a flurry of research in algorithmic fairness, what is less recognized is that modern antidiscrimination law may prohibit the adoption of such techniques. We make three contributions. First, we discuss how such approaches will likely be deemed "algorithmic affirmative action," posing serious legal risks of violating equal protection, particularly under the higher education jurisprudence. Such cases have increasingly turned toward anticlassification, demanding "individualized consideration" and barring formal, quantitative weights for race regardless of purpose. This case law is hence fundamentally incompatible with fairness in machine learning. Second, we argue that the government-contracting cases offer an alternative grounding for algorithmic fairness, as these cases permit explicit and quantitative race-based remedies based on historical discrimination by the actor. Third, while limited, this doctrinal approach also guides the future of algorithmic fairness, mandating that adjustments be calibrated to the entity's responsibility for historical discrimination causing present-day disparities. The contractor cases provide a legally viable path for algorithmic fairness under current constitutional doctrine but call for more research at the intersection of algorithmic fairness and causal inference to ensure that bias mitigation is tailored to specific causes and mechanisms of bias.

</p>
</details>

<details><summary><b>Upper and Lower Bounds on the Performance of Kernel PCA</b>
<a href="https://arxiv.org/abs/2012.10369">arxiv:2012.10369</a>
&#x1F4C8; 5 <br>
<p>Maxime Haddouche, Benjamin Guedj, Omar Rivasplata, John Shawe-Taylor</p></summary>
<p>

**Abstract:** Principal Component Analysis (PCA) is a popular method for dimension reduction and has attracted an unfailing interest for decades. Recently, kernel PCA has emerged as an extension of PCA but, despite its use in practice, a sound theoretical understanding of kernel PCA is missing. In this paper, we contribute lower and upper bounds on the efficiency of kernel PCA, involving the empirical eigenvalues of the kernel Gram matrix. Two bounds are for fixed estimators, and two are for randomized estimators through the PAC-Bayes theory. We control how much information is captured by kernel PCA on average, and we dissect the bounds to highlight strengths and limitations of the kernel PCA algorithm. Therefore, we contribute to the better understanding of kernel PCA. Our bounds are briefly illustrated on a toy numerical example.

</p>
</details>

<details><summary><b>MASSIVE: Tractable and Robust Bayesian Learning of Many-Dimensional Instrumental Variable Models</b>
<a href="https://arxiv.org/abs/2012.10141">arxiv:2012.10141</a>
&#x1F4C8; 5 <br>
<p>Ioan Gabriel Bucur, Tom Claassen, Tom Heskes</p></summary>
<p>

**Abstract:** The recent availability of huge, many-dimensional data sets, like those arising from genome-wide association studies (GWAS), provides many opportunities for strengthening causal inference. One popular approach is to utilize these many-dimensional measurements as instrumental variables (instruments) for improving the causal effect estimate between other pairs of variables. Unfortunately, searching for proper instruments in a many-dimensional set of candidates is a daunting task due to the intractable model space and the fact that we cannot directly test which of these candidates are valid, so most existing search methods either rely on overly stringent modeling assumptions or fail to capture the inherent model uncertainty in the selection process. We show that, as long as at least some of the candidates are (close to) valid, without knowing a priori which ones, they collectively still pose enough restrictions on the target interaction to obtain a reliable causal effect estimate. We propose a general and efficient causal inference algorithm that accounts for model uncertainty by performing Bayesian model averaging over the most promising many-dimensional instrumental variable models, while at the same time employing weaker assumptions regarding the data generating process. We showcase the efficiency, robustness and predictive performance of our algorithm through experimental results on both simulated and real-world data.

</p>
</details>

<details><summary><b>Reinforcement Learning for Test Case Prioritization</b>
<a href="https://arxiv.org/abs/2012.11364">arxiv:2012.11364</a>
&#x1F4C8; 4 <br>
<p>João Lousada, Miguel Ribeiro</p></summary>
<p>

**Abstract:** In modern software engineering, Continuous Integration (CI) has become an indispensable step towards systematically managing the life cycles of software development. Large companies struggle with keeping the pipeline updated and operational, in useful time, due to the large amount of changes and addition of features, that build on top of each other and have several developers, working on different platforms. Associated with such software changes, there is always a strong component of Testing. As teams and projects grow, exhaustive testing quickly becomes inhibitive, becoming adamant to select the most relevant test cases earlier, without compromising software quality. This paper extends recent studies on applying Reinforcement Learning to optimize testing strategies. We test its ability to adapt to new environments, by testing it on novel data extracted from a financial institution, yielding a Normalized percentage of Fault Detection (NAPFD) of over $0.6$ using the Network Approximator and Test Case Failure Reward. Additionally, we studied the impact of using Decision Tree (DT) Approximator as a model for memory representation, which failed to produce significant improvements relative to Artificial Neural Networks.

</p>
</details>

<details><summary><b>T-GAP: Learning to Walk across Time for Temporal Knowledge Graph Completion</b>
<a href="https://arxiv.org/abs/2012.10595">arxiv:2012.10595</a>
&#x1F4C8; 4 <br>
<p>Jaehun Jung, Jinhong Jung, U Kang</p></summary>
<p>

**Abstract:** Temporal knowledge graphs (TKGs) inherently reflect the transient nature of real-world knowledge, as opposed to static knowledge graphs. Naturally, automatic TKG completion has drawn much research interests for a more realistic modeling of relational reasoning. However, most of the existing mod-els for TKG completion extend static KG embeddings that donot fully exploit TKG structure, thus lacking in 1) account-ing for temporally relevant events already residing in the lo-cal neighborhood of a query, and 2) path-based inference that facilitates multi-hop reasoning and better interpretability. In this paper, we propose T-GAP, a novel model for TKG completion that maximally utilizes both temporal information and graph structure in its encoder and decoder. T-GAP encodes query-specific substructure of TKG by focusing on the temporal displacement between each event and the query times-tamp, and performs path-based inference by propagating attention through the graph. Our empirical experiments demonstrate that T-GAP not only achieves superior performance against state-of-the-art baselines, but also competently generalizes to queries with unseen timestamps. Through extensive qualitative analyses, we also show that T-GAP enjoys from transparent interpretability, and follows human intuition in its reasoning process.

</p>
</details>

<details><summary><b>Finding Sparse Structure for Domain Specific Neural Machine Translation</b>
<a href="https://arxiv.org/abs/2012.10586">arxiv:2012.10586</a>
&#x1F4C8; 4 <br>
<p>Jianze Liang, Chengqi Zhao, Mingxuan Wang, Xipeng Qiu, Lei Li</p></summary>
<p>

**Abstract:** Fine-tuning is a major approach for domain adaptation in Neural Machine Translation (NMT). However, unconstrained fine-tuning requires very careful hyper-parameter tuning otherwise it is easy to fall into over-fitting on the target domain and degradation on the general domain. To mitigate it, we propose PRUNE-TUNE, a novel domain adaptation method via gradual pruning. It learns tiny domain-specific subnetworks for tuning. During adaptation to a new domain, we only tune its corresponding subnetwork. PRUNE-TUNE alleviates the over-fitting and the degradation problem without model modification. Additionally, with no overlapping between domain-specific subnetworks, PRUNE-TUNE is also capable of sequential multi-domain learning. Empirical experiment results show that PRUNE-TUNE outperforms several strong competitors in the target domain test set without the quality degradation of the general domain in both single and multiple domain settings.

</p>
</details>

<details><summary><b>Identification of Metallic Objects using Spectral MPT Signatures: Object Characterisation and Invariants</b>
<a href="https://arxiv.org/abs/2012.10376">arxiv:2012.10376</a>
&#x1F4C8; 4 <br>
<p>P. D. Ledger, B. A. Wilson, A. A. S. Amad, W. R. B. Lionheart</p></summary>
<p>

**Abstract:** The early detection of terrorist threats, such as guns and knives, through improved metal detection, has the potential to reduce the number of attacks and improve public safety and security. To achieve this, there is considerable potential to use the fields applied and measured by a metal detector to discriminate between different shapes and different metals since, hidden within the field perturbation, is object characterisation information. The magnetic polarizability tensor (MPT) offers an economical characterisation of metallic objects that can be computed for different threat and non-threat objects and has an established theoretical background, which shows that the induced voltage is a function of the hidden object's MPT coefficients. In this paper, we describe the additional characterisation information that measurements of the induced voltage over a range of frequencies offer compared to measurements at a single frequency. We call such object characterisations its MPT spectral signature. Then, we present a series of alternative rotational invariants for the purpose of classifying hidden objects using MPT spectral signatures. Finally, we include examples of computed MPT spectral signature characterisations of realistic threat and non-threat objects that can be used to train machine learning algorithms for classification purposes.

</p>
</details>

<details><summary><b>Temporal Bilinear Encoding Network of Audio-Visual Features at Low Sampling Rates</b>
<a href="https://arxiv.org/abs/2012.10283">arxiv:2012.10283</a>
&#x1F4C8; 4 <br>
<p>Feiyan Hu, Eva Mohedano, Noel O'Connor, Kevin McGuinness</p></summary>
<p>

**Abstract:** Current deep learning based video classification architectures are typically trained end-to-end on large volumes of data and require extensive computational resources. This paper aims to exploit audio-visual information in video classification with a 1 frame per second sampling rate. We propose Temporal Bilinear Encoding Networks (TBEN) for encoding both audio and visual long range temporal information using bilinear pooling and demonstrate bilinear pooling is better than average pooling on the temporal dimension for videos with low sampling rate. We also embed the label hierarchy in TBEN to further improve the robustness of the classifier. Experiments on the FGA240 fine-grained classification dataset using TBEN achieve a new state-of-the-art (hit@1=47.95%). We also exploit the possibility of incorporating TBEN with multiple decoupled modalities like visual semantic and motion features: experiments on UCF101 sampled at 1 FPS achieve close to state-of-the-art accuracy (hit@1=91.03%) while requiring significantly less computational resources than competing approaches for both training and prediction.

</p>
</details>

<details><summary><b>Investigating the Ground-level Ozone Formation and Future Trend in Taiwan</b>
<a href="https://arxiv.org/abs/2012.10058">arxiv:2012.10058</a>
&#x1F4C8; 4 <br>
<p>Yu-Wen Chen, Sourav Medya, Yi-Chun Chen</p></summary>
<p>

**Abstract:** Tropospheric ozone (O3) is an influential ground-level air pollutant which can severely damage the environment. Thus evaluating the importance of various factors related to the O3 formation process is essential. However, O3 simulated by the available climate models exhibits large variance in different places, indicating the insufficiency of models in explaining the O3 formation process correctly. In this paper, we aim to understand the impact of various factors on O3 formation and predict the O3 concentrations. Six well-known supervised learning methods are evaluated to estimate the observed O3 using sixteen meteorological and chemical variables. We find that the XGBoost and the convolution neural network (CNN) models achieve most accurate predictions. We also demonstrate the importance of several variables empirically. The results suggest that while Nitrogen Oxides negatively contributes to predicting O3, the amount of solar radiation makes significantly positive contribution. Furthermore, we apply the XGBoost model on climate O3 prediction and show its competence in calibrating the O3 simulated by a global climate model.

</p>
</details>

<details><summary><b>Learning by Fixing: Solving Math Word Problems with Weak Supervision</b>
<a href="https://arxiv.org/abs/2012.10582">arxiv:2012.10582</a>
&#x1F4C8; 3 <br>
<p>Yining Hong, Qing Li, Daniel Ciao, Siyuan Haung, Song-Chun Zhu</p></summary>
<p>

**Abstract:** Previous neural solvers of math word problems (MWPs) are learned with full supervision and fail to generate diverse solutions. In this paper, we address this issue by introducing a \textit{weakly-supervised} paradigm for learning MWPs. Our method only requires the annotations of the final answers and can generate various solutions for a single problem. To boost weakly-supervised learning, we propose a novel \textit{learning-by-fixing} (LBF) framework, which corrects the misperceptions of the neural network via symbolic reasoning. Specifically, for an incorrect solution tree generated by the neural network, the \textit{fixing} mechanism propagates the error from the root node to the leaf nodes and infers the most probable fix that can be executed to get the desired answer. To generate more diverse solutions, \textit{tree regularization} is applied to guide the efficient shrinkage and exploration of the solution space, and a \textit{memory buffer} is designed to track and save the discovered various fixes for each problem. Experimental results on the Math23K dataset show the proposed LBF framework significantly outperforms reinforcement learning baselines in weakly-supervised learning. Furthermore, it achieves comparable top-1 and much better top-3/5 answer accuracies than fully-supervised methods, demonstrating its strength in producing diverse solutions.

</p>
</details>

<details><summary><b>Identifying Invariant Texture Violation for Robust Deepfake Detection</b>
<a href="https://arxiv.org/abs/2012.10580">arxiv:2012.10580</a>
&#x1F4C8; 3 <br>
<p>Xinwei Sun, Botong Wu, Wei Chen</p></summary>
<p>

**Abstract:** Existing deepfake detection methods have reported promising in-distribution results, by accessing published large-scale dataset. However, due to the non-smooth synthesis method, the fake samples in this dataset may expose obvious artifacts (e.g., stark visual contrast, non-smooth boundary), which were heavily relied on by most of the frame-level detection methods above. As these artifacts do not come up in real media forgeries, the above methods can suffer from a large degradation when applied to fake images that close to reality. To improve the robustness for high-realism fake data, we propose the Invariant Texture Learning (InTeLe) framework, which only accesses the published dataset with low visual quality. Our method is based on the prior that the microscopic facial texture of the source face is inevitably violated by the texture transferred from the target person, which can hence be regarded as the invariant characterization shared among all fake images. To learn such an invariance for deepfake detection, our InTeLe introduces an auto-encoder framework with different decoders for pristine and fake images, which are further appended with a shallow classifier in order to separate out the obvious artifact-effect. Equipped with such a separation, the extracted embedding by encoder can capture the texture violation in fake images, followed by the classifier for the final pristine/fake prediction. As a theoretical guarantee, we prove the identifiability of such an invariance texture violation, i.e., to be precisely inferred from observational data. The effectiveness and utility of our method are demonstrated by promising generalization ability from low-quality images with obvious artifacts to fake images with high realism.

</p>
</details>

<details><summary><b>CityLearn: Standardizing Research in Multi-Agent Reinforcement Learning for Demand Response and Urban Energy Management</b>
<a href="https://arxiv.org/abs/2012.10504">arxiv:2012.10504</a>
&#x1F4C8; 3 <br>
<p>Jose R Vazquez-Canteli, Sourav Dey, Gregor Henze, Zoltan Nagy</p></summary>
<p>

**Abstract:** Rapid urbanization, increasing integration of distributed renewable energy resources, energy storage, and electric vehicles introduce new challenges for the power grid. In the US, buildings represent about 70% of the total electricity demand and demand response has the potential for reducing peaks of electricity by about 20%. Unlocking this potential requires control systems that operate on distributed systems, ideally data-driven and model-free. For this, reinforcement learning (RL) algorithms have gained increased interest in the past years. However, research in RL for demand response has been lacking the level of standardization that propelled the enormous progress in RL research in the computer science community. To remedy this, we created CityLearn, an OpenAI Gym Environment which allows researchers to implement, share, replicate, and compare their implementations of RL for demand response. Here, we discuss this environment and The CityLearn Challenge, a RL competition we organized to propel further progress in this field.

</p>
</details>

<details><summary><b>ShineOn: Illuminating Design Choices for Practical Video-based Virtual Clothing Try-on</b>
<a href="https://arxiv.org/abs/2012.10495">arxiv:2012.10495</a>
&#x1F4C8; 3 <br>
<p>Gaurav Kuppa, Andrew Jong, Vera Liu, Ziwei Liu, Teng-Sheng Moh</p></summary>
<p>

**Abstract:** Virtual try-on has garnered interest as a neural rendering benchmark task to evaluate complex object transfer and scene composition. Recent works in virtual clothing try-on feature a plethora of possible architectural and data representation choices. However, they present little clarity on quantifying the isolated visual effect of each choice, nor do they specify the hyperparameter details that are key to experimental reproduction. Our work, ShineOn, approaches the try-on task from a bottom-up approach and aims to shine light on the visual and quantitative effects of each experiment. We build a series of scientific experiments to isolate effective design choices in video synthesis for virtual clothing try-on. Specifically, we investigate the effect of different pose annotations, self-attention layer placement, and activation functions on the quantitative and qualitative performance of video virtual try-on. We find that DensePose annotations not only enhance face details but also decrease memory usage and training time. Next, we find that attention layers improve face and neck quality. Finally, we show that GELU and ReLU activation functions are the most effective in our experiments despite the appeal of newer activations such as Swish and Sine. We will release a well-organized code base, hyperparameters, and model checkpoints to support the reproducibility of our results. We expect our extensive experiments and code to greatly inform future design choices in video virtual try-on. Our code may be accessed at https://github.com/andrewjong/ShineOn-Virtual-Tryon.

</p>
</details>

<details><summary><b>A Benchmark Arabic Dataset for Commonsense Explanation</b>
<a href="https://arxiv.org/abs/2012.10251">arxiv:2012.10251</a>
&#x1F4C8; 3 <br>
<p>Saja AL-Tawalbeh, Mohammad AL-Smadi</p></summary>
<p>

**Abstract:** Language comprehension and commonsense knowledge validation by machines are challenging tasks that are still under researched and evaluated for Arabic text. In this paper, we present a benchmark Arabic dataset for commonsense explanation. The dataset consists of Arabic sentences that does not make sense along with three choices to select among them the one that explains why the sentence is false. Furthermore, this paper presents baseline results to assist and encourage the future evaluation of research in this field. The dataset is distributed under the Creative Commons CC-BY-SA 4.0 license and can be found on GitHub

</p>
</details>

<details><summary><b>Exact Reduction of Huge Action Spaces in General Reinforcement Learning</b>
<a href="https://arxiv.org/abs/2012.10200">arxiv:2012.10200</a>
&#x1F4C8; 3 <br>
<p>Sultan Javed Majeed, Marcus Hutter</p></summary>
<p>

**Abstract:** The reinforcement learning (RL) framework formalizes the notion of learning with interactions. Many real-world problems have large state-spaces and/or action-spaces such as in Go, StarCraft, protein folding, and robotics or are non-Markovian, which cause significant challenges to RL algorithms. In this work we address the large action-space problem by sequentializing actions, which can reduce the action-space size significantly, even down to two actions at the expense of an increased planning horizon. We provide explicit and exact constructions and equivalence proofs for all quantities of interest for arbitrary history-based processes. In the case of MDPs, this could help RL algorithms that bootstrap. In this work we show how action-binarization in the non-MDP case can significantly improve Extreme State Aggregation (ESA) bounds. ESA allows casting any (non-MDP, non-ergodic, history-based) RL problem into a fixed-sized non-Markovian state-space with the help of a surrogate Markovian process. On the upside, ESA enjoys similar optimality guarantees as Markovian models do. But a downside is that the size of the aggregated state-space becomes exponential in the size of the action-space. In this work, we patch this issue by binarizing the action-space. We provide an upper bound on the number of states of this binarized ESA that is logarithmic in the original action-space size, a double-exponential improvement.

</p>
</details>

<details><summary><b>STNet: Scale Tree Network with Multi-level Auxiliator for Crowd Counting</b>
<a href="https://arxiv.org/abs/2012.10189">arxiv:2012.10189</a>
&#x1F4C8; 3 <br>
<p>Mingjie Wang, Hao Cai, Xianfeng Han, Jun Zhou, Minglun Gong</p></summary>
<p>

**Abstract:** Crowd counting remains a challenging task because the presence of drastic scale variation, density inconsistency, and complex background can seriously degrade the counting accuracy. To battle the ingrained issue of accuracy degradation, we propose a novel and powerful network called Scale Tree Network (STNet) for accurate crowd counting. STNet consists of two key components: a Scale-Tree Diversity Enhancer and a Semi-supervised Multi-level Auxiliator. Specifically, the Diversity Enhancer is designed to enrich scale diversity, which alleviates limitations of existing methods caused by insufficient level of scales. A novel tree structure is adopted to hierarchically parse coarse-to-fine crowd regions. Furthermore, a simple yet effective Multi-level Auxiliator is presented to aid in exploiting generalisable shared characteristics at multiple levels, allowing more accurate pixel-wise background cognition. The overall STNet is trained in an end-to-end manner, without the needs for manually tuning loss weights between the main and the auxiliary tasks. Extensive experiments on four challenging crowd datasets demonstrate the superiority of the proposed method.

</p>
</details>

<details><summary><b>Voronoi Progressive Widening: Efficient Online Solvers for Continuous Space MDPs and POMDPs with Provably Optimal Components</b>
<a href="https://arxiv.org/abs/2012.10140">arxiv:2012.10140</a>
&#x1F4C8; 3 <br>
<p>Michael H. Lim, Claire J. Tomlin, Zachary N. Sunberg</p></summary>
<p>

**Abstract:** Markov decision processes (MDPs) and partially observable MDPs (POMDPs) can effectively represent complex real-world decision and control problems. However, continuous space MDPs and POMDPs, i.e. those having continuous state, action and observation spaces, are extremely difficult to solve, and there are few online algorithms with convergence guarantees. This paper introduces Voronoi Progressive Widening (VPW), a general technique to modify tree search algorithms to effectively handle continuous or hybrid action spaces, and proposes and evaluates three continuous space solvers: VOSS, VOWSS, and VOMCPOW. VOSS and VOWSS are theoretical tools based on sparse sampling and Voronoi optimistic optimization designed to justify VPW-based online solvers. While previous algorithms have enjoyed convergence guarantees for problems with continuous state and observation spaces, VOWSS is the first with global convergence guarantees for problems that additionally have continuous action spaces. VOMCPOW is a versatile and efficient VPW-based algorithm that consistently outperforms POMCPOW and BOMCP in several simulation experiments.

</p>
</details>

<details><summary><b>On the human-recognizability phenomenon of adversarially trained deep image classifiers</b>
<a href="https://arxiv.org/abs/2101.05219">arxiv:2101.05219</a>
&#x1F4C8; 2 <br>
<p>Jonathan Helland, Nathan VanHoudnos</p></summary>
<p>

**Abstract:** In this work, we investigate the phenomenon that robust image classifiers have human-recognizable features -- often referred to as interpretability -- as revealed through the input gradients of their score functions and their subsequent adversarial perturbations. In particular, we demonstrate that state-of-the-art methods for adversarial training incorporate two terms -- one that orients the decision boundary via minimizing the expected loss, and another that induces smoothness of the classifier's decision surface by penalizing the local Lipschitz constant. Through this demonstration, we provide a unified discussion of gradient and Jacobian-based regularizers that have been used to encourage adversarial robustness in prior works. Following this discussion, we give qualitative evidence that the coupling of smoothness and orientation of the decision boundary is sufficient to induce the aforementioned human-recognizability phenomenon.

</p>
</details>

<details><summary><b>Efficient Object-Level Visual Context Modeling for Multimodal Machine Translation: Masking Irrelevant Objects Helps Grounding</b>
<a href="https://arxiv.org/abs/2101.05208">arxiv:2101.05208</a>
&#x1F4C8; 2 <br>
<p>Dexin Wang, Deyi Xiong</p></summary>
<p>

**Abstract:** Visual context provides grounding information for multimodal machine translation (MMT). However, previous MMT models and probing studies on visual features suggest that visual information is less explored in MMT as it is often redundant to textual information. In this paper, we propose an object-level visual context modeling framework (OVC) to efficiently capture and explore visual information for multimodal machine translation. With detected objects, the proposed OVC encourages MMT to ground translation on desirable visual objects by masking irrelevant objects in the visual modality. We equip the proposed with an additional object-masking loss to achieve this goal. The object-masking loss is estimated according to the similarity between masked objects and the source texts so as to encourage masking source-irrelevant objects. Additionally, in order to generate vision-consistent target words, we further propose a vision-weighted translation loss for OVC. Experiments on MMT datasets demonstrate that the proposed OVC model outperforms state-of-the-art MMT models and analyses show that masking irrelevant objects helps grounding in MMT.

</p>
</details>

<details><summary><b>Communication-Aware Collaborative Learning</b>
<a href="https://arxiv.org/abs/2012.10569">arxiv:2012.10569</a>
&#x1F4C8; 2 <br>
<p>Avrim Blum, Shelby Heinecke, Lev Reyzin</p></summary>
<p>

**Abstract:** Algorithms for noiseless collaborative PAC learning have been analyzed and optimized in recent years with respect to sample complexity. In this paper, we study collaborative PAC learning with the goal of reducing communication cost at essentially no penalty to the sample complexity. We develop communication efficient collaborative PAC learning algorithms using distributed boosting. We then consider the communication cost of collaborative learning in the presence of classification noise. As an intermediate step, we show how collaborative PAC learning algorithms can be adapted to handle classification noise. With this insight, we develop communication efficient algorithms for collaborative PAC learning robust to classification noise.

</p>
</details>

<details><summary><b>Computer-aided abnormality detection in chest radiographs in a clinical setting via domain-adaptation</b>
<a href="https://arxiv.org/abs/2012.10564">arxiv:2012.10564</a>
&#x1F4C8; 2 <br>
<p>Abhishek K Dubey, Michael T Young, Christopher Stanley, Dalton Lunga, Jacob Hinkle</p></summary>
<p>

**Abstract:** Deep learning (DL) models are being deployed at medical centers to aid radiologists for diagnosis of lung conditions from chest radiographs. Such models are often trained on a large volume of publicly available labeled radiographs. These pre-trained DL models' ability to generalize in clinical settings is poor because of the changes in data distributions between publicly available and privately held radiographs. In chest radiographs, the heterogeneity in distributions arises from the diverse conditions in X-ray equipment and their configurations used for generating the images. In the machine learning community, the challenges posed by the heterogeneity in the data generation source is known as domain shift, which is a mode shift in the generative model. In this work, we introduce a domain-shift detection and removal method to overcome this problem. Our experimental results show the proposed method's effectiveness in deploying a pre-trained DL model for abnormality detection in chest radiographs in a clinical setting.

</p>
</details>

<details><summary><b>Dataset Security for Machine Learning: Data Poisoning, Backdoor Attacks, and Defenses</b>
<a href="https://arxiv.org/abs/2012.10544">arxiv:2012.10544</a>
&#x1F4C8; 2 <br>
<p>Micah Goldblum, Dimitris Tsipras, Chulin Xie, Xinyun Chen, Avi Schwarzschild, Dawn Song, Aleksander Madry, Bo Li, Tom Goldstein</p></summary>
<p>

**Abstract:** As machine learning systems grow in scale, so do their training data requirements, forcing practitioners to automate and outsource the curation of training data in order to achieve state-of-the-art performance. The absence of trustworthy human supervision over the data collection process exposes organizations to security vulnerabilities; training data can be manipulated to control and degrade the downstream behaviors of learned models. The goal of this work is to systematically categorize and discuss a wide range of dataset vulnerabilities and exploits, approaches for defending against these threats, and an array of open problems in this space. In addition to describing various poisoning and backdoor threat models and the relationships among them, we develop their unified taxonomy.

</p>
</details>

<details><summary><b>Biomedical Knowledge Graph Refinement and Completion using Graph Representation Learning and Top-K Similarity Measure</b>
<a href="https://arxiv.org/abs/2012.10540">arxiv:2012.10540</a>
&#x1F4C8; 2 <br>
<p>Islam Akef Ebeid, Majdi Hassan, Tingyi Wanyan, Jack Roper, Abhik Seal, Ying Ding</p></summary>
<p>

**Abstract:** Knowledge Graphs have been one of the fundamental methods for integrating heterogeneous data sources. Integrating heterogeneous data sources is crucial, especially in the biomedical domain, where central data-driven tasks such as drug discovery rely on incorporating information from different biomedical databases. These databases contain various biological entities and relations such as proteins (PDB), genes (Gene Ontology), drugs (DrugBank), diseases (DDB), and protein-protein interactions (BioGRID). The process of semantically integrating heterogeneous biomedical databases is often riddled with imperfections. The quality of data-driven drug discovery relies on the accuracy of the mining methods used and the data's quality as well. Thus, having complete and refined biomedical knowledge graphs is central to achieving more accurate drug discovery outcomes. Here we propose using the latest graph representation learning and embedding models to refine and complete biomedical knowledge graphs. This preliminary work demonstrates learning discrete representations of the integrated biomedical knowledge graph Chem2Bio2RD [3]. We perform a knowledge graph completion and refinement task using a simple top-K cosine similarity measure between the learned embedding vectors to predict missing links between drugs and targets present in the data. We show that this simple procedure can be used alternatively to binary classifiers in link prediction.

</p>
</details>

<details><summary><b>A Graph Attention Based Approach for Trajectory Prediction in Multi-agent Sports Games</b>
<a href="https://arxiv.org/abs/2012.10531">arxiv:2012.10531</a>
&#x1F4C8; 2 <br>
<p>Ding Ding, H. Howie Huang</p></summary>
<p>

**Abstract:** This work investigates the problem of multi-agents trajectory prediction. Prior approaches lack of capability of capturing fine-grained dependencies among coordinated agents. In this paper, we propose a spatial-temporal trajectory prediction approach that is able to learn the strategy of a team with multiple coordinated agents. In particular, we use graph-based attention model to learn the dependency of the agents. In addition, instead of utilizing the recurrent networks (e.g., VRNN, LSTM), our method uses a Temporal Convolutional Network (TCN) as the sequential model to support long effective history and provide important features such as parallelism and stable gradients. We demonstrate the validation and effectiveness of our approach on two different sports game datasets: basketball and soccer datasets. The result shows that compared to related approaches, our model that infers the dependency of players yields substantially improved performance. Code is available at https://github.com/iHeartGraph/predict

</p>
</details>

<details><summary><b>RAILS: A Robust Adversarial Immune-inspired Learning System</b>
<a href="https://arxiv.org/abs/2012.10485">arxiv:2012.10485</a>
&#x1F4C8; 2 <br>
<p>Ren Wang, Tianqi Chen, Stephen Lindsly, Alnawaz Rehemtulla, Alfred Hero, Indika Rajapakse</p></summary>
<p>

**Abstract:** Adversarial attacks against deep neural networks are continuously evolving. Without effective defenses, they can lead to catastrophic failure. The long-standing and arguably most powerful natural defense system is the mammalian immune system, which has successfully defended against attacks by novel pathogens for millions of years. In this paper, we propose a new adversarial defense framework, called the Robust Adversarial Immune-inspired Learning System (RAILS). RAILS incorporates an Adaptive Immune System Emulation (AISE), which emulates in silico the biological mechanisms that are used to defend the host against attacks by pathogens. We use RAILS to harden Deep k-Nearest Neighbor (DkNN) architectures against evasion attacks. Evolutionary programming is used to simulate processes in the natural immune system: B-cell flocking, clonal expansion, and affinity maturation. We show that the RAILS learning curve exhibits similar diversity-selection learning phases as observed in our in vitro biological experiments. When applied to adversarial image classification on three different datasets, RAILS delivers an additional 5.62%/12.56%/4.74% robustness improvement as compared to applying DkNN alone, without appreciable loss of accuracy on clean data.

</p>
</details>

<details><summary><b>Reinforcement Learning based Multi-Robot Classification via Scalable Communication Structure</b>
<a href="https://arxiv.org/abs/2012.10480">arxiv:2012.10480</a>
&#x1F4C8; 2 <br>
<p>Guangyi Liu, Arash Amini, Martin Takáč, Héctor Muñoz-Avila, Nader Motee</p></summary>
<p>

**Abstract:** In the multi-robot collaboration domain, training with Reinforcement Learning (RL) can become intractable, and performance starts to deteriorate drastically as the number of robots increases. In this work, we proposed a distributed multi-robot learning architecture with a scalable communication structure capable of learning a robust communication policy for time-varying communication topology. We construct the communication structure with Long-Short Term Memory (LSTM) cells and star graphs, in which the computational complexity of the proposed learning algorithm scales linearly with the number of robots and suitable for application with a large number of robots. The proposed methodology is validated with a map classification problem in the simulated environment. It is shown that the proposed architecture achieves a comparable classification accuracy with the centralized methods, maintains high performance with various numbers of robots without additional training cost, and robust to hacking and loss of the robots in the network.

</p>
</details>

<details><summary><b>On the Efficient Implementation of the Matrix Exponentiated Gradient Algorithm for Low-Rank Matrix Optimization</b>
<a href="https://arxiv.org/abs/2012.10469">arxiv:2012.10469</a>
&#x1F4C8; 2 <br>
<p>Dan Garber, Atara Kaplan</p></summary>
<p>

**Abstract:** Convex optimization over the spectrahedron, i.e., the set of all real $n\times n$ positive semidefinite matrices with unit trace, has important applications in machine learning, signal processing and statistics, mainly as a convex relaxation for optimization with low-rank matrices. It is also one of the most prominent examples in the theory of first-order methods for convex optimization in which non-Euclidean methods can be significantly preferable to their Euclidean counterparts, and in particular the Matrix Exponentiated Gradient (MEG) method which is based on the Bregman distance induced by the (negative) von Neumann entropy. Unfortunately, implementing MEG requires a full SVD computation on each iteration, which is not scalable to high-dimensional problems.
  In this work we propose efficient implementations of MEG, both with deterministic and stochastic gradients, which are tailored for optimization with low-rank matrices, and only use a single low-rank SVD computation on each iteration. We also provide efficiently-computable certificates for the correct convergence of our methods. Mainly, we prove that under a strict complementarity condition, the suggested methods converge from a "warm-start" initialization with similar rates to their full-SVD-based counterparts. Finally, we bring empirical experiments which both support our theoretical findings and demonstrate the practical appeal of our methods.

</p>
</details>

<details><summary><b>Small Business Classification By Name: Addressing Gender and Geographic Origin Biases</b>
<a href="https://arxiv.org/abs/2012.10348">arxiv:2012.10348</a>
&#x1F4C8; 2 <br>
<p>Daniel Shapiro</p></summary>
<p>

**Abstract:** Small business classification is a difficult and important task within many applications, including customer segmentation. Training on small business names introduces gender and geographic origin biases. A model for predicting one of 66 business types based only upon the business name was developed in this work (top-1 f1-score = 60.2%). Two approaches to removing the bias from this model are explored: replacing given names with a placeholder token, and augmenting the training data with gender-swapped examples. The results for these approaches is reported, and the bias in the model was reduced by hiding given names from the model. However, bias reduction was accomplished at the expense of classification performance (top-1 f1-score = 56.6%). Augmentation of the training data with gender-swapping samples proved less effective at bias reduction than the name hiding approach on the evaluated dataset.

</p>
</details>

<details><summary><b>Artificial Neural Networks to Impute Rounded Zeros in Compositional Data</b>
<a href="https://arxiv.org/abs/2012.10300">arxiv:2012.10300</a>
&#x1F4C8; 2 <br>
<p>Matthias Templ</p></summary>
<p>

**Abstract:** Methods of deep learning have become increasingly popular in recent years, but they have not arrived in compositional data analysis. Imputation methods for compositional data are typically applied on additive, centered or isometric log-ratio representations of the data. Generally, methods for compositional data analysis can only be applied to observed positive entries in a data matrix. Therefore one tries to impute missing values or measurements that were below a detection limit. In this paper, a new method for imputing rounded zeros based on artificial neural networks is shown and compared with conventional methods. We are also interested in the question whether for ANNs, a representation of the data in log-ratios for imputation purposes, is relevant. It can be shown, that ANNs are competitive or even performing better when imputing rounded zeros of data sets with moderate size. They deliver better results when data sets are big. Also, we can see that log-ratio transformations within the artificial neural network imputation procedure nevertheless help to improve the results. This proves that the theory of compositional data analysis and the fulfillment of all properties of compositional data analysis is still very important in the age of deep learning.

</p>
</details>

<details><summary><b>Neural Network Embeddings for Test Case Prioritization</b>
<a href="https://arxiv.org/abs/2012.10154">arxiv:2012.10154</a>
&#x1F4C8; 2 <br>
<p>João Lousada, Miguel Ribeiro</p></summary>
<p>

**Abstract:** In modern software engineering, Continuous Integration (CI) has become an indispensable step towards systematically managing the life cycles of software development. Large companies struggle with keeping the pipeline updated and operational, in useful time, due to the large amount of changes and addition of features, that build on top of each other and have several developers, working on different platforms. Associated with such software changes, there is always a strong component of Testing. As teams and projects grow, exhaustive testing quickly becomes inhibitive, becoming adamant to select the most relevant test cases earlier, without compromising software quality. We have developed a new tool called Neural Network Embeeding for Test Case Prioritization (NNE-TCP) is a novel Machine-Learning (ML) framework that analyses which files were modified when there was a test status transition and learns relationships between these files and tests by mapping them into multidimensional vectors and grouping them by similarity. When new changes are made, tests that are more likely to be linked to the files modified are prioritized, reducing the resources needed to find newly introduced faults. Furthermore, NNE-TCP enables entity visualization in low-dimensional space, allowing for other manners of grouping files and tests by similarity and to reduce redundancies. By applying NNE-TCP, we show for the first time that the connection between modified files and tests is relevant and competitive relative to other traditional methods.

</p>
</details>

<details><summary><b>Semantics and explanation: why counterfactual explanations produce adversarial examples in deep neural networks</b>
<a href="https://arxiv.org/abs/2012.10076">arxiv:2012.10076</a>
&#x1F4C8; 2 <br>
<p>Kieran Browne, Ben Swift</p></summary>
<p>

**Abstract:** Recent papers in explainable AI have made a compelling case for counterfactual modes of explanation. While counterfactual explanations appear to be extremely effective in some instances, they are formally equivalent to adversarial examples. This presents an apparent paradox for explainability researchers: if these two procedures are formally equivalent, what accounts for the explanatory divide apparent between counterfactual explanations and adversarial examples? We resolve this paradox by placing emphasis back on the semantics of counterfactual expressions. Producing satisfactory explanations for deep learning systems will require that we find ways to interpret the semantics of hidden layer representations in deep neural networks.

</p>
</details>

<details><summary><b>Mention Extraction and Linking for SQL Query Generation</b>
<a href="https://arxiv.org/abs/2012.10074">arxiv:2012.10074</a>
&#x1F4C8; 2 <br>
<p>Jianqiang Ma, Zeyu Yan, Shuai Pang, Yang Zhang, Jianping Shen</p></summary>
<p>

**Abstract:** On the WikiSQL benchmark, state-of-the-art text-to-SQL systems typically take a slot-filling approach by building several dedicated models for each type of slots. Such modularized systems are not only complex butalso of limited capacity for capturing inter-dependencies among SQL clauses. To solve these problems, this paper proposes a novel extraction-linking approach, where a unified extractor recognizes all types of slot mentions appearing in the question sentence before a linker maps the recognized columns to the table schema to generate executable SQL queries. Trained with automatically generated annotations, the proposed method achieves the first place on the WikiSQL benchmark.

</p>
</details>

<details><summary><b>Attention-Based LSTM Network for COVID-19 Clinical Trial Parsing</b>
<a href="https://arxiv.org/abs/2012.10063">arxiv:2012.10063</a>
&#x1F4C8; 2 <br>
<p>Xiong Liu, Luca A. Finelli, Greg L. Hersch, Iya Khalil</p></summary>
<p>

**Abstract:** COVID-19 clinical trial design is a critical task in developing therapeutics for the prevention and treatment of COVID-19. In this study, we apply a deep learning approach to extract eligibility criteria variables from COVID-19 trials to enable quantitative analysis of trial design and optimization. Specifically, we train attention-based bidirectional Long Short-Term Memory (Att-BiLSTM) models and use the optimal model to extract entities (i.e., variables) from the eligibility criteria of COVID-19 trials. We compare the performance of Att-BiLSTM with traditional ontology-based method. The result on a benchmark dataset shows that Att-BiLSTM outperforms the ontology model. Att-BiLSTM achieves a precision of 0.942, recall of 0.810, and F1 of 0.871, while the ontology model only achieves a precision of 0.715, recall of 0.659, and F1 of 0.686. Our analyses demonstrate that Att-BiLSTM is an effective approach for characterizing patient populations in COVID-19 clinical trials.

</p>
</details>

<details><summary><b>Instance Space Analysis for the Car Sequencing Problem</b>
<a href="https://arxiv.org/abs/2012.10053">arxiv:2012.10053</a>
&#x1F4C8; 2 <br>
<p>Yuan Sun, Samuel Esler, Dhananjay Thiruvady, Andreas T. Ernst, Xiaodong Li, Kerri Morgan</p></summary>
<p>

**Abstract:** In this paper, we investigate an important research question in the car sequencing problem, that is, what characteristics make an instance hard to solve? To do so, we carry out an Instance Space Analysis for the car sequencing problem, by extracting a vector of problem features to characterize an instance and projecting feature vectors onto a two-dimensional space using principal component analysis. The resulting two dimensional visualizations provide insights into both the characteristics of the instances used for testing and to compare how these affect different optimisation algorithms. This guides us in constructing a new set of benchmark instances with a range of instance properties. These are shown to be both more diverse than the previous benchmarks and include many hard to solve instances. We systematically compare the performance of six algorithms for solving the car sequencing problem. The methods tested include three existing algorithms from the literature and three new ones. Importantly, we build machine learning models to identify the niche in the instance space that an algorithm is expected to perform well on. Our results show that the new algorithms are state-of-the-art. This analysis helps to understand problem hardness and select an appropriate algorithm for solving a given car sequencing problem instance.

</p>
</details>

<details><summary><b>Influence Maximization Under Generic Threshold-based Non-submodular Model</b>
<a href="https://arxiv.org/abs/2012.12309">arxiv:2012.12309</a>
&#x1F4C8; 1 <br>
<p>Liang Ma</p></summary>
<p>

**Abstract:** As a widely observable social effect, influence diffusion refers to a process where innovations, trends, awareness, etc. spread across the network via the social impact among individuals. Motivated by such social effect, the concept of influence maximization is coined, where the goal is to select a bounded number of the most influential nodes (seed nodes) from a social network so that they can jointly trigger the maximal influence diffusion. A rich body of research in this area is performed under statistical diffusion models with provable submodularity, which essentially simplifies the problem as the optimal result can be approximated by the simple greedy search. When the diffusion models are non-submodular, however, the research community mostly focuses on how to bound/approximate them by tractable submodular functions so as to estimate the optimal result. In other words, there is still a lack of efficient methods that can directly resolve non-submodular influence maximization problems. In this regard, we fill the gap by proposing seed selection strategies using network graphical properties in a generalized threshold-based model, called influence barricade model, which is non-submodular. Specifically, under this model, we first establish theories to reveal graphical conditions that ensure the network generated by node removals has the same optimal seed set as that in the original network. We then exploit these theoretical conditions to develop efficient algorithms by strategically removing less-important nodes and selecting seeds only in the remaining network. To the best of our knowledge, this is the first graph-based approach that directly tackles non-submodular influence maximization.

</p>
</details>

<details><summary><b>A hybrid MGA-MSGD ANN training approach for approximate solution of linear elliptic PDEs</b>
<a href="https://arxiv.org/abs/2012.11517">arxiv:2012.11517</a>
&#x1F4C8; 1 <br>
<p>Hamidreza Dehghani, Andreas Zilian</p></summary>
<p>

**Abstract:** We introduce a hybrid "Modified Genetic Algorithm-Multilevel Stochastic Gradient Descent" (MGA-MSGD) training algorithm that considerably improves accuracy and efficiency of solving 3D mechanical problems described, in strong-form, by PDEs via ANNs (Artificial Neural Networks). This presented approach allows the selection of a number of locations of interest at which the state variables are expected to fulfil the governing equations associated with a physical problem. Unlike classical PDE approximation methods such as finite differences or the finite element method, there is no need to establish and reconstruct the physical field quantity throughout the computational domain in order to predict the mechanical response at specific locations of interest. The basic idea of MGA-MSGD is the manipulation of the learnable parameters' components responsible for the error explosion so that we can train the network with relatively larger learning rates which avoids trapping in local minima. The proposed training approach is less sensitive to the learning rate value, training points density and distribution, and the random initial parameters. The distance function to minimise is where we introduce the PDEs including any physical laws and conditions (so-called, Physics Informed ANN). The Genetic algorithm is modified to be suitable for this type of ANN in which a Coarse-level Stochastic Gradient Descent (CSGD) is exploited to make the decision of the offspring qualification. Employing the presented approach, a considerable improvement in both accuracy and efficiency, compared with standard training algorithms such as classical SGD and Adam optimiser, is observed. The local displacement accuracy is studied and ensured by introducing the results of Finite Element Method (FEM) at sufficiently fine mesh as the reference displacements. A slightly more complex problem is solved ensuring its feasibility.

</p>
</details>

<details><summary><b>Modeling Silicon-Photonic Neural Networks under Uncertainties</b>
<a href="https://arxiv.org/abs/2012.10594">arxiv:2012.10594</a>
&#x1F4C8; 1 <br>
<p>Sanmitra Banerjee, Mahdi Nikdast, Krishnendu Chakrabarty</p></summary>
<p>

**Abstract:** Silicon-photonic neural networks (SPNNs) offer substantial improvements in computing speed and energy efficiency compared to their digital electronic counterparts. However, the energy efficiency and accuracy of SPNNs are highly impacted by uncertainties that arise from fabrication-process and thermal variations. In this paper, we present the first comprehensive and hierarchical study on the impact of random uncertainties on the classification accuracy of a Mach-Zehnder Interferometer (MZI)-based SPNN. We show that such impact can vary based on both the location and characteristics (e.g., tuned phase angles) of a non-ideal silicon-photonic device. Simulation results show that in an SPNN with two hidden layers and 1374 tunable-thermal-phase shifters, random uncertainties even in mature fabrication processes can lead to a catastrophic 70% accuracy loss.

</p>
</details>

<details><summary><b>Ekya: Continuous Learning of Video Analytics Models on Edge Compute Servers</b>
<a href="https://arxiv.org/abs/2012.10557">arxiv:2012.10557</a>
&#x1F4C8; 1 <br>
<p>Romil Bhardwaj, Zhengxu Xia, Ganesh Ananthanarayanan, Junchen Jiang, Nikolaos Karianakis, Yuanchao Shu, Kevin Hsieh, Victor Bahl, Ion Stoica</p></summary>
<p>

**Abstract:** Video analytics applications use edge compute servers for the analytics of the videos (for bandwidth and privacy). Compressed models that are deployed on the edge servers for inference suffer from data drift, where the live video data diverges from the training data. Continuous learning handles data drift by periodically retraining the models on new data. Our work addresses the challenge of jointly supporting inference and retraining tasks on edge servers, which requires navigating the fundamental tradeoff between the retrained model's accuracy and the inference accuracy. Our solution Ekya balances this tradeoff across multiple models and uses a micro-profiler to identify the models that will benefit the most by retraining. Ekya's accuracy gain compared to a baseline scheduler is 29% higher, and the baseline requires 4x more GPU resources to achieve the same accuracy as Ekya.

</p>
</details>

<details><summary><b>Reduction of the Number of Variables in Parametric Constrained Least-Squares Problems</b>
<a href="https://arxiv.org/abs/2012.10423">arxiv:2012.10423</a>
&#x1F4C8; 1 <br>
<p>Alberto Bemporad, Gionata Cimini</p></summary>
<p>

**Abstract:** For linearly constrained least-squares problems that depend on a vector of parameters, this paper proposes techniques for reducing the number of involved optimization variables. After first eliminating equality constraints in a numerically robust way by QR factorization, we propose a technique based on singular value decomposition (SVD) and unsupervised learning, that we call $K$-SVD, and neural classifiers to automatically partition the set of parameter vectors in $K$ nonlinear regions in which the original problem is approximated by using a smaller set of variables. For the special case of parametric constrained least-squares problems that arise from model predictive control (MPC) formulations, we propose a novel and very efficient QR factorization method for equality constraint elimination. Together with SVD or $K$-SVD, the method provides a numerically robust alternative to standard condensing and move blocking, and to other complexity reduction methods for MPC based on basis functions. We show the good performance of the proposed techniques in numerical tests and in a linearized MPC problem of a nonlinear benchmark process.

</p>
</details>

<details><summary><b>Convergence dynamics of Generative Adversarial Networks: the dual metric flows</b>
<a href="https://arxiv.org/abs/2012.10410">arxiv:2012.10410</a>
&#x1F4C8; 1 <br>
<p>Gabriel Turinici</p></summary>
<p>

**Abstract:** Fitting neural networks often resorts to stochastic (or similar) gradient descent which is a noise-tolerant (and efficient) resolution of a gradient descent dynamics. It outputs a sequence of networks parameters, which sequence evolves during the training steps. The gradient descent is the limit, when the learning rate is small and the batch size is infinite, of this set of increasingly optimal network parameters obtained during training. In this contribution, we investigate instead the convergence in the Generative Adversarial Networks used in machine learning. We study the limit of small learning rate, and show that, similar to single network training, the GAN learning dynamics tend, for vanishing learning rate to some limit dynamics. This leads us to consider evolution equations in metric spaces (which is the natural framework for evolving probability laws)that we call dual flows. We give formal definitions of solutions and prove the convergence. The theory is then applied to specific instances of GANs and we discuss how this insight helps understand and mitigate the mode collapse.

</p>
</details>

<details><summary><b>Universal Approximation in Dropout Neural Networks</b>
<a href="https://arxiv.org/abs/2012.10351">arxiv:2012.10351</a>
&#x1F4C8; 1 <br>
<p>Oxana A. Manita, Mark A. Peletier, Jacobus W. Portegies, Jaron Sanders, Albert Senen-Cerda</p></summary>
<p>

**Abstract:** We prove two universal approximation theorems for a range of dropout neural networks. These are feed-forward neural networks in which each edge is given a random $\{0,1\}$-valued filter, that have two modes of operation: in the first each edge output is multiplied by its random filter, resulting in a random output, while in the second each edge output is multiplied by the expectation of its filter, leading to a deterministic output. It is common to use the random mode during training and the deterministic mode during testing and prediction.
  Both theorems are of the following form: Given a function to approximate and a threshold $\varepsilon>0$, there exists a dropout network that is $\varepsilon$-close in probability and in $L^q$. The first theorem applies to dropout networks in the random mode. It assumes little on the activation function, applies to a wide class of networks, and can even be applied to approximation schemes other than neural networks. The core is an algebraic property that shows that deterministic networks can be exactly matched in expectation by random networks. The second theorem makes stronger assumptions and gives a stronger result. Given a function to approximate, it provides existence of a network that approximates in both modes simultaneously. Proof components are a recursive replacement of edges by independent copies, and a special first-layer replacement that couples the resulting larger network to the input.
  The functions to be approximated are assumed to be elements of general normed spaces, and the approximations are measured in the corresponding norms. The networks are constructed explicitly. Because of the different methods of proof, the two results give independent insight into the approximation properties of random dropout networks. With this, we establish that dropout neural networks broadly satisfy a universal-approximation property.

</p>
</details>

<details><summary><b>Learning from History for Byzantine Robust Optimization</b>
<a href="https://arxiv.org/abs/2012.10333">arxiv:2012.10333</a>
&#x1F4C8; 1 <br>
<p>Sai Praneeth Karimireddy, Lie He, Martin Jaggi</p></summary>
<p>

**Abstract:** Byzantine robustness has received significant attention recently given its importance for distributed and federated learning. In spite of this, we identify severe flaws in existing algorithms even when the data across the participants is assumed to be identical. First, we show that most existing robust aggregation rules may not converge even in the absence of any Byzantine attackers, because they are overly sensitive to the distribution of the noise in the stochastic gradients. Secondly, we show that even if the aggregation rules may succeed in limiting the influence of the attackers in a single round, the attackers can couple their attacks across time eventually leading to divergence. To address these issues, we present two surprisingly simple strategies: a new iterative clipping procedure, and incorporating worker momentum to overcome time-coupled attacks. This is the first provably robust method for the standard stochastic non-convex optimization setting.

</p>
</details>

<details><summary><b>Deep learning and high harmonic generation</b>
<a href="https://arxiv.org/abs/2012.10328">arxiv:2012.10328</a>
&#x1F4C8; 1 <br>
<p>M. Lytova, M. Spanner, I. Tamblyn</p></summary>
<p>

**Abstract:** Using machine learning, we explore the utility of various deep neural networks (NN) when applied to high harmonic generation (HHG) scenarios. First, we train the NNs to predict the time-dependent dipole and spectra of HHG emission from reduced-dimensionality models of di- and triatomic systems based of on sets of randomly generated parameters (laser pulse intensity, internuclear distance, and molecular orientation). These networks, once trained, are useful tools to rapidly generate the HHG spectra of our systems. Similarly, we have trained the NNs to solve the inverse problem - to determine the molecular parameters based on HHG spectra or dipole acceleration data. These types of networks could then be used as spectroscopic tools to invert HHG spectra in order to recover the underlying physical parameters of a system. Next, we demonstrate that transfer learning can be applied to our networks to expand the range of applicability of the networks with only a small number of new test cases added to our training sets. Finally, we demonstrate NNs that can be used to classify molecules by type: di- or triatomic, symmetric or asymmetric, wherein we can even rely on fairly simple fully connected neural networks. With outlooks toward training with experimental data, these NN topologies offer a novel set of spectroscopic tools that could be incorporated into HHG experiments.

</p>
</details>

<details><summary><b>Kernel Methods for Unobserved Confounding: Negative Controls, Proxies, and Instruments</b>
<a href="https://arxiv.org/abs/2012.10315">arxiv:2012.10315</a>
&#x1F4C8; 1 <br>
<p>Rahul Singh</p></summary>
<p>

**Abstract:** Negative control is a strategy for learning the causal relationship between treatment and outcome in the presence of unmeasured confounding. The treatment effect can nonetheless be identified if two auxiliary variables are available: a negative control treatment (which has no effect on the actual outcome), and a negative control outcome (which is not affected by the actual treatment). These auxiliary variables can also be viewed as proxies for a traditional set of control variables, and they bear resemblance to instrumental variables. I propose a new family of non-parametric algorithms for learning treatment effects with negative controls. I consider treatment effects of the population, of sub-populations, and of alternative populations. I allow for data that may be discrete or continuous, and low-, high-, or infinite-dimensional. I impose the additional structure of the reproducing kernel Hilbert space (RKHS), a popular non-parametric setting in machine learning. I prove uniform consistency and provide finite sample rates of convergence. I evaluate the estimators in simulations.

</p>
</details>

<details><summary><b>Adversarially Robust Estimate and Risk Analysis in Linear Regression</b>
<a href="https://arxiv.org/abs/2012.10278">arxiv:2012.10278</a>
&#x1F4C8; 1 <br>
<p>Yue Xing, Ruizhi Zhang, Guang Cheng</p></summary>
<p>

**Abstract:** Adversarially robust learning aims to design algorithms that are robust to small adversarial perturbations on input variables. Beyond the existing studies on the predictive performance to adversarial samples, our goal is to understand statistical properties of adversarially robust estimates and analyze adversarial risk in the setup of linear regression models. By discovering the statistical minimax rate of convergence of adversarially robust estimators, we emphasize the importance of incorporating model information, e.g., sparsity, in adversarially robust learning. Further, we reveal an explicit connection of adversarial and standard estimates, and propose a straightforward two-stage adversarial learning framework, which facilitates to utilize model structure information to improve adversarial robustness. In theory, the consistency of the adversarially robust estimator is proven and its Bahadur representation is also developed for the statistical inference purpose. The proposed estimator converges in a sharp rate under either low-dimensional or sparse scenario. Moreover, our theory confirms two phenomena in adversarially robust learning: adversarial robustness hurts generalization, and unlabeled data help improve the generalization. In the end, we conduct numerical simulations to verify our theory.

</p>
</details>

<details><summary><b>Generative Neural Samplers for the Quantum Heisenberg Chain</b>
<a href="https://arxiv.org/abs/2012.10264">arxiv:2012.10264</a>
&#x1F4C8; 1 <br>
<p>Johanna Vielhaben, Nils Strodthoff</p></summary>
<p>

**Abstract:** Generative neural samplers offer a complementary approach to Monte Carlo methods for problems in statistical physics and quantum field theory. This work tests the ability of generative neural samplers to estimate observables for real-world low-dimensional spin systems. It maps out how autoregressive models can sample configurations of a quantum Heisenberg chain via a classical approximation based on the Suzuki-Trotter transformation. We present results for energy, specific heat and susceptibility for the isotropic XXX and the anisotropic XY chain that are in good agreement with Monte Carlo results within the same approximation scheme.

</p>
</details>

<details><summary><b>Resource-efficient DNNs for Keyword Spotting using Neural Architecture Search and Quantization</b>
<a href="https://arxiv.org/abs/2012.10138">arxiv:2012.10138</a>
&#x1F4C8; 1 <br>
<p>David Peter, Wolfgang Roth, Franz Pernkopf</p></summary>
<p>

**Abstract:** This paper introduces neural architecture search (NAS) for the automatic discovery of small models for keyword spotting (KWS) in limited resource environments. We employ a differentiable NAS approach to optimize the structure of convolutional neural networks (CNNs) to maximize the classification accuracy while minimizing the number of operations per inference. Using NAS only, we were able to obtain a highly efficient model with 95.4% accuracy on the Google speech commands dataset with 494.8 kB of memory usage and 19.6 million operations. Additionally, weight quantization is used to reduce the memory consumption even further. We show that weight quantization to low bit-widths (e.g. 1 bit) can be used without substantial loss in accuracy. By increasing the number of input features from 10 MFCC to 20 MFCC we were able to increase the accuracy to 96.3% at 340.1 kB of memory usage and 27.1 million operations.

</p>
</details>

<details><summary><b>Stable Implementation of Probabilistic ODE Solvers</b>
<a href="https://arxiv.org/abs/2012.10106">arxiv:2012.10106</a>
&#x1F4C8; 1 <br>
<p>Nicholas Krämer, Philipp Hennig</p></summary>
<p>

**Abstract:** Probabilistic solvers for ordinary differential equations (ODEs) provide efficient quantification of numerical uncertainty associated with simulation of dynamical systems. Their convergence rates have been established by a growing body of theoretical analysis. However, these algorithms suffer from numerical instability when run at high order or with small step-sizes -- that is, exactly in the regime in which they achieve the highest accuracy. The present work proposes and examines a solution to this problem. It involves three components: accurate initialisation, a coordinate change preconditioner that makes numerical stability concerns step-size-independent, and square-root implementation. Using all three techniques enables numerical computation of probabilistic solutions of ODEs with algorithms of order up to 11, as demonstrated on a set of challenging test problems. The resulting rapid convergence is shown to be competitive to high-order, state-of-the-art, classical methods. As a consequence, a barrier between analysing probabilistic ODE solvers and applying them to interesting machine learning problems is effectively removed.

</p>
</details>

<details><summary><b>A closed form scale bound for the $(ε, δ)$-differentially private Gaussian Mechanism valid for all privacy regimes</b>
<a href="https://arxiv.org/abs/2012.10523">arxiv:2012.10523</a>
&#x1F4C8; 0 <br>
<p>Staal A. Vinterbo</p></summary>
<p>

**Abstract:** The standard closed form lower bound on $σ$ for providing $(ε, δ)$-differential privacy by adding zero mean Gaussian noise with variance $σ^2$ is $σ> Δ\sqrt {2}ε^{-1} \sqrt {\log\left( 5/4δ^{-1} \right)}$ for $ε\in (0,1)$. We present a similar closed form bound $σ\geq Δ(\sqrt{2}ε)^{-1} \left(\sqrt{z}+\sqrt{z+ε}\right)$ for $z=-\log\left(δ\left(2-δ\right)\right)$ that is valid for all $ε> 0$ and is always lower (better) for $ε< 1$ and $δ\leq 0.946$. Both bounds are based on fulfilling a particular sufficient condition. For $δ< 1$, we present an analytical bound that is optimal for this condition and is necessarily larger than $Δ/\sqrt{2ε}$.

</p>
</details>


[Next Page](2020/2020-12/2020-12-17.md)
