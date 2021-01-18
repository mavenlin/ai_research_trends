## Summary for 2020-12-22, created on 2021-01-18


<details><summary><b>Neuroevolutionary learning of particles and protocols for self-assembly</b>
<a href="https://arxiv.org/abs/2012.11832">arxiv:2012.11832</a>
&#x1F4C8; 85 <br>
<p>Stephen Whitelam, Isaac Tamblyn</p></summary>
<p>

**Abstract:** Within simulations of molecules deposited on a surface we show that neuroevolutionary learning can design particles and time-dependent protocols to promote self-assembly, without input from physical concepts such as thermal equilibrium or mechanical stability and without prior knowledge of candidate or competing structures. The learning algorithm is capable of both directed and exploratory design: it can assemble a material with a user-defined property, or search for novelty in the space of specified order parameters. In the latter mode it explores the space of what can be made rather than the space of structures that are low in energy but not necessarily kinetically accessible.

</p>
</details>

<details><summary><b>YolactEdge: Real-time Instance Segmentation on the Edge (Jetson AGX Xavier: 30 FPS, RTX 2080 Ti: 170 FPS)</b>
<a href="https://arxiv.org/abs/2012.12259">arxiv:2012.12259</a>
&#x1F4C8; 71 <br>
<p>Haotian Liu, Rafael A. Rivera Soto, Fanyi Xiao, Yong Jae Lee</p></summary>
<p>

**Abstract:** We propose YolactEdge, the first competitive instance segmentation approach that runs on small edge devices at real-time speeds. Specifically, YolactEdge runs at up to 30.8 FPS on a Jetson AGX Xavier (and 172.7 FPS on an RTX 2080 Ti) with a ResNet-101 backbone on 550x550 resolution images. To achieve this, we make two improvements to the state-of-the-art image-based real-time method YOLACT: (1) TensorRT optimization while carefully trading off speed and accuracy, and (2) a novel feature warping module to exploit temporal redundancy in videos. Experiments on the YouTube VIS and MS COCO datasets demonstrate that YolactEdge produces a 3-5x speed up over existing real-time methods while producing competitive mask and box detection accuracy. We also conduct ablation studies to dissect our design choices and modules. Code and models are available at https://github.com/haotian-liu/yolact_edge.

</p>
</details>

<details><summary><b>Is the brain macroscopically linear? A system identification of resting state dynamics</b>
<a href="https://arxiv.org/abs/2012.12351">arxiv:2012.12351</a>
&#x1F4C8; 30 <br>
<p>Erfan Nozari, Jennifer Stiso, Lorenzo Caciagli, Eli J. Cornblath, Xiaosong He, Maxwell A. Bertolero, Arun S. Mahadevan, George J. Pappas, Danielle S. Bassett</p></summary>
<p>

**Abstract:** A central challenge in the computational modeling of neural dynamics is the trade-off between accuracy and simplicity. At the level of individual neurons, nonlinear dynamics are both experimentally established and essential for neuronal functioning. An implicit assumption has thus formed that an accurate computational model of whole-brain dynamics must also be highly nonlinear, whereas linear models may provide a first-order approximation. Here, we provide a rigorous and data-driven investigation of this hypothesis at the level of whole-brain blood-oxygen-level-dependent (BOLD) and macroscopic field potential dynamics by leveraging the theory of system identification. Using functional MRI (fMRI) and intracranial EEG (iEEG), we model the resting state activity of 700 subjects in the Human Connectome Project (HCP) and 122 subjects from the Restoring Active Memory (RAM) project using state-of-the-art linear and nonlinear model families. We assess relative model fit using predictive power, computational complexity, and the extent of residual dynamics unexplained by the model. Contrary to our expectations, linear auto-regressive models achieve the best measures across all three metrics, eliminating the trade-off between accuracy and simplicity. To understand and explain this linearity, we highlight four properties of macroscopic neurodynamics which can counteract or mask microscopic nonlinear dynamics: averaging over space, averaging over time, observation noise, and limited data samples. Whereas the latter two are technological limitations and can improve in the future, the former two are inherent to aggregated macroscopic brain activity. Our results, together with the unparalleled interpretability of linear models, can greatly facilitate our understanding of macroscopic neural dynamics and the principled design of model-based interventions for the treatment of neuropsychiatric disorders.

</p>
</details>

<details><summary><b>Unadversarial Examples: Designing Objects for Robust Vision</b>
<a href="https://arxiv.org/abs/2012.12235">arxiv:2012.12235</a>
&#x1F4C8; 25 <br>
<p>Hadi Salman, Andrew Ilyas, Logan Engstrom, Sai Vemprala, Aleksander Madry, Ashish Kapoor</p></summary>
<p>

**Abstract:** We study a class of realistic computer vision settings wherein one can influence the design of the objects being recognized. We develop a framework that leverages this capability to significantly improve vision models' performance and robustness. This framework exploits the sensitivity of modern machine learning algorithms to input perturbations in order to design "robust objects," i.e., objects that are explicitly optimized to be confidently detected or classified. We demonstrate the efficacy of the framework on a wide variety of vision-based tasks ranging from standard benchmarks, to (in-simulation) robotics, to real-world experiments. Our code can be found at https://git.io/unadversarial .

</p>
</details>

<details><summary><b>TorchMD: A deep learning framework for molecular simulations</b>
<a href="https://arxiv.org/abs/2012.12106">arxiv:2012.12106</a>
&#x1F4C8; 25 <br>
<p>Stefan Doerr, Maciej Majewsk, Adrià Pérez, Andreas Krämer, Cecilia Clementi, Frank Noe, Toni Giorgino, Gianni De Fabritiis</p></summary>
<p>

**Abstract:** Molecular dynamics simulations provide a mechanistic description of molecules by relying on empirical potentials. The quality and transferability of such potentials can be improved leveraging data-driven models derived with machine learning approaches. Here, we present TorchMD, a framework for molecular simulations with mixed classical and machine learning potentials. All of force computations including bond, angle, dihedral, Lennard-Jones and Coulomb interactions are expressed as PyTorch arrays and operations. Moreover, TorchMD enables learning and simulating neural network potentials. We validate it using standard Amber all-atom simulations, learning an ab-initio potential, performing an end-to-end training and finally learning and simulating a coarse-grained model for protein folding. We believe that TorchMD provides a useful tool-set to support molecular simulations of machine learning potentials. Code and data are freely available at \url{github.com/torchmd}.

</p>
</details>

<details><summary><b>ActionBert: Leveraging User Actions for Semantic Understanding of User Interfaces</b>
<a href="https://arxiv.org/abs/2012.12350">arxiv:2012.12350</a>
&#x1F4C8; 9 <br>
<p>Zecheng He, Srinivas Sunkara, Xiaoxue Zang, Ying Xu, Lijuan Liu, Nevan Wichers, Gabriel Schubiner, Ruby Lee, Jindong Chen</p></summary>
<p>

**Abstract:** As mobile devices are becoming ubiquitous, regularly interacting with a variety of user interfaces (UIs) is a common aspect of daily life for many people. To improve the accessibility of these devices and to enable their usage in a variety of settings, building models that can assist users and accomplish tasks through the UI is vitally important. However, there are several challenges to achieve this. First, UI components of similar appearance can have different functionalities, making understanding their function more important than just analyzing their appearance. Second, domain-specific features like Document Object Model (DOM) in web pages and View Hierarchy (VH) in mobile applications provide important signals about the semantics of UI elements, but these features are not in a natural language format. Third, owing to a large diversity in UIs and absence of standard DOM or VH representations, building a UI understanding model with high coverage requires large amounts of training data.
  Inspired by the success of pre-training based approaches in NLP for tackling a variety of problems in a data-efficient way, we introduce a new pre-trained UI representation model called ActionBert. Our methodology is designed to leverage visual, linguistic and domain-specific features in user interaction traces to pre-train generic feature representations of UIs and their components. Our key intuition is that user actions, e.g., a sequence of clicks on different UI components, reveals important information about their functionality. We evaluate the proposed model on a wide variety of downstream tasks, ranging from icon classification to UI component retrieval based on its natural language description. Experiments show that the proposed ActionBert model outperforms multi-modal baselines across all downstream tasks by up to 15.5%.

</p>
</details>

<details><summary><b>Few-Shot Text Generation with Pattern-Exploiting Training</b>
<a href="https://arxiv.org/abs/2012.11926">arxiv:2012.11926</a>
&#x1F4C8; 7 <br>
<p>Timo Schick, Hinrich Schütze</p></summary>
<p>

**Abstract:** Providing pretrained language models with simple task descriptions or prompts in natural language yields impressive few-shot results for a wide range of text classification tasks when combined with gradient-based learning from examples. In this paper, we show that the underlying idea can also be applied to text generation tasks: We adapt Pattern-Exploiting Training (PET), a recently proposed few-shot approach, for finetuning generative language models on text generation tasks. On several text summarization and headline generation datasets, our proposed variant of PET gives consistent improvements over a strong baseline in few-shot settings.

</p>
</details>

<details><summary><b>AudioViewer: Learning to Visualize Sound</b>
<a href="https://arxiv.org/abs/2012.13341">arxiv:2012.13341</a>
&#x1F4C8; 6 <br>
<p>Yuchi Zhang, Willis Peng, Bastian Wandt, Helge Rhodin</p></summary>
<p>

**Abstract:** Sensory substitution can help persons with perceptual deficits. In this work, we attempt to visualize audio with video. Our long-term goal is to create sound perception for hearing impaired people, for instance, to facilitate feedback for training deaf speech. Different from existing models that translate between speech and text or text and images, we target an immediate and low-level translation that applies to generic environment sounds and human speech without delay. No canonical mapping is known for this artificial translation task. Our design is to translate from audio to video by compressing both into a common latent space with shared structure. Our core contribution is the development and evaluation of learned mappings that respect human perception limits and maximize user comfort by enforcing priors and combining strategies from unpaired image translation and disentanglement. We demonstrate qualitatively and quantitatively that our AudioViewer model maintains important audio features in the generated video and that generated videos of faces and numbers are well suited for visualizing high-dimensional audio features since they can easily be parsed by humans to match and distinguish between sounds, words, and speakers.

</p>
</details>

<details><summary><b>Global Models for Time Series Forecasting: A Simulation Study</b>
<a href="https://arxiv.org/abs/2012.12485">arxiv:2012.12485</a>
&#x1F4C8; 5 <br>
<p>Hansika Hewamalage, Christoph Bergmeir, Kasun Bandara</p></summary>
<p>

**Abstract:** In the current context of Big Data, the nature of many forecasting problems has changed from predicting isolated time series to predicting many time series from similar sources. This has opened up the opportunity to develop competitive global forecasting models that simultaneously learn from many time series. But, it still remains unclear when global forecasting models can outperform the univariate benchmarks, especially along the dimensions of the homogeneity/heterogeneity of series, the complexity of patterns in the series, the complexity of forecasting models, and the lengths/number of series. Our study attempts to address this problem through investigating the effect from these factors, by simulating a number of datasets that have controllable time series characteristics. Specifically, we simulate time series from simple data generating processes (DGP), such as Auto Regressive (AR) and Seasonal AR, to complex DGPs, such as Chaotic Logistic Map, Self-Exciting Threshold Auto-Regressive, and Mackey-Glass Equations. The data heterogeneity is introduced by mixing time series generated from several DGPs into a single dataset. The lengths and the number of series in the dataset are varied in different scenarios. We perform experiments on these datasets using global forecasting models including Recurrent Neural Networks (RNN), Feed-Forward Neural Networks, Pooled Regression (PR) models and Light Gradient Boosting Models (LGBM), and compare their performance against standard statistical univariate forecasting techniques. Our experiments demonstrate that when trained as global forecasting models, techniques such as RNNs and LGBMs, which have complex non-linear modelling capabilities, are competitive methods in general under challenging forecasting scenarios such as series having short lengths, datasets with heterogeneous series and having minimal prior knowledge of the patterns of the series.

</p>
</details>

<details><summary><b>Graph-Evolving Meta-Learning for Low-Resource Medical Dialogue Generation</b>
<a href="https://arxiv.org/abs/2012.11988">arxiv:2012.11988</a>
&#x1F4C8; 5 <br>
<p>Shuai Lin, Pan Zhou, Xiaodan Liang, Jianheng Tang, Ruihui Zhao, Ziliang Chen, Liang Lin</p></summary>
<p>

**Abstract:** Human doctors with well-structured medical knowledge can diagnose a disease merely via a few conversations with patients about symptoms. In contrast, existing knowledge-grounded dialogue systems often require a large number of dialogue instances to learn as they fail to capture the correlations between different diseases and neglect the diagnostic experience shared among them. To address this issue, we propose a more natural and practical paradigm, i.e., low-resource medical dialogue generation, which can transfer the diagnostic experience from source diseases to target ones with a handful of data for adaptation. It is capitalized on a commonsense knowledge graph to characterize the prior disease-symptom relations. Besides, we develop a Graph-Evolving Meta-Learning (GEML) framework that learns to evolve the commonsense graph for reasoning disease-symptom correlations in a new disease, which effectively alleviates the needs of a large number of dialogues. More importantly, by dynamically evolving disease-symptom graphs, GEML also well addresses the real-world challenges that the disease-symptom correlations of each disease may vary or evolve along with more diagnostic cases. Extensive experiment results on the CMDD dataset and our newly-collected Chunyu dataset testify the superiority of our approach over state-of-the-art approaches. Besides, our GEML can generate an enriched dialogue-sensitive knowledge graph in an online manner, which could benefit other tasks grounded on knowledge graph.

</p>
</details>

<details><summary><b>Analyzing the response to TV serials retelecast during COVID19 lockdown in India</b>
<a href="https://arxiv.org/abs/2101.02628">arxiv:2101.02628</a>
&#x1F4C8; 4 <br>
<p>Sandeep Ranjan</p></summary>
<p>

**Abstract:** TV serials are a popular source of entertainment. The ongoing COVID19 lockdown has a high probability of degrading the publics mental health. The Government of India started the retelecast of yesteryears popular TV serials on public broadcaster Doordarshan from 28th March 2020 to 31st July 2020. Tweets corresponding to the Doordarshan hashtag were mined to create a dataset. The experiment aims to analyze the publics response to the retelecast of TV serials by calculating the sentiment score of the tweet dataset. Datasets mean sentiment score of 0.65 and high share 64.58% of positive tweets signifies the acceptance of Doordarshans retelecast decision. The sentiment analysis result also reflects the positive state of mind of the public.

</p>
</details>

<details><summary><b>IIRC: Incremental Implicitly-Refined Classification</b>
<a href="https://arxiv.org/abs/2012.12477">arxiv:2012.12477</a>
&#x1F4C8; 4 <br>
<p>Mohamed Abdelsalam, Mojtaba Faramarzi, Shagun Sodhani, Sarath Chandar</p></summary>
<p>

**Abstract:** We introduce the "Incremental Implicitly-Refined Classi-fication (IIRC)" setup, an extension to the class incremental learning setup where the incoming batches of classes have two granularity levels. i.e., each sample could have a high-level (coarse) label like "bear" and a low-level (fine) label like "polar bear". Only one label is provided at a time, and the model has to figure out the other label if it has already learnfed it. This setup is more aligned with real-life scenarios, where a learner usually interacts with the same family of entities multiple times, discovers more granularity about them, while still trying not to forget previous knowledge. Moreover, this setup enables evaluating models for some important lifelong learning challenges that cannot be easily addressed under the existing setups. These challenges can be motivated by the example "if a model was trained on the class bear in one task and on polar bear in another task, will it forget the concept of bear, will it rightfully infer that a polar bear is still a bear? and will it wrongfully associate the label of polar bear to other breeds of bear?". We develop a standardized benchmark that enables evaluating models on the IIRC setup. We evaluate several state-of-the-art lifelong learning algorithms and highlight their strengths and limitations. For example, distillation-based methods perform relatively well but are prone to incorrectly predicting too many labels per image. We hope that the proposed setup, along with the benchmark, would provide a meaningful problem setting to the practitioners

</p>
</details>

<details><summary><b>Self-supervised self-supervision by combining deep learning and probabilistic logic</b>
<a href="https://arxiv.org/abs/2012.12474">arxiv:2012.12474</a>
&#x1F4C8; 4 <br>
<p>Hunter Lang, Hoifung Poon</p></summary>
<p>

**Abstract:** Labeling training examples at scale is a perennial challenge in machine learning. Self-supervision methods compensate for the lack of direct supervision by leveraging prior knowledge to automatically generate noisy labeled examples. Deep probabilistic logic (DPL) is a unifying framework for self-supervised learning that represents unknown labels as latent variables and incorporates diverse self-supervision using probabilistic logic to train a deep neural network end-to-end using variational EM. While DPL is successful at combining pre-specified self-supervision, manually crafting self-supervision to attain high accuracy may still be tedious and challenging. In this paper, we propose Self-Supervised Self-Supervision (S4), which adds to DPL the capability to learn new self-supervision automatically. Starting from an initial "seed," S4 iteratively uses the deep neural network to propose new self supervision. These are either added directly (a form of structured self-training) or verified by a human expert (as in feature-based active learning). Experiments show that S4 is able to automatically propose accurate self-supervision and can often nearly match the accuracy of supervised methods with a tiny fraction of the human effort.

</p>
</details>

<details><summary><b>Latent Feature Representation via Unsupervised Learning for Pattern Discovery in Massive Electron Microscopy Image Volumes</b>
<a href="https://arxiv.org/abs/2012.12175">arxiv:2012.12175</a>
&#x1F4C8; 4 <br>
<p>Gary B Huang, Huei-Fang Yang, Shin-ya Takemura, Pat Rivlin, Stephen M Plaza</p></summary>
<p>

**Abstract:** We propose a method to facilitate exploration and analysis of new large data sets. In particular, we give an unsupervised deep learning approach to learning a latent representation that captures semantic similarity in the data set. The core idea is to use data augmentations that preserve semantic meaning to generate synthetic examples of elements whose feature representations should be close to one another.
  We demonstrate the utility of our method applied to nano-scale electron microscopy data, where even relatively small portions of animal brains can require terabytes of image data. Although supervised methods can be used to predict and identify known patterns of interest, the scale of the data makes it difficult to mine and analyze patterns that are not known a priori. We show the ability of our learned representation to enable query by example, so that if a scientist notices an interesting pattern in the data, they can be presented with other locations with matching patterns. We also demonstrate that clustering of data in the learned space correlates with biologically-meaningful distinctions. Finally, we introduce a visualization tool and software ecosystem to facilitate user-friendly interactive analysis and uncover interesting biological patterns. In short, our work opens possible new avenues in understanding of and discovery in large data sets, arising in domains such as EM analysis.

</p>
</details>

<details><summary><b>This is not the Texture you are looking for! Introducing Novel Counterfactual Explanations for Non-Experts using Generative Adversarial Learning</b>
<a href="https://arxiv.org/abs/2012.11905">arxiv:2012.11905</a>
&#x1F4C8; 4 <br>
<p>Silvan Mertes, Tobias Huber, Katharina Weitz, Alexander Heimerl, Elisabeth André</p></summary>
<p>

**Abstract:** With the ongoing rise of machine learning, the need for methods for explaining decisions made by artificial intelligence systems is becoming a more and more important topic. Especially for image classification tasks, many state-of-the-art tools to explain such classifiers rely on visual highlighting of important areas of the input data. Contrary, counterfactual explanation systems try to enable a counterfactual reasoning by modifying the input image in a way such that the classifier would have made a different prediction. By doing so, the users of counterfactual explanation systems are equipped with a completely different kind of explanatory information. However, methods for generating realistic counterfactual explanations for image classifiers are still rare. In this work, we present a novel approach to generate such counterfactual image explanations based on adversarial image-to-image translation techniques. Additionally, we conduct a user study to evaluate our approach in a use case which was inspired by a healthcare scenario. Our results show that our approach leads to significantly better results regarding mental models, explanation satisfaction, trust, emotions, and self-efficacy than two state-of-the art systems that work with saliency maps, namely LIME and LRP.

</p>
</details>

<details><summary><b>Modelling Human Routines: Conceptualising Social Practice Theory for Agent-Based Simulation</b>
<a href="https://arxiv.org/abs/2012.11903">arxiv:2012.11903</a>
&#x1F4C8; 4 <br>
<p>Rijk Mercuur, Virginia Dignum, Catholijn M. Jonker</p></summary>
<p>

**Abstract:** Our routines play an important role in a wide range of social challenges such as climate change, disease outbreaks and coordinating staff and patients in a hospital. To use agent-based simulations (ABS) to understand the role of routines in social challenges we need an agent framework that integrates routines. This paper provides the domain-independent Social Practice Agent (SoPrA) framework that satisfies requirements from the literature to simulate our routines. By choosing the appropriate concepts from the literature on agent theory, social psychology and social practice theory we ensure SoPrA correctly depicts current evidence on routines. By creating a consistent, modular and parsimonious framework suitable for multiple domains we enhance the usability of SoPrA. SoPrA provides ABS researchers with a conceptual, formal and computational framework to simulate routines and gain new insights into social systems.

</p>
</details>

<details><summary><b>Personalized Adaptive Meta Learning for Cold-start User Preference Prediction</b>
<a href="https://arxiv.org/abs/2012.11842">arxiv:2012.11842</a>
&#x1F4C8; 4 <br>
<p>Runsheng Yu, Yu Gong, Xu He, Bo An, Yu Zhu, Qingwen Liu, Wenwu Ou</p></summary>
<p>

**Abstract:** A common challenge in personalized user preference prediction is the cold-start problem. Due to the lack of user-item interactions, directly learning from the new users' log data causes serious over-fitting problem. Recently, many existing studies regard the cold-start personalized preference prediction as a few-shot learning problem, where each user is the task and recommended items are the classes, and the gradient-based meta learning method (MAML) is leveraged to address this challenge. However, in real-world application, the users are not uniformly distributed (i.e., different users may have different browsing history, recommended items, and user profiles. We define the major users as the users in the groups with large numbers of users sharing similar user information, and other users are the minor users), existing MAML approaches tend to fit the major users and ignore the minor users. To address this cold-start task-overfitting problem, we propose a novel personalized adaptive meta learning approach to consider both the major and the minor users with three key contributions: 1) We are the first to present a personalized adaptive learning rate meta-learning approach to improve the performance of MAML by focusing on both the major and minor users. 2) To provide better personalized learning rates for each user, we introduce a similarity-based method to find similar users as a reference and a tree-based method to store users' features for fast search. 3) To reduce the memory usage, we design a memory agnostic regularizer to further reduce the space complexity to constant while maintain the performance. Experiments on MovieLens, BookCrossing, and real-world production datasets reveal that our method outperforms the state-of-the-art methods dramatically for both the minor and major users.

</p>
</details>

<details><summary><b>Open Set Domain Adaptation by Extreme Value Theory</b>
<a href="https://arxiv.org/abs/2101.02561">arxiv:2101.02561</a>
&#x1F4C8; 3 <br>
<p>Yiming Xu, Diego Klabjan</p></summary>
<p>

**Abstract:** Common domain adaptation techniques assume that the source domain and the target domain share an identical label space, which is problematic since when target samples are unlabeled we have no knowledge on whether the two domains share the same label space. When this is not the case, the existing methods fail to perform well because the additional unknown classes are also matched with the source domain during adaptation. In this paper, we tackle the open set domain adaptation problem under the assumption that the source and the target label spaces only partially overlap, and the task becomes when the unknown classes exist, how to detect the target unknown classes and avoid aligning them with the source domain. We propose to utilize an instance-level reweighting strategy for domain adaptation where the weights indicate the likelihood of a sample belonging to known classes and to model the tail of the entropy distribution with Extreme Value Theory for unknown class detection. Experiments on conventional domain adaptation datasets show that the proposed method outperforms the state-of-the-art models.

</p>
</details>

<details><summary><b>Limitations of Deep Neural Networks: a discussion of G. Marcus' critical appraisal of deep learning</b>
<a href="https://arxiv.org/abs/2012.15754">arxiv:2012.15754</a>
&#x1F4C8; 3 <br>
<p>Stefanos Tsimenidis</p></summary>
<p>

**Abstract:** Deep neural networks have triggered a revolution in artificial intelligence, having been applied with great results in medical imaging, semi-autonomous vehicles, ecommerce, genetics research, speech recognition, particle physics, experimental art, economic forecasting, environmental science, industrial manufacturing, and a wide variety of applications in nearly every field. This sudden success, though, may have intoxicated the research community and blinded them to the potential pitfalls of assigning deep learning a higher status than warranted. Also, research directed at alleviating the weaknesses of deep learning may seem less attractive to scientists and engineers, who focus on the low-hanging fruit of finding more and more applications for deep learning models, thus letting short-term benefits hamper long-term scientific progress. Gary Marcus wrote a paper entitled Deep Learning: A Critical Appraisal, and here we discuss Marcus' core ideas, as well as attempt a general assessment of the subject. This study examines some of the limitations of deep neural networks, with the intention of pointing towards potential paths for future research, and of clearing up some metaphysical misconceptions, held by numerous researchers, that may misdirect them.

</p>
</details>

<details><summary><b>Augmenting Policy Learning with Routines Discovered from a Demonstration</b>
<a href="https://arxiv.org/abs/2012.12469">arxiv:2012.12469</a>
&#x1F4C8; 3 <br>
<p>Zelin Zhao, Chuang Gan, Jiajun Wu, Xiaoxiao Guo, Joshua B. Tenenbaum</p></summary>
<p>

**Abstract:** Humans can abstract prior knowledge from very little data and use it to boost skill learning. In this paper, we propose routine-augmented policy learning (RAPL), which discovers routines composed of primitive actions from a single demonstration and uses discovered routines to augment policy learning. To discover routines from the demonstration, we first abstract routine candidates by identifying grammar over the demonstrated action trajectory. Then, the best routines measured by length and frequency are selected to form a routine library. We propose to learn policy simultaneously at primitive-level and routine-level with discovered routines, leveraging the temporal structure of routines. Our approach enables imitating expert behavior at multiple temporal scales for imitation learning and promotes reinforcement learning exploration. Extensive experiments on Atari games demonstrate that RAPL improves the state-of-the-art imitation learning method SQIL and reinforcement learning method A2C. Further, we show that discovered routines can generalize to unseen levels and difficulties on the CoinRun benchmark.

</p>
</details>

<details><summary><b>Future-Guided Incremental Transformer for Simultaneous Translation</b>
<a href="https://arxiv.org/abs/2012.12465">arxiv:2012.12465</a>
&#x1F4C8; 3 <br>
<p>Shaolei Zhang, Yang Feng, Liangyou Li</p></summary>
<p>

**Abstract:** Simultaneous translation (ST) starts translations synchronously while reading source sentences, and is used in many online scenarios. The previous wait-k policy is concise and achieved good results in ST. However, wait-k policy faces two weaknesses: low training speed caused by the recalculation of hidden states and lack of future source information to guide training. For the low training speed, we propose an incremental Transformer with an average embedding layer (AEL) to accelerate the speed of calculation of the hidden states during training. For future-guided training, we propose a conventional Transformer as the teacher of the incremental Transformer, and try to invisibly embed some future information in the model through knowledge distillation. We conducted experiments on Chinese-English and German-English simultaneous translation tasks and compared with the wait-k policy to evaluate the proposed method. Our method can effectively increase the training speed by about 28 times on average at different k and implicitly embed some predictive abilities in the model, achieving better translation quality than wait-k baseline.

</p>
</details>

<details><summary><b>Skeleton-based Approaches based on Machine Vision: A Survey</b>
<a href="https://arxiv.org/abs/2012.12447">arxiv:2012.12447</a>
&#x1F4C8; 3 <br>
<p>Jie Li, Binglin Li, Min Gao</p></summary>
<p>

**Abstract:** Recently, skeleton-based approaches have achieved rapid progress on the basis of great success in skeleton representation. Plenty of researches focus on solving specific problems according to skeleton features. Some skeleton-based approaches have been mentioned in several overviews on object detection as a non-essential part. Nevertheless, there has not been any thorough analysis of skeleton-based approaches attentively. Instead of describing these techniques in terms of theoretical constructs, we devote to summarizing skeleton-based approaches with regard to application fields and given tasks as comprehensively as possible. This paper is conducive to further understanding of skeleton-based application and dealing with particular issues.

</p>
</details>

<details><summary><b>Emergent Hand Morphology and Control from Optimizing Robust Grasps of Diverse Objects</b>
<a href="https://arxiv.org/abs/2012.12209">arxiv:2012.12209</a>
&#x1F4C8; 3 <br>
<p>Xinlei Pan, Animesh Garg, Animashree Anandkumar, Yuke Zhu</p></summary>
<p>

**Abstract:** Evolution in nature illustrates that the creatures' biological structure and their sensorimotor skills adapt to the environmental changes for survival. Likewise, the ability to morph and acquire new skills can facilitate an embodied agent to solve tasks of varying complexities. In this work, we introduce a data-driven approach where effective hand designs naturally emerge for the purpose of grasping diverse objects. Jointly optimizing morphology and control imposes computational challenges since it requires constant evaluation of a black-box function that measures the performance of a combination of embodiment and behavior. We develop a novel Bayesian Optimization algorithm that efficiently co-designs the morphology and grasping skills jointly through learned latent-space representations. We design the grasping tasks based on a taxonomy of three human grasp types: power grasp, pinch grasp, and lateral grasp. Through experimentation and comparative study, we demonstrate the effectiveness of our approach in discovering robust and cost-efficient hand morphologies for grasping novel objects.

</p>
</details>

<details><summary><b>Quantum Convolutional Neural Networks for High Energy Physics Data Analysis</b>
<a href="https://arxiv.org/abs/2012.12177">arxiv:2012.12177</a>
&#x1F4C8; 3 <br>
<p>Samuel Yen-Chi Chen, Tzu-Chieh Wei, Chao Zhang, Haiwang Yu, Shinjae Yoo</p></summary>
<p>

**Abstract:** This work presents a quantum convolutional neural network (QCNN) for the classification of high energy physics events. The proposed model is tested using a simulated dataset from the Deep Underground Neutrino Experiment. The proposed architecture demonstrates the quantum advantage of learning faster than the classical convolutional neural networks (CNNs) under a similar number of parameters. In addition to faster convergence, the QCNN achieves greater test accuracy compared to CNNs. Based on experimental results, it is a promising direction to study the application of QCNN and other quantum machine learning models in high energy physics and additional scientific fields.

</p>
</details>

<details><summary><b>Fundamental Limits on the Maximum Deviations in Control Systems: How Short Can Distribution Tails be Made by Feedback?</b>
<a href="https://arxiv.org/abs/2012.12174">arxiv:2012.12174</a>
&#x1F4C8; 3 <br>
<p>Song Fang, Quanyan Zhu</p></summary>
<p>

**Abstract:** In this paper, we adopt an information-theoretic approach to investigate the fundamental lower bounds on the maximum deviations in feedback control systems, where the plant is linear time-invariant while the controller can generically be any causal functions as long as it stabilizes the plant. It is seen in general that the lower bounds are characterized by the unstable poles (or nonminimum-phase zeros) of the plant as well as the level of randomness (as quantified by the conditional entropy) contained in the disturbance. Such bounds provide fundamental limits on how short the distribution tails in control systems can be made by feedback.

</p>
</details>

<details><summary><b>Image to Bengali Caption Generation Using Deep CNN and Bidirectional Gated Recurrent Unit</b>
<a href="https://arxiv.org/abs/2012.12139">arxiv:2012.12139</a>
&#x1F4C8; 3 <br>
<p>Al Momin Faruk, Hasan Al Faraby, Md. Muzahidul Azad, Md. Riduyan Fedous, Md. Kishor Morol</p></summary>
<p>

**Abstract:** There is very little notable research on generating descriptions of the Bengali language. About 243 million people speak in Bengali, and it is the 7th most spoken language on the planet. The purpose of this research is to propose a CNN and Bidirectional GRU based architecture model that generates natural language captions in the Bengali language from an image. Bengali people can use this research to break the language barrier and better understand each other's perspectives. It will also help many blind people with their everyday lives. This paper used an encoder-decoder approach to generate captions. We used a pre-trained Deep convolutional neural network (DCNN) called InceptonV3image embedding model as the encoder for analysis, classification, and annotation of the dataset's images Bidirectional Gated Recurrent unit (BGRU) layer as the decoder to generate captions. Argmax and Beam search is used to produce the highest possible quality of the captions. A new dataset called BNATURE is used, which comprises 8000 images with five captions per image. It is used for training and testing the proposed model. We obtained BLEU-1, BLEU-2, BLEU-3, BLEU-4 and Meteor is 42.6, 27.95, 23, 66, 16.41, 28.7 respectively.

</p>
</details>

<details><summary><b>Prediction of Chronic Kidney Disease Using Deep Neural Network</b>
<a href="https://arxiv.org/abs/2012.12089">arxiv:2012.12089</a>
&#x1F4C8; 3 <br>
<p>Iliyas Ibrahim Iliyas, Isah Rambo Saidu, Ali Baba Dauda, Suleiman Tasiu</p></summary>
<p>

**Abstract:** Deep neural Network (DNN) is becoming a focal point in Machine Learning research. Its application is penetrating into different fields and solving intricate and complex problems. DNN is now been applied in health image processing to detect various ailment such as cancer and diabetes. Another disease that is causing threat to our health is the kidney disease. This disease is becoming prevalent due to substances and elements we intake. Death is imminent and inevitable within few days without at least one functioning kidney. Ignoring the kidney malfunction can cause chronic kidney disease leading to death. Frequently, Chronic Kidney Disease (CKD) and its symptoms are mild and gradual, often go unnoticed for years only to be realized lately. Bade, a Local Government of Yobe state in Nigeria has been a center of attention by medical practitioners due to the prevalence of CKD. Unfortunately, a technical approach in culminating the disease is yet to be attained. We obtained a record of 400 patients with 10 attributes as our dataset from Bade General Hospital. We used DNN model to predict the absence or presence of CKD in the patients. The model produced an accuracy of 98%. Furthermore, we identified and highlighted the Features importance to provide the ranking of the features used in the prediction of the CKD. The outcome revealed that two attributes; Creatinine and Bicarbonate have the highest influence on the CKD prediction.

</p>
</details>

<details><summary><b>Finding Global Minima via Kernel Approximations</b>
<a href="https://arxiv.org/abs/2012.11978">arxiv:2012.11978</a>
&#x1F4C8; 3 <br>
<p>Alessandro Rudi, Ulysse Marteau-Ferey, Francis Bach</p></summary>
<p>

**Abstract:** We consider the global minimization of smooth functions based solely on function evaluations. Algorithms that achieve the optimal number of function evaluations for a given precision level typically rely on explicitly constructing an approximation of the function which is then minimized with algorithms that have exponential running-time complexity. In this paper, we consider an approach that jointly models the function to approximate and finds a global minimum. This is done by using infinite sums of square smooth functions and has strong links with polynomial sum-of-squares hierarchies. Leveraging recent representation properties of reproducing kernel Hilbert spaces, the infinite-dimensional optimization problem can be solved by subsampling in time polynomial in the number of function evaluations, and with theoretical guarantees on the obtained minimum.
  Given $n$ samples, the computational cost is $O(n^{3.5})$ in time, $O(n^2)$ in space, and we achieve a convergence rate to the global optimum that is $O(n^{-m/d + 1/2 + 3/d})$ where $m$ is the degree of differentiability of the function and $d$ the number of dimensions. The rate is nearly optimal in the case of Sobolev functions and more generally makes the proposed method particularly suitable for functions that have a large number of derivatives. Indeed, when $m$ is in the order of $d$, the convergence rate to the global optimum does not suffer from the curse of dimensionality, which affects only the worst-case constants (that we track explicitly through the paper).

</p>
</details>

<details><summary><b>Learning to Retrieve Entity-Aware Knowledge and Generate Responses with Copy Mechanism for Task-Oriented Dialogue Systems</b>
<a href="https://arxiv.org/abs/2012.11937">arxiv:2012.11937</a>
&#x1F4C8; 3 <br>
<p>Chao-Hong Tan, Xiaoyu Yang, Zi'ou Zheng, Tianda Li, Yufei Feng, Jia-Chen Gu, Quan Liu, Dan Liu, Zhen-Hua Ling, Xiaodan Zhu</p></summary>
<p>

**Abstract:** Task-oriented conversational modeling with unstructured knowledge access, as track 1 of the 9th Dialogue System Technology Challenges (DSTC 9), requests to build a system to generate response given dialogue history and knowledge access. This challenge can be separated into three subtasks, (1) knowledge-seeking turn detection, (2) knowledge selection, and (3) knowledge-grounded response generation. We use pre-trained language models, ELECTRA and RoBERTa, as our base encoder for different subtasks. For subtask 1 and 2, the coarse-grained information like domain and entity are used to enhance knowledge usage. For subtask 3, we use a latent variable to encode dialog history and selected knowledge better and generate responses combined with copy mechanism. Meanwhile, some useful post-processing strategies are performed on the model's final output to make further knowledge usage in the generation task. As shown in released evaluation results, our proposed system ranks second under objective metrics and ranks fourth under human metrics.

</p>
</details>

<details><summary><b>Interpreting Deep Learning Models for Epileptic Seizure Detection on EEG signals</b>
<a href="https://arxiv.org/abs/2012.11933">arxiv:2012.11933</a>
&#x1F4C8; 3 <br>
<p>Valentin Gabeff, Tomas Teijeiro, Marina Zapater, Leila Cammoun, Sylvain Rheims, Philippe Ryvlin, David Atienza</p></summary>
<p>

**Abstract:** While Deep Learning (DL) is often considered the state-of-the art for Artificial Intelligence-based medical decision support, it remains sparsely implemented in clinical practice and poorly trusted by clinicians due to insufficient interpretability of neural network models. We have tackled this issue by developing interpretable DL models in the context of online detection of epileptic seizure, based on EEG signal. This has conditioned the preparation of the input signals, the network architecture, and the post-processing of the output in line with the domain knowledge. Specifically, we focused the discussion on three main aspects: 1) how to aggregate the classification results on signal segments provided by the DL model into a larger time scale, at the seizure-level; 2) what are the relevant frequency patterns learned in the first convolutional layer of different models, and their relation with the delta, theta, alpha, beta and gamma frequency bands on which the visual interpretation of EEG is based; and 3) the identification of the signal waveforms with larger contribution towards the ictal class, according to the activation differences highlighted using the DeepLIFT method. Results show that the kernel size in the first layer determines the interpretability of the extracted features and the sensitivity of the trained models, even though the final performance is very similar after post-processing. Also, we found that amplitude is the main feature leading to an ictal prediction, suggesting that a larger patient population would be required to learn more complex frequency patterns. Still, our methodology was successfully able to generalize patient inter-variability for the majority of the studied population with a classification F1-score of 0.873 and detecting 90% of the seizures.

</p>
</details>

<details><summary><b>Graph Autoencoders with Deconvolutional Networks</b>
<a href="https://arxiv.org/abs/2012.11898">arxiv:2012.11898</a>
&#x1F4C8; 3 <br>
<p>Jia Li, Tomas Yu, Da-Cheng Juan, Arjun Gopalan, Hong Cheng, Andrew Tomkins</p></summary>
<p>

**Abstract:** Recent studies have indicated that Graph Convolutional Networks (GCNs) act as a \emph{low pass} filter in spectral domain and encode smoothed node representations. In this paper, we consider their opposite, namely Graph Deconvolutional Networks (GDNs) that reconstruct graph signals from smoothed node representations. We motivate the design of Graph Deconvolutional Networks via a combination of inverse filters in spectral domain and de-noising layers in wavelet domain, as the inverse operation results in a \emph{high pass} filter and may amplify the noise. Based on the proposed GDN, we further propose a graph autoencoder framework that first encodes smoothed graph representations with GCN and then decodes accurate graph signals with GDN. We demonstrate the effectiveness of the proposed method on several tasks including unsupervised graph-level representation , social recommendation and graph generation

</p>
</details>

<details><summary><b>Deep learning-based virtual refocusing of images using an engineered point-spread function</b>
<a href="https://arxiv.org/abs/2012.11892">arxiv:2012.11892</a>
&#x1F4C8; 3 <br>
<p>Xilin Yang, Luzhe Huang, Yilin Luo, Yichen Wu, Hongda Wang, Yair Rivenson, Aydogan Ozcan</p></summary>
<p>

**Abstract:** We present a virtual image refocusing method over an extended depth of field (DOF) enabled by cascaded neural networks and a double-helix point-spread function (DH-PSF). This network model, referred to as W-Net, is composed of two cascaded generator and discriminator network pairs. The first generator network learns to virtually refocus an input image onto a user-defined plane, while the second generator learns to perform a cross-modality image transformation, improving the lateral resolution of the output image. Using this W-Net model with DH-PSF engineering, we extend the DOF of a fluorescence microscope by ~20-fold. This approach can be applied to develop deep learning-enabled image reconstruction methods for localization microscopy techniques that utilize engineered PSFs to improve their imaging performance, including spatial resolution and volumetric imaging throughput.

</p>
</details>

<details><summary><b>Undivided Attention: Are Intermediate Layers Necessary for BERT?</b>
<a href="https://arxiv.org/abs/2012.11881">arxiv:2012.11881</a>
&#x1F4C8; 3 <br>
<p>Sharath Nittur Sridhar, Anthony Sarah</p></summary>
<p>

**Abstract:** In recent times, BERT-based models have been extremely successful in solving a variety of natural language processing (NLP) tasks such as reading comprehension, natural language inference, sentiment analysis, etc. All BERT-based architectures have a self-attention block followed by a block of intermediate layers as the basic building component. However, a strong justification for the inclusion of these intermediate layers remains missing in the literature. In this work we investigate the importance of intermediate layers on the overall network performance of downstream tasks. We show that reducing the number of intermediate layers and modifying the architecture for BERT-Base results in minimal loss in fine-tuning accuracy for downstream tasks while decreasing the number of parameters and training time of the model. Additionally, we use the central kernel alignment (CKA) similarity metric and probing classifiers to demonstrate that removing intermediate layers has little impact on the learned self-attention representations.

</p>
</details>

<details><summary><b>Objective Evaluation of Deep Uncertainty Predictions for COVID-19 Detection</b>
<a href="https://arxiv.org/abs/2012.11840">arxiv:2012.11840</a>
&#x1F4C8; 3 <br>
<p>Hamzeh Asgharnezhad, Afshar Shamsi, Roohallah Alizadehsani, Abbas Khosravi, Saeid Nahavandi, Zahra Alizadeh Sani, Dipti Srinivasan</p></summary>
<p>

**Abstract:** Deep neural networks (DNNs) have been widely applied for detecting COVID-19 in medical images. Existing studies mainly apply transfer learning and other data representation strategies to generate accurate point estimates. The generalization power of these networks is always questionable due to being developed using small datasets and failing to report their predictive confidence. Quantifying uncertainties associated with DNN predictions is a prerequisite for their trusted deployment in medical settings. Here we apply and evaluate three uncertainty quantification techniques for COVID-19 detection using chest X-Ray (CXR) images. The novel concept of uncertainty confusion matrix is proposed and new performance metrics for the objective evaluation of uncertainty estimates are introduced. Through comprehensive experiments, it is shown that networks pertained on CXR images outperform networks pretrained on natural image datasets such as ImageNet. Qualitatively and quantitatively evaluations also reveal that the predictive uncertainty estimates are statistically higher for erroneous predictions than correct predictions. Accordingly, uncertainty quantification methods are capable of flagging risky predictions with high uncertainty estimates. We also observe that ensemble methods more reliably capture uncertainties during the inference.

</p>
</details>

<details><summary><b>Refined bounds for randomized experimental design</b>
<a href="https://arxiv.org/abs/2012.15726">arxiv:2012.15726</a>
&#x1F4C8; 2 <br>
<p>Geovani Rizk, Igor Colin, Albert Thomas, Moez Draief</p></summary>
<p>

**Abstract:** Experimental design is an approach for selecting samples among a given set so as to obtain the best estimator for a given criterion. In the context of linear regression, several optimal designs have been derived, each associated with a different criterion: mean square error, robustness, \emph{etc}. Computing such designs is generally an NP-hard problem and one can instead rely on a convex relaxation that considers probability distributions over the samples. Although greedy strategies and rounding procedures have received a lot of attention, straightforward sampling from the optimal distribution has hardly been investigated. In this paper, we propose theoretical guarantees for randomized strategies on E and G-optimal design. To this end, we develop a new concentration inequality for the eigenvalues of random matrices using a refined version of the intrinsic dimension that enables us to quantify the performance of such randomized strategies. Finally, we evidence the validity of our analysis through experiments, with particular attention on the G-optimal design applied to the best arm identification problem for linear bandits.

</p>
</details>

<details><summary><b>Confronting Abusive Language Online: A Survey from the Ethical and Human Rights Perspective</b>
<a href="https://arxiv.org/abs/2012.12305">arxiv:2012.12305</a>
&#x1F4C8; 2 <br>
<p>Svetlana Kiritchenko, Isar Nejadgholi, Kathleen C. Fraser</p></summary>
<p>

**Abstract:** The pervasiveness of abusive content on the internet can lead to severe psychological and physical harm. Significant effort in Natural Language Processing (NLP) research has been devoted to addressing this problem through abusive content detection and related sub-areas, such as the detection of hate speech, toxicity, cyberbullying, etc. Although current technologies achieve high classification performance in research studies, it has been observed that the real-life application of this technology can cause unintended harms, such as the silencing of under-represented groups. We review a large body of NLP research on automatic abuse detection with a new focus on ethical challenges, organized around eight established ethical principles: privacy, accountability, safety and security, transparency and explainability, fairness and non-discrimination, human control of technology, professional responsibility, and promotion of human values. In many cases, these principles relate not only to situational ethical codes, which may be context-dependent, but are in fact connected to universal human rights, such as the right to privacy, freedom from discrimination, and freedom of expression. We highlight the need to examine the broad social impacts of this technology, and to bring ethical and human rights considerations to every stage of the application life-cycle, from task formulation and dataset design, to model training and evaluation, to application deployment. Guided by these principles, we identify several opportunities for rights-respecting, socio-technical solutions to detect and confront online abuse, including 'nudging', 'quarantining', value sensitive design, counter-narratives, style transfer, and AI-driven public education applications.

</p>
</details>

<details><summary><b>C-Watcher: A Framework for Early Detection of High-Risk Neighborhoods Ahead of COVID-19 Outbreak</b>
<a href="https://arxiv.org/abs/2012.12169">arxiv:2012.12169</a>
&#x1F4C8; 2 <br>
<p>Congxi Xiao, Jingbo Zhou, Jizhou Huang, An Zhuo, Ji Liu, Haoyi Xiong, Dejing Dou</p></summary>
<p>

**Abstract:** The novel coronavirus disease (COVID-19) has crushed daily routines and is still rampaging through the world. Existing solution for nonpharmaceutical interventions usually needs to timely and precisely select a subset of residential urban areas for containment or even quarantine, where the spatial distribution of confirmed cases has been considered as a key criterion for the subset selection. While such containment measure has successfully stopped or slowed down the spread of COVID-19 in some countries, it is criticized for being inefficient or ineffective, as the statistics of confirmed cases are usually time-delayed and coarse-grained. To tackle the issues, we propose C-Watcher, a novel data-driven framework that aims at screening every neighborhood in a target city and predicting infection risks, prior to the spread of COVID-19 from epicenters to the city. In terms of design, C-Watcher collects large-scale long-term human mobility data from Baidu Maps, then characterizes every residential neighborhood in the city using a set of features based on urban mobility patterns. Furthermore, to transfer the firsthand knowledge (witted in epicenters) to the target city before local outbreaks, we adopt a novel adversarial encoder framework to learn "city-invariant" representations from the mobility-related features for precise early detection of high-risk neighborhoods, even before any confirmed cases known, in the target city. We carried out extensive experiments on C-Watcher using the real-data records in the early stage of COVID-19 outbreaks, where the results demonstrate the efficiency and effectiveness of C-Watcher for early detection of high-risk neighborhoods from a large number of cities.

</p>
</details>

<details><summary><b>High-Speed Robot Navigation using Predicted Occupancy Maps</b>
<a href="https://arxiv.org/abs/2012.12142">arxiv:2012.12142</a>
&#x1F4C8; 2 <br>
<p>Kapil D. Katyal, Adam Polevoy, Joseph Moore, Craig Knuth, Katie M. Popek</p></summary>
<p>

**Abstract:** Safe and high-speed navigation is a key enabling capability for real world deployment of robotic systems. A significant limitation of existing approaches is the computational bottleneck associated with explicit mapping and the limited field of view (FOV) of existing sensor technologies. In this paper, we study algorithmic approaches that allow the robot to predict spaces extending beyond the sensor horizon for robust planning at high speeds. We accomplish this using a generative neural network trained from real-world data without requiring human annotated labels. Further, we extend our existing control algorithms to support leveraging the predicted spaces to improve collision-free planning and navigation at high speeds. Our experiments are conducted on a physical robot based on the MIT race car using an RGBD sensor where were able to demonstrate improved performance at 4 m/s compared to a controller not operating on predicted regions of the map.

</p>
</details>

<details><summary><b>Domain Adaptation of NMT models for English-Hindi Machine Translation Task at AdapMT ICON 2020</b>
<a href="https://arxiv.org/abs/2012.12112">arxiv:2012.12112</a>
&#x1F4C8; 2 <br>
<p>Ramchandra Joshi, Rushabh Karnavat, Kaustubh Jirapure, Raviraj Joshi</p></summary>
<p>

**Abstract:** Recent advancements in Neural Machine Translation (NMT) models have proved to produce a state of the art results on machine translation for low resource Indian languages. This paper describes the neural machine translation systems for the English-Hindi language presented in AdapMT Shared Task ICON 2020. The shared task aims to build a translation system for Indian languages in specific domains like Artificial Intelligence (AI) and Chemistry using a small in-domain parallel corpus. We evaluated the effectiveness of two popular NMT models i.e, LSTM, and Transformer architectures for the English-Hindi machine translation task based on BLEU scores. We train these models primarily using the out of domain data and employ simple domain adaptation techniques based on the characteristics of the in-domain dataset. The fine-tuning and mixed-domain data approaches are used for domain adaptation. Our team was ranked first in the chemistry and general domain En-Hi translation task and second in the AI domain En-Hi translation task.

</p>
</details>

<details><summary><b>QVMix and QVMix-Max: Extending the Deep Quality-Value Family of Algorithms to Cooperative Multi-Agent Reinforcement Learning</b>
<a href="https://arxiv.org/abs/2012.12062">arxiv:2012.12062</a>
&#x1F4C8; 2 <br>
<p>Pascal Leroy, Damien Ernst, Pierre Geurts, Gilles Louppe, Jonathan Pisane, Matthia Sabatelli</p></summary>
<p>

**Abstract:** This paper introduces four new algorithms that can be used for tackling multi-agent reinforcement learning (MARL) problems occurring in cooperative settings. All algorithms are based on the Deep Quality-Value (DQV) family of algorithms, a set of techniques that have proven to be successful when dealing with single-agent reinforcement learning problems (SARL). The key idea of DQV algorithms is to jointly learn an approximation of the state-value function $V$, alongside an approximation of the state-action value function $Q$. We follow this principle and generalise these algorithms by introducing two fully decentralised MARL algorithms (IQV and IQV-Max) and two algorithms that are based on the centralised training with decentralised execution training paradigm (QVMix and QVMix-Max). We compare our algorithms with state-of-the-art MARL techniques on the popular StarCraft Multi-Agent Challenge (SMAC) environment. We show competitive results when QVMix and QVMix-Max are compared to well-known MARL techniques such as QMIX and MAVEN and show that QVMix can even outperform them on some of the tested environments, being the algorithm which performs best overall. We hypothesise that this is due to the fact that QVMix suffers less from the overestimation bias of the $Q$ function.

</p>
</details>

<details><summary><b>Information Leakage Games: Exploring Information as a Utility Function</b>
<a href="https://arxiv.org/abs/2012.12060">arxiv:2012.12060</a>
&#x1F4C8; 2 <br>
<p>Mário S. Alvim, Konstantinos Chatzikokolakis, Yusuke Kawamoto, Catuscia Palamidessi</p></summary>
<p>

**Abstract:** A common goal in the areas of secure information flow and privacy is to build effective defenses against unwanted leakage of information. To this end, one must be able to reason about potential attacks and their interplay with possible defenses. In this paper we propose a game-theoretic framework to formalize strategies of attacker and defender in the context of information leakage, and provide a basis for developing optimal defense methods. A crucial novelty of our games is that their utility is given by information leakage, which in some cases may behave in a non-linear way. This causes a significant deviation from classic game theory, in which utility functions are linear with respect to players' strategies. Hence, a key contribution of this paper is the establishment of the foundations of information leakage games. We consider two main categories of games, depending on the particular notion of information leakage being captured. The first category, which we call QIF-games, is tailored for the theory of quantitative information flow (QIF). The second one, which we call DP-games, corresponds to differential privacy (DP).

</p>
</details>

<details><summary><b>Do We Really Need Scene-specific Pose Encoders?</b>
<a href="https://arxiv.org/abs/2012.12014">arxiv:2012.12014</a>
&#x1F4C8; 2 <br>
<p>Yoli Shavit, Ron Ferens</p></summary>
<p>

**Abstract:** Visual pose regression models estimate the camera pose from a query image with a single forward pass. Current models learn pose encoding from an image using deep convolutional networks which are trained per scene. The resulting encoding is typically passed to a multi-layer perceptron in order to regress the pose. In this work, we propose that scene-specific pose encoders are not required for pose regression and that encodings trained for visual similarity can be used instead. In order to test our hypothesis, we take a shallow architecture of several fully connected layers and train it with pre-computed encodings from a generic image retrieval model. We find that these encodings are not only sufficient to regress the camera pose, but that, when provided to a branching fully connected architecture, a trained model can achieve competitive results and even surpass current \textit{state-of-the-art} pose regressors in some cases. Moreover, we show that for outdoor localization, the proposed architecture is the only pose regressor, to date, consistently localizing in under 2 meters and 5 degrees.

</p>
</details>

<details><summary><b>Uncertainty and Surprisal Jointly Deliver the Punchline: Exploiting Incongruity-Based Features for Humor Recognition</b>
<a href="https://arxiv.org/abs/2012.12007">arxiv:2012.12007</a>
&#x1F4C8; 2 <br>
<p>Yubo Xie, Junze Li, Pearl Pu</p></summary>
<p>

**Abstract:** Humor recognition has been widely studied as a text classification problem using data-driven approaches. However, most existing work does not examine the actual joke mechanism to understand humor. We break down any joke into two distinct components: the set-up and the punchline, and further explore the special relationship between them. Inspired by the incongruity theory of humor, we model the set-up as the part developing semantic uncertainty, and the punchline disrupting audience expectations. With increasingly powerful language models, we were able to feed the set-up along with the punchline into the GPT-2 language model, and calculate the uncertainty and surprisal values of the jokes. By conducting experiments on the SemEval 2021 Task 7 dataset, we found that these two features have better capabilities of telling jokes from non-jokes, compared with existing baselines.

</p>
</details>

<details><summary><b>g2tmn at Constraint@AAAI2021: Exploiting CT-BERT and Ensembling Learning for COVID-19 Fake News Detection</b>
<a href="https://arxiv.org/abs/2012.11967">arxiv:2012.11967</a>
&#x1F4C8; 2 <br>
<p>Anna Glazkova, Maksim Glazkov, Timofey Trifonov</p></summary>
<p>

**Abstract:** The COVID-19 pandemic has had a huge impact on various areas of human life. Hence, the coronavirus pandemic and its consequences are being actively discussed on social media. However, not all social media posts are truthful. Many of them spread fake news that cause panic among readers, misinform people and thus exacerbate the effect of the pandemic. In this paper, we present our results at the Constraint@AAAI2021 Shared Task: COVID-19 Fake News Detection in English. In particular, we propose our approach using the transformer-based ensemble of COVID-Twitter-BERT (CT-BERT) models. We describe the models used, the ways of text preprocessing and adding extra data. As a result, our best model achieved the weighted F1-score of 98.69 on the test set (the first place in the leaderboard) of this shared task that attracted 166 submitted teams in total.

</p>
</details>

<details><summary><b>A Hierarchical Reasoning Graph Neural Network for The Automatic Scoring of Answer Transcriptions in Video Job Interviews</b>
<a href="https://arxiv.org/abs/2012.11960">arxiv:2012.11960</a>
&#x1F4C8; 2 <br>
<p>Kai Chen, Meng Niu, Qingcai Chen</p></summary>
<p>

**Abstract:** We address the task of automatically scoring the competency of candidates based on textual features, from the automatic speech recognition (ASR) transcriptions in the asynchronous video job interview (AVI). The key challenge is how to construct the dependency relation between questions and answers, and conduct the semantic level interaction for each question-answer (QA) pair. However, most of the recent studies in AVI focus on how to represent questions and answers better, but ignore the dependency information and interaction between them, which is critical for QA evaluation. In this work, we propose a Hierarchical Reasoning Graph Neural Network (HRGNN) for the automatic assessment of question-answer pairs. Specifically, we construct a sentence-level relational graph neural network to capture the dependency information of sentences in or between the question and the answer. Based on these graphs, we employ a semantic-level reasoning graph attention network to model the interaction states of the current QA session. Finally, we propose a gated recurrent unit encoder to represent the temporal question-answer pairs for the final prediction. Empirical results conducted on CHNAT (a real-world dataset) validate that our proposed model significantly outperforms text-matching based benchmark models. Ablation studies and experimental results with 10 random seeds also show the effectiveness and stability of our models.

</p>
</details>

<details><summary><b>A Feasibility study for Deep learning based automated brain tumor segmentation using Magnetic Resonance Images</b>
<a href="https://arxiv.org/abs/2012.11952">arxiv:2012.11952</a>
&#x1F4C8; 2 <br>
<p>Shanaka Ramesh Gunasekara, HNTK Kaldera, Maheshi B. Dissanayake</p></summary>
<p>

**Abstract:** Deep learning algorithms have accounted for the rapid acceleration of research in artificial intelligence in medical image analysis, interpretation, and segmentation with many potential applications across various sub disciplines in medicine. However, only limited number of research which investigates these application scenarios, are deployed into the clinical sector for the evaluation of the real requirement and the practical challenges of the model deployment. In this research, a deep convolutional neural network (CNN) based classification network and Faster RCNN based localization network were developed for brain tumor MR image classification and tumor localization. A typical edge detection algorithm called Prewitt was used for tumor segmentation task, based on the output of the tumor localization. Overall performance of the proposed tumor segmentation architecture, was analyzed using objective quality parameters including Accuracy, Boundary Displacement Error (BDE), Dice score and confidence interval. A subjective quality assessment of the model was conducted based on the Double Stimulus Impairment Scale (DSIS) protocol using the input of medical expertise. It was observed that the confidence level of our segmented output was in a similar range to that of experts. Also, the Neurologists have rated the output of our model as highly accurate segmentation.

</p>
</details>

<details><summary><b>Intelligent Resource Allocation in Dense LoRa Networks using Deep Reinforcement Learning</b>
<a href="https://arxiv.org/abs/2012.11867">arxiv:2012.11867</a>
&#x1F4C8; 2 <br>
<p>Inaam Ilahi, Muhammad Usama, Muhammad Omer Farooq, Muhammad Umar Janjua, Junaid Qadir</p></summary>
<p>

**Abstract:** The anticipated increase in the count of IoT devices in the coming years motivates the development of efficient algorithms that can help in their effective management while keeping the power consumption low. In this paper, we propose LoRaDRL and provide a detailed performance evaluation. We propose a multi-channel scheme for LoRaDRL. We perform extensive experiments, and our results demonstrate that the proposed algorithm not only significantly improves long-range wide area network (LoRaWAN)'s packet delivery ratio (PDR) but is also able to support mobile end-devices (EDs) while ensuring lower power consumption. Most previous works focus on proposing different MAC protocols for improving the network capacity. We show that through the use of LoRaDRL, we can achieve the same efficiency with ALOHA while moving the complexity from EDs to the gateway thus making the EDs simpler and cheaper. Furthermore, we test the performance of LoRaDRL under large-scale frequency jamming attacks and show its adaptiveness to the changes in the environment. We show that LoRaDRL's output improves the performance of state-of-the-art techniques resulting in some cases an improvement of more than 500% in terms of PDR compared to learning-based techniques.

</p>
</details>

<details><summary><b>Efficient and Visualizable Convolutional Neural Networks for COVID-19 Classification Using Chest CT</b>
<a href="https://arxiv.org/abs/2012.11860">arxiv:2012.11860</a>
&#x1F4C8; 2 <br>
<p>Aksh Garg, Sana Salehi, Marianna La Rocca, Rachael Garner, Dominique Duncan</p></summary>
<p>

**Abstract:** The novel 2019 coronavirus disease (COVID-19) has infected over 65 million people worldwide as of December 4, 2020, pushing the world to the brink of social and economic collapse. With cases rising rapidly, deep learning has emerged as a promising diagnosis technique. However, identifying the most accurate models to characterize COVID-19 patients is challenging because comparing results obtained with different types of data and acquisition processes is non-trivial. In this paper, we evaluated and compared 40 different convolutional neural network architectures for COVID-19 diagnosis, serving as the first to consider the EfficientNet family for COVID-19 diagnosis. EfficientNet-B5 is identified as the best model with an accuracy of 0.9931+/-0.0021, F1 score of 0.9931+/-0.0020, sensitivity of 0.9952+/-0.0020, and specificity of 0.9912+/-0.0048. Intermediate activation maps and Gradient-weighted Class Activation Mappings offer human-interpretable evidence of the model's perception of ground-class opacities and consolidations, hinting towards a promising use-case of artificial intelligence-assisted radiology tools.

</p>
</details>

<details><summary><b>A Second-Order Approach to Learning with Instance-Dependent Label Noise</b>
<a href="https://arxiv.org/abs/2012.11854">arxiv:2012.11854</a>
&#x1F4C8; 2 <br>
<p>Zhaowei Zhu, Tongliang Liu, Yang Liu</p></summary>
<p>

**Abstract:** The presence of label noise often misleads the training of deep neural networks. Departing from the recent literature which largely assumes the label noise rate is only determined by the true class, the errors in human-annotated labels are more likely to be dependent on the difficulty levels of tasks, resulting in settings with instance-dependent label noise. We show theoretically that the heterogeneous instance-dependent label noise is effectively down-weighting the examples with higher noise rates in a non-uniform way and thus causes imbalances, rendering the strategy of directly applying methods for class-dependent label noise questionable. In this paper, we propose and study the potentials of a second-order approach that leverages the estimation of several covariance terms defined between the instance-dependent noise rates and the Bayes optimal label. We show that this set of second-order information successfully captures the induced imbalances. We further proceed to show that with the help of the estimated second-order information, we identify a new loss function whose expected risk of a classifier under instance-dependent label noise can be shown to be equivalent to a new problem with only class-dependent label noise. This fact allows us to develop effective loss functions to correctly evaluate models. We provide an efficient procedure to perform the estimations without accessing either ground truth labels or prior knowledge of the noise rates. Experiments on CIFAR10 and CIFAR100 with synthetic instance-dependent label noise and Clothing1M with real-world human label noise verify our approach.

</p>
</details>

<details><summary><b>Adversarial Multiscale Feature Learning for Overlapping Chromosome Segmentation</b>
<a href="https://arxiv.org/abs/2012.11847">arxiv:2012.11847</a>
&#x1F4C8; 2 <br>
<p>Liye Mei, Yalan Yu, Yueyun Weng, Xiaopeng Guo, Yan Liu, Du Wang, Sheng Liu, Fuling Zhou, Cheng Lei</p></summary>
<p>

**Abstract:** Chromosome karyotype analysis is of great clinical importance in the diagnosis and treatment of diseases, especially for genetic diseases. Since manual analysis is highly time and effort consuming, computer-assisted automatic chromosome karyotype analysis based on images is routinely used to improve the efficiency and accuracy of the analysis. Due to the strip shape of the chromosomes, they easily get overlapped with each other when imaged, significantly affecting the accuracy of the analysis afterward. Conventional overlapping chromosome segmentation methods are usually based on manually tagged features, hence, the performance of which is easily affected by the quality, such as resolution and brightness, of the images. To address the problem, in this paper, we present an adversarial multiscale feature learning framework to improve the accuracy and adaptability of overlapping chromosome segmentation. Specifically, we first adopt the nested U-shape network with dense skip connections as the generator to explore the optimal representation of the chromosome images by exploiting multiscale features. Then we use the conditional generative adversarial network (cGAN) to generate images similar to the original ones, the training stability of which is enhanced by applying the least-square GAN objective. Finally, we employ Lovasz-Softmax to help the model converge in a continuous optimization setting. Comparing with the established algorithms, the performance of our framework is proven superior by using public datasets in eight evaluation criteria, showing its great potential in overlapping chromosome segmentation

</p>
</details>

<details><summary><b>Uncertainty Bounds for Multivariate Machine Learning Predictions on High-Strain Brittle Fracture</b>
<a href="https://arxiv.org/abs/2012.15739">arxiv:2012.15739</a>
&#x1F4C8; 1 <br>
<p>Cristina Garcia-Cardona, M. Giselle Fernández-Godino, Daniel O'Malley, Tanmoy Bhattacharya</p></summary>
<p>

**Abstract:** Simulation of the crack network evolution on high strain rate impact experiments performed in brittle materials is very compute-intensive. The cost increases even more if multiple simulations are needed to account for the randomness in crack length, location, and orientation, which is inherently found in real-world materials. Constructing a machine learning emulator can make the process faster by orders of magnitude. There has been little work, however, on assessing the error associated with their predictions. Estimating these errors is imperative for meaningful overall uncertainty quantification. In this work, we extend the heteroscedastic uncertainty estimates to bound a multiple output machine learning emulator. We find that the response prediction is robust with a somewhat conservative estimate of uncertainty.

</p>
</details>

<details><summary><b>Hardware-accelerated Simulation-based Inference of Stochastic Epidemiology Models for COVID-19</b>
<a href="https://arxiv.org/abs/2012.14332">arxiv:2012.14332</a>
&#x1F4C8; 1 <br>
<p>Sourabh Kulkarni, Mario Michael Krell, Seth Nabarro, Csaba Andras Moritz</p></summary>
<p>

**Abstract:** Epidemiology models are central in understanding and controlling large scale pandemics. Several epidemiology models require simulation-based inference such as Approximate Bayesian Computation (ABC) to fit their parameters to observations. ABC inference is highly amenable to efficient hardware acceleration. In this work, we develop parallel ABC inference of a stochastic epidemiology model for COVID-19. The statistical inference framework is implemented and compared on Intel Xeon CPU, NVIDIA Tesla V100 GPU and the Graphcore Mk1 IPU, and the results are discussed in the context of their computational architectures. Results show that GPUs are 4x and IPUs are 30x faster than Xeon CPUs. Extensive performance analysis indicates that the difference between IPU and GPU can be attributed to higher communication bandwidth, closeness of memory to compute, and higher compute power in the IPU. The proposed framework scales across 16 IPUs, with scaling overhead not exceeding 8% for the experiments performed. We present an example of our framework in practice, performing inference on the epidemiology model across three countries, and giving a brief overview of the results.

</p>
</details>

<details><summary><b>Function Design for Improved Competitive Ratio in Online Resource Allocation with Procurement Costs</b>
<a href="https://arxiv.org/abs/2012.12457">arxiv:2012.12457</a>
&#x1F4C8; 1 <br>
<p>Mitas Ray, Omid Sadeghi, Lillian J. Ratliff, Maryam Fazel</p></summary>
<p>

**Abstract:** We study the problem of online resource allocation, where multiple customers arrive sequentially and the seller must irrevocably allocate resources to each incoming customer while also facing a procurement cost for the total allocation. Assuming resource procurement follows an a priori known marginally increasing cost function, the objective is to maximize the reward obtained from fulfilling the customers' requests sans the cumulative procurement cost. We analyze the competitive ratio of a primal-dual algorithm in this setting, and develop an optimization framework for synthesizing a surrogate function for the procurement cost function to be used by the algorithm, in order to improve the competitive ratio of the primal-dual algorithm. Our first design method focuses on polynomial procurement cost functions and uses the optimal surrogate function to provide a more refined bound than the state of the art. Our second design method uses quasiconvex optimization to find optimal design parameters for a general class of procurement cost functions. Numerical examples are used to illustrate the design techniques. We conclude by extending the analysis to devise a posted pricing mechanism in which the algorithm does not require the customers' preferences to be revealed.

</p>
</details>

<details><summary><b>Stochastic Gradient Variance Reduction by Solving a Filtering Problem</b>
<a href="https://arxiv.org/abs/2012.12418">arxiv:2012.12418</a>
&#x1F4C8; 1 <br>
<p>Xingyi Yang</p></summary>
<p>

**Abstract:** Deep neural networks (DNN) are typically optimized using stochastic gradient descent (SGD). However, the estimation of the gradient using stochastic samples tends to be noisy and unreliable, resulting in large gradient variance and bad convergence. In this paper, we propose \textbf{Filter Gradient Decent}~(FGD), an efficient stochastic optimization algorithm that makes the consistent estimation of the local gradient by solving an adaptive filtering problem with different design of filters. Our method reduces variance in stochastic gradient descent by incorporating the historical states to enhance the current estimation. It is able to correct noisy gradient direction as well as to accelerate the convergence of learning. We demonstrate the effectiveness of the proposed Filter Gradient Descent on numerical optimization and training neural networks, where it achieves superior and robust performance compared with traditional momentum-based methods. To the best of our knowledge, we are the first to provide a practical solution that integrates filtering into gradient estimation by making the analogy between gradient estimation and filtering problems in signal processing. (The code is provided in https://github.com/Adamdad/Filter-Gradient-Decent)

</p>
</details>

<details><summary><b>Towards Histopathological Stain Invariance by Unsupervised Domain Augmentation using Generative Adversarial Networks</b>
<a href="https://arxiv.org/abs/2012.12413">arxiv:2012.12413</a>
&#x1F4C8; 1 <br>
<p>Jelica Vasiljević, Friedrich Feuerhake, Cédric Wemmert, Thomas Lampert</p></summary>
<p>

**Abstract:** The application of supervised deep learning methods in digital pathology is limited due to their sensitivity to domain shift. Digital Pathology is an area prone to high variability due to many sources, including the common practice of evaluating several consecutive tissue sections stained with different staining protocols. Obtaining labels for each stain is very expensive and time consuming as it requires a high level of domain knowledge. In this article, we propose an unsupervised augmentation approach based on adversarial image-to-image translation, which facilitates the training of stain invariant supervised convolutional neural networks. By training the network on one commonly used staining modality and applying it to images that include corresponding, but differently stained, tissue structures, the presented method demonstrates significant improvements over other approaches. These benefits are illustrated in the problem of glomeruli segmentation in seven different staining modalities (PAS, Jones H&E, CD68, Sirius Red, CD34, H&E and CD3) and analysis of the learned representations demonstrate their stain invariance.

</p>
</details>

<details><summary><b>Scalable Optical Learning Operator</b>
<a href="https://arxiv.org/abs/2012.12404">arxiv:2012.12404</a>
&#x1F4C8; 1 <br>
<p>Uğur Teğin, Mustafa Yıldırım, İlker Oğuz, Christophe Moser, Demetri Psaltis</p></summary>
<p>

**Abstract:** Today's heavy machine learning tasks are fueled by large datasets. Computing is performed with power hungry processors whose performance is ultimately limited by the data transfer to and from memory. Optics is one of the powerful means of communicating and processing information and there is intense current interest in optical information processing for realizing high-speed computations. Here we present and experimentally demonstrate an optical computing framework based on spatiotemporal effects in multimode fibers for a range of learning tasks from classifying COVID-19 X-ray lung images and speech recognition to predicting age from face images. The presented framework overcomes the energy scaling problem of existing systems without compromising speed. We leveraged simultaneous, linear, and nonlinear interaction of spatial modes as a computation engine. We numerically and experimentally showed the ability of the method to execute several different tasks with accuracy comparable to a digital implementation. Our results indicate that a powerful supercomputer would be required to duplicate the performance of the multimode fiber-based computer.

</p>
</details>

<details><summary><b>Fractal Dimension Generalization Measure</b>
<a href="https://arxiv.org/abs/2012.12384">arxiv:2012.12384</a>
&#x1F4C8; 1 <br>
<p>Valeri Alexiev</p></summary>
<p>

**Abstract:** Developing a robust generalization measure for the performance of machine learning models is an important and challenging task. A lot of recent research in the area focuses on the model decision boundary when predicting generalization. In this paper, as part of the "Predicting Generalization in Deep Learning" competition, we analyse the complexity of decision boundaries using the concept of fractal dimension and develop a generalization measure based on that technique.

</p>
</details>

<details><summary><b>Distributed Q-Learning with State Tracking for Multi-agent Networked Control</b>
<a href="https://arxiv.org/abs/2012.12383">arxiv:2012.12383</a>
&#x1F4C8; 1 <br>
<p>Hang Wang, Sen Lin, Hamid Jafarkhani, Junshan Zhang</p></summary>
<p>

**Abstract:** This paper studies distributed Q-learning for Linear Quadratic Regulator (LQR) in a multi-agent network. The existing results often assume that agents can observe the global system state, which may be infeasible in large-scale systems due to privacy concerns or communication constraints. In this work, we consider a setting with unknown system models and no centralized coordinator. We devise a state tracking (ST) based Q-learning algorithm to design optimal controllers for agents. Specifically, we assume that agents maintain local estimates of the global state based on their local information and communications with neighbors. At each step, every agent updates its local global state estimation, based on which it solves an approximate Q-factor locally through policy iteration. Assuming decaying injected excitation noise during the policy evaluation, we prove that the local estimation converges to the true global state, and establish the convergence of the proposed distributed ST-based Q-learning algorithm. The experimental studies corroborate our theoretical results by showing that our proposed method achieves comparable performance with the centralized case.

</p>
</details>

<details><summary><b>Generative Interventions for Causal Learning</b>
<a href="https://arxiv.org/abs/2012.12265">arxiv:2012.12265</a>
&#x1F4C8; 1 <br>
<p>Chengzhi Mao, Amogh Gupta, Augustine Cha, Hao Wang, Junfeng Yang, Carl Vondrick</p></summary>
<p>

**Abstract:** We introduce a framework for learning robust visual representations that generalize to new viewpoints, backgrounds, and scene contexts. Discriminative models often learn naturally occurring spurious correlations, which cause them to fail on images outside of the training distribution. In this paper, we show that we can steer generative models to manufacture interventions on features caused by confounding factors. Experiments, visualizations, and theoretical results show this method learns robust representations more consistent with the underlying causal relationships. Our approach improves performance on multiple datasets demanding out-of-distribution generalization, and we demonstrate state-of-the-art performance generalizing from ImageNet to ObjectNet dataset.

</p>
</details>

<details><summary><b>Autonomous Charging of Electric Vehicle Fleets to Enhance Renewable Generation Dispatchability</b>
<a href="https://arxiv.org/abs/2012.12257">arxiv:2012.12257</a>
&#x1F4C8; 1 <br>
<p>Reza Bayani, Saeed D. Manshadi, Guangyi Liu, Yawei Wang, Renchang Dai</p></summary>
<p>

**Abstract:** A total 19% of generation capacity in California is offered by PV units and over some months, more than 10% of this energy is curtailed. In this research, a novel approach to reduce renewable generation curtailments and increasing system flexibility by means of electric vehicles' charging coordination is represented. The presented problem is a sequential decision making process, and is solved by fitted Q-iteration algorithm which unlike other reinforcement learning methods, needs fewer episodes of learning. Three case studies are presented to validate the effectiveness of the proposed approach. These cases include aggregator load following, ramp service and utilization of non-deterministic PV generation. The results suggest that through this framework, EVs successfully learn how to adjust their charging schedule in stochastic scenarios where their trip times, as well as solar power generation are unknown beforehand.

</p>
</details>

<details><summary><b>Improving Sample and Feature Selection with Principal Covariates Regression</b>
<a href="https://arxiv.org/abs/2012.12253">arxiv:2012.12253</a>
&#x1F4C8; 1 <br>
<p>Rose K. Cersonsky, Benjamin A. Helfrecht, Edgar A. Engel, Michele Ceriotti</p></summary>
<p>

**Abstract:** Selecting the most relevant features and samples out of a large set of candidates is a task that occurs very often in the context of automated data analysis, where it can be used to improve the computational performance, and also often the transferability, of a model. Here we focus on two popular sub-selection schemes which have been applied to this end: CUR decomposition, that is based on a low-rank approximation of the feature matrix and Farthest Point Sampling, that relies on the iterative identification of the most diverse samples and discriminating features. We modify these unsupervised approaches, incorporating a supervised component following the same spirit as the Principal Covariates Regression (PCovR) method. We show that incorporating target information provides selections that perform better in supervised tasks, which we demonstrate with ridge regression, kernel ridge regression, and sparse kernel regression. We also show that incorporating aspects of simple supervised learning models can improve the accuracy of more complex models, such as feed-forward neural networks. We present adjustments to minimize the impact that any subselection may incur when performing unsupervised tasks. We demonstrate the significant improvements associated with the use of PCov-CUR and PCov-FPS selections for applications to chemistry and materials science, typically reducing by a factor of two the number of features and samples which are required to achieve a given level of regression accuracy.

</p>
</details>

<details><summary><b>Learning to Initialize Gradient Descent Using Gradient Descent</b>
<a href="https://arxiv.org/abs/2012.12141">arxiv:2012.12141</a>
&#x1F4C8; 1 <br>
<p>Kartik Ahuja, Amit Dhurandhar, Kush R. Varshney</p></summary>
<p>

**Abstract:** Non-convex optimization problems are challenging to solve; the success and computational expense of a gradient descent algorithm or variant depend heavily on the initialization strategy. Often, either random initialization is used or initialization rules are carefully designed by exploiting the nature of the problem class. As a simple alternative to hand-crafted initialization rules, we propose an approach for learning "good" initialization rules from previous solutions. We provide theoretical guarantees that establish conditions that are sufficient in all cases and also necessary in some under which our approach performs better than random initialization. We apply our methodology to various non-convex problems such as generating adversarial examples, generating post hoc explanations for black-box machine learning models, and allocating communication spectrum, and show consistent gains over other initialization techniques.

</p>
</details>

<details><summary><b>Algorithms for Solving Nonlinear Binary Optimization Problems in Robust Causal Inference</b>
<a href="https://arxiv.org/abs/2012.12130">arxiv:2012.12130</a>
&#x1F4C8; 1 <br>
<p>Md Saiful Islam, Md Sarowar Morshed, Md. Noor-E-Alam</p></summary>
<p>

**Abstract:** Identifying cause-effect relation among variables is a key step in the decision-making process. While causal inference requires randomized experiments, researchers and policymakers are increasingly using observational studies to test causal hypotheses due to the wide availability of observational data and the infeasibility of experiments. The matching method is the most used technique to make causal inference from observational data. However, the pair assignment process in one-to-one matching creates uncertainty in the inference because of different choices made by the experimenter. Recently, discrete optimization models are proposed to tackle such uncertainty. Although a robust inference is possible with discrete optimization models, they produce nonlinear problems and lack scalability. In this work, we propose greedy algorithms to solve the robust causal inference test instances from observational data with continuous outcomes. We propose a unique framework to reformulate the nonlinear binary optimization problems as feasibility problems. By leveraging the structure of the feasibility formulation, we develop greedy schemes that are efficient in solving robust test problems. In many cases, the proposed algorithms achieve global optimal solution. We perform experiments on three real-world datasets to demonstrate the effectiveness of the proposed algorithms and compare our result with the state-of-the-art solver. Our experiments show that the proposed algorithms significantly outperform the exact method in terms of computation time while achieving the same conclusion for causal tests. Both numerical experiments and complexity analysis demonstrate that the proposed algorithms ensure the scalability required for harnessing the power of big data in the decision-making process.

</p>
</details>

<details><summary><b>Data Assimilation in the Latent Space of a Neural Network</b>
<a href="https://arxiv.org/abs/2012.12056">arxiv:2012.12056</a>
&#x1F4C8; 1 <br>
<p>Maddalena Amendola, Rossella Arcucci, Laetitia Mottet, Cesar Quilodran Casas, Shiwei Fan, Christopher Pain, Paul Linden, Yi-Ke Guo</p></summary>
<p>

**Abstract:** There is an urgent need to build models to tackle Indoor Air Quality issue. Since the model should be accurate and fast, Reduced Order Modelling technique is used to reduce the dimensionality of the problem. The accuracy of the model, that represent a dynamic system, is improved integrating real data coming from sensors using Data Assimilation techniques. In this paper, we formulate a new methodology called Latent Assimilation that combines Data Assimilation and Machine Learning. We use a Convolutional neural network to reduce the dimensionality of the problem, a Long-Short-Term-Memory to build a surrogate model of the dynamic system and an Optimal Interpolated Kalman Filter to incorporate real data. Experimental results are provided for CO2 concentration within an indoor space. This methodology can be used for example to predict in real-time the load of virus, such as the SARS-COV-2, in the air by linking it to the concentration of CO2.

</p>
</details>

<details><summary><b>Unsupervised Functional Data Analysis via Nonlinear Dimension Reduction</b>
<a href="https://arxiv.org/abs/2012.11987">arxiv:2012.11987</a>
&#x1F4C8; 1 <br>
<p>Moritz Herrmann, Fabian Scheipl</p></summary>
<p>

**Abstract:** In recent years, manifold methods have moved into focus as tools for dimension reduction. Assuming that the high-dimensional data actually lie on or close to a low-dimensional nonlinear manifold, these methods have shown convincing results in several settings. This manifold assumption is often reasonable for functional data, i.e., data representing continuously observed functions, as well. However, the performance of manifold methods recently proposed for tabular or image data has not been systematically assessed in the case of functional data yet. Moreover, it is unclear how to evaluate the quality of learned embeddings that do not yield invertible mappings, since the reconstruction error cannot be used as a performance measure for such representations. In this work, we describe and investigate the specific challenges for nonlinear dimension reduction posed by the functional data setting. The contributions of the paper are three-fold: First of all, we define a theoretical framework which allows to systematically assess specific challenges that arise in the functional data context, transfer several nonlinear dimension reduction methods for tabular and image data to functional data, and show that manifold methods can be used successfully in this setting. Secondly, we subject performance assessment and tuning strategies to a thorough and systematic evaluation based on several different functional data settings and point out some previously undescribed weaknesses and pitfalls which can jeopardize reliable judgment of embedding quality. Thirdly, we propose a nuanced approach to make trustworthy decisions for or against competing nonconforming embeddings more objectively.

</p>
</details>

<details><summary><b>Fast Fluid Simulations in 3D with Physics-Informed Deep Learning</b>
<a href="https://arxiv.org/abs/2012.11893">arxiv:2012.11893</a>
&#x1F4C8; 1 <br>
<p>Nils Wandel, Michael Weinmann, Reinhard Klein</p></summary>
<p>

**Abstract:** Physically plausible fluid simulations play an important role in modern computer graphics. However, in order to achieve real-time performance, computational speed needs to be traded-off with physical accuracy. Surrogate fluid models based on neural networks are a promising candidate to achieve both: fast fluid simulations and high physical accuracy. However, these approaches do not generalize to new fluid domains, rely on massive amounts of training data or require complex pipelines for training and inference.
  In this work, we present a 3D extension to our recently proposed fluid training framework, which addresses the aforementioned issues in 2D. Our method allows to train fluid models that generalize to new fluid domains without requiring fluid simulation data and simplifies the training and inference pipeline as the fluid models directly map a fluid state and boundary conditions at a moment t to a subsequent state at t+dt. To this end, we introduce a physics-informed loss function based on the residuals of the Navier-Stokes equations on a 3D staggered Marker-and-Cell grid. Furthermore, we propose an efficient 3D U-Net based architecture in order to cope with the high demands of 3D grids in terms of memory and computational complexity. Our method allows for real-time fluid simulations on a 128x64x64 grid that include various fluid phenomena such as the Magnus effect or Karman vortex streets, and generalize to domain geometries not considered during training. Our method indicates strong improvements in terms of accuracy, speed and generalization capabilities over current 3D NN-based fluid models.

</p>
</details>

<details><summary><b>Selective Forgetting of Deep Networks at a Finer Level than Samples</b>
<a href="https://arxiv.org/abs/2012.11849">arxiv:2012.11849</a>
&#x1F4C8; 1 <br>
<p>Tomohiro Hayase, Suguru Yasutomi, Takashi Katoh</p></summary>
<p>

**Abstract:** Selective forgetting or removing information from deep neural networks (DNNs) is essential for continual learning and is challenging in controlling the DNNs. Such forgetting is crucial also in a practical sense since the deployed DNNs may be trained on the data with outliers, poisoned by attackers, or with leaked/sensitive information. In this paper, we formulate selective forgetting for classification tasks at a finer level than the samples' level. We specify the finer level based on four datasets distinguished by two conditions: whether they contain information to be forgotten and whether they are available for the forgetting procedure. Additionally, we reveal the need for such formulation with the datasets by showing concrete and practical situations. Moreover, we introduce the forgetting procedure as an optimization problem on three criteria; the forgetting, the correction, and the remembering term. Experimental results show that the proposed methods can make the model forget to use specific information for classification. Notably, in specific cases, our methods improved the model's accuracy on the datasets, which contains information to be forgotten but is unavailable in the forgetting procedure. Such data are unexpectedly found and misclassified in actual situations.

</p>
</details>

<details><summary><b>Towards Automated Satellite Conjunction Management with Bayesian Deep Learning</b>
<a href="https://arxiv.org/abs/2012.12450">arxiv:2012.12450</a>
&#x1F4C8; 0 <br>
<p>Francesco Pinto, Giacomo Acciarini, Sascha Metz, Sarah Boufelja, Sylvester Kaczmarek, Klaus Merz, José A. Martinez-Heras, Francesca Letizia, Christopher Bridges, Atılım Güneş Baydin</p></summary>
<p>

**Abstract:** After decades of space travel, low Earth orbit is a junkyard of discarded rocket bodies, dead satellites, and millions of pieces of debris from collisions and explosions. Objects in high enough altitudes do not re-enter and burn up in the atmosphere, but stay in orbit around Earth for a long time. With a speed of 28,000 km/h, collisions in these orbits can generate fragments and potentially trigger a cascade of more collisions known as the Kessler syndrome. This could pose a planetary challenge, because the phenomenon could escalate to the point of hindering future space operations and damaging satellite infrastructure critical for space and Earth science applications. As commercial entities place mega-constellations of satellites in orbit, the burden on operators conducting collision avoidance manoeuvres will increase. For this reason, development of automated tools that predict potential collision events (conjunctions) is critical. We introduce a Bayesian deep learning approach to this problem, and develop recurrent neural network architectures (LSTMs) that work with time series of conjunction data messages (CDMs), a standard data format used by the space community. We show that our method can be used to model all CDM features simultaneously, including the time of arrival of future CDMs, providing predictions of conjunction event evolution with associated uncertainties.

</p>
</details>

<details><summary><b>Partial Identifiability in Discrete Data With Measurement Error</b>
<a href="https://arxiv.org/abs/2012.12449">arxiv:2012.12449</a>
&#x1F4C8; 0 <br>
<p>Noam Finkelstein, Roy Adams, Suchi Saria, Ilya Shpitser</p></summary>
<p>

**Abstract:** When data contains measurement errors, it is necessary to make assumptions relating the observed, erroneous data to the unobserved true phenomena of interest. These assumptions should be justifiable on substantive grounds, but are often motivated by mathematical convenience, for the sake of exactly identifying the target of inference. We adopt the view that it is preferable to present bounds under justifiable assumptions than to pursue exact identification under dubious ones. To that end, we demonstrate how a broad class of modeling assumptions involving discrete variables, including common measurement error and conditional independence assumptions, can be expressed as linear constraints on the parameters of the model. We then use linear programming techniques to produce sharp bounds for factual and counterfactual distributions under measurement error in such models. We additionally propose a procedure for obtaining outer bounds on non-linear models. Our method yields sharp bounds in a number of important settings -- such as the instrumental variable scenario with measurement error -- for which no bounds were previously known.

</p>
</details>

<details><summary><b>Unbiased Gradient Estimation for Distributionally Robust Learning</b>
<a href="https://arxiv.org/abs/2012.12367">arxiv:2012.12367</a>
&#x1F4C8; 0 <br>
<p>Soumyadip Ghosh, Mark Squillante</p></summary>
<p>

**Abstract:** Seeking to improve model generalization, we consider a new approach based on distributionally robust learning (DRL) that applies stochastic gradient descent to the outer minimization problem. Our algorithm efficiently estimates the gradient of the inner maximization problem through multi-level Monte Carlo randomization. Leveraging theoretical results that shed light on why standard gradient estimators fail, we establish the optimal parameterization of the gradient estimators of our approach that balances a fundamental tradeoff between computation time and statistical variance. Numerical experiments demonstrate that our DRL approach yields significant benefits over previous work.

</p>
</details>

<details><summary><b>Unbiased Subdata Selection for Fair Classification: A Unified Framework and Scalable Algorithms</b>
<a href="https://arxiv.org/abs/2012.12356">arxiv:2012.12356</a>
&#x1F4C8; 0 <br>
<p>Qing Ye, Weijun Xie</p></summary>
<p>

**Abstract:** As an important problem in modern data analytics, classification has witnessed varieties of applications from different domains. Different from conventional classification approaches, fair classification concerns the issues of unintentional biases against the sensitive features (e.g., gender, race). Due to high nonconvexity of fairness measures, existing methods are often unable to model exact fairness, which can cause inferior fair classification outcomes. This paper fills the gap by developing a novel unified framework to jointly optimize accuracy and fairness. The proposed framework is versatile and can incorporate different fairness measures studied in literature precisely as well as can be applicable to many classifiers including deep classification models. Specifically, in this paper, we first prove Fisher consistency of the proposed framework. We then show that many classification models within this framework can be recast as mixed-integer convex programs, which can be solved effectively by off-the-shelf solvers when the instance sizes are moderate and can be used as benchmarks to compare the efficiency of approximation algorithms. We prove that in the proposed framework, when the classification outcomes are known, the resulting problem, termed "unbiased subdata selection," is strongly polynomial-solvable and can be used to enhance the classification fairness by selecting more representative data points. This motivates us to develop an iterative refining strategy (IRS) to solve the large-scale instances, where we improve the classification accuracy and conduct the unbiased subdata selection in an alternating fashion. We study the convergence property of IRS and derive its approximation bound. More broadly, this framework can be leveraged to improve classification models with unbalanced data by taking F1 score into consideration.

</p>
</details>

<details><summary><b>An overview on deep learning-based approximation methods for partial differential equations</b>
<a href="https://arxiv.org/abs/2012.12348">arxiv:2012.12348</a>
&#x1F4C8; 0 <br>
<p>Christian Beck, Martin Hutzenthaler, Arnulf Jentzen, Benno Kuckuck</p></summary>
<p>

**Abstract:** It is one of the most challenging problems in applied mathematics to approximatively solve high-dimensional partial differential equations (PDEs). Recently, several deep learning-based approximation algorithms for attacking this problem have been proposed and tested numerically on a number of examples of high-dimensional PDEs. This has given rise to a lively field of research in which deep learning-based methods and related Monte Carlo methods are applied to the approximation of high-dimensional PDEs. In this article we offer an introduction to this field of research, we review some of the main ideas of deep learning-based approximation methods for PDEs, we revisit one of the central mathematical results for deep neural network approximations for PDEs, and we provide an overview of the recent literature in this area of research.

</p>
</details>

<details><summary><b>Evolutionary Variational Optimization of Generative Models</b>
<a href="https://arxiv.org/abs/2012.12294">arxiv:2012.12294</a>
&#x1F4C8; 0 <br>
<p>Jakob Drefs, Enrico Guiraud, Jörg Lücke</p></summary>
<p>

**Abstract:** We combine two popular optimization approaches to derive learning algorithms for generative models: variational optimization and evolutionary algorithms. The combination is realized for generative models with discrete latents by using truncated posteriors as the family of variational distributions. The variational parameters of truncated posteriors are sets of latent states. By interpreting these states as genomes of individuals and by using the variational lower bound to define a fitness, we can apply evolutionary algorithms to realize the variational loop. The used variational distributions are very flexible and we show that evolutionary algorithms can effectively and efficiently optimize the variational bound. Furthermore, the variational loop is generally applicable ("black box") with no analytical derivations required. To show general applicability, we apply the approach to three generative models (we use noisy-OR Bayes Nets, Binary Sparse Coding, and Spike-and-Slab Sparse Coding). To demonstrate effectiveness and efficiency of the novel variational approach, we use the standard competitive benchmarks of image denoising and inpainting. The benchmarks allow quantitative comparisons to a wide range of methods including probabilistic approaches, deep deterministic and generative networks, and non-local image processing methods. In the category of "zero-shot" learning (when only the corrupted image is used for training), we observed the evolutionary variational algorithm to significantly improve the state-of-the-art in many benchmark settings. For one well-known inpainting benchmark, we also observed state-of-the-art performance across all categories of algorithms although we only train on the corrupted image. In general, our investigations highlight the importance of research on optimization methods for generative models to achieve performance improvements.

</p>
</details>

<details><summary><b>Group-Aware Robot Navigation in Crowded Environments</b>
<a href="https://arxiv.org/abs/2012.12291">arxiv:2012.12291</a>
&#x1F4C8; 0 <br>
<p>Kapil Katyal, Yuxiang Gao, Jared Markowitz, I-Jeng Wang, Chien-Ming Huang</p></summary>
<p>

**Abstract:** Human-aware robot navigation promises a range of applications in which mobile robots bring versatile assistance to people in common human environments. While prior research has mostly focused on modeling pedestrians as independent, intentional individuals, people move in groups; consequently, it is imperative for mobile robots to respect human groups when navigating around people. This paper explores learning group-aware navigation policies based on dynamic group formation using deep reinforcement learning. Through simulation experiments, we show that group-aware policies, compared to baseline policies that neglect human groups, achieve greater robot navigation performance (e.g., fewer collisions), minimize violation of social norms and discomfort, and reduce the robot's movement impact on pedestrians. Our results contribute to the development of social navigation and the integration of mobile robots into human environments.

</p>
</details>

<details><summary><b>Iteratively Reweighted Least Squares for $\ell_1$-minimization with Global Linear Convergence Rate</b>
<a href="https://arxiv.org/abs/2012.12250">arxiv:2012.12250</a>
&#x1F4C8; 0 <br>
<p>Christian Kümmerle, Claudio Mayrink Verdun, Dominik Stöger</p></summary>
<p>

**Abstract:** Iteratively Reweighted Least Squares (IRLS), whose history goes back more than 80 years, represents an important family of algorithms for non-smooth optimization as it is able to optimize these problems by solving a sequence of linear systems. In 2010, Daubechies, DeVore, Fornasier, and Güntürk proved that IRLS for $\ell_1$-minimization, an optimization program ubiquitous in the field of compressed sensing, globally converges to a sparse solution. While this algorithm has been popular in applications in engineering and statistics, fundamental algorithmic questions have remained unanswered. As a matter of fact, existing convergence guarantees only provide global convergence without any rate, except for the case that the support of the underlying signal has already been identified. In this paper, we prove that IRLS for $\ell_1$-minimization converges to a sparse solution with a global linear rate. We support our theory by numerical experiments indicating that our linear rate essentially captures the correct dimension dependence.

</p>
</details>


[Next Page](2020/2020-12/2020-12-21.md)
