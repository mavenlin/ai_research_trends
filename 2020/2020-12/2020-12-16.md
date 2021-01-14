## Summary for 2020-12-16, created on 2021-01-13


<details><summary><b>Variational Quantum Algorithms</b>
<a href="https://arxiv.org/abs/2012.09265">arxiv:2012.09265</a>
&#x1F4C8; 1670 <br>
<p>M. Cerezo, Andrew Arrasmith, Ryan Babbush, Simon C. Benjamin, Suguru Endo, Keisuke Fujii, Jarrod R. McClean, Kosuke Mitarai, Xiao Yuan, Lukasz Cincio, Patrick J. Coles</p></summary>
<p>

**Abstract:** Applications such as simulating large quantum systems or solving large-scale linear algebra problems are immensely challenging for classical computers due their extremely high computational cost. Quantum computers promise to unlock these applications, although fault-tolerant quantum computers will likely not be available for several years. Currently available quantum devices have serious constraints, including limited qubit numbers and noise processes that limit circuit depth. Variational Quantum Algorithms (VQAs), which employ a classical optimizer to train a parametrized quantum circuit, have emerged as a leading strategy to address these constraints. VQAs have now been proposed for essentially all applications that researchers have envisioned for quantum computers, and they appear to the best hope for obtaining quantum advantage. Nevertheless, challenges remain including the trainability, accuracy, and efficiency of VQAs. In this review article we present an overview of the field of VQAs. Furthermore, we discuss strategies to overcome their challenges as well as the exciting prospects for using them as a means to obtain quantum advantage.

</p>
</details>

<details><summary><b>Learning Continuous Image Representation with Local Implicit Image Function</b>
<a href="https://arxiv.org/abs/2012.09161">arxiv:2012.09161</a>
&#x1F4C8; 161 <br>
<p>Yinbo Chen, Sifei Liu, Xiaolong Wang</p></summary>
<p>

**Abstract:** How to represent an image? While the visual world is presented in a continuous manner, machines store and see the images in a discrete way with 2D arrays of pixels. In this paper, we seek to learn a continuous representation for images. Inspired by the recent progress in 3D reconstruction with implicit function, we propose Local Implicit Image Function (LIIF), which takes an image coordinate and the 2D deep features around the coordinate as inputs, predicts the RGB value at a given coordinate as an output. Since the coordinates are continuous, LIIF can be presented in an arbitrary resolution. To generate the continuous representation for pixel-based images, we train an encoder and LIIF representation via a self-supervised task with super-resolution. The learned continuous representation can be presented in arbitrary resolution even extrapolate to $\times 30$ higher resolution, where the training tasks are not provided. We further show that LIIF representation builds a bridge between discrete and continuous representation in 2D, it naturally supports the learning tasks with size-varied image ground-truths and significantly outperforms the method with resizing the ground-truths. Our project page with code is at https://yinboc.github.io/liif/ .

</p>
</details>

<details><summary><b>Distilling Optimal Neural Networks: Rapid Search in Diverse Spaces</b>
<a href="https://arxiv.org/abs/2012.08859">arxiv:2012.08859</a>
&#x1F4C8; 79 <br>
<p>Bert Moons, Parham Noorzad, Andrii Skliar, Giovanni Mariani, Dushyant Mehta, Chris Lott, Tijmen Blankevoort</p></summary>
<p>

**Abstract:** This work presents DONNA (Distilling Optimal Neural Network Architectures), a novel pipeline for rapid neural architecture search and search space exploration, targeting multiple different hardware platforms and user scenarios. In DONNA, a search consists of three phases. First, an accuracy predictor is built for a diverse search space using blockwise knowledge distillation. This predictor enables searching across diverse macro-architectural network parameters such as layer types, attention mechanisms, and channel widths, as well as across micro-architectural parameters such as block repeats, kernel sizes, and expansion rates. Second, a rapid evolutionary search phase finds a Pareto-optimal set of architectures in terms of accuracy and latency for any scenario using the predictor and on-device measurements. Third, Pareto-optimal models can be quickly finetuned to full accuracy. With this approach, DONNA finds architectures that outperform the state of the art. In ImageNet classification, architectures found by DONNA are 20% faster than EfficientNet-B0 and MobileNetV2 on a Nvidia V100 GPU at similar accuracy and 10% faster with 0.5% higher accuracy than MobileNetV2-1.4x on a Samsung S20 smartphone. In addition to neural architecture search, DONNA is used for search-space exploration and hardware-aware model compression.

</p>
</details>

<details><summary><b>A Hybrid Graph Neural Network Approach for Detecting PHP Vulnerabilities</b>
<a href="https://arxiv.org/abs/2012.08835">arxiv:2012.08835</a>
&#x1F4C8; 40 <br>
<p>Rishi Rabheru, Hazim Hanif, Sergio Maffeis</p></summary>
<p>

**Abstract:** This paper presents DeepTective, a deep learning approach to detect vulnerabilities in PHP source code. Our approach implements a novel hybrid technique that combines Gated Recurrent Units and Graph Convolutional Networks to detect SQLi, XSS and OSCI vulnerabilities leveraging both syntactic and semantic information. We evaluate DeepTective and compare it to the state of the art on an established synthetic dataset and on a novel real-world dataset collected from GitHub. Experimental results show that DeepTective achieves near perfect classification on the synthetic dataset, and an F1 score of 88.12% on the realistic dataset, outperforming related approaches. We validate DeepTective in the wild by discovering 4 novel vulnerabilities in established WordPress plugins.

</p>
</details>

<details><summary><b>Adversarial trading</b>
<a href="https://arxiv.org/abs/2101.03128">arxiv:2101.03128</a>
&#x1F4C8; 13 <br>
<p>Alexandre Miot</p></summary>
<p>

**Abstract:** Adversarial samples have drawn a lot of attention from the Machine Learning community in the past few years. An adverse sample is an artificial data point coming from an imperceptible modification of a sample point aiming at misleading. Surprisingly, in financial research, little has been done in relation to this topic from a concrete trading point of view. We show that those adversarial samples can be implemented in a trading environment and have a negative impact on certain market participants. This could have far reaching implications for financial markets either from a trading or a regulatory point of view.

</p>
</details>

<details><summary><b>Graph integration of structured, semistructured and unstructured data for data journalism</b>
<a href="https://arxiv.org/abs/2012.08830">arxiv:2012.08830</a>
&#x1F4C8; 11 <br>
<p>Angelos-Christos Anadiotis, Oana Balalau, Catarina Conceicao, Helena Galhardas, Mhd Yamen Haddad, Ioana Manolescu, Tayeb Merabti, Jingmao You</p></summary>
<p>

**Abstract:** Digital data is a gold mine for modern journalism. However, datasets which interest journalists are extremely heterogeneous, ranging from highly structured (relational databases), semi-structured (JSON, XML, HTML), graphs (e.g., RDF), and text. Journalists (and other classes of users lacking advanced IT expertise, such as most non-governmental-organizations, or small public administrations) need to be able to make sense of such heterogeneous corpora, even if they lack the ability to define and deploy custom extract-transform-load workflows, especially for dynamically varying sets of data sources.
  We describe a complete approach for integrating dynamic sets of heterogeneous datasets along the lines described above: the challenges we faced to make such graphs useful, allow their integration to scale, and the solutions we proposed for these problems. Our approach is implemented within the ConnectionLens system; we validate it through a set of experiments.

</p>
</details>

<details><summary><b>S3CNet: A Sparse Semantic Scene Completion Network for LiDAR Point Clouds</b>
<a href="https://arxiv.org/abs/2012.09242">arxiv:2012.09242</a>
&#x1F4C8; 10 <br>
<p>Ran Cheng, Christopher Agia, Yuan Ren, Xinhai Li, Liu Bingbing</p></summary>
<p>

**Abstract:** With the increasing reliance of self-driving and similar robotic systems on robust 3D vision, the processing of LiDAR scans with deep convolutional neural networks has become a trend in academia and industry alike. Prior attempts on the challenging Semantic Scene Completion task - which entails the inference of dense 3D structure and associated semantic labels from "sparse" representations - have been, to a degree, successful in small indoor scenes when provided with dense point clouds or dense depth maps often fused with semantic segmentation maps from RGB images. However, the performance of these systems drop drastically when applied to large outdoor scenes characterized by dynamic and exponentially sparser conditions. Likewise, processing of the entire sparse volume becomes infeasible due to memory limitations and workarounds introduce computational inefficiency as practitioners are forced to divide the overall volume into multiple equal segments and infer on each individually, rendering real-time performance impossible. In this work, we formulate a method that subsumes the sparsity of large-scale environments and present S3CNet, a sparse convolution based neural network that predicts the semantically completed scene from a single, unified LiDAR point cloud. We show that our proposed method outperforms all counterparts on the 3D task, achieving state-of-the art results on the SemanticKITTI benchmark. Furthermore, we propose a 2D variant of S3CNet with a multi-view fusion strategy to complement our 3D network, providing robustness to occlusions and extreme sparsity in distant regions. We conduct experiments for the 2D semantic scene completion task and compare the results of our sparse 2D network against several leading LiDAR segmentation models adapted for bird's eye view segmentation on two open-source datasets.

</p>
</details>

<details><summary><b>DECOR-GAN: 3D Shape Detailization by Conditional Refinement</b>
<a href="https://arxiv.org/abs/2012.09159">arxiv:2012.09159</a>
&#x1F4C8; 10 <br>
<p>Zhiqin Chen, Vladimir Kim, Matthew Fisher, Noam Aigerman, Hao Zhang, Siddhartha Chaudhuri</p></summary>
<p>

**Abstract:** We introduce a deep generative network for 3D shape detailization, akin to stylization with the style being geometric details. We address the challenge of creating large varieties of high-resolution and detailed 3D geometry from a small set of exemplars by treating the problem as that of geometric detail transfer. Given a low-resolution coarse voxel shape, our network refines it, via voxel upsampling, into a higher-resolution shape enriched with geometric details. The output shape preserves the overall structure (or content) of the input, while its detail generation is conditioned on an input "style code" corresponding to a detailed exemplar. Our 3D detailization via conditional refinement is realized by a generative adversarial network, coined DECOR-GAN. The network utilizes a 3D CNN generator for upsampling coarse voxels and a 3D PatchGAN discriminator to enforce local patches of the generated model to be similar to those in the training detailed shapes. During testing, a style code is fed into the generator to condition the refinement. We demonstrate that our method can refine a coarse shape into a variety of detailed shapes with different styles. The generated results are evaluated in terms of content preservation, plausibility, and diversity. Comprehensive ablation studies are conducted to validate our network designs.

</p>
</details>

<details><summary><b>Relational Boosted Bandits</b>
<a href="https://arxiv.org/abs/2012.09220">arxiv:2012.09220</a>
&#x1F4C8; 9 <br>
<p>Ashutosh Kakadiya, Sriraam Natarajan, Balaraman Ravindran</p></summary>
<p>

**Abstract:** Contextual bandits algorithms have become essential in real-world user interaction problems in recent years. However, these algorithms rely on context as attribute value representation, which makes them unfeasible for real-world domains like social networks are inherently relational. We propose Relational Boosted Bandits(RB2), acontextual bandits algorithm for relational domains based on (relational) boosted trees. RB2 enables us to learn interpretable and explainable models due to the more descriptive nature of the relational representation. We empirically demonstrate the effectiveness and interpretability of RB2 on tasks such as link prediction, relational classification, and recommendations.

</p>
</details>

<details><summary><b>Sparse Signal Models for Data Augmentation in Deep Learning ATR</b>
<a href="https://arxiv.org/abs/2012.09284">arxiv:2012.09284</a>
&#x1F4C8; 7 <br>
<p>Tushar Agarwal, Nithin Sugavanam, Emre Ertin</p></summary>
<p>

**Abstract:** Automatic Target Recognition (ATR) algorithms classify a given Synthetic Aperture Radar (SAR) image into one of the known target classes using a set of training images available for each class. Recently, learning methods have shown to achieve state-of-the-art classification accuracy if abundant training data is available, sampled uniformly over the classes, and their poses. In this paper, we consider the task of ATR with a limited set of training images. We propose a data augmentation approach to incorporate domain knowledge and improve the generalization power of a data-intensive learning algorithm, such as a Convolutional neural network (CNN). The proposed data augmentation method employs a limited persistence sparse modeling approach, capitalizing on commonly observed characteristics of wide-angle synthetic aperture radar (SAR) imagery. Specifically, we exploit the sparsity of the scattering centers in the spatial domain and the smoothly-varying structure of the scattering coefficients in the azimuthal domain to solve the ill-posed problem of over-parametrized model fitting. Using this estimated model, we synthesize new images at poses and sub-pixel translations not available in the given data to augment CNN's training data. The experimental results show that for the training data starved region, the proposed method provides a significant gain in the resulting ATR algorithm's generalization performance.

</p>
</details>

<details><summary><b>Clustering with Iterated Linear Optimization</b>
<a href="https://arxiv.org/abs/2012.09202">arxiv:2012.09202</a>
&#x1F4C8; 7 <br>
<p>Pedro Felzenszwalb, Caroline Klivans, Alice Paul</p></summary>
<p>

**Abstract:** We introduce a novel method for clustering using a semidefinite programming (SDP) relaxation of the Max k-Cut problem. The approach is based on a new methodology for rounding the solution of an SDP using iterated linear optimization. We show the vertices of the Max k-Cut SDP relaxation correspond to partitions of the data into at most k sets. We also show the vertices are attractive fixed points of iterated linear optimization. We interpret the process of fixed point iteration with linear optimization as repeated relaxations of the closest vertex problem. Our experiments show that using fixed point iteration for rounding the Max k-Cut SDP relaxation leads to significantly better results when compared to randomized rounding.

</p>
</details>

<details><summary><b>Classifying Sequences of Extreme Length with Constant Memory Applied to Malware Detection</b>
<a href="https://arxiv.org/abs/2012.09390">arxiv:2012.09390</a>
&#x1F4C8; 6 <br>
<p>Edward Raff, William Fleshman, Richard Zak, Hyrum S. Anderson, Bobby Filar, Mark McLean</p></summary>
<p>

**Abstract:** Recent works within machine learning have been tackling inputs of ever-increasing size, with cybersecurity presenting sequence classification problems of particularly extreme lengths. In the case of Windows executable malware detection, inputs may exceed $100$ MB, which corresponds to a time series with $T=100,000,000$ steps. To date, the closest approach to handling such a task is MalConv, a convolutional neural network capable of processing up to $T=2,000,000$ steps. The $\mathcal{O}(T)$ memory of CNNs has prevented further application of CNNs to malware. In this work, we develop a new approach to temporal max pooling that makes the required memory invariant to the sequence length $T$. This makes MalConv $116\times$ more memory efficient, and up to $25.8\times$ faster to train on its original dataset, while removing the input length restrictions to MalConv. We re-invest these gains into improving the MalConv architecture by developing a new Global Channel Gating design, giving us an attention mechanism capable of learning feature interactions across 100 million time steps in an efficient manner, a capability lacked by the original MalConv CNN. Our implementation can be found at https://github.com/NeuromorphicComputationResearchProgram/MalConv2

</p>
</details>

<details><summary><b>Machine Learning for Detecting Data Exfiltration</b>
<a href="https://arxiv.org/abs/2012.09344">arxiv:2012.09344</a>
&#x1F4C8; 6 <br>
<p>Bushra Sabir, Faheem Ullah, M. Ali Babar, Raj Gaire</p></summary>
<p>

**Abstract:** Context: Research at the intersection of cybersecurity, Machine Learning (ML), and Software Engineering (SE) has recently taken significant steps in proposing countermeasures for detecting sophisticated data exfiltration attacks. It is important to systematically review and synthesize the ML-based data exfiltration countermeasures for building a body of knowledge on this important topic. Objective: This paper aims at systematically reviewing ML-based data exfiltration countermeasures to identify and classify ML approaches, feature engineering techniques, evaluation datasets, and performance metrics used for these countermeasures. This review also aims at identifying gaps in research on ML-based data exfiltration countermeasures. Method: We used a Systematic Literature Review (SLR) method to select and review {92} papers. Results: The review has enabled us to (a) classify the ML approaches used in the countermeasures into data-driven, and behaviour-driven approaches, (b) categorize features into six types: behavioural, content-based, statistical, syntactical, spatial and temporal, (c) classify the evaluation datasets into simulated, synthesized, and real datasets and (d) identify 11 performance measures used by these studies. Conclusion: We conclude that: (i) the integration of data-driven and behaviour-driven approaches should be explored; (ii) There is a need of developing high quality and large size evaluation datasets; (iii) Incremental ML model training should be incorporated in countermeasures; (iv) resilience to adversarial learning should be considered and explored during the development of countermeasures to avoid poisoning attacks; and (v) the use of automated feature engineering should be encouraged for efficiently detecting data exfiltration attacks.

</p>
</details>

<details><summary><b>Shape My Face: Registering 3D Face Scans by Surface-to-Surface Translation</b>
<a href="https://arxiv.org/abs/2012.09235">arxiv:2012.09235</a>
&#x1F4C8; 6 <br>
<p>Mehdi Bahri, Eimear O' Sullivan, Shunwang Gong, Feng Liu, Xiaoming Liu, Michael M. Bronstein, Stefanos Zafeiriou</p></summary>
<p>

**Abstract:** Existing surface registration methods focus on fitting in-sample data with little to no generalization ability and require both heavy pre-processing and careful hand-tuning. In this paper, we cast the registration task as a surface-to-surface translation problem, and design a model to reliably capture the latent geometric information directly from raw 3D face scans. We introduce Shape-My-Face (SMF), a powerful encoder-decoder architecture based on an improved point cloud encoder, a novel visual attention mechanism, graph convolutional decoders with skip connections, and a specialized mouth model that we smoothly integrate with the mesh convolutions. Compared to the previous state-of-the-art learning algorithms for non-rigid registration of face scans, SMF only requires the raw data to be rigidly aligned (with scaling) with a pre-defined face template. Additionally, our model provides topologically-sound meshes with minimal supervision, offers faster training time, has orders of magnitude fewer trainable parameters, is more robust to noise, and can generalize to previously unseen datasets. We extensively evaluate the quality of our registrations on diverse data. We demonstrate the robustness and generalizability of our model with in-the-wild face scans across different modalities, sensor types, and resolutions. Finally, we show that, by learning to register scans, SMF produces a hybrid linear and non-linear morphable model that can be used for generation, shape morphing, and expression transfer through manipulation of the latent space, including in-the-wild. We train SMF on a dataset of human faces comprising 9 large-scale databases on commodity hardware.

</p>
</details>

<details><summary><b>Optimal transport for vector Gaussian mixture models</b>
<a href="https://arxiv.org/abs/2012.09226">arxiv:2012.09226</a>
&#x1F4C8; 6 <br>
<p>Jiening Zhu, Kaiming Xu, Allen Tannenbaum</p></summary>
<p>

**Abstract:** Vector Gaussian mixture models form an important special subset of vector-valued distributions. Any physical entity that can mutate or transit among alternative manifestations distributed in a given space falls into this category. A key example is color imagery. In this note, we vectorize the Gaussian mixture model and study different optimal mass transport related problems for such models. The benefits of using vector Gaussian mixture for optimal mass transport include computational efficiency and the ability to preserve structure.

</p>
</details>

<details><summary><b>Assessing COVID-19 Impacts on College Students via Automated Processing of Free-form Text</b>
<a href="https://arxiv.org/abs/2012.09369">arxiv:2012.09369</a>
&#x1F4C8; 5 <br>
<p>Ravi Sharma, Sri Divya Pagadala, Pratool Bharti, Sriram Chellappan, Trine Schmidt, Raj Goyal</p></summary>
<p>

**Abstract:** In this paper, we report experimental results on assessing the impact of COVID-19 on college students by processing free-form texts generated by them. By free-form texts, we mean textual entries posted by college students (enrolled in a four year US college) via an app specifically designed to assess and improve their mental health. Using a dataset comprising of more than 9000 textual entries from 1451 students collected over four months (split between pre and post COVID-19), and established NLP techniques, a) we assess how topics of most interest to student change between pre and post COVID-19, and b) we assess the sentiments that students exhibit in each topic between pre and post COVID-19. Our analysis reveals that topics like Education became noticeably less important to students post COVID-19, while Health became much more trending. We also found that across all topics, negative sentiment among students post COVID-19 was much higher compared to pre-COVID-19. We expect our study to have an impact on policy-makers in higher education across several spectra, including college administrators, teachers, parents, and mental health counselors.

</p>
</details>

<details><summary><b>Do You Do Yoga? Understanding Twitter Users' Types and Motivations using Social and Textual Information</b>
<a href="https://arxiv.org/abs/2012.09332">arxiv:2012.09332</a>
&#x1F4C8; 5 <br>
<p>Tunazzina Islam, Dan Goldwasser</p></summary>
<p>

**Abstract:** Leveraging social media data to understand people's lifestyle choices is an exciting domain to explore but requires a multiview formulation of the data. In this paper, we propose a joint embedding model based on the fusion of neural networks with attention mechanism by incorporating social and textual information of users to understand their activities and motivations. We use well-being related tweets from Twitter, focusing on 'Yoga'. We demonstrate our model on two downstream tasks: (i) finding user type such as either practitioner or promotional (promoting yoga studio/gym), other; (ii) finding user motivation i.e. health benefit, spirituality, love to tweet/retweet about yoga but do not practice yoga.

</p>
</details>

<details><summary><b>Series Saliency: Temporal Interpretation for Multivariate Time Series Forecasting</b>
<a href="https://arxiv.org/abs/2012.09324">arxiv:2012.09324</a>
&#x1F4C8; 5 <br>
<p>Qingyi Pan, Wenbo Hu, Jun Zhu</p></summary>
<p>

**Abstract:** Time series forecasting is an important yet challenging task. Though deep learning methods have recently been developed to give superior forecasting results, it is crucial to improve the interpretability of time series models. Previous interpretation methods, including the methods for general neural networks and attention-based methods, mainly consider the interpretation in the feature dimension while ignoring the crucial temporal dimension. In this paper, we present the series saliency framework for temporal interpretation for multivariate time series forecasting, which considers the forecasting interpretation in both feature and temporal dimensions. By extracting the "series images" from the sliding windows of the time series, we apply the saliency map segmentation following the smallest destroying region principle. The series saliency framework can be employed to any well-defined deep learning models and works as a data augmentation to get more accurate forecasts. Experimental results on several real datasets demonstrate that our framework generates temporal interpretations for the time series forecasting task while produces accurate time series forecast.

</p>
</details>

<details><summary><b>Neural Pruning via Growing Regularization</b>
<a href="https://arxiv.org/abs/2012.09243">arxiv:2012.09243</a>
&#x1F4C8; 5 <br>
<p>Huan Wang, Can Qin, Yulun Zhang, Yun Fu</p></summary>
<p>

**Abstract:** Regularization has long been utilized to learn sparsity in deep neural network pruning. However, its role is mainly explored in the small penalty strength regime. In this work, we extend its application to a new scenario where the regularization grows large gradually to tackle two central problems of pruning: pruning schedule and weight importance scoring. (1) The former topic is newly brought up in this work, which we find critical to the pruning performance while receives little research attention. Specifically, we propose an L2 regularization variant with rising penalty factors and show it can bring significant accuracy gains compared with its one-shot counterpart, even when the same weights are removed. (2) The growing penalty scheme also brings us an approach to exploit the Hessian information for more accurate pruning without knowing their specific values, thus not bothered by the common Hessian approximation problems. Empirically, the proposed algorithms are easy to implement and scalable to large datasets and networks in both structured and unstructured pruning. Their effectiveness is demonstrated with modern deep neural networks on the CIFAR and ImageNet datasets, achieving competitive results compared to many state-of-the-art algorithms. Our code and trained models are publicly available at https://github.com/mingsuntse/regularization-pruning.

</p>
</details>

<details><summary><b>uBAM: Unsupervised Behavior Analysis and Magnification using Deep Learning</b>
<a href="https://arxiv.org/abs/2012.09237">arxiv:2012.09237</a>
&#x1F4C8; 5 <br>
<p>Biagio Brattoli, Uta Buechler, Michael Dorkenwald, Philipp Reiser, Linard Filli, Fritjof Helmchen, Anna-Sophia Wahl, Bjoern Ommer</p></summary>
<p>

**Abstract:** Motor behavior analysis is essential to biomedical research and clinical diagnostics as it provides a non-invasive strategy for identifying motor impairment and its change caused by interventions. State-of-the-art instrumented movement analysis is time- and cost-intensive, since it requires placing physical or virtual markers. Besides the effort required for marking keypoints or annotations necessary for training or finetuning a detector, users need to know the interesting behavior beforehand to provide meaningful keypoints. We introduce uBAM, a novel, automatic deep learning algorithm for behavior analysis by discovering and magnifying deviations. We propose an unsupervised learning of posture and behavior representations that enable an objective behavior comparison across subjects. A generative model with novel disentanglement of appearance and behavior magnifies subtle behavior differences across subjects directly in a video without requiring a detour via keypoints or annotations. Evaluations on rodents and human patients with neurological diseases demonstrate the wide applicability of our approach.

</p>
</details>

<details><summary><b>MINIROCKET: A Very Fast (Almost) Deterministic Transform for Time Series Classification</b>
<a href="https://arxiv.org/abs/2012.08791">arxiv:2012.08791</a>
&#x1F4C8; 5 <br>
<p>Angus Dempster, Daniel F. Schmidt, Geoffrey I. Webb</p></summary>
<p>

**Abstract:** Until recently, the most accurate methods for time series classification were limited by high computational complexity. ROCKET achieves state-of-the-art accuracy with a fraction of the computational expense of most existing methods by transforming input time series using random convolutional kernels, and using the transformed features to train a linear classifier. We reformulate ROCKET into a new method, MINIROCKET, making it up to 75 times faster on larger datasets, and making it almost deterministic (and optionally, with additional computational expense, fully deterministic), while maintaining essentially the same accuracy. Using this method, it is possible to train and test a classifier on all of 109 datasets from the UCR archive to state-of-the-art accuracy in less than 10 minutes. MINIROCKET is significantly faster than any other method of comparable accuracy (including ROCKET), and significantly more accurate than any other method of even roughly-similar computational expense. As such, we suggest that MINIROCKET should now be considered and used as the default variant of ROCKET.

</p>
</details>

<details><summary><b>Evaluation of deep learning-based myocardial infarction quantification using Segment CMR software</b>
<a href="https://arxiv.org/abs/2012.09070">arxiv:2012.09070</a>
&#x1F4C8; 4 <br>
<p>Olivier Rukundo</p></summary>
<p>

**Abstract:** In this paper, the author evaluates the preliminary work related to automating the quantification of the size of the myocardial infarction (MI) using deep learning in Segment cardiovascular magnetic resonance (CMR) software. Here, deep learning is used to automate the segmentation of myocardial boundaries before triggering the automatic quantification of the size of the MI using the expectation-maximization, weighted intensity, a priori information (EWA) algorithm incorporated in the Segment CMR software. Experimental evaluation of the size of the MI shows that more than 50 % (average infarct scar volume), 75% (average infarct scar percentage), and 65 % (average microvascular obstruction percentage) of the network-based results are approximately very close to the expert delineation-based results. Also, in an experiment involving the visualization of myocardial and infarct contours, in all images of the selected stack, the network and expert-based results tie in terms of the number of infarcted and contoured images.

</p>
</details>

<details><summary><b>Using Meta-Knowledge Mined from Identifiers to Improve Intent Recognition in Neuro-Symbolic Algorithms</b>
<a href="https://arxiv.org/abs/2012.09005">arxiv:2012.09005</a>
&#x1F4C8; 4 <br>
<p>Claudio Pinhanez, Paulo Cavalin, Victor Ribeiro, Heloisa Candello, Julio Nogima, Ana Appel, Mauro Pichiliani, Maira Gatti de Bayser, Melina Guerra, Henrique Ferreira, Gabriel Malfatti</p></summary>
<p>

**Abstract:** In this paper we explore the use of meta-knowledge embedded in intent identifiers to improve intent recognition in conversational systems. As evidenced by the analysis of thousands of real-world chatbots and in interviews with professional chatbot curators, developers and domain experts tend to organize the set of chatbot intents by identifying them using proto-taxonomies, i.e., meta-knowledge connecting high-level, symbolic concepts shared across different intents. By using neuro-symbolic algorithms able to incorporate such proto-taxonomies to expand intent representation, we show that such mined meta-knowledge can improve accuracy in intent recognition. In a dataset with intents and example utterances from hundreds of professional chatbots, we saw improvements of more than 10% in the equal error rate (EER) in almost a third of the chatbots when we apply those algorithms in comparison to a baseline of the same algorithms without the meta-knowledge. The meta-knowledge proved to be even more relevant in detecting out-of-scope utterances, decreasing the false acceptance rate (FAR) in more than 20\% in about half of the chatbots. The experiments demonstrate that such symbolic meta-knowledge structures can be effectively mined and used by neuro-symbolic algorithms, apparently by incorporating into the learning process higher-level structures of the problem being solved. Based on these results, we also discuss how the use of mined meta-knowledge can be an answer for the challenge of knowledge acquisition in neuro-symbolic algorithms.

</p>
</details>

<details><summary><b>Clinical Temporal Relation Extraction with Probabilistic Soft Logic Regularization and Global Inference</b>
<a href="https://arxiv.org/abs/2012.08790">arxiv:2012.08790</a>
&#x1F4C8; 4 <br>
<p>Yichao Zhou, Yu Yan, Rujun Han, J. Harry Caufield, Kai-Wei Chang, Yizhou Sun, Peipei Ping, Wei Wang</p></summary>
<p>

**Abstract:** There has been a steady need in the medical community to precisely extract the temporal relations between clinical events. In particular, temporal information can facilitate a variety of downstream applications such as case report retrieval and medical question answering. Existing methods either require expensive feature engineering or are incapable of modeling the global relational dependencies among the events. In this paper, we propose a novel method, Clinical Temporal ReLation Exaction with Probabilistic Soft Logic Regularization and Global Inference (CTRL-PG) to tackle the problem at the document level. Extensive experiments on two benchmark datasets, I2B2-2012 and TB-Dense, demonstrate that CTRL-PG significantly outperforms baseline methods for temporal relation extraction.

</p>
</details>

<details><summary><b>Machine Learning Algorithm for NLOS Millimeter Wave in 5G V2X Communication</b>
<a href="https://arxiv.org/abs/2012.12123">arxiv:2012.12123</a>
&#x1F4C8; 3 <br>
<p>Deepika Mohan, G. G. Md. Nawaz Ali, Peter Han Joo Chong</p></summary>
<p>

**Abstract:** The 5G vehicle-to-everything (V2X) communication for autonomous and semi-autonomous driving utilizes the wireless technology for communication and the Millimeter Wave bands are widely implemented in this kind of vehicular network application. The main purpose of this paper is to broadcast the messages from the mmWave Base Station to vehicles at LOS (Line-of-sight) and NLOS (Non-LOS). Relay using Machine Learning (RML) algorithm is formulated to train the mmBS for identifying the blockages within its coverage area and broadcast the messages to the vehicles at NLOS using a LOS nodes as a relay. The transmission of information is faster with higher throughput and it covers a wider bandwidth which is reused, therefore when performing machine learning within the coverage area of mmBS most of the vehicles in NLOS can be benefited. A unique method of relay mechanism combined with machine learning is proposed to communicate with mobile nodes at NLOS.

</p>
</details>

<details><summary><b>Spatial Context-Aware Self-Attention Model For Multi-Organ Segmentation</b>
<a href="https://arxiv.org/abs/2012.09279">arxiv:2012.09279</a>
&#x1F4C8; 3 <br>
<p>Hao Tang, Xingwei Liu, Kun Han, Shanlin Sun, Narisu Bai, Xuming Chen, Huang Qian, Yong Liu, Xiaohui Xie</p></summary>
<p>

**Abstract:** Multi-organ segmentation is one of most successful applications of deep learning in medical image analysis. Deep convolutional neural nets (CNNs) have shown great promise in achieving clinically applicable image segmentation performance on CT or MRI images. State-of-the-art CNN segmentation models apply either 2D or 3D convolutions on input images, with pros and cons associated with each method: 2D convolution is fast, less memory-intensive but inadequate for extracting 3D contextual information from volumetric images, while the opposite is true for 3D convolution. To fit a 3D CNN model on CT or MRI images on commodity GPUs, one usually has to either downsample input images or use cropped local regions as inputs, which limits the utility of 3D models for multi-organ segmentation. In this work, we propose a new framework for combining 3D and 2D models, in which the segmentation is realized through high-resolution 2D convolutions, but guided by spatial contextual information extracted from a low-resolution 3D model. We implement a self-attention mechanism to control which 3D features should be used to guide 2D segmentation. Our model is light on memory usage but fully equipped to take 3D contextual information into account. Experiments on multiple organ segmentation datasets demonstrate that by taking advantage of both 2D and 3D models, our method is consistently outperforms existing 2D and 3D models in organ segmentation accuracy, while being able to directly take raw whole-volume image data as inputs.

</p>
</details>

<details><summary><b>Sample-Efficient Reinforcement Learning via Counterfactual-Based Data Augmentation</b>
<a href="https://arxiv.org/abs/2012.09092">arxiv:2012.09092</a>
&#x1F4C8; 3 <br>
<p>Chaochao Lu, Biwei Huang, Ke Wang, José Miguel Hernández-Lobato, Kun Zhang, Bernhard Schölkopf</p></summary>
<p>

**Abstract:** Reinforcement learning (RL) algorithms usually require a substantial amount of interaction data and perform well only for specific tasks in a fixed environment. In some scenarios such as healthcare, however, usually only few records are available for each patient, and patients may show different responses to the same treatment, impeding the application of current RL algorithms to learn optimal policies. To address the issues of mechanism heterogeneity and related data scarcity, we propose a data-efficient RL algorithm that exploits structural causal models (SCMs) to model the state dynamics, which are estimated by leveraging both commonalities and differences across subjects. The learned SCM enables us to counterfactually reason what would have happened had another treatment been taken. It helps avoid real (possibly risky) exploration and mitigates the issue that limited experiences lead to biased policies. We propose counterfactual RL algorithms to learn both population-level and individual-level policies. We show that counterfactual outcomes are identifiable under mild conditions and that Q- learning on the counterfactual-based augmented data set converges to the optimal value function. Experimental results on synthetic and real-world data demonstrate the efficacy of the proposed approach.

</p>
</details>

<details><summary><b>You Are What You Tweet: Profiling Users by Past Tweets to Improve Hate Speech Detection</b>
<a href="https://arxiv.org/abs/2012.09090">arxiv:2012.09090</a>
&#x1F4C8; 3 <br>
<p>Prateek Chaudhry, Matthew Lease</p></summary>
<p>

**Abstract:** Hate speech detection research has predominantly focused on purely content-based methods, without exploiting any additional context. We briefly critique pros and cons of this task formulation. We then investigate profiling users by their past utterances as an informative prior to better predict whether new utterances constitute hate speech. To evaluate this, we augment three Twitter hate speech datasets with additional timeline data, then embed this additional context into a strong baseline model. Promising results suggest merit for further investigation, though analysis is complicated by differences in annotation schemes and processes, as well as Twitter API limitations and data sharing policies.

</p>
</details>

<details><summary><b>AdjointBackMap: Reconstructing Effective Decision Hypersurfaces from CNN Layers Using Adjoint Operators</b>
<a href="https://arxiv.org/abs/2012.09020">arxiv:2012.09020</a>
&#x1F4C8; 3 <br>
<p>Qing Wan, Yoonsuck Choe</p></summary>
<p>

**Abstract:** There are several effective methods in explaining the inner workings of convolutional neural networks (CNNs). However, in general, finding the inverse of the function performed by CNNs as a whole is an ill-posed problem. In this paper, we propose a method based on adjoint operators to reconstruct, given an arbitrary unit in the CNN (except for the first convolutional layer), its effective hypersurface in the input space that replicates that unit's decision surface conditioned on a particular input image. Our results show that the hypersurface reconstructed this way, when multiplied by the original input image, would give nearly exact output value of that unit. We find that the CNN unit's decision surface is largely conditioned on the input, and this may explain why adversarial inputs can effectively deceive CNNs.

</p>
</details>

<details><summary><b>Deep Reinforcement Learning of Graph Matching</b>
<a href="https://arxiv.org/abs/2012.08950">arxiv:2012.08950</a>
&#x1F4C8; 3 <br>
<p>Chang Liu, Runzhong Wang, Zetian Jiang, Junchi Yan</p></summary>
<p>

**Abstract:** Graph matching under node and pairwise constraints has been a building block in areas from combinatorial optimization, machine learning to computer vision, for effective structural representation and association. We present a reinforcement learning solver that seeks the node correspondence between two graphs, whereby the node embedding model on the association graph is learned to sequentially find the node-to-node matching. Our method differs from the previous deep graph matching model in the sense that they are focused on the front-end feature and affinity function learning while our method aims to learn the backend decision making given the affinity objective function whatever obtained by learning or not. Such an objective function maximization setting naturally fits with the reinforcement learning mechanism, of which the learning procedure is label-free. Besides, the model is not restricted to a fixed number of nodes for matching. These features make it more suitable for practical usage. Extensive experimental results on both synthetic datasets, natural images, and QAPLIB showcase the superior performance regarding both matching accuracy and efficiency. To our best knowledge, this is the first deep reinforcement learning solver for graph matching.

</p>
</details>

<details><summary><b>Multilingual Evidence Retrieval and Fact Verification to Combat Global Disinformation: The Power of Polyglotism</b>
<a href="https://arxiv.org/abs/2012.08919">arxiv:2012.08919</a>
&#x1F4C8; 3 <br>
<p>Denisa A. O. Roberts</p></summary>
<p>

**Abstract:** This article investigates multilingual evidence retrieval and fact verification as a step to combat global disinformation, a first effort of this kind, to the best of our knowledge. A 400 example mixed language English-Romanian dataset is created for cross-lingual transfer learning evaluation. We make code, datasets, and trained models available upon publication.

</p>
</details>

<details><summary><b>Learning from the Best: Rationalizing Prediction by Adversarial Information Calibration</b>
<a href="https://arxiv.org/abs/2012.08884">arxiv:2012.08884</a>
&#x1F4C8; 3 <br>
<p>Lei Sha, Oana-Maria Camburu, Thomas Lukasiewicz</p></summary>
<p>

**Abstract:** Explaining the predictions of AI models is paramount in safety-critical applications, such as in legal or medical domains. One form of explanation for a prediction is an extractive rationale, i.e., a subset of features of an instance that lead the model to give its prediction on the instance. Previous works on generating extractive rationales usually employ a two-phase model: a selector that selects the most important features (i.e., the rationale) followed by a predictor that makes the prediction based exclusively on the selected features. One disadvantage of these works is that the main signal for learning to select features comes from the comparison of the answers given by the predictor and the ground-truth answers. In this work, we propose to squeeze more information from the predictor via an information calibration method. More precisely, we train two models jointly: one is a typical neural model that solves the task at hand in an accurate but black-box manner, and the other is a selector-predictor model that additionally produces a rationale for its prediction. The first model is used as a guide to the second model. We use an adversarial-based technique to calibrate the information extracted by the two models such that the difference between them is an indicator of the missed or over-selected features. In addition, for natural language tasks, we propose to use a language-model-based regularizer to encourage the extraction of fluent rationales. Experimental results on a sentiment analysis task as well as on three tasks from the legal domain show the effectiveness of our approach to rationale extraction.

</p>
</details>

<details><summary><b>Multi-type Disentanglement without Adversarial Training</b>
<a href="https://arxiv.org/abs/2012.08883">arxiv:2012.08883</a>
&#x1F4C8; 3 <br>
<p>Lei Sha, Thomas Lukasiewicz</p></summary>
<p>

**Abstract:** Controlling the style of natural language by disentangling the latent space is an important step towards interpretable machine learning. After the latent space is disentangled, the style of a sentence can be transformed by tuning the style representation without affecting other features of the sentence. Previous works usually use adversarial training to guarantee that disentangled vectors do not affect each other. However, adversarial methods are difficult to train. Especially when there are multiple features (e.g., sentiment, or tense, which we call style types in this paper), each feature requires a separate discriminator for extracting a disentangled style vector corresponding to that feature. In this paper, we propose a unified distribution-controlling method, which provides each specific style value (the value of style types, e.g., positive sentiment, or past tense) with a unique representation. This method contributes a solid theoretical basis to avoid adversarial training in multi-type disentanglement. We also propose multiple loss functions to achieve a style-content disentanglement as well as a disentanglement among multiple style types. In addition, we observe that if two different style types always have some specific style values that occur together in the dataset, they will affect each other when transferring the style values. We call this phenomenon training bias, and we propose a loss function to alleviate such training bias while disentangling multiple types. We conduct experiments on two datasets (Yelp service reviews and Amazon product reviews) to evaluate the style-disentangling effect and the unsupervised style transfer performance on two style types: sentiment and tense. The experimental results show the effectiveness of our model.

</p>
</details>

<details><summary><b>Programmable Quantum Annealers as Noisy Gibbs Samplers</b>
<a href="https://arxiv.org/abs/2012.08827">arxiv:2012.08827</a>
&#x1F4C8; 3 <br>
<p>Marc Vuffray, Carleton Coffrin, Yaroslav A. Kharkov, Andrey Y. Lokhov</p></summary>
<p>

**Abstract:** Drawing independent samples from high-dimensional probability distributions represents the major computational bottleneck for modern algorithms, including powerful machine learning frameworks such as deep learning. The quest for discovering larger families of distributions for which sampling can be efficiently realized has inspired an exploration beyond established computing methods and turning to novel physical devices that leverage the principles of quantum computation. Quantum annealing embodies a promising computational paradigm that is intimately related to the complexity of energy landscapes in Gibbs distributions, which relate the probabilities of system states to the energies of these states. Here, we study the sampling properties of physical realizations of quantum annealers which are implemented through programmable lattices of superconducting flux qubits. Comprehensive statistical analysis of the data produced by these quantum machines shows that quantum annealers behave as samplers that generate independent configurations from low-temperature noisy Gibbs distributions. We show that the structure of the output distribution probes the intrinsic physical properties of the quantum device such as effective temperature of individual qubits and magnitude of local qubit noise, which result in a non-linear response function and spurious interactions that are absent in the hardware implementation. We anticipate that our methodology will find widespread use in characterization of future generations of quantum annealers and other emerging analog computing devices.

</p>
</details>

<details><summary><b>Provable Benefits of Overparameterization in Model Compression: From Double Descent to Pruning Neural Networks</b>
<a href="https://arxiv.org/abs/2012.08749">arxiv:2012.08749</a>
&#x1F4C8; 3 <br>
<p>Xiangyu Chang, Yingcong Li, Samet Oymak, Christos Thrampoulidis</p></summary>
<p>

**Abstract:** Deep networks are typically trained with many more parameters than the size of the training dataset. Recent empirical evidence indicates that the practice of overparameterization not only benefits training large models, but also assists - perhaps counterintuitively - building lightweight models. Specifically, it suggests that overparameterization benefits model pruning / sparsification. This paper sheds light on these empirical findings by theoretically characterizing the high-dimensional asymptotics of model pruning in the overparameterized regime. The theory presented addresses the following core question: "should one train a small model from the beginning, or first train a large model and then prune?". We analytically identify regimes in which, even if the location of the most informative features is known, we are better off fitting a large model and then pruning rather than simply training with the known informative features. This leads to a new double descent in the training of sparse models: growing the original model, while preserving the target sparsity, improves the test accuracy as one moves beyond the overparameterization threshold. Our analysis further reveals the benefit of retraining by relating it to feature correlations. We find that the above phenomena are already present in linear and random-features models. Our technical approach advances the toolset of high-dimensional analysis and precisely characterizes the asymptotic distribution of over-parameterized least-squares. The intuition gained by analytically studying simpler models is numerically verified on neural networks.

</p>
</details>

<details><summary><b>Deep Learning of Cell Classification using Microscope Images of Intracellular Microtubule Networks</b>
<a href="https://arxiv.org/abs/2012.12125">arxiv:2012.12125</a>
&#x1F4C8; 2 <br>
<p>Aleksei Shpilman, Dmitry Boikiy, Marina Polyakova, Daniel Kudenko, Anton Burakov, Elena Nadezhdina</p></summary>
<p>

**Abstract:** Microtubule networks (MTs) are a component of a cell that may indicate the presence of various chemical compounds and can be used to recognize properties such as treatment resistance. Therefore, the classification of MT images is of great relevance for cell diagnostics. Human experts find it particularly difficult to recognize the levels of chemical compound exposure of a cell. Improving the accuracy with automated techniques would have a significant impact on cell therapy. In this paper we present the application of Deep Learning to MT image classification and evaluate it on a large MT image dataset of animal cells with three degrees of exposure to a chemical agent. The results demonstrate that the learned deep network performs on par or better at the corresponding cell classification task than human experts. Specifically, we show that the task of recognizing different levels of chemical agent exposure can be handled significantly better by the neural network than by human experts.

</p>
</details>

<details><summary><b>Non-autoregressive electron flow generation for reaction prediction</b>
<a href="https://arxiv.org/abs/2012.12124">arxiv:2012.12124</a>
&#x1F4C8; 2 <br>
<p>Hangrui Bi, Hengyi Wang, Chence Shi, Jian Tang</p></summary>
<p>

**Abstract:** Reaction prediction is a fundamental problem in computational chemistry. Existing approaches typically generate a chemical reaction by sampling tokens or graph edits sequentially, conditioning on previously generated outputs. These autoregressive generating methods impose an arbitrary ordering of outputs and prevent parallel decoding during inference. We devise a novel decoder that avoids such sequential generating and predicts the reaction in a Non-Autoregressive manner. Inspired by physical-chemistry insights, we represent edge edits in a molecule graph as electron flows, which can then be predicted in parallel. To capture the uncertainty of reactions, we introduce latent variables to generate multi-modal outputs. Following previous works, we evaluate our model on USPTO MIT dataset. Our model achieves both an order of magnitude lower inference latency, with state-of-the-art top-1 accuracy and comparable performance on Top-K sampling.

</p>
</details>

<details><summary><b>Automatic source localization and spectra generation from deconvolved beamforming maps</b>
<a href="https://arxiv.org/abs/2012.09643">arxiv:2012.09643</a>
&#x1F4C8; 2 <br>
<p>Armin Goudarzi, Carsten Spehr, Steffen Herbold</p></summary>
<p>

**Abstract:** We present two methods for the automated detection of aeroacoustic source positions in deconvolved beamforming maps and the extraction of their corresponding spectra. We evaluate these methods on two scaled airframe half-model wind-tunnel measurements. The first relies on the spatial normal distribution of aeroacoustic broadband sources in CLEAN-SC maps. The second uses hierarchical clustering methods. Both methods predict a spatial probability estimation based on which aeroacoustic spectra are generated.

</p>
</details>

<details><summary><b>Balancing Geometry and Density: Path Distances on High-Dimensional Data</b>
<a href="https://arxiv.org/abs/2012.09385">arxiv:2012.09385</a>
&#x1F4C8; 2 <br>
<p>Anna Little, Daniel McKenzie, James Murphy</p></summary>
<p>

**Abstract:** New geometric and computational analyses of power-weighted shortest-path distances (PWSPDs) are presented. By illuminating the way these metrics balance density and geometry in the underlying data, we clarify their key parameters and discuss how they may be chosen in practice. Comparisons are made with related data-driven metrics, which illustrate the broader role of density in kernel-based unsupervised and semi-supervised machine learning. Computationally, we relate PWSPDs on complete weighted graphs to their analogues on weighted nearest neighbor graphs, providing high probability guarantees on their equivalence that are near-optimal. Connections with percolation theory are developed to establish estimates on the bias and variance of PWSPDs in the finite sample setting. The theoretical results are bolstered by illustrative experiments, demonstrating the versatility of PWSPDs for a wide range of data settings. Throughout the paper, our results require only that the underlying data is sampled from a low-dimensional manifold, and depend crucially on the intrinsic dimension of this manifold, rather than its ambient dimension.

</p>
</details>

<details><summary><b>Speech Enhancement with Zero-Shot Model Selection</b>
<a href="https://arxiv.org/abs/2012.09359">arxiv:2012.09359</a>
&#x1F4C8; 2 <br>
<p>Ryandhimas E. Zezario, Chiou-Shann Fuh, Hsin-Min Wang, Yu Tsao</p></summary>
<p>

**Abstract:** Recent research on speech enhancement (SE) has seen the emergence of deep learning-based methods. It is still a challenging task to determine effective ways to increase the generalizability of SE under diverse test conditions. In this paper, we combine zero-shot learning and ensemble learning to propose a zero-shot model selection (ZMOS) approach to increase the generalization of SE performance. The proposed approach is realized in two phases, namely offline and online phases. The offline phase clusters the entire set of training data into multiple subsets, and trains a specialized SE model (termed component SE model) with each subset. The online phase selects the most suitable component SE model to carry out enhancement. Two selection strategies are developed: selection based on quality score (QS) and selection based on quality embedding (QE). Both QS and QE are obtained by a Quality-Net, a non-intrusive quality assessment network. In the offline phase, the QS or QE of a train-ing utterance is used to group the training data into clusters. In the online phase, the QS or QE of the test utterance is used to identify the appropriate component SE model to perform enhancement on the test utterance. Experimental results have confirmed that the proposed ZMOS approach can achieve better performance in both seen and unseen noise types compared to the baseline systems, which indicates the effectiveness of the proposed approach to provide robust SE performance.

</p>
</details>

<details><summary><b>Measuring Disentanglement: A Review of Metrics</b>
<a href="https://arxiv.org/abs/2012.09276">arxiv:2012.09276</a>
&#x1F4C8; 2 <br>
<p>Julian Zaidi, Jonathan Boilard, Ghyslain Gagnon, Marc-André Carbonneau</p></summary>
<p>

**Abstract:** Learning to disentangle and represent factors of variation in data is an important problem in AI. While many advances are made to learn these representations, it is still unclear how to quantify disentanglement. Several metrics exist, however little is known on their implicit assumptions, what they truly measure and their limits. As a result, it is difficult to interpret results when comparing different representations. In this work, we survey supervised disentanglement metrics and thoroughly analyze them. We propose a new taxonomy in which all metrics fall into one of three families: intervention-based, predictor-based and information-based. We conduct extensive experiments, where we isolate representation properties to compare all metrics on many aspects. From experiment results and analysis, we provide insights on relations between disentangled representation properties. Finally, we provide guidelines on how to measure disentanglement and report the results.

</p>
</details>

<details><summary><b>Transfer Learning Through Weighted Loss Function and Group Normalization for Vessel Segmentation from Retinal Images</b>
<a href="https://arxiv.org/abs/2012.09250">arxiv:2012.09250</a>
&#x1F4C8; 2 <br>
<p>Abdullah Sarhan, Jon Rokne, Reda Alhajj, Andrew Crichton</p></summary>
<p>

**Abstract:** The vascular structure of blood vessels is important in diagnosing retinal conditions such as glaucoma and diabetic retinopathy. Accurate segmentation of these vessels can help in detecting retinal objects such as the optic disc and optic cup and hence determine if there are damages to these areas. Moreover, the structure of the vessels can help in diagnosing glaucoma. The rapid development of digital imaging and computer-vision techniques has increased the potential for developing approaches for segmenting retinal vessels. In this paper, we propose an approach for segmenting retinal vessels that uses deep learning along with transfer learning. We adapted the U-Net structure to use a customized InceptionV3 as the encoder and used multiple skip connections to form the decoder. Moreover, we used a weighted loss function to handle the issue of class imbalance in retinal images. Furthermore, we contributed a new dataset to this field. We tested our approach on six publicly available datasets and a newly created dataset. We achieved an average accuracy of 95.60% and a Dice coefficient of 80.98%. The results obtained from comprehensive experiments demonstrate the robustness of our approach to the segmentation of blood vessels in retinal images obtained from different sources. Our approach results in greater segmentation accuracy than other approaches.

</p>
</details>

<details><summary><b>Learning-NUM: Network Utility Maximization with Unknown Utility Functions and Queueing Delay</b>
<a href="https://arxiv.org/abs/2012.09222">arxiv:2012.09222</a>
&#x1F4C8; 2 <br>
<p>Xinzhe Fu, Eytan Modiano</p></summary>
<p>

**Abstract:** Network Utility Maximization (NUM) studies the problems of allocating traffic rates to network users in order to maximize the users' total utility subject to network resource constraints. In this paper, we propose a new NUM framework, Learning-NUM, where the users' utility functions are unknown apriori and the utility function values of the traffic rates can be observed only after the corresponding traffic is delivered to the destination, which means that the utility feedback experiences \textit{queueing delay}.
  The goal is to design a policy that gradually learns the utility functions and makes rate allocation and network scheduling/routing decisions so as to maximize the total utility obtained over a finite time horizon $T$. In addition to unknown utility functions and stochastic constraints, a central challenge of our problem lies in the queueing delay of the observations, which may be unbounded and depends on the decisions of the policy.
  We first show that the expected total utility obtained by the best dynamic policy is upper bounded by the solution to a static optimization problem. Without the presence of feedback delay, we design an algorithm based on the ideas of gradient estimation and Max-Weight scheduling. To handle the feedback delay, we embed the algorithm in a parallel-instance paradigm to form a policy that achieves $\tilde{O}(T^{3/4})$-regret, i.e., the difference between the expected utility obtained by the best dynamic policy and our policy is in $\tilde{O}(T^{3/4})$. Finally, to demonstrate the practical applicability of the Learning-NUM framework, we apply it to three application scenarios including database query, job scheduling and video streaming. We further conduct simulations on the job scheduling application to evaluate the empirical performance of our policy.

</p>
</details>

<details><summary><b>LIREx: Augmenting Language Inference with Relevant Explanation</b>
<a href="https://arxiv.org/abs/2012.09157">arxiv:2012.09157</a>
&#x1F4C8; 2 <br>
<p>Xinyan Zhao, V. G. Vinod Vydiswaran</p></summary>
<p>

**Abstract:** Natural language explanations (NLEs) are a special form of data annotation in which annotators identify rationales (most significant text tokens) when assigning labels to data instances, and write out explanations for the labels in natural language based on the rationales. NLEs have been shown to capture human reasoning better, but not as beneficial for natural language inference (NLI). In this paper, we analyze two primary flaws in the way NLEs are currently used to train explanation generators for language inference tasks. We find that the explanation generators do not take into account the variability inherent in human explanation of labels, and that the current explanation generation models generate spurious explanations. To overcome these limitations, we propose a novel framework, LIREx, that incorporates both a rationale-enabled explanation generator and an instance selector to select only relevant, plausible NLEs to augment NLI models. When evaluated on the standardized SNLI data set, LIREx achieved an accuracy of 91.87%, an improvement of 0.32 over the baseline and matching the best-reported performance on the data set. It also achieves significantly better performance than previous studies when transferred to the out-of-domain MultiNLI data set. Qualitative analysis shows that LIREx generates flexible, faithful, and relevant NLEs that allow the model to be more robust to spurious explanations. The code is available at https://github.com/zhaoxy92/LIREx.

</p>
</details>

<details><summary><b>Planning From Pixels in Atari With Learned Symbolic Representations</b>
<a href="https://arxiv.org/abs/2012.09126">arxiv:2012.09126</a>
&#x1F4C8; 2 <br>
<p>Andrea Dittadi, Frederik K. Drachmann, Thomas Bolander</p></summary>
<p>

**Abstract:** Width-based planning methods have been shown to yield state-of-the-art performance in the Atari 2600 domain using pixel input. One successful approach, RolloutIW, represents states with the B-PROST boolean feature set. An augmented version of RolloutIW, $π$-IW, shows that learned features can be competitive with handcrafted ones for width-based search. In this paper, we leverage variational autoencoders (VAEs) to learn features directly from pixels in a principled manner, and without supervision. The inference model of the trained VAEs extracts boolean features from pixels, and RolloutIW plans with these features. The resulting combination outperforms the original RolloutIW and human professional play on Atari 2600 and drastically reduces the size of the feature set.

</p>
</details>

<details><summary><b>Building and Using Personal Knowledge Graph to Improve Suicidal Ideation Detection on Social Media</b>
<a href="https://arxiv.org/abs/2012.09123">arxiv:2012.09123</a>
&#x1F4C8; 2 <br>
<p>Lei Cao, Huijun Zhang, Ling Feng</p></summary>
<p>

**Abstract:** A large number of individuals are suffering from suicidal ideation in the world. There are a number of causes behind why an individual might suffer from suicidal ideation. As the most popular platform for self-expression, emotion release, and personal interaction, individuals may exhibit a number of symptoms of suicidal ideation on social media. Nevertheless, challenges from both data and knowledge aspects remain as obstacles, constraining the social media-based detection performance. Data implicitness and sparsity make it difficult to discover the inner true intentions of individuals based on their posts. Inspired by psychological studies, we build and unify a high-level suicide-oriented knowledge graph with deep neural networks for suicidal ideation detection on social media. We further design a two-layered attention mechanism to explicitly reason and establish key risk factors to individual's suicidal ideation. The performance study on microblog and Reddit shows that: 1) with the constructed personal knowledge graph, the social media-based suicidal ideation detection can achieve over 93% accuracy; and 2) among the six categories of personal factors, post, personality, and experience are the top-3 key indicators. Under these categories, posted text, stress level, stress duration, posted image, and ruminant thinking contribute to one's suicidal ideation detection.

</p>
</details>

<details><summary><b>On Avoiding the Union Bound When Answering Multiple Differentially Private Queries</b>
<a href="https://arxiv.org/abs/2012.09116">arxiv:2012.09116</a>
&#x1F4C8; 2 <br>
<p>Badih Ghazi, Ravi Kumar, Pasin Manurangsi</p></summary>
<p>

**Abstract:** In this work, we study the problem of answering $k$ queries with $(ε, δ)$-differential privacy, where each query has sensitivity one. We give an algorithm for this task that achieves an expected $\ell_\infty$ error bound of $O(\frac{1}ε\sqrt{k \log \frac{1}δ})$, which is known to be tight (Steinke and Ullman, 2016).
  A very recent work by Dagan and Kur (2020) provides a similar result, albeit via a completely different approach. One difference between our work and theirs is that our guarantee holds even when $δ< 2^{-Ω(k/(\log k)^8)}$ whereas theirs does not apply in this case. On the other hand, the algorithm of Dagan and Kur has a remarkable advantage that the $\ell_{\infty}$ error bound of $O(\frac{1}ε\sqrt{k \log \frac{1}δ})$ holds not only in expectation but always (i.e., with probability one) while we can only get a high probability (or expected) guarantee on the error.

</p>
</details>

<details><summary><b>TEMImageNet and AtomSegNet Deep Learning Training Library and Models for High-Precision Atom Segmentation, Localization, Denoising, and Super-resolution Processing of Atom-Resolution Scanning TEM Images</b>
<a href="https://arxiv.org/abs/2012.09093">arxiv:2012.09093</a>
&#x1F4C8; 2 <br>
<p>Ruoqian Lin, Rui Zhang, Chunyang Wang, Xiao-Qing Yang, Huolin L. Xin</p></summary>
<p>

**Abstract:** Atom segmentation and localization, noise reduction and super-resolution processing of atomic-resolution scanning transmission electron microscopy (STEM) images with high precision and robustness is a challenging task. Although several conventional algorithms, such has thresholding, edge detection and clustering, can achieve reasonable performance in some predefined sceneries, they tend to fail when interferences from the background are strong and unpredictable. Particularly, for atomic-resolution STEM images, so far there is no well-established algorithm that is robust enough to segment or detect all atomic columns when there is large thickness variation in a recorded image. Herein, we report the development of a training library and a deep learning method that can perform robust and precise atom segmentation, localization, denoising, and super-resolution processing of experimental images. Despite using simulated images as training datasets, the deep-learning model can self-adapt to experimental STEM images and shows outstanding performance in atom detection and localization in challenging contrast conditions and the precision is consistently better than the state-of-the-art two-dimensional Gaussian fit method. Taking a step further, we have deployed our deep-learning models to a desktop app with a graphical user interface and the app is free and open-source. We have also built a TEM ImageNet project website for easy browsing and downloading of the training data.

</p>
</details>

<details><summary><b>Investigating ADR mechanisms with knowledge graph mining and explainable AI</b>
<a href="https://arxiv.org/abs/2012.09077">arxiv:2012.09077</a>
&#x1F4C8; 2 <br>
<p>Emmanuel Bresso, Pierre Monnin, Cédric Bousquet, François-Elie Calvier, Ndeye-Coumba Ndiaye, Nadine Petitpain, Malika Smaïl-Tabbone, Adrien Coulet</p></summary>
<p>

**Abstract:** Adverse Drug Reactions (ADRs) are characterized within randomized clinical trials and postmarketing pharmacovigilance, but their molecular mechanism remains unknown in most cases. Aside from clinical trials, many elements of knowledge about drug ingredients are available in open-access knowledge graphs. In addition, drug classifications that label drugs as either causative or not for several ADRs, have been established. We propose to mine knowledge graphs for identifying biomolecular features that may enable reproducing automatically expert classifications that distinguish drug causative or not for a given type of ADR. In an explainable AI perspective, we explore simple classification techniques such as Decision Trees and Classification Rules because they provide human-readable models, which explain the classification itself, but may also provide elements of explanation for molecular mechanisms behind ADRs. In summary, we mine a knowledge graph for features; we train classifiers at distinguishing, drugs associated or not with ADRs; we isolate features that are both efficient in reproducing expert classifications and interpretable by experts (i.e., Gene Ontology terms, drug targets, or pathway names); and we manually evaluate how they may be explanatory. Extracted features reproduce with a good fidelity classifications of drugs causative or not for DILI and SCAR. Experts fully agreed that 73% and 38% of the most discriminative features are possibly explanatory for DILI and SCAR, respectively; and partially agreed (2/3) for 90% and 77% of them. Knowledge graphs provide diverse features to enable simple and explainable models to distinguish between drugs that are causative or not for ADRs. In addition to explaining classifications, most discriminative features appear to be good candidates for investigating ADR mechanisms further.

</p>
</details>

<details><summary><b>ColorShapeLinks: A Board Game AI Competition for Educators and Students</b>
<a href="https://arxiv.org/abs/2012.09015">arxiv:2012.09015</a>
&#x1F4C8; 2 <br>
<p>Nuno Fachada</p></summary>
<p>

**Abstract:** ColorShapeLinks is an AI board game competition framework specially designed for students and educators in videogame development, with openness and accessibility in mind. The competition is based on an arbitrarily-sized version of the Simplexity board game, the motto of which, "simple to learn, complex to master", is curiously also applicable to AI agents. ColorShapeLinks offers graphical and text-based frontends and a completely open and documented development framework built using industry standard tools and following software engineering best practices. ColorShapeLinks is not only a competition, but both a game and a framework which educators and students can extend and use to host their own competitions.

</p>
</details>

<details><summary><b>Discovering New Intents with Deep Aligned Clustering</b>
<a href="https://arxiv.org/abs/2012.08987">arxiv:2012.08987</a>
&#x1F4C8; 2 <br>
<p>Hanlei Zhang, Hua Xu, Ting-En Lin, Rui Lv</p></summary>
<p>

**Abstract:** Discovering new intents is a crucial task in a dialogue system. Most existing methods are limited in transferring the prior knowledge from known intents to new intents. These methods also have difficulties in providing high-quality supervised signals to learn clustering-friendly features for grouping unlabeled intents. In this work, we propose an effective method (Deep Aligned Clustering) to discover new intents with the aid of limited known intent data. Firstly, we leverage a few labeled known intent samples as prior knowledge to pre-train the model. Then, we perform k-means to produce cluster assignments as pseudo-labels. Moreover, we propose an alignment strategy to tackle the label inconsistency during clustering assignments. Finally, we learn the intent representations under the supervision of the aligned pseudo-labels. With an unknown number of new intents, we predict the number of intent categories by eliminating low-confidence intent-wise clusters. Extensive experiments on two benchmark datasets show that our method is more robust and achieves substantial improvements over the state-of-the-art methods.(Code available at https://github.com/hanleizhang/DeepAligned-Clustering)

</p>
</details>

<details><summary><b>AutoDis: Automatic Discretization for Embedding Numerical Features in CTR Prediction</b>
<a href="https://arxiv.org/abs/2012.08986">arxiv:2012.08986</a>
&#x1F4C8; 2 <br>
<p>Huifeng Guo, Bo Chen, Ruiming Tang, Zhenguo Li, Xiuqiang He</p></summary>
<p>

**Abstract:** Learning sophisticated feature interactions is crucial for Click-Through Rate (CTR) prediction in recommender systems. Various deep CTR models follow an Embedding & Feature Interaction paradigm. The majority focus on designing network architectures in Feature Interaction module to better model feature interactions while the Embedding module, serving as a bottleneck between data and Feature Interaction module, has been overlooked. The common methods for numerical feature embedding are Normalization and Discretization. The former shares a single embedding for intra-field features and the latter transforms the features into categorical form through various discretization approaches. However, the first approach surfers from low capacity and the second one limits performance as well because the discretization rule cannot be optimized with the ultimate goal of CTR model. To fill the gap of representing numerical features, in this paper, we propose AutoDis, a framework that discretizes features in numerical fields automatically and is optimized with CTR models in an end-to-end manner. Specifically, we introduce a set of meta-embeddings for each numerical field to model the relationship among the intra-field features and propose an automatic differentiable discretization and aggregation approach to capture the correlations between the numerical features and meta-embeddings. Comprehensive experiments on two public and one industrial datasets are conducted to validate the effectiveness of AutoDis over the SOTA methods.

</p>
</details>

<details><summary><b>Learning-Based Algorithms for Vessel Tracking: A Review</b>
<a href="https://arxiv.org/abs/2012.08929">arxiv:2012.08929</a>
&#x1F4C8; 2 <br>
<p>Dengqiang Jia, Xiahai Zhuang</p></summary>
<p>

**Abstract:** Developing efficient vessel-tracking algorithms is crucial for imaging-based diagnosis and treatment of vascular diseases. Vessel tracking aims to solve recognition problems such as key (seed) point detection, centerline extraction, and vascular segmentation. Extensive image-processing techniques have been developed to overcome the problems of vessel tracking that are mainly attributed to the complex morphologies of vessels and image characteristics of angiography. This paper presents a literature review on vessel-tracking methods, focusing on machine-learning-based methods. First, the conventional machine-learning-based algorithms are reviewed, and then, a general survey of deep-learning-based frameworks is provided. On the basis of the reviewed methods, the evaluation issues are introduced. The paper is concluded with discussions about the remaining exigencies and future research.

</p>
</details>

<details><summary><b>Unsupervised Image Segmentation using Mutual Mean-Teaching</b>
<a href="https://arxiv.org/abs/2012.08922">arxiv:2012.08922</a>
&#x1F4C8; 2 <br>
<p>Zhichao Wu, Lei Guo, Hao Zhang, Dan Xu</p></summary>
<p>

**Abstract:** Unsupervised image segmentation aims at assigning the pixels with similar feature into a same cluster without annotation, which is an important task in computer vision. Due to lack of prior knowledge, most of existing model usually need to be trained several times to obtain suitable results. To address this problem, we propose an unsupervised image segmentation model based on the Mutual Mean-Teaching (MMT) framework to produce more stable results. In addition, since the labels of pixels from two model are not matched, a label alignment algorithm based on the Hungarian algorithm is proposed to match the cluster labels. Experimental results demonstrate that the proposed model is able to segment various types of images and achieves better performance than the existing methods.

</p>
</details>

<details><summary><b>Clustering Ensemble Meets Low-rank Tensor Approximation</b>
<a href="https://arxiv.org/abs/2012.08916">arxiv:2012.08916</a>
&#x1F4C8; 2 <br>
<p>Yuheng Jia, Hui Liu, Junhui Hou, Qingfu Zhang</p></summary>
<p>

**Abstract:** This paper explores the problem of clustering ensemble, which aims to combine multiple base clusterings to produce better performance than that of the individual one. The existing clustering ensemble methods generally construct a co-association matrix, which indicates the pairwise similarity between samples, as the weighted linear combination of the connective matrices from different base clusterings, and the resulting co-association matrix is then adopted as the input of an off-the-shelf clustering algorithm, e.g., spectral clustering. However, the co-association matrix may be dominated by poor base clusterings, resulting in inferior performance. In this paper, we propose a novel low-rank tensor approximation-based method to solve the problem from a global perspective. Specifically, by inspecting whether two samples are clustered to an identical cluster under different base clusterings, we derive a coherent-link matrix, which contains limited but highly reliable relationships between samples. We then stack the coherent-link matrix and the co-association matrix to form a three-dimensional tensor, the low-rankness property of which is further explored to propagate the information of the coherent-link matrix to the co-association matrix, producing a refined co-association matrix. We formulate the proposed method as a convex constrained optimization problem and solve it efficiently. Experimental results over 7 benchmark data sets show that the proposed model achieves a breakthrough in clustering performance, compared with 12 state-of-the-art methods. To the best of our knowledge, this is the first work to explore the potential of low-rank tensor on clustering ensemble, which is fundamentally different from previous approaches.

</p>
</details>

<details><summary><b>Multi-Task Learning in Diffractive Deep Neural Networks via Hardware-Software Co-design</b>
<a href="https://arxiv.org/abs/2012.08906">arxiv:2012.08906</a>
&#x1F4C8; 2 <br>
<p>Yingjie Li, Ruiyang Chen, Berardi Sensale Rodriguez, Weilu Gao, Cunxi Yu</p></summary>
<p>

**Abstract:** Deep neural networks (DNNs) have substantial computational requirements, which greatly limit their performance in resource-constrained environments. Recently, there are increasing efforts on optical neural networks and optical computing based DNNs hardware, which bring significant advantages for deep learning systems in terms of their power efficiency, parallelism and computational speed. Among them, free-space diffractive deep neural networks (D$^2$NNs) based on the light diffraction, feature millions of neurons in each layer interconnected with neurons in neighboring layers. However, due to the challenge of implementing reconfigurability, deploying different DNNs algorithms requires re-building and duplicating the physical diffractive systems, which significantly degrades the hardware efficiency in practical application scenarios. Thus, this work proposes a novel hardware-software co-design method that enables robust and noise-resilient Multi-task Learning in D$^2$2NNs. Our experimental results demonstrate significant improvements in versatility and hardware efficiency, and also demonstrate the robustness of proposed multi-task D$^2$NN architecture under wide noise ranges of all system components. In addition, we propose a domain-specific regularization algorithm for training the proposed multi-task architecture, which can be used to flexibly adjust the desired performance for each task.

</p>
</details>

<details><summary><b>A connection between the pattern classification problem and the General Linear Model for statistical inference</b>
<a href="https://arxiv.org/abs/2012.08903">arxiv:2012.08903</a>
&#x1F4C8; 2 <br>
<p>Juan Manuel Gorriz, SIPBA group, John Suckling</p></summary>
<p>

**Abstract:** A connection between the General Linear Model (GLM) in combination with classical statistical inference and the machine learning (MLE)-based inference is described in this paper. Firstly, the estimation of the GLM parameters is expressed as a Linear Regression Model (LRM) of an indicator matrix, that is, in terms of the inverse problem of regressing the observations. In other words, both approaches, i.e. GLM and LRM, apply to different domains, the observation and the label domains, and are linked by a normalization value at the least-squares solution. Subsequently, from this relationship we derive a statistical test based on a more refined predictive algorithm, i.e. the (non)linear Support Vector Machine (SVM) that maximizes the class margin of separation, within a permutation analysis. The MLE-based inference employs a residual score and includes the upper bound to compute a better estimation of the actual (real) error. Experimental results demonstrate how the parameter estimations derived from each model resulted in different classification performances in the equivalent inverse problem. Moreover, using real data the aforementioned predictive algorithms within permutation tests, including such model-free estimators, are able to provide a good trade-off between type I error and statistical power.

</p>
</details>

<details><summary><b>Lévy walks derived from a Bayesian decision-making model in non-stationary environments</b>
<a href="https://arxiv.org/abs/2012.08858">arxiv:2012.08858</a>
&#x1F4C8; 2 <br>
<p>Shuji Shinohara, Nobuhito Manome, Yoshihiro Nakajima, Yukio Pegio Gunji, Toru Moriyama, Hiroshi Okamoto, Shunji Mitsuyoshi, Ung-il Chung</p></summary>
<p>

**Abstract:** Lévy walks are found in the migratory behaviour patterns of various organisms, and the reason for this phenomenon has been much discussed. We use simulations to demonstrate that learning causes the changes in confidence level during decision-making in non-stationary environments, and results in Lévy-walk-like patterns. One inference algorithm involving confidence is Bayesian inference. We propose an algorithm that introduces the effects of learning and forgetting into Bayesian inference, and simulate an imitation game in which two decision-making agents incorporating the algorithm estimate each other's internal models from their opponent's observational data. For forgetting without learning, agent confidence levels remained low due to a lack of information on the counterpart and Brownian walks occurred for a wide range of forgetting rates. Conversely, when learning was introduced, high confidence levels occasionally occurred even at high forgetting rates, and Brownian walks universally became Lévy walks through a mixture of high- and low-confidence states.

</p>
</details>

<details><summary><b>Time-Aware Tensor Decomposition for Missing Entry Prediction</b>
<a href="https://arxiv.org/abs/2012.08855">arxiv:2012.08855</a>
&#x1F4C8; 2 <br>
<p>Dawon Ahn, Jun-Gi Jang, U Kang</p></summary>
<p>

**Abstract:** Given a time-evolving tensor with missing entries, how can we effectively factorize it for precisely predicting the missing entries? Tensor factorization has been extensively utilized for analyzing various multi-dimensional real-world data. However, existing models for tensor factorization have disregarded the temporal property for tensor factorization while most real-world data are closely related to time. Moreover, they do not address accuracy degradation due to the sparsity of time slices. The essential problems of how to exploit the temporal property for tensor decomposition and consider the sparsity of time slices remain unresolved. In this paper, we propose TATD (Time-Aware Tensor Decomposition), a novel tensor decomposition method for real-world temporal tensors. TATD is designed to exploit temporal dependency and time-varying sparsity of real-world temporal tensors. We propose a new smoothing regularization with Gaussian kernel for modeling time dependency. Moreover, we improve the performance of TATD by considering time-varying sparsity. We design an alternating optimization scheme suitable for temporal tensor factorization with our smoothing regularization. Extensive experiments show that TATD provides the state-of-the-art accuracy for decomposing temporal tensors.

</p>
</details>

<details><summary><b>Using noise resilience for ranking generalization of deep neural networks</b>
<a href="https://arxiv.org/abs/2012.08854">arxiv:2012.08854</a>
&#x1F4C8; 2 <br>
<p>Depen Morwani, Rahul Vashisht, Harish G. Ramaswamy</p></summary>
<p>

**Abstract:** Recent papers have shown that sufficiently overparameterized neural networks can perfectly fit even random labels. Thus, it is crucial to understand the underlying reason behind the generalization performance of a network on real-world data. In this work, we propose several measures to predict the generalization error of a network given the training data and its parameters. Using one of these measures, based on noise resilience of the network, we secured 5th position in the predicting generalization in deep learning (PGDL) competition at NeurIPS 2020.

</p>
</details>

<details><summary><b>More Industry-friendly: Federated Learning with High Efficient Design</b>
<a href="https://arxiv.org/abs/2012.08809">arxiv:2012.08809</a>
&#x1F4C8; 2 <br>
<p>Dingwei Li, Qinglong Chang, Lixue Pang, Yanfang Zhang, Xudong Sun, Jikun Ding, Liang Zhang</p></summary>
<p>

**Abstract:** Although many achievements have been made since Google threw out the paradigm of federated learning (FL), there still exists much room for researchers to optimize its efficiency. In this paper, we propose a high efficient FL method equipped with the double head design aiming for personalization optimization over non-IID dataset, and the gradual model sharing design for communication saving. Experimental results show that, our method has more stable accuracy performance and better communication efficient across various data distributions than other state of art methods (SOTAs), makes it more industry-friendly.

</p>
</details>

<details><summary><b>Study on the Large Batch Size Training of Neural Networks Based on the Second Order Gradient</b>
<a href="https://arxiv.org/abs/2012.08795">arxiv:2012.08795</a>
&#x1F4C8; 2 <br>
<p>Fengli Gao, Huicai Zhong</p></summary>
<p>

**Abstract:** Large batch size training in deep neural networks (DNNs) possesses a well-known 'generalization gap' that remarkably induces generalization performance degradation. However, it remains unclear how varying batch size affects the structure of a NN. Here, we combine theory with experiments to explore the evolution of the basic structural properties, including gradient, parameter update step length, and loss update step length of NNs under varying batch sizes. We provide new guidance to improve generalization, which is further verified by two designed methods involving discarding small-loss samples and scheduling batch size. A curvature-based learning rate (CBLR) algorithm is proposed to better fit the curvature variation, a sensitive factor affecting large batch size training, across layers in a NN. As an approximation of CBLR, the median-curvature LR (MCLR) algorithm is found to gain comparable performance to Layer-wise Adaptive Rate Scaling (LARS) algorithm. Our theoretical results and algorithm offer geometry-based explanations to the existing studies. Furthermore, we demonstrate that the layer wise LR algorithms, for example LARS, can be regarded as special instances of CBLR. Finally, we deduce a theoretical geometric picture of large batch size training, and show that all the network parameters tend to center on their related minima.

</p>
</details>

<details><summary><b>Building domain specific lexicon based on TikTok comment dataset</b>
<a href="https://arxiv.org/abs/2012.08773">arxiv:2012.08773</a>
&#x1F4C8; 2 <br>
<p>Hao Jiaxiang</p></summary>
<p>

**Abstract:** In the sentiment analysis task, predicting the sentiment tendency of a sentence is an important branch. Previous research focused more on sentiment analysis in English, for example, analyzing the sentiment tendency of sentences based on Valence, Arousal, Dominance of sentences. the emotional tendency is different between the two languages. For example, the sentence order between Chinese and English may present different emotions. This paper tried a method that builds a domain-specific lexicon. In this way, the model can classify Chinese words with emotional tendency. In this approach, based on the [13], an ultra-dense space embedding table is trained through word embedding of Chinese TikTok review and emotional lexicon sources(seed words). The result of the model is a domain-specific lexicon, which presents the emotional tendency of words. I collected Chinese TikTok comments as training data. By comparing The training results with the PCA method to evaluate the performance of the model in Chinese sentiment classification, the results show that the model has done well in Chinese. The source code has released on github:https://github.com/h2222/douyin_comment_dataset

</p>
</details>

<details><summary><b>Ensemble model for pre-discharge icd10 coding prediction</b>
<a href="https://arxiv.org/abs/2012.11333">arxiv:2012.11333</a>
&#x1F4C8; 1 <br>
<p>Yassien Shaalan, Alexander Dokumentov, Piyapong Khumrin, Krit Khwanngern, Anawat Wisetborisu, Thanakom Hatsadeang, Nattapat Karaket, Witthawin Achariyaviriya, Sansanee Auephanwiriyakul, Nipon Theera-Umpon, Terence Siganakis</p></summary>
<p>

**Abstract:** The translation of medical diagnosis to clinical coding has wide range of applications in billing, aetiology analysis, and auditing. Currently, coding is a manual effort while the automation of such task is not straight forward. Among the challenges are the messy and noisy clinical records, case complexities, along with the huge ICD10 code space. Previous work mainly relied on discharge notes for prediction and was applied to a very limited data scale. We propose an ensemble model incorporating multiple clinical data sources for accurate code predictions. We further propose an assessment mechanism to provide confidence rates in predicted outcomes. Extensive experiments were performed on two new real-world clinical datasets (inpatient & outpatient) with unaltered case-mix distributions from Maharaj Nakorn Chiang Mai Hospital. We obtain multi-label classification accuracies of 0.73 and 0.58 for average precision, 0.56 and 0.35 for F1-scores and 0.71 and 0.4 accuracy in predicting principal diagnosis for inpatient and outpatient datasets respectively.

</p>
</details>

<details><summary><b>CARLA Real Traffic Scenarios -- novel training ground and benchmark for autonomous driving</b>
<a href="https://arxiv.org/abs/2012.11329">arxiv:2012.11329</a>
&#x1F4C8; 1 <br>
<p>Błażej Osiński, Piotr Miłoś, Adam Jakubowski, Paweł Zięcina, Michał Martyniak, Christopher Galias, Antonia Breuer, Silviu Homoceanu, Henryk Michalewski</p></summary>
<p>

**Abstract:** This work introduces interactive traffic scenarios in the CARLA simulator, which are based on real-world traffic. We concentrate on tactical tasks lasting several seconds, which are especially challenging for current control methods. The CARLA Real Traffic Scenarios (CRTS) is intended to be a training and testing ground for autonomous driving systems. To this end, we open-source the code under a permissive license and present a set of baseline policies. CRTS combines the realism of traffic scenarios and the flexibility of simulation. We use it to train agents using a reinforcement learning algorithm. We show how to obtain competitive polices and evaluate experimentally how observation types and reward schemes affect the training process and the resulting agent's behavior.

</p>
</details>

<details><summary><b>Collaborative residual learners for automatic icd10 prediction using prescribed medications</b>
<a href="https://arxiv.org/abs/2012.11327">arxiv:2012.11327</a>
&#x1F4C8; 1 <br>
<p>Yassien Shaalan, Alexander Dokumentov, Piyapong Khumrin, Krit Khwanngern, Anawat Wisetborisu, Thanakom Hatsadeang, Nattapat Karaket, Witthawin Achariyaviriya, Sansanee Auephanwiriyakul, Nipon Theera-Umpon, Terence Siganakis</p></summary>
<p>

**Abstract:** Clinical coding is an administrative process that involves the translation of diagnostic data from episodes of care into a standard code format such as ICD10. It has many critical applications such as billing and aetiology research. The automation of clinical coding is very challenging due to data sparsity, low interoperability of digital health systems, complexity of real-life diagnosis coupled with the huge size of ICD10 code space. Related work suffer from low applicability due to reliance on many data sources, inefficient modelling and less generalizable solutions. We propose a novel collaborative residual learning based model to automatically predict ICD10 codes employing only prescriptions data. Extensive experiments were performed on two real-world clinical datasets (outpatient & inpatient) from Maharaj Nakorn Chiang Mai Hospital with real case-mix distributions. We obtain multi-label classification accuracy of 0.71 and 0.57 of average precision, 0.57 and 0.38 of F1-score and 0.73 and 0.44 of accuracy in predicting principal diagnosis for inpatient and outpatient datasets respectively.

</p>
</details>

<details><summary><b>Optimized Random Forest Model for Botnet Detection Based on DNS Queries</b>
<a href="https://arxiv.org/abs/2012.11326">arxiv:2012.11326</a>
&#x1F4C8; 1 <br>
<p>Abdallah Moubayed, MohammadNoor Injadat, Abdallah Shami</p></summary>
<p>

**Abstract:** The Domain Name System (DNS) protocol plays a major role in today's Internet as it translates between website names and corresponding IP addresses. However, due to the lack of processes for data integrity and origin authentication, the DNS protocol has several security vulnerabilities. This often leads to a variety of cyber-attacks, including botnet network attacks. One promising solution to detect DNS-based botnet attacks is adopting machine learning (ML) based solutions. To that end, this paper proposes a novel optimized ML-based framework to detect botnets based on their corresponding DNS queries. More specifically, the framework consists of using information gain as a feature selection method and genetic algorithm (GA) as a hyper-parameter optimization model to tune the parameters of a random forest (RF) classifier. The proposed framework is evaluated using a state-of-the-art TI-2016 DNS dataset. Experimental results show that the proposed optimized framework reduced the feature set size by up to 60%. Moreover, it achieved a high detection accuracy, precision, recall, and F-score compared to the default classifier. This highlights the effectiveness and robustness of the proposed framework in detecting botnet attacks.

</p>
</details>

<details><summary><b>Detecting Botnet Attacks in IoT Environments: An Optimized Machine Learning Approach</b>
<a href="https://arxiv.org/abs/2012.11325">arxiv:2012.11325</a>
&#x1F4C8; 1 <br>
<p>MohammadNoor Injadat, Abdallah Moubayed, Abdallah Shami</p></summary>
<p>

**Abstract:** The increased reliance on the Internet and the corresponding surge in connectivity demand has led to a significant growth in Internet-of-Things (IoT) devices. The continued deployment of IoT devices has in turn led to an increase in network attacks due to the larger number of potential attack surfaces as illustrated by the recent reports that IoT malware attacks increased by 215.7% from 10.3 million in 2017 to 32.7 million in 2018. This illustrates the increased vulnerability and susceptibility of IoT devices and networks. Therefore, there is a need for proper effective and efficient attack detection and mitigation techniques in such environments. Machine learning (ML) has emerged as one potential solution due to the abundance of data generated and available for IoT devices and networks. Hence, they have significant potential to be adopted for intrusion detection for IoT environments. To that end, this paper proposes an optimized ML-based framework consisting of a combination of Bayesian optimization Gaussian Process (BO-GP) algorithm and decision tree (DT) classification model to detect attacks on IoT devices in an effective and efficient manner. The performance of the proposed framework is evaluated using the Bot-IoT-2018 dataset. Experimental results show that the proposed optimized framework has a high detection accuracy, precision, recall, and F-score, highlighting its effectiveness and robustness for the detection of botnet attacks in IoT environments.

</p>
</details>

<details><summary><b>LiveMap: Real-Time Dynamic Map in Automotive Edge Computing</b>
<a href="https://arxiv.org/abs/2012.10252">arxiv:2012.10252</a>
&#x1F4C8; 1 <br>
<p>Qiang Liu, Tao Han,  Jiang,  Xie, BaekGyu Kim</p></summary>
<p>

**Abstract:** Autonomous driving needs various line-of-sight sensors to perceive surroundings that could be impaired under diverse environment uncertainties such as visual occlusion and extreme weather. To improve driving safety, we explore to wirelessly share perception information among connected vehicles within automotive edge computing networks. Sharing massive perception data in real time, however, is challenging under dynamic networking conditions and varying computation workloads. In this paper, we propose LiveMap, a real-time dynamic map, that detects, matches, and tracks objects on the road with crowdsourcing data from connected vehicles in sub-second. We develop the data plane of LiveMap that efficiently processes individual vehicle data with object detection, projection, feature extraction, object matching, and effectively integrates objects from multiple vehicles with object combination. We design the control plane of LiveMap that allows adaptive offloading of vehicle computations, and develop an intelligent vehicle scheduling and offloading algorithm to reduce the offloading latency of vehicles based on deep reinforcement learning (DRL) techniques. We implement LiveMap on a small-scale testbed and develop a large-scale network simulator. We evaluate the performance of LiveMap with both experiments and simulations, and the results show LiveMap reduces 34.1% average latency than the baseline solution.

</p>
</details>

<details><summary><b>Computational discovery of new 2D materials using deep learning generative models</b>
<a href="https://arxiv.org/abs/2012.09314">arxiv:2012.09314</a>
&#x1F4C8; 1 <br>
<p>Yuqi Song, Edirisuriya M. Dilanga Siriwardane, Yong Zhao, Jianjun Hu</p></summary>
<p>

**Abstract:** Two dimensional (2D) materials have emerged as promising functional materials with many applications such as semiconductors and photovoltaics because of their unique optoelectronic properties. While several thousand 2D materials have been screened in existing materials databases, discovering new 2D materials remains to be challenging. Herein we propose a deep learning generative model for composition generation combined with random forest based 2D materials classifier to discover new hypothetical 2D materials. Furthermore, a template based element substitution structure prediction approach is developed to predict the crystal structures of a subset of the newly predicted hypothetical formulas, which allows us to confirm their structure stability using DFT calculations. So far, we have discovered 267,489 new potential 2D materials compositions and confirmed twelve 2D/layered materials by DFT formation energy calculation. Our results show that generative machine learning models provide an effective way to explore the vast chemical design space for new 2D materials discovery.

</p>
</details>

<details><summary><b>Infrastructure for Artificial Intelligence, Quantum and High Performance Computing</b>
<a href="https://arxiv.org/abs/2012.09303">arxiv:2012.09303</a>
&#x1F4C8; 1 <br>
<p>William Gropp, Sujata Banerjee, Ian Foster</p></summary>
<p>

**Abstract:** High Performance Computing (HPC), Artificial Intelligence (AI)/Machine Learning (ML), and Quantum Computing (QC) and communications offer immense opportunities for innovation and impact on society. Researchers in these areas depend on access to computing infrastructure, but these resources are in short supply and are typically siloed in support of their research communities, making it more difficult to pursue convergent and interdisciplinary research. Such research increasingly depends on complex workflows that require different resources for each stage. This paper argues that a more-holistic approach to computing infrastructure, one that recognizes both the convergence of some capabilities and the complementary capabilities from new computing approaches, be it commercial cloud to Quantum Computing, is needed to support computer science research.

</p>
</details>

<details><summary><b>Detection of data drift and outliers affecting machine learning model performance over time</b>
<a href="https://arxiv.org/abs/2012.09258">arxiv:2012.09258</a>
&#x1F4C8; 1 <br>
<p>Samuel Ackerman, Eitan Farchi, Orna Raz, Marcel Zalmanovici, Parijat Dube</p></summary>
<p>

**Abstract:** A trained ML model is deployed on another `test' dataset where target feature values (labels) are unknown. Drift is distribution change between the training and deployment data, which is concerning if model performance changes. For a cat/dog image classifier, for instance, drift during deployment could be rabbit images (new class) or cat/dog images with changed characteristics (change in distribution). We wish to detect these changes but can't measure accuracy without deployment data labels. We instead detect drift indirectly by nonparametrically testing the distribution of model prediction confidence for changes. This generalizes our method and sidesteps domain-specific feature representation.
  We address important statistical issues, particularly Type-1 error control in sequential testing, using Change Point Models (CPMs; see Adams and Ross 2012). We also use nonparametric outlier methods to show the user suspicious observations for model diagnosis, since the before/after change confidence distributions overlap significantly. In experiments to demonstrate robustness, we train on a subset of MNIST digit classes, then insert drift (e.g., unseen digit class) in deployment data in various settings (gradual/sudden changes in the drift proportion). A novel loss function is introduced to compare the performance (detection delay, Type-1 and 2 errors) of a drift detector under different levels of drift class contamination.

</p>
</details>

<details><summary><b>An Assessment of the Usability of Machine Learning Based Tools for the Security Operations Center</b>
<a href="https://arxiv.org/abs/2012.09013">arxiv:2012.09013</a>
&#x1F4C8; 1 <br>
<p>Sean Oesch, Robert Bridges, Jared Smith, Justin Beaver, John Goodall, Kelly Huffer, Craig Miles, Dan Scofield</p></summary>
<p>

**Abstract:** Gartner, a large research and advisory company, anticipates that by 2024 80% of security operation centers (SOCs) will use machine learning (ML) based solutions to enhance their operations. In light of such widespread adoption, it is vital for the research community to identify and address usability concerns. This work presents the results of the first in situ usability assessment of ML-based tools. With the support of the US Navy, we leveraged the national cyber range, a large, air-gapped cyber testbed equipped with state-of-the-art network and user emulation capabilities, to study six US Naval SOC analysts' usage of two tools. Our analysis identified several serious usability issues, including multiple violations of established usability heuristics form user interface design. We also discovered that analysts lacked a clear mental model of how these tools generate scores, resulting in mistrust and/or misuse of the tools themselves. Surprisingly, we found no correlation between analysts' level of education or years of experience and their performance with either tool, suggesting that other factors such as prior background knowledge or personality play a significant role in ML-based tool usage. Our findings demonstrate that ML-based security tool vendors must put a renewed focus on working with analysts, both experienced and inexperienced, to ensure that their systems are usable and useful in real-world security operations settings.

</p>
</details>

<details><summary><b>Visually Grounding Instruction for History-Dependent Manipulation</b>
<a href="https://arxiv.org/abs/2012.08977">arxiv:2012.08977</a>
&#x1F4C8; 1 <br>
<p>Hyemin Ahn, Obin Kwon, Kyoungdo Kim, Dongheui Lee, Songhwai Oh</p></summary>
<p>

**Abstract:** This paper emphasizes the importance of robot's ability to refer its task history, when it executes a series of pick-and-place manipulations by following text instructions given one by one. The advantage of referring the manipulation history can be categorized into two folds: (1) the instructions omitting details or using co-referential expressions can be interpreted, and (2) the visual information of objects occluded by previous manipulations can be inferred. For this challenge, we introduce the task of history-dependent manipulation which is to visually ground a series of text instructions for proper manipulations depending on the task history. We also suggest a relevant dataset and a methodology based on the deep neural network, and show that our network trained with a synthetic dataset can be applied to the real world based on images transferred into synthetic-style based on the CycleGAN.

</p>
</details>

<details><summary><b>Scenario-aware and Mutual-based approach for Multi-scenario Recommendation in E-Commerce</b>
<a href="https://arxiv.org/abs/2012.08952">arxiv:2012.08952</a>
&#x1F4C8; 1 <br>
<p>Yuting Chen, Yanshi Wang, Yabo Ni, An-Xiang Zeng, Lanfen Lin</p></summary>
<p>

**Abstract:** Recommender systems (RSs) are essential for e-commerce platforms to help meet the enormous needs of users. How to capture user interests and make accurate recommendations for users in heterogeneous e-commerce scenarios is still a continuous research topic. However, most existing studies overlook the intrinsic association of the scenarios: the log data collected from platforms can be naturally divided into different scenarios (e.g., country, city, culture).
  We observed that the scenarios are heterogeneous because of the huge differences among them. Therefore, a unified model is difficult to effectively capture complex correlations (e.g., differences and similarities) between multiple scenarios thus seriously reducing the accuracy of recommendation results.
  In this paper, we target the problem of multi-scenario recommendation in e-commerce, and propose a novel recommendation model named Scenario-aware Mutual Learning (SAML) that leverages the differences and similarities between multiple scenarios. We first introduce scenario-aware feature representation, which transforms the embedding and attention modules to map the features into both global and scenario-specific subspace in parallel. Then we introduce an auxiliary network to model the shared knowledge across all scenarios, and use a multi-branch network to model differences among specific scenarios. Finally, we employ a novel mutual unit to adaptively learn the similarity between various scenarios and incorporate it into multi-branch network. We conduct extensive experiments on both public and industrial datasets, empirical results show that SAML consistently and significantly outperforms state-of-the-art methods.

</p>
</details>

<details><summary><b>On The Verification of Neural ODEs with Stochastic Guarantees</b>
<a href="https://arxiv.org/abs/2012.08863">arxiv:2012.08863</a>
&#x1F4C8; 1 <br>
<p>Sophie Gruenbacher, Ramin Hasani, Mathias Lechner, Jacek Cyranka, Scott A. Smolka, Radu Grosu</p></summary>
<p>

**Abstract:** We show that Neural ODEs, an emerging class of time-continuous neural networks, can be verified by solving a set of global-optimization problems. For this purpose, we introduce Stochastic Lagrangian Reachability (SLR), an abstraction-based technique for constructing a tight Reachtube (an over-approximation of the set of reachable states over a given time-horizon), and provide stochastic guarantees in the form of confidence intervals for the Reachtube bounds. SLR inherently avoids the infamous wrapping effect (accumulation of over-approximation errors) by performing local optimization steps to expand safe regions instead of repeatedly forward-propagating them as is done by deterministic reachability methods. To enable fast local optimizations, we introduce a novel forward-mode adjoint sensitivity method to compute gradients without the need for backpropagation. Finally, we establish asymptotic and non-asymptotic convergence rates for SLR.

</p>
</details>

<details><summary><b>A comparative evaluation of machine learning methods for robot navigation through human crowds</b>
<a href="https://arxiv.org/abs/2012.08822">arxiv:2012.08822</a>
&#x1F4C8; 1 <br>
<p>Anastasia Gaydashenko, Daniel Kudenko, Aleksei Shpilman</p></summary>
<p>

**Abstract:** Robot navigation through crowds poses a difficult challenge to AI systems, since the methods should result in fast and efficient movement but at the same time are not allowed to compromise safety. Most approaches to date were focused on the combination of pathfinding algorithms with machine learning for pedestrian walking prediction. More recently, reinforcement learning techniques have been proposed in the research literature. In this paper, we perform a comparative evaluation of pathfinding/prediction and reinforcement learning approaches on a crowd movement dataset collected from surveillance videos taken at Grand Central Station in New York. The results demonstrate the strong superiority of state-of-the-art reinforcement learning approaches over pathfinding with state-of-the-art behaviour prediction techniques.

</p>
</details>

<details><summary><b>Learning-based Prediction and Uplink Retransmission for Wireless Virtual Reality (VR) Network</b>
<a href="https://arxiv.org/abs/2012.12725">arxiv:2012.12725</a>
&#x1F4C8; 0 <br>
<p>Xiaonan Liu, Xinyu Li, Yansha Deng</p></summary>
<p>

**Abstract:** Wireless Virtual Reality (VR) users are able to enjoy immersive experience from anywhere at anytime. However, providing full spherical VR video with high quality under limited VR interaction latency is challenging. If the viewpoint of the VR user can be predicted in advance, only the required viewpoint is needed to be rendered and delivered, which can reduce the VR interaction latency. Therefore, in this paper, we use offline and online learning algorithms to predict viewpoint of the VR user using real VR dataset. For the offline learning algorithm, the trained learning model is directly used to predict the viewpoint of VR users in continuous time slots. While for the online learning algorithm, based on the VR user's actual viewpoint delivered through uplink transmission, we compare it with the predicted viewpoint and update the parameters of the online learning algorithm to further improve the prediction accuracy. To guarantee the reliability of the uplink transmission, we integrate the Proactive retransmission scheme into our proposed online learning algorithm. Simulation results show that our proposed online learning algorithm for uplink wireless VR network with the proactive retransmission scheme only exhibits about 5% prediction error.

</p>
</details>

<details><summary><b>Applying Deutsch's concept of good explanations to artificial intelligence and neuroscience -- an initial exploration</b>
<a href="https://arxiv.org/abs/2012.09318">arxiv:2012.09318</a>
&#x1F4C8; 0 <br>
<p>Daniel C. Elton</p></summary>
<p>

**Abstract:** Artificial intelligence has made great strides since the deep learning revolution, but AI systems still struggle to extrapolate outside of their training data and adapt to new situations. For inspiration we look to the domain of science, where scientists have been able to develop theories which show remarkable ability to extrapolate and sometimes predict the existence of phenomena which have never been observed before. According to David Deutsch, this type of extrapolation, which he calls "reach", is due to scientific theories being hard to vary. In this work we investigate Deutsch's hard-to-vary principle and how it relates to more formalized principles in deep learning such as the bias-variance trade-off and Occam's razor. We distinguish internal variability, how much a model/theory can be varied internally while still yielding the same predictions, with external variability, which is how much a model must be varied to accurately predict new, out-of-distribution data. We discuss how to measure internal variability using the size of the Rashomon set and how to measure external variability using Kolmogorov complexity. We explore what role hard-to-vary explanations play in intelligence by looking at the human brain and distinguish two learning systems in the brain. The first system operates similar to deep learning and likely underlies most of perception and motor control while the second is a more creative system capable of generating hard-to-vary explanations of the world. We argue that figuring out how replicate this second system, which is capable of generating hard-to-vary explanations, is a key challenge which needs to be solved in order to realize artificial general intelligence. We make contact with the framework of Popperian epistemology which rejects induction and asserts that knowledge generation is an evolutionary process which proceeds through conjecture and refutation.

</p>
</details>

<details><summary><b>Beyond the Hype: A Real-World Evaluation of the Impact and Cost of Machine Learning--Based Malware Detection</b>
<a href="https://arxiv.org/abs/2012.09214">arxiv:2012.09214</a>
&#x1F4C8; 0 <br>
<p>Robert A. Bridges, Sean Oesch, Miki E. Verma, Michael D. Iannacone, Kelly M. T. Huffer, Brian Jewell, Jeff A. Nichols, Brian Weber, Justin M. Beaver, Jared M. Smith, Daniel Scofield, Craig Miles, Thomas Plummer, Mark Daniell, Anne M. Tall</p></summary>
<p>

**Abstract:** There is a lack of scientific testing of commercially available malware detectors, especially those that boast accurate classification of never-before-seen (zero-day) files using machine learning (ML). The result is that the efficacy and trade-offs among the different available approaches are opaque. In this paper, we address this gap in the scientific literature with an evaluation of commercially available malware detection tools. We tested each tool against 3,536 total files (2,554 72% malicious, 982 28% benign) including over 400 zero-day malware, and tested with a variety of file types and protocols for delivery. Specifically, we investigate three questions: Do ML-based malware detectors provide better detection than signature-based detectors? Is it worth purchasing a network-level malware detector to complement host-based detection? What is the trade-off in detection time and detection accuracy among commercially available tools using static and dynamic analysis? We present statistical results on detection time and accuracy, consider complementary analysis (using multiple tools together), and provide a novel application of a recent cost-benefit evaluation procedure by Iannaconne \& Bridges that incorporates all the above metrics into a single quantifiable cost to help security operation centers select the right tools for their use case. Our results show that while ML-based tools are more effective at detecting zero-days and malicious executables, they work best when used in combination with a signature-based solution. In addition, network-based tools had poor detection rates on protocols other than the HTTP or SMTP, making them a poor choice if used on their own. Surprisingly, we also found that all the tools tested had lower than expected detection rates, completely missing 37% of malicious files tested and failing to detect any polyglot files.

</p>
</details>

<details><summary><b>Predicting Generalization in Deep Learning via Metric Learning -- PGDL Shared task</b>
<a href="https://arxiv.org/abs/2012.09117">arxiv:2012.09117</a>
&#x1F4C8; 0 <br>
<p>Sebastian Mežnar, Blaž Škrlj</p></summary>
<p>

**Abstract:** The competition "Predicting Generalization in Deep Learning (PGDL)" aims to provide a platform for rigorous study of generalization of deep learning models and offer insight into the progress of understanding and explaining these models. This report presents the solution that was submitted by the user \emph{smeznar} which achieved the eight place in the competition. In the proposed approach, we create simple metrics and find their best combination with automatic testing on the provided dataset, exploring how combinations of various properties of the input neural network architectures can be used for the prediction of their generalization.

</p>
</details>

<details><summary><b>R$^2$-Net: Relation of Relation Learning Network for Sentence Semantic Matching</b>
<a href="https://arxiv.org/abs/2012.08920">arxiv:2012.08920</a>
&#x1F4C8; 0 <br>
<p>Kun Zhang, Le Wu, Guangyi Lv, Meng Wang, Enhong Chen, Shulan Ruan</p></summary>
<p>

**Abstract:** Sentence semantic matching is one of the fundamental tasks in natural language processing, which requires an agent to determine the semantic relation among input sentences. Recently, deep neural networks have achieved impressive performance in this area, especially BERT. Despite the effectiveness of these models, most of them treat output labels as meaningless one-hot vectors, underestimating the semantic information and guidance of relations that these labels reveal, especially for tasks with a small number of labels. To address this problem, we propose a Relation of Relation Learning Network (R2-Net) for sentence semantic matching. Specifically, we first employ BERT to encode the input sentences from a global perspective. Then a CNN-based encoder is designed to capture keywords and phrase information from a local perspective. To fully leverage labels for better relation information extraction, we introduce a self-supervised relation of relation classification task for guiding R2-Net to consider more about labels. Meanwhile, a triplet loss is employed to distinguish the intra-class and inter-class relations in a finer granularity. Empirical experiments on two sentence semantic matching tasks demonstrate the superiority of our proposed model. As a byproduct, we have released the codes to facilitate other researches.

</p>
</details>

<details><summary><b>On $O( \max \{n_1, n_2 \}\log ( \max \{ n_1, n_2 \} n_3) )$ Sample Entries for $n_1 \times n_2 \times n_3$ Tensor Completion via Unitary Transformation</b>
<a href="https://arxiv.org/abs/2012.08784">arxiv:2012.08784</a>
&#x1F4C8; 0 <br>
<p>Guang-Jing Song, Michael K. Ng, Xiongjun Zhang</p></summary>
<p>

**Abstract:** One of the key problems in tensor completion is the number of uniformly random sample entries required for recovery guarantee. The main aim of this paper is to study $n_1 \times n_2 \times n_3$ third-order tensor completion and investigate into incoherence conditions of $n_3$ low-rank $n_1$-by-$n_2$ matrix slices under the transformed tensor singular value decomposition where the unitary transformation is applied along $n_3$-dimension. We show that such low-rank tensors can be recovered exactly with high probability when the number of randomly observed entries is of order $O( r\max \{n_1, n_2 \} \log ( \max \{ n_1, n_2 \} n_3))$, where $r$ is the sum of the ranks of these $n_3$ matrix slices in the transformed tensor. By utilizing synthetic data and imaging data sets, we demonstrate that the theoretical result can be obtained under valid incoherence conditions, and the tensor completion performance of the proposed method is also better than that of existing methods in terms of sample sizes requirement.

</p>
</details>


[Next Page](2020/2020-12/2020-12-15.md)
