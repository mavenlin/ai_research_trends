## Summary for 2020-12-21, created on 2021-01-17


<details><summary><b>Hardware and Software Optimizations for Accelerating Deep Neural Networks: Survey of Current Trends, Challenges, and the Road Ahead</b>
<a href="https://arxiv.org/abs/2012.11233">arxiv:2012.11233</a>
&#x1F4C8; 79 <br>
<p>Maurizio Capra, Beatrice Bussolino, Alberto Marchisio, Guido Masera, Maurizio Martina, Muhammad Shafique</p></summary>
<p>

**Abstract:** Currently, Machine Learning (ML) is becoming ubiquitous in everyday life. Deep Learning (DL) is already present in many applications ranging from computer vision for medicine to autonomous driving of modern cars as well as other sectors in security, healthcare, and finance. However, to achieve impressive performance, these algorithms employ very deep networks, requiring a significant computational power, both during the training and inference time. A single inference of a DL model may require billions of multiply-and-accumulated operations, making the DL extremely compute- and energy-hungry. In a scenario where several sophisticated algorithms need to be executed with limited energy and low latency, the need for cost-effective hardware platforms capable of implementing energy-efficient DL execution arises. This paper first introduces the key properties of two brain-inspired models like Deep Neural Network (DNN), and Spiking Neural Network (SNN), and then analyzes techniques to produce efficient and high-performance designs. This work summarizes and compares the works for four leading platforms for the execution of algorithms such as CPU, GPU, FPGA and ASIC describing the main solutions of the state-of-the-art, giving much prominence to the last two solutions since they offer greater design flexibility and bear the potential of high energy-efficiency, especially for the inference process. In addition to hardware solutions, this paper discusses some of the important security issues that these DNN and SNN models may have during their execution, and offers a comprehensive section on benchmarking, explaining how to assess the quality of different networks and hardware systems designed for them.

</p>
</details>

<details><summary><b>Evaluating Agents without Rewards</b>
<a href="https://arxiv.org/abs/2012.11538">arxiv:2012.11538</a>
&#x1F4C8; 41 <br>
<p>Brendon Matusch, Jimmy Ba, Danijar Hafner</p></summary>
<p>

**Abstract:** Reinforcement learning has enabled agents to solve challenging tasks in unknown environments. However, manually crafting reward functions can be time consuming, expensive, and error prone to human error. Competing objectives have been proposed for agents to learn without external supervision, but it has been unclear how well they reflect task rewards or human behavior. To accelerate the development of intrinsic objectives, we retrospectively compute potential objectives on pre-collected datasets of agent behavior, rather than optimizing them online, and compare them by analyzing their correlations. We study input entropy, information gain, and empowerment across seven agents, three Atari games, and the 3D game Minecraft. We find that all three intrinsic objectives correlate more strongly with a human behavior similarity metric than with task reward. Moreover, input entropy and information gain correlate more strongly with human similarity than task reward does, suggesting the use of intrinsic objectives for designing agents that behave similarly to human players.

</p>
</details>

<details><summary><b>A Distributional Approach to Controlled Text Generation</b>
<a href="https://arxiv.org/abs/2012.11635">arxiv:2012.11635</a>
&#x1F4C8; 22 <br>
<p>Muhammad Khalifa, Hady Elsahar, Marc Dymetman</p></summary>
<p>

**Abstract:** We propose a Distributional Approach to address Controlled Text Generation from pre-trained Language Models (LMs). This view permits to define, in a single formal framework, "pointwise" and "distributional" constraints over the target LM -- to our knowledge, this is the first approach with such generality -- while minimizing KL divergence with the initial LM distribution. The optimal target distribution is then uniquely determined as an explicit EBM (Energy-Based Model) representation. From that optimal representation we then train the target controlled autoregressive LM through an adaptive distributional variant of Policy Gradient. We conduct a first set of experiments over pointwise constraints showing the advantages of our approach over a set of baselines, in terms of obtaining a controlled LM balancing constraint satisfaction with divergence from the initial LM (GPT-2). We then perform experiments over distributional constraints, a unique feature of our approach, demonstrating its potential as a remedy to the problem of Bias in Language Models. Through an ablation study we show the effectiveness of our adaptive technique for obtaining faster convergence.

</p>
</details>

<details><summary><b>Online Bag-of-Visual-Words Generation for Unsupervised Representation Learning</b>
<a href="https://arxiv.org/abs/2012.11552">arxiv:2012.11552</a>
&#x1F4C8; 22 <br>
<p>Spyros Gidaris, Andrei Bursuc, Gilles Puy, Nikos Komodakis, Matthieu Cord, Patrick Pérez</p></summary>
<p>

**Abstract:** Learning image representations without human supervision is an important and active research field. Several recent approaches have successfully leveraged the idea of making such a representation invariant under different types of perturbations, especially via contrastive-based instance discrimination training. Although effective visual representations should indeed exhibit such invariances, there are other important characteristics, such as encoding contextual reasoning skills, for which alternative reconstruction-based approaches might be better suited.
  With this in mind, we propose a teacher-student scheme to learn representations by training a convnet to reconstruct a bag-of-visual-words (BoW) representation of an image, given as input a perturbed version of that same image. Our strategy performs an online training of both the teacher network (whose role is to generate the BoW targets) and the student network (whose role is to learn representations), along with an online update of the visual-words vocabulary (used for the BoW targets). This idea effectively enables fully online BoW-guided unsupervised learning. Extensive experiments demonstrate the interest of our BoW-based strategy which surpasses previous state-of-the-art methods (including contrastive-based ones) in several applications. For instance, in downstream tasks such Pascal object detection, Pascal classification and Places205 classification, our method improves over all prior unsupervised approaches, thus establishing new state-of-the-art results that are also significantly better even than those of supervised pre-training. We provide the implementation code at https://github.com/valeoai/obow.

</p>
</details>

<details><summary><b>Semantic Audio-Visual Navigation</b>
<a href="https://arxiv.org/abs/2012.11583">arxiv:2012.11583</a>
&#x1F4C8; 10 <br>
<p>Changan Chen, Ziad Al-Halah, Kristen Grauman</p></summary>
<p>

**Abstract:** Recent work on audio-visual navigation assumes a constantly-sounding target and restricts the role of audio to signaling the target's spatial placement. We introduce semantic audio-visual navigation, where objects in the environment make sounds consistent with their semantic meanings (e.g., toilet flushing, door creaking) and acoustic envents are sporadic or short in duration. We propose a transformer-based model to tackle this new semantic AudioGoal task, incorporating an inferred goal descriptor that captures both spatial and semantic properties of the target. Our model's persistent multimodal memory enables it to reach the goal even long after the acoustic event stops. In support of the new task, we also expand the SoundSpaces audio simulation platform to provide semantically grounded object sounds for an array of objects in Matterport3D. Our method strongly outperforms existing audio-visual navigation methods by learning to associate semantic, acoustic, and visual cues.

</p>
</details>

<details><summary><b>Offline Reinforcement Learning from Images with Latent Space Models</b>
<a href="https://arxiv.org/abs/2012.11547">arxiv:2012.11547</a>
&#x1F4C8; 10 <br>
<p>Rafael Rafailov, Tianhe Yu, Aravind Rajeswaran, Chelsea Finn</p></summary>
<p>

**Abstract:** Offline reinforcement learning (RL) refers to the problem of learning policies from a static dataset of environment interactions. Offline RL enables extensive use and re-use of historical datasets, while also alleviating safety concerns associated with online exploration, thereby expanding the real-world applicability of RL. Most prior work in offline RL has focused on tasks with compact state representations. However, the ability to learn directly from rich observation spaces like images is critical for real-world applications such as robotics. In this work, we build on recent advances in model-based algorithms for offline RL, and extend them to high-dimensional visual observation spaces. Model-based offline RL algorithms have achieved state of the art results in state based tasks and have strong theoretical guarantees. However, they rely crucially on the ability to quantify uncertainty in the model predictions, which is particularly challenging with image observations. To overcome this challenge, we propose to learn a latent-state dynamics model, and represent the uncertainty in the latent space. Our approach is both tractable in practice and corresponds to maximizing a lower bound of the ELBO in the unknown POMDP. In experiments on a range of challenging image-based locomotion and manipulation tasks, we find that our algorithm significantly outperforms previous offline model-free RL methods as well as state-of-the-art online visual model-based RL methods. Moreover, we also find that our approach excels on an image-based drawer closing task on a real robot using a pre-existing dataset. All results including videos can be found online at https://sites.google.com/view/lompo/ .

</p>
</details>

<details><summary><b>Neural Methods for Effective, Efficient, and Exposure-Aware Information Retrieval</b>
<a href="https://arxiv.org/abs/2012.11685">arxiv:2012.11685</a>
&#x1F4C8; 8 <br>
<p>Bhaskar Mitra</p></summary>
<p>

**Abstract:** Neural networks with deep architectures have demonstrated significant performance improvements in computer vision, speech recognition, and natural language processing. The challenges in information retrieval (IR), however, are different from these other application areas. A common form of IR involves ranking of documents -- or short passages -- in response to keyword-based queries. Effective IR systems must deal with query-document vocabulary mismatch problem, by modeling relationships between different query and document terms and how they indicate relevance. Models should also consider lexical matches when the query contains rare terms -- such as a person's name or a product model number -- not seen during training, and to avoid retrieving semantically related but irrelevant results. In many real-life IR tasks, the retrieval involves extremely large collections -- such as the document index of a commercial Web search engine -- containing billions of documents. Efficient IR methods should take advantage of specialized IR data structures, such as inverted index, to efficiently retrieve from large collections. Given an information need, the IR system also mediates how much exposure an information artifact receives by deciding whether it should be displayed, and where it should be positioned, among other results. Exposure-aware IR systems may optimize for additional objectives, besides relevance, such as parity of exposure for retrieved items and content publishers. In this thesis, we present novel neural architectures and methods motivated by the specific needs and challenges of IR tasks.

</p>
</details>

<details><summary><b>Ordered Counterfactual Explanation by Mixed-Integer Linear Optimization</b>
<a href="https://arxiv.org/abs/2012.11782">arxiv:2012.11782</a>
&#x1F4C8; 7 <br>
<p>Kentaro Kanamori, Takuya Takagi, Ken Kobayashi, Yuichi Ike, Kento Uemura, Hiroki Arimura</p></summary>
<p>

**Abstract:** Post-hoc explanation methods for machine learning models have been widely used to support decision-making. One of the popular methods is Counterfactual Explanation (CE), which provides a user with a perturbation vector of features that alters the prediction result. Given a perturbation vector, a user can interpret it as an "action" for obtaining one's desired decision result. In practice, however, showing only a perturbation vector is often insufficient for users to execute the action. The reason is that if there is an asymmetric interaction among features, such as causality, the total cost of the action is expected to depend on the order of changing features. Therefore, practical CE methods are required to provide an appropriate order of changing features in addition to a perturbation vector. For this purpose, we propose a new framework called Ordered Counterfactual Explanation (OrdCE). We introduce a new objective function that evaluates a pair of an action and an order based on feature interaction. To extract an optimal pair, we propose a mixed-integer linear optimization approach with our objective function. Numerical experiments on real datasets demonstrated the effectiveness of our OrdCE in comparison with unordered CE methods.

</p>
</details>

<details><summary><b>Object-Centric Diagnosis of Visual Reasoning</b>
<a href="https://arxiv.org/abs/2012.11587">arxiv:2012.11587</a>
&#x1F4C8; 7 <br>
<p>Jianwei Yang, Jiayuan Mao, Jiajun Wu, Devi Parikh, David D. Cox, Joshua B. Tenenbaum, Chuang Gan</p></summary>
<p>

**Abstract:** When answering questions about an image, it not only needs knowing what -- understanding the fine-grained contents (e.g., objects, relationships) in the image, but also telling why -- reasoning over grounding visual cues to derive the answer for a question. Over the last few years, we have seen significant progress on visual question answering. Though impressive as the accuracy grows, it still lags behind to get knowing whether these models are undertaking grounding visual reasoning or just leveraging spurious correlations in the training data. Recently, a number of works have attempted to answer this question from perspectives such as grounding and robustness. However, most of them are either focusing on the language side or coarsely studying the pixel-level attention maps. In this paper, by leveraging the step-wise object grounding annotations provided in the GQA dataset, we first present a systematical object-centric diagnosis of visual reasoning on grounding and robustness, particularly on the vision side. According to the extensive comparisons across different models, we find that even models with high accuracy are not good at grounding objects precisely, nor robust to visual content perturbations. In contrast, symbolic and modular models have a relatively better grounding and robustness, though at the cost of accuracy. To reconcile these different aspects, we further develop a diagnostic model, namely Graph Reasoning Machine. Our model replaces purely symbolic visual representation with probabilistic scene graph and then applies teacher-forcing training for the visual reasoning module. The designed model improves the performance on all three metrics over the vanilla neural-symbolic model while inheriting the transparency. Further ablation studies suggest that this improvement is mainly due to more accurate image understanding and proper intermediate reasoning supervisions.

</p>
</details>

<details><summary><b>Building LEGO Using Deep Generative Models of Graphs</b>
<a href="https://arxiv.org/abs/2012.11543">arxiv:2012.11543</a>
&#x1F4C8; 7 <br>
<p>Rylee Thompson, Elahe Ghalebi, Terrance DeVries, Graham W. Taylor</p></summary>
<p>

**Abstract:** Generative models are now used to create a variety of high-quality digital artifacts. Yet their use in designing physical objects has received far less attention. In this paper, we advocate for the construction toy, LEGO, as a platform for developing generative models of sequential assembly. We develop a generative model based on graph-structured neural networks that can learn from human-built structures and produce visually compelling designs. Our code is released at: https://github.com/uoguelph-mlrg/GenerativeLEGO.

</p>
</details>

<details><summary><b>An Overview of Facial Micro-Expression Analysis: Data, Methodology and Challenge</b>
<a href="https://arxiv.org/abs/2012.11307">arxiv:2012.11307</a>
&#x1F4C8; 7 <br>
<p>Hong-Xia Xie, Ling Lo, Hong-Han Shuai, Wen-Huang Cheng</p></summary>
<p>

**Abstract:** Facial micro-expressions indicate brief and subtle facial movements that appear during emotional communication. In comparison to macro-expressions, micro-expressions are more challenging to be analyzed due to the short span of time and the fine-grained changes. In recent years, micro-expression recognition (MER) has drawn much attention because it can benefit a wide range of applications, e.g. police interrogation, clinical diagnosis, depression analysis, and business negotiation. In this survey, we offer a fresh overview to discuss new research directions and challenges these days for MER tasks. For example, we review MER approaches from three novel aspects: macro-to-micro adaptation, recognition based on key apex frames, and recognition based on facial action units. Moreover, to mitigate the problem of limited and biased ME data, synthetic data generation is surveyed for the diversity enrichment of micro-expression data. Since micro-expression spotting can boost micro-expression analysis, the state-of-the-art spotting works are also introduced in this paper. At last, we discuss the challenges in MER research and provide potential solutions as well as possible directions for further investigation.

</p>
</details>

<details><summary><b>SENTRY: Selective Entropy Optimization via Committee Consistency for Unsupervised Domain Adaptation</b>
<a href="https://arxiv.org/abs/2012.11460">arxiv:2012.11460</a>
&#x1F4C8; 6 <br>
<p>Viraj Prabhu, Shivam Khare, Deeksha Kartik, Judy Hoffman</p></summary>
<p>

**Abstract:** Many existing approaches for unsupervised domain adaptation (UDA) focus on adapting under only data distribution shift and offer limited success under additional cross-domain label distribution shift. Recent work based on self-training using target pseudo-labels has shown promise, but on challenging shifts pseudo-labels may be highly unreliable, and using them for self-training may cause error accumulation and domain misalignment. We propose Selective Entropy Optimization via Committee Consistency (SENTRY), a UDA algorithm that judges the reliability of a target instance based on its predictive consistency under a committee of random image transformations. Our algorithm then selectively minimizes predictive entropy to increase confidence on highly consistent target instances, while maximizing predictive entropy to reduce confidence on highly inconsistent ones. In combination with pseudo-label based approximate target class balancing, our approach leads to significant improvements over the state-of-the-art on 27/31 domain shifts from standard UDA benchmarks as well as benchmarks designed to stress-test adaptation under label distribution shift.

</p>
</details>

<details><summary><b>Learning Disentangled Semantic Representation for Domain Adaptation</b>
<a href="https://arxiv.org/abs/2012.11807">arxiv:2012.11807</a>
&#x1F4C8; 5 <br>
<p>Ruichu Cai, Zijian Li, Pengfei Wei, Jie Qiao, Kun Zhang, Zhifeng Hao</p></summary>
<p>

**Abstract:** Domain adaptation is an important but challenging task. Most of the existing domain adaptation methods struggle to extract the domain-invariant representation on the feature space with entangling domain information and semantic information. Different from previous efforts on the entangled feature space, we aim to extract the domain invariant semantic information in the latent disentangled semantic representation (DSR) of the data. In DSR, we assume the data generation process is controlled by two independent sets of variables, i.e., the semantic latent variables and the domain latent variables. Under the above assumption, we employ a variational auto-encoder to reconstruct the semantic latent variables and domain latent variables behind the data. We further devise a dual adversarial network to disentangle these two sets of reconstructed latent variables. The disentangled semantic latent variables are finally adapted across the domains. Experimental studies testify that our model yields state-of-the-art performance on several domain adaptation benchmark datasets.

</p>
</details>

<details><summary><b>Alleviating Noisy Data in Image Captioning with Cooperative Distillation</b>
<a href="https://arxiv.org/abs/2012.11691">arxiv:2012.11691</a>
&#x1F4C8; 5 <br>
<p>Pierre Dognin, Igor Melnyk, Youssef Mroueh, Inkit Padhi, Mattia Rigotti, Jarret Ross, Yair Schiff</p></summary>
<p>

**Abstract:** Image captioning systems have made substantial progress, largely due to the availability of curated datasets like Microsoft COCO or Vizwiz that have accurate descriptions of their corresponding images. Unfortunately, scarce availability of such cleanly labeled data results in trained algorithms producing captions that can be terse and idiosyncratically specific to details in the image. We propose a new technique, cooperative distillation that combines clean curated datasets with the web-scale automatically extracted captions of the Google Conceptual Captions dataset (GCC), which can have poor descriptions of images, but is abundant in size and therefore provides a rich vocabulary resulting in more expressive captions.

</p>
</details>

<details><summary><b>LQF: Linear Quadratic Fine-Tuning</b>
<a href="https://arxiv.org/abs/2012.11140">arxiv:2012.11140</a>
&#x1F4C8; 5 <br>
<p>Alessandro Achille, Aditya Golatkar, Avinash Ravichandran, Marzia Polito, Stefano Soatto</p></summary>
<p>

**Abstract:** Classifiers that are linear in their parameters, and trained by optimizing a convex loss function, have predictable behavior with respect to changes in the training data, initial conditions, and optimization. Such desirable properties are absent in deep neural networks (DNNs), typically trained by non-linear fine-tuning of a pre-trained model. Previous attempts to linearize DNNs have led to interesting theoretical insights, but have not impacted the practice due to the substantial performance gap compared to standard non-linear optimization. We present the first method for linearizing a pre-trained model that achieves comparable performance to non-linear fine-tuning on most of real-world image classification tasks tested, thus enjoying the interpretability of linear models without incurring punishing losses in performance. LQF consists of simple modifications to the architecture, loss function and optimization typically used for classification: Leaky-ReLU instead of ReLU, mean squared loss instead of cross-entropy, and pre-conditioning using Kronecker factorization. None of these changes in isolation is sufficient to approach the performance of non-linear fine-tuning. When used in combination, they allow us to reach comparable performance, and even superior in the low-data regime, while enjoying the simplicity, robustness and interpretability of linear-quadratic optimization.

</p>
</details>

<details><summary><b>AttentionLite: Towards Efficient Self-Attention Models for Vision</b>
<a href="https://arxiv.org/abs/2101.05216">arxiv:2101.05216</a>
&#x1F4C8; 4 <br>
<p>Souvik Kundu, Sairam Sundaresan</p></summary>
<p>

**Abstract:** We propose a novel framework for producing a class of parameter and compute efficient models called AttentionLitesuitable for resource-constrained applications. Prior work has primarily focused on optimizing models either via knowledge distillation or pruning. In addition to fusing these two mechanisms, our joint optimization framework also leverages recent advances in self-attention as a substitute for convolutions. We can simultaneously distill knowledge from a compute-heavy teacher while also pruning the student model in a single pass of training thereby reducing training and fine-tuning times considerably. We evaluate the merits of our proposed approach on the CIFAR-10, CIFAR-100, and Tiny-ImageNet datasets. Not only do our AttentionLite models significantly outperform their unoptimized counterparts in accuracy, we find that in some cases, that they perform almost as well as their compute-heavy teachers while consuming only a fraction of the parameters and FLOPs. Concretely, AttentionLite models can achieve upto30x parameter efficiency and 2x computation efficiency with no significant accuracy drop compared to their teacher.

</p>
</details>

<details><summary><b>Empirically Classifying Network Mechanisms</b>
<a href="https://arxiv.org/abs/2012.15863">arxiv:2012.15863</a>
&#x1F4C8; 4 <br>
<p>Ryan E. Langendorf, Matthew G. Burgess</p></summary>
<p>

**Abstract:** Network models are used to study interconnected systems across many physical, biological, and social disciplines. Such models often assume a particular network-generating mechanism, which when fit to data produces estimates of mechanism-specific parameters that describe how systems function. For instance, a social network model might assume new individuals connect to others with probability proportional to their number of pre-existing connections ('preferential attachment'), and then estimate the disparity in interactions between famous and obscure individuals with similar qualifications. However, without a means of testing the relevance of the assumed mechanism, conclusions from such models could be misleading. Here we introduce a simple empirical approach which can mechanistically classify arbitrary network data. Our approach compares empirical networks to model networks from a user-provided candidate set of mechanisms, and classifies each network--with high accuracy--as originating from either one of the mechanisms or none of them. We tested 373 empirical networks against five of the most widely studied network mechanisms and found that most (228) were unlike any of these mechanisms. This raises the possibility that some empirical networks arise from mixtures of mechanisms. We show that mixtures are often unidentifiable because different mixtures can produce functionally equivalent networks. In such systems, which are governed by multiple mechanisms, our approach can still accurately predict out-of-sample functional properties.

</p>
</details>

<details><summary><b>SChuBERT: Scholarly Document Chunks with BERT-encoding boost Citation Count Prediction</b>
<a href="https://arxiv.org/abs/2012.11740">arxiv:2012.11740</a>
&#x1F4C8; 4 <br>
<p>Thomas van Dongen, Gideon Maillette de Buy Wenniger, Lambert Schomaker</p></summary>
<p>

**Abstract:** Predicting the number of citations of scholarly documents is an upcoming task in scholarly document processing. Besides the intrinsic merit of this information, it also has a wider use as an imperfect proxy for quality which has the advantage of being cheaply available for large volumes of scholarly documents. Previous work has dealt with number of citations prediction with relatively small training data sets, or larger datasets but with short, incomplete input text. In this work we leverage the open access ACL Anthology collection in combination with the Semantic Scholar bibliometric database to create a large corpus of scholarly documents with associated citation information and we propose a new citation prediction model called SChuBERT. In our experiments we compare SChuBERT with several state-of-the-art citation prediction models and show that it outperforms previous methods by a large margin. We also show the merit of using more training data and longer input for number of citations prediction.

</p>
</details>

<details><summary><b>Social NCE: Contrastive Learning of Socially-aware Motion Representations</b>
<a href="https://arxiv.org/abs/2012.11717">arxiv:2012.11717</a>
&#x1F4C8; 4 <br>
<p>Yuejiang Liu, Qi Yan, Alexandre Alahi</p></summary>
<p>

**Abstract:** Learning socially-aware motion representations is at the core of recent advances in human trajectory forecasting and robot navigation in crowded spaces. Yet existing methods often struggle to generalize to challenging scenarios and even output unacceptable solutions (e.g., collisions). In this work, we propose to address this issue via contrastive learning. Concretely, we introduce a social contrastive loss that encourages the encoded motion representation to preserve sufficient information for distinguishing a positive future event from a set of negative ones. We explicitly draw these negative samples based on our domain knowledge about socially unfavorable scenarios in the multi-agent context. Experimental results show that the proposed method consistently boosts the performance of previous trajectory forecasting, behavioral cloning, and reinforcement learning algorithms in various settings. Our method makes little assumptions about neural architecture designs, and hence can be used as a generic way to incorporate negative data augmentation into motion representation learning.

</p>
</details>

<details><summary><b>Facebook Ad Engagement in the Russian Active Measures Campaign of 2016</b>
<a href="https://arxiv.org/abs/2012.11690">arxiv:2012.11690</a>
&#x1F4C8; 4 <br>
<p>Mirela Silva, Luiz Giovanini, Juliana Fernandes, Daniela Oliveira, Catia S. Silva</p></summary>
<p>

**Abstract:** This paper examines 3,517 Facebook ads created by Russia's Internet Research Agency (IRA) between June 2015 and August 2017 in its active measures disinformation campaign targeting the 2016 U.S. general election. We aimed to unearth the relationship between ad engagement (as measured by ad clicks) and 41 features related to ads' metadata, sociolinguistic structures, and sentiment. Our analysis was three-fold: (i) understand the relationship between engagement and features via correlation analysis; (ii) find the most relevant feature subsets to predict engagement via feature selection; and (iii) find the semantic topics that best characterize the dataset via topic modeling. We found that ad expenditure, text size, ad lifetime, and sentiment were the top features predicting users' engagement to the ads. Additionally, positive sentiment ads were more engaging than negative ads, and sociolinguistic features (e.g., use of religion-relevant words) were identified as highly important in the makeup of an engaging ad. Linear SVM and Logistic Regression classifiers achieved the highest mean F-scores (93.6% for both models), determining that the optimal feature subset contains 12 and 6 features, respectively. Finally, we corroborate the findings of related works that the IRA specifically targeted Americans on divisive ad topics (e.g., LGBT rights, African American reparations).

</p>
</details>

<details><summary><b>Natural vs Balanced Distribution in Deep Learning on Whole Slide Images for Cancer Detection</b>
<a href="https://arxiv.org/abs/2012.11684">arxiv:2012.11684</a>
&#x1F4C8; 4 <br>
<p>Ismat Ara Reshma, Sylvain Cussat-Blanc, Radu Tudor Ionescu, Hervé Luga, Josiane Mothe</p></summary>
<p>

**Abstract:** The class distribution of data is one of the factors that regulates the performance of machine learning models. However, investigations on the impact of different distributions available in the literature are very few, sometimes absent for domain-specific tasks. In this paper, we analyze the impact of natural and balanced distributions of the training set in deep learning (DL) models applied on histological images, also known as whole slide images (WSIs). WSIs are considered as the gold standard for cancer diagnosis. In recent years, researchers have turned their attention to DL models to automate and accelerate the diagnosis process. In the training of such DL models, filtering out the non-regions-of-interest from the WSIs and adopting an artificial distribution (usually, a balanced distribution) is a common trend. In our analysis, we show that keeping the WSIs data in their usual distribution (which we call natural distribution) for DL training produces fewer false positives (FPs) with comparable false negatives (FNs) than the artificially-obtained balanced distribution. We conduct an empirical comparative study with 10 random folds for each distribution, comparing the resulting average performance levels in terms of five different evaluation metrics. Experimental results show the effectiveness of the natural distribution over the balanced one across all the evaluation metrics.

</p>
</details>

<details><summary><b>BERTChem-DDI : Improved Drug-Drug Interaction Prediction from text using Chemical Structure Information</b>
<a href="https://arxiv.org/abs/2012.11599">arxiv:2012.11599</a>
&#x1F4C8; 4 <br>
<p>Ishani Mondal</p></summary>
<p>

**Abstract:** Traditional biomedical version of embeddings obtained from pre-trained language models have recently shown state-of-the-art results for relation extraction (RE) tasks in the medical domain. In this paper, we explore how to incorporate domain knowledge, available in the form of molecular structure of drugs, for predicting Drug-Drug Interaction from textual corpus. We propose a method, BERTChem-DDI, to efficiently combine drug embeddings obtained from the rich chemical structure of drugs along with off-the-shelf domain-specific BioBERT embedding-based RE architecture. Experiments conducted on the DDIExtraction 2013 corpus clearly indicate that this strategy improves other strong baselines architectures by 3.4\% macro F1-score.

</p>
</details>

<details><summary><b>Can we learn where people come from? Retracing of origins in merging situations</b>
<a href="https://arxiv.org/abs/2012.11527">arxiv:2012.11527</a>
&#x1F4C8; 4 <br>
<p>Marion Gödel, Luca Spataro, Gerta Köster</p></summary>
<p>

**Abstract:** One crucial information for a pedestrian crowd simulation is the number of agents moving from an origin to a certain target. While this setup has a large impact on the simulation, it is in most setups challenging to find the number of agents that should be spawned at a source in the simulation. Often, number are chosen based on surveys and experience of modelers and event organizers. These approaches are important and useful but reach their limits when we want to perform real-time predictions. In this case, a static information about the inflow is not sufficient. Instead, we need a dynamic information that can be retrieved each time the prediction is started. Nowadays, sensor data such as video footage or GPS tracks of a crowd are often available. If we can estimate the number of pedestrians who stem from a certain origin from this sensor data, we can dynamically initialize the simulation. In this study, we use density heatmaps that can be derived from sensor data as input for a random forest regressor to predict the origin distributions. We study three different datasets: A simulated dataset, experimental data, and a hybrid approach with both experimental and simulated data. In the hybrid setup, the model is trained with simulated data and then tested on experimental data. The results demonstrate that the random forest model is able to predict the origin distribution based on a single density heatmap for all three configurations. This is especially promising for applying the approach on real data since there is often only a limited amount of data available.

</p>
</details>

<details><summary><b>Zeroth-Order Hybrid Gradient Descent: Towards A Principled Black-Box Optimization Framework</b>
<a href="https://arxiv.org/abs/2012.11518">arxiv:2012.11518</a>
&#x1F4C8; 4 <br>
<p>Pranay Sharma, Kaidi Xu, Sijia Liu, Pin-Yu Chen, Xue Lin, Pramod K. Varshney</p></summary>
<p>

**Abstract:** In this work, we focus on the study of stochastic zeroth-order (ZO) optimization which does not require first-order gradient information and uses only function evaluations. The problem of ZO optimization has emerged in many recent machine learning applications, where the gradient of the objective function is either unavailable or difficult to compute. In such cases, we can approximate the full gradients or stochastic gradients through function value based gradient estimates. Here, we propose a novel hybrid gradient estimator (HGE), which takes advantage of the query-efficiency of random gradient estimates as well as the variance-reduction of coordinate-wise gradient estimates. We show that with a graceful design in coordinate importance sampling, the proposed HGE-based ZO optimization method is efficient both in terms of iteration complexity as well as function query cost. We provide a thorough theoretical analysis of the convergence of our proposed method for non-convex, convex, and strongly-convex optimization. We show that the convergence rate that we derive generalizes the results for some prominent existing methods in the nonconvex case, and matches the optimal result in the convex case. We also corroborate the theory with a real-world black-box attack generation application to demonstrate the empirical advantage of our method over state-of-the-art ZO optimization approaches.

</p>
</details>

<details><summary><b>Towards the Localisation of Lesions in Diabetic Retinopathy</b>
<a href="https://arxiv.org/abs/2012.11432">arxiv:2012.11432</a>
&#x1F4C8; 4 <br>
<p>Samuel Ofosu Mensah, Bubacarr Bah, Willie Brink</p></summary>
<p>

**Abstract:** Convolutional Neural Networks (CNN) has successfully been used to classify diabetic retinopathy (DR) fundus images in recent times. However, deeper representations in CNN only capture higher-level semantics at the expense of losing spatial information. To make predictions very usable for ophthalmologists, we use a post-attention technique called Gradient-weighted Class Activation Mapping (Grad-CAM) on the penultimate layer of deep learning models to produce coarse localisation maps on DR fundus images. This is to help identify discriminative regions in the images, consequently providing enough evidence for ophthalmologists to make a diagnosis and saving lives by early diagnosis. Specifically, this study uses pre-trained weights from four (4) state-of-the-art deep learning models to produce and compare the localisation maps of DR fundus images. The models used include VGG16, ResNet50, InceptionV3, and InceptionResNetV2. We find that InceptionV3 achieves the best performance with a test classification accuracy of 96.07% and localise lesions better and faster than the other models.

</p>
</details>

<details><summary><b>Learning Compositional Sparse Gaussian Processes with a Shrinkage Prior</b>
<a href="https://arxiv.org/abs/2012.11339">arxiv:2012.11339</a>
&#x1F4C8; 4 <br>
<p>Anh Tong, Toan Tran, Hung Bui, Jaesik Choi</p></summary>
<p>

**Abstract:** Choosing a proper set of kernel functions is an important problem in learning Gaussian Process (GP) models since each kernel structure has different model complexity and data fitness. Recently, automatic kernel composition methods provide not only accurate prediction but also attractive interpretability through search-based methods. However, existing methods suffer from slow kernel composition learning. To tackle large-scaled data, we propose a new sparse approximate posterior for GPs, MultiSVGP, constructed from groups of inducing points associated with individual additive kernels in compositional kernels. We demonstrate that this approximation provides a better fit to learn compositional kernels given empirical observations. We also theoretically justification on error bound when compared to the traditional sparse GP. In contrast to the search-based approach, we present a novel probabilistic algorithm to learn a kernel composition by handling the sparsity in the kernel selection with Horseshoe prior. We demonstrate that our model can capture characteristics of time series with significant reductions in computational time and have competitive regression performance on real-world data sets.

</p>
</details>

<details><summary><b>Medical Entity Linking using Triplet Network</b>
<a href="https://arxiv.org/abs/2012.11164">arxiv:2012.11164</a>
&#x1F4C8; 4 <br>
<p>Ishani Mondal, Sukannya Purkayastha, Sudeshna Sarkar, Pawan Goyal, Jitesh Pillai, Amitava Bhattacharyya, Mahanandeeshwar Gattu</p></summary>
<p>

**Abstract:** Entity linking (or Normalization) is an essential task in text mining that maps the entity mentions in the medical text to standard entities in a given Knowledge Base (KB). This task is of great importance in the medical domain. It can also be used for merging different medical and clinical ontologies. In this paper, we center around the problem of disease linking or normalization. This task is executed in two phases: candidate generation and candidate scoring. In this paper, we present an approach to rank the candidate Knowledge Base entries based on their similarity with disease mention. We make use of the Triplet Network for candidate ranking. While the existing methods have used carefully generated sieves and external resources for candidate generation, we introduce a robust and portable candidate generation scheme that does not make use of the hand-crafted rules. Experimental results on the standard benchmark NCBI disease dataset demonstrate that our system outperforms the prior methods by a significant margin.

</p>
</details>

<details><summary><b>Narrative Incoherence Detection</b>
<a href="https://arxiv.org/abs/2012.11157">arxiv:2012.11157</a>
&#x1F4C8; 4 <br>
<p>Deng Cai, Yizhe Zhang, Yichen Huang, Wai Lam, Bill Dolan</p></summary>
<p>

**Abstract:** Motivated by the increasing popularity of intelligent editing assistant, we introduce and investigate the task of narrative incoherence detection: Given a (corrupted) long-form narrative, decide whether there exists some semantic discrepancy in the narrative flow. Specifically, we focus on the missing sentence and incoherent sentence detection. Despite its simple setup, this task is challenging as the model needs to understand and analyze a multi-sentence narrative text, and make decisions at the sentence level. As an initial step towards this task, we implement several baselines either directly analyzing the raw text (\textit{token-level}) or analyzing learned sentence representations (\textit{sentence-level}). We observe that while token-level modeling enjoys greater expressive power and hence better performance, sentence-level modeling possesses an advantage in efficiency and flexibility. With pre-training on large-scale data and cycle-consistent sentence embedding, our extended sentence-level model can achieve comparable detection accuracy to the token-level model. As a by-product, such a strategy enables simultaneous incoherence detection and infilling/modification suggestions.

</p>
</details>

<details><summary><b>Improving Unsupervised Image Clustering With Robust Learning</b>
<a href="https://arxiv.org/abs/2012.11150">arxiv:2012.11150</a>
&#x1F4C8; 4 <br>
<p>Sungwon Park, Sungwon Han, Sundong Kim, Danu Kim, Sungkyu Park, Seunghoon Hong, Meeyoung Cha</p></summary>
<p>

**Abstract:** Unsupervised image clustering methods often introduce alternative objectives to indirectly train the model and are subject to faulty predictions and overconfident results. To overcome these challenges, the current research proposes an innovative model RUC that is inspired by robust learning. RUC's novelty is at utilizing pseudo-labels of existing image clustering models as a noisy dataset that may include misclassified samples. Its retraining process can revise misaligned knowledge and alleviate the overconfidence problem in predictions. This model's flexible structure makes it possible to be used as an add-on module to state-of-the-art clustering methods and helps them achieve better performance on multiple datasets. Extensive experiments show that the proposed model can adjust the model confidence with better calibration and gain additional robustness against adversarial noise.

</p>
</details>

<details><summary><b>Towards Incorporating Entity-specific Knowledge Graph Information in Predicting Drug-Drug Interactions</b>
<a href="https://arxiv.org/abs/2012.11142">arxiv:2012.11142</a>
&#x1F4C8; 4 <br>
<p>Ishani Mondal</p></summary>
<p>

**Abstract:** Off-the-shelf biomedical embeddings obtained from the recently released various pre-trained language models (such as BERT, XLNET) have demonstrated state-of-the-art results (in terms of accuracy) for the various natural language understanding tasks (NLU) in the biomedical domain. Relation Classification (RC) falls into one of the most critical tasks. In this paper, we explore how to incorporate domain knowledge of the biomedical entities (such as drug, disease, genes), obtained from Knowledge Graph (KG) Embeddings, for predicting Drug-Drug Interaction from textual corpus. We propose a new method, BERTKG-DDI, to combine drug embeddings obtained from its interaction with other biomedical entities along with domain-specific BioBERT embedding-based RC architecture. Experiments conducted on the DDIExtraction 2013 corpus clearly indicate that this strategy improves other baselines architectures by 4.1% macro F1-score.

</p>
</details>

<details><summary><b>Subject-independent Human Pose Image Construction with Commodity Wi-Fi</b>
<a href="https://arxiv.org/abs/2012.11812">arxiv:2012.11812</a>
&#x1F4C8; 3 <br>
<p>Shuang Zhou, Lingchao Guo, Zhaoming Lu, Xiangming Wen, Wei Zheng, Yiming Wang</p></summary>
<p>

**Abstract:** Recently, commodity Wi-Fi devices have been shown to be able to construct human pose images, i.e., human skeletons, as fine-grained as cameras. Existing papers achieve good results when constructing the images of subjects who are in the prior training samples. However, the performance drops when it comes to new subjects, i.e., the subjects who are not in the training samples. This paper focuses on solving the subject-generalization problem in human pose image construction. To this end, we define the subject as the domain. Then we design a Domain-Independent Neural Network (DINN) to extract subject-independent features and convert them into fine-grained human pose images. We also propose a novel training method to train the DINN and it has no re-training overhead comparing with the domain-adversarial approach. We build a prototype system and experimental results demonstrate that our system can construct fine-grained human pose images of new subjects with commodity Wi-Fi in both the visible and through-wall scenarios, which shows the effectiveness and the subject-generalization ability of our model.

</p>
</details>

<details><summary><b>Progressive One-shot Human Parsing</b>
<a href="https://arxiv.org/abs/2012.11810">arxiv:2012.11810</a>
&#x1F4C8; 3 <br>
<p>Haoyu He, Jing Zhang, Bhavani Thuraisingham, Dacheng Tao</p></summary>
<p>

**Abstract:** Prior human parsing models are limited to parsing humans into classes pre-defined in the training data, which is not flexible to generalize to unseen classes, e.g., new clothing in fashion analysis. In this paper, we propose a new problem named one-shot human parsing (OSHP) that requires to parse human into an open set of reference classes defined by any single reference example. During training, only base classes defined in the training set are exposed, which can overlap with part of reference classes. In this paper, we devise a novel Progressive One-shot Parsing network (POPNet) to address two critical challenges , i.e., testing bias and small sizes. POPNet consists of two collaborative metric learning modules named Attention Guidance Module and Nearest Centroid Module, which can learn representative prototypes for base classes and quickly transfer the ability to unseen classes during testing, thereby reducing testing bias. Moreover, POPNet adopts a progressive human parsing framework that can incorporate the learned knowledge of parent classes at the coarse granularity to help recognize the descendant classes at the fine granularity, thereby handling the small sizes issue. Experiments on the ATR-OS benchmark tailored for OSHP demonstrate POPNet outperforms other representative one-shot segmentation models by large margins and establishes a strong baseline. Source code can be found at https://github.com/Charleshhy/One-shot-Human-Parsing.

</p>
</details>

<details><summary><b>Fast Physical Activity Suggestions: Efficient Hyperparameter Learning in Mobile Health</b>
<a href="https://arxiv.org/abs/2012.11646">arxiv:2012.11646</a>
&#x1F4C8; 3 <br>
<p>Marianne Menictas, Sabina Tomkins, Susan Murphy</p></summary>
<p>

**Abstract:** Users can be supported to adopt healthy behaviors, such as regular physical activity, via relevant and timely suggestions on their mobile devices. Recently, reinforcement learning algorithms have been found to be effective for learning the optimal context under which to provide suggestions. However, these algorithms are not necessarily designed for the constraints posed by mobile health (mHealth) settings, that they be efficient, domain-informed and computationally affordable. We propose an algorithm for providing physical activity suggestions in mHealth settings. Using domain-science, we formulate a contextual bandit algorithm which makes use of a linear mixed effects model. We then introduce a procedure to efficiently perform hyper-parameter updating, using far less computational resources than competing approaches. Not only is our approach computationally efficient, it is also easily implemented with closed form matrix algebraic updates and we show improvements over state of the art approaches both in speed and accuracy of up to 99% and 56% respectively.

</p>
</details>

<details><summary><b>myGym: Modular Toolkit for Visuomotor Robotic Tasks</b>
<a href="https://arxiv.org/abs/2012.11643">arxiv:2012.11643</a>
&#x1F4C8; 3 <br>
<p>Michal Vavrecka, Nikita Sokovnin, Megi Mejdrechova, Gabriela Sejnova, Marek Otahal</p></summary>
<p>

**Abstract:** We introduce a novel virtual robotic toolkit myGym, developed for reinforcement learning (RL), intrinsic motivation and imitation learning tasks trained in a 3D simulator. The trained tasks can then be easily transferred to real-world robotic scenarios. The modular structure of the simulator enables users to train and validate their algorithms on a large number of scenarios with various robots, environments and tasks. Compared to existing toolkits (e.g. OpenAI Gym, Roboschool) which are suitable for classical RL, myGym is also prepared for visuomotor (combining vision & movement) unsupervised tasks that require intrinsic motivation, i.e. the robots are able to generate their own goals. There are also collaborative scenarios intended for human-robot interaction. The toolkit provides pretrained visual modules for visuomotor tasks allowing rapid prototyping, and, moreover, users can customize the visual submodules and retrain with their own set of objects. In practice, the user selects the desired environment, robot, objects, task and type of reward as simulation parameters, and the training, visualization and testing themselves are handled automatically. The user can thus fully focus on development of the neural network architecture while controlling the behaviour of the environment using predefined parameters.

</p>
</details>

<details><summary><b>Deep Learning in Detection and Diagnosis of Covid-19 using Radiology Modalities: A Systematic Review</b>
<a href="https://arxiv.org/abs/2012.11577">arxiv:2012.11577</a>
&#x1F4C8; 3 <br>
<p>Mustafa Ghaderzadeh, Farkhondeh Asadi</p></summary>
<p>

**Abstract:** Purpose: Early detection and diagnosis of Covid-19 and accurate separation of patients with non-Covid-19 cases at the lowest cost and in the early stages of the disease are one of the main challenges in the epidemic of Covid-19. Concerning the novelty of the disease, the diagnostic methods based on radiological images suffer shortcomings despite their many uses in diagnostic centers. Accordingly, medical and computer researchers tended to use machine-learning models to analyze radiology images.
  Methods: Present systematic review was conducted by searching three databases of PubMed, Scopus, and Web of Science from November 1, 2019, to July 20, 2020 Based on a search strategy, the keywords were Covid-19, Deep learning, Diagnosis and Detection leading to the extraction of 168 articles that ultimately, 37 articles were selected as the research population by applying inclusion and exclusion criteria. Result: This review study provides an overview of the current state of all models for the detection and diagnosis of Covid-19 through radiology modalities and their processing based on deep learning. According to the finding, Deep learning Based models have an extraordinary capacity to achieve an accurate and efficient system for the detection and diagnosis of Covid-19, which using of them in the processing of CT-Scan and X-Ray images, would lead to a significant increase in sensitivity and specificity values.
  Conclusion: The Application of Deep Learning (DL) in the field of Covid-19 radiologic image processing leads to the reduction of false-positive and negative errors in the detection and diagnosis of this disease and provides an optimal opportunity to provide fast, cheap, and safe diagnostic services to patients.

</p>
</details>

<details><summary><b>Variational Transport: A Convergent Particle-BasedAlgorithm for Distributional Optimization</b>
<a href="https://arxiv.org/abs/2012.11554">arxiv:2012.11554</a>
&#x1F4C8; 3 <br>
<p>Zhuoran Yang, Yufeng Zhang, Yongxin Chen, Zhaoran Wang</p></summary>
<p>

**Abstract:** We consider the optimization problem of minimizing a functional defined over a family of probability distributions, where the objective functional is assumed to possess a variational form. Such a distributional optimization problem arises widely in machine learning and statistics, with Monte-Carlo sampling, variational inference, policy optimization, and generative adversarial network as examples. For this problem, we propose a novel particle-based algorithm, dubbed as variational transport, which approximately performs Wasserstein gradient descent over the manifold of probability distributions via iteratively pushing a set of particles. Specifically, we prove that moving along the geodesic in the direction of functional gradient with respect to the second-order Wasserstein distance is equivalent to applying a pushforward mapping to a probability distribution, which can be approximated accurately by pushing a set of particles. Specifically, in each iteration of variational transport, we first solve the variational problem associated with the objective functional using the particles, whose solution yields the Wasserstein gradient direction. Then we update the current distribution by pushing each particle along the direction specified by such a solution. By characterizing both the statistical error incurred in estimating the Wasserstein gradient and the progress of the optimization algorithm, we prove that when the objective function satisfies a functional version of the Polyak-Łojasiewicz (PL) (Polyak, 1963) and smoothness conditions, variational transport converges linearly to the global minimum of the objective functional up to a certain statistical error, which decays to zero sublinearly as the number of particles goes to infinity.

</p>
</details>

<details><summary><b>The Importance of Modeling Data Missingness in Algorithmic Fairness: A Causal Perspective</b>
<a href="https://arxiv.org/abs/2012.11448">arxiv:2012.11448</a>
&#x1F4C8; 3 <br>
<p>Naman Goel, Alfonso Amayuelas, Amit Deshpande, Amit Sharma</p></summary>
<p>

**Abstract:** Training datasets for machine learning often have some form of missingness. For example, to learn a model for deciding whom to give a loan, the available training data includes individuals who were given a loan in the past, but not those who were not. This missingness, if ignored, nullifies any fairness guarantee of the training procedure when the model is deployed. Using causal graphs, we characterize the missingness mechanisms in different real-world scenarios. We show conditions under which various distributions, used in popular fairness algorithms, can or can not be recovered from the training data. Our theoretical results imply that many of these algorithms can not guarantee fairness in practice. Modeling missingness also helps to identify correct design principles for fair algorithms. For example, in multi-stage settings where decisions are made in multiple screening rounds, we use our framework to derive the minimal distributions required to design a fair algorithm. Our proposed algorithm decentralizes the decision-making process and still achieves similar performance to the optimal algorithm that requires centralization and non-recoverable distributions.

</p>
</details>

<details><summary><b>Variational Quantum Cloning: Improving Practicality for Quantum Cryptanalysis</b>
<a href="https://arxiv.org/abs/2012.11424">arxiv:2012.11424</a>
&#x1F4C8; 3 <br>
<p>Brian Coyle, Mina Doosti, Elham Kashefi, Niraj Kumar</p></summary>
<p>

**Abstract:** Cryptanalysis on standard quantum cryptographic systems generally involves finding optimal adversarial attack strategies on the underlying protocols. The core principle of modelling quantum attacks in many cases reduces to the adversary's ability to clone unknown quantum states which facilitates the extraction of some meaningful secret information. Explicit optimal attack strategies typically require high computational resources due to large circuit depths or, in many cases, are unknown. In this work, we propose variational quantum cloning (VQC), a quantum machine learning based cryptanalysis algorithm which allows an adversary to obtain optimal (approximate) cloning strategies with short depth quantum circuits, trained using hybrid classical-quantum techniques. The algorithm contains operationally meaningful cost functions with theoretical guarantees, quantum circuit structure learning and gradient descent based optimisation. Our approach enables the end-to-end discovery of hardware efficient quantum circuits to clone specific families of quantum states, which in turn leads to an improvement in cloning fidelites when implemented on quantum hardware: the Rigetti Aspen chip. Finally, we connect these results to quantum cryptographic primitives, in particular quantum coin flipping. We derive attacks on two protocols as examples, based on quantum cloning and facilitated by VQC. As a result, our algorithm can improve near term attacks on these protocols, using approximate quantum cloning as a resource.

</p>
</details>

<details><summary><b>Spatial Monte Carlo Integration with Annealed Importance Sampling</b>
<a href="https://arxiv.org/abs/2012.11198">arxiv:2012.11198</a>
&#x1F4C8; 3 <br>
<p>Muneki Yasuda, Kaiji Sekimoto</p></summary>
<p>

**Abstract:** Evaluating expectations on a pairwise Boltzmann machine (PBM) (or Ising model) is important for various applications, including the statistical machine learning. However, in general the evaluation is computationally difficult because it involves intractable multiple summations or integrations; therefore, it requires an approximation. Monte Carlo integration (MCI) is a well-known approximation method; a more effective MCI-like approximation method was proposed recently, called spatial Monte Carlo integration (SMCI). However, the estimations obtained from SMCI (and MCI) tend to perform poorly in PBMs with low temperature owing to degradation of the sampling quality. Annealed importance sampling (AIS) is a type of importance sampling based on Markov chain Monte Carlo methods, and it can suppress performance degradation in low temperature regions by the force of importance weights. In this study, a new method is proposed to evaluate the expectations on PBMs combining AIS and SMCI. The proposed method performs efficiently in both high- and low-temperature regions, which is theoretically and numerically demonstrated.

</p>
</details>

<details><summary><b>Mobile Robot Planner with Low-cost Cameras Using Deep Reinforcement Learning</b>
<a href="https://arxiv.org/abs/2012.11160">arxiv:2012.11160</a>
&#x1F4C8; 3 <br>
<p>Minh Q. Tran, Ngoc Q. Ly</p></summary>
<p>

**Abstract:** This study develops a robot mobility policy based on deep reinforcement learning. Since traditional methods of conventional robotic navigation depend on accurate map reproduction as well as require high-end sensors, learning-based methods are positive trends, especially deep reinforcement learning. The problem is modeled in the form of a Markov Decision Process (MDP) with the agent being a mobile robot. Its state of view is obtained by the input sensors such as laser findings or cameras and the purpose is navigating to the goal without any collision. There have been many deep learning methods that solve this problem. However, in order to bring robots to market, low-cost mass production is also an issue that needs to be addressed. Therefore, this work attempts to construct a pseudo laser findings system based on direct depth matrix prediction from a single camera image while still retaining stable performances. Experiment results show that they are directly comparable with others using high-priced sensors.

</p>
</details>

<details><summary><b>A complete, parallel and autonomous photonic neural network in a semiconductor multimode laser</b>
<a href="https://arxiv.org/abs/2012.11153">arxiv:2012.11153</a>
&#x1F4C8; 3 <br>
<p>Xavier Porte, Anas Skalli, Nasibeh Haghighi, Stephan Reitzenstein, James A. Lott, Daniel Brunner</p></summary>
<p>

**Abstract:** Neural networks are one of the disruptive computing concepts of our time. However, they fundamentally differ from classical, algorithmic computing in a number of fundamental aspects. These differences result in equally fundamental, severe and relevant challenges for neural network computing using current computing substrates. Neural networks urge for parallelism across the entire processor and for a co-location of memory and arithmetic, i.e. beyond von Neumann architectures. Parallelism in particular made photonics a highly promising platform, yet until now scalable and integratable concepts are scarce. Here, we demonstrate for the first time how a fully parallel and fully implemented photonic neural network can be realized using spatially distributed modes of an efficient and fast semiconductor laser. Importantly, all neural network connections are realized in hardware, and our processor produces results without pre- or post-processing. 130+ nodes are implemented in a large-area vertical cavity surface emitting laser, input and output weights are realized via the complex transmission matrix of a multimode fiber and a digital micro-mirror array, respectively. We train the readout weights to perform 2-bit header recognition, a 2-bit XOR and 2-bit digital analog conversion, and obtain < 0.9 10^-3 and 2.9 10^-2 error rates for digit recognition and XOR, respectively. Finally, the digital analog conversion can be realized with a standard deviation of only 5.4 10^-2. Our system is scalable to much larger sizes and to bandwidths in excess of 20 GHz.

</p>
</details>

<details><summary><b>A Review of Artificial Intelligence Technologies for Early Prediction of Alzheimer's Disease</b>
<a href="https://arxiv.org/abs/2101.01781">arxiv:2101.01781</a>
&#x1F4C8; 2 <br>
<p>Kuo Yang, Emad A. Mohammed</p></summary>
<p>

**Abstract:** Alzheimer's Disease (AD) is a severe brain disorder, destroying memories and brain functions. AD causes chronically, progressively, and irreversibly cognitive declination and brain damages. The reliable and effective evaluation of early dementia has become essential research with medical imaging technologies and computer-aided algorithms. This trend has moved to modern Artificial Intelligence (AI) technologies motivated by deeplearning success in image classification and natural language processing. The purpose of this review is to provide an overview of the latest research involving deep-learning algorithms in evaluating the process of dementia, diagnosing the early stage of AD, and discussing an outlook for this research. This review introduces various applications of modern AI algorithms in AD diagnosis, including Convolutional Neural Network (CNN), Recurrent Neural Network (RNN), Automatic Image Segmentation, Autoencoder, Graph CNN (GCN), Ensemble Learning, and Transfer Learning. The advantages and disadvantages of the proposed methods and their performance are discussed. The conclusion section summarizes the primary contributions and medical imaging preprocessing techniques applied in the reviewed research. Finally, we discuss the limitations and future outlooks.

</p>
</details>

<details><summary><b>Understanding Health Misinformation Transmission: An Interpretable Deep Learning Approach to Manage Infodemics</b>
<a href="https://arxiv.org/abs/2101.01076">arxiv:2101.01076</a>
&#x1F4C8; 2 <br>
<p>Jiaheng Xie, Yidong Chai, Xiao Liu</p></summary>
<p>

**Abstract:** Health misinformation on social media devastates physical and mental health, invalidates health gains, and potentially costs lives. Understanding how health misinformation is transmitted is an urgent goal for researchers, social media platforms, health sectors, and policymakers to mitigate those ramifications. Deep learning methods have been deployed to predict the spread of misinformation. While achieving the state-of-the-art predictive performance, deep learning methods lack the interpretability due to their blackbox nature. To remedy this gap, this study proposes a novel interpretable deep learning approach, Generative Adversarial Network based Piecewise Wide and Attention Deep Learning (GAN-PiWAD), to predict health misinformation transmission in social media. Improving upon state-of-the-art interpretable methods, GAN-PiWAD captures the interactions among multi-modal data, offers unbiased estimation of the total effect of each feature, and models the dynamic total effect of each feature when its value varies. We select features according to social exchange theory and evaluate GAN-PiWAD on 4,445 misinformation videos. The proposed approach outperformed strong benchmarks. Interpretation of GAN-PiWAD indicates video description, negative video content, and channel credibility are key features that drive viral transmission of misinformation. This study contributes to IS with a novel interpretable deep learning method that is generalizable to understand other human decision factors. Our findings provide direct implications for social media platforms and policymakers to design proactive interventions to identify misinformation, control transmissions, and manage infodemics.

</p>
</details>

<details><summary><b>NetReAct: Interactive Learning for Network Summarization</b>
<a href="https://arxiv.org/abs/2012.11821">arxiv:2012.11821</a>
&#x1F4C8; 2 <br>
<p>Sorour E. Amiri, Bijaya Adhikari, John Wenskovitch, Alexander Rodriguez, Michelle Dowling, Chris North, B. Aditya Prakash</p></summary>
<p>

**Abstract:** Generating useful network summaries is a challenging and important problem with several applications like sensemaking, visualization, and compression. However, most of the current work in this space do not take human feedback into account while generating summaries. Consider an intelligence analysis scenario, where the analyst is exploring a similarity network between documents. The analyst can express her agreement/disagreement with the visualization of the network summary via iterative feedback, e.g. closing or moving documents ("nodes") together. How can we use this feedback to improve the network summary quality? In this paper, we present NetReAct, a novel interactive network summarization algorithm which supports the visualization of networks induced by text corpora to perform sensemaking. NetReAct incorporates human feedback with reinforcement learning to summarize and visualize document networks. Using scenarios from two datasets, we show how NetReAct is successful in generating high-quality summaries and visualizations that reveal hidden patterns better than other non-trivial baselines.

</p>
</details>

<details><summary><b>Semi-Supervised Disentangled Framework for Transferable Named Entity Recognition</b>
<a href="https://arxiv.org/abs/2012.11805">arxiv:2012.11805</a>
&#x1F4C8; 2 <br>
<p>Zhifeng Hao, Di Lv, Zijian Li, Ruichu Cai, Wen Wen, Boyan Xu</p></summary>
<p>

**Abstract:** Named entity recognition (NER) for identifying proper nouns in unstructured text is one of the most important and fundamental tasks in natural language processing. However, despite the widespread use of NER models, they still require a large-scale labeled data set, which incurs a heavy burden due to manual annotation. Domain adaptation is one of the most promising solutions to this problem, where rich labeled data from the relevant source domain are utilized to strengthen the generalizability of a model based on the target domain. However, the mainstream cross-domain NER models are still affected by the following two challenges (1) Extracting domain-invariant information such as syntactic information for cross-domain transfer. (2) Integrating domain-specific information such as semantic information into the model to improve the performance of NER. In this study, we present a semi-supervised framework for transferable NER, which disentangles the domain-invariant latent variables and domain-specific latent variables. In the proposed framework, the domain-specific information is integrated with the domain-specific latent variables by using a domain predictor. The domain-specific and domain-invariant latent variables are disentangled using three mutual information regularization terms, i.e., maximizing the mutual information between the domain-specific latent variables and the original embedding, maximizing the mutual information between the domain-invariant latent variables and the original embedding, and minimizing the mutual information between the domain-specific and domain-invariant latent variables. Extensive experiments demonstrated that our model can obtain state-of-the-art performance with cross-domain and cross-lingual NER benchmark data sets.

</p>
</details>

<details><summary><b>To Talk or to Work: Flexible Communication Compression for Energy Efficient Federated Learning over Heterogeneous Mobile Edge Devices</b>
<a href="https://arxiv.org/abs/2012.11804">arxiv:2012.11804</a>
&#x1F4C8; 2 <br>
<p>Liang Li, Dian Shi, Ronghui Hou, Hui Li, Miao Pan, Zhu Han</p></summary>
<p>

**Abstract:** Recent advances in machine learning, wireless communication, and mobile hardware technologies promisingly enable federated learning (FL) over massive mobile edge devices, which opens new horizons for numerous intelligent mobile applications. Despite the potential benefits, FL imposes huge communication and computation burdens on participating devices due to periodical global synchronization and continuous local training, raising great challenges to battery constrained mobile devices. In this work, we target at improving the energy efficiency of FL over mobile edge networks to accommodate heterogeneous participating devices without sacrificing the learning performance. To this end, we develop a convergence-guaranteed FL algorithm enabling flexible communication compression. Guided by the derived convergence bound, we design a compression control scheme to balance the energy consumption of local computing (i.e., "working") and wireless communication (i.e., "talking") from the long-term learning perspective. In particular, the compression parameters are elaborately chosen for FL participants adapting to their computing and communication environments. Extensive simulations are conducted using various datasets to validate our theoretical analysis, and the results also demonstrate the efficacy of the proposed scheme in energy saving.

</p>
</details>

<details><summary><b>Scalable Deep Reinforcement Learning for Routing and Spectrum Access in Physical Layer</b>
<a href="https://arxiv.org/abs/2012.11783">arxiv:2012.11783</a>
&#x1F4C8; 2 <br>
<p>Wei Cui, Wei Yu</p></summary>
<p>

**Abstract:** This paper proposes a novel and scalable reinforcement learning approach for simultaneous routing and spectrum access in wireless ad-hoc networks. In most previous works on reinforcement learning for network optimization, routing and spectrum access are tackled as separate tasks; further, the wireless links in the network are assumed to be fixed, and a different agent is trained for each transmission node -- this limits scalability and generalizability. In this paper, we account for the inherent signal-to-interference-plus-noise ratio (SINR) in the physical layer and propose a more scalable approach in which a single agent is associated with each flow. Specifically, a single agent makes all routing and spectrum access decisions as it moves along the frontier nodes of each flow. The agent is trained according to the physical layer characteristics of the environment using the future bottleneck SINR as a novel reward definition. This allows a highly effective routing strategy based on the geographic locations of the nodes in the wireless ad-hoc network. The proposed deep reinforcement learning strategy is capable of accounting for the mutual interference between the links. It learns to avoid interference by intelligently allocating spectrum slots and making routing decisions for the entire network in a scalable manner.

</p>
</details>

<details><summary><b>Differentially Private Synthetic Medical Data Generation using Convolutional GANs</b>
<a href="https://arxiv.org/abs/2012.11774">arxiv:2012.11774</a>
&#x1F4C8; 2 <br>
<p>Amirsina Torfi, Edward A. Fox, Chandan K. Reddy</p></summary>
<p>

**Abstract:** Deep learning models have demonstrated superior performance in several application problems, such as image classification and speech processing. However, creating a deep learning model using health record data requires addressing certain privacy challenges that bring unique concerns to researchers working in this domain. One effective way to handle such private data issues is to generate realistic synthetic data that can provide practically acceptable data quality and correspondingly the model performance. To tackle this challenge, we develop a differentially private framework for synthetic data generation using Rényi differential privacy. Our approach builds on convolutional autoencoders and convolutional generative adversarial networks to preserve some of the critical characteristics of the generated synthetic data. In addition, our model can also capture the temporal information and feature correlations that might be present in the original data. We demonstrate that our model outperforms existing state-of-the-art models under the same privacy budget using several publicly available benchmark medical datasets in both supervised and unsupervised settings.

</p>
</details>

<details><summary><b>Power-SLIC: Diagram-based superpixel generation</b>
<a href="https://arxiv.org/abs/2012.11772">arxiv:2012.11772</a>
&#x1F4C8; 2 <br>
<p>Maximilian Fiedler, Andreas Alpers</p></summary>
<p>

**Abstract:** Superpixel algorithms, which group pixels similar in color and other low-level properties, are increasingly used for pre-processing in image segmentation. Commonly important criteria for the computation of superpixels are boundary adherence, speed, and regularity.
  Boundary adherence and regularity are typically contradictory goals. Most recent algorithms have focused on improving boundary adherence. Motivated by improving superpixel regularity, we propose a diagram-based superpixel generation method called Power-SLIC.
  On the BSDS500 data set, Power-SLIC outperforms other state-of-the-art algorithms in terms of compactness and boundary precision, and its boundary adherence is the most robust against varying levels of Gaussian noise. In terms of speed, Power-SLIC is competitive with SLIC.

</p>
</details>

<details><summary><b>Self-Progressing Robust Training</b>
<a href="https://arxiv.org/abs/2012.11769">arxiv:2012.11769</a>
&#x1F4C8; 2 <br>
<p>Minhao Cheng, Pin-Yu Chen, Sijia Liu, Shiyu Chang, Cho-Jui Hsieh, Payel Das</p></summary>
<p>

**Abstract:** Enhancing model robustness under new and even adversarial environments is a crucial milestone toward building trustworthy machine learning systems. Current robust training methods such as adversarial training explicitly uses an "attack" (e.g., $\ell_{\infty}$-norm bounded perturbation) to generate adversarial examples during model training for improving adversarial robustness. In this paper, we take a different perspective and propose a new framework called SPROUT, self-progressing robust training. During model training, SPROUT progressively adjusts training label distribution via our proposed parametrized label smoothing technique, making training free of attack generation and more scalable. We also motivate SPROUT using a general formulation based on vicinity risk minimization, which includes many robust training methods as special cases. Compared with state-of-the-art adversarial training methods (PGD-l_inf and TRADES) under l_inf-norm bounded attacks and various invariance tests, SPROUT consistently attains superior performance and is more scalable to large neural networks. Our results shed new light on scalable, effective and attack-independent robust training methods.

</p>
</details>

<details><summary><b>Contraband Materials Detection Within Volumetric 3D Computed Tomography Baggage Security Screening Imagery</b>
<a href="https://arxiv.org/abs/2012.11753">arxiv:2012.11753</a>
&#x1F4C8; 2 <br>
<p>Qian Wang, Toby P. Breckon</p></summary>
<p>

**Abstract:** Automatic prohibited object detection within 2D/3D X-ray Computed Tomography (CT) has been studied in literature to enhance the aviation security screening at checkpoints. Deep Convolutional Neural Networks (CNN) have demonstrated superior performance in 2D X-ray imagery. However, there exists very limited proof of how deep neural networks perform in materials detection within volumetric 3D CT baggage screening imagery. We attempt to close this gap by applying Deep Neural Networks in 3D contraband substance detection based on their material signatures. Specifically, we formulate it as a 3D semantic segmentation problem to identify material types for all voxels based on which contraband materials can be detected. To this end, we firstly investigate 3D CNN based semantic segmentation algorithms such as 3D U-Net and its variants. In contrast to the original dense representation form of volumetric 3D CT data, we propose to convert the CT volumes into sparse point clouds which allows the use of point cloud processing approaches such as PointNet++ towards more efficient processing. Experimental results on a publicly available dataset (NEU ATR) demonstrate the effectiveness of both 3D U-Net and PointNet++ in materials detection in 3D CT imagery for baggage security screening.

</p>
</details>

<details><summary><b>A Fast Edge-Based Synchronizer for Tasks in Real-Time Artificial Intelligence Applications</b>
<a href="https://arxiv.org/abs/2012.11731">arxiv:2012.11731</a>
&#x1F4C8; 2 <br>
<p>Richard Olaniyan, Muthucumaru Maheswaran</p></summary>
<p>

**Abstract:** Real-time artificial intelligence (AI) applications mapped onto edge computing need to perform data capture, process data, and device actuation within given bounds while using the available devices. Task synchronization across the devices is an important problem that affects the timely progress of an AI application by determining the quality of the captured data, time to process the data, and the quality of actuation. In this paper, we develop a fast edge-based synchronization scheme that can time align the execution of input-output tasks as well compute tasks. The primary idea of the fast synchronizer is to cluster the devices into groups that are highly synchronized in their task executions and statically determine few synchronization points using a game-theoretic solver. The cluster of devices use a late notification protocol to select the best point among the pre-computed synchronization points to reach a time aligned task execution as quickly as possible. We evaluate the performance of our synchronization scheme using trace-driven simulations and we compare the performance with existing distributed synchronization schemes for real-time AI application tasks. We implement our synchronization scheme and compare its training accuracy and training time with other parameter server synchronization frameworks.

</p>
</details>

<details><summary><b>Cross-Domain Latent Modulation for Variational Transfer Learning</b>
<a href="https://arxiv.org/abs/2012.11727">arxiv:2012.11727</a>
&#x1F4C8; 2 <br>
<p>Jinyong Hou, Jeremiah D. Deng, Stephen Cranefield, Xuejie Ding</p></summary>
<p>

**Abstract:** We propose a cross-domain latent modulation mechanism within a variational autoencoders (VAE) framework to enable improved transfer learning. Our key idea is to procure deep representations from one data domain and use it as perturbation to the reparameterization of the latent variable in another domain. Specifically, deep representations of the source and target domains are first extracted by a unified inference model and aligned by employing gradient reversal. Second, the learned deep representations are cross-modulated to the latent encoding of the alternate domain. The consistency between the reconstruction from the modulated latent encoding and the generation using deep representation samples is then enforced in order to produce inter-class alignment in the latent space. We apply the proposed model to a number of transfer learning tasks including unsupervised domain adaptation and image-toimage translation. Experimental results show that our model gives competitive performance.

</p>
</details>

<details><summary><b>Image Captioning as an Assistive Technology: Lessons Learned from VizWiz 2020 Challenge</b>
<a href="https://arxiv.org/abs/2012.11696">arxiv:2012.11696</a>
&#x1F4C8; 2 <br>
<p>Pierre Dognin, Igor Melnyk, Youssef Mroueh, Inkit Padhi, Mattia Rigotti, Jarret Ross, Yair Schiff, Richard A. Young, Brian Belgodere</p></summary>
<p>

**Abstract:** Image captioning has recently demonstrated impressive progress largely owing to the introduction of neural network algorithms trained on curated dataset like MS-COCO. Often work in this field is motivated by the promise of deployment of captioning systems in practical applications. However, the scarcity of data and contexts in many competition datasets renders the utility of systems trained on these datasets limited as an assistive technology in real-world settings, such as helping visually impaired people navigate and accomplish everyday tasks. This gap motivated the introduction of the novel VizWiz dataset, which consists of images taken by the visually impaired and captions that have useful, task-oriented information. In an attempt to help the machine learning computer vision field realize its promise of producing technologies that have positive social impact, the curators of the VizWiz dataset host several competitions, including one for image captioning. This work details the theory and engineering from our winning submission to the 2020 captioning competition. Our work provides a step towards improved assistive image captioning systems.

</p>
</details>

<details><summary><b>Rapidly adapting robot swarms with Swarm Map-based Bayesian Optimisation</b>
<a href="https://arxiv.org/abs/2012.11444">arxiv:2012.11444</a>
&#x1F4C8; 2 <br>
<p>David M. Bossens, Danesh Tarapore</p></summary>
<p>

**Abstract:** Rapid performance recovery from unforeseen environmental perturbations remains a grand challenge in swarm robotics. To solve this challenge, we investigate a behaviour adaptation approach, where one searches an archive of controllers for potential recovery solutions. To apply behaviour adaptation in swarm robotic systems, we propose two algorithms: (i) Swarm Map-based Optimisation (SMBO), which selects and evaluates one controller at a time, for a homogeneous swarm, in a centralised fashion; and (ii) Swarm Map-based Optimisation Decentralised (SMBO-Dec), which performs an asynchronous batch-based Bayesian optimisation to simultaneously explore different controllers for groups of robots in the swarm. We set up foraging experiments with a variety of disturbances: injected faults to proximity sensors, ground sensors, and the actuators of individual robots, with 100 unique combinations for each type. We also investigate disturbances in the operating environment of the swarm, where the swarm has to adapt to drastic changes in the number of resources available in the environment, and to one of the robots behaving disruptively towards the rest of the swarm, with 30 unique conditions for each such perturbation. The viability of SMBO and SMBO-Dec is demonstrated, comparing favourably to variants of random search and gradient descent, and various ablations, and improving performance up to 80% compared to the performance at the time of fault injection within at most 30 evaluations.

</p>
</details>

<details><summary><b>Unifying Homophily and Heterophily Network Transformation via Motifs</b>
<a href="https://arxiv.org/abs/2012.11400">arxiv:2012.11400</a>
&#x1F4C8; 2 <br>
<p>Yan Ge, Jun Ma, Li Zhang, Haiping Lu</p></summary>
<p>

**Abstract:** Higher-order proximity (HOP) is fundamental for most network embedding methods due to its significant effects on the quality of node embedding and performance on downstream network analysis tasks. Most existing HOP definitions are based on either homophily to place close and highly interconnected nodes tightly in embedding space or heterophily to place distant but structurally similar nodes together after embedding. In real-world networks, both can co-exist, and thus considering only one could limit the prediction performance and interpretability. However, there is no general and universal solution that takes both into consideration. In this paper, we propose such a simple yet powerful framework called homophily and heterophliy preserving network transformation (H2NT) to capture HOP that flexibly unifies homophily and heterophily. Specifically, H2NT utilises motif representations to transform a network into a new network with a hybrid assumption via micro-level and macro-level walk paths. H2NT can be used as an enhancer to be integrated with any existing network embedding methods without requiring any changes to latter methods. Because H2NT can sparsify networks with motif structures, it can also improve the computational efficiency of existing network embedding methods when integrated. We conduct experiments on node classification, structural role classification and motif prediction to show the superior prediction performance and computational efficiency over state-of-the-art methods. In particular, DeepWalk-based H2 NT achieves 24% improvement in terms of precision on motif prediction, while reducing 46% computational time compared to the original DeepWalk.

</p>
</details>

<details><summary><b>Difference Rewards Policy Gradients</b>
<a href="https://arxiv.org/abs/2012.11258">arxiv:2012.11258</a>
&#x1F4C8; 2 <br>
<p>Jacopo Castellini, Sam Devlin, Frans A. Oliehoek, Rahul Savani</p></summary>
<p>

**Abstract:** Policy gradient methods have become one of the most popular classes of algorithms for multi-agent reinforcement learning. A key challenge, however, that is not addressed by many of these methods is multi-agent credit assignment: assessing an agent's contribution to the overall performance, which is crucial for learning good policies. We propose a novel algorithm called Dr.Reinforce that explicitly tackles this by combining difference rewards with policy gradients to allow for learning decentralized policies when the reward function is known. By differencing the reward function directly, Dr.Reinforce avoids difficulties associated with learning the Q-function as done by Counterfactual Multiagent Policy Gradients (COMA), a state-of-the-art difference rewards method. For applications where the reward function is unknown, we show the effectiveness of a version of Dr.Reinforce that learns an additional reward network that is used to estimate the difference rewards.

</p>
</details>

<details><summary><b>Neural Joint Entropy Estimation</b>
<a href="https://arxiv.org/abs/2012.11197">arxiv:2012.11197</a>
&#x1F4C8; 2 <br>
<p>Yuval Shalev, Amichai Painsky, Irad Ben-Gal</p></summary>
<p>

**Abstract:** Estimating the entropy of a discrete random variable is a fundamental problem in information theory and related fields. This problem has many applications in various domains, including machine learning, statistics and data compression. Over the years, a variety of estimation schemes have been suggested. However, despite significant progress, most methods still struggle when the sample is small, compared to the variable's alphabet size. In this work, we introduce a practical solution to this problem, which extends the work of McAllester and Statos (2020). The proposed scheme uses the generalization abilities of cross-entropy estimation in deep neural networks (DNNs) to introduce improved entropy estimation accuracy. Furthermore, we introduce a family of estimators for related information-theoretic measures, such as conditional entropy and mutual information. We show that these estimators are strongly consistent and demonstrate their performance in a variety of use-cases. First, we consider large alphabet entropy estimation. Then, we extend the scope to mutual information estimation. Next, we apply the proposed scheme to conditional mutual information estimation, as we focus on independence testing tasks. Finally, we study a transfer entropy estimation problem. The proposed estimators demonstrate improved performance compared to existing methods in all tested setups.

</p>
</details>

<details><summary><b>Unsupervised Cross-Lingual Speech Emotion Recognition Using DomainAdversarial Neural Network</b>
<a href="https://arxiv.org/abs/2012.11174">arxiv:2012.11174</a>
&#x1F4C8; 2 <br>
<p>Xiong Cai, Zhiyong Wu, Kuo Zhong, Bin Su, Dongyang Dai, Helen Meng</p></summary>
<p>

**Abstract:** By using deep learning approaches, Speech Emotion Recog-nition (SER) on a single domain has achieved many excellentresults. However, cross-domain SER is still a challenging taskdue to the distribution shift between source and target domains.In this work, we propose a Domain Adversarial Neural Net-work (DANN) based approach to mitigate this distribution shiftproblem for cross-lingual SER. Specifically, we add a languageclassifier and gradient reversal layer after the feature extractor toforce the learned representation both language-independent andemotion-meaningful. Our method is unsupervised, i. e., labelson target language are not required, which makes it easier to ap-ply our method to other languages. Experimental results showthe proposed method provides an average absolute improve-ment of 3.91% over the baseline system for arousal and valenceclassification task. Furthermore, we find that batch normaliza-tion is beneficial to the performance gain of DANN. Thereforewe also explore the effect of different ways of data combinationfor batch normalization.

</p>
</details>

<details><summary><b>A Comprehensive Survey of Machine Learning Based Localization with Wireless Signals</b>
<a href="https://arxiv.org/abs/2012.11171">arxiv:2012.11171</a>
&#x1F4C8; 2 <br>
<p>Daoud Burghal, Ashwin T. Ravi, Varun Rao, Abdullah A. Alghafis, Andreas F. Molisch</p></summary>
<p>

**Abstract:** The last few decades have witnessed a growing interest in location-based services. Using localization systems based on Radio Frequency (RF) signals has proven its efficacy for both indoor and outdoor applications. However, challenges remain with respect to both complexity and accuracy of such systems. Machine Learning (ML) is one of the most promising methods for mitigating these problems, as ML (especially deep learning) offers powerful practical data-driven tools that can be integrated into localization systems. In this paper, we provide a comprehensive survey of ML-based localization solutions that use RF signals. The survey spans different aspects, ranging from the system architectures, to the input features, the ML methods, and the datasets.
  A main point of the paper is the interaction between the domain knowledge arising from the physics of localization systems, and the various ML approaches. Besides the ML methods, the utilized input features play a major role in shaping the localization solution; we present a detailed discussion of the different features and what could influence them, be it the underlying wireless technology or standards or the preprocessing techniques. A detailed discussion is dedicated to the different ML methods that have been applied to localization problems, discussing the underlying problem and the solution structure. Furthermore, we summarize the different ways the datasets were acquired, and then list the publicly available ones. Overall, the survey categorizes and partly summarizes insights from almost 400 papers in this field.
  This survey is self-contained, as we provide a concise review of the main ML and wireless propagation concepts, which shall help the researchers in either field navigate through the surveyed solutions, and suggested open problems.

</p>
</details>

<details><summary><b>Multi-stream Convolutional Neural Network with Frequency Selection for Robust Speaker Verification</b>
<a href="https://arxiv.org/abs/2012.11159">arxiv:2012.11159</a>
&#x1F4C8; 2 <br>
<p>Wei Yao, Shen Chen, Jiamin Cui, Yaolin Lou</p></summary>
<p>

**Abstract:** Speaker verification aims to verify whether an input speech corresponds to the claimed speaker, and conventionally, this kind of system is deployed based on single-stream scenario, wherein the feature extractor operates in full frequency range. In this paper, we hypothesize that machine can learn enough knowledge to do classification task when listening to partial frequency range instead of full frequency range, which is so called frequency selection technique, and further propose a novel framework of multi-stream Convolutional Neural Network (CNN) with this technique for speaker verification tasks. The proposed framework accommodates diverse temporal embeddings generated from multiple streams to enhance the robustness of acoustic modeling. For the diversity of temporal embeddings, we consider feature augmentation with frequency selection, which is to manually segment the full-band of frequency into several sub-bands, and the feature extractor of each stream can select which sub-bands to use as target frequency domain. Different from conventional single-stream solution wherein each utterance would only be processed for one time, in this framework, there are multiple streams processing it in parallel. The input utterance for each stream is pre-processed by a frequency selector within specified frequency range, and post-processed by mean normalization. The normalized temporal embeddings of each stream will flow into a pooling layer to generate fused embeddings. We conduct extensive experiments on VoxCeleb dataset, and the experimental results demonstrate that multi-stream CNN significantly outperforms single-stream baseline with 20.53 % of relative improvement in minimum Decision Cost Function (minDCF).

</p>
</details>

<details><summary><b>Efficient On-Chip Learning for Optical Neural Networks Through Power-Aware Sparse Zeroth-Order Optimization</b>
<a href="https://arxiv.org/abs/2012.11148">arxiv:2012.11148</a>
&#x1F4C8; 2 <br>
<p>Jiaqi Gu, Chenghao Feng, Zheng Zhao, Zhoufeng Ying, Ray T. Chen, David Z. Pan</p></summary>
<p>

**Abstract:** Optical neural networks (ONNs) have demonstrated record-breaking potential in high-performance neuromorphic computing due to their ultra-high execution speed and low energy consumption. However, current learning protocols fail to provide scalable and efficient solutions to photonic circuit optimization in practical applications. In this work, we propose a novel on-chip learning framework to release the full potential of ONNs for power-efficient in situ training. Instead of deploying implementation-costly back-propagation, we directly optimize the device configurations with computation budgets and power constraints. We are the first to model the ONN on-chip learning as a resource-constrained stochastic noisy zeroth-order optimization problem, and propose a novel mixed-training strategy with two-level sparsity and power-aware dynamic pruning to offer a scalable on-chip training solution in practical ONN deployment. Compared with previous methods, we are the first to optimize over 2,500 optical components on chip. We can achieve much better optimization stability, 3.7x-7.6x higher efficiency, and save >90% power under practical device variations and thermal crosstalk.

</p>
</details>

<details><summary><b>APIK: Active Physics-Informed Kriging Model with Partial Differential Equations</b>
<a href="https://arxiv.org/abs/2012.11798">arxiv:2012.11798</a>
&#x1F4C8; 1 <br>
<p>Jialei Chen, Zhehui Chen, Chuck Zhang, C. F. Jeff Wu</p></summary>
<p>

**Abstract:** Kriging (or Gaussian process regression) is a popular machine learning method for its flexibility and closed-form prediction expressions. However, one of the key challenges in applying kriging to engineering systems is that the available measurement data is scarce due to the measurement limitations and high sensing costs. On the other hand, physical knowledge of the engineering system is often available and represented in the form of partial differential equations (PDEs). We present in this work a PDE Informed Kriging model (PIK), which introduces PDE information via a set of PDE points and conducts posterior prediction similar to the standard kriging method. The proposed PIK model can incorporate physical knowledge from both linear and nonlinear PDEs. To further improve learning performance, we propose an Active PIK framework (APIK) that designs PDE points to leverage the PDE information based on the PIK model and measurement data. The selected PDE points not only explore the whole input space but also exploit the locations where the PDE information is critical in reducing predictive uncertainty. Finally, an expectation-maximization algorithm is developed for parameter estimation. We demonstrate the effectiveness of APIK in two synthetic examples, a shock wave case study, and a laser heating case study.

</p>
</details>

<details><summary><b>Tight Bounds on the Smallest Eigenvalue of the Neural Tangent Kernel for Deep ReLU Networks</b>
<a href="https://arxiv.org/abs/2012.11654">arxiv:2012.11654</a>
&#x1F4C8; 1 <br>
<p>Quynh Nguyen, Marco Mondelli, Guido Montufar</p></summary>
<p>

**Abstract:** A recent line of work has analyzed the theoretical properties of deep neural networks via the Neural Tangent Kernel (NTK). In particular, the smallest eigenvalue of the NTK has been related to memorization capacity, convergence of gradient descent algorithms and generalization of deep nets. However, existing results either provide bounds in the two-layer setting or assume that the spectrum of the NTK is bounded away from 0 for multi-layer networks. In this paper, we provide tight bounds on the smallest eigenvalue of NTK matrices for deep ReLU networks, both in the limiting case of infinite widths and for finite widths. In the finite-width setting, the network architectures we consider are quite general: we require the existence of a wide layer with roughly order of $N$ neurons, $N$ being the number of data samples; and the scaling of the remaining widths is arbitrary (up to logarithmic factors). To obtain our results, we analyze various quantities of independent interest: we give lower bounds on the smallest singular value of feature matrices, and upper bounds on the Lipschitz constant of input-output feature maps.

</p>
</details>

<details><summary><b>Defence against adversarial attacks using classical and quantum-enhanced Boltzmann machines</b>
<a href="https://arxiv.org/abs/2012.11619">arxiv:2012.11619</a>
&#x1F4C8; 1 <br>
<p>Aidan Kehoe, Peter Wittek, Yanbo Xue, Alejandro Pozas-Kerstjens</p></summary>
<p>

**Abstract:** We provide a robust defence to adversarial attacks on discriminative algorithms. Neural networks are naturally vulnerable to small, tailored perturbations in the input data that lead to wrong predictions. On the contrary, generative models attempt to learn the distribution underlying a dataset, making them inherently more robust to small perturbations. We use Boltzmann machines for discrimination purposes as attack-resistant classifiers, and compare them against standard state-of-the-art adversarial defences. We find improvements ranging from 5% to 72% against attacks with Boltzmann machines on the MNIST dataset. We furthermore complement the training with quantum-enhanced sampling from the D-Wave 2000Q annealer, finding results comparable with classical techniques and with marginal improvements in some cases. These results underline the relevance of probabilistic methods in constructing neural networks and demonstrate the power of quantum computers, even with limited hardware capabilities. This work is dedicated to the memory of Peter Wittek.

</p>
</details>

<details><summary><b>The COVID-19 pandemic: socioeconomic and health disparities</b>
<a href="https://arxiv.org/abs/2012.11399">arxiv:2012.11399</a>
&#x1F4C8; 1 <br>
<p>Behzad Javaheri</p></summary>
<p>

**Abstract:** Disadvantaged groups around the world have suffered and endured higher mortality during the current COVID-19 pandemic. This contrast disparity suggests that socioeconomic and health-related factors may drive inequality in disease outcome. To identify these factors correlated with COVID-19 outcome, country aggregate data provided by the Lancet COVID-19 Commission subjected to correlation analysis. Socioeconomic and health-related variables were used to predict mortality in the top 5 most affected countries using ridge regression and extreme gradient boosting (XGBoost) models. Our data reveal that predictors related to demographics and social disadvantage correlate with COVID-19 mortality per million and that XGBoost performed better than ridge regression. Taken together, our findings suggest that the health consequence of the current pandemic is not just confined to indiscriminate impact of a viral infection but that these preventable effects are amplified based on pre-existing health and socioeconomic inequalities.

</p>
</details>

<details><summary><b>Incremental Verification of Fixed-Point Implementations of Neural Networks</b>
<a href="https://arxiv.org/abs/2012.11220">arxiv:2012.11220</a>
&#x1F4C8; 1 <br>
<p>Luiz Sena, Erickson Alves, Iury Bessa, Eddie Filho, Lucas Cordeiro</p></summary>
<p>

**Abstract:** Implementations of artificial neural networks (ANNs) might lead to failures, which are hardly predicted in the design phase since ANNs are highly parallel and their parameters are barely interpretable. Here, we develop and evaluate a novel symbolic verification framework using incremental bounded model checking (BMC), satisfiability modulo theories (SMT), and invariant inference, to obtain adversarial cases and validate coverage methods in a multi-layer perceptron (MLP). We exploit incremental BMC based on interval analysis to compute boundaries from a neuron's input. Then, the latter are propagated to effectively find a neuron's output since it is the input of the next one. This paper describes the first bit-precise symbolic verification framework to reason over actual implementations of ANNs in CUDA, based on invariant inference, therefore providing further guarantees about finite-precision arithmetic and its rounding errors, which are routinely ignored in the existing literature. We have implemented the proposed approach on top of the efficient SMT-based bounded model checker (ESBMC), and its experimental results show that it can successfully verify safety properties, in actual implementations of ANNs, and generate real adversarial cases in MLPs. Our approach was able to verify and produce adversarial examples for 85.8% of 21 test cases considering different input images, and 100% of the properties related to covering methods. Although our verification time is higher than existing approaches, our methodology can consider fixed-point implementation aspects that are disregarded by the state-of-the-art verification methodologies.

</p>
</details>

<details><summary><b>Can I Still Trust You?: Understanding the Impact of Distribution Shifts on Algorithmic Recourses</b>
<a href="https://arxiv.org/abs/2012.11788">arxiv:2012.11788</a>
&#x1F4C8; 0 <br>
<p>Kaivalya Rawal, Ece Kamar, Himabindu Lakkaraju</p></summary>
<p>

**Abstract:** As predictive models are being increasingly deployed to make a variety of consequential decisions ranging from hiring decisions to loan approvals, there is growing emphasis on designing algorithms that can provide reliable recourses to affected individuals. To this end, several recourse generation algorithms have been proposed in recent literature. However, there is little to no work on systematically assessing if these algorithms are actually generating recourses that are reliable. In this work, we assess the reliability of algorithmic recourses through the lens of distribution shifts i.e., we empirically and theoretically study if and what kind of recourses generated by state-of-the-art algorithms are robust to distribution shifts. To the best of our knowledge, this work makes the first attempt at addressing this critical question. We experiment with multiple synthetic and real world datasets capturing different kinds of distribution shifts including temporal shifts, geospatial shifts, and shifts due to data corrections. Our results demonstrate that all the aforementioned distribution shifts could potentially invalidate the recourses generated by state-of-the-art algorithms. In addition, we also find that recourse interventions themselves may cause distribution shifts which in turn invalidate previously prescribed recourses. Our theoretical results establish that the recourses (counterfactuals) that are close to the model decision boundary are more likely to be invalidated upon model updation. However, state-of-the-art algorithms tend to prefer exactly these recourses because their cost functions penalize recourses (counterfactuals) that require large modifications to the original instance. Our findings not only expose fundamental flaws in recourse finding strategies but also pave new way for rethinking the design and development of recourse generation algorithms.

</p>
</details>

<details><summary><b>Adversarial Training for a Continuous Robustness Control Problem in Power Systems</b>
<a href="https://arxiv.org/abs/2012.11390">arxiv:2012.11390</a>
&#x1F4C8; 0 <br>
<p>Loïc Omnes, Antoine Marot, Benjamin Donnot</p></summary>
<p>

**Abstract:** We propose a new adversarial training approach for injecting robustness when designing controllers for upcoming cyber-physical power systems. Previous approaches relying deeply on simulations are not able to cope with the rising complexity and are too costly when used online in terms of computation budget. In comparison, our method proves to be computationally efficient online while displaying useful robustness properties. To do so we model an adversarial framework, propose the implementation of a fixed opponent policy and test it on a L2RPN (Learning to Run a Power Network) environment. That environment is a synthetic but realistic modeling of a cyber-physical system accounting for one third of the IEEE 118 grid. Using adversarial testing, we analyze the results of submitted trained agents from the robustness track of the L2RPN competition. We then further assess the performance of those agents in regards to the continuous N-1 problem through tailored evaluation metrics. We discover that some agents trained in an adversarial way demonstrate interesting preventive behaviors in that regard, which we discuss.

</p>
</details>


[Next Page](2020/2020-12/2020-12-20.md)
