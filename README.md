## Summary for 2021-11-02, created on 2021-12-14


<details><summary><b>StyleGAN of All Trades: Image Manipulation with Only Pretrained StyleGAN</b>
<a href="https://arxiv.org/abs/2111.01619">arxiv:2111.01619</a>
&#x1F4C8; 2970 <br>
<p>Min Jin Chong, Hsin-Ying Lee, David Forsyth</p></summary>
<p>

**Abstract:** Recently, StyleGAN has enabled various image manipulation and editing tasks thanks to the high-quality generation and the disentangled latent space. However, additional architectures or task-specific training paradigms are usually required for different tasks. In this work, we take a deeper look at the spatial properties of StyleGAN. We show that with a pretrained StyleGAN along with some operations, without any additional architecture, we can perform comparably to the state-of-the-art methods on various tasks, including image blending, panorama generation, generation from a single image, controllable and local multimodal image to image translation, and attributes transfer. The proposed method is simple, effective, efficient, and applicable to any existing pretrained StyleGAN model.

</p>
</details>

<details><summary><b>Recent Advances in End-to-End Automatic Speech Recognition</b>
<a href="https://arxiv.org/abs/2111.01690">arxiv:2111.01690</a>
&#x1F4C8; 76 <br>
<p>Jinyu Li</p></summary>
<p>

**Abstract:** Recently, the speech community is seeing a significant trend of moving from deep neural network based hybrid modeling to end-to-end (E2E) modeling for automatic speech recognition (ASR). While E2E models achieve the state-of-the-art results in most benchmarks in terms of ASR accuracy, hybrid models are still used in a large proportion of commercial ASR systems at the current time. There are lots of practical factors that affect the production model deployment decision. Traditional hybrid models, being optimized for production for decades, are usually good at these factors. Without providing excellent solutions to all these factors, it is hard for E2E models to be widely commercialized. In this paper, we will overview the recent advances in E2E models, focusing on technologies addressing those challenges from the industry's perspective.

</p>
</details>

<details><summary><b>PatchGame: Learning to Signal Mid-level Patches in Referential Games</b>
<a href="https://arxiv.org/abs/2111.01785">arxiv:2111.01785</a>
&#x1F4C8; 67 <br>
<p>Kamal Gupta, Gowthami Somepalli, Anubhav Gupta, Vinoj Jayasundara, Matthias Zwicker, Abhinav Shrivastava</p></summary>
<p>

**Abstract:** We study a referential game (a type of signaling game) where two agents communicate with each other via a discrete bottleneck to achieve a common goal. In our referential game, the goal of the speaker is to compose a message or a symbolic representation of "important" image patches, while the task for the listener is to match the speaker's message to a different view of the same image. We show that it is indeed possible for the two agents to develop a communication protocol without explicit or implicit supervision. We further investigate the developed protocol and show the applications in speeding up recent Vision Transformers by using only important patches, and as pre-training for downstream recognition tasks (e.g., classification). Code available at https://github.com/kampta/PatchGame.

</p>
</details>

<details><summary><b>A Framework for Real-World Multi-Robot Systems Running Decentralized GNN-Based Policies</b>
<a href="https://arxiv.org/abs/2111.01777">arxiv:2111.01777</a>
&#x1F4C8; 44 <br>
<p>Jan Blumenkamp, Steven Morad, Jennifer Gielis, Qingbiao Li, Amanda Prorok</p></summary>
<p>

**Abstract:** Graph Neural Networks (GNNs) are a paradigm-shifting neural architecture to facilitate the learning of complex multi-agent behaviors. Recent work has demonstrated remarkable performance in tasks such as flocking, multi-agent path planning and cooperative coverage. However, the policies derived through GNN-based learning schemes have not yet been deployed to the real-world on physical multi-robot systems. In this work, we present the design of a system that allows for fully decentralized execution of GNN-based policies. We create a framework based on ROS2 and elaborate its details in this paper. We demonstrate our framework on a case-study that requires tight coordination between robots, and present first-of-a-kind results that show successful real-world deployment of GNN-based policies on a decentralized multi-robot system relying on Adhoc communication. A video demonstration of this case-study can be found online. https://www.youtube.com/watch?v=COh-WLn4iO4

</p>
</details>

<details><summary><b>Procedural Generalization by Planning with Self-Supervised World Models</b>
<a href="https://arxiv.org/abs/2111.01587">arxiv:2111.01587</a>
&#x1F4C8; 26 <br>
<p>Ankesh Anand, Jacob Walker, Yazhe Li, Eszter Vértes, Julian Schrittwieser, Sherjil Ozair, Théophane Weber, Jessica B. Hamrick</p></summary>
<p>

**Abstract:** One of the key promises of model-based reinforcement learning is the ability to generalize using an internal model of the world to make predictions in novel environments and tasks. However, the generalization ability of model-based agents is not well understood because existing work has focused on model-free agents when benchmarking generalization. Here, we explicitly measure the generalization ability of model-based agents in comparison to their model-free counterparts. We focus our analysis on MuZero (Schrittwieser et al., 2020), a powerful model-based agent, and evaluate its performance on both procedural and task generalization. We identify three factors of procedural generalization -- planning, self-supervised representation learning, and procedural data diversity -- and show that by combining these techniques, we achieve state-of-the art generalization performance and data efficiency on Procgen (Cobbe et al., 2019). However, we find that these factors do not always provide the same benefits for the task generalization benchmarks in Meta-World (Yu et al., 2019), indicating that transfer remains a challenge and may require different approaches than procedural generalization. Overall, we suggest that building generalizable agents requires moving beyond the single-task, model-free paradigm and towards self-supervised model-based agents that are trained in rich, procedural, multi-task environments.

</p>
</details>

<details><summary><b>OpenPrompt: An Open-source Framework for Prompt-learning</b>
<a href="https://arxiv.org/abs/2111.01998">arxiv:2111.01998</a>
&#x1F4C8; 25 <br>
<p>Ning Ding, Shengding Hu, Weilin Zhao, Yulin Chen, Zhiyuan Liu, Hai-Tao Zheng, Maosong Sun</p></summary>
<p>

**Abstract:** Prompt-learning has become a new paradigm in modern natural language processing, which directly adapts pre-trained language models (PLMs) to $cloze$-style prediction, autoregressive modeling, or sequence to sequence generation, resulting in promising performances on various tasks. However, no standard implementation framework of prompt-learning is proposed yet, and most existing prompt-learning codebases, often unregulated, only provide limited implementations for specific scenarios. Since there are many details such as templating strategy, initializing strategy, and verbalizing strategy, etc. need to be considered in prompt-learning, practitioners face impediments to quickly adapting the desired prompt learning methods to their applications. In this paper, we present {OpenPrompt}, a unified easy-to-use toolkit to conduct prompt-learning over PLMs. OpenPrompt is a research-friendly framework that is equipped with efficiency, modularity, and extendibility, and its combinability allows the freedom to combine different PLMs, task formats, and prompting modules in a unified paradigm. Users could expediently deploy prompt-learning frameworks and evaluate the generalization of them on different NLP tasks without constraints. OpenPrompt is publicly released at {\url{ https://github.com/thunlp/OpenPrompt}}.

</p>
</details>

<details><summary><b>Obvious Manipulability of Voting Rules</b>
<a href="https://arxiv.org/abs/2111.01983">arxiv:2111.01983</a>
&#x1F4C8; 16 <br>
<p>Haris Aziz, Alexander Lam</p></summary>
<p>

**Abstract:** The Gibbard-Satterthwaite theorem states that no unanimous and non-dictatorial voting rule is strategyproof. We revisit voting rules and consider a weaker notion of strategyproofness called not obvious manipulability that was proposed by Troyan and Morrill (2020). We identify several classes of voting rules that satisfy this notion. We also show that several voting rules including k-approval fail to satisfy this property. We characterize conditions under which voting rules are obviously manipulable. One of our insights is that certain rules are obviously manipulable when the number of alternatives is relatively large compared to the number of voters. In contrast to the Gibbard-Satterthwaite theorem, many of the rules we examined are not obviously manipulable. This reflects the relatively easier satisfiability of the notion and the zero information assumption of not obvious manipulability, as opposed to the perfect information assumption of strategyproofness. We also present algorithmic results for computing obvious manipulations and report on experiments.

</p>
</details>

<details><summary><b>Realistic galaxy image simulation via score-based generative models</b>
<a href="https://arxiv.org/abs/2111.01713">arxiv:2111.01713</a>
&#x1F4C8; 10 <br>
<p>Michael J. Smith, James E. Geach, Ryan A. Jackson, Nikhil Arora, Connor Stone, Stéphane Courteau</p></summary>
<p>

**Abstract:** We show that a Denoising Diffusion Probabalistic Model (DDPM), a class of score-based generative model, can be used to produce realistic yet fake images that mimic observations of galaxies. Our method is tested with Dark Energy Spectroscopic Instrument grz imaging of galaxies from the Photometry and Rotation curve OBservations from Extragalactic Surveys (PROBES) sample and galaxies selected from the Sloan Digital Sky Survey. Subjectively, the generated galaxies are highly realistic when compared with samples from the real dataset. We quantify the similarity by borrowing from the deep generative learning literature, using the `Fréchet Inception Distance' to test for subjective and morphological similarity. We also introduce the `Synthetic Galaxy Distance' metric to compare the emergent physical properties (such as total magnitude, colour and half light radius) of a ground truth parent and synthesised child dataset. We argue that the DDPM approach produces sharper and more realistic images than other generative methods such as Adversarial Networks (with the downside of more costly inference), and could be used to produce large samples of synthetic observations tailored to a specific imaging survey. We demonstrate two potential uses of the DDPM: (1) accurate in-painting of occluded data, such as satellite trails, and (2) domain transfer, where new input images can be processed to mimic the properties of the DDPM training set. Here we `DESI-fy' cartoon images as a proof of concept for domain transfer. Finally, we suggest potential applications for score-based approaches that could motivate further research on this topic within the astronomical community.

</p>
</details>

<details><summary><b>Subquadratic Overparameterization for Shallow Neural Networks</b>
<a href="https://arxiv.org/abs/2111.01875">arxiv:2111.01875</a>
&#x1F4C8; 9 <br>
<p>Chaehwan Song, Ali Ramezani-Kebrya, Thomas Pethick, Armin Eftekhari, Volkan Cevher</p></summary>
<p>

**Abstract:** Overparameterization refers to the important phenomenon where the width of a neural network is chosen such that learning algorithms can provably attain zero loss in nonconvex training. The existing theory establishes such global convergence using various initialization strategies, training modifications, and width scalings. In particular, the state-of-the-art results require the width to scale quadratically with the number of training data under standard initialization strategies used in practice for best generalization performance. In contrast, the most recent results obtain linear scaling either with requiring initializations that lead to the "lazy-training", or training only a single layer. In this work, we provide an analytical framework that allows us to adopt standard initialization strategies, possibly avoid lazy training, and train all layers simultaneously in basic shallow neural networks while attaining a desirable subquadratic scaling on the network width. We achieve the desiderata via Polyak-Lojasiewicz condition, smoothness, and standard assumptions on data, and use tools from random matrix theory.

</p>
</details>

<details><summary><b>Geometry-aware Bayesian Optimization in Robotics using Riemannian Matérn Kernels</b>
<a href="https://arxiv.org/abs/2111.01460">arxiv:2111.01460</a>
&#x1F4C8; 9 <br>
<p>Noémie Jaquier, Viacheslav Borovitskiy, Andrei Smolensky, Alexander Terenin, Tamim Asfour, Leonel Rozo</p></summary>
<p>

**Abstract:** Bayesian optimization is a data-efficient technique which can be used for control parameter tuning, parametric policy adaptation, and structure design in robotics. Many of these problems require optimization of functions defined on non-Euclidean domains like spheres, rotation groups, or spaces of positive-definite matrices. To do so, one must place a Gaussian process prior, or equivalently define a kernel, on the space of interest. Effective kernels typically reflect the geometry of the spaces they are defined on, but designing them is generally non-trivial. Recent work on the Riemannian Matérn kernels, based on stochastic partial differential equations and spectral theory of the Laplace-Beltrami operator, offers promising avenues towards constructing such geometry-aware kernels. In this paper, we study techniques for implementing these kernels on manifolds of interest in robotics, demonstrate their performance on a set of artificial benchmark functions, and illustrate geometry-aware Bayesian optimization for a variety of robotic applications, covering orientation control, manipulability optimization, and motion planning, while showing its improved performance.

</p>
</details>

<details><summary><b>Recursive Bayesian Networks: Generalising and Unifying Probabilistic Context-Free Grammars and Dynamic Bayesian Networks</b>
<a href="https://arxiv.org/abs/2111.01853">arxiv:2111.01853</a>
&#x1F4C8; 6 <br>
<p>Robert Lieck, Martin Rohrmeier</p></summary>
<p>

**Abstract:** Probabilistic context-free grammars (PCFGs) and dynamic Bayesian networks (DBNs) are widely used sequence models with complementary strengths and limitations. While PCFGs allow for nested hierarchical dependencies (tree structures), their latent variables (non-terminal symbols) have to be discrete. In contrast, DBNs allow for continuous latent variables, but the dependencies are strictly sequential (chain structure). Therefore, neither can be applied if the latent variables are assumed to be continuous and also to have a nested hierarchical dependency structure. In this paper, we present Recursive Bayesian Networks (RBNs), which generalise and unify PCFGs and DBNs, combining their strengths and containing both as special cases. RBNs define a joint distribution over tree-structured Bayesian networks with discrete or continuous latent variables. The main challenge lies in performing joint inference over the exponential number of possible structures and the continuous variables. We provide two solutions: 1) For arbitrary RBNs, we generalise inside and outside probabilities from PCFGs to the mixed discrete-continuous case, which allows for maximum posterior estimates of the continuous latent variables via gradient descent, while marginalising over network structures. 2) For Gaussian RBNs, we additionally derive an analytic approximation, allowing for robust parameter optimisation and Bayesian inference. The capacity and diverse applications of RBNs are illustrated on two examples: In a quantitative evaluation on synthetic data, we demonstrate and discuss the advantage of RBNs for segmentation and tree induction from noisy sequences, compared to change point detection and hierarchical clustering. In an application to musical data, we approach the unsolved problem of hierarchical music analysis from the raw note level and compare our results to expert annotations.

</p>
</details>

<details><summary><b>Spatio-Temporal Variational Gaussian Processes</b>
<a href="https://arxiv.org/abs/2111.01732">arxiv:2111.01732</a>
&#x1F4C8; 6 <br>
<p>Oliver Hamelijnck, William J. Wilkinson, Niki A. Loppi, Arno Solin, Theodoros Damoulas</p></summary>
<p>

**Abstract:** We introduce a scalable approach to Gaussian process inference that combines spatio-temporal filtering with natural gradient variational inference, resulting in a non-conjugate GP method for multivariate data that scales linearly with respect to time. Our natural gradient approach enables application of parallel filtering and smoothing, further reducing the temporal span complexity to be logarithmic in the number of time steps. We derive a sparse approximation that constructs a state-space model over a reduced set of spatial inducing points, and show that for separable Markov kernels the full and sparse cases exactly recover the standard variational GP, whilst exhibiting favourable computational properties. To further improve the spatial scaling we propose a mean-field assumption of independence between spatial locations which, when coupled with sparsity and parallelisation, leads to an efficient and accurate method for large spatio-temporal problems.

</p>
</details>

<details><summary><b>Zero-Shot Translation using Diffusion Models</b>
<a href="https://arxiv.org/abs/2111.01471">arxiv:2111.01471</a>
&#x1F4C8; 6 <br>
<p>Eliya Nachmani, Shaked Dovrat</p></summary>
<p>

**Abstract:** In this work, we show a novel method for neural machine translation (NMT), using a denoising diffusion probabilistic model (DDPM), adjusted for textual data, following recent advances in the field. We show that it's possible to translate sentences non-autoregressively using a diffusion model conditioned on the source sentence. We also show that our model is able to translate between pairs of languages unseen during training (zero-shot learning).

</p>
</details>

<details><summary><b>LogAvgExp Provides a Principled and Performant Global Pooling Operator</b>
<a href="https://arxiv.org/abs/2111.01742">arxiv:2111.01742</a>
&#x1F4C8; 5 <br>
<p>Scott C. Lowe, Thomas Trappenberg, Sageev Oore</p></summary>
<p>

**Abstract:** We seek to improve the pooling operation in neural networks, by applying a more theoretically justified operator. We demonstrate that LogSumExp provides a natural OR operator for logits. When one corrects for the number of elements inside the pooling operator, this becomes $\text{LogAvgExp} := \log(\text{mean}(\exp(x)))$. By introducing a single temperature parameter, LogAvgExp smoothly transitions from the max of its operands to the mean (found at the limiting cases $t \to 0^+$ and $t \to +\infty$). We experimentally tested LogAvgExp, both with and without a learnable temperature parameter, in a variety of deep neural network architectures for computer vision.

</p>
</details>

<details><summary><b>Meta-Learning the Search Distribution of Black-Box Random Search Based Adversarial Attacks</b>
<a href="https://arxiv.org/abs/2111.01714">arxiv:2111.01714</a>
&#x1F4C8; 5 <br>
<p>Maksym Yatsura, Jan Hendrik Metzen, Matthias Hein</p></summary>
<p>

**Abstract:** Adversarial attacks based on randomized search schemes have obtained state-of-the-art results in black-box robustness evaluation recently. However, as we demonstrate in this work, their efficiency in different query budget regimes depends on manual design and heuristic tuning of the underlying proposal distributions. We study how this issue can be addressed by adapting the proposal distribution online based on the information obtained during the attack. We consider Square Attack, which is a state-of-the-art score-based black-box attack, and demonstrate how its performance can be improved by a learned controller that adjusts the parameters of the proposal distribution online during the attack. We train the controller using gradient-based end-to-end training on a CIFAR10 model with white box access. We demonstrate that plugging the learned controller into the attack consistently improves its black-box robustness estimate in different query regimes by up to 20% for a wide range of different models with black-box access. We further show that the learned adaptation principle transfers well to the other data distributions such as CIFAR100 or ImageNet and to the targeted attack setting.

</p>
</details>

<details><summary><b>Improving Classifier Training Efficiency for Automatic Cyberbullying Detection with Feature Density</b>
<a href="https://arxiv.org/abs/2111.01689">arxiv:2111.01689</a>
&#x1F4C8; 5 <br>
<p>Juuso Eronen, Michal Ptaszynski, Fumito Masui, Aleksander Smywiński-Pohl, Gniewosz Leliwa, Michal Wroczynski</p></summary>
<p>

**Abstract:** We study the effectiveness of Feature Density (FD) using different linguistically-backed feature preprocessing methods in order to estimate dataset complexity, which in turn is used to comparatively estimate the potential performance of machine learning (ML) classifiers prior to any training. We hypothesise that estimating dataset complexity allows for the reduction of the number of required experiments iterations. This way we can optimize the resource-intensive training of ML models which is becoming a serious issue due to the increases in available dataset sizes and the ever rising popularity of models based on Deep Neural Networks (DNN). The problem of constantly increasing needs for more powerful computational resources is also affecting the environment due to alarmingly-growing amount of CO2 emissions caused by training of large-scale ML models. The research was conducted on multiple datasets, including popular datasets, such as Yelp business review dataset used for training typical sentiment analysis models, as well as more recent datasets trying to tackle the problem of cyberbullying, which, being a serious social problem, is also a much more sophisticated problem form the point of view of linguistic representation. We use cyberbullying datasets collected for multiple languages, namely English, Japanese and Polish. The difference in linguistic complexity of datasets allows us to additionally discuss the efficacy of linguistically-backed word preprocessing.

</p>
</details>

<details><summary><b>Characterizing and Understanding the Generalization Error of Transfer Learning with Gibbs Algorithm</b>
<a href="https://arxiv.org/abs/2111.01635">arxiv:2111.01635</a>
&#x1F4C8; 5 <br>
<p>Yuheng Bu, Gholamali Aminian, Laura Toni, Miguel Rodrigues, Gregory Wornell</p></summary>
<p>

**Abstract:** We provide an information-theoretic analysis of the generalization ability of Gibbs-based transfer learning algorithms by focusing on two popular transfer learning approaches, $α$-weighted-ERM and two-stage-ERM. Our key result is an exact characterization of the generalization behaviour using the conditional symmetrized KL information between the output hypothesis and the target training samples given the source samples. Our results can also be applied to provide novel distribution-free generalization error upper bounds on these two aforementioned Gibbs algorithms. Our approach is versatile, as it also characterizes the generalization errors and excess risks of these two Gibbs algorithms in the asymptotic regime, where they converge to the $α$-weighted-ERM and two-stage-ERM, respectively. Based on our theoretical results, we show that the benefits of transfer learning can be viewed as a bias-variance trade-off, with the bias induced by the source distribution and the variance induced by the lack of target samples. We believe this viewpoint can guide the choice of transfer learning algorithms in practice.

</p>
</details>

<details><summary><b>Training Certifiably Robust Neural Networks with Efficient Local Lipschitz Bounds</b>
<a href="https://arxiv.org/abs/2111.01395">arxiv:2111.01395</a>
&#x1F4C8; 5 <br>
<p>Yujia Huang, Huan Zhang, Yuanyuan Shi, J Zico Kolter, Anima Anandkumar</p></summary>
<p>

**Abstract:** Certified robustness is a desirable property for deep neural networks in safety-critical applications, and popular training algorithms can certify robustness of a neural network by computing a global bound on its Lipschitz constant. However, such a bound is often loose: it tends to over-regularize the neural network and degrade its natural accuracy. A tighter Lipschitz bound may provide a better tradeoff between natural and certified accuracy, but is generally hard to compute exactly due to non-convexity of the network. In this work, we propose an efficient and trainable \emph{local} Lipschitz upper bound by considering the interactions between activation functions (e.g. ReLU) and weight matrices. Specifically, when computing the induced norm of a weight matrix, we eliminate the corresponding rows and columns where the activation function is guaranteed to be a constant in the neighborhood of each given data point, which provides a provably tighter bound than the global Lipschitz constant of the neural network. Our method can be used as a plug-in module to tighten the Lipschitz bound in many certifiable training algorithms. Furthermore, we propose to clip activation functions (e.g., ReLU and MaxMin) with a learnable upper threshold and a sparsity loss to assist the network to achieve an even tighter local Lipschitz bound. Experimentally, we show that our method consistently outperforms state-of-the-art methods in both clean and certified accuracy on MNIST, CIFAR-10 and TinyImageNet datasets with various network architectures.

</p>
</details>

<details><summary><b>Skin Cancer Classification using Inception Network and Transfer Learning</b>
<a href="https://arxiv.org/abs/2111.02402">arxiv:2111.02402</a>
&#x1F4C8; 4 <br>
<p>Priscilla Benedetti, Damiano Perri, Marco Simonetti, Osvaldo Gervasi, Gianluca Reali, Mauro Femminella</p></summary>
<p>

**Abstract:** Medical data classification is typically a challenging task due to imbalance between classes. In this paper, we propose an approach to classify dermatoscopic images from HAM10000 (Human Against Machine with 10000 training images) dataset, consisting of seven imbalanced types of skin lesions, with good precision and low resources requirements. Classification is done by using a pretrained convolutional neural network. We evaluate the accuracy and performance of the proposal and illustrate possible extensions.

</p>
</details>

<details><summary><b>PhyloTransformer: A Discriminative Model for Mutation Prediction Based on a Multi-head Self-attention Mechanism</b>
<a href="https://arxiv.org/abs/2111.01969">arxiv:2111.01969</a>
&#x1F4C8; 4 <br>
<p>Yingying Wu, Shusheng Xu, Shing-Tung Yau, Yi Wu</p></summary>
<p>

**Abstract:** Severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) has caused an ongoing pandemic infecting 219 million people as of 10/19/21, with a 3.6% mortality rate. Natural selection can generate favorable mutations with improved fitness advantages; however, the identified coronaviruses may be the tip of the iceberg, and potentially more fatal variants of concern (VOCs) may emerge over time. Understanding the patterns of emerging VOCs and forecasting mutations that may lead to gain of function or immune escape is urgently required. Here we developed PhyloTransformer, a Transformer-based discriminative model that engages a multi-head self-attention mechanism to model genetic mutations that may lead to viral reproductive advantage. In order to identify complex dependencies between the elements of each input sequence, PhyloTransformer utilizes advanced modeling techniques, including a novel Fast Attention Via positive Orthogonal Random features approach (FAVOR+) from Performer, and the Masked Language Model (MLM) from Bidirectional Encoder Representations from Transformers (BERT). PhyloTransformer was trained with 1,765,297 genetic sequences retrieved from the Global Initiative for Sharing All Influenza Data (GISAID) database. Firstly, we compared the prediction accuracy of novel mutations and novel combinations using extensive baseline models; we found that PhyloTransformer outperformed every baseline method with statistical significance. Secondly, we examined predictions of mutations in each nucleotide of the receptor binding motif (RBM), and we found our predictions were precise and accurate. Thirdly, we predicted modifications of N-glycosylation sites to identify mutations associated with altered glycosylation that may be favored during viral evolution. We anticipate that PhyloTransformer may guide proactive vaccine design for effective targeting of future SARS-CoV-2 variants.

</p>
</details>

<details><summary><b>3-D PET Image Generation with tumour masks using TGAN</b>
<a href="https://arxiv.org/abs/2111.01866">arxiv:2111.01866</a>
&#x1F4C8; 4 <br>
<p>Robert V Bergen, Jean-Francois Rajotte, Fereshteh Yousefirizi, Ivan S Klyuzhin, Arman Rahmim, Raymond T. Ng</p></summary>
<p>

**Abstract:** Training computer-vision related algorithms on medical images for disease diagnosis or image segmentation is difficult due to the lack of training data, labeled samples, and privacy concerns. For this reason, a robust generative method to create synthetic data is highly sought after. However, most three-dimensional image generators require additional image input or are extremely memory intensive. To address these issues we propose adapting video generation techniques for 3-D image generation. Using the temporal GAN (TGAN) architecture, we show we are able to generate realistic head and neck PET images. We also show that by conditioning the generator on tumour masks, we are able to control the geometry and location of the tumour in the generated images. To test the utility of the synthetic images, we train a segmentation model using the synthetic images. Synthetic images conditioned on real tumour masks are automatically segmented, and the corresponding real images are also segmented. We evaluate the segmentations using the Dice score and find the segmentation algorithm performs similarly on both datasets (0.65 synthetic data, 0.70 real data). Various radionomic features are then calculated over the segmented tumour volumes for each data set. A comparison of the real and synthetic feature distributions show that seven of eight feature distributions had statistically insignificant differences (p>0.05). Correlation coefficients were also calculated between all radionomic features and it is shown that all of the strong statistical correlations in the real data set are preserved in the synthetic data set.

</p>
</details>

<details><summary><b>Efficient Hierarchical Bayesian Inference for Spatio-temporal Regression Models in Neuroimaging</b>
<a href="https://arxiv.org/abs/2111.01692">arxiv:2111.01692</a>
&#x1F4C8; 4 <br>
<p>Ali Hashemi, Yijing Gao, Chang Cai, Sanjay Ghosh, Klaus-Robert Müller, Srikantan S. Nagarajan, Stefan Haufe</p></summary>
<p>

**Abstract:** Several problems in neuroimaging and beyond require inference on the parameters of multi-task sparse hierarchical regression models. Examples include M/EEG inverse problems, neural encoding models for task-based fMRI analyses, and climate science. In these domains, both the model parameters to be inferred and the measurement noise may exhibit a complex spatio-temporal structure. Existing work either neglects the temporal structure or leads to computationally demanding inference schemes. Overcoming these limitations, we devise a novel flexible hierarchical Bayesian framework within which the spatio-temporal dynamics of model parameters and noise are modeled to have Kronecker product covariance structure. Inference in our framework is based on majorization-minimization optimization and has guaranteed convergence properties. Our highly efficient algorithms exploit the intrinsic Riemannian geometry of temporal autocovariance matrices. For stationary dynamics described by Toeplitz matrices, the theory of circulant embeddings is employed. We prove convex bounding properties and derive update rules of the resulting algorithms. On both synthetic and real neural data from M/EEG, we demonstrate that our methods lead to improved performance.

</p>
</details>

<details><summary><b>Fitness Landscape Footprint: A Framework to Compare Neural Architecture Search Problems</b>
<a href="https://arxiv.org/abs/2111.01584">arxiv:2111.01584</a>
&#x1F4C8; 4 <br>
<p>Kalifou René Traoré, Andrés Camero, Xiao Xiang Zhu</p></summary>
<p>

**Abstract:** Neural architecture search is a promising area of research dedicated to automating the design of neural network models. This field is rapidly growing, with a surge of methodologies ranging from Bayesian optimization,neuroevoltion, to differentiable search, and applications in various contexts. However, despite all great advances, few studies have presented insights on the difficulty of the problem itself, thus the success (or fail) of these methodologies remains unexplained. In this sense, the field of optimization has developed methods that highlight key aspects to describe optimization problems. The fitness landscape analysis stands out when it comes to characterize reliably and quantitatively search algorithms. In this paper, we propose to use fitness landscape analysis to study a neural architecture search problem. Particularly, we introduce the fitness landscape footprint, an aggregation of eight (8)general-purpose metrics to synthesize the landscape of an architecture search problem. We studied two problems, the classical image classification benchmark CIFAR-10, and the Remote-Sensing problem So2Sat LCZ42. The results present a quantitative appraisal of the problems, allowing to characterize the relative difficulty and other characteristics, such as the ruggedness or the persistence, that helps to tailor a search strategy to the problem. Also, the footprint is a tool that enables the comparison of multiple problems.

</p>
</details>

<details><summary><b>Convolutional generative adversarial imputation networks for spatio-temporal missing data in storm surge simulations</b>
<a href="https://arxiv.org/abs/2111.02823">arxiv:2111.02823</a>
&#x1F4C8; 3 <br>
<p>Ehsan Adeli, Jize Zhang, Alexandros A. Taflanidis</p></summary>
<p>

**Abstract:** Imputation of missing data is a task that plays a vital role in a number of engineering and science applications. Often such missing data arise in experimental observations from limitations of sensors or post-processing transformation errors. Other times they arise from numerical and algorithmic constraints in computer simulations. One such instance and the application emphasis of this paper are numerical simulations of storm surge. The simulation data corresponds to time-series surge predictions over a number of save points within the geographic domain of interest, creating a spatio-temporal imputation problem where the surge points are heavily correlated spatially and temporally, and the missing values regions are structurally distributed at random. Very recently, machine learning techniques such as neural network methods have been developed and employed for missing data imputation tasks. Generative Adversarial Nets (GANs) and GAN-based techniques have particularly attracted attention as unsupervised machine learning methods. In this study, the Generative Adversarial Imputation Nets (GAIN) performance is improved by applying convolutional neural networks instead of fully connected layers to better capture the correlation of data and promote learning from the adjacent surge points. Another adjustment to the method needed specifically for the studied data is to consider the coordinates of the points as additional features to provide the model more information through the convolutional layers. We name our proposed method as Convolutional Generative Adversarial Imputation Nets (Conv-GAIN). The proposed method's performance by considering the improvements and adaptations required for the storm surge data is assessed and compared to the original GAIN and a few other techniques. The results show that Conv-GAIN has better performance than the alternative methods on the studied data.

</p>
</details>

<details><summary><b>Discovering and Exploiting Sparse Rewards in a Learned Behavior Space</b>
<a href="https://arxiv.org/abs/2111.01919">arxiv:2111.01919</a>
&#x1F4C8; 3 <br>
<p>Giuseppe Paolo, Alexandre Coninx, Alban Laflaquière, Stephane Doncieux</p></summary>
<p>

**Abstract:** Learning optimal policies in sparse rewards settings is difficult as the learning agent has little to no feedback on the quality of its actions. In these situations, a good strategy is to focus on exploration, hopefully leading to the discovery of a reward signal to improve on. A learning algorithm capable of dealing with this kind of settings has to be able to (1) explore possible agent behaviors and (2) exploit any possible discovered reward. Efficient exploration algorithms have been proposed that require to define a behavior space, that associates to an agent its resulting behavior in a space that is known to be worth exploring. The need to define this space is a limitation of these algorithms. In this work, we introduce STAX, an algorithm designed to learn a behavior space on-the-fly and to explore it while efficiently optimizing any reward discovered. It does so by separating the exploration and learning of the behavior space from the exploitation of the reward through an alternating two-steps process. In the first step, STAX builds a repertoire of diverse policies while learning a low-dimensional representation of the high-dimensional observations generated during the policies evaluation. In the exploitation step, emitters are used to optimize the performance of the discovered rewarding solutions. Experiments conducted on three different sparse reward environments show that STAX performs comparably to existing baselines while requiring much less prior information about the task as it autonomously builds the behavior space.

</p>
</details>

<details><summary><b>From Strings to Data Science: a Practical Framework for Automated String Handling</b>
<a href="https://arxiv.org/abs/2111.01868">arxiv:2111.01868</a>
&#x1F4C8; 3 <br>
<p>John W. van Lith, Joaquin Vanschoren</p></summary>
<p>

**Abstract:** Many machine learning libraries require that string features be converted to a numerical representation for the models to work as intended. Categorical string features can represent a wide variety of data (e.g., zip codes, names, marital status), and are notoriously difficult to preprocess automatically. In this paper, we propose a framework to do so based on best practices, domain knowledge, and novel techniques. It automatically identifies different types of string features, processes them accordingly, and encodes them into numerical representations. We also provide an open source Python implementation to automatically preprocess categorical string data in tabular datasets and demonstrate promising results on a wide range of datasets.

</p>
</details>

<details><summary><b>UnProjection: Leveraging Inverse-Projections for Visual Analytics of High-Dimensional Data</b>
<a href="https://arxiv.org/abs/2111.01744">arxiv:2111.01744</a>
&#x1F4C8; 3 <br>
<p>Mateus Espadoto, Gabriel Appleby, Ashley Suh, Dylan Cashman, Mingwei Li, Carlos Scheidegger, Erik W Anderson, Remco Chang, Alexandru C Telea</p></summary>
<p>

**Abstract:** Projection techniques are often used to visualize high-dimensional data, allowing users to better understand the overall structure of multi-dimensional spaces on a 2D screen. Although many such methods exist, comparably little work has been done on generalizable methods of inverse-projection -- the process of mapping the projected points, or more generally, the projection space back to the original high-dimensional space. In this paper we present NNInv, a deep learning technique with the ability to approximate the inverse of any projection or mapping. NNInv learns to reconstruct high-dimensional data from any arbitrary point on a 2D projection space, giving users the ability to interact with the learned high-dimensional representation in a visual analytics system. We provide an analysis of the parameter space of NNInv, and offer guidance in selecting these parameters. We extend validation of the effectiveness of NNInv through a series of quantitative and qualitative analyses. We then demonstrate the method's utility by applying it to three visualization tasks: interactive instance interpolation, classifier agreement, and gradient visualization.

</p>
</details>

<details><summary><b>Predicting the Location of Bicycle-sharing Stations using OpenStreetMap Data</b>
<a href="https://arxiv.org/abs/2111.01722">arxiv:2111.01722</a>
&#x1F4C8; 3 <br>
<p>Kamil Raczycki</p></summary>
<p>

**Abstract:** Planning the layout of bicycle-sharing stations is a complex process, especially in cities where bicycle sharing systems are just being implemented. Urban planners often have to make a lot of estimates based on both publicly available data and privately provided data from the administration and then use the Location-Allocation model popular in the field. Many municipalities in smaller cities may have difficulty hiring specialists to carry out such planning. This thesis proposes a new solution to streamline and facilitate the process of such planning by using spatial embedding methods. Based only on publicly available data from OpenStreetMap, and station layouts from 34 cities in Europe, a method has been developed to divide cities into micro-regions using the Uber H3 discrete global grid system and to indicate regions where it is worth placing a station based on existing systems in different cities using transfer learning. The result of the work is a mechanism to support planners in their decision making when planning a station layout with a choice of reference cities.

</p>
</details>

<details><summary><b>Assessing Effectiveness of Using Internal Signals for Check-Worthy Claim Identification in Unlabeled Data for Automated Fact-Checking</b>
<a href="https://arxiv.org/abs/2111.01706">arxiv:2111.01706</a>
&#x1F4C8; 3 <br>
<p>Archita Pathak, Rohini K. Srihari</p></summary>
<p>

**Abstract:** While recent work on automated fact-checking has focused mainly on verifying and explaining claims, for which the list of claims is readily available, identifying check-worthy claim sentences from a text remains challenging. Current claim identification models rely on manual annotations for each sentence in the text, which is an expensive task and challenging to conduct on a frequent basis across multiple domains. This paper explores methodology to identify check-worthy claim sentences from fake news articles, irrespective of domain, without explicit sentence-level annotations. We leverage two internal supervisory signals - headline and the abstractive summary - to rank the sentences based on semantic similarity. We hypothesize that this ranking directly correlates to the check-worthiness of the sentences. To assess the effectiveness of this hypothesis, we build pipelines that leverage the ranking of sentences based on either the headline or the abstractive summary. The top-ranked sentences are used for the downstream fact-checking tasks of evidence retrieval and the article's veracity prediction by the pipeline. Our findings suggest that the top 3 ranked sentences contain enough information for evidence-based fact-checking of a fake news article. We also show that while the headline has more gisting similarity with how a fact-checking website writes a claim, the summary-based pipeline is the most promising for an end-to-end fact-checking system.

</p>
</details>

<details><summary><b>Explainable Medical Image Segmentation via Generative Adversarial Networks and Layer-wise Relevance Propagation</b>
<a href="https://arxiv.org/abs/2111.01665">arxiv:2111.01665</a>
&#x1F4C8; 3 <br>
<p>Awadelrahman M. A. Ahmed, Leen A. M. Ali</p></summary>
<p>

**Abstract:** This paper contributes to automating medical image segmentation by proposing generative adversarial network-based models to segment both polyps and instruments in endoscopy images. A major contribution of this work is to provide explanations for the predictions using a layer-wise relevance propagation approach designating which input image pixels are relevant to the predictions and to what extent. On the polyp segmentation task, the models achieved 0.84 of accuracy and 0.46 on Jaccard index. On the instrument segmentation task, the models achieved 0.96 of accuracy and 0.70 on Jaccard index. The code is available at https://github.com/Awadelrahman/MedAI.

</p>
</details>

<details><summary><b>UQuAD1.0: Development of an Urdu Question Answering Training Data for Machine Reading Comprehension</b>
<a href="https://arxiv.org/abs/2111.01543">arxiv:2111.01543</a>
&#x1F4C8; 3 <br>
<p>Samreen Kazi, Shakeel Khoja</p></summary>
<p>

**Abstract:** In recent years, low-resource Machine Reading Comprehension (MRC) has made significant progress, with models getting remarkable performance on various language datasets. However, none of these models have been customized for the Urdu language. This work explores the semi-automated creation of the Urdu Question Answering Dataset (UQuAD1.0) by combining machine-translated SQuAD with human-generated samples derived from Wikipedia articles and Urdu RC worksheets from Cambridge O-level books. UQuAD1.0 is a large-scale Urdu dataset intended for extractive machine reading comprehension tasks consisting of 49k question Answers pairs in question, passage, and answer format. In UQuAD1.0, 45000 pairs of QA were generated by machine translation of the original SQuAD1.0 and approximately 4000 pairs via crowdsourcing. In this study, we used two types of MRC models: rule-based baseline and advanced Transformer-based models. However, we have discovered that the latter outperforms the others; thus, we have decided to concentrate solely on Transformer-based architectures. Using XLMRoBERTa and multi-lingual BERT, we acquire an F1 score of 0.66 and 0.63, respectively.

</p>
</details>

<details><summary><b>Dealing With Misspecification In Fixed-Confidence Linear Top-m Identification</b>
<a href="https://arxiv.org/abs/2111.01479">arxiv:2111.01479</a>
&#x1F4C8; 3 <br>
<p>Clémence Réda, Andrea Tirinzoni, Rémy Degenne</p></summary>
<p>

**Abstract:** We study the problem of the identification of m arms with largest means under a fixed error rate $δ$ (fixed-confidence Top-m identification), for misspecified linear bandit models. This problem is motivated by practical applications, especially in medicine and recommendation systems, where linear models are popular due to their simplicity and the existence of efficient algorithms, but in which data inevitably deviates from linearity. In this work, we first derive a tractable lower bound on the sample complexity of any $δ$-correct algorithm for the general Top-m identification problem. We show that knowing the scale of the deviation from linearity is necessary to exploit the structure of the problem. We then describe the first algorithm for this setting, which is both practical and adapts to the amount of misspecification. We derive an upper bound to its sample complexity which confirms this adaptivity and that matches the lower bound when $δ$ $\rightarrow$ 0. Finally, we evaluate our algorithm on both synthetic and real-world data, showing competitive performance with respect to existing baselines.

</p>
</details>

<details><summary><b>Synthesizing Speech from Intracranial Depth Electrodes using an Encoder-Decoder Framework</b>
<a href="https://arxiv.org/abs/2111.01457">arxiv:2111.01457</a>
&#x1F4C8; 3 <br>
<p>Jonas Kohler, Maarten C. Ottenhoff, Sophocles Goulis, Miguel Angrick, Albert J. Colon, Louis Wagner, Simon Tousseyn, Pieter L. Kubben, Christian Herff</p></summary>
<p>

**Abstract:** Speech Neuroprostheses have the potential to enable communication for people with dysarthria or anarthria. Recent advances have demonstrated high-quality text decoding and speech synthesis from electrocorticographic grids placed on the cortical surface. Here, we investigate a less invasive measurement modality, namely stereotactic EEG (sEEG) that provides sparse sampling from multiple brain regions, including subcortical regions. To evaluate whether sEEG can also be used to synthesize high-quality audio from neural recordings, we employ a recurrent encoder-decoder framework based on modern deep learning methods. We demonstrate that high-quality speech can be reconstructed from these minimally invasive recordings, despite a limited amount of training data. Finally, we utilize variational feature dropout to successfully identify the most informative electrode contacts.

</p>
</details>

<details><summary><b>A Review of Dialogue Systems: From Trained Monkeys to Stochastic Parrots</b>
<a href="https://arxiv.org/abs/2111.01414">arxiv:2111.01414</a>
&#x1F4C8; 3 <br>
<p>Atharv Singh Patlan, Shiven Tripathi, Shubham Korde</p></summary>
<p>

**Abstract:** In spoken dialogue systems, we aim to deploy artificial intelligence to build automated dialogue agents that can converse with humans. Dialogue systems are increasingly being designed to move beyond just imitating conversation and also improve from such interactions over time. In this survey, we present a broad overview of methods developed to build dialogue systems over the years. Different use cases for dialogue systems ranging from task-based systems to open domain chatbots motivate and necessitate specific systems. Starting from simple rule-based systems, research has progressed towards increasingly complex architectures trained on a massive corpus of datasets, like deep learning systems. Motivated with the intuition of resembling human dialogues, progress has been made towards incorporating emotions into the natural language generator, using reinforcement learning. While we see a trend of highly marginal improvement on some metrics, we find that limited justification exists for the metrics, and evaluation practices are not uniform. To conclude, we flag these concerns and highlight possible research directions.

</p>
</details>

<details><summary><b>Integrating Pretrained Language Model for Dialogue Policy Learning</b>
<a href="https://arxiv.org/abs/2111.01398">arxiv:2111.01398</a>
&#x1F4C8; 3 <br>
<p>Hongru Wang, Huimin Wang, Zezhong Wang, Kam-Fai Wong</p></summary>
<p>

**Abstract:** Reinforcement Learning (RL) has been witnessed its potential for training a dialogue policy agent towards maximizing the accumulated rewards given from users. However, the reward can be very sparse for it is usually only provided at the end of a dialog session, which causes unaffordable interaction requirements for an acceptable dialog agent. Distinguished from many efforts dedicated to optimizing the policy and recovering the reward alternatively which suffers from easily getting stuck in local optima and model collapse, we decompose the adversarial training into two steps: 1) we integrate a pre-trained language model as a discriminator to judge whether the current system action is good enough for the last user action (i.e., \textit{next action prediction}); 2) the discriminator gives and extra local dense reward to guide the agent's exploration. The experimental result demonstrates that our method significantly improves the complete rate (~4.4\%) and success rate (~8.0\%) of the dialogue system.

</p>
</details>

<details><summary><b>Understanding Entropic Regularization in GANs</b>
<a href="https://arxiv.org/abs/2111.01387">arxiv:2111.01387</a>
&#x1F4C8; 3 <br>
<p>Daria Reshetova, Yikun Bai, Xiugang Wu, Ayfer Ozgur</p></summary>
<p>

**Abstract:** Generative Adversarial Networks are a popular method for learning distributions from data by modeling the target distribution as a function of a known distribution. The function, often referred to as the generator, is optimized to minimize a chosen distance measure between the generated and target distributions. One commonly used measure for this purpose is the Wasserstein distance. However, Wasserstein distance is hard to compute and optimize, and in practice entropic regularization techniques are used to improve numerical convergence. The influence of regularization on the learned solution, however, remains not well-understood. In this paper, we study how several popular entropic regularizations of Wasserstein distance impact the solution in a simple benchmark setting where the generator is linear and the target distribution is high-dimensional Gaussian. We show that entropy regularization promotes the solution sparsification, while replacing the Wasserstein distance with the Sinkhorn divergence recovers the unregularized solution. Both regularization techniques remove the curse of dimensionality suffered by Wasserstein distance. We show that the optimal generator can be learned to accuracy $ε$ with $O(1/ε^2)$ samples from the target distribution. We thus conclude that these regularization techniques can improve the quality of the generator learned from empirical data for a large class of distributions.

</p>
</details>

<details><summary><b>Distributed Sparse Feature Selection in Communication-Restricted Networks</b>
<a href="https://arxiv.org/abs/2111.02802">arxiv:2111.02802</a>
&#x1F4C8; 2 <br>
<p>Hanie Barghi, Amir Najafi, Seyed Abolfazl Motahari</p></summary>
<p>

**Abstract:** This paper aims to propose and theoretically analyze a new distributed scheme for sparse linear regression and feature selection. The primary goal is to learn the few causal features of a high-dimensional dataset based on noisy observations from an unknown sparse linear model. However, the presumed training set which includes $n$ data samples in $\mathbb{R}^p$ is already distributed over a large network with $N$ clients connected through extremely low-bandwidth links. Also, we consider the asymptotic configuration of $1\ll N\ll n\ll p$. In order to infer the causal dimensions from the whole dataset, we propose a simple, yet effective method for information sharing in the network. In this regard, we theoretically show that the true causal features can be reliably recovered with negligible bandwidth usage of $O\left(N\log p\right)$ across the network. This yields a significantly lower communication cost in comparison with the trivial case of transmitting all the samples to a single node (centralized scenario), which requires $O\left(np\right)$ transmissions. Even more sophisticated schemes such as ADMM still have a communication complexity of $O\left(Np\right)$. Surprisingly, our sample complexity bound is proved to be the same (up to a constant factor) as the optimal centralized approach for a fixed performance measure in each node, while that of a naïve decentralized technique grows linearly with $N$. Theoretical guarantees in this paper are based on the recent analytic framework of debiased LASSO in Javanmard et al. (2019), and are supported by several computer experiments performed on both synthetic and real-world datasets.

</p>
</details>

<details><summary><b>Scalable mixed-domain Gaussian processes</b>
<a href="https://arxiv.org/abs/2111.02019">arxiv:2111.02019</a>
&#x1F4C8; 2 <br>
<p>Juho Timonen, Harri Lähdesmäki</p></summary>
<p>

**Abstract:** Gaussian process (GP) models that combine both categorical and continuous input variables have found use e.g. in longitudinal data analysis and computer experiments. However, standard inference for these models has the typical cubic scaling, and common scalable approximation schemes for GPs cannot be applied since the covariance function is non-continuous. In this work, we derive a basis function approximation scheme for mixed-domain covariance functions, which scales linearly with respect to the number of observations and total number of basis functions. The proposed approach is naturally applicable to Bayesian GP regression with arbitrary observation models. We demonstrate the approach in a longitudinal data modelling context and show that it approximates the exact GP model accurately, requiring only a fraction of the runtime compared to fitting the corresponding exact model.

</p>
</details>

<details><summary><b>HASHTAG: Hash Signatures for Online Detection of Fault-Injection Attacks on Deep Neural Networks</b>
<a href="https://arxiv.org/abs/2111.01932">arxiv:2111.01932</a>
&#x1F4C8; 2 <br>
<p>Mojan Javaheripi, Farinaz Koushanfar</p></summary>
<p>

**Abstract:** We propose HASHTAG, the first framework that enables high-accuracy detection of fault-injection attacks on Deep Neural Networks (DNNs) with provable bounds on detection performance. Recent literature in fault-injection attacks shows the severe DNN accuracy degradation caused by bit flips. In this scenario, the attacker changes a few weight bits during DNN execution by tampering with the program's DRAM memory. To detect runtime bit flips, HASHTAG extracts a unique signature from the benign DNN prior to deployment. The signature is later used to validate the integrity of the DNN and verify the inference output on the fly. We propose a novel sensitivity analysis scheme that accurately identifies the most vulnerable DNN layers to the fault-injection attack. The DNN signature is then constructed by encoding the underlying weights in the vulnerable layers using a low-collision hash function. When the DNN is deployed, new hashes are extracted from the target layers during inference and compared against the ground-truth signatures. HASHTAG incorporates a lightweight methodology that ensures a low-overhead and real-time fault detection on embedded platforms. Extensive evaluations with the state-of-the-art bit-flip attack on various DNNs demonstrate the competitive advantage of HASHTAG in terms of both attack detection and execution overhead.

</p>
</details>

<details><summary><b>A trained humanoid robot can perform human-like crossmodal social attention conflict resolution</b>
<a href="https://arxiv.org/abs/2111.01906">arxiv:2111.01906</a>
&#x1F4C8; 2 <br>
<p>Di Fu, Fares Abawi, Hugo Carneiro, Matthias Kerzel, Ziwei Chen, Erik Strahl, Xun Liu, Stefan Wermter</p></summary>
<p>

**Abstract:** Due to the COVID-19 pandemic, robots could be seen as potential resources in tasks like helping people work remotely, sustaining social distancing, and improving mental or physical health. To enhance human-robot interaction, it is essential for robots to become more socialised, via processing multiple social cues in a complex real-world environment. Our study adopted a neurorobotic paradigm of gaze-triggered audio-visual crossmodal integration to make an iCub robot express human-like social attention responses. At first, a behavioural experiment was conducted on 37 human participants. To improve ecological validity, a round-table meeting scenario with three masked animated avatars was designed with the middle one capable of performing gaze shift, and the other two capable of generating sound. The gaze direction and the sound location are either congruent or incongruent. Masks were used to cover all facial visual cues other than the avatars' eyes. We observed that the avatar's gaze could trigger crossmodal social attention with better human performance in the audio-visual congruent condition than in the incongruent condition. Then, our computational model, GASP, was trained to implement social cue detection, audio-visual saliency prediction, and selective attention. After finishing the model training, the iCub robot was exposed to similar laboratory conditions as human participants, demonstrating that it can replicate similar attention responses as humans regarding the congruency and incongruency performance, while overall the human performance was still superior. Therefore, this interdisciplinary work provides new insights on mechanisms of crossmodal social attention and how it can be modelled in robots in a complex environment.

</p>
</details>

<details><summary><b>Equivariant Deep Dynamical Model for Motion Prediction</b>
<a href="https://arxiv.org/abs/2111.01892">arxiv:2111.01892</a>
&#x1F4C8; 2 <br>
<p>Bahar Azari, Deniz Erdoğmuş</p></summary>
<p>

**Abstract:** Learning representations through deep generative modeling is a powerful approach for dynamical modeling to discover the most simplified and compressed underlying description of the data, to then use it for other tasks such as prediction. Most learning tasks have intrinsic symmetries, i.e., the input transformations leave the output unchanged, or the output undergoes a similar transformation. The learning process is, however, usually uninformed of these symmetries. Therefore, the learned representations for individually transformed inputs may not be meaningfully related. In this paper, we propose an SO(3) equivariant deep dynamical model (EqDDM) for motion prediction that learns a structured representation of the input space in the sense that the embedding varies with symmetry transformations. EqDDM is equipped with equivariant networks to parameterize the state-space emission and transition models. We demonstrate the superior predictive performance of the proposed model on various motion data.

</p>
</details>

<details><summary><b>Conformal testing: binary case with Markov alternatives</b>
<a href="https://arxiv.org/abs/2111.01885">arxiv:2111.01885</a>
&#x1F4C8; 2 <br>
<p>Vladimir Vovk, Ilia Nouretdinov, Alex Gammerman</p></summary>
<p>

**Abstract:** We continue study of conformal testing in binary model situations. In this note we consider Markov alternatives to the null hypothesis of exchangeability. We propose two new classes of conformal test martingales; one class is statistically efficient in our experiments, and the other class partially sacrifices statistical efficiency to gain computational efficiency.

</p>
</details>

<details><summary><b>A Survey of Fairness-Aware Federated Learning</b>
<a href="https://arxiv.org/abs/2111.01872">arxiv:2111.01872</a>
&#x1F4C8; 2 <br>
<p>Yuxin Shi, Han Yu, Cyril Leung</p></summary>
<p>

**Abstract:** Recent advances in Federated Learning (FL) have brought large-scale machine learning opportunities for massive distributed clients with performance and data privacy guarantees. However, most current works only focus on the interest of the central controller in FL, and ignore the interests of clients. This may result in unfairness which discourages clients from actively participating in the learning process and damages the sustainability of the whole FL system. Therefore, the topic of ensuring fairness in an FL is attracting a great deal of research interest. In recent years, diverse Fairness-Aware FL (FAFL) approaches have been proposed in an effort to achieve fairness in FL from different viewpoints. However, there is no comprehensive survey which helps readers gain insight into this interdisciplinary field. This paper aims to provide such a survey. By examining the fundamental and simplifying assumptions, as well as the notions of fairness adopted by existing literature in this field, we propose a taxonomy of FAFL approaches covering major steps in FL, including client selection, optimization, contribution evaluation and incentive distribution. In addition, we discuss the main metrics for experimentally evaluating the performance of FAFL approaches, and suggest some promising future research directions.

</p>
</details>

<details><summary><b>Off-Policy Correction for Deep Deterministic Policy Gradient Algorithms via Batch Prioritized Experience Replay</b>
<a href="https://arxiv.org/abs/2111.01865">arxiv:2111.01865</a>
&#x1F4C8; 2 <br>
<p>Dogan C. Cicek, Enes Duran, Baturay Saglam, Furkan B. Mutlu, Suleyman S. Kozat</p></summary>
<p>

**Abstract:** The experience replay mechanism allows agents to use the experiences multiple times. In prior works, the sampling probability of the transitions was adjusted according to their importance. Reassigning sampling probabilities for every transition in the replay buffer after each iteration is highly inefficient. Therefore, experience replay prioritization algorithms recalculate the significance of a transition when the corresponding transition is sampled to gain computational efficiency. However, the importance level of the transitions changes dynamically as the policy and the value function of the agent are updated. In addition, experience replay stores the transitions are generated by the previous policies of the agent that may significantly deviate from the most recent policy of the agent. Higher deviation from the most recent policy of the agent leads to more off-policy updates, which is detrimental for the agent. In this paper, we develop a novel algorithm, Batch Prioritizing Experience Replay via KL Divergence (KLPER), which prioritizes batch of transitions rather than directly prioritizing each transition. Moreover, to reduce the off-policyness of the updates, our algorithm selects one batch among a certain number of batches and forces the agent to learn through the batch that is most likely generated by the most recent policy of the agent. We combine our algorithm with Deep Deterministic Policy Gradient and Twin Delayed Deep Deterministic Policy Gradient and evaluate it on various continuous control tasks. KLPER provides promising improvements for deep deterministic continuous control algorithms in terms of sample efficiency, final performance, and stability of the policy during the training.

</p>
</details>

<details><summary><b>A Recommendation System to Enhance Midwives' Capacities in Low-Income Countries</b>
<a href="https://arxiv.org/abs/2111.01786">arxiv:2111.01786</a>
&#x1F4C8; 2 <br>
<p>Anna Guitart, Afsaneh Heydari, Eniola Olaleye, Jelena Ljubicic, Ana Fernández del Río, África Periáñez, Lauren Bellhouse</p></summary>
<p>

**Abstract:** Maternal and child mortality is a public health problem that disproportionately affects low- and middle-income countries. Every day, 800 women and 6,700 newborns die from complications related to pregnancy or childbirth. And for every maternal death, about 20 women suffer serious birth injuries. However, nearly all of these deaths and negative health outcomes are preventable. Midwives are key to revert this situation, and thus it is essential to strengthen their capacities and the quality of their education. This is the aim of the Safe Delivery App, a digital job aid and learning tool to enhance the knowledge, confidence and skills of health practitioners. Here, we use the behavioral logs of the App to implement a recommendation system that presents each midwife with suitable contents to continue gaining expertise. We focus on predicting the click-through rate, the probability that a given user will click on a recommended content. We evaluate four deep learning models and show that all of them produce highly accurate predictions.

</p>
</details>

<details><summary><b>Nearly Optimal Algorithms for Level Set Estimation</b>
<a href="https://arxiv.org/abs/2111.01768">arxiv:2111.01768</a>
&#x1F4C8; 2 <br>
<p>Blake Mason, Romain Camilleri, Subhojyoti Mukherjee, Kevin Jamieson, Robert Nowak, Lalit Jain</p></summary>
<p>

**Abstract:** The level set estimation problem seeks to find all points in a domain ${\cal X}$ where the value of an unknown function $f:{\cal X}\rightarrow \mathbb{R}$ exceeds a threshold $α$. The estimation is based on noisy function evaluations that may be acquired at sequentially and adaptively chosen locations in ${\cal X}$. The threshold value $α$ can either be \emph{explicit} and provided a priori, or \emph{implicit} and defined relative to the optimal function value, i.e. $α= (1-ε)f(x_\ast)$ for a given $ε> 0$ where $f(x_\ast)$ is the maximal function value and is unknown. In this work we provide a new approach to the level set estimation problem by relating it to recent adaptive experimental design methods for linear bandits in the Reproducing Kernel Hilbert Space (RKHS) setting. We assume that $f$ can be approximated by a function in the RKHS up to an unknown misspecification and provide novel algorithms for both the implicit and explicit cases in this setting with strong theoretical guarantees. Moreover, in the linear (kernel) setting, we show that our bounds are nearly optimal, namely, our upper bounds match existing lower bounds for threshold linear bandits. To our knowledge this work provides the first instance-dependent, non-asymptotic upper bounds on sample complexity of level-set estimation that match information theoretic lower bounds.

</p>
</details>

<details><summary><b>Regularization for Shuffled Data Problems via Exponential Family Priors on the Permutation Group</b>
<a href="https://arxiv.org/abs/2111.01767">arxiv:2111.01767</a>
&#x1F4C8; 2 <br>
<p>Zhenbang Wang, Emanuel Ben-David, Martin Slawski</p></summary>
<p>

**Abstract:** In the analysis of data sets consisting of (X, Y)-pairs, a tacit assumption is that each pair corresponds to the same observation unit. If, however, such pairs are obtained via record linkage of two files, this assumption can be violated as a result of mismatch error rooting, for example, in the lack of reliable identifiers in the two files. Recently, there has been a surge of interest in this setting under the term "Shuffled data" in which the underlying correct pairing of (X, Y)-pairs is represented via an unknown index permutation. Explicit modeling of the permutation tends to be associated with substantial overfitting, prompting the need for suitable methods of regularization. In this paper, we propose a flexible exponential family prior on the permutation group for this purpose that can be used to integrate various structures such as sparse and locally constrained shuffling. This prior turns out to be conjugate for canonical shuffled data problems in which the likelihood conditional on a fixed permutation can be expressed as product over the corresponding (X,Y)-pairs. Inference is based on the EM algorithm in which the intractable E-step is approximated by the Fisher-Yates algorithm. The M-step is shown to admit a significant reduction from $n^2$ to $n$ terms if the likelihood of (X,Y)-pairs has exponential family form as in the case of generalized linear models. Comparisons on synthetic and real data show that the proposed approach compares favorably to competing methods.

</p>
</details>

<details><summary><b>Spiking Generative Adversarial Networks With a Neural Network Discriminator: Local Training, Bayesian Models, and Continual Meta-Learning</b>
<a href="https://arxiv.org/abs/2111.01750">arxiv:2111.01750</a>
&#x1F4C8; 2 <br>
<p>Bleema Rosenfeld, Osvaldo Simeone, Bipin Rajendran</p></summary>
<p>

**Abstract:** Neuromorphic data carries information in spatio-temporal patterns encoded by spikes. Accordingly, a central problem in neuromorphic computing is training spiking neural networks (SNNs) to reproduce spatio-temporal spiking patterns in response to given spiking stimuli. Most existing approaches model the input-output behavior of an SNN in a deterministic fashion by assigning each input to a specific desired output spiking sequence. In contrast, in order to fully leverage the time-encoding capacity of spikes, this work proposes to train SNNs so as to match distributions of spiking signals rather than individual spiking signals. To this end, the paper introduces a novel hybrid architecture comprising a conditional generator, implemented via an SNN, and a discriminator, implemented by a conventional artificial neural network (ANN). The role of the ANN is to provide feedback during training to the SNN within an adversarial iterative learning strategy that follows the principle of generative adversarial network (GANs). In order to better capture multi-modal spatio-temporal distribution, the proposed approach -- termed SpikeGAN -- is further extended to support Bayesian learning of the generator's weight. Finally, settings with time-varying statistics are addressed by proposing an online meta-learning variant of SpikeGAN. Experiments bring insights into the merits of the proposed approach as compared to existing solutions based on (static) belief networks and maximum likelihood (or empirical risk minimization).

</p>
</details>

<details><summary><b>Designing Inherently Interpretable Machine Learning Models</b>
<a href="https://arxiv.org/abs/2111.01743">arxiv:2111.01743</a>
&#x1F4C8; 2 <br>
<p>Agus Sudjianto, Aijun Zhang</p></summary>
<p>

**Abstract:** Interpretable machine learning (IML) becomes increasingly important in highly regulated industry sectors related to the health and safety or fundamental rights of human beings. In general, the inherently IML models should be adopted because of their transparency and explainability, while black-box models with model-agnostic explainability can be more difficult to defend under regulatory scrutiny. For assessing inherent interpretability of a machine learning model, we propose a qualitative template based on feature effects and model architecture constraints. It provides the design principles for high-performance IML model development, with examples given by reviewing our recent works on ExNN, GAMI-Net, SIMTree, and the Aletheia toolkit for local linear interpretability of deep ReLU networks. We further demonstrate how to design an interpretable ReLU DNN model with evaluation of conceptual soundness for a real case study of predicting credit default in home lending. We hope that this work will provide a practical guide of developing inherently IML models in high risk applications in banking industry, as well as other sectors.

</p>
</details>

<details><summary><b>Bayes-Newton Methods for Approximate Bayesian Inference with PSD Guarantees</b>
<a href="https://arxiv.org/abs/2111.01721">arxiv:2111.01721</a>
&#x1F4C8; 2 <br>
<p>William J. Wilkinson, Simo Särkkä, Arno Solin</p></summary>
<p>

**Abstract:** We formulate natural gradient variational inference (VI), expectation propagation (EP), and posterior linearisation (PL) as extensions of Newton's method for optimising the parameters of a Bayesian posterior distribution. This viewpoint explicitly casts inference algorithms under the framework of numerical optimisation. We show that common approximations to Newton's method from the optimisation literature, namely Gauss-Newton and quasi-Newton methods (e.g., the BFGS algorithm), are still valid under this 'Bayes-Newton' framework. This leads to a suite of novel algorithms which are guaranteed to result in positive semi-definite covariance matrices, unlike standard VI and EP. Our unifying viewpoint provides new insights into the connections between various inference schemes. All the presented methods apply to any model with a Gaussian prior and non-conjugate likelihood, which we demonstrate with (sparse) Gaussian processes and state space models.

</p>
</details>

<details><summary><b>OSOA: One-Shot Online Adaptation of Deep Generative Models for Lossless Compression</b>
<a href="https://arxiv.org/abs/2111.01662">arxiv:2111.01662</a>
&#x1F4C8; 2 <br>
<p>Chen Zhang, Shifeng Zhang, Fabio Maria Carlucci, Zhenguo Li</p></summary>
<p>

**Abstract:** Explicit deep generative models (DGMs), e.g., VAEs and Normalizing Flows, have shown to offer an effective data modelling alternative for lossless compression. However, DGMs themselves normally require large storage space and thus contaminate the advantage brought by accurate data density estimation. To eliminate the requirement of saving separate models for different target datasets, we propose a novel setting that starts from a pretrained deep generative model and compresses the data batches while adapting the model with a dynamical system for only one epoch. We formalise this setting as that of One-Shot Online Adaptation (OSOA) of DGMs for lossless compression and propose a vanilla algorithm under this setting. Experimental results show that vanilla OSOA can save significant time versus training bespoke models and space versus using one model for all targets. With the same adaptation step number or adaptation time, it is shown vanilla OSOA can exhibit better space efficiency, e.g., $47\%$ less space, than fine-tuning the pretrained model and saving the fine-tuned model. Moreover, we showcase the potential of OSOA and motivate more sophisticated OSOA algorithms by showing further space or time efficiency with multiple updates per batch and early stopping.

</p>
</details>

<details><summary><b>Elucidating Noisy Data via Uncertainty-Aware Robust Learning</b>
<a href="https://arxiv.org/abs/2111.01632">arxiv:2111.01632</a>
&#x1F4C8; 2 <br>
<p>Jeongeun Park, Seungyoun Shin, Sangheum Hwang, Sungjoon Choi</p></summary>
<p>

**Abstract:** Robust learning methods aim to learn a clean target distribution from noisy and corrupted training data where a specific corruption pattern is often assumed a priori. Our proposed method can not only successfully learn the clean target distribution from a dirty dataset but also can estimate the underlying noise pattern. To this end, we leverage a mixture-of-experts model that can distinguish two different types of predictive uncertainty, aleatoric and epistemic uncertainty. We show that the ability to estimate the uncertainty plays a significant role in elucidating the corruption patterns as these two objectives are tightly intertwined. We also present a novel validation scheme for evaluating the performance of the corruption pattern estimation. Our proposed method is extensively assessed in terms of both robustness and corruption pattern estimation through a number of domains, including computer vision and natural language processing.

</p>
</details>

<details><summary><b>Learning Robotic Ultrasound Scanning Skills via Human Demonstrations and Guided Explorations</b>
<a href="https://arxiv.org/abs/2111.01625">arxiv:2111.01625</a>
&#x1F4C8; 2 <br>
<p>Xutian Deng, Yiting Chen, Fei Chen, Miao Li</p></summary>
<p>

**Abstract:** Medical ultrasound has become a routine examination approach nowadays and is widely adopted for different medical applications, so it is desired to have a robotic ultrasound system to perform the ultrasound scanning autonomously. However, the ultrasound scanning skill is considerably complex, which highly depends on the experience of the ultrasound physician. In this paper, we propose a learning-based approach to learn the robotic ultrasound scanning skills from human demonstrations. First, the robotic ultrasound scanning skill is encapsulated into a high-dimensional multi-modal model, which takes the ultrasound images, the pose/position of the probe and the contact force into account. Second, we leverage the power of imitation learning to train the multi-modal model with the training data collected from the demonstrations of experienced ultrasound physicians. Finally, a post-optimization procedure with guided explorations is proposed to further improve the performance of the learned model. Robotic experiments are conducted to validate the advantages of our proposed framework and the learned models.

</p>
</details>

<details><summary><b>Likelihood-Free Inference in State-Space Models with Unknown Dynamics</b>
<a href="https://arxiv.org/abs/2111.01555">arxiv:2111.01555</a>
&#x1F4C8; 2 <br>
<p>Alexander Aushev, Thong Tran, Henri Pesonen, Andrew Howes, Samuel Kaski</p></summary>
<p>

**Abstract:** We introduce a method for inferring and predicting latent states in the important and difficult case of state-space models where observations can only be simulated, and transition dynamics are unknown. In this setting, the likelihood of observations is not available and only synthetic observations can be generated from a black-box simulator. We propose a way of doing likelihood-free inference (LFI) of states and state prediction with a limited number of simulations. Our approach uses a multi-output Gaussian process for state inference, and a Bayesian Neural Network as a model of the transition dynamics for state prediction. We improve upon existing LFI methods for the inference task, while also accurately learning transition dynamics. The proposed method is necessary for modelling inverse problems in dynamical systems with computationally expensive simulations, as demonstrated in experiments with non-stationary user models.

</p>
</details>

<details><summary><b>HydraText: Multi-objective Optimization for Adversarial Textual Attack</b>
<a href="https://arxiv.org/abs/2111.01528">arxiv:2111.01528</a>
&#x1F4C8; 2 <br>
<p>Shengcai Liu, Ning Lu, Cheng Chen, Chao Qian, Ke Tang</p></summary>
<p>

**Abstract:** The field of adversarial textual attack has significantly grown over the last years, where the commonly considered objective is to craft adversarial examples that can successfully fool the target models. However, the imperceptibility of attacks, which is also an essential objective, is often left out by previous studies. In this work, we advocate considering both objectives at the same time, and propose a novel multi-optimization approach (dubbed HydraText) with provable performance guarantee to achieve successful attacks with high imperceptibility. We demonstrate the efficacy of HydraText through extensive experiments under both score-based and decision-based settings, involving five modern NLP models across five benchmark datasets. In comparison to existing state-of-the-art attacks, HydraText consistently achieves simultaneously higher success rates, lower modification rates, and higher semantic similarity to the original texts. A human evaluation study shows that the adversarial examples crafted by HydraText maintain validity and naturality well. Finally, these examples also exhibit good transferability and can bring notable robustness improvement to the target models by adversarial training.

</p>
</details>

<details><summary><b>A Hybrid Approach for Learning to Shift and Grasp with Elaborate Motion Primitives</b>
<a href="https://arxiv.org/abs/2111.01510">arxiv:2111.01510</a>
&#x1F4C8; 2 <br>
<p>Zohar Feldman, Hanna Ziesche, Ngo Anh Vien, Dotan Di Castro</p></summary>
<p>

**Abstract:** Many possible fields of application of robots in real world settings hinge on the ability of robots to grasp objects. As a result, robot grasping has been an active field of research for many years. With our publication we contribute to the endeavor of enabling robots to grasp, with a particular focus on bin picking applications. Bin picking is especially challenging due to the often cluttered and unstructured arrangement of objects and the often limited graspability of objects by simple top down grasps. To tackle these challenges, we propose a fully self-supervised reinforcement learning approach based on a hybrid discrete-continuous adaptation of soft actor-critic (SAC). We employ parametrized motion primitives for pushing and grasping movements in order to enable a flexibly adaptable behavior to the difficult setups we consider. Furthermore, we use data augmentation to increase sample efficiency. We demonnstrate our proposed method on challenging picking scenarios in which planar grasp learning or action discretization methods would face a lot of difficulties

</p>
</details>

<details><summary><b>DAGSurv: Directed Acyclic Graph Based Survival Analysis Using Deep Neural Networks</b>
<a href="https://arxiv.org/abs/2111.01482">arxiv:2111.01482</a>
&#x1F4C8; 2 <br>
<p>Ansh Kumar Sharma, Rahul Kukreja, Ranjitha Prasad, Shilpa Rao</p></summary>
<p>

**Abstract:** Causal structures for observational survival data provide crucial information regarding the relationships between covariates and time-to-event. We derive motivation from the information theoretic source coding argument, and show that incorporating the knowledge of the directed acyclic graph (DAG) can be beneficial if suitable source encoders are employed. As a possible source encoder in this context, we derive a variational inference based conditional variational autoencoder for causal structured survival prediction, which we refer to as DAGSurv. We illustrate the performance of DAGSurv on low and high-dimensional synthetic datasets, and real-world datasets such as METABRIC and GBSG. We demonstrate that the proposed method outperforms other survival analysis baselines such as Cox Proportional Hazards, DeepSurv and Deephit, which are oblivious to the underlying causal relationship between data entities.

</p>
</details>

<details><summary><b>Variational message passing (VMP) applied to LDA</b>
<a href="https://arxiv.org/abs/2111.01480">arxiv:2111.01480</a>
&#x1F4C8; 2 <br>
<p>Rebecca M. C. Taylor, Johan A. du Preez</p></summary>
<p>

**Abstract:** Variational Bayes (VB) applied to latent Dirichlet allocation (LDA) is the original inference mechanism for LDA. Many variants of VB for LDA, as well as for VB in general, have been developed since LDA's inception in 2013, but standard VB is still widely applied to LDA. Variational message passing (VMP) is the message passing equivalent of VB and is a useful tool for constructing a variational inference solution for a large variety of conjugate exponential graphical models (there is also a non conjugate variant available for other models). In this article we present the VMP equations for LDA and also provide a brief discussion of the equations. We hope that this will assist others when deriving variational inference solutions to other similar graphical models.

</p>
</details>

<details><summary><b>WaveSense: Efficient Temporal Convolutions with Spiking Neural Networks for Keyword Spotting</b>
<a href="https://arxiv.org/abs/2111.01456">arxiv:2111.01456</a>
&#x1F4C8; 2 <br>
<p>Philipp Weidel, Sadique Sheik</p></summary>
<p>

**Abstract:** Ultra-low power local signal processing is a crucial aspect for edge applications on always-on devices. Neuromorphic processors emulating spiking neural networks show great computational power while fulfilling the limited power budget as needed in this domain. In this work we propose spiking neural dynamics as a natural alternative to dilated temporal convolutions. We extend this idea to WaveSense, a spiking neural network inspired by the WaveNet architecture. WaveSense uses simple neural dynamics, fixed time-constants and a simple feed-forward architecture and hence is particularly well suited for a neuromorphic implementation. We test the capabilities of this model on several datasets for keyword-spotting. The results show that the proposed network beats the state of the art of other spiking neural networks and reaches near state-of-the-art performance of artificial neural networks such as CNNs and LSTMs.

</p>
</details>

<details><summary><b>Efficient Learning of the Parameters of Non-Linear Models using Differentiable Resampling in Particle Filters</b>
<a href="https://arxiv.org/abs/2111.01409">arxiv:2111.01409</a>
&#x1F4C8; 2 <br>
<p>Conor Rosato, Paul Horridge, Thomas B. Schön, Simon Maskell</p></summary>
<p>

**Abstract:** It has been widely documented that the sampling and resampling steps in particle filters cannot be differentiated. The {\itshape reparameterisation trick} was introduced to allow the sampling step to be reformulated into a differentiable function. We extend the {\itshape reparameterisation trick} to include the stochastic input to resampling therefore limiting the discontinuities in the gradient calculation after this step. Knowing the gradients of the prior and likelihood allows us to run particle Markov Chain Monte Carlo (p-MCMC) and use the No-U-Turn Sampler (NUTS) as the proposal when estimating parameters.
  We compare the Metropolis-adjusted Langevin algorithm (MALA), Hamiltonian Monte Carlo with different number of steps and NUTS. We consider two state-space models and show that NUTS improves the mixing of the Markov chain and can produce more accurate results in less computational time.

</p>
</details>

<details><summary><b>Solving Partial Differential Equations with Point Source Based on Physics-Informed Neural Networks</b>
<a href="https://arxiv.org/abs/2111.01394">arxiv:2111.01394</a>
&#x1F4C8; 2 <br>
<p>Xiang Huang, Hongsheng Liu, Beiji Shi, Zidong Wang, Kang Yang, Yang Li, Bingya Weng, Min Wang, Haotian Chu, Jing Zhou, Fan Yu, Bei Hua, Lei Chen, Bin Dong</p></summary>
<p>

**Abstract:** In recent years, deep learning technology has been used to solve partial differential equations (PDEs), among which the physics-informed neural networks (PINNs) emerges to be a promising method for solving both forward and inverse PDE problems. PDEs with a point source that is expressed as a Dirac delta function in the governing equations are mathematical models of many physical processes. However, they cannot be solved directly by conventional PINNs method due to the singularity brought by the Dirac delta function. We propose a universal solution to tackle this problem with three novel techniques. Firstly the Dirac delta function is modeled as a continuous probability density function to eliminate the singularity; secondly a lower bound constrained uncertainty weighting algorithm is proposed to balance the PINNs losses between point source area and other areas; and thirdly a multi-scale deep neural network with periodic activation function is used to improve the accuracy and convergence speed of the PINNs method. We evaluate the proposed method with three representative PDEs, and the experimental results show that our method outperforms existing deep learning-based methods with respect to the accuracy, the efficiency and the versatility.

</p>
</details>

<details><summary><b>Overlapping and nonoverlapping models</b>
<a href="https://arxiv.org/abs/2111.01392">arxiv:2111.01392</a>
&#x1F4C8; 2 <br>
<p>Huan Qing</p></summary>
<p>

**Abstract:** Consider a directed network with $K_{r}$ row communities and $K_{c}$ column communities. Previous works found that modeling directed networks in which all nodes have overlapping property requires $K_{r}=K_{c}$ for identifiability. In this paper, we propose an overlapping and nonoverlapping model to study directed networks in which row nodes have overlapping property while column nodes do not. The proposed model is identifiable when $K_{r}\leq K_{c}$. Meanwhile, we provide one identifiable model as extension of ONM to model directed networks with variation in node degree. Two spectral algorithms with theoretical guarantee on consistent estimations are designed to fit the models. A small scale of numerical studies are used to illustrate the algorithms.

</p>
</details>

<details><summary><b>Is Complexity Important for Philosophy of Mind?</b>
<a href="https://arxiv.org/abs/2112.03877">arxiv:2112.03877</a>
&#x1F4C8; 1 <br>
<p>Kristina Šekrst, Sandro Skansi</p></summary>
<p>

**Abstract:** Computational complexity has often been ignored in philosophy of mind, in philosophical artificial intelligence studies. The purpose of this paper is threefold. First and foremost, to show the importance of complexity rather than computability in philosophical and AI problems. Second, to rephrase the notion of computability in terms of solvability, i.e. treating computability as non-sufficient for establishing intelligence. The Church-Turing thesis is therefore revisited and rephrased in order to capture the ontological background of spatial and temporal complexity. Third, to emphasize ontological differences between different time complexities, which seem to provide a solid base towards better understanding of artificial intelligence in general.

</p>
</details>

<details><summary><b>Automated, real-time hospital ICU emergency signaling: A field-level implementation</b>
<a href="https://arxiv.org/abs/2111.01999">arxiv:2111.01999</a>
&#x1F4C8; 1 <br>
<p>Nazifa M Shemonti, Shifat Uddin, Saifur Rahman, Tarem Ahmed, Mohammad Faisal Uddin</p></summary>
<p>

**Abstract:** Contemporary patient surveillance systems have streamlined central surveillance into the electronic health record interface. They are able to process the sheer volume of patient data by adopting machine learning approaches. However, these systems are not suitable for implementation in many hospitals, mostly in developing countries, with limited human, financial, and technological resources. Through conducting thorough research on intensive care facilities, we designed a novel central patient monitoring system and in this paper, we describe the working prototype of our system. The proposed prototype comprises of inexpensive peripherals and simplistic user interface. Our central patient monitoring system implements Kernel-based On-line Anomaly Detection (KOAD) algorithm for emergency event signaling. By evaluating continuous patient data, we show that the system is able to detect critical events in real-time reliably and has low false alarm rate.

</p>
</details>

<details><summary><b>A MIMO Radar-Based Metric Learning Approach for Activity Recognition</b>
<a href="https://arxiv.org/abs/2111.01939">arxiv:2111.01939</a>
&#x1F4C8; 1 <br>
<p>Fady Aziz, Omar Metwally, Pascal Weller, Urs Schneider, Marco F. Huber</p></summary>
<p>

**Abstract:** Human activity recognition is seen of great importance in the medical and surveillance fields. Radar has shown great feasibility for this field based on the captured micro-Doppler (μ-D) signatures. In this paper, a MIMO radar is used to formulate a novel micro-motion spectrogram for the angular velocity (μ-ω) in non-tangential scenarios. Combining both the μ-D and the μ-ω signatures have shown better performance. Classification accuracy of 88.9% was achieved based on a metric learning approach. The experimental setup was designed to capture micro-motion signatures on different aspect angles and line of sight (LOS). The utilized training dataset was of smaller size compared to the state-of-the-art techniques, where eight activities were captured. A few-shot learning approach is used to adapt the pre-trained model for fall detection. The final model has shown a classification accuracy of 86.42% for ten activities.

</p>
</details>

<details><summary><b>Duality for Continuous Graphical Models</b>
<a href="https://arxiv.org/abs/2111.01938">arxiv:2111.01938</a>
&#x1F4C8; 1 <br>
<p>Mehdi Molkaraie</p></summary>
<p>

**Abstract:** The dual normal factor graph and the factor graph duality theorem have been considered for discrete graphical models. In this paper, we show an application of the factor graph duality theorem to continuous graphical models. Specifically, we propose a method to solve exactly the Gaussian graphical models defined on the ladder graph if certain conditions on the local covariance matrices are satisfied. Unlike the conventional approaches, the efficiency of the method depends on the position of the zeros in the local covariance matrices. The method and details of the dualization are illustrated on two toy examples.

</p>
</details>

<details><summary><b>Audacity of huge: overcoming challenges of data scarcity and data quality for machine learning in computational materials discovery</b>
<a href="https://arxiv.org/abs/2111.01905">arxiv:2111.01905</a>
&#x1F4C8; 1 <br>
<p>Aditya Nandy, Chenru Duan, Heather J. Kulik</p></summary>
<p>

**Abstract:** Machine learning (ML)-accelerated discovery requires large amounts of high-fidelity data to reveal predictive structure-property relationships. For many properties of interest in materials discovery, the challenging nature and high cost of data generation has resulted in a data landscape that is both scarcely populated and of dubious quality. Data-driven techniques starting to overcome these limitations include the use of consensus across functionals in density functional theory, the development of new functionals or accelerated electronic structure theories, and the detection of where computationally demanding methods are most necessary. When properties cannot be reliably simulated, large experimental data sets can be used to train ML models. In the absence of manual curation, increasingly sophisticated natural language processing and automated image analysis are making it possible to learn structure-property relationships from the literature. Models trained on these data sets will improve as they incorporate community feedback.

</p>
</details>

<details><summary><b>Source-to-Source Automatic Differentiation of OpenMP Parallel Loops</b>
<a href="https://arxiv.org/abs/2111.01861">arxiv:2111.01861</a>
&#x1F4C8; 1 <br>
<p>Jan Hückelheim, Laurent Hascoët</p></summary>
<p>

**Abstract:** This paper presents our work toward correct and efficient automatic differentiation of OpenMP parallel worksharing loops in forward and reverse mode. Automatic differentiation is a method to obtain gradients of numerical programs, which are crucial in optimization, uncertainty quantification, and machine learning. The computational cost to compute gradients is a common bottleneck in practice. For applications that are parallelized for multicore CPUs or GPUs using OpenMP, one also wishes to compute the gradients in parallel. We propose a framework to reason about the correctness of the generated derivative code, from which we justify our OpenMP extension to the differentiation model. We implement this model in the automatic differentiation tool Tapenade and present test cases that are differentiated following our extended differentiation procedure. Performance of the generated derivative programs in forward and reverse mode is better than sequential, although our reverse mode often scales worse than the input programs.

</p>
</details>

<details><summary><b>Coordinate Linear Variance Reduction for Generalized Linear Programming</b>
<a href="https://arxiv.org/abs/2111.01842">arxiv:2111.01842</a>
&#x1F4C8; 1 <br>
<p>Chaobing Song, Cheuk Yin Lin, Stephen J. Wright, Jelena Diakonikolas</p></summary>
<p>

**Abstract:** We study a class of generalized linear programs (GLP) in a large-scale setting, which includes possibly simple nonsmooth convex regularizer and simple convex set constraints. By reformulating GLP as an equivalent convex-concave min-max problem, we show that the linear structure in the problem can be used to design an efficient, scalable first-order algorithm, to which we give the name \emph{Coordinate Linear Variance Reduction} (\textsc{clvr}; pronounced "clever"). \textsc{clvr} is an incremental coordinate method with implicit variance reduction that outputs an \emph{affine combination} of the dual variable iterates. \textsc{clvr} yields improved complexity results for (GLP) that depend on the max row norm of the linear constraint matrix in (GLP) rather than the spectral norm. When the regularization terms and constraints are separable, \textsc{clvr} admits an efficient lazy update strategy that makes its complexity bounds scale with the number of nonzero elements of the linear constraint matrix in (GLP) rather than the matrix dimensions. We show that Distributionally Robust Optimization (DRO) problems with ambiguity sets based on both $f$-divergence and Wasserstein metrics can be reformulated as (GLPs) by introducing sparsely connected auxiliary variables. We complement our theoretical guarantees with numerical experiments that verify our algorithm's practical effectiveness, both in terms of wall-clock time and number of data passes.

</p>
</details>

<details><summary><b>OnSlicing: Online End-to-End Network Slicing with Reinforcement Learning</b>
<a href="https://arxiv.org/abs/2111.01616">arxiv:2111.01616</a>
&#x1F4C8; 1 <br>
<p>Qiang Liu, Nakjung Choi, Tao Han</p></summary>
<p>

**Abstract:** Network slicing allows mobile network operators to virtualize infrastructures and provide customized slices for supporting various use cases with heterogeneous requirements. Online deep reinforcement learning (DRL) has shown promising potential in solving network problems and eliminating the simulation-to-reality discrepancy. Optimizing cross-domain resources with online DRL is, however, challenging, as the random exploration of DRL violates the service level agreement (SLA) of slices and resource constraints of infrastructures. In this paper, we propose OnSlicing, an online end-to-end network slicing system, to achieve minimal resource usage while satisfying slices' SLA. OnSlicing allows individualized learning for each slice and maintains its SLA by using a novel constraint-aware policy update method and proactive baseline switching mechanism. OnSlicing complies with resource constraints of infrastructures by using a unique design of action modification in slices and parameter coordination in infrastructures. OnSlicing further mitigates the poor performance of online learning during the early learning stage by offline imitating a rule-based solution. Besides, we design four new domain managers to enable dynamic resource configuration in radio access, transport, core, and edge networks, respectively, at a timescale of subseconds. We implement OnSlicing on an end-to-end slicing testbed designed based on OpenAirInterface with both 4G LTE and 5G NR, OpenDayLight SDN platform, and OpenAir-CN core network. The experimental results show that OnSlicing achieves 61.3% usage reduction as compared to the rule-based solution and maintains nearly zero violation (0.06%) throughout the online learning phase. As online learning is converged, OnSlicing reduces 12.5% usage without any violations as compared to the state-of-the-art online DRL solution.

</p>
</details>

<details><summary><b>Strategyproof and Proportionally Fair Facility Location</b>
<a href="https://arxiv.org/abs/2111.01566">arxiv:2111.01566</a>
&#x1F4C8; 1 <br>
<p>Haris Aziz, Alexander Lam, Barton E. Lee, Toby Walsh</p></summary>
<p>

**Abstract:** We focus on a simple, one-dimensional collective decision problem (often referred to as the facility location problem) and explore issues of strategyproofness and proportional fairness. We present several characterization results for mechanisms that satisfy strategyproofness and varying levels of proportional fairness. We also characterize one of the mechanisms as the unique equilibrium outcome for any mechanism that satisfies natural fairness and monotonicity properties. Finally, we identify strategyproof and proportionally fair mechanisms that provide the best welfare-optimal approximation among all mechanisms that satisfy the corresponding fairness axiom.

</p>
</details>

<details><summary><b>FedFly: Towards Migration in Edge-based Distributed Federated Learning</b>
<a href="https://arxiv.org/abs/2111.01516">arxiv:2111.01516</a>
&#x1F4C8; 1 <br>
<p>Rehmat Ullah, Di Wu, Paul Harvey, Peter Kilpatrick, Ivor Spence, Blesson Varghese</p></summary>
<p>

**Abstract:** Federated learning (FL) is a privacy-preserving distributed machine learning technique that trains models without having direct access to the original data generated on devices. Since devices may be resource constrained, offloading can be used to improve FL performance by transferring computational workload from devices to edge servers. However, due to mobility, devices participating in FL may leave the network during training and need to connect to a different edge server. This is challenging because the offloaded computations from edge server need to be migrated. In line with this assertion, we present FedFly, which is, to the best of our knowledge, the first work to migrate a deep neural network (DNN) when devices move between edge servers during FL training. Our empirical results on the CIFAR-10 dataset, with both balanced and imbalanced data distribution support our claims that FedFly can reduce training time by up to 33% when a device moves after 50% of the training is completed, and by up to 45% when 90% of the training is completed when compared to state-of-the-art offloading approach in FL. FedFly has negligible overhead of 2 seconds and does not compromise accuracy. Finally, we highlight a number of open research issues for further investigation. FedFly can be downloaded from https://github.com/qub-blesson/FedFly

</p>
</details>

<details><summary><b>ArchABM: an agent-based simulator of human interaction with the built environment. $CO_2$ and viral load analysis for indoor air quality</b>
<a href="https://arxiv.org/abs/2111.01484">arxiv:2111.01484</a>
&#x1F4C8; 1 <br>
<p>Iñigo Martinez, Jan L. Bruse, Ane M. Florez-Tapia, Elisabeth Viles, Igor G. Olaizola</p></summary>
<p>

**Abstract:** Recent evidence suggests that SARS-CoV-2, which is the virus causing a global pandemic in 2020, is predominantly transmitted via airborne aerosols in indoor environments. This calls for novel strategies when assessing and controlling a building's indoor air quality (IAQ). IAQ can generally be controlled by ventilation and/or policies to regulate human-building-interaction. However, in a building, occupants use rooms in different ways, and it may not be obvious which measure or combination of measures leads to a cost- and energy-effective solution ensuring good IAQ across the entire building. Therefore, in this article, we introduce a novel agent-based simulator, ArchABM, designed to assist in creating new or adapt existing buildings by estimating adequate room sizes, ventilation parameters and testing the effect of policies while taking into account IAQ as a result of complex human-building interaction patterns. A recently published aerosol model was adapted to calculate time-dependent carbon dioxide ($CO_2$) and virus quanta concentrations in each room and inhaled $CO_2$ and virus quanta for each occupant over a day as a measure of physiological response. ArchABM is flexible regarding the aerosol model and the building layout due to its modular architecture, which allows implementing further models, any number and size of rooms, agents, and actions reflecting human-building interaction patterns. We present a use case based on a real floor plan and working schedules adopted in our research center. This study demonstrates how advanced simulation tools can contribute to improving IAQ across a building, thereby ensuring a healthy indoor environment.

</p>
</details>

<details><summary><b>iCallee: Recovering Call Graphs for Binaries</b>
<a href="https://arxiv.org/abs/2111.01415">arxiv:2111.01415</a>
&#x1F4C8; 1 <br>
<p>Wenyu Zhu, Zhiyao Feng, Zihan Zhang, Zhijian Ou, Min Yang, Chao Zhang</p></summary>
<p>

**Abstract:** Recovering programs' call graphs is crucial for inter-procedural analysis tasks and applications based on them. The core challenge is recognizing targets of indirect calls (i.e., indirect callees). It becomes more challenging if target programs are in binary forms, due to information loss in binaries. Existing indirect callee recognition solutions for binaries all have high false positives and negatives, making call graphs inaccurate.
  In this paper, we propose a new solution iCallee based on the Siamese Neural Network, inspired by the advances in question-answering applications. The key insight is that, neural networks can learn to answer whether a callee function is a potential target of an indirect callsite by comprehending their contexts, i.e., instructions nearby callsites and of callees. Following this insight, we first preprocess target binaries to extract contexts of callsites and callees. Then, we build a customized Natural Language Processing (NLP) model applicable to assembly language. Further, we collect abundant pairs of callsites and callees, and embed their contexts with the NLP model, then train a Siamese network and a classifier to answer the callsite-callee question. We have implemented a prototype of iCallee and evaluated it on several groups of targets. Evaluation results showed that, our solution could match callsites to callees with an F1-Measure of 93.7%, recall of 93.8%, and precision of 93.5%, much better than state-of-the-art solutions. To show its usefulness, we apply iCallee to two specific applications - binary code similarity detection and binary program hardening, and found that it could greatly improve state-of-the-art solutions.

</p>
</details>

<details><summary><b>A Comparative Analysis of Machine Learning Algorithms for Intrusion Detection in Edge-Enabled IoT Networks</b>
<a href="https://arxiv.org/abs/2111.01383">arxiv:2111.01383</a>
&#x1F4C8; 1 <br>
<p>Poornima Mahadevappa, Syeda Mariam Muzammal, Raja Kumar Murugesan</p></summary>
<p>

**Abstract:** A significant increase in the number of interconnected devices and data communication through wireless networks has given rise to various threats, risks and security concerns. Internet of Things (IoT) applications is deployed in almost every field of daily life, including sensitive environments. The edge computing paradigm has complemented IoT applications by moving the computational processing near the data sources. Among various security models, Machine Learning (ML) based intrusion detection is the most conceivable defense mechanism to combat the anomalous behavior in edge-enabled IoT networks. The ML algorithms are used to classify the network traffic into normal and malicious attacks. Intrusion detection is one of the challenging issues in the area of network security. The research community has proposed many intrusion detection systems. However, the challenges involved in selecting suitable algorithm(s) to provide security in edge-enabled IoT networks exist. In this paper, a comparative analysis of conventional machine learning classification algorithms has been performed to categorize the network traffic on NSL-KDD dataset using Jupyter on Pycharm tool. It can be observed that Multi-Layer Perception (MLP) has dependencies between input and output and relies more on network configuration for intrusion detection. Therefore, MLP can be more appropriate for edge-based IoT networks with a better training time of 1.2 seconds and testing accuracy of 79%.

</p>
</details>

<details><summary><b>IoT to monitor people flow in areas of public interest</b>
<a href="https://arxiv.org/abs/2111.04465">arxiv:2111.04465</a>
&#x1F4C8; 0 <br>
<p>Damiano Perri, Marco Simonetti, Alex Bordini, Simone Cimarelli, Osvaldo Gervasi</p></summary>
<p>

**Abstract:** The unexpected historical period we are living has abruptly pushed us to loosen any sort of interaction between individuals, gradually forcing us to deal with new ways to allow compliance with safety distances; indeed the present situation has demonstrated more than ever how critical it is to be able to properly organize our travel plans, put people in safe conditions, and avoid harmful circumstances. The aim of this research is to set up a system to monitor the flow of people inside public places and facilities of interest (museums, theatres, cinemas, etc.) without collecting personal or sensitive data. Weak monitoring of people flows (i.e. monitoring without personal identification of the monitored subjects) through Internet of Things tools might be a viable solution to minimize lineups and overcrowding. Our study, which began as an experiment in the Umbria region of Italy, aims to be one of several answers to automated planning of people's flows in order to make our land more liveable. We intend to show that the Internet of Things gives almost unlimited tools and possibilities, from developing a basic information process to implementing a true portal which enables business people to connect with interested consumers.

</p>
</details>

<details><summary><b>The effect of synaptic weight initialization in feature-based successor representation learning</b>
<a href="https://arxiv.org/abs/2111.02017">arxiv:2111.02017</a>
&#x1F4C8; 0 <br>
<p>Hyunsu Lee</p></summary>
<p>

**Abstract:** After discovering place cells, the idea of the hippocampal (HPC) function to represent geometric spaces has been extended to predictions, imaginations, and conceptual cognitive maps. Recent research arguing that the HPC represents a predictive map; and it has shown that the HPC predicts visits to specific locations. This predictive map theory is based on successor representation (SR) from reinforcement learning. Feature-based SR (SF), which uses a neural network as a function approximation to learn SR, seems more plausible neurobiological model. However, it is not well known how different methods of weight (W) initialization affect SF learning.
  In this study, SF learners were exposed to simple maze environments to analyze SF learning efficiency and W patterns pattern changes. Three kinds of W initialization pattern were used: identity matrix, zero matrix, and small random matrix. The SF learner initiated with random weight matrix showed better performance than other three RL agents. We will discuss the neurobiological meaning of SF weight matrix. Through this approach, this paper tried to increase our understanding of intelligence from neuroscientific and artificial intelligence perspective.

</p>
</details>

<details><summary><b>Oblique and rotation double random forest</b>
<a href="https://arxiv.org/abs/2111.02010">arxiv:2111.02010</a>
&#x1F4C8; 0 <br>
<p>M. A. Ganaie, M. Tanveer, P. N. Suganthan, V. Snasel</p></summary>
<p>

**Abstract:** An ensemble of decision trees is known as Random Forest. As suggested by Breiman, the strength of unstable learners and the diversity among them are the ensemble models' core strength. In this paper, we propose two approaches known as oblique and rotation double random forests. In the first approach, we propose a rotation based double random forest. In rotation based double random forests, transformation or rotation of the feature space is generated at each node. At each node different random feature subspace is chosen for evaluation, hence the transformation at each node is different. Different transformations result in better diversity among the base learners and hence, better generalization performance. With the double random forest as base learner, the data at each node is transformed via two different transformations namely, principal component analysis and linear discriminant analysis. In the second approach, we propose oblique double random forest. Decision trees in random forest and double random forest are univariate, and this results in the generation of axis parallel split which fails to capture the geometric structure of the data. Also, the standard random forest may not grow sufficiently large decision trees resulting in suboptimal performance. To capture the geometric properties and to grow the decision trees of sufficient depth, we propose oblique double random forest. The oblique double random forest models are multivariate decision trees. At each non-leaf node, multisurface proximal support vector machine generates the optimal plane for better generalization performance. Also, different regularization techniques (Tikhonov regularisation and axis-parallel split regularisation) are employed for tackling the small sample size problems in the decision trees of oblique double random forest.

</p>
</details>

<details><summary><b>AI Ethics Statements -- Analysis and lessons learnt from NeurIPS Broader Impact Statements</b>
<a href="https://arxiv.org/abs/2111.01705">arxiv:2111.01705</a>
&#x1F4C8; 0 <br>
<p>Carolyn Ashurst, Emmie Hine, Paul Sedille, Alexis Carlier</p></summary>
<p>

**Abstract:** Ethics statements have been proposed as a mechanism to increase transparency and promote reflection on the societal impacts of published research. In 2020, the machine learning (ML) conference NeurIPS broke new ground by requiring that all papers include a broader impact statement. This requirement was removed in 2021, in favour of a checklist approach. The 2020 statements therefore provide a unique opportunity to learn from the broader impact experiment: to investigate the benefits and challenges of this and similar governance mechanisms, as well as providing an insight into how ML researchers think about the societal impacts of their own work. Such learning is needed as NeurIPS and other venues continue to question and adapt their policies. To enable this, we have created a dataset containing the impact statements from all NeurIPS 2020 papers, along with additional information such as affiliation type, location and subject area, and a simple visualisation tool for exploration. We also provide an initial quantitative analysis of the dataset, covering representation, engagement, common themes, and willingness to discuss potential harms alongside benefits. We investigate how these vary by geography, affiliation type and subject area. Drawing on these findings, we discuss the potential benefits and negative outcomes of ethics statement requirements, and their possible causes and associated challenges. These lead us to several lessons to be learnt from the 2020 requirement: (i) the importance of creating the right incentives, (ii) the need for clear expectations and guidance, and (iii) the importance of transparency and constructive deliberation. We encourage other researchers to use our dataset to provide additional analysis, to further our understanding of how researchers responded to this requirement, and to investigate the benefits and challenges of this and related mechanisms.

</p>
</details>

<details><summary><b>Towards an Optimal Hybrid Algorithm for EV Charging Stations Placement using Quantum Annealing and Genetic Algorithms</b>
<a href="https://arxiv.org/abs/2111.01622">arxiv:2111.01622</a>
&#x1F4C8; 0 <br>
<p>Aman Chandra, Jitesh Lalwani, Babita Jajodia</p></summary>
<p>

**Abstract:** Quantum Annealing is a heuristic for solving optimization problems that have seen a recent surge in usage owing to the success of D-Wave Systems. This paper aims to find a good heuristic for solving the Electric Vehicle Charger Placement (EVCP) problem, a problem that stands to be very important given the costs of setting up an electric vehicle (EV) charger and the expected surge in electric vehicles across the world. The same problem statement can also be generalised to the optimal placement of any entity in a grid and can be explored for further uses. Finally, the authors introduce a novel heuristic combining Quantum Annealing and Genetic Algorithms to solve the problem. The proposed hybrid approach entails seeding the genetic algorithm with the results of a quantum annealer. Our experiments show this method decreases the minimum distance from POIs by 42.89% compared to vanilla quantum annealing over our sample EVCP datasets.

</p>
</details>

<details><summary><b>Learning Size and Shape of Calabi-Yau Spaces</b>
<a href="https://arxiv.org/abs/2111.01436">arxiv:2111.01436</a>
&#x1F4C8; 0 <br>
<p>Magdalena Larfors, Andre Lukas, Fabian Ruehle, Robin Schneider</p></summary>
<p>

**Abstract:** We present a new machine learning library for computing metrics of string compactification spaces. We benchmark the performance on Monte-Carlo sampled integrals against previous numerical approximations and find that our neural networks are more sample- and computation-efficient. We are the first to provide the possibility to compute these metrics for arbitrary, user-specified shape and size parameters of the compact space and observe a linear relation between optimization of the partial differential equation we are training against and vanishing Ricci curvature.

</p>
</details>

<details><summary><b>Cascadable all-optical NAND gates using diffractive networks</b>
<a href="https://arxiv.org/abs/2111.01404">arxiv:2111.01404</a>
&#x1F4C8; 0 <br>
<p>Yi Luo, Deniz Mengu, Aydogan Ozcan</p></summary>
<p>

**Abstract:** Owing to its potential advantages such as scalability, low latency and power efficiency, optical computing has seen rapid advances over the last decades. A core unit of a potential all-optical processor would be the NAND gate, which can be cascaded to perform an arbitrary logical operation. Here, we present the design and analysis of cascadable all-optical NAND gates using diffractive neural networks. We encoded the logical values at the input and output planes of a diffractive NAND gate using the relative optical power of two spatially-separated apertures. Based on this architecture, we numerically optimized the design of a diffractive neural network composed of 4 passive layers to all-optically perform NAND operation using the diffraction of light, and cascaded these diffractive NAND gates to perform complex logical functions by successively feeding the output of one diffractive NAND gate into another. We demonstrated the cascadability of our diffractive NAND gates by using identical diffractive designs to all-optically perform AND and OR operations, as well as a half-adder. Cascadable all-optical NAND gates composed of spatially-engineered passive diffractive layers can serve as a core component of various optical computing platforms.

</p>
</details>


[Next Page](2021/2021-11/2021-11-01.md)
