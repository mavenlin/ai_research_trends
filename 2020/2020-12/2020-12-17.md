## Summary for 2020-12-17, created on 2021-01-14


<details><summary><b>Towards Understanding Ensemble, Knowledge Distillation and Self-Distillation in Deep Learning</b>
<a href="https://arxiv.org/abs/2012.09816">arxiv:2012.09816</a>
&#x1F4C8; 246 <br>
<p>Zeyuan Allen-Zhu, Yuanzhi Li</p></summary>
<p>

**Abstract:** We formally study how Ensemble of deep learning models can improve test accuracy, and how the superior performance of ensemble can be distilled into a single model using Knowledge Distillation. We consider the challenging case where the ensemble is simply an average of the outputs of a few independently trained neural networks with the SAME architecture, trained using the SAME algorithm on the SAME data set, and they only differ by the random seeds used in the initialization. We empirically show that ensemble/knowledge distillation in deep learning works very differently from traditional learning theory, especially differently from ensemble of random feature mappings or the neural-tangent-kernel feature mappings, and is potentially out of the scope of existing theorems. Thus, to properly understand ensemble and knowledge distillation in deep learning, we develop a theory showing that when data has a structure we refer to as "multi-view", then ensemble of independently trained neural networks can provably improve test accuracy, and such superior test accuracy can also be provably distilled into a single model by training a single model to match the output of the ensemble instead of the true label. Our result sheds light on how ensemble works in deep learning in a way that is completely different from traditional theorems, and how the "dark knowledge" is hidden in the outputs of the ensemble -- that can be used in knowledge distillation -- comparing to the true data labels. In the end, we prove that self-distillation can also be viewed as implicitly combining ensemble and knowledge distillation to improve test accuracy.

</p>
</details>

<details><summary><b>Computational principles of intelligence: learning and reasoning with neural networks</b>
<a href="https://arxiv.org/abs/2012.09477">arxiv:2012.09477</a>
&#x1F4C8; 189 <br>
<p>Abel Torres Montoya</p></summary>
<p>

**Abstract:** Despite significant achievements and current interest in machine learning and artificial intelligence, the quest for a theory of intelligence, allowing general and efficient problem solving, has done little progress. This work tries to contribute in this direction by proposing a novel framework of intelligence based on three principles. First, the generative and mirroring nature of learned representations of inputs. Second, a grounded, intrinsically motivated and iterative process for learning, problem solving and imagination. Third, an ad hoc tuning of the reasoning mechanism over causal compositional representations using inhibition rules. Together, those principles create a systems approach offering interpretability, continuous learning, common sense and more. This framework is being developed from the following perspectives: as a general problem solving method, as a human oriented tool and finally, as model of information processing in the brain.

</p>
</details>

<details><summary><b>Intrinsically Motivated Goal-Conditioned Reinforcement Learning: a Short Survey</b>
<a href="https://arxiv.org/abs/2012.09830">arxiv:2012.09830</a>
&#x1F4C8; 135 <br>
<p>Cédric Colas, Tristan Karch, Olivier Sigaud, Pierre-Yves Oudeyer</p></summary>
<p>

**Abstract:** Building autonomous machines that can explore open-ended environments, discover possible interactions and autonomously build repertoires of skills is a general objective of artificial intelligence. Developmental approaches argue that this can only be achieved by autonomous and intrinsically motivated learning agents that can generate, select and learn to solve their own problems. In recent years, we have seen a convergence of developmental approaches, and developmental robotics in particular, with deep reinforcement learning (RL) methods, forming the new domain of developmental machine learning. Within this new domain, we review here a set of methods where deep RL algorithms are trained to tackle the developmental robotics problem of the autonomous acquisition of open-ended repertoires of skills. Intrinsically motivated goal-conditioned RL algorithms train agents to learn to represent, generate and pursue their own goals. The self-generation of goals requires the learning of compact goal encodings as well as their associated goal-achievement functions, which results in new challenges compared to traditional RL algorithms designed to tackle pre-defined sets of goals using external reward signals. This paper proposes a typology of these methods at the intersection of deep RL and developmental approaches, surveys recent approaches and discusses future avenues.

</p>
</details>

<details><summary><b>Worldsheet: Wrapping the World in a 3D Sheet for View Synthesis from a Single Image</b>
<a href="https://arxiv.org/abs/2012.09854">arxiv:2012.09854</a>
&#x1F4C8; 134 <br>
<p>Ronghang Hu, Deepak Pathak</p></summary>
<p>

**Abstract:** We present Worldsheet, a method for novel view synthesis using just a single RGB image as input. This is a challenging problem as it requires an understanding of the 3D geometry of the scene as well as texture mapping to generate both visible and occluded regions from new view-points. Our main insight is that simply shrink-wrapping a planar mesh sheet onto the input image, consistent with the learned intermediate depth, captures underlying geometry sufficient enough to generate photorealistic unseen views with arbitrarily large view-point changes. To operationalize this, we propose a novel differentiable texture sampler that allows our wrapped mesh sheet to be textured; which is then transformed into a target image via differentiable rendering. Our approach is category-agnostic, end-to-end trainable without using any 3D supervision and requires a single image at test time. Worldsheet consistently outperforms prior state-of-the-art methods on single-image view synthesis across several datasets. Furthermore, this simple idea captures novel views surprisingly well on a wide range of high resolution in-the-wild images in converting them into a navigable 3D pop-up. Video results and code at https://worldsheet.github.io

</p>
</details>

<details><summary><b>Neural Radiance Flow for 4D View Synthesis and Video Processing</b>
<a href="https://arxiv.org/abs/2012.09790">arxiv:2012.09790</a>
&#x1F4C8; 79 <br>
<p>Yilun Du, Yinan Zhang, Hong-Xing Yu, Joshua B. Tenenbaum, Jiajun Wu</p></summary>
<p>

**Abstract:** We present a method, Neural Radiance Flow (NeRFlow),to learn a 4D spatial-temporal representation of a dynamic scene from a set of RGB images. Key to our approach is the use of a neural implicit representation that learns to capture the 3D occupancy, radiance, and dynamics of the scene. By enforcing consistency across different modalities, our representation enables multi-view rendering in diverse dynamic scenes, including water pouring, robotic interaction, and real images, outperforming state-of-the-art methods for spatial-temporal view synthesis. Our approach works even when inputs images are captured with only one camera. We further demonstrate that the learned representation can serve as an implicit scene prior, enabling video processing tasks such as image super-resolution and de-noising without any additional supervision.

</p>
</details>

<details><summary><b>Continual Lifelong Learning in Natural Language Processing: A Survey</b>
<a href="https://arxiv.org/abs/2012.09823">arxiv:2012.09823</a>
&#x1F4C8; 52 <br>
<p>Magdalena Biesialska, Katarzyna Biesialska, Marta R. Costa-jussà</p></summary>
<p>

**Abstract:** Continual learning (CL) aims to enable information systems to learn from a continuous data stream across time. However, it is difficult for existing deep learning architectures to learn a new task without largely forgetting previously acquired knowledge. Furthermore, CL is particularly challenging for language learning, as natural language is ambiguous: it is discrete, compositional, and its meaning is context-dependent. In this work, we look at the problem of CL through the lens of various NLP tasks. Our survey discusses major challenges in CL and current methods applied in neural network models. We also provide a critical review of the existing CL evaluation methods and datasets in NLP. Finally, we present our outlook on future research directions.

</p>
</details>

<details><summary><b>ViNG: Learning Open-World Navigation with Visual Goals</b>
<a href="https://arxiv.org/abs/2012.09812">arxiv:2012.09812</a>
&#x1F4C8; 52 <br>
<p>Dhruv Shah, Benjamin Eysenbach, Gregory Kahn, Nicholas Rhinehart, Sergey Levine</p></summary>
<p>

**Abstract:** We propose a learning-based navigation system for reaching visually indicated goals and demonstrate this system on a real mobile robot platform. Learning provides an appealing alternative to conventional methods for robotic navigation: instead of reasoning about environments in terms of geometry and maps, learning can enable a robot to learn about navigational affordances, understand what types of obstacles are traversable (e.g., tall grass) or not (e.g., walls), and generalize over patterns in the environment. However, unlike conventional planning algorithms, it is harder to change the goal for a learned policy during deployment. We propose a method for learning to navigate towards a goal image of the desired destination. By combining a learned policy with a topological graph constructed out of previously observed data, our system can determine how to reach this visually indicated goal even in the presence of variable appearance and lighting. Three key insights, waypoint proposal, graph pruning and negative mining, enable our method to learn to navigate in real-world environments using only offline data, a setting where prior methods struggle. We instantiate our method on a real outdoor ground robot and show that our system, which we call ViNG, outperforms previously-proposed methods for goal-conditioned reinforcement learning, including other methods that incorporate reinforcement learning and search. We also study how ViNG generalizes to unseen environments and evaluate its ability to adapt to such an environment with growing experience. Finally, we demonstrate ViNG on a number of real-world applications, such as last-mile delivery and warehouse inspection. We encourage the reader to check out the videos of our experiments and demonstrations at our project website https://sites.google.com/view/ving-robot

</p>
</details>

<details><summary><b>Toward Transformer-Based Object Detection</b>
<a href="https://arxiv.org/abs/2012.09958">arxiv:2012.09958</a>
&#x1F4C8; 26 <br>
<p>Josh Beal, Eric Kim, Eric Tzeng, Dong Huk Park, Andrew Zhai, Dmitry Kislyuk</p></summary>
<p>

**Abstract:** Transformers have become the dominant model in natural language processing, owing to their ability to pretrain on massive amounts of data, then transfer to smaller, more specific tasks via fine-tuning. The Vision Transformer was the first major attempt to apply a pure transformer model directly to images as input, demonstrating that as compared to convolutional networks, transformer-based architectures can achieve competitive results on benchmark classification tasks. However, the computational complexity of the attention operator means that we are limited to low-resolution inputs. For more complex tasks such as detection or segmentation, maintaining a high input resolution is crucial to ensure that models can properly identify and reflect fine details in their output. This naturally raises the question of whether or not transformer-based architectures such as the Vision Transformer are capable of performing tasks other than classification. In this paper, we determine that Vision Transformers can be used as a backbone by a common detection task head to produce competitive COCO results. The model that we propose, ViT-FRCNN, demonstrates several known properties associated with transformers, including large pretraining capacity and fast fine-tuning performance. We also investigate improvements over a standard detection backbone, including superior performance on out-of-domain images, better performance on large objects, and a lessened reliance on non-maximum suppression. We view ViT-FRCNN as an important stepping stone toward a pure-transformer solution of complex vision tasks such as object detection.

</p>
</details>

<details><summary><b>Deep Molecular Dreaming: Inverse machine learning for de-novo molecular design and interpretability with surjective representations</b>
<a href="https://arxiv.org/abs/2012.09712">arxiv:2012.09712</a>
&#x1F4C8; 26 <br>
<p>Cynthia Shen, Mario Krenn, Sagi Eppel, Alan Aspuru-Guzik</p></summary>
<p>

**Abstract:** Computer-based de-novo design of functional molecules is one of the most prominent challenges in cheminformatics today. As a result, generative and evolutionary inverse designs from the field of artificial intelligence have emerged at a rapid pace, with aims to optimize molecules for a particular chemical property. These models 'indirectly' explore the chemical space; by learning latent spaces, policies, distributions or by applying mutations on populations of molecules. However, the recent development of the SELFIES string representation of molecules, a surjective alternative to SMILES, have made possible other potential techniques. Based on SELFIES, we therefore propose PASITHEA, a direct gradient-based molecule optimization that applies inceptionism techniques from computer vision. PASITHEA exploits the use of gradients by directly reversing the learning process of a neural network, which is trained to predict real-valued chemical properties. Effectively, this forms an inverse regression model, which is capable of generating molecular variants optimized for a certain property. Although our results are preliminary, we observe a shift in distribution of a chosen property during inverse-training, a clear indication of PASITHEA's viability. A striking property of inceptionism is that we can directly probe the model's understanding of the chemical space it was trained on. We expect that extending PASITHEA to larger datasets, molecules and more complex properties will lead to advances in the design of new functional molecules as well as the interpretation and explanation of machine learning models.

</p>
</details>

<details><summary><b>Research Reproducibility as a Survival Analysis</b>
<a href="https://arxiv.org/abs/2012.09932">arxiv:2012.09932</a>
&#x1F4C8; 20 <br>
<p>Edward Raff</p></summary>
<p>

**Abstract:** There has been increasing concern within the machine learning community that we are in a reproducibility crisis. As many have begun to work on this problem, all work we are aware of treat the issue of reproducibility as an intrinsic binary property: a paper is or is not reproducible. Instead, we consider modeling the reproducibility of a paper as a survival analysis problem. We argue that this perspective represents a more accurate model of the underlying meta-science question of reproducible research, and we show how a survival analysis allows us to draw new insights that better explain prior longitudinal data. The data and code can be found at https://github.com/EdwardRaff/Research-Reproducibility-Survival-Analysis

</p>
</details>

<details><summary><b>Content Masked Loss: Human-Like Brush Stroke Planning in a Reinforcement Learning Painting Agent</b>
<a href="https://arxiv.org/abs/2012.10043">arxiv:2012.10043</a>
&#x1F4C8; 9 <br>
<p>Peter Schaldenbrand, Jean Oh</p></summary>
<p>

**Abstract:** The objective of most Reinforcement Learning painting agents is to minimize the loss between a target image and the paint canvas. Human painter artistry emphasizes important features of the target image rather than simply reproducing it (DiPaola 2007). Using adversarial or L2 losses in the RL painting models, although its final output is generally a work of finesse, produces a stroke sequence that is vastly different from that which a human would produce since the model does not have knowledge about the abstract features in the target image. In order to increase the human-like planning of the model without the use of expensive human data, we introduce a new loss function for use with the model's reward function: Content Masked Loss. In the context of robot painting, Content Masked Loss employs an object detection model to extract features which are used to assign higher weight to regions of the canvas that a human would find important for recognizing content. The results, based on 332 human evaluators, show that the digital paintings produced by our Content Masked model show detectable subject matter earlier in the stroke sequence than existing methods without compromising on the quality of the final painting.

</p>
</details>

<details><summary><b>High-Throughput Synchronous Deep RL</b>
<a href="https://arxiv.org/abs/2012.09849">arxiv:2012.09849</a>
&#x1F4C8; 9 <br>
<p>Iou-Jen Liu, Raymond A. Yeh, Alexander G. Schwing</p></summary>
<p>

**Abstract:** Deep reinforcement learning (RL) is computationally demanding and requires processing of many data points. Synchronous methods enjoy training stability while having lower data throughput. In contrast, asynchronous methods achieve high throughput but suffer from stability issues and lower sample efficiency due to `stale policies.' To combine the advantages of both methods we propose High-Throughput Synchronous Deep Reinforcement Learning (HTS-RL). In HTS-RL, we perform learning and rollouts concurrently, devise a system design which avoids `stale policies' and ensure that actors interact with environment replicas in an asynchronous manner while maintaining full determinism. We evaluate our approach on Atari games and the Google Research Football environment. Compared to synchronous baselines, HTS-RL is 2-6$\times$ faster. Compared to state-of-the-art asynchronous methods, HTS-RL has competitive throughput and consistently achieves higher average episode rewards.

</p>
</details>

<details><summary><b>Task Uncertainty Loss Reduce Negative Transfer in Asymmetric Multi-task Feature Learning</b>
<a href="https://arxiv.org/abs/2012.09575">arxiv:2012.09575</a>
&#x1F4C8; 9 <br>
<p>Rafael Peres da Silva, Chayaporn Suphavilai, Niranjan Nagarajan</p></summary>
<p>

**Abstract:** Multi-task learning (MTL) is frequently used in settings where a target task has to be learnt based on limited training data, but knowledge can be leveraged from related auxiliary tasks. While MTL can improve task performance overall relative to single-task learning (STL), these improvements can hide negative transfer (NT), where STL may deliver better performance for many individual tasks. Asymmetric multitask feature learning (AMTFL) is an approach that tries to address this by allowing tasks with higher loss values to have smaller influence on feature representations for learning other tasks. Task loss values do not necessarily indicate reliability of models for a specific task. We present examples of NT in two orthogonal datasets (image recognition and pharmacogenomics) and tackle this challenge by using aleatoric homoscedastic uncertainty to capture the relative confidence between tasks, and set weights for task loss. Our results show that this approach reduces NT providing a new approach to enable robust MTL.

</p>
</details>

<details><summary><b>Learning Cross-Domain Correspondence for Control with Dynamics Cycle-Consistency</b>
<a href="https://arxiv.org/abs/2012.09811">arxiv:2012.09811</a>
&#x1F4C8; 8 <br>
<p>Qiang Zhang, Tete Xiao, Alexei A. Efros, Lerrel Pinto, Xiaolong Wang</p></summary>
<p>

**Abstract:** At the heart of many robotics problems is the challenge of learning correspondences across domains. For instance, imitation learning requires obtaining correspondence between humans and robots; sim-to-real requires correspondence between physics simulators and the real world; transfer learning requires correspondences between different robotics environments. This paper aims to learn correspondence across domains differing in representation (vision vs. internal state), physics parameters (mass and friction), and morphology (number of limbs). Importantly, correspondences are learned using unpaired and randomly collected data from the two domains. We propose \textit{dynamics cycles} that align dynamic robot behavior across two domains using a cycle-consistency constraint. Once this correspondence is found, we can directly transfer the policy trained on one domain to the other, without needing any additional fine-tuning on the second domain. We perform experiments across a variety of problem domains, both in simulation and on real robot. Our framework is able to align uncalibrated monocular video of a real robot arm to dynamic state-action trajectories of a simulated arm without paired data. Video demonstrations of our results are available at: https://sjtuzq.github.io/cycle_dynamics.html .

</p>
</details>

<details><summary><b>A new semi-supervised self-training method for lung cancer prediction</b>
<a href="https://arxiv.org/abs/2012.09472">arxiv:2012.09472</a>
&#x1F4C8; 7 <br>
<p>Kelvin Shak, Mundher Al-Shabi, Andrea Liew, Boon Leong Lan, Wai Yee Chan, Kwan Hoong Ng, Maxine Tan</p></summary>
<p>

**Abstract:** Background and Objective: Early detection of lung cancer is crucial as it has high mortality rate with patients commonly present with the disease at stage 3 and above. There are only relatively few methods that simultaneously detect and classify nodules from computed tomography (CT) scans. Furthermore, very few studies have used semi-supervised learning for lung cancer prediction. This study presents a complete end-to-end scheme to detect and classify lung nodules using the state-of-the-art Self-training with Noisy Student method on a comprehensive CT lung screening dataset of around 4,000 CT scans.
  Methods: We used three datasets, namely LUNA16, LIDC and NLST, for this study. We first utilise a three-dimensional deep convolutional neural network model to detect lung nodules in the detection stage. The classification model known as Maxout Local-Global Network uses non-local networks to detect global features including shape features, residual blocks to detect local features including nodule texture, and a Maxout layer to detect nodule variations. We trained the first Self-training with Noisy Student model to predict lung cancer on the unlabelled NLST datasets. Then, we performed Mixup regularization to enhance our scheme and provide robustness to erroneous labels.
  Results and Conclusions: Our new Mixup Maxout Local-Global network achieves an AUC of 0.87 on 2,005 completely independent testing scans from the NLST dataset. Our new scheme significantly outperformed the next highest performing method at the 5% significance level using DeLong's test (p = 0.0001). This study presents a new complete end-to-end scheme to predict lung cancer using Self-training with Noisy Student combined with Mixup regularization. On a completely independent dataset of 2,005 scans, we achieved state-of-the-art performance even with more images as compared to other methods.

</p>
</details>

<details><summary><b>Curiosity in exploring chemical space: Intrinsic rewards for deep molecular reinforcement learning</b>
<a href="https://arxiv.org/abs/2012.11293">arxiv:2012.11293</a>
&#x1F4C8; 6 <br>
<p>Luca A. Thiede, Mario Krenn, AkshatKumar Nigam, Alan Aspuru-Guzik</p></summary>
<p>

**Abstract:** Computer-aided design of molecules has the potential to disrupt the field of drug and material discovery. Machine learning, and deep learning, in particular, have been topics where the field has been developing at a rapid pace. Reinforcement learning is a particularly promising approach since it allows for molecular design without prior knowledge. However, the search space is vast and efficient exploration is desirable when using reinforcement learning agents. In this study, we propose an algorithm to aid efficient exploration. The algorithm is inspired by a concept known in the literature as curiosity. We show on three benchmarks that a curious agent finds better performing molecules. This indicates an exciting new research direction for reinforcement learning agents that can explore the chemical space out of their own motivation. This has the potential to eventually lead to unexpected new molecules that no human has thought about so far.

</p>
</details>

<details><summary><b>On the eigenvector bias of Fourier feature networks: From regression to solving multi-scale PDEs with physics-informed neural networks</b>
<a href="https://arxiv.org/abs/2012.10047">arxiv:2012.10047</a>
&#x1F4C8; 6 <br>
<p>Sifan Wang, Hanwen Wang, Paris Perdikaris</p></summary>
<p>

**Abstract:** Physics-informed neural networks (PINNs) are demonstrating remarkable promise in integrating physical models with gappy and noisy observational data, but they still struggle in cases where the target functions to be approximated exhibit high-frequency or multi-scale features. In this work we investigate this limitation through the lens of Neural Tangent Kernel (NTK) theory and elucidate how PINNs are biased towards learning functions along the dominant eigen-directions of their limiting NTK. Using this observation, we construct novel architectures that employ spatio-temporal and multi-scale random Fourier features, and justify how such coordinate embedding layers can lead to robust and accurate PINN models. Numerical examples are presented for several challenging cases where conventional PINN models fail, including wave propagation and reaction-diffusion dynamics, illustrating how the proposed methods can be used to effectively tackle both forward and inverse problems involving partial differential equations with multi-scale behavior. All code an data accompanying this manuscript will be made publicly available at \url{https://github.com/PredictiveIntelligenceLab/MultiscalePINNs}.

</p>
</details>

<details><summary><b>SpAtten: Efficient Sparse Attention Architecture with Cascade Token and Head Pruning</b>
<a href="https://arxiv.org/abs/2012.09852">arxiv:2012.09852</a>
&#x1F4C8; 6 <br>
<p>Hanrui Wang, Zhekai Zhang, Song Han</p></summary>
<p>

**Abstract:** The attention mechanism is becoming increasingly popular in Natural Language Processing (NLP) applications, showing superior performance than convolutional and recurrent architectures. However, general-purpose platforms such as CPUs and GPUs are inefficient when performing attention inference due to complicated data movement and low arithmetic intensity. Moreover, existing NN accelerators mainly focus on optimizing convolutional or recurrent models, and cannot efficiently support attention. In this paper, we present SpAtten, an efficient algorithm-architecture co-design that leverages token sparsity, head sparsity, and quantization opportunities to reduce the attention computation and memory access. Inspired by the high redundancy of human languages, we propose the novel cascade token pruning to prune away unimportant tokens in the sentence. We also propose cascade head pruning to remove unessential heads. Cascade pruning is fundamentally different from weight pruning since there is no trainable weight in the attention mechanism, and the pruned tokens and heads are selected on the fly. To efficiently support them on hardware, we design a novel top-k engine to rank token and head importance scores with high throughput. Furthermore, we propose progressive quantization that first fetches MSBs only and performs the computation; if the confidence is low, it fetches LSBs and recomputes the attention outputs, trading computation for memory reduction.
  Extensive experiments on 30 benchmarks show that, on average, SpAtten reduces DRAM access by 10.0x with no accuracy loss, and achieves 1.6x, 3.0x, 162x, 347x speedup, and 1,4x, 3.2x, 1193x, 4059x energy savings over A3 accelerator, MNNFast accelerator, TITAN Xp GPU, Xeon CPU, respectively.

</p>
</details>

<details><summary><b>Towards Resolving the Implicit Bias of Gradient Descent for Matrix Factorization: Greedy Low-Rank Learning</b>
<a href="https://arxiv.org/abs/2012.09839">arxiv:2012.09839</a>
&#x1F4C8; 6 <br>
<p>Zhiyuan Li, Yuping Luo, Kaifeng Lyu</p></summary>
<p>

**Abstract:** Matrix factorization is a simple and natural test-bed to investigate the implicit regularization of gradient descent. Gunasekar et al. (2018) conjectured that Gradient Flow with infinitesimal initialization converges to the solution that minimizes the nuclear norm, but a series of recent papers argued that the language of norm minimization is not sufficient to give a full characterization for the implicit regularization. In this work, we provide theoretical and empirical evidence that for depth-2 matrix factorization, gradient flow with infinitesimal initialization is mathematically equivalent to a simple heuristic rank minimization algorithm, Greedy Low-Rank Learning, under some reasonable assumptions. This generalizes the rank minimization view from previous works to a much broader setting and enables us to construct counter-examples to refute the conjecture from Gunasekar et al. (2018). We also extend the results to the case where depth $\ge 3$, and we show that the benefit of being deeper is that the above convergence has a much weaker dependence over initialization magnitude so that this rank minimization is more likely to take effect for initialization with practical scale.

</p>
</details>

<details><summary><b>Rank-One Measurements of Low-Rank PSD Matrices Have Small Feasible Sets</b>
<a href="https://arxiv.org/abs/2012.09768">arxiv:2012.09768</a>
&#x1F4C8; 6 <br>
<p>T. Mitchell Roddenberry, Santiago Segarra, Anastasios Kyrillidis</p></summary>
<p>

**Abstract:** We study the role of the constraint set in determining the solution to low-rank, positive semidefinite (PSD) matrix sensing problems. The setting we consider involves rank-one sensing matrices: In particular, given a set of rank-one projections of an approximately low-rank PSD matrix, we characterize the radius of the set of PSD matrices that satisfy the measurements. This result yields a sampling rate to guarantee singleton solution sets when the true matrix is exactly low-rank, such that the choice of the objective function or the algorithm to be used is inconsequential in its recovery. We discuss applications of this contribution and compare it to recent literature regarding implicit regularization for similar problems. We demonstrate practical implications of this result by applying conic projection methods for PSD matrix recovery without incorporating low-rank regularization.

</p>
</details>

<details><summary><b>Detection and Prediction of Nutrient Deficiency Stress using Longitudinal Aerial Imagery</b>
<a href="https://arxiv.org/abs/2012.09654">arxiv:2012.09654</a>
&#x1F4C8; 6 <br>
<p>Saba Dadsetan, Gisele Rose, Naira Hovakimyan, Jennifer Hobbs</p></summary>
<p>

**Abstract:** Early, precise detection of nutrient deficiency stress (NDS) has key economic as well as environmental impact; precision application of chemicals in place of blanket application reduces operational costs for the growers while reducing the amount of chemicals which may enter the environment unnecessarily. Furthermore, earlier treatment reduces the amount of loss and therefore boosts crop production during a given season. With this in mind, we collect sequences of high-resolution aerial imagery and construct semantic segmentation models to detect and predict NDS across the field. Our work sits at the intersection of agriculture, remote sensing, and modern computer vision and deep learning. First, we establish a baseline for full-field detection of NDS and quantify the impact of pretraining, backbone architecture, input representation, and sampling strategy. We then quantify the amount of information available at different points in the season by building a single-timestamp model based on a UNet. Next, we construct our proposed spatiotemporal architecture, which combines a UNet with a convolutional LSTM layer, to accurately detect regions of the field showing NDS; this approach has an impressive IOU score of 0.53. Finally, we show that this architecture can be trained to predict regions of the field which are expected to show NDS in a later flight -- potentially more than three weeks in the future -- maintaining an IOU score of 0.47-0.51 depending on how far in advance the prediction is made. We will also release a dataset which we believe will benefit the computer vision, remote sensing, as well as agriculture fields. This work contributes to the recent developments in deep learning for remote sensing and agriculture, while addressing a key social challenge with implications for economics and sustainability.

</p>
</details>

<details><summary><b>On the experimental feasibility of quantum state reconstruction via machine learning</b>
<a href="https://arxiv.org/abs/2012.09432">arxiv:2012.09432</a>
&#x1F4C8; 6 <br>
<p>Sanjaya Lohani, Thomas A. Searles, Brian T. Kirby, Ryan T. Glasser</p></summary>
<p>

**Abstract:** We determine the resource scaling of machine learning-based quantum state reconstruction methods, in terms of both inference and training, for systems of up to four qubits. Further, we examine system performance in the low-count regime, likely to be encountered in the tomography of high-dimensional systems. Finally, we implement our quantum state reconstruction method on a IBM Q quantum computer and confirm our results.

</p>
</details>

<details><summary><b>DenseHMM: Learning Hidden Markov Models by Learning Dense Representations</b>
<a href="https://arxiv.org/abs/2012.09783">arxiv:2012.09783</a>
&#x1F4C8; 5 <br>
<p>Joachim Sicking, Maximilian Pintz, Maram Akila, Tim Wirtz</p></summary>
<p>

**Abstract:** We propose DenseHMM - a modification of Hidden Markov Models (HMMs) that allows to learn dense representations of both the hidden states and the observables. Compared to the standard HMM, transition probabilities are not atomic but composed of these representations via kernelization. Our approach enables constraint-free and gradient-based optimization. We propose two optimization schemes that make use of this: a modification of the Baum-Welch algorithm and a direct co-occurrence optimization. The latter one is highly scalable and comes empirically without loss of performance compared to standard HMMs. We show that the non-linearity of the kernelization is crucial for the expressiveness of the representations. The properties of the DenseHMM like learned co-occurrences and log-likelihoods are studied empirically on synthetic and biomedical datasets.

</p>
</details>

<details><summary><b>End-to-end Deep Object Tracking with Circular Loss Function for Rotated Bounding Box</b>
<a href="https://arxiv.org/abs/2012.09771">arxiv:2012.09771</a>
&#x1F4C8; 5 <br>
<p>Vladislav Belyaev, Aleksandra Malysheva, Aleksei Shpilman</p></summary>
<p>

**Abstract:** The task object tracking is vital in numerous applications such as autonomous driving, intelligent surveillance, robotics, etc. This task entails the assigning of a bounding box to an object in a video stream, given only the bounding box for that object on the first frame. In 2015, a new type of video object tracking (VOT) dataset was created that introduced rotated bounding boxes as an extension of axis-aligned ones. In this work, we introduce a novel end-to-end deep learning method based on the Transformer Multi-Head Attention architecture. We also present a new type of loss function, which takes into account the bounding box overlap and orientation.
  Our Deep Object Tracking model with Circular Loss Function (DOTCL) shows an considerable improvement in terms of robustness over current state-of-the-art end-to-end deep learning models. It also outperforms state-of-the-art object tracking methods on VOT2018 dataset in terms of expected average overlap (EAO) metric.

</p>
</details>

<details><summary><b>Hardness of Learning Halfspaces with Massart Noise</b>
<a href="https://arxiv.org/abs/2012.09720">arxiv:2012.09720</a>
&#x1F4C8; 5 <br>
<p>Ilias Diakonikolas, Daniel M. Kane</p></summary>
<p>

**Abstract:** We study the complexity of PAC learning halfspaces in the presence of Massart (bounded) noise. Specifically, given labeled examples $(x, y)$ from a distribution $D$ on $\mathbb{R}^{n} \times \{ \pm 1\}$ such that the marginal distribution on $x$ is arbitrary and the labels are generated by an unknown halfspace corrupted with Massart noise at rate $η<1/2$, we want to compute a hypothesis with small misclassification error. Characterizing the efficient learnability of halfspaces in the Massart model has remained a longstanding open problem in learning theory.
  Recent work gave a polynomial-time learning algorithm for this problem with error $η+ε$. This error upper bound can be far from the information-theoretically optimal bound of $\mathrm{OPT}+ε$. More recent work showed that {\em exact learning}, i.e., achieving error $\mathrm{OPT}+ε$, is hard in the Statistical Query (SQ) model. In this work, we show that there is an exponential gap between the information-theoretically optimal error and the best error that can be achieved by a polynomial-time SQ algorithm. In particular, our lower bound implies that no efficient SQ algorithm can approximate the optimal error within any polynomial factor.

</p>
</details>

<details><summary><b>Image-Based Jet Analysis</b>
<a href="https://arxiv.org/abs/2012.09719">arxiv:2012.09719</a>
&#x1F4C8; 5 <br>
<p>Michael Kagan</p></summary>
<p>

**Abstract:** Image-based jet analysis is built upon the jet image representation of jets that enables a direct connection between high energy physics and the fields of computer vision and deep learning. Through this connection, a wide array of new jet analysis techniques have emerged. In this text, we survey jet image based classification models, built primarily on the use of convolutional neural networks, examine the methods to understand what these models have learned and what is their sensitivity to uncertainties, and review the recent successes in moving these models from phenomenological studies to real world application on experiments at the LHC. Beyond jet classification, several other applications of jet image based techniques, including energy estimation, pileup noise reduction, data generation, and anomaly detection, are discussed.

</p>
</details>

<details><summary><b>Attention-based Image Upsampling</b>
<a href="https://arxiv.org/abs/2012.09904">arxiv:2012.09904</a>
&#x1F4C8; 4 <br>
<p>Souvik Kundu, Hesham Mostafa, Sharath Nittur Sridhar, Sairam Sundaresan</p></summary>
<p>

**Abstract:** Convolutional layers are an integral part of many deep neural network solutions in computer vision. Recent work shows that replacing the standard convolution operation with mechanisms based on self-attention leads to improved performance on image classification and object detection tasks. In this work, we show how attention mechanisms can be used to replace another canonical operation: strided transposed convolution. We term our novel attention-based operation attention-based upsampling since it increases/upsamples the spatial dimensions of the feature maps. Through experiments on single image super-resolution and joint-image upsampling tasks, we show that attention-based upsampling consistently outperforms traditional upsampling methods based on strided transposed convolution or based on adaptive filters while using fewer parameters. We show that the inherent flexibility of the attention mechanism, which allows it to use separate sources for calculating the attention coefficients and the attention targets, makes attention-based upsampling a natural choice when fusing information from multiple image modalities.

</p>
</details>

<details><summary><b>Describing the Structural Phenotype of the Glaucomatous Optic Nerve Head Using Artificial Intelligence</b>
<a href="https://arxiv.org/abs/2012.09755">arxiv:2012.09755</a>
&#x1F4C8; 4 <br>
<p>Satish K. Panda, Haris Cheong, Tin A. Tun, Sripad K. Devella, Ramaswami Krishnadas, Martin L. Buist, Shamira Perera, Ching-Yu Cheng, Tin Aung, Alexandre H. Thiéry, Michaël J. A. Girard</p></summary>
<p>

**Abstract:** The optic nerve head (ONH) typically experiences complex neural- and connective-tissue structural changes with the development and progression of glaucoma, and monitoring these changes could be critical for improved diagnosis and prognosis in the glaucoma clinic. The gold-standard technique to assess structural changes of the ONH clinically is optical coherence tomography (OCT). However, OCT is limited to the measurement of a few hand-engineered parameters, such as the thickness of the retinal nerve fiber layer (RNFL), and has not yet been qualified as a stand-alone device for glaucoma diagnosis and prognosis applications. We argue this is because the vast amount of information available in a 3D OCT scan of the ONH has not been fully exploited. In this study we propose a deep learning approach that can: \textbf{(1)} fully exploit information from an OCT scan of the ONH; \textbf{(2)} describe the structural phenotype of the glaucomatous ONH; and that can \textbf{(3)} be used as a robust glaucoma diagnosis tool. Specifically, the structural features identified by our algorithm were found to be related to clinical observations of glaucoma. The diagnostic accuracy from these structural features was $92.0 \pm 2.3 \%$ with a sensitivity of $90.0 \pm 2.4 \% $ (at $95 \%$ specificity). By changing their magnitudes in steps, we were able to reveal how the morphology of the ONH changes as one transitions from a `non-glaucoma' to a `glaucoma' condition. We believe our work may have strong clinical implication for our understanding of glaucoma pathogenesis, and could be improved in the future to also predict future loss of vision.

</p>
</details>

<details><summary><b>Model-free and Bayesian Ensembling Model-based Deep Reinforcement Learning for Particle Accelerator Control Demonstrated on the FERMI FEL</b>
<a href="https://arxiv.org/abs/2012.09737">arxiv:2012.09737</a>
&#x1F4C8; 4 <br>
<p>Simon Hirlaender, Niky Bruchon</p></summary>
<p>

**Abstract:** Reinforcement learning holds tremendous promise in accelerator controls. The primary goal of this paper is to show how this approach can be utilised on an operational level on accelerator physics problems. Despite the success of model-free reinforcement learning in several domains, sample-efficiency still is a bottle-neck, which might be encompassed by model-based methods. We compare well-suited purely model-based to model-free reinforcement learning applied to the intensity optimisation on the FERMI FEL system. We find that the model-based approach demonstrates higher representational power and sample-efficiency, while the asymptotic performance of the model-free method is slightly superior. The model-based algorithm is implemented in a DYNA-style using an uncertainty aware model, and the model-free algorithm is based on tailored deep Q-learning. In both cases, the algorithms were implemented in a way, which presents increased noise robustness as omnipresent in accelerator control problems. Code is released in https://github.com/MathPhysSim/FERMI_RL_Paper.

</p>
</details>

<details><summary><b>SRoll3: A neural network approach to reduce large-scale systematic effects in the Planck High Frequency Instrument maps</b>
<a href="https://arxiv.org/abs/2012.09702">arxiv:2012.09702</a>
&#x1F4C8; 4 <br>
<p>Manuel López-Radcenco, Jean-Marc Delouis, Laurent Vibert</p></summary>
<p>

**Abstract:** In the present work, we propose a neural network based data inversion approach to reduce structured contamination sources, with a particular focus on the mapmaking for Planck High Frequency Instrument (Planck-HFI) data and the removal of large-scale systematic effects within the produced sky maps. The removal of contamination sources is rendered possible by the structured nature of these sources, which is characterized by local spatiotemporal interactions producing couplings between different spatiotemporal scales. We focus on exploring neural networks as a means of exploiting these couplings to learn optimal low-dimensional representations, optimized with respect to the contamination source removal and mapmaking objectives, to achieve robust and effective data inversion. We develop multiple variants of the proposed approach, and consider the inclusion of physics informed constraints and transfer learning techniques. Additionally, we focus on exploiting data augmentation techniques to integrate expert knowledge into an otherwise unsupervised network training approach. We validate the proposed method on Planck-HFI 545 GHz Far Side Lobe simulation data, considering ideal and non-ideal cases involving partial, gap-filled and inconsistent datasets, and demonstrate the potential of the neural network based dimensionality reduction to accurately model and remove large-scale systematic effects. We also present an application to real Planck-HFI 857 GHz data, which illustrates the relevance of the proposed method to accurately model and capture structured contamination sources, with reported gains of up to one order of magnitude in terms of contamination removal performance. Importantly, the methods developed in this work are to be integrated in a new version of the SRoll algorithm (SRoll3), and we describe here SRoll3 857 GHz detector maps that will be released to the community.

</p>
</details>

<details><summary><b>Polynomial-Time Algorithms for Counting and Sampling Markov Equivalent DAGs</b>
<a href="https://arxiv.org/abs/2012.09679">arxiv:2012.09679</a>
&#x1F4C8; 4 <br>
<p>Marcel Wienöbst, Max Bannach, Maciej Liśkiewicz</p></summary>
<p>

**Abstract:** Counting and uniform sampling of directed acyclic graphs (DAGs) from a Markov equivalence class are fundamental tasks in graphical causal analysis. In this paper, we show that these tasks can be performed in polynomial time, solving a long-standing open problem in this area. Our algorithms are effective and easily implementable. Experimental results show that the algorithms significantly outperform state-of-the-art methods.

</p>
</details>

<details><summary><b>cif-based collaborative decoding for end-to-end contextual speech recognition</b>
<a href="https://arxiv.org/abs/2012.09466">arxiv:2012.09466</a>
&#x1F4C8; 4 <br>
<p>Minglun Han, Linhao Dong, Shiyu Zhou, Bo Xu</p></summary>
<p>

**Abstract:** End-to-end (E2E) models have achieved promising results on multiple speech recognition benchmarks, and shown the potential to become the mainstream. However, the unified structure and the E2E training hamper injecting contextual information into them for contextual biasing. Though contextual LAS (CLAS) gives an excellent all-neural solution, the degree of biasing to given context information is not explicitly controllable. In this paper, we focus on incorporating context information into the continuous integrate-and-fire (CIF) based model that supports contextual biasing in a more controllable fashion. Specifically, an extra context processing network is introduced to extract contextual embeddings, integrate acoustically relevant context information and decode the contextual output distribution, thus forming a collaborative decoding with the decoder of the CIF-based model. Evaluated on the named entity rich evaluation sets of HKUST/AISHELL-2, our method brings relative character error rate (CER) reduction of 8.83%/21.13% and relative named entity character error rate (NE-CER) reduction of 40.14%/51.50% when compared with a strong baseline. Besides, it keeps the performance on original evaluation set without degradation.

</p>
</details>

<details><summary><b>Unsupervised Learning of Discourse Structures using a Tree Autoencoder</b>
<a href="https://arxiv.org/abs/2012.09446">arxiv:2012.09446</a>
&#x1F4C8; 4 <br>
<p>Patrick Huber, Giuseppe Carenini</p></summary>
<p>

**Abstract:** Discourse information, as postulated by popular discourse theories, such as RST and PDTB, has been shown to improve an increasing number of downstream NLP tasks, showing positive effects and synergies of discourse with important real-world applications. While methods for incorporating discourse become more and more sophisticated, the growing need for robust and general discourse structures has not been sufficiently met by current discourse parsers, usually trained on small scale datasets in a strictly limited number of domains. This makes the prediction for arbitrary tasks noisy and unreliable. The overall resulting lack of high-quality, high-quantity discourse trees poses a severe limitation to further progress. In order the alleviate this shortcoming, we propose a new strategy to generate tree structures in a task-agnostic, unsupervised fashion by extending a latent tree induction framework with an auto-encoding objective. The proposed approach can be applied to any tree-structured objective, such as syntactic parsing, discourse parsing and others. However, due to the especially difficult annotation process to generate discourse trees, we initially develop a method to generate larger and more diverse discourse treebanks. In this paper we are inferring general tree structures of natural text in multiple domains, showing promising results on a diverse set of tasks.

</p>
</details>

<details><summary><b>Maximum Entropy competes with Maximum Likelihood</b>
<a href="https://arxiv.org/abs/2012.09430">arxiv:2012.09430</a>
&#x1F4C8; 4 <br>
<p>A. E. Allahverdyan, N. H. Martirosyan</p></summary>
<p>

**Abstract:** Maximum entropy (MAXENT) method has a large number of applications in theoretical and applied machine learning, since it provides a convenient non-parametric tool for estimating unknown probabilities. The method is a major contribution of statistical physics to probabilistic inference. However, a systematic approach towards its validity limits is currently missing. Here we study MAXENT in a Bayesian decision theory set-up, i.e. assuming that there exists a well-defined prior Dirichlet density for unknown probabilities, and that the average Kullback-Leibler (KL) distance can be employed for deciding on the quality and applicability of various estimators. These allow to evaluate the relevance of various MAXENT constraints, check its general applicability, and compare MAXENT with estimators having various degrees of dependence on the prior, viz. the regularized maximum likelihood (ML) and the Bayesian estimators. We show that MAXENT applies in sparse data regimes, but needs specific types of prior information. In particular, MAXENT can outperform the optimally regularized ML provided that there are prior rank correlations between the estimated random quantity and its probabilities.

</p>
</details>

<details><summary><b>Learning Fair Policies in Decentralized Cooperative Multi-Agent Reinforcement Learning</b>
<a href="https://arxiv.org/abs/2012.09421">arxiv:2012.09421</a>
&#x1F4C8; 4 <br>
<p>Matthieu Zimmer, Umer Siddique, Paul Weng</p></summary>
<p>

**Abstract:** We consider the problem of learning fair policies in (deep) cooperative multi-agent reinforcement learning (MARL). We formalize it in a principled way as the problem of optimizing a welfare function that explicitly encodes two important aspects of fairness: efficiency and equity. As a solution method, we propose a novel neural network architecture, which is composed of two sub-networks specifically designed for taking into account the two aspects of fairness. In experiments, we demonstrate the importance of the two sub-networks for fair optimization. Our overall approach is general as it can accommodate any (sub)differentiable welfare function. Therefore, it is compatible with various notions of fairness that have been proposed in the literature (e.g., lexicographic maximin, generalized Gini social welfare function, proportional fairness). Our solution method is generic and can be implemented in various MARL settings: centralized training and decentralized execution, or fully decentralized. Finally, we experimentally validate our approach in various domains and show that it can perform much better than previous methods.

</p>
</details>

<details><summary><b>Temporal LiDAR Frame Prediction for Autonomous Driving</b>
<a href="https://arxiv.org/abs/2012.09409">arxiv:2012.09409</a>
&#x1F4C8; 4 <br>
<p>David Deng, Avideh Zakhor</p></summary>
<p>

**Abstract:** Anticipating the future in a dynamic scene is critical for many fields such as autonomous driving and robotics. In this paper we propose a class of novel neural network architectures to predict future LiDAR frames given previous ones. Since the ground truth in this application is simply the next frame in the sequence, we can train our models in a self-supervised fashion. Our proposed architectures are based on FlowNet3D and Dynamic Graph CNN. We use Chamfer Distance (CD) and Earth Mover's Distance (EMD) as loss functions and evaluation metrics. We train and evaluate our models using the newly released nuScenes dataset, and characterize their performance and complexity with several baselines. Compared to directly using FlowNet3D, our proposed architectures achieve CD and EMD nearly an order of magnitude lower. In addition, we show that our predictions generate reasonable scene flow approximations without using any labelled supervision.

</p>
</details>

<details><summary><b>Autoregressive Reasoning over Chains of Facts with Transformers</b>
<a href="https://arxiv.org/abs/2012.11321">arxiv:2012.11321</a>
&#x1F4C8; 3 <br>
<p>Ruben Cartuyvels, Graham Spinks, Marie-Francine Moens</p></summary>
<p>

**Abstract:** This paper proposes an iterative inference algorithm for multi-hop explanation regeneration, that retrieves relevant factual evidence in the form of text snippets, given a natural language question and its answer. Combining multiple sources of evidence or facts for multi-hop reasoning becomes increasingly hard when the number of sources needed to make an inference grows. Our algorithm copes with this by decomposing the selection of facts from a corpus autoregressively, conditioning the next iteration on previously selected facts. This allows us to use a pairwise learning-to-rank loss. We validate our method on datasets of the TextGraphs 2019 and 2020 Shared Tasks for explanation regeneration. Existing work on this task either evaluates facts in isolation or artificially limits the possible chains of facts, thus limiting multi-hop inference. We demonstrate that our algorithm, when used with a pre-trained transformer model, outperforms the previous state-of-the-art in terms of precision, training time and inference efficiency.

</p>
</details>

<details><summary><b>Automatic detection of abnormal EEG signals using wavelet feature extraction and gradient boosting decision tree</b>
<a href="https://arxiv.org/abs/2012.10034">arxiv:2012.10034</a>
&#x1F4C8; 3 <br>
<p>Hezam Albaqami, Ghulam Mubashar Hassan, Abdulhamit Subasi, Amitava Datta</p></summary>
<p>

**Abstract:** Electroencephalography is frequently used for diagnostic evaluation of various brain-related disorders due to its excellent resolution, non-invasive nature and low cost. However, manual analysis of EEG signals could be strenuous and a time-consuming process for experts. It requires long training time for physicians to develop expertise in it and additionally experts have low inter-rater agreement (IRA) among themselves. Therefore, many Computer Aided Diagnostic (CAD) based studies have considered the automation of interpreting EEG signals to alleviate the workload and support the final diagnosis. In this paper, we present an automatic binary classification framework for brain signals in multichannel EEG recordings. We propose to use Wavelet Packet Decomposition (WPD) techniques to decompose the EEG signals into frequency sub-bands and extract a set of statistical features from each of the selected coefficients. Moreover, we propose a novel method to reduce the dimension of the feature space without compromising the quality of the extracted features. The extracted features are classified using different Gradient Boosting Decision Tree (GBDT) based classification frameworks, which are CatBoost, XGBoost and LightGBM. We used Temple University Hospital EEG Abnormal Corpus V2.0.0 to test our proposed technique. We found that CatBoost classifier achieves the binary classification accuracy of 87.68%, and outperforms state-of-the-art techniques on the same dataset by more than 1% in accuracy and more than 3% in sensitivity. The obtained results in this research provide important insights into the usefulness of WPD feature extraction and GBDT classifiers for EEG classification.

</p>
</details>

<details><summary><b>Exploring Fluent Query Reformulations with Text-to-Text Transformers and Reinforcement Learning</b>
<a href="https://arxiv.org/abs/2012.10033">arxiv:2012.10033</a>
&#x1F4C8; 3 <br>
<p>Jerry Zikun Chen, Shi Yu, Haoran Wang</p></summary>
<p>

**Abstract:** Query reformulation aims to alter potentially noisy or ambiguous text sequences into coherent ones closer to natural language questions. In this process, it is also crucial to maintain and even enhance performance in a downstream environments like question answering when rephrased queries are given as input. We explore methods to generate these query reformulations by training reformulators using text-to-text transformers and apply policy-based reinforcement learning algorithms to further encourage reward learning. Query fluency is numerically evaluated by the same class of model fine-tuned on a human-evaluated well-formedness dataset. The reformulator leverages linguistic knowledge obtained from transfer learning and generates more well-formed reformulations than a translation-based model in qualitative and quantitative analysis. During reinforcement learning, it better retains fluency while optimizing the RL objective to acquire question answering rewards and can generalize to out-of-sample textual data in qualitative evaluations. Our RL framework is demonstrated to be flexible, allowing reward signals to be sourced from different downstream environments such as intent classification.

</p>
</details>

<details><summary><b>An Improved Approach for Estimating Social POI Boundaries With Textual Attributes on Social Media</b>
<a href="https://arxiv.org/abs/2012.09990">arxiv:2012.09990</a>
&#x1F4C8; 3 <br>
<p>Cong Tran, Dung D. Vu, Won-Yong Shin</p></summary>
<p>

**Abstract:** It has been insufficiently explored how to perform density-based clustering by exploiting textual attributes on social media. In this paper, we aim at discovering a social point-of-interest (POI) boundary, formed as a convex polygon. More specifically, we present a new approach and algorithm, built upon our earlier work on social POI boundary estimation (SoBEst). This SoBEst approach takes into account both relevant and irrelevant records within a geographic area, where relevant records contain a POI name or its variations in their text field. Our study is motivated by the following empirical observation: a fixed representative coordinate of each POI that SoBEst basically assumes may be far away from the centroid of the estimated social POI boundary for certain POIs. Thus, using SoBEst in such cases may possibly result in unsatisfactory performance on the boundary estimation quality (BEQ), which is expressed as a function of the $F$-measure. To solve this problem, we formulate a joint optimization problem of simultaneously finding the radius of a circle and the POI's representative coordinate $c$ by allowing to update $c$. Subsequently, we design an iterative SoBEst (I-SoBEst) algorithm, which enables us to achieve a higher degree of BEQ for some POIs. The computational complexity of the proposed I-SoBEst algorithm is shown to scale linearly with the number of records. We demonstrate the superiority of our algorithm over competing clustering methods including the original SoBEst.

</p>
</details>

<details><summary><b>Can Transformers Reason About Effects of Actions?</b>
<a href="https://arxiv.org/abs/2012.09938">arxiv:2012.09938</a>
&#x1F4C8; 3 <br>
<p>Pratyay Banerjee, Chitta Baral, Man Luo, Arindam Mitra, Kuntal Pal, Tran C. Son, Neeraj Varshney</p></summary>
<p>

**Abstract:** A recent work has shown that transformers are able to "reason" with facts and rules in a limited setting where the rules are natural language expressions of conjunctions of conditions implying a conclusion. Since this suggests that transformers may be used for reasoning with knowledge given in natural language, we do a rigorous evaluation of this with respect to a common form of knowledge and its corresponding reasoning -- the reasoning about effects of actions. Reasoning about action and change has been a top focus in the knowledge representation subfield of AI from the early days of AI and more recently it has been a highlight aspect in common sense question answering. We consider four action domains (Blocks World, Logistics, Dock-Worker-Robots and a Generic Domain) in natural language and create QA datasets that involve reasoning about the effects of actions in these domains. We investigate the ability of transformers to (a) learn to reason in these domains and (b) transfer that learning from the generic domains to the other domains.

</p>
</details>

<details><summary><b>Multi-FinGAN: Generative Coarse-To-Fine Sampling of Multi-Finger Grasps</b>
<a href="https://arxiv.org/abs/2012.09696">arxiv:2012.09696</a>
&#x1F4C8; 3 <br>
<p>Jens Lundell, Enric Corona, Tran Nguyen Le, Francesco Verdoja, Philippe Weinzaepfel, Gregory Rogez, Francesc Moreno-Noguer, Ville Kyrki</p></summary>
<p>

**Abstract:** While there exists a large number of methods for manipulating rigid objects with parallel-jaw grippers, grasping with multi-finger robotic hands remains a quite unexplored research topic. Reasoning and planning collision-free trajectories on the additional degrees of freedom of several fingers represents an important challenge that, so far, involves computationally costly and slow processes. In this work, we present Multi-FinGAN, a fast generative multi-finger grasp sampling method that synthesizes high quality grasps directly from RGB-D images in about a second. We achieve this by training in an end-to-end fashion a coarse-to-fine model composed of a classification network that distinguishes grasp types according to a specific taxonomy and a refinement network that produces refined grasp poses and joint angles. We experimentally validate and benchmark our method against standard grasp-sampling methods on 790 grasps in simulation and 20 grasps on a real Franka Emika Panda. All experimental results using our method show consistent improvements both in terms of grasp quality metrics and grasp success rate. Remarkably, our approach is up to 20-30 times faster than the baseline, a significant improvement that opens the door to feedback-based grasp re-planning and task informative grasping.

</p>
</details>

<details><summary><b>RainBench: Towards Global Precipitation Forecasting from Satellite Imagery</b>
<a href="https://arxiv.org/abs/2012.09670">arxiv:2012.09670</a>
&#x1F4C8; 3 <br>
<p>Christian Schroeder de Witt, Catherine Tong, Valentina Zantedeschi, Daniele De Martini, Freddie Kalaitzis, Matthew Chantry, Duncan Watson-Parris, Piotr Bilinski</p></summary>
<p>

**Abstract:** Extreme precipitation events, such as violent rainfall and hail storms, routinely ravage economies and livelihoods around the developing world. Climate change further aggravates this issue. Data-driven deep learning approaches could widen the access to accurate multi-day forecasts, to mitigate against such events. However, there is currently no benchmark dataset dedicated to the study of global precipitation forecasts. In this paper, we introduce \textbf{RainBench}, a new multi-modal benchmark dataset for data-driven precipitation forecasting. It includes simulated satellite data, a selection of relevant meteorological data from the ERA5 reanalysis product, and IMERG precipitation data. We also release \textbf{PyRain}, a library to process large precipitation datasets efficiently. We present an extensive analysis of our novel dataset and establish baseline results for two benchmark medium-range precipitation forecasting tasks. Finally, we discuss existing data-driven weather forecasting methodologies and suggest future research avenues.

</p>
</details>

<details><summary><b>Multi-Modal Depth Estimation Using Convolutional Neural Networks</b>
<a href="https://arxiv.org/abs/2012.09667">arxiv:2012.09667</a>
&#x1F4C8; 3 <br>
<p>Sadique Adnan Siddiqui, Axel Vierling, Karsten Berns</p></summary>
<p>

**Abstract:** This paper addresses the problem of dense depth predictions from sparse distance sensor data and a single camera image on challenging weather conditions. This work explores the significance of different sensor modalities such as camera, Radar, and Lidar for estimating depth by applying Deep Learning approaches. Although Lidar has higher depth-sensing abilities than Radar and has been integrated with camera images in lots of previous works, depth estimation using CNN's on the fusion of robust Radar distance data and camera images has not been explored much. In this work, a deep regression network is proposed utilizing a transfer learning approach consisting of an encoder where a high performing pre-trained model has been used to initialize it for extracting dense features and a decoder for upsampling and predicting desired depth. The results are demonstrated on Nuscenes, KITTI, and a Synthetic dataset which was created using the CARLA simulator. Also, top-view zoom-camera images captured from the crane on a construction site are evaluated to estimate the distance of the crane boom carrying heavy loads from the ground to show the usability in safety-critical applications.

</p>
</details>

<details><summary><b>XAI-P-T: A Brief Review of Explainable Artificial Intelligence from Practice to Theory</b>
<a href="https://arxiv.org/abs/2012.09636">arxiv:2012.09636</a>
&#x1F4C8; 3 <br>
<p>Nazanin Fouladgar, Kary Främling</p></summary>
<p>

**Abstract:** In this work, we report the practical and theoretical aspects of Explainable AI (XAI) identified in some fundamental literature. Although there is a vast body of work on representing the XAI backgrounds, most of the corpuses pinpoint a discrete direction of thoughts. Providing insights into literature in practice and theory concurrently is still a gap in this field. This is important as such connection facilitates a learning process for the early stage XAI researchers and give a bright stand for the experienced XAI scholars. Respectively, we first focus on the categories of black-box explanation and give a practical example. Later, we discuss how theoretically explanation has been grounded in the body of multidisciplinary fields. Finally, some directions of future works are presented.

</p>
</details>

<details><summary><b>Distance-aware Molecule Graph Attention Network for Drug-Target Binding Affinity Prediction</b>
<a href="https://arxiv.org/abs/2012.09624">arxiv:2012.09624</a>
&#x1F4C8; 3 <br>
<p>Jingbo Zhou, Shuangli Li, Liang Huang, Haoyi Xiong, Fan Wang, Tong Xu, Hui Xiong, Dejing Dou</p></summary>
<p>

**Abstract:** Accurately predicting the binding affinity between drugs and proteins is an essential step for computational drug discovery. Since graph neural networks (GNNs) have demonstrated remarkable success in various graph-related tasks, GNNs have been considered as a promising tool to improve the binding affinity prediction in recent years. However, most of the existing GNN architectures can only encode the topological graph structure of drugs and proteins without considering the relative spatial information among their atoms. Whereas, different from other graph datasets such as social networks and commonsense knowledge graphs, the relative spatial position and chemical bonds among atoms have significant impacts on the binding affinity. To this end, in this paper, we propose a diStance-aware Molecule graph Attention Network (S-MAN) tailored to drug-target binding affinity prediction. As a dedicated solution, we first propose a position encoding mechanism to integrate the topological structure and spatial position information into the constructed pocket-ligand graph. Moreover, we propose a novel edge-node hierarchical attentive aggregation structure which has edge-level aggregation and node-level aggregation. The hierarchical attentive aggregation can capture spatial dependencies among atoms, as well as fuse the position-enhanced information with the capability of discriminating multiple spatial relations among atoms. Finally, we conduct extensive experiments on two standard datasets to demonstrate the effectiveness of S-MAN.

</p>
</details>

<details><summary><b>Weakly-Supervised Action Localization and Action Recognition using Global-Local Attention of 3D CNN</b>
<a href="https://arxiv.org/abs/2012.09542">arxiv:2012.09542</a>
&#x1F4C8; 3 <br>
<p>Novanto Yudistira, Muthu Subash Kavitha, Takio Kurita</p></summary>
<p>

**Abstract:** 3D Convolutional Neural Network (3D CNN) captures spatial and temporal information on 3D data such as video sequences. However, due to the convolution and pooling mechanism, the information loss seems unavoidable. To improve the visual explanations and classification in 3D CNN, we propose two approaches; i) aggregate layer-wise global to local (global-local) discrete gradients using trained 3DResNext network, and ii) implement attention gating network to improve the accuracy of the action recognition. The proposed approach intends to show the usefulness of every layer termed as global-local attention in 3D CNN via visual attribution, weakly-supervised action localization, and action recognition. Firstly, the 3DResNext is trained and applied for action classification using backpropagation concerning the maximum predicted class. The gradients and activations of every layer are then up-sampled. Later, aggregation is used to produce more nuanced attention, which points out the most critical part of the predicted class's input videos. We use contour thresholding of final attention for final localization. We evaluate spatial and temporal action localization in trimmed videos using fine-grained visual explanation via 3DCam. Experimental results show that the proposed approach produces informative visual explanations and discriminative attention. Furthermore, the action recognition via attention gating on each layer produces better classification results than the baseline model.

</p>
</details>

<details><summary><b>Enhancing Balanced Graph Edge Partition with Effective Local Search</b>
<a href="https://arxiv.org/abs/2012.09451">arxiv:2012.09451</a>
&#x1F4C8; 3 <br>
<p>Zhenyu Guo, Mingyu Xiao, Yi Zhou, Dongxiang Zhang, Kian-Lee Tan</p></summary>
<p>

**Abstract:** Graph partition is a key component to achieve workload balance and reduce job completion time in parallel graph processing systems. Among the various partition strategies, edge partition has demonstrated more promising performance in power-law graphs than vertex partition and thereby has been more widely adopted as the default partition strategy by existing graph systems. The graph edge partition problem, which is to split the edge set into multiple balanced parts to minimize the total number of copied vertices, has been widely studied from the view of optimization and algorithms. In this paper, we study local search algorithms for this problem to further improve the partition results from existing methods. More specifically, we propose two novel concepts, namely adjustable edges and blocks. Based on these, we develop a greedy heuristic as well as an improved search algorithm utilizing the property of the max-flow model. To evaluate the performance of our algorithms, we first provide adequate theoretical analysis in terms of the approximation quality. We significantly improve the previously known approximation ratio for this problem. Then we conduct extensive experiments on a large number of benchmark datasets and state-of-the-art edge partition strategies. The results show that our proposed local search framework can further improve the quality of graph partition by a wide margin.

</p>
</details>

<details><summary><b>Causality-Aware Neighborhood Methods for Recommender Systems</b>
<a href="https://arxiv.org/abs/2012.09442">arxiv:2012.09442</a>
&#x1F4C8; 3 <br>
<p>Masahiro Sato, Sho Takemori, Janmajay Singh, Qian Zhang</p></summary>
<p>

**Abstract:** The business objectives of recommenders, such as increasing sales, are aligned with the causal effect of recommendations. Previous recommenders targeting for the causal effect employ the inverse propensity scoring (IPS) in causal inference. However, IPS is prone to suffer from high variance. The matching estimator is another representative method in causal inference field. It does not use propensity and hence free from the above variance problem. In this work, we unify traditional neighborhood recommendation methods with the matching estimator, and develop robust ranking methods for the causal effect of recommendations. Our experiments demonstrate that the proposed methods outperform various baselines in ranking metrics for the causal effect. The results suggest that the proposed methods can achieve more sales and user engagement than previous recommenders.

</p>
</details>

<details><summary><b>Smoothed Gaussian Mixture Models for Video Classification and Recommendation</b>
<a href="https://arxiv.org/abs/2012.11673">arxiv:2012.11673</a>
&#x1F4C8; 2 <br>
<p>Sirjan Kafle, Aman Gupta, Xue Xia, Ananth Sankar, Xi Chen, Di Wen, Liang Zhang</p></summary>
<p>

**Abstract:** Cluster-and-aggregate techniques such as Vector of Locally Aggregated Descriptors (VLAD), and their end-to-end discriminatively trained equivalents like NetVLAD have recently been popular for video classification and action recognition tasks. These techniques operate by assigning video frames to clusters and then representing the video by aggregating residuals of frames with respect to the mean of each cluster. Since some clusters may see very little video-specific data, these features can be noisy. In this paper, we propose a new cluster-and-aggregate method which we call smoothed Gaussian mixture model (SGMM), and its end-to-end discriminatively trained equivalent, which we call deep smoothed Gaussian mixture model (DSGMM). SGMM represents each video by the parameters of a Gaussian mixture model (GMM) trained for that video. Low-count clusters are addressed by smoothing the video-specific estimates with a universal background model (UBM) trained on a large number of videos. The primary benefit of SGMM over VLAD is smoothing which makes it less sensitive to small number of training samples. We show, through extensive experiments on the YouTube-8M classification task, that SGMM/DSGMM is consistently better than VLAD/NetVLAD by a small but statistically significant margin. We also show results using a dataset created at LinkedIn to predict if a member will watch an uploaded video.

</p>
</details>

<details><summary><b>EVA: Generating Longitudinal Electronic Health Records Using Conditional Variational Autoencoders</b>
<a href="https://arxiv.org/abs/2012.10020">arxiv:2012.10020</a>
&#x1F4C8; 2 <br>
<p>Siddharth Biswal, Soumya Ghosh, Jon Duke, Bradley Malin, Walter Stewart, Jimeng Sun</p></summary>
<p>

**Abstract:** Researchers require timely access to real-world longitudinal electronic health records (EHR) to develop, test, validate, and implement machine learning solutions that improve the quality and efficiency of healthcare. In contrast, health systems value deeply patient privacy and data security. De-identified EHRs do not adequately address the needs of health systems, as de-identified data are susceptible to re-identification and its volume is also limited. Synthetic EHRs offer a potential solution. In this paper, we propose EHR Variational Autoencoder (EVA) for synthesizing sequences of discrete EHR encounters (e.g., clinical visits) and encounter features (e.g., diagnoses, medications, procedures). We illustrate that EVA can produce realistic EHR sequences, account for individual differences among patients, and can be conditioned on specific disease conditions, thus enabling disease-specific studies. We design efficient, accurate inference algorithms by combining stochastic gradient Markov Chain Monte Carlo with amortized variational inference. We assess the utility of the methods on large real-world EHR repositories containing over 250, 000 patients. Our experiments, which include user studies with knowledgeable clinicians, indicate the generated EHR sequences are realistic. We confirmed the performance of predictive models trained on the synthetic data are similar with those trained on real EHRs. Additionally, our findings indicate that augmenting real data with synthetic EHRs results in the best predictive performance - improving the best baseline by as much as 8% in top-20 recall.

</p>
</details>

<details><summary><b>Binomial Tails for Community Analysis</b>
<a href="https://arxiv.org/abs/2012.09968">arxiv:2012.09968</a>
&#x1F4C8; 2 <br>
<p>Omid Madani, Thanh Ngo, Weifei Zeng, Sai Ankith Averine, Sasidhar Evuru, Varun Malhotra, Shashidhar Gandham, Navindra Yadav</p></summary>
<p>

**Abstract:** An important task of community discovery in networks is assessing significance of the results and robust ranking of the generated candidate groups. Often in practice, numerous candidate communities are discovered, and focusing the analyst's time on the most salient and promising findings is crucial. We develop simple efficient group scoring functions derived from tail probabilities using binomial models. Experiments on synthetic and numerous real-world data provides evidence that binomial scoring leads to a more robust ranking than other inexpensive scoring functions, such as conductance. Furthermore, we obtain confidence values ($p$-values) that can be used for filtering and labeling the discovered groups. Our analyses shed light on various properties of the approach. The binomial tail is simple and versatile, and we describe two other applications for community analysis: degree of community membership (which in turn yields group-scoring functions), and the discovery of significant edges in the community-induced graph.

</p>
</details>

<details><summary><b>Named Entity Recognition in the Legal Domain using a Pointer Generator Network</b>
<a href="https://arxiv.org/abs/2012.09936">arxiv:2012.09936</a>
&#x1F4C8; 2 <br>
<p>Stavroula Skylaki, Ali Oskooei, Omar Bari, Nadja Herger, Zac Kriegman</p></summary>
<p>

**Abstract:** Named Entity Recognition (NER) is the task of identifying and classifying named entities in unstructured text. In the legal domain, named entities of interest may include the case parties, judges, names of courts, case numbers, references to laws etc. We study the problem of legal NER with noisy text extracted from PDF files of filed court cases from US courts. The "gold standard" training data for NER systems provide annotation for each token of the text with the corresponding entity or non-entity label. We work with only partially complete training data, which differ from the gold standard NER data in that the exact location of the entities in the text is unknown and the entities may contain typos and/or OCR mistakes. To overcome the challenges of our noisy training data, e.g. text extraction errors and/or typos and unknown label indices, we formulate the NER task as a text-to-text sequence generation task and train a pointer generator network to generate the entities in the document rather than label them. We show that the pointer generator can be effective for NER in the absence of gold standard data and outperforms the common NER neural network architectures in long legal documents.

</p>
</details>

<details><summary><b>Efficient CNN-LSTM based Image Captioning using Neural Network Compression</b>
<a href="https://arxiv.org/abs/2012.09708">arxiv:2012.09708</a>
&#x1F4C8; 2 <br>
<p>Harshit Rampal, Aman Mohanty</p></summary>
<p>

**Abstract:** Modern Neural Networks are eminent in achieving state of the art performance on tasks under Computer Vision, Natural Language Processing and related verticals. However, they are notorious for their voracious memory and compute appetite which further obstructs their deployment on resource limited edge devices. In order to achieve edge deployment, researchers have developed pruning and quantization algorithms to compress such networks without compromising their efficacy. Such compression algorithms are broadly experimented on standalone CNN and RNN architectures while in this work, we present an unconventional end to end compression pipeline of a CNN-LSTM based Image Captioning model. The model is trained using VGG16 or ResNet50 as an encoder and an LSTM decoder on the flickr8k dataset. We then examine the effects of different compression architectures on the model and design a compression architecture that achieves a 73.1% reduction in model size, 71.3% reduction in inference time and a 7.7% increase in BLEU score as compared to its uncompressed counterpart.

</p>
</details>

<details><summary><b>Estimating mixed-memberships using the Symmetric Laplacian Inverse Matrix</b>
<a href="https://arxiv.org/abs/2012.09561">arxiv:2012.09561</a>
&#x1F4C8; 2 <br>
<p>Huan Qing, Jingli Wang</p></summary>
<p>

**Abstract:** Community detection has been well studied in network analysis, and one popular technique is spectral clustering which is fast and statistically analyzable for detect-ing clusters for given networks. But the more realistic case of mixed membership community detection remains a challenge. In this paper, we propose a new spectral clustering method Mixed-SLIM for mixed membership community detection. Mixed-SLIM is designed based on the symmetrized Laplacian inverse matrix (SLIM) (Jing et al. 2021) under the degree-corrected mixed membership (DCMM) model. We show that this algorithm and its regularized version Mixed-SLIM τ are asymptotically consistent under mild conditions. Meanwhile, we provide Mixed-SLIM appro and its regularized version Mixed-SLIM τappro by approximating the SLIM matrix when dealing with large networks in practice. These four Mixed-SLIM methods outperform state-of-art methods in simulations and substantial empirical datasets for both community detection and mixed membership community detection problems.

</p>
</details>

<details><summary><b>Experts with Lower-Bounded Loss Feedback: A Unifying Framework</b>
<a href="https://arxiv.org/abs/2012.09537">arxiv:2012.09537</a>
&#x1F4C8; 2 <br>
<p>Eyal Gofer, Guy Gilboa</p></summary>
<p>

**Abstract:** The most prominent feedback models for the best expert problem are the full information and bandit models. In this work we consider a simple feedback model that generalizes both, where on every round, in addition to a bandit feedback, the adversary provides a lower bound on the loss of each expert. Such lower bounds may be obtained in various scenarios, for instance, in stock trading or in assessing errors of certain measurement devices. For this model we prove optimal regret bounds (up to logarithmic factors) for modified versions of Exp3, generalizing algorithms and bounds both for the bandit and the full-information settings. Our second-order unified regret analysis simulates a two-step loss update and highlights three Hessian or Hessian-like expressions, which map to the full-information regret, bandit regret, and a hybrid of both. Our results intersect with those for bandits with graph-structured feedback, in that both settings can accommodate feedback from an arbitrary subset of experts on each round. However, our model also accommodates partial feedback at the single-expert level, by allowing non-trivial lower bounds on each loss.

</p>
</details>

<details><summary><b>Towards Optimal District Heating Temperature Control in China with Deep Reinforcement Learning</b>
<a href="https://arxiv.org/abs/2012.09508">arxiv:2012.09508</a>
&#x1F4C8; 2 <br>
<p>Adrien Le-Coz, Tahar Nabil, Francois Courtot</p></summary>
<p>

**Abstract:** Achieving efficiency gains in Chinese district heating networks, thereby reducing their carbon footprint, requires new optimal control methods going beyond current industry tools. Focusing on the secondary network, we propose a data-driven deep reinforcement learning (DRL) approach to address this task. We build a recurrent neural network, trained on simulated data, to predict the indoor temperatures. This model is then used to train two DRL agents, with or without expert guidance, for the optimal control of the supply water temperature. Our tests in a multi-apartment setting show that both agents can ensure a higher thermal comfort and at the same time a smaller energy cost, compared to an optimized baseline strategy.

</p>
</details>

<details><summary><b>ReferentialGym: A Nomenclature and Framework for Language Emergence & Grounding in (Visual) Referential Games</b>
<a href="https://arxiv.org/abs/2012.09486">arxiv:2012.09486</a>
&#x1F4C8; 2 <br>
<p>Kevin Denamganaï, James Alfred Walker</p></summary>
<p>

**Abstract:** Natural languages are powerful tools wielded by human beings to communicate information and co-operate towards common goals. Their values lie in some main properties like compositionality, hierarchy and recurrent syntax, which computational linguists have been researching the emergence of in artificial languages induced by language games. Only relatively recently, the AI community has started to investigate language emergence and grounding working towards better human-machine interfaces. For instance, interactive/conversational AI assistants that are able to relate their vision to the ongoing conversation.
  This paper provides two contributions to this research field. Firstly, a nomenclature is proposed to understand the main initiatives in studying language emergence and grounding, accounting for the variations in assumptions and constraints. Secondly, a PyTorch based deep learning framework is introduced, entitled ReferentialGym, which is dedicated to furthering the exploration of language emergence and grounding. By providing baseline implementations of major algorithms and metrics, in addition to many different features and approaches, ReferentialGym attempts to ease the entry barrier to the field and provide the community with common implementations.

</p>
</details>

<details><summary><b>Helping Reduce Environmental Impact of Aviation with Machine Learning</b>
<a href="https://arxiv.org/abs/2012.09433">arxiv:2012.09433</a>
&#x1F4C8; 2 <br>
<p>Ashish Kapoor</p></summary>
<p>

**Abstract:** Commercial aviation is one of the biggest contributors towards climate change. We propose to reduce environmental impact of aviation by considering solutions that would reduce the flight time. Specifically, we first consider improving winds aloft forecast so that flight planners could use better information to find routes that are efficient. Secondly, we propose an aircraft routing method that seeks to find the fastest route to the destination by considering uncertainty in the wind forecasts and then optimally trading-off between exploration and exploitation.

</p>
</details>

<details><summary><b>The Variational Method of Moments</b>
<a href="https://arxiv.org/abs/2012.09422">arxiv:2012.09422</a>
&#x1F4C8; 2 <br>
<p>Andrew Bennett, Nathan Kallus</p></summary>
<p>

**Abstract:** The conditional moment problem is a powerful formulation for describing structural causal parameters in terms of observables, a prominent example being instrumental variable regression. A standard approach is to reduce the problem to a finite set of marginal moment conditions and apply the optimally weighted generalized method of moments (OWGMM), but this requires we know a finite set of identifying moments, can still be inefficient even if identifying, or can be unwieldy and impractical if we use a growing sieve of moments. Motivated by a variational minimax reformulation of OWGMM, we define a very general class of estimators for the conditional moment problem, which we term the variational method of moments (VMM) and which naturally enables controlling infinitely-many moments. We provide a detailed theoretical analysis of multiple VMM estimators, including based on kernel methods and neural networks, and provide appropriate conditions under which these estimators are consistent, asymptotically normal, and semiparametrically efficient in the full conditional moment model. This is in contrast to other recently proposed methods for solving conditional moment problems based on adversarial machine learning, which do not incorporate optimal weighting, do not establish asymptotic normality, and are not semiparametrically efficient.

</p>
</details>

<details><summary><b>FantastIC4: A Hardware-Software Co-Design Approach for Efficiently Running 4bit-Compact Multilayer Perceptrons</b>
<a href="https://arxiv.org/abs/2012.11331">arxiv:2012.11331</a>
&#x1F4C8; 1 <br>
<p>Simon Wiedemann, Suhas Shivapakash, Pablo Wiedemann, Daniel Becking, Wojciech Samek, Friedel Gerfers, Thomas Wiegand</p></summary>
<p>

**Abstract:** With the growing demand for deploying deep learning models to the "edge", it is paramount to develop techniques that allow to execute state-of-the-art models within very tight and limited resource constraints. In this work we propose a software-hardware optimization paradigm for obtaining a highly efficient execution engine of deep neural networks (DNNs) that are based on fully-connected layers. Our approach is centred around compression as a means for reducing the area as well as power requirements of, concretely, multilayer perceptrons (MLPs) with high predictive performances. Firstly, we design a novel hardware architecture named FantastIC4, which (1) supports the efficient on-chip execution of multiple compact representations of fully-connected layers and (2) minimizes the required number of multipliers for inference down to only 4 (thus the name). Moreover, in order to make the models amenable for efficient execution on FantastIC4, we introduce a novel entropy-constrained training method that renders them to be robust to 4bit quantization and highly compressible in size simultaneously. The experimental results show that we can achieve throughputs of 2.45 TOPS with a total power consumption of 3.6W on a Virtual Ultrascale FPGA XCVU440 device implementation, and achieve a total power efficiency of 20.17 TOPS/W on a 22nm process ASIC version. When compared to the other state-of-the-art accelerators designed for the Google Speech Command (GSC) dataset, FantastIC4 is better by 51$\times$ in terms of throughput and 145$\times$ in terms of area efficiency (GOPS/W).

</p>
</details>

<details><summary><b>Bayesian Convolutional Neural Networks as probabilistic surrogates for the fast prediction of stress fields in structures with microscale features</b>
<a href="https://arxiv.org/abs/2012.11330">arxiv:2012.11330</a>
&#x1F4C8; 1 <br>
<p>Vasilis Krokos, Viet Bui Xuan, Stéphane P. A. Bordas, Philippe Young, Pierre Kerfriden</p></summary>
<p>

**Abstract:** Finite Element Analysis (FEA) for stress prediction in structures with microstructural features is computationally expensive since those features are much smaller than the other geometric features of the structure. The accurate prediction of the additional stress generated by such microstructural features therefore requires a very fine FE mesh. Omitting or averaging the effect of the microstructural features from FEA models is standard practice, resulting in faster calculations of global stress fields, which, assuming some degree of scale separability, may then be complemented by local defect analyses. The purpose of this work is to train an Encoder-Decoder Convolutional Neural Networks (CNN) to automatically add local fine-scale stress corrections to coarse stress predictions around defects. We wish to understand to what extent such a framework may provide reliable stress predictions inside and outside the training set, i.e. for unseen coarse scale geometries and stress distributions and/or unseen defect geometries. Ultimately, we aim to develop efficient offline data generation and online data acquisition methods to maximise the domain of validity of the CNN predictions. To achieve these ambitious goals, we will deploy a Bayesian approach providing not point estimates, but credible intervals of the fine-scale stress field, as a means to evaluate the uncertainty of the predictions. The uncertainty quantified by the network will automatically encompass the lack of knowledge due to unseen macro and micro features, and the lack of knowledge due to the potential lack of scale separability. This uncertainty will be used in a Selective Learning framework to reduce the data requirements of the network. In this work we will investigate stress prediction in 2D composite structures with randomly distributed circular pores.

</p>
</details>

<details><summary><b>Searching for Possible Exoplanet Transits from BRITE Data through a Machine Learning Technique</b>
<a href="https://arxiv.org/abs/2012.10035">arxiv:2012.10035</a>
&#x1F4C8; 1 <br>
<p>Li-Chin Yeh, Ing-Guey Jiang</p></summary>
<p>

**Abstract:** The photometric light curves of BRITE satellites were examined through a machine learning technique to investigate whether there are possible exoplanets moving around nearby bright stars. Focusing on different transit periods, several convolutional neural networks were constructed to search for transit candidates. The convolutional neural networks were trained with synthetic transit signals combined with BRITE light curves until the accuracy rate was higher than 99.7 $\%$. Our method could efficiently lead to a small number of possible transit candidates. Among these ten candidates, two of them, HD37465, and HD186882 systems, were followed up through future observations with a higher priority. The codes of convolutional neural networks employed in this study are publicly available at http://www.phys.nthu.edu.tw/$\sim$jiang/BRITE2020YehJiangCNN.tar.gz.

</p>
</details>

<details><summary><b>Prediction of brain strain across head impact subtypes using 18 brain injury criteria</b>
<a href="https://arxiv.org/abs/2012.10006">arxiv:2012.10006</a>
&#x1F4C8; 1 <br>
<p>Xianghao Zhan, Yiheng Li, Yuzhe Liu, August G. Domel, Hossein Vahid Alidazeh, Samuel J. Raymond, Jesse Ruan, Saeed Barbat, Stephen Tiernan, Olivier Gevaert, Michael Zeineh, Gerald Grant</p></summary>
<p>

**Abstract:** Multiple brain injury criteria (BIC) are developed to quickly quantify brain injury risks after head impacts. These BIC originated from different types of head impacts (e.g., sports and car crashes) are widely used in risk evaluation. However, the predictability of the BIC on different types of head impacts has not been evaluated. Physiologically, the brain strain is often considered the key parameter of brain injury. To evaluate the BIC's ability to predict brain strain across five datasets comprising different head impact subtypes, linear regression was used to model 95% maximum principal strain, 95% maximum principal strain at corpus callosum, and cumulative strain damage (15%) on 18 BIC. The results show significant differences in the relationship between BIC and brain strain across datasets, indicating the same BIC value may indicate different brain strain in different head impact subtypes. The accuracy of regression is generally decreasing if the BIC regression models are fit on a dataset with a different head impact subtype rather than on the dataset with the same subtype. Given this finding, this study raises concerns for applying BIC to predict the brain strain for head impacts different from the head impacts on which the BIC was developed.

</p>
</details>

<details><summary><b>Data-driven rogue waves and parameter discovery in the defocusing NLS equation with a potential using the PINN deep learning</b>
<a href="https://arxiv.org/abs/2012.09984">arxiv:2012.09984</a>
&#x1F4C8; 1 <br>
<p>Li Wang, Zhenya Yan</p></summary>
<p>

**Abstract:** The physics-informed neural networks (PINNs) can be used to deep learn the nonlinear partial differential equations and other types of physical models. In this paper, we use the multi-layer PINN deep learning method to study the data-driven rogue wave solutions of the defocusing nonlinear Schrödinger (NLS) equation with the time-dependent potential by considering several initial conditions such as the rogue wave, Jacobi elliptic cosine function, two-Gaussian function, or three-hyperbolic-secant function, and periodic boundary conditions. Moreover, the multi-layer PINN algorithm can also be used to learn the parameter in the defocusing NLS equation with the time-dependent potential under the sense of the rogue wave solution. These results will be useful to further discuss the rogue wave solutions of the defocusing NLS equation with a potential in the study of deep learning neural networks.

</p>
</details>

<details><summary><b>High Dimensional Level Set Estimation with Bayesian Neural Network</b>
<a href="https://arxiv.org/abs/2012.09973">arxiv:2012.09973</a>
&#x1F4C8; 1 <br>
<p>Huong Ha, Sunil Gupta, Santu Rana, Svetha Venkatesh</p></summary>
<p>

**Abstract:** Level Set Estimation (LSE) is an important problem with applications in various fields such as material design, biotechnology, machine operational testing, etc. Existing techniques suffer from the scalability issue, that is, these methods do not work well with high dimensional inputs. This paper proposes novel methods to solve the high dimensional LSE problems using Bayesian Neural Networks. In particular, we consider two types of LSE problems: (1) \textit{explicit} LSE problem where the threshold level is a fixed user-specified value, and, (2) \textit{implicit} LSE problem where the threshold level is defined as a percentage of the (unknown) maximum of the objective function. For each problem, we derive the corresponding theoretic information based acquisition function to sample the data points so as to maximally increase the level set accuracy. Furthermore, we also analyse the theoretical time complexity of our proposed acquisition functions, and suggest a practical methodology to efficiently tune the network hyper-parameters to achieve high model accuracy. Numerical experiments on both synthetic and real-world datasets show that our proposed method can achieve better results compared to existing state-of-the-art approaches.

</p>
</details>

<details><summary><b>Guiding Neural Network Initialization via Marginal Likelihood Maximization</b>
<a href="https://arxiv.org/abs/2012.09943">arxiv:2012.09943</a>
&#x1F4C8; 1 <br>
<p>Anthony S. Tai, Chunfeng Huang</p></summary>
<p>

**Abstract:** We propose a simple, data-driven approach to help guide hyperparameter selection for neural network initialization. We leverage the relationship between neural network and Gaussian process models having corresponding activation and covariance functions to infer the hyperparameter values desirable for model initialization. Our experiment shows that marginal likelihood maximization provides recommendations that yield near-optimal prediction performance on MNIST classification task under experiment constraints. Furthermore, our empirical results indicate consistency in the proposed technique, suggesting that computation cost for the procedure could be significantly reduced with smaller training sets.

</p>
</details>

<details><summary><b>Increasing the efficiency of randomized trial estimates via linear adjustment for a prognostic score</b>
<a href="https://arxiv.org/abs/2012.09935">arxiv:2012.09935</a>
&#x1F4C8; 1 <br>
<p>Alejandro Schuler, David Walsh, Diana Hall, Jon Walsh, Charles Fisher</p></summary>
<p>

**Abstract:** Estimating causal effects from randomized experiments is central to clinical research. Reducing the statistical uncertainty in these analyses is an important objective for statisticians. Registries, prior trials, and health records constitute a growing compendium of historical data on patients under standard-of-care conditions that may be exploitable to this end. However, most methods for historical borrowing achieve reductions in variance by sacrificing strict type-I error rate control. Here, we propose a use of historical data that exploits linear covariate adjustment to improve the efficiency of trial analyses without incurring bias. Specifically, we train a prognostic model on the historical data, then estimate the treatment effect using a linear regression while adjusting for the trial subjects' predicted outcomes (their prognostic scores). We prove that, under certain conditions, this prognostic covariate adjustment procedure attains the minimum variance possible among a large class of estimators. When those conditions are not met, prognostic covariate adjustment is still more efficient than raw covariate adjustment and the gain in efficiency is proportional to a measure of the predictive accuracy of the prognostic model. We demonstrate the approach using simulations and a reanalysis of an Alzheimer's Disease clinical trial and observe meaningful reductions in mean-squared error and the estimated variance. Lastly, we provide a simplified formula for asymptotic variance that enables power and sample size calculations that account for the gains from the prognostic model for clinical trial design.

</p>
</details>

<details><summary><b>Game-theoretic Models of Moral and Other-Regarding Agents</b>
<a href="https://arxiv.org/abs/2012.09759">arxiv:2012.09759</a>
&#x1F4C8; 1 <br>
<p>Gabriel Istrate</p></summary>
<p>

**Abstract:** We investigate Kantian equilibria in finite normal form games, a class of non-Nashian, morally motivated courses of action that was recently proposed in the economics literature. We highlight a number of problems with such equilibria, including computational intractability, a high price of miscoordination, and expensive/problematic extension to general normal form games. We point out that such a proper generalization will likely involve the concept of program equilibrium. Finally we propose some general, intuitive, computationally tractable, other-regarding equilibria related to Kantian equilibria, as well as a class of courses of action that interpolates between purely self-regarding and Kantian behavior.

</p>
</details>

<details><summary><b>Individually Conditional Individual Mutual Information Bound on Generalization Error</b>
<a href="https://arxiv.org/abs/2012.09922">arxiv:2012.09922</a>
&#x1F4C8; 0 <br>
<p>Ruida Zhou, Chao Tian, Tie Liu</p></summary>
<p>

**Abstract:** We propose a new information-theoretic bound on generalization error based on a combination of the error decomposition technique of Bu et al. and the conditional mutual information (CMI) construction of Steinke and Zakynthinou. In a previous work, Haghifam et al. proposed a different bound combining the two aforementioned techniques, which we refer to as the conditional individual mutual information (CIMI) bound. However, in a simple Gaussian setting, both the CMI and the CIMI bounds are order-wise worse than that by Bu et al.. This observation motivated us to propose the new bound, which overcomes this issue by reducing the conditioning terms in the conditional mutual information. In the process of establishing this bound, a conditional decoupling lemma is established, which also leads to a meaningful dichotomy and comparison among these information-theoretic bounds.

</p>
</details>

<details><summary><b>Learning and Sharing: A Multitask Genetic Programming Approach to Image Feature Learning</b>
<a href="https://arxiv.org/abs/2012.09444">arxiv:2012.09444</a>
&#x1F4C8; 0 <br>
<p>Ying Bi, Bing Xue, Mengjie Zhang</p></summary>
<p>

**Abstract:** Using evolutionary computation algorithms to solve multiple tasks with knowledge sharing is a promising approach. Image feature learning can be considered as a multitask problem because different tasks may have a similar feature space. Genetic programming (GP) has been successfully applied to image feature learning for classification. However, most of the existing GP methods solve one task, independently, using sufficient training data. No multitask GP method has been developed for image feature learning. Therefore, this paper develops a multitask GP approach to image feature learning for classification with limited training data. Owing to the flexible representation of GP, a new knowledge sharing mechanism based on a new individual representation is developed to allow GP to automatically learn what to share across two tasks and to improve its learning performance. The shared knowledge is encoded as a common tree, which can represent the common/general features of two tasks. With the new individual representation, each task is solved using the features extracted from a common tree and a task-specific tree representing task-specific features. To learn the best common and task-specific trees, a new evolutionary process and new fitness functions are developed. The performance of the proposed approach is examined on six multitask problems of 12 image classification datasets with limited training data and compared with three GP and 14 non-GP-based competitive methods. Experimental results show that the new approach outperforms these compared methods in almost all the comparisons. Further analysis reveals that the new approach learns simple yet effective common trees with high effectiveness and transferability.

</p>
</details>


[Next Page](2020/2020-12/2020-12-16.md)
