## Summary for 2020-12-23, created on 2021-01-19


<details><summary><b>Solving Mixed Integer Programs Using Neural Networks</b>
<a href="https://arxiv.org/abs/2012.13349">arxiv:2012.13349</a>
&#x1F4C8; 154 <br>
<p>Vinod Nair, Sergey Bartunov, Felix Gimeno, Ingrid von Glehn, Pawel Lichocki, Ivan Lobov, Brendan O'Donoghue, Nicolas Sonnerat, Christian Tjandraatmadja, Pengming Wang, Ravichandra Addanki, Tharindi Hapuarachchi, Thomas Keck, James Keeling, Pushmeet Kohli, Ira Ktena, Yujia Li, Oriol Vinyals, Yori Zwols</p></summary>
<p>

**Abstract:** Mixed Integer Programming (MIP) solvers rely on an array of sophisticated heuristics developed with decades of research to solve large-scale MIP instances encountered in practice. Machine learning offers to automatically construct better heuristics from data by exploiting shared structure among instances in the data. This paper applies learning to the two key sub-tasks of a MIP solver, generating a high-quality joint variable assignment, and bounding the gap in objective value between that assignment and an optimal one. Our approach constructs two corresponding neural network-based components, Neural Diving and Neural Branching, to use in a base MIP solver such as SCIP. Neural Diving learns a deep neural network to generate multiple partial assignments for its integer variables, and the resulting smaller MIPs for un-assigned variables are solved with SCIP to construct high quality joint assignments. Neural Branching learns a deep neural network to make variable selection decisions in branch-and-bound to bound the objective value gap with a small tree. This is done by imitating a new variant of Full Strong Branching we propose that scales to large instances using GPUs. We evaluate our approach on six diverse real-world datasets, including two Google production datasets and MIPLIB, by training separate neural networks on each. Most instances in all the datasets combined have $10^3-10^6$ variables and constraints after presolve, which is significantly larger than previous learning approaches. Comparing solvers with respect to primal-dual gap averaged over a held-out set of instances, the learning-augmented SCIP is 2x to 10x better on all datasets except one on which it is $10^5$x better, at large time limits. To the best of our knowledge, ours is the first learning approach to demonstrate such large improvements over SCIP on both large-scale real-world application datasets and MIPLIB.

</p>
</details>

<details><summary><b>GANDA: A deep generative adversarial network predicts the spatial distribution of nanoparticles in tumor pixelly</b>
<a href="https://arxiv.org/abs/2012.12561">arxiv:2012.12561</a>
&#x1F4C8; 84 <br>
<p>Jiulou Zhang, Yuxia Tang, Shouju Wang</p></summary>
<p>

**Abstract:** Intratumoral nanoparticles (NPs) distribution is critical for the diagnostic and therapeutic effect, but methods to predict the distribution remain unavailable due to the complex bio-nano interactions. Here, we developed a Generative Adversarial Network for Distribution Analysis (GANDA) to make pixels-to-pixels prediction of the NPs distribution across tumors. This predictive model used deep learning approaches to automatically learn the features of tumor vessels and cell nuclei from whole-slide images of tumor sections. We showed that the GANDA could generate images of NPs distribution with the same spatial resolution as original images of tumor vessels and nuclei. The GANDA enabled quantitative analysis of NPs distribution (R2=0.93) and extravasation without knowing their real distribution. This model provides opportunities to investigate how influencing factors affect NPs distribution in individual tumors and may guide nanomedicine optimization for personalized treatments.

</p>
</details>

<details><summary><b>Learning emergent PDEs in a learned emergent space</b>
<a href="https://arxiv.org/abs/2012.12738">arxiv:2012.12738</a>
&#x1F4C8; 55 <br>
<p>Felix P. Kemeth, Tom Bertalan, Thomas Thiem, Felix Dietrich, Sung Joon Moon, Carlo R. Laing, Ioannis G. Kevrekidis</p></summary>
<p>

**Abstract:** We extract data-driven, intrinsic spatial coordinates from observations of the dynamics of large systems of coupled heterogeneous agents. These coordinates then serve as an emergent space in which to learn predictive models in the form of partial differential equations (PDEs) for the collective description of the coupled-agent system. They play the role of the independent spatial variables in this PDE (as opposed to the dependent, possibly also data-driven, state variables). This leads to an alternative description of the dynamics, local in these emergent coordinates, thus facilitating an alternative modeling path for complex coupled-agent systems. We illustrate this approach on a system where each agent is a limit cycle oscillator (a so-called Stuart-Landau oscillator); the agents are heterogeneous (they each have a different intrinsic frequency $ω$) and are coupled through the ensemble average of their respective variables. After fast initial transients, we show that the collective dynamics on a slow manifold can be approximated through a learned model based on local "spatial" partial derivatives in the emergent coordinates. The model is then used for prediction in time, as well as to capture collective bifurcations when system parameters vary. The proposed approach thus integrates the automatic, data-driven extraction of emergent space coordinates parametrizing the agent dynamics, with machine-learning assisted identification of an "emergent PDE" description of the dynamics in this parametrization.

</p>
</details>

<details><summary><b>A Survey on Visual Transformer</b>
<a href="https://arxiv.org/abs/2012.12556">arxiv:2012.12556</a>
&#x1F4C8; 55 <br>
<p>Kai Han, Yunhe Wang, Hanting Chen, Xinghao Chen, Jianyuan Guo, Zhenhua Liu, Yehui Tang, An Xiao, Chunjing Xu, Yixing Xu, Zhaohui Yang, Yiman Zhang, Dacheng Tao</p></summary>
<p>

**Abstract:** Transformer is a type of deep neural network mainly based on self-attention mechanism which is originally applied in natural language processing field. Inspired by the strong representation ability of transformer, researchers propose to extend transformer for computer vision tasks. Transformer-based models show competitive and even better performance on various visual benchmarks compared to other network types such as convolutional networks and recurrent networks. With high performance and without inductive bias defined by human, transformer is receiving more and more attention from the visual community. In this paper we provide a literature review of these visual transformer models by categorizing them in different tasks and analyze the advantages and disadvantages of these methods. In particular, the main categories include the basic image classification, high-level vision, low-level vision and video processing. The self-attention in computer vision is also briefly revisited as self-attention is the base component in transformer. Efficient transformer methods are included for pushing transformer into real applications on the devices. Finally, we give a discussion about the challenges and further research directions for visual transformers.

</p>
</details>

<details><summary><b>Learned Indexes for a Google-scale Disk-based Database</b>
<a href="https://arxiv.org/abs/2012.12501">arxiv:2012.12501</a>
&#x1F4C8; 55 <br>
<p>Hussam Abu-Libdeh, Deniz Altınbüken, Alex Beutel, Ed H. Chi, Lyric Doshi, Tim Kraska,  Xiaozhou,  Li, Andy Ly, Christopher Olston</p></summary>
<p>

**Abstract:** There is great excitement about learned index structures, but understandable skepticism about the practicality of a new method uprooting decades of research on B-Trees. In this paper, we work to remove some of that uncertainty by demonstrating how a learned index can be integrated in a distributed, disk-based database system: Google's Bigtable. We detail several design decisions we made to integrate learned indexes in Bigtable. Our results show that integrating learned index significantly improves the end-to-end read latency and throughput for Bigtable.

</p>
</details>

<details><summary><b>Hiding Among the Clones: A Simple and Nearly Optimal Analysis of Privacy Amplification by Shuffling</b>
<a href="https://arxiv.org/abs/2012.12803">arxiv:2012.12803</a>
&#x1F4C8; 52 <br>
<p>Vitaly Feldman, Audra McMillan, Kunal Talwar</p></summary>
<p>

**Abstract:** Recent work of Erlingsson, Feldman, Mironov, Raghunathan, Talwar, and Thakurta [EFMRTT19] demonstrates that random shuffling amplifies differential privacy guarantees of locally randomized data. Such amplification implies substantially stronger privacy guarantees for systems in which data is contributed anonymously [BEMMRLRKTS17] and has lead to significant interest in the shuffle model of privacy [CSUZZ19,EFMRTT19].
  We show that random shuffling of $n$ data records that are input to $\varepsilon_0$-differentially private local randomizers results in an $(O((1-e^{-\varepsilon_0})\sqrt{\frac{e^{\varepsilon_0}\log(1/δ)}{n}}), δ)$-differentially private algorithm. This significantly improves over previous work and achieves the asymptotically optimal dependence in $\varepsilon_0$. Our result is based on a new approach that is simpler than previous work and extends to approximate differential privacy with nearly the same guarantees. Our work also yields an empirical method to derive tighter bounds the resulting $\varepsilon$ and we show that it gets to within a small constant factor of the optimal bound. As a direct corollary of our analysis, we derive a simple and asymptotically optimal algorithm for discrete distribution estimation in the shuffle model of privacy. We also observe that our result implies the first asymptotically optimal privacy analysis of noisy stochastic gradient descent that applies to sampling without replacement.

</p>
</details>

<details><summary><b>Poisoning Attacks on Cyber Attack Detectors for Industrial Control Systems</b>
<a href="https://arxiv.org/abs/2012.15740">arxiv:2012.15740</a>
&#x1F4C8; 48 <br>
<p>Moshe Kravchik, Battista Biggio, Asaf Shabtai</p></summary>
<p>

**Abstract:** Recently, neural network (NN)-based methods, including autoencoders, have been proposed for the detection of cyber attacks targeting industrial control systems (ICSs). Such detectors are often retrained, using data collected during system operation, to cope with the natural evolution (i.e., concept drift) of the monitored signals. However, by exploiting this mechanism, an attacker can fake the signals provided by corrupted sensors at training time and poison the learning process of the detector such that cyber attacks go undetected at test time. With this research, we are the first to demonstrate such poisoning attacks on ICS cyber attack online NN detectors. We propose two distinct attack algorithms, namely, interpolation- and back-gradient based poisoning, and demonstrate their effectiveness on both synthetic and real-world ICS data. We also discuss and analyze some potential mitigation strategies.

</p>
</details>

<details><summary><b>Focal Frequency Loss for Generative Models</b>
<a href="https://arxiv.org/abs/2012.12821">arxiv:2012.12821</a>
&#x1F4C8; 19 <br>
<p>Liming Jiang, Bo Dai, Wayne Wu, Chen Change Loy</p></summary>
<p>

**Abstract:** Despite the remarkable success of generative models in creating photorealistic images using deep neural networks, gaps could still exist between the real and generated images, especially in the frequency domain. In this study, we find that narrowing the frequency domain gap can ameliorate the image synthesis quality further. To this end, we propose the focal frequency loss, a novel objective function that brings optimization of generative models into the frequency domain. The proposed loss allows the model to dynamically focus on the frequency components that are hard to synthesize by down-weighting the easy frequencies. This objective function is complementary to existing spatial losses, offering great impedance against the loss of important frequency information due to the inherent crux of neural networks. We demonstrate the versatility and effectiveness of focal frequency loss to improve various baselines in both perceptual quality and quantitative performance.

</p>
</details>

<details><summary><b>Bridging Textual and Tabular Data for Cross-Domain Text-to-SQL Semantic Parsing</b>
<a href="https://arxiv.org/abs/2012.12627">arxiv:2012.12627</a>
&#x1F4C8; 12 <br>
<p>Xi Victoria Lin, Richard Socher, Caiming Xiong</p></summary>
<p>

**Abstract:** We present BRIDGE, a powerful sequential architecture for modeling dependencies between natural language questions and relational databases in cross-DB semantic parsing. BRIDGE represents the question and DB schema in a tagged sequence where a subset of the fields are augmented with cell values mentioned in the question. The hybrid sequence is encoded by BERT with minimal subsequent layers and the text-DB contextualization is realized via the fine-tuned deep attention in BERT. Combined with a pointer-generator decoder with schema-consistency driven search space pruning, BRIDGE attained state-of-the-art performance on popular cross-DB text-to-SQL benchmarks, Spider (71.1\% dev, 67.5\% test with ensemble model) and WikiSQL (92.6\% dev, 91.9\% test). Our analysis shows that BRIDGE effectively captures the desired cross-modal dependencies and has the potential to generalize to more text-DB related tasks. Our implementation is available at \url{https://github.com/salesforce/TabularSemanticParsing}.

</p>
</details>

<details><summary><b>Machine Learning Advances for Time Series Forecasting</b>
<a href="https://arxiv.org/abs/2012.12802">arxiv:2012.12802</a>
&#x1F4C8; 9 <br>
<p>Ricardo P. Masini, Marcelo C. Medeiros, Eduardo F. Mendes</p></summary>
<p>

**Abstract:** In this paper we survey the most recent advances in supervised machine learning and high-dimensional models for time series forecasting. We consider both linear and nonlinear alternatives. Among the linear methods we pay special attention to penalized regressions and ensemble of models. The nonlinear methods considered in the paper include shallow and deep neural networks, in their feed-forward and recurrent versions, and tree-based methods, such as random forests and boosted trees. We also consider ensemble and hybrid models by combining ingredients from different alternatives. Tests for superior predictive ability are briefly reviewed. Finally, we discuss application of machine learning in economics and finance and provide an illustration with high-frequency financial data.

</p>
</details>

<details><summary><b>Deep manifold learning reveals hidden dynamics of proteasome autoregulation</b>
<a href="https://arxiv.org/abs/2012.12854">arxiv:2012.12854</a>
&#x1F4C8; 8 <br>
<p>Zhaolong Wu, Shuwen Zhang, Wei Li Wang, Yinping Ma, Yuanchen Dong, Youdong Mao</p></summary>
<p>

**Abstract:** The 2.5-MDa 26S proteasome maintains proteostasis and regulates myriad cellular processes. How polyubiquitylated substrate interactions regulate proteasome activity is not understood. Here we introduce a deep manifold learning framework, named AlphaCryo4D, which enables atomic-level cryogenic electron microscopy (cryo-EM) reconstructions of nonequilibrium conformational continuum and reconstitutes hidden dynamics of proteasome autoregulation in the act of substrate degradation. AlphaCryo4D integrates 3D deep residual learning with manifold embedding of free-energy landscapes, which directs 3D clustering via an energy-based particle-voting algorithm. In blind assessments using simulated heterogeneous cryo-EM datasets, AlphaCryo4D achieved 3D classification accuracy three times that of conventional method and reconstructed continuous conformational changes of a 130-kDa protein at sub-3-angstrom resolution. By using AlphaCryo4D to analyze a single experimental cryo-EM dataset, we identified 64 conformers of the substrate-bound human 26S proteasome, revealing conformational entanglement of two regulatory particles in the doubly capped holoenzymes and their energetic differences with singly capped ones. Novel ubiquitin-binding sites are discovered on the RPN2, RPN10 and Alpha5 subunits to remodel polyubiquitin chains for deubiquitylation and recycle. Importantly, AlphaCryo4D choreographs single-nucleotide-exchange dynamics of proteasomal AAA-ATPase motor during translocation initiation, which upregulates proteolytic activity by allosterically promoting nucleophilic attack. Our systemic analysis illuminates a grand hierarchical allostery for proteasome autoregulation.

</p>
</details>

<details><summary><b>Compliance Generation for Privacy Documents under GDPR: A Roadmap for Implementing Automation and Machine Learning</b>
<a href="https://arxiv.org/abs/2012.12718">arxiv:2012.12718</a>
&#x1F4C8; 8 <br>
<p>David Restrepo Amariles, Aurore Clément Troussel, Rajaa El Hamdani</p></summary>
<p>

**Abstract:** Most prominent research today addresses compliance with data protection laws through consumer-centric and public-regulatory approaches. We shift this perspective with the Privatech project to focus on corporations and law firms as agents of compliance. To comply with data protection laws, data processors must implement accountability measures to assess and document compliance in relation to both privacy documents and privacy practices. In this paper, we survey, on the one hand, current research on GDPR automation, and on the other hand, the operational challenges corporations face to comply with GDPR, and that may benefit from new forms of automation. We attempt to bridge the gap. We provide a roadmap for compliance assessment and generation by identifying compliance issues, breaking them down into tasks that can be addressed through machine learning and automation, and providing notes about related developments in the Privatech project.

</p>
</details>

<details><summary><b>Self-Supervised Representation Learning for Astronomical Images</b>
<a href="https://arxiv.org/abs/2012.13083">arxiv:2012.13083</a>
&#x1F4C8; 7 <br>
<p>Md Abul Hayat, George Stein, Peter Harrington, Zarija Lukić, Mustafa Mustafa</p></summary>
<p>

**Abstract:** Sky surveys are the largest data generators in astronomy, making automated tools for extracting meaningful scientific information an absolute necessity. We show that, without the need for labels, self-supervised learning recovers representations of sky survey images that are semantically useful for a variety of scientific tasks. These representations can be directly used as features, or fine-tuned, to outperform supervised methods trained only on labeled data. We apply a contrastive learning framework on multi-band galaxy photometry from the Sloan Digital Sky Survey (SDSS) to learn image representations. We then use them for galaxy morphology classification, and fine-tune them for photometric redshift estimation, using labels from the Galaxy Zoo 2 dataset and SDSS spectroscopy. In both downstream tasks, using the same learned representations, we outperform the supervised state-of-the-art results, and we show that our approach can achieve the accuracy of supervised models while using 2-4 times fewer labels for training.

</p>
</details>

<details><summary><b>ProofWriter: Generating Implications, Proofs, and Abductive Statements over Natural Language</b>
<a href="https://arxiv.org/abs/2012.13048">arxiv:2012.13048</a>
&#x1F4C8; 5 <br>
<p>Oyvind Tafjord, Bhavana Dalvi Mishra, Peter Clark</p></summary>
<p>

**Abstract:** Transformers have been shown to emulate logical deduction over natural language theories (logical rules expressed in natural language), reliably assigning true/false labels to candidate implications. However, their ability to generate implications of a theory has not yet been demonstrated, and methods for reconstructing proofs of answers are imperfect. In this work we show that a generative model, called ProofWriter, can reliably generate both implications of a theory and the natural language proof(s) that support them. In particular, iterating a 1-step implication generator results in proofs that are highly reliable, and represent actual model decisions (rather than post-hoc rationalizations). On the RuleTaker dataset, the accuracy of ProofWriter's proofs exceed previous methods by +9% absolute, and in a way that generalizes to proof depths unseen in training and on out-of-domain problems. We also show that generative techniques can perform a type of abduction with high precision: Given a theory and an unprovable conclusion, identify a missing fact that allows the conclusion to be proved, along with a proof. These results significantly improve the viability of neural methods for systematically reasoning over natural language.

</p>
</details>

<details><summary><b>Multiclass Spinal Cord Tumor Segmentation on MRI with Deep Learning</b>
<a href="https://arxiv.org/abs/2012.12820">arxiv:2012.12820</a>
&#x1F4C8; 5 <br>
<p>Andreanne Lemay, Charley Gros, Zhizheng Zhuo, Jie Zhang, Yunyun Duan, Julien Cohen-Adad, Yaou Liu</p></summary>
<p>

**Abstract:** Spinal cord tumors lead to neurological morbidity and mortality. Being able to obtain morphometric quantification (size, location, growth rate) of the tumor, edema, and cavity can result in improved monitoring and treatment planning. Such quantification requires the segmentation of these structures into three separate classes. However, manual segmentation of 3-dimensional structures is time-consuming and tedious, motivating the development of automated methods. Here, we tailor a model adapted to the spinal cord tumor segmentation task. Data were obtained from 343 patients using gadolinium-enhanced T1-weighted and T2-weighted MRI scans with cervical, thoracic, and/or lumbar coverage. The dataset includes the three most common intramedullary spinal cord tumor types: astrocytomas, ependymomas, and hemangioblastomas. The proposed approach is a cascaded architecture with U-Net-based models that segments tumors in a two-stage process: locate and label. The model first finds the spinal cord and generates bounding box coordinates. The images are cropped according to this output, leading to a reduced field of view, which mitigates class imbalance. The tumor is then segmented. The segmentation of the tumor, cavity, and edema (as a single class) reached 76.7 $\pm$ 1.5% of Dice score and the segmentation of tumors alone reached 61.8 $\pm$ 4.0% Dice score. The true positive detection rate was above 87% for tumor, edema, and cavity. To the best of our knowledge, this is the first fully automatic deep learning model for spinal cord tumor segmentation. The multiclass segmentation pipeline is available in the Spinal Cord Toolbox (https://spinalcordtoolbox.com/). It can be run with custom data on a regular computer within seconds.

</p>
</details>

<details><summary><b>Regret Bound Balancing and Elimination for Model Selection in Bandits and RL</b>
<a href="https://arxiv.org/abs/2012.13045">arxiv:2012.13045</a>
&#x1F4C8; 4 <br>
<p>Aldo Pacchiano, Christoph Dann, Claudio Gentile, Peter Bartlett</p></summary>
<p>

**Abstract:** We propose a simple model selection approach for algorithms in stochastic bandit and reinforcement learning problems. As opposed to prior work that (implicitly) assumes knowledge of the optimal regret, we only require that each base algorithm comes with a candidate regret bound that may or may not hold during all rounds. In each round, our approach plays a base algorithm to keep the candidate regret bounds of all remaining base algorithms balanced, and eliminates algorithms that violate their candidate bound. We prove that the total regret of this approach is bounded by the best valid candidate regret bound times a multiplicative factor. This factor is reasonably small in several applications, including linear bandits and MDPs with nested function classes, linear bandits with unknown misspecification, and LinUCB applied to linear bandits with different confidence parameters. We further show that, under a suitable gap-assumption, this factor only scales with the number of base algorithms and not their complexity when the number of rounds is large enough. Finally, unlike recent efforts in model selection for linear stochastic bandits, our approach is versatile enough to also cover cases where the context information is generated by an adversarial environment, rather than a stochastic one.

</p>
</details>

<details><summary><b>Warping of Radar Data into Camera Image for Cross-Modal Supervision in Automotive Applications</b>
<a href="https://arxiv.org/abs/2012.12809">arxiv:2012.12809</a>
&#x1F4C8; 4 <br>
<p>Christopher Grimm, Tai Fei, Ernst Warsitz, Ridha Farhoud, Tobias Breddermann, Reinhold Haeb-Umbach</p></summary>
<p>

**Abstract:** In this paper, we present a novel framework to project automotive radar range-Doppler (RD) spectrum into camera image. The utilized warping operation is designed to be fully differentiable, which allows error backpropagation through the operation. This enables the training of neural networks (NN) operating exclusively on RD spectrum by utilizing labels provided from camera vision models. As the warping operation relies on accurate scene flow, additionally, we present a novel scene flow estimation algorithm fed from camera, lidar and radar, enabling us to improve the accuracy of the warping operation. We demonstrate the framework in multiple applications like direction-of-arrival (DoA) estimation, target detection, semantic segmentation and estimation of radar power from camera data. Extensive evaluations have been carried out for the DoA application and suggest superior quality for NN based estimators compared to classical estimators. The novel scene flow estimation approach is benchmarked against state-of-the-art scene flow algorithms and outperforms them by roughly a third.

</p>
</details>

<details><summary><b>Multi-Modality Cut and Paste for 3D Object Detection</b>
<a href="https://arxiv.org/abs/2012.12741">arxiv:2012.12741</a>
&#x1F4C8; 4 <br>
<p>Wenwei Zhang, Zhe Wang, Chen Change Loy</p></summary>
<p>

**Abstract:** Three-dimensional (3D) object detection is essential in autonomous driving. There are observations that multi-modality methods based on both point cloud and imagery features perform only marginally better or sometimes worse than approaches that solely use single-modality point cloud. This paper investigates the reason behind this counter-intuitive phenomenon through a careful comparison between augmentation techniques used by single modality and multi-modality methods. We found that existing augmentations practiced in single-modality detection are equally useful for multi-modality detection. Then we further present a new multi-modality augmentation approach, Multi-mOdality Cut and pAste (MoCa). MoCa boosts detection performance by cutting point cloud and imagery patches of ground-truth objects and pasting them into different scenes in a consistent manner while avoiding collision between objects. We also explore beneficial architecture design and optimization practices in implementing a good multi-modality detector. Without using ensemble of detectors, our multi-modality detector achieves new state-of-the-art performance on nuScenes dataset and competitive performance on KITTI 3D benchmark. Our method also wins the best PKL award in the 3rd nuScenes detection challenge. Code and models will be released at https://github.com/open-mmlab/mmdetection3d.

</p>
</details>

<details><summary><b>AutonoML: Towards an Integrated Framework for Autonomous Machine Learning</b>
<a href="https://arxiv.org/abs/2012.12600">arxiv:2012.12600</a>
&#x1F4C8; 4 <br>
<p>David Jacob Kedziora, Katarzyna Musial, Bogdan Gabrys</p></summary>
<p>

**Abstract:** Over the last decade, the long-running endeavour to automate high-level processes in machine learning (ML) has risen to mainstream prominence, stimulated by advances in optimisation techniques and their impact on selecting ML models/algorithms. Central to this drive is the appeal of engineering a computational system that both discovers and deploys high-performance solutions to arbitrary ML problems with minimal human interaction. Beyond this, an even loftier goal is the pursuit of autonomy, which describes the capability of the system to independently adjust an ML solution over a lifetime of changing contexts. However, these ambitions are unlikely to be achieved in a robust manner without the broader synthesis of various mechanisms and theoretical frameworks, which, at the present time, remain scattered across numerous research threads. Accordingly, this review seeks to motivate a more expansive perspective on what constitutes an automated/autonomous ML system, alongside consideration of how best to consolidate those elements. In doing so, we survey developments in the following research areas: hyperparameter optimisation, multi-component models, neural architecture search, automated feature engineering, meta-learning, multi-level ensembling, dynamic adaptation, multi-objective evaluation, resource constraints, flexible user involvement, and the principles of generalisation. We also develop a conceptual framework throughout the review, augmented by each topic, to illustrate one possible way of fusing high-level mechanisms into an autonomous ML system. Ultimately, we conclude that the notion of architectural integration deserves more discussion, without which the field of automated ML risks stifling both its technical advantages and general uptake.

</p>
</details>

<details><summary><b>Wheel-Rail Interface Condition Estimation (W-RICE)</b>
<a href="https://arxiv.org/abs/2012.13096">arxiv:2012.13096</a>
&#x1F4C8; 3 <br>
<p>Sundar Shrestha, Anand Koirala, Maksym Spiryagin, Qing Wu</p></summary>
<p>

**Abstract:** The surface roughness between the wheel and rail has a huge influence on rolling noise level. The presence of the third body such as frost or grease at wheel-rail interface contributes towards change in adhesion coefficient resulting in the generation of noise at various levels. Therefore, it is possible to estimate adhesion conditions between the wheel and rail from the analysis of noise patterns originating from wheel-rail interaction. In this study, a new approach to estimate adhesion condition is proposed which takes rolling noise as input.

</p>
</details>

<details><summary><b>Multi-modal Identification of State-Sponsored Propaganda on Social Media</b>
<a href="https://arxiv.org/abs/2012.13042">arxiv:2012.13042</a>
&#x1F4C8; 3 <br>
<p>Xiaobo Guo, Soroush Vosoughi</p></summary>
<p>

**Abstract:** The prevalence of state-sponsored propaganda on the Internet has become a cause for concern in the recent years. While much effort has been made to identify state-sponsored Internet propaganda, the problem remains far from being solved because the ambiguous definition of propaganda leads to unreliable data labelling, and the huge amount of potential predictive features causes the models to be inexplicable. This paper is the first attempt to build a balanced dataset for this task. The dataset is comprised of propaganda by three different organizations across two time periods. A multi-model framework for detecting propaganda messages solely based on the visual and textual content is proposed which achieves a promising performance on detecting propaganda by the three organizations both for the same time period (training and testing on data from the same time period) (F1=0.869) and for different time periods (training on past, testing on future) (F1=0.697). To reduce the influence of false positive predictions, we change the threshold to test the relationship between the false positive and true positive rates and provide explanations for the predictions made by our models with visualization tools to enhance the interpretability of our framework. Our new dataset and general framework provide a strong benchmark for the task of identifying state-sponsored Internet propaganda and point out a potential path for future work on this task.

</p>
</details>

<details><summary><b>Private-Shared Disentangled Multimodal VAE for Learning of Hybrid Latent Representations</b>
<a href="https://arxiv.org/abs/2012.13024">arxiv:2012.13024</a>
&#x1F4C8; 3 <br>
<p>Mihee Lee, Vladimir Pavlovic</p></summary>
<p>

**Abstract:** Multi-modal generative models represent an important family of deep models, whose goal is to facilitate representation learning on data with multiple views or modalities. However, current deep multi-modal models focus on the inference of shared representations, while neglecting the important private aspects of data within individual modalities. In this paper, we introduce a disentangled multi-modal variational autoencoder (DMVAE) that utilizes disentangled VAE strategy to separate the private and shared latent spaces of multiple modalities. We specifically consider the instance where the latent factor may be of both continuous and discrete nature, leading to the family of general hybrid DMVAE models. We demonstrate the utility of DMVAE on a semi-supervised learning task, where one of the modalities contains partial data labels, both relevant and irrelevant to the other modality. Our experiments on several benchmarks indicate the importance of the private-shared disentanglement as well as the hybrid latent representation.

</p>
</details>

<details><summary><b>Low-latency Perception in Off-Road Dynamical Low Visibility Environments</b>
<a href="https://arxiv.org/abs/2012.13014">arxiv:2012.13014</a>
&#x1F4C8; 3 <br>
<p>Nelson Alves, Marco Ruiz, Marco Reis, Tiago Cajahyba, Davi Oliveira, Ana Barreto, Eduardo F. Simas Filho, Wagner L. A. de Oliveira, Leizer Schnitman, Roberto L. S. Monteiro</p></summary>
<p>

**Abstract:** This work proposes a perception system for autonomous vehicles and advanced driver assistance specialized on unpaved roads and off-road environments. In this research, the authors have investigated the behavior of Deep Learning algorithms applied to semantic segmentation of off-road environments and unpaved roads under differents adverse conditions of visibility. Almost 12,000 images of different unpaved and off-road environments were collected and labeled. It was assembled an off-road proving ground exclusively for its development. The proposed dataset also contains many adverse situations such as rain, dust, and low light. To develop the system, we have used convolutional neural networks trained to segment obstacles and areas where the car can pass through. We developed a Configurable Modular Segmentation Network (CMSNet) framework to help create different architectures arrangements and test them on the proposed dataset. Besides, we also have ported some CMSNet configurations by removing and fusing many layers using TensorRT, C++, and CUDA to achieve embedded real-time inference and allow field tests. The main contributions of this work are: a new dataset for unpaved roads and off-roads environments containing many adverse conditions such as night, rain, and dust; a CMSNet framework; an investigation regarding the feasibility of applying deep learning to detect region where the vehicle can pass through when there is no clear boundary of the track; a study of how our proposed segmentation algorithms behave in different severity levels of visibility impairment; and an evaluation of field tests carried out with semantic segmentation architectures ported for real-time inference.

</p>
</details>

<details><summary><b>Learning by Self-Explanation, with Application to Neural Architecture Search</b>
<a href="https://arxiv.org/abs/2012.12899">arxiv:2012.12899</a>
&#x1F4C8; 3 <br>
<p>Ramtin Hosseini, Pengtao Xie</p></summary>
<p>

**Abstract:** Learning by self-explanation, where students explain a learned topic to themselves for deepening their understanding of this topic, is a broadly used methodology in human learning and shows great effectiveness in improving learning outcome. We are interested in investigating whether this powerful learning technique can be borrowed from humans to improve the learning abilities of machines. We propose a novel learning approach called learning by self-explanation (LeaSE). In our approach, an explainer model improves its learning ability by trying to clearly explain to an audience model regarding how a prediction outcome is made. We propose a multi-level optimization framework to formulate LeaSE which involves four stages of learning: explainer learns; explainer explains; audience learns; explainer and audience validate themselves. We develop an efficient algorithm to solve the LeaSE problem. We apply our approach to neural architecture search on CIFAR-100, CIFAR-10, and ImageNet. The results demonstrate the effectiveness of our method.

</p>
</details>

<details><summary><b>A Multimodal Framework for the Detection of Hateful Memes</b>
<a href="https://arxiv.org/abs/2012.12871">arxiv:2012.12871</a>
&#x1F4C8; 3 <br>
<p>Phillip Lippe, Nithin Holla, Shantanu Chandra, Santhosh Rajamanickam, Georgios Antoniou, Ekaterina Shutova, Helen Yannakoudakis</p></summary>
<p>

**Abstract:** An increasingly common expression of online hate speech is multimodal in nature and comes in the form of memes. Designing systems to automatically detect hateful content is of paramount importance if we are to mitigate its undesirable effects on the society at large. The detection of multimodal hate speech is an intrinsically difficult and open problem: memes convey a message using both images and text and, hence, require multimodal reasoning and joint visual and language understanding. In this work, we seek to advance this line of research and develop a multimodal framework for the detection of hateful memes. We improve the performance of existing multimodal approaches beyond simple fine-tuning and, among others, show the effectiveness of upsampling of contrastive examples to encourage multimodality and ensemble learning based on cross-validation to improve robustness. We furthermore analyze model misclassifications and discuss a number of hypothesis-driven augmentations and their effects on performance, presenting important implications for future research in the field. Our best approach comprises an ensemble of UNITER-based models and achieves an AUROC score of 80.53, placing us 4th on phase 2 of the 2020 Hateful Memes Challenge organized by Facebook.

</p>
</details>

<details><summary><b>EmotionGIF-IITP-AINLPML: Ensemble-based Automated Deep Neural System for predicting category(ies) of a GIF response</b>
<a href="https://arxiv.org/abs/2012.12756">arxiv:2012.12756</a>
&#x1F4C8; 3 <br>
<p>Soumitra Ghosh, Arkaprava Roy, Asif Ekbal, Pushpak Bhattacharyya</p></summary>
<p>

**Abstract:** In this paper, we describe the systems submitted by our IITP-AINLPML team in the shared task of SocialNLP 2020, EmotionGIF 2020, on predicting the category(ies) of a GIF response for a given unlabelled tweet. For the round 1 phase of the task, we propose an attention-based Bi-directional GRU network trained on both the tweet (text) and their replies (text wherever available) and the given category(ies) for its GIF response. In the round 2 phase, we build several deep neural-based classifiers for the task and report the final predictions through a majority voting based ensemble technique. Our proposed models attain the best Mean Recall (MR) scores of 52.92% and 53.80% in round 1 and round 2, respectively.

</p>
</details>

<details><summary><b>Gradient-free quantum optimization on NISQ devices</b>
<a href="https://arxiv.org/abs/2012.13453">arxiv:2012.13453</a>
&#x1F4C8; 2 <br>
<p>L. Franken, B. Georgiev, S. Muecke, M. Wolter, N. Piatkowski, C. Bauckhage</p></summary>
<p>

**Abstract:** Variational Quantum Eigensolvers (VQEs) have recently attracted considerable attention. Yet, in practice, they still suffer from the efforts for estimating cost function gradients for large parameter sets or resource-demanding reinforcement strategies. Here, we therefore consider recent advances in weight-agnostic learning and propose a strategy that addresses the trade-off between finding appropriate circuit architectures and parameter tuning. We investigate the use of NEAT-inspired algorithms which evaluate circuits via genetic competition and thus circumvent issues due to exceeding numbers of parameters. Our methods are tested both via simulation and on real quantum hardware and are used to solve the transverse Ising Hamiltonian and the Sherrington-Kirkpatrick spin model.

</p>
</details>

<details><summary><b>Cooperative Policy Learning with Pre-trained Heterogeneous Observation Representations</b>
<a href="https://arxiv.org/abs/2012.13099">arxiv:2012.13099</a>
&#x1F4C8; 2 <br>
<p>Wenlei Shi, Xinran Wei, Jia Zhang, Xiaoyuan Ni, Arthur Jiang, Jiang Bian, Tie-Yan Liu</p></summary>
<p>

**Abstract:** Multi-agent reinforcement learning (MARL) has been increasingly explored to learn the cooperative policy towards maximizing a certain global reward. Many existing studies take advantage of graph neural networks (GNN) in MARL to propagate critical collaborative information over the interaction graph, built upon inter-connected agents. Nevertheless, the vanilla GNN approach yields substantial defects in dealing with complex real-world scenarios since the generic message passing mechanism is ineffective between heterogeneous vertices and, moreover, simple message aggregation functions are incapable of accurately modeling the combinational interactions from multiple neighbors. While adopting complex GNN models with more informative message passing and aggregation mechanisms can obviously benefit heterogeneous vertex representations and cooperative policy learning, it could, on the other hand, increase the training difficulty of MARL and demand more intense and direct reward signals compared to the original global reward. To address these challenges, we propose a new cooperative learning framework with pre-trained heterogeneous observation representations. Particularly, we employ an encoder-decoder based graph attention to learn the intricate interactions and heterogeneous representations that can be more easily leveraged by MARL. Moreover, we design a pre-training with local actor-critic algorithm to ease the difficulty in cooperative policy learning. Extensive experiments over real-world scenarios demonstrate that our new approach can significantly outperform existing MARL baselines as well as operational research solutions that are widely-used in industry.

</p>
</details>

<details><summary><b>High-Dimensional Bayesian Optimization via Tree-Structured Additive Models</b>
<a href="https://arxiv.org/abs/2012.13088">arxiv:2012.13088</a>
&#x1F4C8; 2 <br>
<p>Eric Han, Ishank Arora, Jonathan Scarlett</p></summary>
<p>

**Abstract:** Bayesian Optimization (BO) has shown significant success in tackling expensive low-dimensional black-box optimization problems. Many optimization problems of interest are high-dimensional, and scaling BO to such settings remains an important challenge. In this paper, we consider generalized additive models in which low-dimensional functions with overlapping subsets of variables are composed to model a high-dimensional target function. Our goal is to lower the computational resources required and facilitate faster model learning by reducing the model complexity while retaining the sample-efficiency of existing methods. Specifically, we constrain the underlying dependency graphs to tree structures in order to facilitate both the structure learning and optimization of the acquisition function. For the former, we propose a hybrid graph learning algorithm based on Gibbs sampling and mutation. In addition, we propose a novel zooming-based algorithm that permits generalized additive models to be employed more efficiently in the case of continuous domains. We demonstrate and discuss the efficacy of our approach via a range of experiments on synthetic functions and real-world datasets.

</p>
</details>

<details><summary><b>White matter hyperintensities volume and cognition: Assessment of a deep learning based lesion detection and quantification algorithm on the Alzheimers Disease Neuroimaging Initiative</b>
<a href="https://arxiv.org/abs/2012.13059">arxiv:2012.13059</a>
&#x1F4C8; 2 <br>
<p>Lavanya Umapathy, Gloria Guzman Perez-Carillo, Blair Winegar, Srinivasan Vedantham, Maria Altbach, Ali Bilgin</p></summary>
<p>

**Abstract:** The relationship between cognition and white matter hyperintensities (WMH) volumes often depends on the accuracy of the lesion segmentation algorithm used. As such, accurate detection and quantification of WMH is of great interest. Here, we use a deep learning-based WMH segmentation algorithm, StackGen-Net, to detect and quantify WMH on 3D FLAIR volumes from ADNI. We used a subset of subjects (n=20) and obtained manual WMH segmentations by an experienced neuro-radiologist to demonstrate the accuracy of our algorithm. On a larger cohort of subjects (n=290), we observed that larger WMH volumes correlated with worse performance on executive function (P=.004), memory (P=.01), and language (P=.005).

</p>
</details>

<details><summary><b>Causal Inference Using Linear Time-Varying Filters with Additive Noise</b>
<a href="https://arxiv.org/abs/2012.13025">arxiv:2012.13025</a>
&#x1F4C8; 2 <br>
<p>Kang Du, Yu Xiang</p></summary>
<p>

**Abstract:** Causal inference using the restricted structural causal model framework hinges largely on the asymmetry between cause and effect from the data generating mechanisms. For linear models and additive noise models, the asymmetry arises from non-Gaussianity or non-linearity, respectively. This methodology can be adapted to stationary time series, however, inferring causal relationships from non-stationary time series remains a challenging task. In the work, we focus on slowly-varying nonstationary processes and propose to break the symmetry by exploiting the nonstationarity of the data. Our main theoretical result shows that causal direction is identifiable in generic cases when cause and effect are connected via a time-varying filter. We propose a causal discovery procedure by leveraging powerful estimates of the bivariate evolutionary spectra. Both synthetic and real-world data simulations that involve high-order and nonsmooth filters are provided to demonstrate the effectiveness of our proposed methodology.

</p>
</details>

<details><summary><b>Representing Partial Programs with Blended Abstract Semantics</b>
<a href="https://arxiv.org/abs/2012.12964">arxiv:2012.12964</a>
&#x1F4C8; 2 <br>
<p>Maxwell Nye, Yewen Pu, Matthew Bowers, Jacob Andreas, Joshua B. Tenenbaum, Armando Solar-Lezama</p></summary>
<p>

**Abstract:** Synthesizing programs from examples requires searching over a vast, combinatorial space of possible programs. In this search process, a key challenge is representing the behavior of a partially written program before it can be executed, to judge if it is on the right track and predict where to search next. We introduce a general technique for representing partially written programs in a program synthesis engine. We take inspiration from the technique of abstract interpretation, in which an approximate execution model is used to determine if an unfinished program will eventually satisfy a goal specification. Here we learn an approximate execution model implemented as a modular neural network. By constructing compositional program representations that implicitly encode the interpretation semantics of the underlying programming language, we can represent partial programs using a flexible combination of concrete execution state and learned neural representations, using the learned approximate semantics when concrete semantics are not known (in unfinished parts of the program). We show that these hybrid neuro-symbolic representations enable execution-guided synthesizers to use more powerful language constructs, such as loops and higher-order functions, and can be used to synthesize programs more accurately for a given search budget than pure neural approaches in several domains.

</p>
</details>

<details><summary><b>On Using Classification Datasets to Evaluate Graph Outlier Detection: Peculiar Observations and New Insights</b>
<a href="https://arxiv.org/abs/2012.12931">arxiv:2012.12931</a>
&#x1F4C8; 2 <br>
<p>Lingxiao Zhao, Leman Akoglu</p></summary>
<p>

**Abstract:** It is common practice of the outlier mining community to repurpose classification datasets toward evaluating various detection models. To that end, often a binary classification dataset is used, where samples from (typically, the larger) one of the classes is designated as the inlier samples, and the other class is substantially down-sampled to create the (ground-truth) outlier samples. In this study, we identify an intriguing issue with repurposing graph classification datasets for graph outlier detection in this manner. Surprisingly, the detection performance of outlier models depends significantly on which class is down-sampled; put differently, accuracy often flips from high to low depending on which of the classes is down-sampled to represent the outlier samples. The problem is notably exacerbated particularly for a certain family of propagation based outlier detection models. Through careful analysis, we show that this issue mainly stems from disparate within-class sample similarity - which is amplified by various propagation based models - that impacts key characteristics of inlier/outlier distributions and indirectly, the difficulty of the outlier detection task and hence performance outcomes. With this study, we aim to draw attention to this (to our knowledge) previously-unnoticed issue, as it has implications for fair and effective evaluation of detection models, and hope that it will motivate the design of better evaluation benchmarks for outlier detection. Finally, we discuss the possibly overarching implications of using propagation based models on datasets with disparate within-class sample similarity beyond outlier detection, specifically for graph classification and graph-level clustering tasks.

</p>
</details>

<details><summary><b>Lattice gauge equivariant convolutional neural networks</b>
<a href="https://arxiv.org/abs/2012.12901">arxiv:2012.12901</a>
&#x1F4C8; 2 <br>
<p>Matteo Favoni, Andreas Ipp, David I. Müller, Daniel Schuh</p></summary>
<p>

**Abstract:** We propose Lattice gauge equivariant Convolutional Neural Networks (L-CNNs) for generic machine learning applications on lattice gauge theoretical problems. At the heart of this network structure is a novel convolutional layer that preserves gauge equivariance while forming arbitrarily shaped Wilson loops in successive bilinear layers. Together with topological information, for example from Polyakov loops, such a network can in principle approximate any gauge covariant function on the lattice. We demonstrate that L-CNNs can learn and generalize gauge invariant quantities that traditional convolutional neural networks are incapable of finding.

</p>
</details>

<details><summary><b>Noisy Labels Can Induce Good Representations</b>
<a href="https://arxiv.org/abs/2012.12896">arxiv:2012.12896</a>
&#x1F4C8; 2 <br>
<p>Jingling Li, Mozhi Zhang, Keyulu Xu, John P. Dickerson, Jimmy Ba</p></summary>
<p>

**Abstract:** The current success of deep learning depends on large-scale labeled datasets. In practice, high-quality annotations are expensive to collect, but noisy annotations are more affordable. Previous works report mixed empirical results when training with noisy labels: neural networks can easily memorize random labels, but they can also generalize from noisy labels. To explain this puzzle, we study how architecture affects learning with noisy labels. We observe that if an architecture "suits" the task, training with noisy labels can induce useful hidden representations, even when the model generalizes poorly; i.e., the last few layers of the model are more negatively affected by noisy labels. This finding leads to a simple method to improve models trained on noisy labels: replacing the final dense layers with a linear model, whose weights are learned from a small set of clean data. We empirically validate our findings across three architectures (Convolutional Neural Networks, Graph Neural Networks, and Multi-Layer Perceptrons) and two domains (graph algorithmic tasks and image classification). Furthermore, we achieve state-of-the-art results on image classification benchmarks by combining our method with existing approaches on noisy label training.

</p>
</details>

<details><summary><b>All That Glitters Is Not Gold: Towards Process Discovery Techniques with Guarantees</b>
<a href="https://arxiv.org/abs/2012.12764">arxiv:2012.12764</a>
&#x1F4C8; 2 <br>
<p>Jan Martijn E. M. van der Werf, Artem Polyvyanyy, Bart R. van Wensveen, Matthieu Brinkhuis, Hajo A. Reijers</p></summary>
<p>

**Abstract:** The aim of a process discovery algorithm is to construct from event data a process model that describes the underlying, real-world process well. Intuitively, the better the quality of the event data, the better the quality of the model that is discovered. However, existing process discovery algorithms do not guarantee this relationship. We demonstrate this by using a range of quality measures for both event data and discovered process models. This paper is a call to the community of IS engineers to complement their process discovery algorithms with properties that relate qualities of their inputs to those of their outputs. To this end, we distinguish four incremental stages for the development of such algorithms, along with concrete guidelines for the formulation of relevant properties and experimental validation. We will also use these stages to reflect on the state of the art, which shows the need to move forward in our thinking about algorithmic process discovery.

</p>
</details>

<details><summary><b>The Less Intelligent the Elements, the More Intelligent the Whole. Or, Possibly Not?</b>
<a href="https://arxiv.org/abs/2012.12689">arxiv:2012.12689</a>
&#x1F4C8; 2 <br>
<p>Guido Fioretti, Andrea Policarpi</p></summary>
<p>

**Abstract:** We dare to make use of a possible analogy between neurons in a brain and people in society, asking ourselves whether individual intelligence is necessary in order to collective wisdom to emerge and, most importantly, what sort of individual intelligence is conducive of greater collective wisdom. We review insights and findings from connectionism, agent-based modeling, group psychology, economics and physics, casting them in terms of changing structure of the system's Lyapunov function. Finally, we apply these insights to the sort and degrees of intelligence of preys and predators in the Lotka-Volterra model, explaining why certain individual understandings lead to co-existence of the two species whereas other usages of their individual intelligence cause global extinction.

</p>
</details>

<details><summary><b>Automated Lay Language Summarization of Biomedical Scientific Reviews</b>
<a href="https://arxiv.org/abs/2012.12573">arxiv:2012.12573</a>
&#x1F4C8; 2 <br>
<p>Yue Guo, Wei Qiu, Yizhong Wang, Trevor Cohen</p></summary>
<p>

**Abstract:** Health literacy has emerged as a crucial factor in making appropriate health decisions and ensuring treatment outcomes. However, medical jargon and the complex structure of professional language in this domain make health information especially hard to interpret. Thus, there is an urgent unmet need for automated methods to enhance the accessibility of the biomedical literature to the general population. This problem can be framed as a type of translation problem between the language of healthcare professionals, and that of the general public. In this paper, we introduce the novel task of automated generation of lay language summaries of biomedical scientific reviews, and construct a dataset to support the development and evaluation of automated methods through which to enhance the accessibility of the biomedical literature. We conduct analyses of the various challenges in solving this task, including not only summarization of the key points but also explanation of background knowledge and simplification of professional language. We experiment with state-of-the-art summarization models as well as several data augmentation techniques, and evaluate their performance using both automated metrics and human assessment. Results indicate that automatically generated summaries produced using contemporary neural architectures can achieve promising quality and readability as compared with reference summaries developed for the lay public by experts (best ROUGE-L of 50.24 and Flesch-Kincaid readability score of 13.30). We also discuss the limitations of the current attempt, providing insights and directions for future work.

</p>
</details>

<details><summary><b>ICMSC: Intra- and Cross-modality Semantic Consistency for Unsupervised Domain Adaptation on Hip Joint Bone Segmentation</b>
<a href="https://arxiv.org/abs/2012.12570">arxiv:2012.12570</a>
&#x1F4C8; 2 <br>
<p>Guodong Zeng, Till D. Lerch, Florian Schmaranzer, Guoyan Zheng, Juergen Burger, Kate Gerber, Moritz Tannast, Klaus Siebenrock, Nicolas Gerber</p></summary>
<p>

**Abstract:** Unsupervised domain adaptation (UDA) for cross-modality medical image segmentation has shown great progress by domain-invariant feature learning or image appearance translation. Adapted feature learning usually cannot detect domain shifts at the pixel level and is not able to achieve good results in dense semantic segmentation tasks. Image appearance translation, e.g. CycleGAN, translates images into different styles with good appearance, despite its population, its semantic consistency is hardly to maintain and results in poor cross-modality segmentation. In this paper, we propose intra- and cross-modality semantic consistency (ICMSC) for UDA and our key insight is that the segmentation of synthesised images in different styles should be consistent. Specifically, our model consists of an image translation module and a domain-specific segmentation module. The image translation module is a standard CycleGAN, while the segmentation module contains two domain-specific segmentation networks. The intra-modality semantic consistency (IMSC) forces the reconstructed image after a cycle to be segmented in the same way as the original input image, while the cross-modality semantic consistency (CMSC) encourages the synthesized images after translation to be segmented exactly the same as before translation. Comprehensive experimental results on cross-modality hip joint bone segmentation show the effectiveness of our proposed method, which achieves an average DICE of 81.61% on the acetabulum and 88.16% on the proximal femur, outperforming other state-of-the-art methods. It is worth to note that without UDA, a model trained on CT for hip joint bone segmentation is non-transferable to MRI and has almost zero-DICE segmentation.

</p>
</details>

<details><summary><b>Evolving Neural Architecture Using One Shot Model</b>
<a href="https://arxiv.org/abs/2012.12540">arxiv:2012.12540</a>
&#x1F4C8; 2 <br>
<p>Nilotpal Sinha, Kuan-Wen Chen</p></summary>
<p>

**Abstract:** Neural Architecture Search (NAS) is emerging as a new research direction which has the potential to replace the hand-crafted neural architectures designed for specific tasks. Previous evolution based architecture search requires high computational resources resulting in high search time. In this work, we propose a novel way of applying a simple genetic algorithm to the NAS problem called EvNAS (Evolving Neural Architecture using One Shot Model) which reduces the search time significantly while still achieving better result than previous evolution based methods. The architectures are represented by using the architecture parameter of the one shot model which results in the weight sharing among the architectures for a given population of architectures and also weight inheritance from one generation to the next generation of architectures. We propose a decoding technique for the architecture parameter which is used to divert majority of the gradient information towards the given architecture and is also used for improving the performance prediction of the given architecture from the one shot model during the search process. Furthermore, we use the accuracy of the partially trained architecture on the validation data as a prediction of its fitness in order to reduce the search time. EvNAS searches for the architecture on the proxy dataset i.e. CIFAR-10 for 4.4 GPU day on a single GPU and achieves top-1 test error of 2.47% with 3.63M parameters which is then transferred to CIFAR-100 and ImageNet achieving top-1 error of 16.37% and top-5 error of 7.4% respectively. All of these results show the potential of evolutionary methods in solving the architecture search problem.

</p>
</details>

<details><summary><b>GAHNE: Graph-Aggregated Heterogeneous Network Embedding</b>
<a href="https://arxiv.org/abs/2012.12517">arxiv:2012.12517</a>
&#x1F4C8; 2 <br>
<p>Xiaohe Li, Lijie Wen, Chen Qian, Jianmin Wang</p></summary>
<p>

**Abstract:** The real-world networks often compose of different types of nodes and edges with rich semantics, widely known as heterogeneous information network (HIN). Heterogeneous network embedding aims to embed nodes into low-dimensional vectors which capture rich intrinsic information of heterogeneous networks. However, existing models either depend on manually designing meta-paths, ignore mutual effects between different semantics, or omit some aspects of information from global networks. To address these limitations, we propose a novel Graph-Aggregated Heterogeneous Network Embedding (GAHNE), which is designed to extract the semantics of HINs as comprehensively as possible to improve the results of downstream tasks based on graph convolutional neural networks. In GAHNE model, we develop several mechanisms that can aggregate semantic representations from different single-type sub-networks as well as fuse the global information into final embeddings. Extensive experiments on three real-world HIN datasets show that our proposed model consistently outperforms the existing state-of-the-art methods.

</p>
</details>

<details><summary><b>Small-Group Learning, with Application to Neural Architecture Search</b>
<a href="https://arxiv.org/abs/2012.12502">arxiv:2012.12502</a>
&#x1F4C8; 2 <br>
<p>Xuefeng Du, Pengtao Xie</p></summary>
<p>

**Abstract:** Small-group learning is a broadly used methodology in human learning and shows great effectiveness in improving learning outcomes: a small group of students work together towards the same learning objective, where they express their understanding of a topic to their peers, compare their ideas, and help each other to trouble-shoot problems. We are interested in investigating whether this powerful learning technique can be borrowed from humans to improve the learning abilities of machines. We propose a novel learning approach called small-group learning (SGL). In our approach, each learner uses its intermediately trained model to generate a pseudo-labeled dataset and re-trains its model using pseudo-labeled datasets generated by other learners. We propose a multi-level optimization framework to formulate SGL which involves three learning stages: learners train their network weights independently; learners train their network weights collaboratively via mutual pseudo-labeling; learners improve their architectures by minimizing validation losses. We develop an efficient algorithm to solve the SGL problem. We apply our approach to neural architecture search and achieve significant improvement on CIFAR-100, CIFAR-10, and ImageNet.

</p>
</details>

<details><summary><b>Probabilistic electric load forecasting through Bayesian Mixture Density Networks</b>
<a href="https://arxiv.org/abs/2012.14389">arxiv:2012.14389</a>
&#x1F4C8; 1 <br>
<p>Alessandro Brusaferri, Matteo Matteucci, Stefano Spinelli, Andrea Vitali</p></summary>
<p>

**Abstract:** Probabilistic load forecasting (PLF) is a key component in the extended tool-chain required for efficient management of smart energy grids. Neural networks are widely considered to achieve improved prediction performances, supporting highly flexible mappings of complex relationships between the target and the conditioning variables set. However, obtaining comprehensive predictive uncertainties from such black-box models is still a challenging and unsolved problem. In this work, we propose a novel PLF approach, framed on Bayesian Mixture Density Networks. Both aleatoric and epistemic uncertainty sources are encompassed within the model predictions, inferring general conditional densities, depending on the input features, within an end-to-end training framework. To achieve reliable and computationally scalable estimators of the posterior distributions, both Mean Field variational inference and deep ensembles are integrated. Experiments have been performed on household short-term load forecasting tasks, showing the capability of the proposed method to achieve robust performances in different operating conditions.

</p>
</details>

<details><summary><b>On Calibration of Scene-Text Recognition Models</b>
<a href="https://arxiv.org/abs/2012.12643">arxiv:2012.12643</a>
&#x1F4C8; 1 <br>
<p>Ron Slossberg, Oron Anschel, Amir Markovitz, Ron Litman, Aviad Aberdam, Shahar Tsiper, Shai Mazor, Jon Wu, R. Manmatha</p></summary>
<p>

**Abstract:** In this work, we study the problem of word-level confidence calibration for scene-text recognition (STR). Although the topic of confidence calibration has been an active research area for the last several decades, the case of structured and sequence prediction calibration has been scarcely explored. We analyze several recent STR methods and show that they are consistently overconfident. We then focus on the calibration of STR models on the word rather than the character level. In particular, we demonstrate that for attention based decoders, calibration of individual character predictions increases word-level calibration error compared to an uncalibrated model. In addition, we apply existing calibration methodologies as well as new sequence-based extensions to numerous STR models, demonstrating reduced calibration error by up to a factor of nearly 7. Finally, we show consistently improved accuracy results by applying our proposed sequence calibration method as a preprocessing step to beam-search.

</p>
</details>

<details><summary><b>Direct Estimation of Spinal Cobb Angles by Structured Multi-Output Regression</b>
<a href="https://arxiv.org/abs/2012.12626">arxiv:2012.12626</a>
&#x1F4C8; 1 <br>
<p>Haoliang Sun, Xiantong Zhen, Chris Bailey, Parham Rasoulinejad, Yilong Yin, Shuo Li</p></summary>
<p>

**Abstract:** The Cobb angle that quantitatively evaluates the spinal curvature plays an important role in the scoliosis diagnosis and treatment. Conventional measurement of these angles suffers from huge variability and low reliability due to intensive manual intervention. However, since there exist high ambiguity and variability around boundaries of vertebrae, it is challenging to obtain Cobb angles automatically. In this paper, we formulate the estimation of the Cobb angles from spinal X-rays as a multi-output regression task. We propose structured support vector regression (S^2VR) to jointly estimate Cobb angles and landmarks of the spine in X-rays in one single framework. The proposed S^2VR can faithfully handle the nonlinear relationship between input images and quantitative outputs, while explicitly capturing the intrinsic correlation of outputs. We introduce the manifold regularization to exploit the geometry of the output space. We propose learning the kernel in S2VR by kernel target alignment to enhance its discriminative ability. The proposed method is evaluated on the spinal X-rays dataset of 439 scoliosis subjects, which achieves the inspiring correlation coefficient of 92.76% with ground truth obtained manually by human experts and outperforms two baseline methods. Our method achieves the direct estimation of Cobb angles with high accuracy, which indicates its great potential in clinical use.

</p>
</details>

<details><summary><b>Commission Fee is not Enough: A Hierarchical Reinforced Framework for Portfolio Management</b>
<a href="https://arxiv.org/abs/2012.12620">arxiv:2012.12620</a>
&#x1F4C8; 1 <br>
<p>Rundong Wang, Hongxin Wei, Bo An, Zhouyan Feng, Jun Yao</p></summary>
<p>

**Abstract:** Portfolio management via reinforcement learning is at the forefront of fintech research, which explores how to optimally reallocate a fund into different financial assets over the long term by trial-and-error. Existing methods are impractical since they usually assume each reallocation can be finished immediately and thus ignoring the price slippage as part of the trading cost. To address these issues, we propose a hierarchical reinforced stock trading system for portfolio management (HRPM). Concretely, we decompose the trading process into a hierarchy of portfolio management over trade execution and train the corresponding policies. The high-level policy gives portfolio weights at a lower frequency to maximize the long term profit and invokes the low-level policy to sell or buy the corresponding shares within a short time window at a higher frequency to minimize the trading cost. We train two levels of policies via pre-training scheme and iterative training scheme for data efficiency. Extensive experimental results in the U.S. market and the China market demonstrate that HRPM achieves significant improvement against many state-of-the-art approaches.

</p>
</details>

<details><summary><b>Distributed Adaptive Control: An ideal Cognitive Architecture candidate for managing a robotic recycling plant</b>
<a href="https://arxiv.org/abs/2012.12586">arxiv:2012.12586</a>
&#x1F4C8; 1 <br>
<p>Oscar Guerrero-Rosado, Paul Verschure</p></summary>
<p>

**Abstract:** In the past decade, society has experienced notable growth in a variety of technological areas. However, the Fourth Industrial Revolution has not been embraced yet. Industry 4.0 imposes several challenges which include the necessity of new architectural models to tackle the uncertainty that open environments represent to cyber-physical systems (CPS). Waste Electrical and Electronic Equipment (WEEE) recycling plants stand for one of such open environments. Here, CPSs must work harmoniously in a changing environment, interacting with similar and not so similar CPSs, and adaptively collaborating with human workers. In this paper, we support the Distributed Adaptive Control (DAC) theory as a suitable Cognitive Architecture for managing a recycling plant. Specifically, a recursive implementation of DAC (between both single-agent and large-scale levels) is proposed to meet the expected demands of the European Project HR-Recycler. Additionally, with the aim of having a realistic benchmark for future implementations of the recursive DAC, a micro-recycling plant prototype is presented.

</p>
</details>

<details><summary><b>BaPipe: Exploration of Balanced Pipeline Parallelism for DNN Training</b>
<a href="https://arxiv.org/abs/2012.12544">arxiv:2012.12544</a>
&#x1F4C8; 1 <br>
<p>Letian Zhao, Rui Xu, Tianqi Wang, Teng Tian, Xiaotian Wang, Wei Wu, Chio-in Ieong, Xi Jin</p></summary>
<p>

**Abstract:** The size of deep neural networks (DNNs) grows rapidly as the complexity of the machine learning algorithm increases. To satisfy the requirement of computation and memory of DNN training, distributed deep learning based on model parallelism has been widely recognized. We propose a new pipeline parallelism training framework, BaPipe, which can automatically explore pipeline parallelism training methods and balanced partition strategies for DNN distributed training. In BaPipe, each accelerator calculates the forward propagation and backward propagation of different parts of networks to implement the intra-batch pipeline parallelism strategy. BaPipe uses a new load balancing automatic exploration strategy that considers the parameters of DNN models and the computation, memory, and communication resources of accelerator clusters. We have trained different DNNs such as VGG-16, ResNet-50, and GNMT on GPU clusters and simulated the performance of different FPGA clusters. Compared with state-of-the-art data parallelism and pipeline parallelism frameworks, BaPipe provides up to 3.2x speedup and 4x memory reduction in various platforms.

</p>
</details>

<details><summary><b>The Translucent Patch: A Physical and Universal Attack on Object Detectors</b>
<a href="https://arxiv.org/abs/2012.12528">arxiv:2012.12528</a>
&#x1F4C8; 1 <br>
<p>Alon Zolfi, Moshe Kravchik, Yuval Elovici, Asaf Shabtai</p></summary>
<p>

**Abstract:** Physical adversarial attacks against object detectors have seen increasing success in recent years. However, these attacks require direct access to the object of interest in order to apply a physical patch. Furthermore, to hide multiple objects, an adversarial patch must be applied to each object. In this paper, we propose a contactless translucent physical patch containing a carefully constructed pattern, which is placed on the camera's lens, to fool state-of-the-art object detectors. The primary goal of our patch is to hide all instances of a selected target class. In addition, the optimization method used to construct the patch aims to ensure that the detection of other (untargeted) classes remains unharmed. Therefore, in our experiments, which are conducted on state-of-the-art object detection models used in autonomous driving, we study the effect of the patch on the detection of both the selected target class and the other classes. We show that our patch was able to prevent the detection of 42.27% of all stop sign instances while maintaining high (nearly 80%) detection of the other classes.

</p>
</details>

<details><summary><b>Diabetic Retinopathy Grading System Based on Transfer Learning</b>
<a href="https://arxiv.org/abs/2012.12515">arxiv:2012.12515</a>
&#x1F4C8; 1 <br>
<p>Eman AbdelMaksoud, Sherif Barakat, Mohammed Elmogy</p></summary>
<p>

**Abstract:** Much effort is being made by the researchers in order to detect and diagnose diabetic retinopathy (DR) accurately automatically. The disease is very dangerous as it can cause blindness suddenly if it is not continuously screened. Therefore, many computers aided diagnosis (CAD) systems have been developed to diagnose the various DR grades. Recently, many CAD systems based on deep learning (DL) methods have been adopted to get deep learning merits in diagnosing the pathological abnormalities of DR disease. In this paper, we present a full based-DL CAD system depending on multi-label classification. In the proposed DL CAD system, we present a customized efficientNet model in order to diagnose the early and advanced grades of the DR disease. Learning transfer is very useful in training small datasets. We utilized IDRiD dataset. It is a multi-label dataset. The experiments manifest that the proposed DL CAD system is robust, reliable, and deigns promising results in detecting and grading DR. The proposed system achieved accuracy (ACC) equals 86%, and the Dice similarity coefficient (DSC) equals 78.45.

</p>
</details>

<details><summary><b>Towards Overcoming False Positives in Visual Relationship Detection</b>
<a href="https://arxiv.org/abs/2012.12510">arxiv:2012.12510</a>
&#x1F4C8; 1 <br>
<p>Daisheng Jin, Xiao Ma, Chongzhi Zhang, Yizhuo Zhou, Jiashu Tao, Mingyuan Zhang, Haiyu Zhao, Shuai Yi, Zhoujun Li, Xianglong Liu, Hongsheng Li</p></summary>
<p>

**Abstract:** In this paper, we investigate the cause of the high false positive rate in Visual Relationship Detection (VRD). We observe that during training, the relationship proposal distribution is highly imbalanced: most of the negative relationship proposals are easy to identify, e.g., the inaccurate object detection, which leads to the under-fitting of low-frequency difficult proposals. This paper presents Spatially-Aware Balanced negative pRoposal sAmpling (SABRA), a robust VRD framework that alleviates the influence of false positives. To effectively optimize the model under imbalanced distribution, SABRA adopts Balanced Negative Proposal Sampling (BNPS) strategy for mini-batch sampling. BNPS divides proposals into 5 well defined sub-classes and generates a balanced training distribution according to the inverse frequency. BNPS gives an easier optimization landscape and significantly reduces the number of false positives. To further resolve the low-frequency challenging false positive proposals with high spatial ambiguity, we improve the spatial modeling ability of SABRA on two aspects: a simple and efficient multi-head heterogeneous graph attention network (MH-GAT) that models the global spatial interactions of objects, and a spatial mask decoder that learns the local spatial configuration. SABRA outperforms SOTA methods by a large margin on two human-object interaction (HOI) datasets and one general VRD dataset.

</p>
</details>

<details><summary><b>A method to integrate and classify normal distributions</b>
<a href="https://arxiv.org/abs/2012.14331">arxiv:2012.14331</a>
&#x1F4C8; 0 <br>
<p>Abhranil Das, Wilson S Geisler</p></summary>
<p>

**Abstract:** Univariate and multivariate normal probability distributions are widely used when modeling decisions under uncertainty. Computing the performance of such models requires integrating these distributions over specific domains, which can vary widely across models. Besides some special cases where these integrals are easy to calculate, there exists no general analytical expression, standard numerical method or software for these integrals. Here we present mathematical results and software that provide (i) the probability in any domain of a normal in any dimensions with any parameters, (ii) the probability density, distribution, and percentage points of any function of a normal vector, (iii) quantities, such as the error matrix and discriminability, which summarize classification performance amongst any number of normal distributions, (iv) dimension reduction and visualizations for all such problems, and (v) tests for how reliably these methods can be used on given data. We illustrate these tools with models for detecting occluding targets in natural scenes and for detecting camouflage.

</p>
</details>

<details><summary><b>A Generalized A* Algorithm for Finding Globally Optimal Paths in Weighted Colored Graphs</b>
<a href="https://arxiv.org/abs/2012.13057">arxiv:2012.13057</a>
&#x1F4C8; 0 <br>
<p>Jaein Lim, Panagiotis Tsiotras</p></summary>
<p>

**Abstract:** Both geometric and semantic information of the search space is imperative for a good plan. We encode those properties in a weighted colored graph (geometric information in terms of edge weight and semantic information in terms of edge and vertex color), and propose a generalized A* to find the shortest path among the set of paths with minimal inclusion of low-ranked color edges. We prove the completeness and optimality of this Class-Ordered A* (COA*) algorithm with respect to the hereto defined notion of optimality. The utility of COA* is numerically validated in a ternary graph with feasible, infeasible, and unknown vertices and edges for the cases of a 2D mobile robot, a 3D robotic arm, and a 5D robotic arm with limited sensing capabilities. We compare the results of COA* to that of the regular A* algorithm, the latter of which finds the shortest path regardless of uncertainty, and we show that the COA* dominates the A* solution in terms of finding less uncertain paths.

</p>
</details>

<details><summary><b>Disentangling semantics in language through VAEs and a certain architectural choice</b>
<a href="https://arxiv.org/abs/2012.13031">arxiv:2012.13031</a>
&#x1F4C8; 0 <br>
<p>Ghazi Felhi, Joseph Le Roux, Djamé Seddah</p></summary>
<p>

**Abstract:** We present an unsupervised method to obtain disentangled representations of sentences that single out semantic content. Using modified Transformers as building blocks, we train a Variational Autoencoder to translate the sentence to a fixed number of hierarchically structured latent variables. We study the influence of each latent variable in generation on the dependency structure of sentences, and on the predicate structure it yields when passed through an Open Information Extraction model. Our model could separate verbs, subjects, direct objects, and prepositional objects into latent variables we identified. We show that varying the corresponding latent variables results in varying these elements in sentences, and that swapping them between couples of sentences leads to the expected partial semantic swap.

</p>
</details>

<details><summary><b>Antitrust and Artificial Intelligence (AAI): Antitrust Vigilance Lifecycle and AI Legal Reasoning Autonomy</b>
<a href="https://arxiv.org/abs/2012.13016">arxiv:2012.13016</a>
&#x1F4C8; 0 <br>
<p>Lance Eliot</p></summary>
<p>

**Abstract:** There is an increasing interest in the entwining of the field of antitrust with the field of Artificial Intelligence (AI), frequently referred to jointly as Antitrust and AI (AAI) in the research literature. This study focuses on the synergies entangling antitrust and AI, doing so to extend the literature by proffering the primary ways that these two fields intersect, consisting of: (1) the application of antitrust to AI, and (2) the application of AI to antitrust. To date, most of the existing research on this intermixing has concentrated on the former, namely the application of antitrust to AI, entailing how the marketplace will be altered by the advent of AI and the potential for adverse antitrust behaviors arising accordingly. Opting to explore more deeply the other side of this coin, this research closely examines the application of AI to antitrust and establishes an antitrust vigilance lifecycle to which AI is predicted to be substantively infused for purposes of enabling and bolstering antitrust detection, enforcement, and post-enforcement monitoring. Furthermore, a gradual and incremental injection of AI into antitrust vigilance is anticipated to occur as significant advances emerge amidst the Levels of Autonomy (LoA) for AI Legal Reasoning (AILR).

</p>
</details>

<details><summary><b>Eurythmic Dancing with Plants -- Measuring Plant Response to Human Body Movement in an Anthroposophic Environment</b>
<a href="https://arxiv.org/abs/2012.12978">arxiv:2012.12978</a>
&#x1F4C8; 0 <br>
<p>Sebastian Duerr, Josephine van Delden, Buenyamin Oezkaya, Peter A. Gloor</p></summary>
<p>

**Abstract:** This paper describes three experiments measuring interaction of humans with garden plants. In particular, body movement of a human conducting eurythmic dances near the plants (beetroots, tomatoes, lettuce) is correlated with the action potential measured by a plant SpikerBox, a device measuring the electrical activity of plants, and the leaf movement of the plant, tracked with a camera. The first experiment shows that our measurement system captures external stimuli identically for different plants, validating the measurement system. The second experiment illustrates that the plants' response is correlated to the movements of the dancer. The third experiment indicates that plants that have been exposed for multiple weeks to eurythmic dancing might respond differently to plants which are exposed for the first time to eurythmic dancing.

</p>
</details>

<details><summary><b>A Modern Analysis of Hutchinson's Trace Estimator</b>
<a href="https://arxiv.org/abs/2012.12895">arxiv:2012.12895</a>
&#x1F4C8; 0 <br>
<p>Maciej Skorski</p></summary>
<p>

**Abstract:** The paper establishes the new state-of-art in the accuracy analysis of Hutchinson's trace estimator. Leveraging tools that have not been previously used in this context, particularly hypercontractive inequalities and concentration properties of sub-gamma distributions, we offer an elegant and modular analysis, as well as numerically superior bounds. Besides these improvements, this work aims to better popularize the aforementioned techniques within the CS community.

</p>
</details>

<details><summary><b>EQ-Net: A Unified Deep Learning Framework for Log-Likelihood Ratio Estimation and Quantization</b>
<a href="https://arxiv.org/abs/2012.12843">arxiv:2012.12843</a>
&#x1F4C8; 0 <br>
<p>Marius Arvinte, Ahmed H. Tewfik, Sriram Vishwanath</p></summary>
<p>

**Abstract:** In this work, we introduce EQ-Net: the first holistic framework that solves both the tasks of log-likelihood ratio (LLR) estimation and quantization using a data-driven method. We motivate our approach with theoretical insights on two practical estimation algorithms at the ends of the complexity spectrum and reveal a connection between the complexity of an algorithm and the information bottleneck method: simpler algorithms admit smaller bottlenecks when representing their solution. This motivates us to propose a two-stage algorithm that uses LLR compression as a pretext task for estimation and is focused on low-latency, high-performance implementations via deep neural networks. We carry out extensive experimental evaluation and demonstrate that our single architecture achieves state-of-the-art results on both tasks when compared to previous methods, with gains in quantization efficiency as high as $20\%$ and reduced estimation latency by up to $60\%$ when measured on general purpose and graphical processing units (GPU). In particular, our approach reduces the GPU inference latency by more than two times in several multiple-input multiple-output (MIMO) configurations. Finally, we demonstrate that our scheme is robust to distributional shifts and retains a significant part of its performance when evaluated on 5G channel models, as well as channel estimation errors.

</p>
</details>

<details><summary><b>Optimal dimension dependence of the Metropolis-Adjusted Langevin Algorithm</b>
<a href="https://arxiv.org/abs/2012.12810">arxiv:2012.12810</a>
&#x1F4C8; 0 <br>
<p>Sinho Chewi, Chen Lu, Kwangjun Ahn, Xiang Cheng, Thibaut Le Gouic, Philippe Rigollet</p></summary>
<p>

**Abstract:** Conventional wisdom in the sampling literature, backed by a popular diffusion scaling limit, suggests that the mixing time of the Metropolis-Adjusted Langevin Algorithm (MALA) scales as $O(d^{1/3})$, where $d$ is the dimension. However, the diffusion scaling limit requires stringent assumptions on the target distribution and is asymptotic in nature. In contrast, the best known non-asymptotic mixing time bound for MALA on the class of log-smooth and strongly log-concave distributions is $O(d)$. In this work, we establish that the mixing time of MALA on this class of target distributions is $\widetildeΘ(d^{1/2})$ under a warm start. Our upper bound proof introduces a new technique based on a projection characterization of the Metropolis adjustment which reduces the study of MALA to the well-studied discretization analysis of the Langevin SDE and bypasses direct computation of the acceptance probability.

</p>
</details>

<details><summary><b>Learning the Gap in the Day-Ahead and Real-Time Locational Marginal Prices in the Electricity Market</b>
<a href="https://arxiv.org/abs/2012.12792">arxiv:2012.12792</a>
&#x1F4C8; 0 <br>
<p>Nika Nizharadze, Arash Farokhi Soofi, Saeed D. Manshadi</p></summary>
<p>

**Abstract:** In this paper, statistical machine learning algorithms, as well as deep neural networks, are used to predict the values of the price gap between day-ahead and real-time electricity markets. Several exogenous features are collected and impacts of these features are examined to capture the best relations between the features and the target variable. Ensemble learning algorithm namely the Random Forest issued to calculate the probability distribution of the predicted electricity prices for day-ahead and real-time markets. Long-Short-Term-Memory (LSTM) is utilized to capture long term dependencies in predicting direct gap values between mentioned markets and the benefits of directly predicting the gap price rather than subtracting the predictions of day-ahead and real-time markets are illustrated. Case studies are implemented on the California Independent System Operator (CAISO) electricity market data for a two years period. The proposed methods are evaluated and neural networks showed promising results in predicting the exact values of the gap.

</p>
</details>

<details><summary><b>Matrix optimization based Euclidean embedding with outliers</b>
<a href="https://arxiv.org/abs/2012.12772">arxiv:2012.12772</a>
&#x1F4C8; 0 <br>
<p>Qian Zhang, Xinyuan Zhao, Chao Ding</p></summary>
<p>

**Abstract:** Euclidean embedding from noisy observations containing outlier errors is an important and challenging problem in statistics and machine learning. Many existing methods would struggle with outliers due to a lack of detection ability. In this paper, we propose a matrix optimization based embedding model that can produce reliable embeddings and identify the outliers jointly. We show that the estimators obtained by the proposed method satisfy a non-asymptotic risk bound, implying that the model provides a high accuracy estimator with high probability when the order of the sample size is roughly the degree of freedom up to a logarithmic factor. Moreover, we show that under some mild conditions, the proposed model also can identify the outliers without any prior information with high probability. Finally, numerical experiments demonstrate that the matrix optimization-based model can produce configurations of high quality and successfully identify outliers even for large networks.

</p>
</details>

<details><summary><b>Second-Moment Loss: A Novel Regression Objective for Improved Uncertainties</b>
<a href="https://arxiv.org/abs/2012.12687">arxiv:2012.12687</a>
&#x1F4C8; 0 <br>
<p>Joachim Sicking, Maram Akila, Maximilian Pintz, Tim Wirtz, Asja Fischer, Stefan Wrobel</p></summary>
<p>

**Abstract:** Quantification of uncertainty is one of the most promising approaches to establish safe machine learning. Despite its importance, it is far from being generally solved, especially for neural networks. One of the most commonly used approaches so far is Monte Carlo dropout, which is computationally cheap and easy to apply in practice. However, it can underestimate the uncertainty. We propose a new objective, referred to as second-moment loss (SML), to address this issue. While the full network is encouraged to model the mean, the dropout networks are explicitly used to optimize the model variance. We analyze the performance of the new objective on various toy and UCI regression datasets. Comparing to the state-of-the-art of deep ensembles, SML leads to comparable prediction accuracies and uncertainty estimates while only requiring a single model. Under distribution shift, we observe moderate improvements. From a safety perspective also the study of worst-case uncertainties is crucial. In this regard we improve considerably. Finally, we show that SML can be successfully applied to SqueezeDet, a modern object detection network. We improve on its uncertainty-related scores while not deteriorating regression quality. As a side result, we introduce an intuitive Wasserstein distance-based uncertainty measure that is non-saturating and thus allows to resolve quality differences between any two uncertainty estimates.

</p>
</details>

<details><summary><b>Probabilistic Iterative Methods for Linear Systems</b>
<a href="https://arxiv.org/abs/2012.12615">arxiv:2012.12615</a>
&#x1F4C8; 0 <br>
<p>Jon Cockayne, Ilse C. F. Ipsen, Chris J. Oates, Tim W. Reid</p></summary>
<p>

**Abstract:** This paper presents a probabilistic perspective on iterative methods for approximating the solution $\mathbf{x}_* \in \mathbb{R}^d$ of a nonsingular linear system $\mathbf{A} \mathbf{x}_* = \mathbf{b}$. In the approach a standard iterative method on $\mathbb{R}^d$ is lifted to act on the space of probability distributions $\mathcal{P}(\mathbb{R}^d)$. Classically, an iterative method produces a sequence $\mathbf{x}_m$ of approximations that converge to $\mathbf{x}_*$. The output of the iterative methods proposed in this paper is, instead, a sequence of probability distributions $μ_m \in \mathcal{P}(\mathbb{R}^d)$. The distributional output both provides a "best guess" for $\mathbf{x}_*$, for example as the mean of $μ_m$, and also probabilistic uncertainty quantification for the value of $\mathbf{x}_*$ when it has not been exactly determined. Theoretical analysis is provided in the prototypical case of a stationary linear iterative method. In this setting we characterise both the rate of contraction of $μ_m$ to an atomic measure on $\mathbf{x}_*$ and the nature of the uncertainty quantification being provided. We conclude with an empirical illustration that highlights the insight into solution uncertainty that can be provided by probabilistic iterative methods.

</p>
</details>

<details><summary><b>IFGAN: Missing Value Imputation using Feature-specific Generative Adversarial Networks</b>
<a href="https://arxiv.org/abs/2012.12581">arxiv:2012.12581</a>
&#x1F4C8; 0 <br>
<p>Wei Qiu, Yangsibo Huang, Quanzheng Li</p></summary>
<p>

**Abstract:** Missing value imputation is a challenging and well-researched topic in data mining. In this paper, we propose IFGAN, a missing value imputation algorithm based on Feature-specific Generative Adversarial Networks (GAN). Our idea is intuitive yet effective: a feature-specific generator is trained to impute missing values, while a discriminator is expected to distinguish the imputed values from observed ones. The proposed architecture is capable of handling different data types, data distributions, missing mechanisms, and missing rates. It also improves post-imputation analysis by preserving inter-feature correlations. We empirically show on several real-life datasets that IFGAN outperforms current state-of-the-art algorithm under various missing conditions.

</p>
</details>


[Next Page](2020/2020-12/2020-12-22.md)
