## Session-based Recommendation System (SBRS)
An SBRS aims to predict either the unknown part (e.g., an item or a batch of items) of a session given the known part, 
or the future session (e.g., the next-basket) given the historical sessions via learning the intra- or inter-session dependencies. 
Such dependencies usually largely rely on the co-occurrence of interactions inside a session and they may be sequential or non-sequential [49]. 
In principle, an SBRS does not necessarily rely on the order information inside sessions, 
but for ordered sessions, the naturally existing sequential dependencies can be utilized for recommendations. 
In comparison, an SRS predicts the successive elements given a sequence of historical ones by learning the sequential dependencies among them. 

![image](https://user-images.githubusercontent.com/4285481/139530028-7c538ff6-a9e7-4937-9561-f41f0b341958.png)




### Survey papers
- [A Survey on Session-based Recommender Systems](https://arxiv.org/abs/1902.04864)
- [Deep Learning for Sequential Recommendation: Algorithms, Influential Factors, and Evaluations](https://arxiv.org/abs/1905.01997)
- https://session-based-recommenders.fastforwardlabs.com
- https://paperswithcode.com/task/session-based-recommendations

### Awesome Opensource
- https://github.com/rn5l/session-rec
- https://github.com/RUCAIBox/RecBole, A unified, comprehensive and efficient recommendation library
- https://github.com/mmaher22/iCV-SBR, Benchmarking of Session-based Recommendation Approaches

### Dataset (E-Commerce)
  - 2015 | YOOCHOOSE - RecSys Challenge | [`URL`](http://2015.recsyschallenge.com/)
  - 2015 | Zalando Fashion Recommendation | [`NA`](https://zalando.com/)
  - 2016 | Diginetica - CIKM Cup | [`URL`](https://cikm2016.cs.iupui.edu/cikm-cup/)
  - 2016 | TMall (Taobao) - IJCAI16 Contest | [`URL`](https://tianchi.aliyun.com/dataset/dataDetail?dataId=53)
  - 2017 | Retail Rocket | [`URL`](https://www.kaggle.com/retailrocket/ecommerce-dataset)


### Technical papers
#### [WWW'10, FPMC, Factorizing Personalized Markov Chains for Next-basket Recommendation](https://dl.acm.org/doi/10.1145/1772690.1772773)
- https://github.com/rdevooght/sequence-based-recommendations/blob/master/factorization/fpmc.py (Numpy)

#### [ICLR'16, GRU4Rec, Session-based Recommendations with Recurrent Neural Networks](https://arxiv.org/abs/1511.06939)
- https://paperswithcode.com/paper/session-based-recommendations-with-recurrent
- https://github.com/hidasib/GRU4Rec (**_Official_**, Theano)
- https://github.com/hungthanhpham94/GRU4REC-pytorch (PyTorch)
- https://github.com/yhs968/pyGRU4REC (PyTorch)
- https://github.com/pcerdam/KerasGRU4Rec (Keras)

#### [CIKM'17, NARM, Neural Attentive Session-based Recommendation](https://arxiv.org/abs/1711.04725)
- https://paperswithcode.com/paper/neural-attentive-session-based-recommendation
- https://github.com/lijingsdu/sessionRec_NARM (**_Official_**, PyTorch)
- https://github.com/Wang-Shuo/Neural-Attentive-Session-Based-Recommendation-PyTorch (PyTorch)
- https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/sequential_recommender/narm.py (PyTorch, RecBole)

#### [CIKM'18, GRU4Rec+, Recurrent Neural Networks with Top-k Gains for Session-based Recommendations](https://arxiv.org/abs/1706.03847)
- https://paperswithcode.com/paper/recurrent-neural-networks-with-top-k-gains
- https://github.com/hungthanhpham94/GRU4REC-pytorch (**_Official_**, PyTorch)

#### [KDD'18, STAMP: Short-Term Attention/Memory Priority Model for Session-based Recommendation](https://dl.acm.org/doi/10.1145/3219819.3219950)
- https://github.com/uestcnlp/STAMP (**_Official_**, TensorFlow)
- https://github.com/rn5l/session-rec/tree/master/algorithms/STAMP (TensorFlow) 
- https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/sequential_recommender/stamp.py (PyTorch, RecBole)

#### [WSDM'18, CASER, Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding](https://arxiv.org/abs/1809.07426)
- https://paperswithcode.com/paper/personalized-top-n-sequential-recommendation
- https://github.com/graytowne/caser_pytorch (PyTorch)
- https://github.com/seunghyunhan/Caser-tensorflow (TensorFlow)
- https://github.com/slientGe/caser (TensorFlow)

#### [WSDM'19, SocialRec, Session-based Social Recommendation via Dynamic Graph Attention Networks](https://arxiv.org/abs/1902.09362)
- https://paperswithcode.com/paper/session-based-social-recommendation-via
- https://github.com/DeepGraphLearning/RecommenderSystems/tree/master/socialRec (**_Official_**, TensorFlow)

#### [AAAI'19, SR-GNN, Session-based Recommendation with Graph Neural Networks](https://arxiv.org/abs/1811.00855)
- https://paperswithcode.com/paper/session-based-recommendation-with-graph
- https://github.com/CRIPAC-DIG/SR-GNN (**_Official_**, PyTorch)

#### [TKDE'19, A-PGNN, Personalizing Graph Neural Networks with Attention Mechanism for Session-based Recommendation](https://arxiv.org/abs/1910.08887)
- https://paperswithcode.com/paper/personalizing-graph-neural-networks-with
- https://github.com/CRIPAC-DIG/A-PGNN (**_Official_**, TensorFlow)

#### [CIKM'19, FGNN, Rethinking the Item Order in Session-based Recommendation with Graph Neural Networks](https://dl.acm.org/doi/abs/10.1145/3357384.3358010)
- https://paperswithcode.com/paper/rethinking-the-item-order-in-session-based
- https://github.com/RuihongQiu/FGNN (**_Official_**, PyTorch + torch_geometric)

#### [CIKM'19, NISER: Normalized Item and Session Representations to Handle Popularity Bias](https://arxiv.org/abs/1909.04276)
- https://paperswithcode.com/paper/niser-normalized-item-and-session
- https://github.com/johnny12150/NISER

#### [SIGIR'19, CSRM, A collaborativesession-based recommendation approach with parallel memory modules](https://dl.acm.org/doi/10.1145/3331184.3331210)
- https://github.com/wmeirui/CSRM_SIGIR2019 (Official, TensorFlow)

#### [AAAI'19, RepeatNet: A Repeat Aware Neural Recommendation Machine for Session-based Recommendation](https://arxiv.org/abs/1812.02646)
- https://paperswithcode.com/paper/repeatnet-a-repeat-aware-neural
- https://github.com/PengjieRen/RepeatNet (Official, Chianer)
- https://github.com/PengjieRen/RepeatNet-pytorch (Official, PyTorch)

#### [SIGIR'20, TAGNN: Target Attentive Graph Neural Networks for Session-based Recommendation](https://dl.acm.org/doi/10.1145/3397271.3401319)
- https://paperswithcode.com/paper/tagnn-target-attentive-graph-neural-networks
- https://github.com/CRIPAC-DIG/TAGNN (Official, PyTorch)

#### [SIGIR'20, GCE-GNN: Global Context Enhanced Graph Neural Network for Session-based Recommendation](https://dl.acm.org/doi/10.1145/3397271.3401142)
- https://paperswithcode.com/paper/global-context-enhanced-graph-neural-networks
- https://github.com/CCIIPLab/GCE-GNN (Official, PyTorch)

#### [AAAI'21, DHCN, Self-Supervised Hypergraph Convolutional Networks for Session-based Recommendation](https://ojs.aaai.org/index.php/AAAI/article/view/16578)
- https://paperswithcode.com/paper/self-supervised-hypergraph-convolutional
- https://github.com/xiaxin1998/DHCN (**_Official_**, PyTorch)

#### [CIKM'21, COTREC, Self-Supervised Graph Co-Training for Session-based Recommendation](https://arxiv.org/abs/2108.10560)
- https://github.com/xiaxin1998/COTREC (**_Official_**, PyTorch)

