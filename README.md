### AdaM: *An An Adaptive Fine-Grained Scheme for Distributed Metadata Management*
This is the source code for the ICPP'19 paper [AdaM: An An Adaptive Fine-Grained Scheme for Distributed Metadata Management](https://dl.acm.org/citation.cfm?doid=3337821.3337822).
![AdaM](https://github.com/caoshiyi/AdaM/blob/master/figures/design1.png?raw=true)

### Abstract
Distributed metadata management, administrating the distribution of metadata nodes on different metadata servers (MDS’s), can substantially improve overall performance of large-scale distributed storage systems if well designed. A major difficulty confronting many metadata management schemes is the trade-off between two conflicting aspects: system load balance and metadata locality preservation. It becomes even more challenging as file access pattern inevitably varies with time. However, existing works dynamically reallocate nodes to different servers adopting historybased coarse-grained methods, failing to make timely and efficient update on distribution of nodes. In this paper, we propose an adaptive fine-grained metadata management scheme, AdaM, leveraging Deep Reinforcement Learning, to address the trade-off dilemma against time-varying access pattern. At each time step, AdaM collects environmental “states" including access pattern, the structure of namespace tree and current distribution of nodes on MDS’s. Then an actor-critic network is trained to reallocate hot metadata nodes to different servers according to the observed “states". Adaptive to varying access pattern, AdaM can automatically migrate hot metadata nodes among servers to keep load balancing while maintaining metadata locality. We test AdaM on real-world data traces. Experimental results demonstrate the superiority of our proposed method over other schemes.

### Citation
If you find the work useful in your research, please consider citing:

        @inproceedings{DBLP:conf/icpp/CaoGGC19,
          author    = {Shiyi Cao and
                       Yuanning Gao and
                       Xiaofeng Gao and
                       Guihai Chen},
          title     = {AdaM: An Adaptive Fine-Grained Scheme for Distributed Metadata Management},
          booktitle = {Proceedings of the 48th International Conference on Parallel Processing,
                       {ICPP} 2019, Kyoto, Japan, August 05-08, 2019},
          pages     = {37:1--37:10},
          year      = {2019}
        }

