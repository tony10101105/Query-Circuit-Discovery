# Automated-Circuit-Labelling
Official code of the project Automated Circuit Labelling

Code is built on [Michael Hanna's EAP-IG Repo](https://github.com/hannamw/eap-ig) and [Arthur Conmy's ADAC Repo](https://github.com/ArthurConmy/Automatic-Circuit-Discovery).

During a preliminary analysis of the indirect object identification (IOI) circuit (GPT2-small), I had an interesting finding regarding attention heads. I wrap it as follows:

The community's previous findings on LLM attention heads:
1. Regarding feature similarity, shallow-layer attention heads are more similar within-layer, and deeper attention heads are more dis-similar within-layer.
2. There are many redundant heads, and we should retain the most diverse heads when doing pruning.
3. Regarding functionality, shallow-layer-attention heads are more generic (eg, duplicate token head), and deeper attention heads are more specialized (eg, S-inhibition head).

Is this the end of the story? 
I think one big mystery is unsolved: Why do LLMs develop multiple attention heads with the same functionality?

Before diving into this mystery, I first study this question: **what are the feature similarities of attention heads with the same functionality?**

It may be intuitive to guess "highly similar". But 1-3. actually do not drive an answer to this question. 

I found that **feature similarities between attention heads with the same functionality are consistently higher** than averaged similarities among all attention heads. Why is the case? I believe answering this question can shed light on the mystery of the development of multiple same-function heads.

My guess: **LLMs develop multiple attention heads with the same functionality so that they can operate independently on feature space to enhance internal model robustness.**
