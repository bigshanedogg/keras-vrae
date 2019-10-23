# keras VRAE(Variational Recurrent AutoEncoder)
[Implementation] A Structured Self-attentive Sentence Embedding

This is the keras implementation of '[A Structured Self-attentive Sentence Embedding](https://arxiv.org/abs/1511.06349)'. 


The point of Self-Attention is
1. To embed contextual information into a matrix rather than a vector, assuming that there can be more than 1 contextual information for 1 sentence.
2. The optimized matrix can be regarded as heatmap of each token of given sentence per context - visualization for interpretable sentence embedding as a side effect.
3. According to 1, The sentence is also a matrix.

Self-Attention is composed of 2-layer feed forward network in fact and specific loss to optimize each rows(hyperparamter r) of matrix A inter-independent, and That's why the model use substraction between dot product of A and identity matrix I instead of KL divergence.

- <b>reference: [[Paper] Generating Sentences from a Continuous Space](https://arxiv.org/abs/1511.06349)</b>
- <b>review: [[Review] Generating Sentences from a Continuous Space](https://bigshanedogg.github.io/posts/variational-recurrent-auto-encoder)</b>
