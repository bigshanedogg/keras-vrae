# generating-sentences-from-a-continuous-space
[Implementation] Generating Sentences from a Continuous Space (VRAE)

This is the keras implementation of '[Generating Sentences from a Continuous Space](https://arxiv.org/abs/1511.06349)'. The Encoder-Deocoder structure is the same as the general Variational AutoEncoder, but the layer uses recurrent layer such as lstm, gru, or rnn, instead of convolutional layer, which can be called VRAE(Variational Recurrent AutoEncoder).

- <b>reference: [[Paper] Generating Sentences from a Continuous Space](https://arxiv.org/abs/1511.06349)</b>
- <b>review: [[Review] Generating Sentences from a Continuous Space](https://bigshanedogg.github.io/posts/variational-recurrent-auto-encoder)</b>
