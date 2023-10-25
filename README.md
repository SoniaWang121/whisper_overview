# Whisper
%%To begin with, whisper is tool
%%
It is a toolkit can do:
![[Pasted image 20231023140908.png]]
![[Pasted image 20231023140858.png]]
%%
 At first, I'm gonna first give you some high level overview of this work.
%%
## Overview
### Context
%%
The paper titled "Robust Speech Recognition by Large Scale Weak Supervision" explores the capabilities of speech processing systems trained on a vast amount of internet audio transcripts. With a whopping 680,000 hours of multilingual and multitask supervision, these models demonstrate impressive generalization capabilities. They perform competitively with previously supervised results in a zero-shot transfer setting, eliminating the need for fine-tuning. Remarkably, these models come close to human accuracy and robustness.
题为“大规模弱监督的鲁棒语音识别”的论文探讨了在大量互联网音频记录上训练的语音处理系统的功能。这些模型经过长达 680,000 小时的多语言和多任务监督，展示了令人印象深刻的泛化能力。它们在零样本传输设置中的表现与之前的监督结果具有竞争力，无需进行微调。值得注意的是，这些模型接近人类的准确性和鲁棒性。
%%
### Contribution
%%
"Let's talk about the landscape of speech recognition. On one hand, we have unsupervised models trained on a whopping 1 million hours of data. Sounds impressive, right? But here's the catch: they still need fine-tuning to be truly effective.“让我们来谈谈语音识别的前景。一方面，我们拥有经过长达 100 万小时数据训练的无监督模型。听起来令人印象深刻，对吧？但问题是：它们仍然需要微调才能真正有效。
%%
- Unsupervised models: need fine-tuning to be truly effective
%%
On the other hand, we have supervised models. These are trained on much less data—just around 5,000 hours. But guess what? They're robust. They adapt well to new, unseen data, making them quite versatile.另一方面，我们有监督模型。这些数据的训练时间要少得多——只有大约 5,000 小时。但猜猜怎么了？他们很坚强。它们能够很好地适应新的、看不见的数据，因此用途广泛。
%%
- Supervised models
%%
Now, you might be thinking, 'Why not have the best of both worlds?' That's where SpeechStew comes in. It combines 7 different datasets for a total of 5,140 hours of high-quality, supervised training. It's not as big as the 1 million hours of unsupervised data, but it's a step in the right direction.现在，您可能会想，“为什么不两全其美呢？”这就是 SpeechStew 的用武之地。它结合了 7 个不同的数据集，进行了总共 5,140 小时的高质量监督训练。它不像 100 万小时的无监督数据那么大，但它是朝着正确方向迈出的一步。

But what if we could push the envelope even further? By loosening the need for perfect, human-validated transcripts, we've managed to scale up to between 10,000 and 30,000 hours of 'noisier' but still valuable training data. And that, folks, is how we're bridging the gap between quantity and quality in speech recognition."但如果我们能进一步挑战极限呢？通过放松对完美、经过人工验证的成绩单的需求，我们成功地将“嘈杂”但仍然有价值的训练数据扩展到 10,000 到 30,000 小时。朋友们，这就是我们缩小语音识别数量和质量之间差距的方法。”
%%
### Approach
%%
"Today, we're excited to introduce Whisper2, a groundbreaking approach to weakly supervised speech recognition. What sets us apart? We've scaled up—big time. We're talking about a massive 680,000 hours of labeled audio data, which is a significant leap from what's currently available.“今天，我们很高兴推出 Whisper2，这是一种弱监督语音识别的突破性方法。是什么让我们与众不同？我们已经大规模扩大规模。我们正在谈论 680,000 小时的大量标记音频数据，这些数据与当前可用的产品相比，这是一个重大飞跃。
%%
- Scaling weakly supervised speech recognition the next order of magnitude to 680,000 hours of labeled audio data
%%
But that's not all. Our model isn't just about quantity; it's about quality and versatility. It performs exceptionally well on existing datasets without any need for fine-tuning. That means it's ready to go, right out of the box.但这还不是全部。我们的模型不仅仅关乎数量，还关乎数量。这是关于质量和多功能性。它在现有数据集上表现得非常好，无需任何微调。这意味着它开箱即用。
%%
- Without any need for fine-tuning
- The dataset is a global affair
%%
And we didn't stop at English. Our dataset is a global affair, covering 96 other languages with 117,000 hours of audio. Plus, we've got 125,000 hours dedicated to translation tasks. The best part? Our tests show that when you go big with the model size, there's no downside to training it on multiple languages and tasks. In fact, it's a win-win situation."我们并没有停留在英语上。我们的数据集是一个全球性的数据集，涵盖 96 种其他语言和 117,000 小时的音频。此外，我们还有 125,000 个小时专门用于翻译任务。最好的部分？我们的测试表明，当您加大模型大小时，在多种语言和任务上对其进行训练并没有什么坏处。事实上，这是一个双赢的局面。”
%%
### Training details
%%
After training the initial models, they iteratively refined the dataset based on model predictions and manual inspection. One observation was the model's tendency to incorrectly guess speaker names, which they addressed through fine-tuning.
训练初始模型后，他们根据模型预测和手动检查迭代细化数据集。一项观察结果是模型倾向于错误地猜测说话者姓名，他们通过微调解决了这一问题。
%%
- Whisper models had a tendency to transcribe plausible but almost always incorrect guesses for the names of speakers.
- Fine-tune Whisper models briefly on the subset of transcripts that do not include speaker annotations which removes this behavior.
## Architecture Overview
%%
The architecture of the system is built upon a standard Transformer encoder-decoder module, a design inspired by the 2017 paper "Attention Is All You Need". The uniqueness of this research doesn't lie in the architecture, but in the data and the special tokens utilized.
该系统的架构建立在标准 Transformer 编码器-解码器模块的基础上，其设计灵感来自 2017 年论文《Attention Is All You Need》。这项研究的独特性不在于架构，而在于数据和所使用的特殊令牌。
The process begins with an audio file that is converted into log Mel spectrograms. These are visual representations of audio, where the x-axis depicts time, the y-axis shows frequency (in Mel scale, which a measure that mimics human pitch perception), and the color indicates amplitude.
该过程从转换为对数梅尔频谱图的音频文件开始。这些是音频的视觉表示，其中 x 轴表示时间，y 轴表示频率（以梅尔标度表示，这是一种模仿人类音调感知的对数度量），颜色表示幅度。
Once the audio is represented this way, it is passed through the encoder stack, consisting of Transformer layers. The final audio representation conditions a causal decoder via a cross-attention mechanism. The real magic happens with special tokens like "start of transcription", "English", and "transcribe". The model produces a timestamp followed by the transcription of the audio.
一旦音频以这种方式表示，它就会通过由 Transformer 层组成的编码器堆栈。最终的音频表示通过交叉注意机制调节因果解码器。真正的魔力发生在特殊的标记上，例如“转录开始”、“英语”和“转录”。该模型会生成一个时间戳，然后生成音频转录。
While the architecture is not the primary contribution of the paper, the dataset and the multitask setup with tokens certainly are. 
虽然架构不是本文的主要贡献，但数据集和带有标记的多任务设置肯定是。
%%
![[Pasted image 20231023140015.png]]
%%
1. **Audio Processing**: The model starts by converting audio into a log Mel spectrogram. This is a way to represent audio that makes it easier for the model to understand. It then passes this through a 1D convolutional layer with activation functions to further process the data.音频处理：该模型首先将音频转换为对数梅尔频谱图。这是一种表示音频的方式，使模型更容易理解。然后，它通过具有激活函数的一维卷积层来进一步处理数据。
2. **Positional Encodings**: After that, positional encodings are added. These help the model understand the sequence or order of the data.位置编码：之后添加位置编码。这些有助于模型理解数据的顺序。
3. **Encoder Blocks**: The processed audio data is then sent through a series of encoder blocks. These are part of the Transformer architecture and are responsible for transforming the input data into a format that the decoder can understand.编码器块：处理后的音频数据然后通过一系列编码器块发送。它们是 Transformer 架构的一部分，负责将输入数据转换为解码器可以理解的格式。
4. **Decoder and Tokens**: The decoder's job is to predict what was said in the audio. To help it do this, various tokens are used. These tokens indicate things like the start of a transcript, the language being spoken, and the task (transcribing or translating). Time stamps can also be added to indicate when in the audio the spoken words occur.解码器和令牌：解码器的工作是预测音频中所说的内容。为了帮助它做到这一点，使用了各种令牌。这些标记表示转录的开始、所讲的语言和任务（转录或翻译）等内容。还可以添加时间戳来指示音频中说出的单词何时出现。
5. **Multi-Task Training**: The model is designed to handle multiple tasks. Sometimes it conditions its predictions based on previous text. This is useful for tasks like ongoing transcription where the audio might be part of a longer conversation.多任务训练：该模型旨在处理多个任务。有时它会根据之前的文本来调整其预测。这对于正在进行的转录等任务非常有用，其中音频可能是较长对话的一部分。
6. **Time Handling**: The model works with 30-second audio clips. If a spoken sentence spans multiple clips, the model adjusts the time window accordingly.时间处理：该模型适用于 30 秒的音频剪辑。如果一个口语句子跨越多个剪辑，模型会相应地调整时间窗口。
%%
%%
8. **Model Variants**: There are different sizes of the model, ranging from 'tiny' to 'large'. Each size has a trade-off between speed and accuracy. A newer version, "large V2," was recently released with some improvements.模型变体：模型有不同尺寸，范围从“微小”到“大”。每种尺寸都需要在速度和准确性之间进行权衡。最近发布了更新版本“large V2”，并进行了一些改进。
%%
### Model Variants
![[Pasted image 20231024203345.png|600]]
%%
10. **Performance Metrics**: The model's performance is usually evaluated using Word Error Rate (WER), which measures how many mistakes the model makes. However, WER is not perfect and can be influenced by the specific style of the dataset used for evaluation.性能指标：模型的性能通常使用字错误率（WER）来评估，它衡量模型犯了多少错误。然而，WER 并不完美，可能会受到用于评估的数据集的特定风格的影响。
%%
### Performance Metrics
![[Pasted image 20231023124517.png]]
%%
10. **Manual Adjustments**: To improve performance, some manual adjustments were made to better adapt the model to specific datasets.手动调整：为了提高性能，进行了一些手动调整以使模型更好地适应特定数据集。
%%
## Pseudocode
The architecture comprises two main components: the audio encoder and the text decoder, as illustrated in the provided diagram.
%%
![[Pasted image 20231023151017.png|350]]
%%
Audio Encoder:
- Features two convolutional 1D (com1D) layers.
- Utilizes non-learnable sinusoidal positional embeddings.
- Contains several residual attention blocks.
- Concludes with a layer normalization.
- In the forward pass, the Mel spectrogram is passed through the com1D, leading to the stem and subsequent blocks.
%%
![[Pasted image 20231023151033.png|400]]![[Pasted image 20231023151630.png|300]]
%%
Text Decoder:
- Contains an embedding table for handling vocabulary.
- Uses learned positional embeddings, initialized with a context length of 448 and a state of approximately 768.
- Comprises blocks that support cross-attention.
- Utilizes a causal mask to ensure tokens attend only to preceding tokens or themselves. This is evident with minus infinite values on the upper triangle, masking out future tokens.
%%
![[Pasted image 20231023151123.png|400]]![[Pasted image 20231023151630.png|300]]

## Empirical Results
%%
The goal of Whisper is to develop a single robust speech processing system that works reliably without the need for dataset specific fine-tuning to achieve high-quality results on specific distributions. To study this capability, it reuse a wide set of existing speech processing datasets to check whether Whisper is able to generalize well across domains, tasks, and languages. Instead of using the standard evaluation protocol for these datasets, which include both a train and test split, we evaluate Whisper in a zero-shot setting without using any of the training data for each of these datasets so that we are measuring broad generalization.
%%
### Evaluation Metrics: Text normalization
**WER** (word error rate)


%%
It perform the following steps to normalize English texts in different styles into a standardized form, which is a best-effort attempt to penalize only when a word error is caused by actually mistranscribing a word, and not by formatting or punctuation differences.
%%
### As good as human being?
Close to performance of professional human transcribers!
![[Pasted image 20231023123802.png]]
%%
Whisper’s performance is close to that of professional human transcribers. This plot shows the WER distributions of 25 recordings from the Kincaid46 dataset transcribed by Whisper, the same 4 commercial ASR systems from Figure 6 (A-D), one computer-assisted human transcription service (E) and 4 human transcription services (F-I). The box plot is superimposed with dots indicating the WERs on individual recordings, and the aggregate WER over the 25 recordings are annotated on each box.
%%
## Code demonstration

<https://colab.research.google.com/drive/1M8zNZ24lGcf05j-u53y73D-OhOv6z0I0?usp=drive_link>
## Disscusion Q1
## Disscusion Q2

## Critical Analysis
### Advantages
Accuracy on speech recognition and translation is near the state-of-the-art level.
%%
Our studies show that, over many existing ASR systems, the models exhibit improved robustness to accents, background noise, and technical language, as well as zero-shot translation from multiple languages into English; and that accuracy on speech recognition and translation is near the state-of-the-art level.
%%
### Were there any errors? 
- The predictions may include texts that are not actually spoken in the audio input
%% 
However, because the models are trained in a weakly supervised manner using large-scale noisy data, the predictions may include texts that are not actually spoken in the audio input (i.e. hallucination). We hypothesize that this happens because, given their general knowledge of language, the models combine trying to predict the next word in audio with trying to transcribe the audio itself.
%%
- Lower accuracy on low-resource and/or low-discoverability languages or languages
%%
Our models perform unevenly across languages, and we observe lower accuracy on low-resource and/or low-discoverability languages or languages where we have less training data. The models also exhibit disparate performance on different accents and dialects of particular languages, which may include a higher word error rate across speakers of different genders, races, ages, or other demographic criteria. 
%%
- Prone to generating repetitive texts
%%
In addition, the sequence-to-sequence architecture of the model makes it prone to generating repetitive texts, which can be mitigated to some degree by beam search and temperature scheduling but not perfectly. It is likely that this behavior and hallucinations may be worse in lower-resource and/or lower-discoverability languages. 
%%
### Broader Impact
%%
Whisper models have potential in enhancing accessibility tools, with prospects of near-real-time speech recognition and translation, bearing significant economic implications. However, releasing Whisper raises dual-use concerns. While intended for beneficial applications, its accessibility could bolster surveillance capabilities, enabling efficient transcription and translation of vast audio communications. 
There's also potential for unintended individual recognition, posing safety concerns. The cost of transcription isn't deemed a primary constraint in scaling surveillance endeavors.
%%
We hope Whisper’s high accuracy and ease of use will allow developers to add voice interfaces to a much wider set of applications.
%%
## Seek for more?
### Research Index of OpenAI
![[Pasted image 20231023121544.png]]
Link: https://openai.com/research?contentTypes=milestone
%%
 that's pretty much it 
 the system is ultimately very elegant in the sense that it uses off-the-shelf Transformer module and basically just leverages huge data and some decoding heuristics tool
%%
## Resource links

1. Paper: <https://cdn.openai.com/papers/whisper.pdf>
2. Code: [https://github.com/openai/whisper](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbTUzOWdvQnBUYXdxektZQXZ0NUp4Z01EdEpGZ3xBQ3Jtc0tsSWlvalQ3MC1obVhsb3hzR2w4U09GejJtLWNBRlU1dGtHYXROX3NrUXN3c1lmcmNhMUN1Z0hFUjU1QXRjQnlJY0pxMlFOQlR1dlJjSEVOdzZBcTdNUV9MR2RRbVY4S05xcGlhbnNyX0ZMSVFpVUdsUQ&q=https%3A%2F%2Fgithub.com%2Fopenai%2Fwhisper&v=AwJf8aQfChE)
3. Model Card: https://github.com/openai/whisper/blob/main/model-card.md
4. Introducing Whisper: <https://openai.com/research/whisper>
5. Hugging Face Community<https://huggingface.co/openai/whisper-large>
## Citation for paper
- Chan et al. Simply mix all available speech recognition data to train one large neural network.
- Galvez et al. The people’s speech: A large-scale diverse english speech recognition dataset for commercial usage.
- Chen et al. Gigaspeech: An evolving, multi-domain asr corpus with 10,000 hours of transcribed audio.
- Baevski et al. wav2vec 2.0: A framework for self-supervised learning of speech representations.  
- Baevski er al.. Unsupervised speech recognition. Advances in Neural Information Processing Systems
- Zhang et al. BigSSL: Exploring the frontier of large-scale semi-supervised learning for automatic speech recognition.
