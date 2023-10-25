# Whisper
%%To begin with, whisper is tool
%%
It is a toolkit can do:
![[Pasted image 20231023140858.png]]
![[Pasted image 20231023140908.png]]
%%
 At first, I'm gonna first give you some high level overview of this work.
%%
## Overview
### Context
%%
The paper titled "Robust Speech Recognition by Large Scale Weak Supervision" explores the capabilities of speech processing systems trained on a vast amount of internet audio transcripts. With a whopping 680,000 hours of multilingual and multitask supervision, these models demonstrate impressive generalization capabilities. They perform competitively with previously supervised results in a zero-shot transfer setting, eliminating the need for fine-tuning. 
题为“大规模弱监督的鲁棒语音识别”的论文探讨了在大量互联网音频记录上训练的语音处理系统的功能。这些模型经过长达 680,000 小时的多语言和多任务监督，展示了令人印象深刻的泛化能力。它们在零样本传输设置中的表现与之前的监督结果具有竞争力，无需进行微调。
%%
#### Paper Title
"Robust Speech Recognition by Large Scale Weak Supervision"
#### Key Feature
- Generalization capabilities
- Transfer setting in zero-shot
- No need for fine-tuning
### Problem
%%
"Let's talk about the landscape of speech recognition. 
On one hand, we have unsupervised models trained on a whopping 1 million hours of data. Sounds impressive, right? But here's the catch: they still need fine-tuning to be truly effective.“让我们来谈谈语音识别的前景。一方面，我们拥有经过长达 100 万小时数据训练的无监督模型。听起来令人印象深刻，对吧？但问题是：它们仍然需要微调才能真正有效。
%%
- Unsupervised models: need fine-tuning to be truly effective
%%
On the other hand, we have supervised models. These are trained on much less data—just around 5,000 hours. But guess what? They're robust. They adapt well to new, unseen data, making them quite versatile.另一方面，我们有监督模型。这些数据的训练时间要少得多——只有大约 5,000 小时。但猜猜怎么了？他们很坚强。它们能够很好地适应新的、看不见的数据，因此用途广泛。
%%
- Supervised models: robust but not enough
%%
Now, you might be thinking, 'Why not have the best of both worlds?' That's where SpeechStew comes in. It combines 7 different datasets for a total of 5,140 hours of high-quality, supervised training. It's not as big as the 1 million hours of unsupervised data, but it's a step in the right direction.现在，您可能会想，“为什么不两全其美呢？”这就是 SpeechStew 的用武之地。它结合了 7 个不同的数据集，进行了总共 5,140 小时的高质量监督训练。它不像 100 万小时的无监督数据那么大，但它是朝着正确方向迈出的一步。
But what if we could push the envelope even further?
they've managed to scale up to between 10,000 and 30,000 hours of 'noisier' but still valuable training data. And that, folks, is how we're bridging the gap between quantity and quality in speech recognition."
但如果我们能进一步挑战极限呢？通过放松对完美、经过人工验证的成绩单的需求，我们成功地将“嘈杂”但仍然有价值的训练数据扩展到 10,000 到 30,000 小时。朋友们，这就是我们缩小语音识别数量和质量之间差距的方法。”
%%
### Approach
%%
"What sets whisper apart? We've scaled up—big time. We're talking about a massive 680,000 hours of labeled audio data, which is a significant leap from what's currently available.“
是什么让whisper与众不同？我们已经大规模扩大规模。我们正在谈论 680,000 小时的大量标记音频数据，这些数据与当前可用的产品相比，这是一个重大飞跃。
%%
- Scaling weakly supervised speech recognition the next order of magnitude to 680,000 hours of labeled audio data
%%
But that's not all. Our model isn't just about quantity; it's about quality. It performs exceptionally well on existing datasets without any need for fine-tuning. That means it's ready to go, right out of the box.但这还不是全部。我们的模型不仅仅关乎数量，还关乎数量。这是关于质量和多功能性。它在现有数据集上表现得非常好，无需任何微调。这意味着它开箱即用。
%%
- Without any need for fine-tuning
- The dataset is a global affair
%%
And we didn't stop at English. Our dataset is a global affair, covering 96 other languages with 117,000 hours of audio. Plus, we've got 125,000 hours dedicated to translation tasks. The best part? Our tests show that when they go big with the model size, there's no downside to training it on multiple languages and tasks. In fact, it's a win-win situation."我们并没有停留在英语上。我们的数据集是一个全球性的数据集，涵盖 96 种其他语言和 117,000 小时的音频。此外，我们还有 125,000 个小时专门用于翻译任务。最好的部分？我们的测试表明，当您加大模型大小时，在多种语言和任务上对其进行训练并没有什么坏处。事实上，这是一个双赢的局面。”
%%
## Architecture
### Overview of Architecture
%%
The architecture of the system is built upon a standard Transformer encoder-decoder module, a design inspired by the 2017 paper "Attention Is All You Need". The uniqueness of this research doesn't lie in the architecture, but in the data and the special tokens utilized.
该系统的架构建立在标准 Transformer 编码器-解码器模块的基础上，其设计灵感来自 2017 年论文《Attention Is All You Need》。这项研究的独特性不在于架构，而在于数据和所使用的特殊令牌。
The process begins with an audio file that is converted into log Mel spectrograms. These are visual representations of audio, where the x-axis depicts time, the y-axis shows frequency (in Mel scale, which a measure that mimics human pitch perception), and the color indicates amplitude.
该过程从转换为对数梅尔频谱图的音频文件开始。这些是音频的视觉表示，其中 x 轴表示时间，y 轴表示频率（以梅尔标度表示，这是一种模仿人类音调感知的对数度量），颜色表示幅度。
%%
![[Pasted image 20231023140015.png]]
%%
3. **Encoder Blocks**: The processed audio data is then sent through a series of encoder blocks. These are part of the Transformer architecture and are responsible for transforming the input data into a format that the decoder can understand.编码器块：处理后的音频数据然后通过一系列编码器块发送。它们是 Transformer 架构的一部分，负责将输入数据转换为解码器可以理解的格式。
4. **Decoder and Tokens**: The decoder's job is to predict what was said in the audio. To help it do this, various tokens are used. These tokens indicate things like the start of a transcript, the language being spoken, and the task (transcribing or translating). Time stamps can also be added to indicate when in the audio the spoken words occur.解码器和令牌：解码器的工作是预测音频中所说的内容。为了帮助它做到这一点，使用了各种令牌。这些标记表示转录的开始、所讲的语言和任务（转录或翻译）等内容。还可以添加时间戳来指示音频中说出的单词何时出现。
5. **Multi-Task Training**: The model is designed to handle multiple tasks. Sometimes it conditions its predictions based on previous text. This is useful for tasks like ongoing transcription where the audio might be part of a longer conversation.多任务训练：该模型旨在处理多个任务。有时它会根据之前的文本来调整其预测。这对于正在进行的转录等任务非常有用，其中音频可能是较长对话的一部分。
6. **Time Handling**: The model works with 30-second audio clips. If a spoken sentence spans multiple clips, the model adjusts the time window accordingly.时间处理：该模型适用于 30 秒的音频剪辑。如果一个口语句子跨越多个剪辑，模型会相应地调整时间窗口。
%%
%%
8. **Model Variants**: There are different sizes of the model, ranging from 'tiny' to 'large'. Each size has a trade-off between speed and accuracy. A newer version, "large V2," was recently released with some improvements.模型变体：模型有不同尺寸，范围从“微小”到“大”。每种尺寸都需要在速度和准确性之间进行权衡。最近发布了更新版本“large V2”，并进行了一些改进。
%%
### Discussion Question #1
How does Whisper handle multitasking?
%%
Whisper employs a multitask format that allows the model to perform various tasks like transcription, translation, voice activity detection, and language identification. All these tasks are specified as a sequence of input tokens to the decoder. This one-to-many mapping allows a single model to replace many different stages of a traditional speech processing pipeline. For example, the model can predict the language being spoken, whether there is speech in an audio segment, and whether to predict timestamps or not. This approach simplifies the system and potentially offers benefits in joint multilingual and multitask Whisper 采用多任务格式，允许模型执行各种任务，如转录、翻译、语音活动检测和语言识别。所有这些任务都被指定为解码器的输入标记序列。这种一对多映射允许单个模型替换传统语音处理管道的许多不同阶段。例如，该模型可以预测所说的语言、音频片段中是否有语音以及是否预测时间戳。这种方法简化了系统，并有可能为联合多语言和多任务培训带来好处。
%%
### Discussion Question #2
Why does Whisper use basic Transformers for its model?
%%
Whisper uses an off-the-shelf encoder-decoder Transformer architecture for its model. This choice was made to focus on studying the capabilities of large-scale supervised pre-training for speech recognition without confounding the findings with model improvements .
Whisper 的模型使用现成的编码器-解码器 Transformer 架构。做出这一选择是为了重点研究语音识别的大规模监督预训练的能力，而不会将研究结果与模型改进混淆
%%
%%
While the architecture is not the primary contribution of the paper, the dataset and the multitask setup with tokens certainly are. 
虽然架构不是本文的主要贡献，但数据集和带有标记的多任务设置肯定是。
%%
### Model Variants
![[Pasted image 20231024203345.png|600]]
%%
10. **Performance Metrics**: The model's performance is usually evaluated using Word Error Rate (WER), which measures how many mistakes the model makes. However, WER is not perfect and can be influenced by the specific style of the dataset used for evaluation.性能指标：模型的性能通常使用字错误率（WER）来评估，它衡量模型犯了多少错误。然而，WER 并不完美，可能会受到用于评估的数据集的特定风格的影响。
%%
### Performance Metrics
![[Pasted image 20231023124517.png|1000]]
%%
10. **Manual Adjustments**: To improve performance, some manual adjustments were made to better adapt the model to specific datasets.手动调整：为了提高性能，进行了一些手动调整以使模型更好地适应特定数据集。
%%
## Pseudocode
%%
After the intro to architecture, let's see through the Pseudocode.

The architecture comprises two main components: the audio encoder and the text decoder, as illustrated in the provided diagram.
该架构包括两个主要组件：音频编码器和文本解码器，如所提供的图表所示。
%%
![[Pasted image 20231023151017.png|750]]
%%
Now, let's enter the encoder to see how it looks. We can see that it has two 1D convolutional layers, which make up the stem of the encoder. Then, we have the positional embeddings, which are non-learnable and are sinusoids.“
现在，让我们进入编码器看看它是什么样子。我们可以看到它有两个一维卷积层，它们组成了然后，我们有位置嵌入，它是不可学习的并且是正弦曲线。
Moving on, we have a series of blocks that are simply residual attention blocks. There's nothing new there. Finally, we end up with layer normalization. Later, we'll see in the actual forward pass how the Mel spectrogram is processed through the 1D convolutional layers and then through the other components of the encoder.继续，我们有一系列的块，它们只是残余注意力块。那里没有什么新鲜事。最后，我们得到了层标准化。稍后，我们将在实际的前向传递中看到如何通过一维卷积层然后通过编码器的其他组件处理梅尔频谱图。
%%
![[Pasted image 20231023151033.png]]![[Pasted image 20231023151630.png]]
%%
Now, let's switch to the text decoder. Here, you can see a couple of things. First, there's an embedding table. Since we're dealing with a decoder model that has a vocabulary, we also have positional embeddings. Unlike the encoder, these are learned positional embeddings. 现在，让我们切换到文本解码器。在这里，您可以看到一些事情。首先，有一个嵌入表。由于我们正在处理具有词汇表的解码器模型，因此我们也有位置嵌入。与编码器不同，这些是学习的位置嵌入。
The decoder also has a series of blocks, but these include cross-attention, adding a third component to the standard Transformer layer. Finally, the decoder uses a causal mask, 
解码器也有一系列块，但其中包括交叉注意力，向标准 Transformer 层添加了第三个组件。最后，解码器使用因果掩码
if I were to print this you can see minus Infinities on the upper triangle which means we're going to mask out the future tokens and this particular token can only attend itself and the previous token. This is standard attention mechanism in Transformer models."
如果我要打印此内容，您可以在上三角形上看到负无穷大，这意味着我们将屏蔽未来的令牌，并且该特定令牌只能参加其自身以及前一个令牌和前一个令牌。这是 Transformer 模型中的标准注意力机制。”
%%
![[Pasted image 20231023151123.png]]
![[Pasted image 20231025100514.png|500]]
## Empirical Results
%%
Figure 2 in the paper illustrates the performance of zero-shot Whisper models in comparison to supervised LibriSpeech models and human performance in terms of Word Error Rate (WER) on various datasets. The key findings are:论文中的图 2 展示了零样本 Whisper 模型与监督 LibriSpeech 模型的性能比较以及人类在各种数据集上的词错误率 (WER) 方面的表现
%%
### Close the gap to human robustness
%%
Supervised LibriSpeech models, despite performing well on the LibriSpeech dev-clean dataset, make roughly twice as many errors as a human on other datasets. This highlights their lack of robustness and adaptability to different data distributions.
监督 LibriSpeech 模型尽管在 LibriSpeech dev-clean 数据集上表现良好，但在其他数据集上犯下的错误大约是人类的两倍。这凸显了它们缺乏对不同数据分布的稳健性和适应性。
%%
- Lack of Robustness in Supervised Models
%%
Zero-shot Whisper models come close to matching or even outperforming human performance on the LibriSpeech dev-clean dataset.
零样本 Whisper 模型在 LibriSpeech dev-clean 数据集上的表现接近甚至超过人类的表现。
%%
- Human-Like Robustness
![[Pasted image 20231025094800.png]]
### As good as human being?
Close to performance of professional human transcribers!
![[Pasted image 20231023123802.png]]
%%
This plot shows the WER distributions from the Kincaid46 dataset transcribed by Whisper, the same 4 commercial ASR systems from Figure 6 (A-D), one computer-assisted human transcription service (E) and 4 human transcription services (F-I). 
We can see that Whisper’s performance is close to that of professional human transcribers. 
该图显示了由 Whisper 转录的 Kincaid46 数据集的 25 个记录、图 6 (A-D) 中的 4 个商业 ASR 系统、一个计算机辅助人工转录服务 (E) 和 4 个人工转录服务 (F-I) 的 WER 分布。
Whisper 的性能接近专业人类转录员。
%%
## Code demonstration
https://colab.research.google.com/drive/1M8zNZ24lGcf05j-u53y73D-OhOv6z0I0?usp=drive_link#scrollTo=j9UgVYrod4SB
%%
Then I will demonstrate how Transcription be performed within Python:
Internally, the `transcribe()` method reads the entire file and processes the audio with a sliding 30-second window, performing autoregressive sequence-to-sequence predictions on each window.
在内部，“transcribe()”方法读取整个文件并使用 30 秒滑动窗口处理音频，在每个窗口上执行自回归序列到序列预测。
I use a taylor's popular song: back to December and to see how the model performs on transcribing the lyrics of the song with the background noise.
I tested 3 different types of models. 
Let's see the results
I am kind of confirmed to a certain extent that the larger the whisper model has better performance. For example, in base, punctuation cannot be transcribed, and there are some errors. Medium model, some non-existent characters appear in the transcribed text. In large models, the performance can be said to be close to perfect.
我测试了3个不同类型的模型。并且一定程度上确认，越大的whisper模型，拥有更好的表现。例如，在base中，不能转录标点符号，同时有一些错误。中型模型，转录文本出现了一些不存在的字符。大模型中，表现可以说接近完美。
Below is an example usage of `whisper.detect_language()` and `whisper.decode()` which provide lower-level access to the model.
下面是“whisper.detect_language()”和“whisper.decode()”的示例用法，它们提供对模型的较低级别访问。
I test this API on a k-pop song and it can successfully detect its language as korean.
%%
## Critical Analysis
### Were there any errors? 
- The predictions may include texts that are not actually spoken in the audio input
%% 
However, because the models are trained in a weakly supervised manner using large-scale noisy data, the predictions may include texts that are not actually spoken in the audio input (i.e. hallucination). We hypothesize that this happens because, given their general knowledge of language, the models combine trying to predict the next word in audio with trying to transcribe the audio itself.
然而，由于模型是使用大规模噪声数据以弱监督方式进行训练的，因此预测可能包括音频输入中实际未说出的文本（即幻觉）。我们假设发生这种情况是因为，考虑到模型对语言的一般知识，模型将预测音频中的下一个单词与尝试转录音频本身结合起来。
%%
- Lower accuracy on low-resource and/or low-discoverability languages or languages
%%
This models perform unevenly across languages, and we observe lower accuracy on low-resource and/or low-discoverability languages or languages where we have less training data. 
我们的模型在不同语言中的表现不均衡，并且我们观察到资源匮乏和/或低可发现性语言或训练数据较少的语言的准确性较低。
%%
- Prone to generating repetitive texts
%%
In addition, the sequence-to-sequence architecture of the model makes it prone to generating repetitive texts, which can be mitigated to some degree by beam search and temperature scheduling but not perfectly. It is likely that this behavior and hallucinations may be worse in lower-resource languages. 
此外，模型的序列到序列架构使其容易生成重复文本，可以通过波束搜索和温度调度在一定程度上缓解这种情况，但并不完美。在资源较低和/或可发现性较低的语言中，这种行为和幻觉可能会更糟。
%%
### Broader Impact
%%
Whisper models have potential in enhancing accessibility tools, with prospects of near-real-time speech recognition and translation, bearing significant economic implications. However, releasing Whisper raises dual-use concerns. While intended for beneficial applications, its accessibility could bolster surveillance capabilities, enabling efficient transcription and translation of vast audio communications. 
There's also potential for unintended individual recognition, posing safety concerns. The cost of transcription isn't deemed a primary constraint in scaling surveillance endeavors.
Whisper 模型具有增强可访问性工具的潜力，具有近实时语音识别和翻译的前景，具有重大的经济影响。然而，Whisper 的发布引发了双重用途的担忧。虽然其目的是为了有益的应用，但其可访问性可以增强监视能力，从而实现大量音频通信的高效转录和翻译。还有可能出现意外的个人识别，从而带来安全问题。
%%
We hope Whisper’s high accuracy and ease of use will allow developers to add voice interfaces to a much wider set of applications.
However,
it also raises dual-use concerns.
## Seek for more?
### Research Index of OpenAI
![[Pasted image 20231023121544.png]]
Link: https://openai.com/research?contentTypes=milestone
%%
 all in all,
 the system is ultimately very elegant in the sense that it uses off-the-shelf Transformer module and basically just leverages huge data and some decoding heuristics tool
%%
## Resource Links

1. Paper: <https://cdn.openai.com/papers/whisper.pdf>
2. Code: [https://github.com/openai/whisper](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbTUzOWdvQnBUYXdxektZQXZ0NUp4Z01EdEpGZ3xBQ3Jtc0tsSWlvalQ3MC1obVhsb3hzR2w4U09GejJtLWNBRlU1dGtHYXROX3NrUXN3c1lmcmNhMUN1Z0hFUjU1QXRjQnlJY0pxMlFOQlR1dlJjSEVOdzZBcTdNUV9MR2RRbVY4S05xcGlhbnNyX0ZMSVFpVUdsUQ&q=https%3A%2F%2Fgithub.com%2Fopenai%2Fwhisper&v=AwJf8aQfChE)
3. Model Card: https://github.com/openai/whisper/blob/main/model-card.md
4. Introducing Whisper: <https://openai.com/research/whisper>
5. Hugging Face Community<https://huggingface.co/openai/whisper-large>
## Citation For paper
- Chan et al. Simply mix all available speech recognition data to train one large neural network.
- Galvez et al. The people’s speech: A large-scale diverse english speech recognition dataset for commercial usage.
- Chen et al. Gigaspeech: An evolving, multi-domain asr corpus with 10,000 hours of transcribed audio.
- Baevski et al. wav2vec 2.0: A framework for self-supervised learning of speech representations.  
- Baevski er al. Unsupervised speech recognition. Advances in Neural Information Processing Systems
- Zhang et al. BigSSL: Exploring the frontier of large-scale semi-supervised learning for automatic speech recognition.
