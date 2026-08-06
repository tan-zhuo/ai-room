export type Lang = 'zh' | 'en' | 'ja'
export const LANGS: Lang[] = ['zh', 'en', 'ja']
export const LANG_LABEL: Record<Lang, string> = { zh: '中文', en: 'EN', ja: '日本語' }

type Dict = Record<string, string>

const en: Dict = {
  'app.title': 'AI ROOM',
  'app.tagline': 'Walk inside a living neural network',
  'arch.mlp': 'MLP',
  'arch.cnn': 'CNN',
  'arch.mlpFull': 'Multi-Layer Perceptron',
  'arch.cnnFull': 'Convolutional Network',
  'toast.comingSoon': 'More architectures (RNN / Transformer) coming soon',
  'controls.play': 'Play',
  'controls.pause': 'Pause',
  'controls.prev': 'Previous step',
  'controls.next': 'Next step',
  'controls.reset': 'Reset',
  'controls.speed': 'Speed',
  'controls.step': 'Step',
  'controls.input': 'Input',
  'controls.randomize': 'Random sample',
  'step.inputLoaded': 'Input loaded',
  'layer.input': 'Input',
  'layer.hidden': 'Hidden {n}',
  'layer.output': 'Output',
  'layer.conv': 'Convolution',
  'layer.pool': 'Max pooling',
  'layer.flatten': 'Flatten',
  'layer.dense': 'Dense',
  'layer.computing': 'computing…',
  'panel.neuron': 'Neuron',
  'panel.pixel': 'Pixel',
  'panel.featureMap': 'Feature map',
  'panel.channel': 'Channel',
  'panel.value': 'Value',
  'panel.inputs': 'Inputs',
  'panel.weight': 'Weight',
  'panel.product': 'Product',
  'panel.sum': 'Weighted sum',
  'panel.bias': 'Bias',
  'panel.preAct': 'Pre-activation z',
  'panel.activation': 'Activation',
  'panel.result': 'Output',
  'panel.kernel': 'Kernel',
  'panel.patch': 'Receptive field',
  'panel.products': 'Element-wise products',
  'panel.window': 'Pooling window',
  'panel.max': 'Max',
  'panel.mapsTo': 'Copied from',
  'panel.probabilities': 'Class probabilities',
  'panel.notYet': 'This layer has not run yet in the current playback — showing final values.',
  'panel.softmaxNote': 'softmax over the whole layer',
  'panel.position': 'Position',
  'panel.close': 'Close',
  'panel.prediction': 'Prediction',
  'class.mlp.0': 'Class α',
  'class.mlp.1': 'Class β',
  'class.mlp.2': 'Class γ',
  'class.cnn.0': 'Vertical',
  'class.cnn.1': 'Horizontal',
  'class.cnn.2': 'Diagonal',
  'class.cnn.3': 'Ring',
  'feature.n': 'Feature {n}',
  'legend.title': 'Legend',
  'legend.posWeight': 'Positive weight / activation',
  'legend.negWeight': 'Negative weight / activation',
  'legend.flow': 'Data flow',
  'legend.glow': 'Brightness = activation strength',
  'help.title': 'Keyboard shortcuts',
  'help.space': 'Play / Pause',
  'help.arrows': 'Previous / Next step',
  'help.r': 'Reset',
  'help.digits': 'Switch architecture',
  'help.l': 'Cycle language',
  'help.f': 'Focus selected node',
  'help.esc': 'Deselect / close panel',
  'help.mouse': 'Drag to orbit · scroll to zoom · right-drag to pan · click a node to inspect',
  'hint.click': 'Click any node to inspect its computation · click a layer title to learn what it does',
  'explain.what': 'What it does',
  'explain.why': 'Why the network needs it',
  'explain.simple': 'In plain words',
  'explain.tip': 'Tip: click any node in this layer to see its exact numbers.',
  'explain.input.what':
    'Holds the raw numbers the network receives — here a 4-dimensional feature vector. Each sphere is one feature value; no computation happens in this layer.',
  'explain.input.why':
    "A network needs a fixed, numeric entry point. Everything the model 'knows' about a sample must be encoded as these numbers before any layer can process them.",
  'explain.input.simple':
    "Think of it as the network's senses: four measurement dials whose readings are handed to the first layer.",
  'explain.hidden.what':
    'Each neuron multiplies every input by its own learned weight, adds them up together with a bias, then applies ReLU — positive values pass through, negative ones become zero.',
  'explain.hidden.why':
    'Stacking these layers lets the network bend and combine features into new ones. Without the nonlinear ReLU, any number of layers would collapse into a single linear formula and could never learn complex patterns.',
  'explain.hidden.simple':
    'A committee of little judges: each one weighs all the evidence differently, and only speaks up (fires) when its weighted sum comes out positive.',
  'explain.output.what':
    'A final dense layer produces one score (logit) per class, and softmax turns those scores into probabilities that add up to 1.',
  'explain.output.why':
    'Turning arbitrary scores into a probability distribution makes the answer easy to read and gives training a clear target to push toward.',
  'explain.output.simple':
    "The scoreboard: whichever class ends up with the highest probability is the network's prediction.",
  'explain.cnnInput.what':
    'An 8×8 grayscale image — each cube is one pixel, brightness is its value. This is the raw data the convolution will scan.',
  'explain.cnnInput.why':
    'Images are grids, and keeping them as grids (instead of flattening right away) lets the network exploit the fact that nearby pixels are related.',
  'explain.cnnInput.simple': 'A tiny picture, laid out as a wall of tiles for the network to look at.',
  'explain.conv.what':
    'A small 3×3 kernel slides across the image; at each position it multiplies the overlapping pixels by its weights and sums them, producing one pixel of a feature map. Each of the 3 kernels detects a different pattern (vertical, horizontal and diagonal edges), and ReLU keeps only the positive responses.',
  'explain.conv.why':
    'Reusing the same small filter everywhere needs far fewer weights than a dense layer — and it can spot its pattern no matter where in the image it appears (translation invariance).',
  'explain.conv.simple':
    "A magnifying glass swept over the picture: wherever the pattern it's looking for shows through, the feature map lights up.",
  'explain.pool.what':
    'Each 2×2 window of a feature map is replaced by its maximum value, shrinking the 6×6 maps down to 3×3.',
  'explain.pool.why':
    'It keeps the strongest evidence while shrinking the data, making the network faster and less sensitive to tiny shifts of the pattern.',
  'explain.pool.simple':
    'Like summarizing a paragraph by keeping only its strongest sentence — smaller, but the message survives.',
  'explain.flatten.what':
    'Rearranges the 3×3×3 stack of pooled feature maps into a single list of 27 numbers. No math happens — only the shape changes.',
  'explain.flatten.why':
    'Dense layers expect a flat vector, so this step is the bridge from spatial feature maps to the classifier.',
  'explain.flatten.simple':
    'Unrolling a stack of grids into one long line, ready for the final decision layers.',
  'explain.dense.what':
    'A fully connected layer: each of its 10 neurons looks at all 27 flattened features, weighs them, adds a bias and applies ReLU.',
  'explain.dense.why':
    'It mixes evidence from every location and every kernel into higher-level combinations that separate the classes.',
  'explain.dense.simple':
    'The deliberation room: all the collected clues are weighed together before the verdict.',
  'arch.text': 'Lang ID',
  'arch.textFull': 'Text Language Detector (built on an MLP)',
  'arch.llm': 'Transformer',
  'arch.llmFull': 'Tiny Transformer (char-level LLM)',
  'nav.models': 'Models',
  'nav.apps': 'AI Apps',
  'nav.menu': 'Menu',
  'nav.more': 'More models and apps coming…',
  'controls.draw': 'Draw',
  'controls.clear': 'Clear',
  'cnn.hand': 'Hand kernels',
  'cnn.handFull': 'Classic hand-crafted edge detectors (Sobel etc.); only the dense head is trained',
  'cnn.learned': 'Learned kernels',
  'cnn.learnedFull': 'Kernels start as random noise and are learned end-to-end via conv backprop',
  'llm.temp': 'Temp',
  'llm.tempTip': 'Sampling temperature — lower is safer, higher is wilder',
  'footer.blog': 'Author blog',
  'footer.source': 'Source code',
  'scale.s': 'S',
  'scale.m': 'M',
  'scale.l': 'L',
  'scale.tooltip': 'Network scale — rebuilds and retrains live',
  'toast.training': 'Retraining the network…',
  'controls.typeText': 'Type any text…',
  'controls.run': 'Run',
  'class.text.0': '中文',
  'class.text.1': 'English',
  'class.text.2': '日本語',
  'textfeat.0': 'Latin %',
  'textfeat.1': 'CJK %',
  'textfeat.2': 'Kana %',
  'textfeat.3': 'Digits %',
  'textfeat.4': 'Punct. %',
  'textfeat.5': 'Spaces %',
  'textfeat.6': 'Word length',
  'textfeat.7': 'Vowel ratio',
  'layer.tokens': 'Tokenizer',
  'layer.embed': 'Embedding',
  'layer.posenc': 'Positional encoding',
  'layer.attn': 'Multi-head attention',
  'layer.attnout': 'Weighted sum A·V',
  'layer.addnorm': 'Add & Norm',
  'layer.ffn': 'Feed-forward',
  'llm.head': 'Head {n}',
  'panel.mean': 'Mean μ',
  'panel.std': 'Std σ',
  'panel.residual': 'Residual: x + sublayer',
  'llm.next': 'Next char',
  'llm.generate': 'Generate',
  'llm.stop': 'Stop',
  'llm.masked': 'Masked (future token)',
  'llm.maskedNote':
    'Token i may only attend to tokens ≤ i — the model must not peek at the future it is trying to predict.',
  'llm.score': 'score',
  'llm.attnWeight': 'attention weight',
  'llm.topCandidates': 'Top candidates',
  'panel.token': 'Token',
  'panel.rawText': 'Input text',
  'explain.textInput.what':
    'The typed text is converted into 8 numeric features — the share of Latin letters, CJK characters, kana, digits, punctuation and spaces, plus average word length and vowel ratio. Only these numbers enter the network.',
  'explain.textInput.why':
    'Networks can only compute with numbers. Turning raw text into a fixed-length feature vector is the simplest form of encoding — real language models do the same job with learned embeddings.',
  'explain.textInput.simple':
    "Like describing a sentence over the phone: you don't read it aloud, you report a few statistics about it.",
  'explain.tokens.what':
    'The typed text is split into single characters; each becomes a token with an id from the 27-symbol vocabulary. The model always looks at the last 8 tokens.',
  'explain.tokens.why':
    'A network cannot read raw text — it needs a discrete, numbered alphabet to look things up in. Real LLMs do exactly this with subword tokens instead of characters.',
  'explain.tokens.simple': 'Cutting the sentence into tiles, one letter per tile.',
  'explain.embed.what':
    'Each token id looks up a learned 10-dimensional vector, and a learned position vector is added so the model knows where in the sequence each token sits.',
  'explain.embed.why':
    'Embeddings turn discrete symbols into coordinates the network can do math on; without positional vectors, "ab" and "ba" would look identical.',
  'explain.embed.simple':
    'Every letter gets a personality profile, plus a note about where it is standing in line.',
  'explain.qkv.what':
    "Each token's embedding is multiplied by three learned matrices, producing a query (what am I looking for?), a key (what do I contain?) and a value (what do I pass along?).",
  'explain.qkv.why':
    "Splitting the roles lets each token both ask questions about the context and answer other tokens' questions — the core trick behind attention.",
  'explain.qkv.simple': 'Every token writes a search query, a business card, and a package to hand over.',
  'explain.posenc.what':
    'Each position i gets a fixed sine/cosine pattern added to its embedding: even dimensions use sin, odd ones cos, at geometrically spaced frequencies. The result X = E + P is what the block actually processes.',
  'explain.posenc.why':
    'Attention by itself has no sense of order — it sees a bag of tokens. The positional pattern stamps "where am I" into every vector, and sinusoids let the model reason about relative distances.',
  'explain.posenc.simple':
    "Like seat numbers sewn into each letter's jacket — in a wave pattern the model can read.",
  'explain.addnorm.what':
    "Two steps: the sublayer's output is added back onto its input (residual connection), then each position's vector is normalized to zero mean and unit variance and rescaled by learned γ and β (LayerNorm).",
  'explain.addnorm.why':
    'Residuals give gradients a highway so deep stacks can train, and let each block learn only a correction. LayerNorm keeps the numbers in a healthy range so training stays stable.',
  'explain.addnorm.simple':
    'Keep the original, add the edits on top — then tidy everything back to a standard volume level.',
  'explain.attn.what':
    "Every query is dotted with every earlier key (scaled by 1/√dₕ) and each row becomes weights via softmax — independently in each of the 2 heads. Cell (i, j) of a head shows how much token i attends to token j under that head's learned criterion; future tokens are masked out.",
  'explain.attn.why':
    'This is how the model decides which earlier characters matter for predicting the next one — the pattern is learned, not hard-coded.',
  'explain.attn.simple':
    'A spotlight each letter shines back over the letters before it — brighter means more relevant.',
  'explain.attnout.what':
    "Each head averages its value vectors using its own attention row as weights; the heads' outputs are concatenated and passed through the output projection W_O, which mixes the heads back into one vector per token: zᵢ = concatₕ(Σⱼ Aₕ[i][j]·vⱼʰ)·W_O.",
  'explain.attnout.why':
    'This actually moves information between positions — the only place in the block where tokens exchange content.',
  'explain.attnout.simple': 'Each letter blends the packages it collected, in proportion to its spotlight.',
  'explain.ffn.what':
    "A small two-layer network (ReLU in the middle) transforms each position's vector independently.",
  'explain.ffn.why':
    'Attention mixes tokens; the feed-forward layer then processes what was gathered, adding nonlinear pattern-matching capacity.',
  'explain.ffn.simple': 'After gathering opinions, each letter thinks on its own for a moment.',
  'explain.llmOutput.what':
    "The last position's vector is projected onto every vocabulary symbol to give scores, and softmax turns them into next-character probabilities.",
  'explain.llmOutput.why':
    'Language modeling is just "predict the next symbol" — sampling from this distribution over and over is how LLMs write text.',
  'explain.llmOutput.simple':
    'The model finishes your sentence: one letter at a time, by voting over the alphabet.',
  'arch.rnn': 'RNN',
  'arch.rnnFull': 'Recurrent Neural Network (Elman)',
  'arch.lstm': 'LSTM',
  'arch.lstmFull': 'Long Short-Term Memory',
  'arch.ae': 'Autoencoder',
  'arch.aeFull': 'Autoencoder — compress & reconstruct',
  'layer.rnnHidden': 'Hidden state',
  'layer.gates': 'Gates f · i · g · o',
  'layer.cell': 'Cell state',
  'layer.encoder': 'Encoder',
  'layer.latent': 'Latent code',
  'layer.decoder': 'Decoder',
  'layer.recon': 'Reconstruction',
  'ae.compare': 'Original vs reconstruction',
  'ae.original': 'original',
  'ae.recon': 'reconstructed',
  'explain.rnnHidden.what':
    'The sheet is read row by row: at each timestep the hidden state is recomputed as h_t = tanh(Wx·x_t + Wh·h_{t-1} + b) — the same two weight matrices reused at every step.',
  'explain.rnnHidden.why':
    'h_{t-1} feeding back in is what gives the network memory: information from earlier characters survives inside the hidden vector. Weight sharing across time keeps the model small.',
  'explain.rnnHidden.simple':
    'Reading a sentence while keeping a single running note that you update after every word.',
  'explain.gates.what':
    'Four small networks read [x_t, h_{t-1}] in parallel: forget gate f and input gate i (sigmoid, 0–1), candidate g (tanh) and output gate o (sigmoid).',
  'explain.gates.why':
    'Plain RNNs forget quickly — gradients vanish over long spans. Gates let the network explicitly decide what to erase, what to write and what to reveal, which is the LSTM fix.',
  'explain.gates.simple':
    'Four dials on a memory box: how much to wipe, how much to write, what to write, and how much to show.',
  'explain.cellstate.what':
    'The long-term memory: c_t = f⊙c_{t-1} + i⊙g. Old content is scaled by the forget gate, new content is added through the input gate.',
  'explain.cellstate.why':
    'Because updates are additive, gradients flow along the cell state almost unchanged — the "conveyor belt" that lets LSTMs remember across many steps.',
  'explain.cellstate.simple': 'A notebook that is partially erased and appended to at every word.',
  'explain.lstmHidden.what':
    'The public face of the memory: h_t = o⊙tanh(c_t). The output gate chooses how much of the cell state is shown to the next layer and the next timestep.',
  'explain.lstmHidden.why':
    'Keeping the internal memory (c) separate from what is exposed (h) lets the network store things it does not want to act on yet.',
  'explain.lstmHidden.simple': 'What you say out loud, versus everything you keep in your head.',
  'explain.aeInput.what':
    'An 8×8 image, flattened into 64 numbers. No labels exist anywhere in this architecture — the input itself is also the training target.',
  'explain.aeInput.why':
    'Autoencoders learn without labels (self-supervised): the task "reproduce your input" forces the network to discover the structure of the data on its own.',
  'explain.aeInput.simple': 'The exam question and the answer sheet are the same picture.',
  'explain.encoder.what':
    'Dense layers that squeeze the 64 input numbers down step by step. Everything the network keeps must fit through here.',
  'explain.encoder.why':
    'Compression forces abstraction: to reconstruct well through a tiny bottleneck, the encoder must keep the essence (which pattern? where?) and drop pixel noise.',
  'explain.encoder.simple': 'Summarizing a picture in a few words before handing it over.',
  'explain.latent.what':
    'The bottleneck: the whole image is now just these few numbers (a 6-dimensional code, tanh-bounded).',
  'explain.latent.why':
    'This is the learned representation — similar inputs land on nearby codes. The same idea powers image compression and generative models, which sample new codes here.',
  'explain.latent.simple':
    "The picture's DNA: a handful of numbers that describe everything worth keeping.",
  'explain.decoder.what':
    'Dense layers that expand the latent code back to 64 numbers, ending in a sigmoid so every reconstructed pixel lands in 0–1.',
  'explain.decoder.why':
    'The decoder proves the code was meaningful: if 6 numbers suffice to redraw the image, the encoder truly captured its structure.',
  'explain.decoder.simple': 'Redrawing the picture from the few words of the summary.',
  'explain.recon.what':
    'The reconstructed image. Training minimizes the mean squared error between this and the input — compare them side by side.',
  'explain.recon.why':
    'Reconstruction error is the whole training signal — no labels needed. Inputs unlike the training data reconstruct badly, which is why autoencoders also detect anomalies.',
  'explain.recon.simple':
    'The picture after a round trip through the bottleneck — close, but never pixel-perfect.',
}

const zh: Dict = {
  'app.title': 'AI ROOM',
  'app.tagline': '走进一个正在运行的神经网络',
  'arch.mlp': 'MLP',
  'arch.cnn': 'CNN',
  'arch.mlpFull': '多层感知机',
  'arch.cnnFull': '卷积神经网络',
  'toast.comingSoon': '更多架构（RNN / Transformer）即将推出',
  'controls.play': '播放',
  'controls.pause': '暂停',
  'controls.prev': '上一步',
  'controls.next': '下一步',
  'controls.reset': '重置',
  'controls.speed': '速度',
  'controls.step': '步骤',
  'controls.input': '输入',
  'controls.randomize': '随机样本',
  'step.inputLoaded': '输入已载入',
  'layer.input': '输入层',
  'layer.hidden': '隐藏层 {n}',
  'layer.output': '输出层',
  'layer.conv': '卷积层',
  'layer.pool': '最大池化',
  'layer.flatten': '展平',
  'layer.dense': '全连接层',
  'layer.computing': '计算中…',
  'panel.neuron': '神经元',
  'panel.pixel': '像素',
  'panel.featureMap': '特征图',
  'panel.channel': '通道',
  'panel.value': '数值',
  'panel.inputs': '输入',
  'panel.weight': '权重',
  'panel.product': '乘积',
  'panel.sum': '加权和',
  'panel.bias': '偏置',
  'panel.preAct': '激活前 z',
  'panel.activation': '激活函数',
  'panel.result': '输出',
  'panel.kernel': '卷积核',
  'panel.patch': '感受野',
  'panel.products': '逐元素乘积',
  'panel.window': '池化窗口',
  'panel.max': '最大值',
  'panel.mapsTo': '来自',
  'panel.probabilities': '类别概率',
  'panel.notYet': '当前播放尚未运行到该层——显示的是最终值。',
  'panel.softmaxNote': '对整层做 softmax',
  'panel.position': '位置',
  'panel.close': '关闭',
  'panel.prediction': '预测',
  'class.mlp.0': '类别 α',
  'class.mlp.1': '类别 β',
  'class.mlp.2': '类别 γ',
  'class.cnn.0': '竖线',
  'class.cnn.1': '横线',
  'class.cnn.2': '斜线',
  'class.cnn.3': '圆环',
  'feature.n': '特征 {n}',
  'legend.title': '图例',
  'legend.posWeight': '正权重 / 正激活',
  'legend.negWeight': '负权重 / 负激活',
  'legend.flow': '数据流',
  'legend.glow': '亮度 = 激活强度',
  'help.title': '键盘快捷键',
  'help.space': '播放 / 暂停',
  'help.arrows': '上一步 / 下一步',
  'help.r': '重置',
  'help.digits': '切换网络架构',
  'help.l': '切换语言',
  'help.f': '聚焦选中节点',
  'help.esc': '取消选择 / 关闭面板',
  'help.mouse': '拖拽旋转 · 滚轮缩放 · 右键平移 · 点击节点查看计算',
  'hint.click': '点击任意节点查看其计算过程 · 点击层标题了解该模块的作用',
  'explain.what': '它做什么',
  'explain.why': '为什么需要它',
  'explain.simple': '通俗理解',
  'explain.tip': '提示：点击该层的任意节点可查看具体数值。',
  'explain.input.what':
    '输入层保存网络接收到的原始数字——这里是一个 4 维特征向量。每个球体代表一个特征值，这一层不做任何计算。',
  'explain.input.why':
    '网络需要一个固定的数字入口。模型对样本的全部“认知”都必须先编码成这些数字，后面的层才能处理。',
  'explain.input.simple': '可以把它当作网络的感官：四个测量表盘，读数被交给第一层。',
  'explain.hidden.what':
    '每个神经元把所有输入分别乘以自己学到的权重，加总后再加上偏置，然后通过 ReLU 激活——正值保留，负值归零。',
  'explain.hidden.why':
    '层层堆叠让网络能把特征弯曲、组合成新的特征。如果没有非线性的 ReLU，再多的层也会塌缩成一个线性公式，学不会复杂模式。',
  'explain.hidden.simple':
    '像一群小评委：每人对证据的加权方式不同，只有加权和为正时才会“发言”（激活）。',
  'explain.output.what':
    '最后一个全连接层为每个类别产生一个得分（logit），softmax 把这些得分转换成总和为 1 的概率。',
  'explain.output.why':
    '把任意大小的得分变成概率分布，结果一目了然，也给训练提供了明确的优化目标。',
  'explain.output.simple': '记分牌：概率最高的类别就是网络的预测。',
  'explain.cnnInput.what':
    '一张 8×8 的灰度图——每个方块是一个像素，亮度即数值。这是卷积将要扫描的原始数据。',
  'explain.cnnInput.why':
    '图像本来就是网格。保持网格形状（而不是立刻展平）让网络能利用“相邻像素相关”这一事实。',
  'explain.cnnInput.simple': '一张小图片，铺成一面瓷砖墙，等着网络来观察。',
  'explain.conv.what':
    '一个 3×3 的卷积核在图像上滑动；每个位置把覆盖到的像素与核内权重相乘再求和，得到特征图上的一个像素。3 个核各自检测不同模式（竖直、水平、对角边缘），ReLU 只保留正响应。',
  'explain.conv.why':
    '同一个小滤波器在全图复用，参数远少于全连接层；而且无论模式出现在图像哪个位置都能被发现（平移不变性）。',
  'explain.conv.simple': '像拿放大镜扫过图片：镜片要找的图案在哪儿透出来，特征图上哪儿就会亮起。',
  'explain.pool.what': '特征图上每个 2×2 窗口被其中的最大值取代，把 6×6 的特征图缩小成 3×3。',
  'explain.pool.why':
    '在压缩数据的同时保留最强的证据，让网络更快，也对图案的微小位移更不敏感。',
  'explain.pool.simple': '像给段落做摘要，只留下最有力的一句话——变小了，但信息还在。',
  'explain.flatten.what':
    '把 3×3×3 的池化特征图堆叠重新排成一条 27 个数字的列表。没有数学运算，只是形状变化。',
  'explain.flatten.why': '全连接层需要一维向量输入，这一步是空间特征图通向分类器的桥梁。',
  'explain.flatten.simple': '把一摞网格摊开拉成一条长线，交给最终的决策层。',
  'explain.dense.what':
    '全连接层：10 个神经元中的每一个都查看全部 27 个展平特征，加权求和、加偏置，再经过 ReLU。',
  'explain.dense.why':
    '它把来自每个位置、每个卷积核的证据混合起来，形成能区分类别的高层组合。',
  'explain.dense.simple': '合议室：所有收集到的线索在这里一起权衡，然后给出结论。',
  'arch.text': '语言识别',
  'arch.textFull': '文本语言识别（基于 MLP 的应用）',
  'arch.llm': 'Transformer',
  'arch.llmFull': '迷你 Transformer（字符级 LLM）',
  'nav.models': '模型',
  'nav.apps': 'AI 应用',
  'nav.menu': '菜单',
  'nav.more': '更多模型与应用陆续加入…',
  'controls.draw': '绘制',
  'controls.clear': '清空',
  'cnn.hand': '手工核',
  'cnn.handFull': '经典手工边缘检测核（Sobel 等），只训练全连接头',
  'cnn.learned': '学习核',
  'cnn.learnedFull': '卷积核从随机噪声开始，通过卷积反向传播端到端学习',
  'llm.temp': '温度',
  'llm.tempTip': '采样温度——越低越保守，越高越放飞',
  'footer.blog': '作者博客',
  'footer.source': '开源代码',
  'scale.s': '小',
  'scale.m': '中',
  'scale.l': '大',
  'scale.tooltip': '网络规模——实时重建并重新训练',
  'toast.training': '正在重新训练网络…',
  'controls.typeText': '输入任意文字…',
  'controls.run': '运行',
  'class.text.0': '中文',
  'class.text.1': 'English',
  'class.text.2': '日本語',
  'textfeat.0': '拉丁字母占比',
  'textfeat.1': '汉字占比',
  'textfeat.2': '假名占比',
  'textfeat.3': '数字占比',
  'textfeat.4': '标点占比',
  'textfeat.5': '空格占比',
  'textfeat.6': '平均词长',
  'textfeat.7': '元音比例',
  'layer.tokens': '分词器',
  'layer.embed': '嵌入层',
  'layer.posenc': '位置编码',
  'layer.attn': '多头注意力',
  'layer.attnout': '加权求和 A·V',
  'layer.addnorm': '残差 + LayerNorm',
  'layer.ffn': '前馈层',
  'llm.head': '注意力头 {n}',
  'panel.mean': '均值 μ',
  'panel.std': '标准差 σ',
  'panel.residual': '残差：x + 子层输出',
  'llm.next': '下一个字符',
  'llm.generate': '连续生成',
  'llm.stop': '停止',
  'llm.masked': '已掩码（未来 token）',
  'llm.maskedNote': 'token i 只能关注 ≤ i 的 token——模型不能偷看它正要预测的未来。',
  'llm.score': '得分',
  'llm.attnWeight': '注意力权重',
  'llm.topCandidates': '最高候选',
  'panel.token': 'Token',
  'panel.rawText': '输入文本',
  'explain.textInput.what':
    '输入的文字被转换成 8 个数值特征——拉丁字母、汉字、假名、数字、标点、空格的占比，以及平均词长和元音比例。进入网络的只有这些数字。',
  'explain.textInput.why':
    '神经网络只能对数字做计算。把原始文本变成定长特征向量是最简单的"编码"方式——真正的语言模型用学习到的词向量做同样的事。',
  'explain.textInput.simple': '就像打电话向网络描述一句话：不是逐字朗读，而是报出几个统计数字。',
  'explain.tokens.what':
    '输入的文字被切分成单个字符，每个字符成为一个 token，并从 27 个符号的词表中获得一个编号。模型始终查看最后 8 个 token。',
  'explain.tokens.why':
    '网络无法直接阅读原始文本——它需要一个离散、有编号的字母表来查表。真正的 LLM 用的是子词（subword）token，原理相同。',
  'explain.tokens.simple': '把句子剪成一块块瓷砖，每块放一个字母。',
  'explain.embed.what':
    '每个 token 编号查出一个学习到的 10 维向量，再加上位置向量，让模型知道每个 token 在序列中的位置。',
  'explain.embed.why':
    '嵌入把离散符号变成可以做数学运算的坐标；没有位置向量，"ab" 和 "ba" 看起来会完全一样。',
  'explain.embed.simple': '每个字母领到一份性格档案，外加一张写着自己排队位置的纸条。',
  'explain.qkv.what':
    '每个 token 的嵌入分别乘以三个学习到的矩阵，得到查询 Query（我在找什么？）、键 Key（我包含什么？）和值 Value（我要传递什么？）。',
  'explain.qkv.why':
    '角色分离让每个 token 既能对上下文提问，也能回答其他 token 的提问——这是注意力机制的核心技巧。',
  'explain.qkv.simple': '每个 token 写下一条搜索请求、一张名片和一个待转交的包裹。',
  'explain.posenc.what':
    '每个位置 i 都有一组固定的正弦/余弦波形加到它的嵌入上：偶数维用 sin、奇数维用 cos，频率按几何级数排布。相加结果 X = E + P 才是后续模块真正处理的输入。',
  'explain.posenc.why':
    '注意力本身没有顺序概念——它看到的只是一袋 token。位置编码把"我在第几个位置"印进每个向量，正弦波形还让模型能推理相对距离。',
  'explain.posenc.simple': '像给每个字母的外套缝上座位号，而且是用模型读得懂的波纹绣出来的。',
  'explain.addnorm.what':
    '两步：先把子层的输出加回它的输入（残差连接），再把每个位置的向量归一化到均值 0、方差 1，并用学习到的 γ 和 β 重新缩放（LayerNorm）。',
  'explain.addnorm.why':
    '残差给梯度开了一条高速公路，深层堆叠才能训得动，每个模块也只需学"修正量"；LayerNorm 把数值拉回健康范围，让训练保持稳定。',
  'explain.addnorm.simple': '保留原稿、把修改意见叠加上去——然后把整体音量调回标准值。',
  'explain.attn.what':
    '每个查询与所有更早的键做点积（除以 √dₕ 缩放），每行经 softmax 变成权重——两个注意力头各自独立进行。某个头的格子 (i, j) 表示在该头学到的标准下 token i 对 token j 的关注程度；未来的 token 被掩码遮住。',
  'explain.attn.why':
    '模型正是通过它来决定哪些更早的字符对预测下一个字符重要——这个模式是学出来的，不是写死的。',
  'explain.attn.simple': '每个字母向前面的字母打一束聚光灯——越亮表示越相关。',
  'explain.attnout.what':
    '每个头用自己的注意力行对值向量加权平均，两个头的结果拼接后再经过输出投影 W_O，把各头的信息混合回每个 token 一个向量：zᵢ = concatₕ(Σⱼ Aₕ[i][j]·vⱼʰ)·W_O。',
  'explain.attnout.why':
    '这一步真正在位置之间搬运信息——是整个模块中 token 之间唯一交换内容的地方。',
  'explain.attnout.simple': '每个字母按聚光灯的亮度比例，把收到的包裹混合在一起。',
  'explain.ffn.what': '一个小型两层网络（中间是 ReLU）独立地变换每个位置的向量。',
  'explain.ffn.why': '注意力负责混合 token；前馈层随后加工收集到的信息，提供非线性的模式识别能力。',
  'explain.ffn.simple': '收集完大家的意见后，每个字母自己静静思考一会儿。',
  'explain.llmOutput.what':
    '最后一个位置的向量被投影到词表的每个符号上得到分数，softmax 把分数变成下一个字符的概率。',
  'explain.llmOutput.why':
    '语言建模就是"预测下一个符号"——反复从这个分布中采样，LLM 就是这样写出文字的。',
  'explain.llmOutput.simple': '模型接着写你的句子：对字母表投票，一次写一个字母。',
  'arch.rnn': 'RNN',
  'arch.rnnFull': '循环神经网络（Elman）',
  'arch.lstm': 'LSTM',
  'arch.lstmFull': '长短期记忆网络',
  'arch.ae': '自编码器',
  'arch.aeFull': '自编码器——压缩与重建',
  'layer.rnnHidden': '隐状态',
  'layer.gates': '门控 f · i · g · o',
  'layer.cell': '细胞状态',
  'layer.encoder': '编码器',
  'layer.latent': '潜向量',
  'layer.decoder': '解码器',
  'layer.recon': '重建输出',
  'ae.compare': '原图 vs 重建',
  'ae.original': '原始值',
  'ae.recon': '重建值',
  'explain.rnnHidden.what':
    '这张表逐行计算：每个时间步的隐状态由 h_t = tanh(Wx·x_t + Wh·h_{t-1} + b) 重新算出——每一步复用同样的两个权重矩阵。',
  'explain.rnnHidden.why':
    'h_{t-1} 的回流就是网络的记忆：更早字符的信息保存在隐向量里向后传递。跨时间共享权重也让模型保持小巧。',
  'explain.rnnHidden.simple': '像一边读句子一边只记一张便签，每读一个词就更新一次。',
  'explain.gates.what':
    '四个小网络并行读取 [x_t, h_{t-1}]：遗忘门 f 和输入门 i（sigmoid，0–1），候选值 g（tanh），输出门 o（sigmoid）。',
  'explain.gates.why':
    '普通 RNN 忘得快——梯度在长跨度上会消失。门控让网络明确决定擦什么、写什么、露什么，这正是 LSTM 的解法。',
  'explain.gates.simple': '记忆盒上的四个旋钮：擦多少、写多少、写什么、给别人看多少。',
  'explain.cellstate.what':
    '长期记忆：c_t = f⊙c_{t-1} + i⊙g。旧内容被遗忘门缩放，新内容经输入门加入。',
  'explain.cellstate.why':
    '因为更新是加法式的，梯度沿细胞状态几乎无损流动——这条"传送带"让 LSTM 能记住很多步之前的信息。',
  'explain.cellstate.simple': '一本每读一个词就擦掉一部分、再补写几行的笔记本。',
  'explain.lstmHidden.what':
    '记忆的公开面孔：h_t = o⊙tanh(c_t)。输出门决定细胞状态里有多少展示给下一层和下一步。',
  'explain.lstmHidden.why':
    '把内部记忆（c）和对外输出（h）分开，网络就能存住暂时不想拿出来用的信息。',
  'explain.lstmHidden.simple': '你说出口的话，和你脑子里记着的全部内容，是两回事。',
  'explain.aeInput.what':
    '一张 8×8 图片，展平成 64 个数。整个架构里没有任何标签——输入本身就是训练目标。',
  'explain.aeInput.why':
    '自编码器无需标签（自监督）："复原你的输入"这个任务逼着网络自己发现数据的结构。',
  'explain.aeInput.simple': '考题和标准答案是同一张图。',
  'explain.encoder.what': '全连接层把 64 个输入数字逐步压缩。网络想保留的一切都必须从这里挤过去。',
  'explain.encoder.why':
    '压缩迫使抽象：要想经过狭窄的瓶颈还能重建，编码器必须保留本质（什么图案？在哪里？），丢掉像素噪声。',
  'explain.encoder.simple': '把一张图先总结成几句话，再交出去。',
  'explain.latent.what': '瓶颈：整张图现在只剩这几个数（6 维编码，tanh 约束在 ±1 内）。',
  'explain.latent.why':
    '这就是学到的表示——相似的输入落在相近的编码上。图像压缩和生成模型用的正是同一个思想：生成模型就是在这里采样新编码。',
  'explain.latent.simple': '这张图的 DNA：几个数字描述了所有值得保留的东西。',
  'explain.decoder.what':
    '全连接层把潜向量展开回 64 个数，最后用 sigmoid 让每个重建像素都落在 0–1。',
  'explain.decoder.why':
    '解码器证明编码是有意义的：如果 6 个数就够重画整张图，说明编码器真的抓住了结构。',
  'explain.decoder.simple': '凭那几句话的总结，把图重新画出来。',
  'explain.recon.what': '重建出的图片。训练就是最小化它与输入之间的均方误差——可以并排对比。',
  'explain.recon.why':
    '重建误差就是全部训练信号——不需要标签。与训练数据不像的输入会重建得很差，所以自编码器也常用来做异常检测。',
  'explain.recon.simple': '穿过瓶颈走了一个来回的图片——很接近，但永远不会逐像素一模一样。',
}

const ja: Dict = {
  'app.title': 'AI ROOM',
  'app.tagline': '動いているニューラルネットワークの中へ',
  'arch.mlp': 'MLP',
  'arch.cnn': 'CNN',
  'arch.mlpFull': '多層パーセプトロン',
  'arch.cnnFull': '畳み込みネットワーク',
  'toast.comingSoon': 'その他のアーキテクチャ（RNN / Transformer）は近日公開',
  'controls.play': '再生',
  'controls.pause': '一時停止',
  'controls.prev': '前のステップ',
  'controls.next': '次のステップ',
  'controls.reset': 'リセット',
  'controls.speed': '速度',
  'controls.step': 'ステップ',
  'controls.input': '入力',
  'controls.randomize': 'ランダムサンプル',
  'step.inputLoaded': '入力を読み込みました',
  'layer.input': '入力層',
  'layer.hidden': '隠れ層 {n}',
  'layer.output': '出力層',
  'layer.conv': '畳み込み層',
  'layer.pool': '最大プーリング',
  'layer.flatten': '平坦化',
  'layer.dense': '全結合層',
  'layer.computing': '計算中…',
  'panel.neuron': 'ニューロン',
  'panel.pixel': 'ピクセル',
  'panel.featureMap': '特徴マップ',
  'panel.channel': 'チャネル',
  'panel.value': '値',
  'panel.inputs': '入力',
  'panel.weight': '重み',
  'panel.product': '積',
  'panel.sum': '加重和',
  'panel.bias': 'バイアス',
  'panel.preAct': '活性化前 z',
  'panel.activation': '活性化関数',
  'panel.result': '出力',
  'panel.kernel': 'カーネル',
  'panel.patch': '受容野',
  'panel.products': '要素ごとの積',
  'panel.window': 'プーリング窓',
  'panel.max': '最大値',
  'panel.mapsTo': 'コピー元',
  'panel.probabilities': 'クラス確率',
  'panel.notYet': '現在の再生ではこの層はまだ実行されていません — 最終値を表示しています。',
  'panel.softmaxNote': '層全体の softmax',
  'panel.position': '位置',
  'panel.close': '閉じる',
  'panel.prediction': '予測',
  'class.mlp.0': 'クラス α',
  'class.mlp.1': 'クラス β',
  'class.mlp.2': 'クラス γ',
  'class.cnn.0': '縦線',
  'class.cnn.1': '横線',
  'class.cnn.2': '斜線',
  'class.cnn.3': 'リング',
  'feature.n': '特徴 {n}',
  'legend.title': '凡例',
  'legend.posWeight': '正の重み / 活性化',
  'legend.negWeight': '負の重み / 活性化',
  'legend.flow': 'データフロー',
  'legend.glow': '明るさ = 活性化の強さ',
  'help.title': 'キーボードショートカット',
  'help.space': '再生 / 一時停止',
  'help.arrows': '前 / 次のステップ',
  'help.r': 'リセット',
  'help.digits': 'アーキテクチャ切替',
  'help.l': '言語切替',
  'help.f': '選択ノードにフォーカス',
  'help.esc': '選択解除 / パネルを閉じる',
  'help.mouse': 'ドラッグで回転 · ホイールでズーム · 右ドラッグで移動 · ノードをクリックで詳細',
  'hint.click': 'ノードをクリックで計算過程を確認 · 層のタイトルをクリックでモジュール解説',
  'explain.what': '何をするか',
  'explain.why': 'なぜ必要か',
  'explain.simple': 'やさしく言うと',
  'explain.tip': 'ヒント：この層のノードをクリックすると実際の数値が見られます。',
  'explain.input.what':
    'ネットワークが受け取る生の数値を保持します——ここでは 4 次元の特徴ベクトルです。各球体が 1 つの特徴値で、この層では計算は行われません。',
  'explain.input.why':
    'ネットワークには固定された数値の入口が必要です。サンプルに関する情報はすべて、この数値として符号化されて初めて各層で処理できます。',
  'explain.input.simple':
    'ネットワークの感覚器官のようなもの：4 つの計器の読みが最初の層に手渡されます。',
  'explain.hidden.what':
    '各ニューロンはすべての入力に学習した重みを掛けて合計し、バイアスを加えた後、ReLU（正の値はそのまま、負の値はゼロ）を適用します。',
  'explain.hidden.why':
    '層を重ねることで特徴を曲げたり組み合わせたりして新しい特徴を作れます。非線形な ReLU がなければ、何層あっても 1 つの線形式に潰れてしまい、複雑なパターンを学べません。',
  'explain.hidden.simple':
    '小さな審査員の集まり：それぞれ証拠の重み付けが異なり、加重和が正のときだけ「発言」（発火）します。',
  'explain.output.what':
    '最後の全結合層がクラスごとのスコア（ロジット）を出し、softmax がそれを合計 1 の確率に変換します。',
  'explain.output.why':
    '任意のスコアを確率分布にすることで結果が読みやすくなり、学習の明確な目標にもなります。',
  'explain.output.simple': 'スコアボード：最も確率が高いクラスがネットワークの予測です。',
  'explain.cnnInput.what':
    '8×8 のグレースケール画像——各キューブが 1 ピクセルで、明るさが値です。畳み込みが走査する生データです。',
  'explain.cnnInput.why':
    '画像はもともと格子です。すぐに平坦化せず格子のまま保つことで、「近くのピクセルは関係が深い」という性質を活かせます。',
  'explain.cnnInput.simple': '小さな絵をタイルの壁として並べ、ネットワークに見せているところです。',
  'explain.conv.what':
    '3×3 のカーネルが画像上をスライドし、各位置で重なったピクセルと重みを掛けて合計し、特徴マップの 1 ピクセルを作ります。3 つのカーネルはそれぞれ別のパターン（縦・横・斜めのエッジ）を検出し、ReLU が正の応答だけを残します。',
  'explain.conv.why':
    '同じ小さなフィルタを画像全体で使い回すため、全結合層よりはるかに少ないパラメータで済み、パターンが画像のどこに現れても検出できます（平行移動不変性）。',
  'explain.conv.simple':
    '虫眼鏡で絵の上を掃くように：探しているパターンが透けて見える場所が光ります。',
  'explain.pool.what':
    '特徴マップの各 2×2 窓をその最大値で置き換え、6×6 のマップを 3×3 に縮小します。',
  'explain.pool.why':
    'データを圧縮しながら最も強い証拠を残すので、ネットワークが速くなり、パターンの小さなずれにも強くなります。',
  'explain.pool.simple':
    '段落を一番力強い一文だけ残して要約するようなもの——小さくなっても意味は残ります。',
  'explain.flatten.what':
    '3×3×3 のプーリング済み特徴マップを、27 個の数値が並ぶ 1 本のリストに並べ替えます。計算はなく、形が変わるだけです。',
  'explain.flatten.why':
    '全結合層は 1 次元ベクトルを入力とするため、空間的な特徴マップと分類器をつなぐ橋になります。',
  'explain.flatten.simple':
    '積み重なった格子をほどいて 1 本の長い列にし、最終判断の層へ渡します。',
  'explain.dense.what':
    '全結合層：10 個のニューロンそれぞれが 27 個の平坦化された特徴すべてを見て、重み付けして合計し、バイアスを加えて ReLU を通します。',
  'explain.dense.why':
    'あらゆる位置・あらゆるカーネルからの証拠を混ぜ合わせ、クラスを分ける高次の組み合わせを作ります。',
  'explain.dense.simple': '評議室：集めた手がかりをまとめて検討し、結論を出す場所です。',
  'arch.text': '言語判定',
  'arch.textFull': 'テキスト言語判定（MLP ベースのアプリ）',
  'arch.llm': 'Transformer',
  'arch.llmFull': 'ミニTransformer（文字レベルLLM）',
  'nav.models': 'モデル',
  'nav.apps': 'AIアプリ',
  'nav.menu': 'メニュー',
  'nav.more': 'モデルとアプリを順次追加…',
  'controls.draw': '描画',
  'controls.clear': 'クリア',
  'cnn.hand': '手作りカーネル',
  'cnn.handFull': '古典的な手作りエッジ検出カーネル（Sobel など）。全結合ヘッドのみ学習',
  'cnn.learned': '学習カーネル',
  'cnn.learnedFull': 'カーネルはランダムノイズから畳み込み逆伝播でエンドツーエンドに学習',
  'llm.temp': '温度',
  'llm.tempTip': 'サンプリング温度——低いほど堅実、高いほど大胆',
  'footer.blog': '作者ブログ',
  'footer.source': 'ソースコード',
  'scale.s': '小',
  'scale.m': '中',
  'scale.l': '大',
  'scale.tooltip': 'ネットワークの規模——その場で再構築・再学習します',
  'toast.training': 'ネットワークを再学習中…',
  'controls.typeText': 'テキストを入力…',
  'controls.run': '実行',
  'class.text.0': '中文',
  'class.text.1': 'English',
  'class.text.2': '日本語',
  'textfeat.0': 'ラテン文字率',
  'textfeat.1': '漢字率',
  'textfeat.2': 'かな率',
  'textfeat.3': '数字率',
  'textfeat.4': '句読点率',
  'textfeat.5': '空白率',
  'textfeat.6': '平均語長',
  'textfeat.7': '母音率',
  'layer.tokens': 'トークナイザ',
  'layer.embed': '埋め込み',
  'layer.posenc': '位置エンコーディング',
  'layer.attn': 'マルチヘッドアテンション',
  'layer.attnout': '加重和 A·V',
  'layer.addnorm': 'Add & Norm（残差+LN）',
  'layer.ffn': 'FFN',
  'llm.head': 'ヘッド {n}',
  'panel.mean': '平均 μ',
  'panel.std': '標準偏差 σ',
  'panel.residual': '残差：x + サブ層出力',
  'llm.next': '次の文字',
  'llm.generate': '連続生成',
  'llm.stop': '停止',
  'llm.masked': 'マスク済み（未来トークン）',
  'llm.maskedNote':
    'トークン i は i 以前のトークンにしか注目できません——予測しようとしている未来を覗いてはいけないからです。',
  'llm.score': 'スコア',
  'llm.attnWeight': 'アテンション重み',
  'llm.topCandidates': '上位候補',
  'panel.token': 'トークン',
  'panel.rawText': '入力テキスト',
  'explain.textInput.what':
    '入力されたテキストは 8 つの数値特徴——ラテン文字・漢字・かな・数字・句読点・空白の割合、平均語長、母音率——に変換されます。ネットワークに入るのはこの数値だけです。',
  'explain.textInput.why':
    'ニューラルネットワークは数値しか計算できません。テキストを固定長の特徴ベクトルにするのは最も簡単な「符号化」で、本物の言語モデルは学習した埋め込みで同じことをします。',
  'explain.textInput.simple':
    '電話で文章を説明するようなもの：読み上げるのではなく、いくつかの統計値を伝えます。',
  'explain.tokens.what':
    '入力テキストは 1 文字ずつに分割され、それぞれが 27 記号の語彙から番号を持つトークンになります。モデルは常に最後の 8 トークンを見ます。',
  'explain.tokens.why':
    'ネットワークは生のテキストを読めません——番号付きの離散的なアルファベットが必要です。本物の LLM は文字の代わりにサブワードトークンで同じことをします。',
  'explain.tokens.simple': '文章をタイルに切り分け、1 枚に 1 文字ずつ載せるイメージです。',
  'explain.embed.what':
    '各トークン番号から学習済みの 10 次元ベクトルを引き、さらに位置ベクトルを加えて、そのトークンが列のどこにいるかを伝えます。',
  'explain.embed.why':
    '埋め込みは離散記号を計算可能な座標に変えます。位置ベクトルがなければ「ab」と「ba」は同じに見えてしまいます。',
  'explain.embed.simple':
    '各文字が性格プロファイルと、列の何番目に立っているかのメモを受け取ります。',
  'explain.qkv.what':
    '各トークンの埋め込みに 3 つの学習済み行列を掛け、クエリ（何を探す？）、キー（何を持つ？）、バリュー（何を渡す？）を作ります。',
  'explain.qkv.why':
    '役割を分けることで、各トークンは文脈に質問することも、他のトークンの質問に答えることもできます——アテンションの核心です。',
  'explain.qkv.simple': '各トークンが検索クエリと名刺と手渡す荷物を書き上げます。',
  'explain.posenc.what':
    '各位置 i の埋め込みに固定の正弦/余弦パターンを加えます：偶数次元は sin、奇数次元は cos、周波数は幾何級数的に配置。その和 X = E + P が以降のブロックの入力になります。',
  'explain.posenc.why':
    'アテンション自体には順序の概念がなく、トークンの袋しか見えません。位置エンコーディングは「自分が何番目か」を各ベクトルに刻み、正弦波は相対距離の推論も可能にします。',
  'explain.posenc.simple':
    '各文字の上着に座席番号を縫い付けるようなもの。しかもモデルが読める波模様で。',
  'explain.addnorm.what':
    '2 段階です：サブ層の出力を入力に足し戻し（残差接続）、次に各位置のベクトルを平均 0・分散 1 に正規化して学習済みの γ と β で再スケールします（LayerNorm）。',
  'explain.addnorm.why':
    '残差は勾配のハイウェイとなり深い積み重ねを学習可能にし、各ブロックは「修正分」だけ学べばよくなります。LayerNorm は数値を健全な範囲に保ち学習を安定させます。',
  'explain.addnorm.simple':
    '原稿を残したまま修正を上に重ね、最後に全体の音量を標準に整えるイメージです。',
  'explain.attn.what':
    '各クエリとそれ以前のキーの内積（1/√dₕ でスケール）を行ごとに softmax で重みへ——これを 2 つのヘッドが独立に行います。あるヘッドのセル (i, j) は、そのヘッドが学んだ基準でトークン i が j にどれだけ注目するかを示します。未来のトークンはマスクされます。',
  'explain.attn.why':
    '次の文字を予測するのにどの過去の文字が重要かを、モデルはここで決めます——このパターンは学習で獲得されたものです。',
  'explain.attn.simple': '各文字が前の文字たちに向けるスポットライト——明るいほど関連が強い。',
  'explain.attnout.what':
    '各ヘッドが自分のアテンション行を重みにバリューを加重平均し、連結した結果を出力射影 W_O に通してヘッドの情報を混ぜ、各トークン 1 本のベクトルに戻します：zᵢ = concatₕ(Σⱼ Aₕ[i][j]·vⱼʰ)·W_O。',
  'explain.attnout.why':
    '位置の間で実際に情報が動くのはここだけ——ブロック内でトークン同士が内容を交換する唯一の場所です。',
  'explain.attnout.simple': '各文字がスポットライトの明るさに応じて、集めた荷物をブレンドします。',
  'explain.ffn.what':
    '小さな 2 層ネットワーク（中間に ReLU）が各位置のベクトルを独立に変換します。',
  'explain.ffn.why':
    'アテンションはトークンを混ぜ、フィードフォワード層は集めた情報を加工して非線形なパターン認識力を加えます。',
  'explain.ffn.simple': '意見を集めた後、各文字がひとりで少し考える時間です。',
  'explain.llmOutput.what':
    '最後の位置のベクトルを語彙の全記号に射影してスコアを出し、softmax が次の文字の確率に変えます。',
  'explain.llmOutput.why':
    '言語モデリングとは「次の記号を予測する」こと——この分布から繰り返しサンプリングして LLM は文章を書きます。',
  'explain.llmOutput.simple': 'モデルがあなたの文を続きから書きます：アルファベットへの投票で 1 文字ずつ。',
  'arch.rnn': 'RNN',
  'arch.rnnFull': 'リカレントNN（Elman）',
  'arch.lstm': 'LSTM',
  'arch.lstmFull': '長・短期記憶ネットワーク',
  'arch.ae': 'オートエンコーダ',
  'arch.aeFull': 'オートエンコーダ——圧縮と再構成',
  'layer.rnnHidden': '隠れ状態',
  'layer.gates': 'ゲート f · i · g · o',
  'layer.cell': 'セル状態',
  'layer.encoder': 'エンコーダ',
  'layer.latent': '潜在コード',
  'layer.decoder': 'デコーダ',
  'layer.recon': '再構成',
  'ae.compare': '元画像 vs 再構成',
  'ae.original': '元の値',
  'ae.recon': '再構成値',
  'explain.rnnHidden.what':
    'このシートは行ごとに計算されます：各時刻の隠れ状態は h_t = tanh(Wx·x_t + Wh·h_{t-1} + b) で更新され、どの時刻でも同じ 2 つの重み行列を使い回します。',
  'explain.rnnHidden.why':
    'h_{t-1} が戻ってくることがネットワークの記憶です：過去の文字の情報が隠れベクトルの中で生き続けます。時間方向の重み共有でモデルも小さく保てます。',
  'explain.rnnHidden.simple':
    '文章を読みながら 1 枚のメモだけを持ち、単語を読むたびに書き換えるイメージ。',
  'explain.gates.what':
    '4 つの小さなネットワークが [x_t, h_{t-1}] を並行して読みます：忘却ゲート f と入力ゲート i（sigmoid、0–1）、候補値 g（tanh）、出力ゲート o（sigmoid）。',
  'explain.gates.why':
    '素朴な RNN はすぐ忘れます——長い区間で勾配が消えるためです。ゲートは何を消し、何を書き、何を見せるかを明示的に決めさせる、LSTM の解決策です。',
  'explain.gates.simple':
    '記憶箱の 4 つのダイヤル：どれだけ消すか、どれだけ書くか、何を書くか、どれだけ見せるか。',
  'explain.cellstate.what':
    '長期記憶：c_t = f⊙c_{t-1} + i⊙g。古い内容は忘却ゲートで薄められ、新しい内容が入力ゲート経由で加わります。',
  'explain.cellstate.why':
    '更新が加算的なので、勾配はセル状態に沿ってほぼ減衰せずに流れます——LSTM が多くのステップを記憶できる「ベルトコンベア」です。',
  'explain.cellstate.simple': '単語を読むたびに一部を消して書き足すノート。',
  'explain.lstmHidden.what':
    '記憶の公開面：h_t = o⊙tanh(c_t)。出力ゲートが、セル状態のどれだけを次の層と次の時刻に見せるかを決めます。',
  'explain.lstmHidden.why':
    '内部記憶（c）と外に出す情報（h）を分けることで、まだ使いたくない情報も貯めておけます。',
  'explain.lstmHidden.simple': '口に出す言葉と、頭の中の全部は別物です。',
  'explain.aeInput.what':
    '8×8 の画像を 64 個の数に平坦化したもの。このアーキテクチャにはラベルが一切なく、入力そのものが学習目標です。',
  'explain.aeInput.why':
    'オートエンコーダはラベルなし（自己教師あり）で学びます：「入力を再現せよ」という課題が、データの構造を自力で発見させます。',
  'explain.aeInput.simple': '試験問題と模範解答が同じ 1 枚の絵。',
  'explain.encoder.what':
    '全結合層が 64 個の入力数値を段階的に圧縮します。残したい情報はすべてここを通り抜けなければなりません。',
  'explain.encoder.why':
    '圧縮は抽象化を強制します：狭いボトルネックを通って再構成するには、本質（どの模様？どこ？）を残しノイズを捨てるしかありません。',
  'explain.encoder.simple': '絵を数語に要約してから手渡すイメージ。',
  'explain.latent.what':
    'ボトルネック：画像全体がこの数個の数（6 次元、tanh で ±1 に制限）になります。',
  'explain.latent.why':
    'これが学習された表現です——似た入力は近いコードに落ちます。画像圧縮も生成モデルも同じ考え方で、生成モデルはここで新しいコードをサンプリングします。',
  'explain.latent.simple': 'この絵の DNA：残す価値のあるすべてを数個の数字で。',
  'explain.decoder.what':
    '全結合層が潜在コードを 64 個の数に展開し、最後に sigmoid で各画素を 0–1 に収めます。',
  'explain.decoder.why':
    'デコーダはコードの意味を証明します：6 個の数で絵を描き直せるなら、エンコーダは本当に構造を捉えています。',
  'explain.decoder.simple': '数語の要約から絵を描き直す作業。',
  'explain.recon.what':
    '再構成された画像。学習は入力との平均二乗誤差を最小化します——並べて見比べてください。',
  'explain.recon.why':
    '再構成誤差が学習信号のすべてで、ラベルは不要です。学習データに似ていない入力はうまく再構成されないため、異常検知にも使われます。',
  'explain.recon.simple':
    'ボトルネックを往復してきた絵——よく似ていますが、画素単位では決して同じになりません。',
}

const dicts: Record<Lang, Dict> = { en, zh, ja }

export function translate(lang: Lang, key: string, params?: Record<string, string | number>): string {
  let s = dicts[lang][key] ?? dicts.en[key] ?? key
  if (params) {
    for (const [k, v] of Object.entries(params)) s = s.replaceAll(`{${k}}`, String(v))
  }
  return s
}

export function detectLang(): Lang {
  const nav = typeof navigator !== 'undefined' ? navigator.language : 'en'
  if (nav.startsWith('zh')) return 'zh'
  if (nav.startsWith('ja')) return 'ja'
  return 'en'
}
