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
  'nav.language': 'Language',
  'controls.fullscreen': 'Fullscreen',
  'controls.exitFullscreen': 'Exit fullscreen',
  'controls.draw': 'Draw',
  'controls.clear': 'Clear',
  'cnn.hand': 'Hand kernels',
  'cnn.handFull': 'Classic hand-crafted edge detectors (Sobel etc.); only the dense head is trained',
  'cnn.learned': 'Learned kernels',
  'cnn.learnedFull': 'Kernels start as random noise and are learned end-to-end via conv backprop',
  'llm.temp': 'Temp',
  'llm.tempTip': 'Sampling temperature — lower is safer, higher is wilder',
  'llm.loop': 'Autoregression: output → input',
  'llm.denseChip': 'Dense FFN',
  'llm.moeChip': 'MoE ×4',
  'llm.variantTip': 'Feed-forward type — MoE routes each token to 2 of 4 experts (retrains live)',
  'llm.gate': 'gate weight',
  'llm.selected': 'selected (top-2)',
  'llm.notSelected': 'not selected — this expert is skipped for this token',
  'llm.notRouted':
    'This token was not routed to this expert — the computation is genuinely skipped. That sparsity is the point of MoE.',
  'layer.router': 'Router (top-2 of 4)',
  'layer.experts': 'Experts',
  'layer.combine': 'Weighted combine',
  'explain.router.what':
    'A small gating network scores all 4 experts for every token and softmaxes the scores. Only the top-2 experts per token are activated; the rest are skipped entirely.',
  'explain.router.why':
    'Routing grows total capacity without growing per-token compute — each token only pays for 2 of the 4 experts. This is how Mixtral / DeepSeek-style LLMs scale.',
  'explain.router.simple':
    'A receptionist who reads each token and sends it to the two most suitable specialists.',
  'explain.experts.what':
    'Four independent feed-forward networks. A dark row means that token was not routed here — its computation is genuinely skipped (sparsity).',
  'explain.experts.why':
    'Experts specialize: different tokens exercise different sub-networks, so the model stores more knowledge than a single FFN of the same per-token cost.',
  'explain.experts.simple':
    'A clinic with four doctors; each patient only sees the two who fit their symptoms.',
  'explain.combine.what':
    "Each token's chosen experts are blended by their gate weights: yₜ = Σ gₑ·Eₑ(xₜ).",
  'explain.combine.why':
    'Gate-weighted mixing keeps everything differentiable, so the router and the experts train together end-to-end.',
  'explain.combine.simple':
    "Merging the two specialists' opinions, weighted by how much the receptionist trusted each.",
  'llm.embedMap': 'Embedding map',
  'llm.embedMapNote':
    'PCA projection of the 28 learned character embeddings — characters the model treats similarly end up close together (vowels cyan, space/period orange).',
  'footer.blog': 'Author blog',
  'footer.source': 'Source code',
  'overview.title': 'Model overview',
  'overview.secIntro': 'What it is',
  'overview.secProblems': 'Problems it solves',
  'overview.secDomains': 'Where it is used',
  'overview.secIndustries': 'Industries it powers',
  'overview.tip':
    'Watch it run in 3D on the left — click any node for the exact math, or a layer title for a module guide.',
  'overview.mlp.intro':
    'The multi-layer perceptron is the foundational neural network: stacked layers of fully-connected neurons, each doing "weighted sum + nonlinear activation". It embodies a simple idea — connect enough simple units and you can approximate almost any function (the universal approximation theorem). Every larger architecture contains MLPs as building blocks.',
  'overview.mlp.problems':
    'It answers "make a judgement from a set of numeric features": estimate health risk from checkup numbers, price a house from its attributes, predict churn from user stats. Anything you can put in a spreadsheet row, an MLP can learn from.',
  'overview.mlp.domains':
    'Classification & regression on tabular data\nScoring layers in recommender systems\nCredit & risk scoring\nThe FFN blocks inside every transformer\nSimple control and forecasting tasks',
  'overview.mlp.industries':
    'Finance — credit scoring, fraud detection\nInsurance — underwriting & pricing\nRetail — churn and demand prediction\nHealthcare — risk stratification\nInside virtually every deep model in production',
  'overview.cnn.intro':
    'The convolutional network slides small learned filters across an image, reusing the same weights everywhere. Translation invariance — "recognize the pattern wherever it appears" — is built into the architecture itself. CNNs (LeNet → AlexNet → ResNet) ignited the deep-learning revolution in vision.',
  'overview.cnn.problems':
    'It taught machines to see: recognizing objects in photos, finding lesions in X-rays, reading license plates and faces. Before CNNs these tasks needed hand-crafted features and worked poorly.',
  'overview.cnn.domains':
    'Image classification & object detection\nMedical image analysis\nFace recognition & security\nAutonomous-driving perception\nIndustrial quality inspection\nOCR text recognition',
  'overview.cnn.industries':
    'Medical imaging — cancer screening\nAutomotive — self-driving perception\nManufacturing — defect detection\nAgriculture — crop monitoring\nSmartphones — photo search, portrait mode\nSecurity & retail analytics',
  'overview.rnn.intro':
    'The recurrent network reads a sequence one element at a time, carrying a continuously-updated hidden state — its "memory of everything so far". The same weights are reused at every timestep, so it handles sequences of any length. It was the first successful answer to "make machines understand order".',
  'overview.rnn.problems':
    'It handles data where order is meaning: a sentence depends on word order, a stock price on its history, speech on the waveform through time. RNNs let a model read and remember at the same time.',
  'overview.rnn.domains':
    'Early machine translation & speech recognition\nTime-series forecasting\nText generation\nMusic generation\nSensor-stream analysis',
  'overview.rnn.industries':
    'Keyboards — next-word prediction (early era)\nFinance — time-series models\nIndustrial IoT — sensor anomaly detection\nVoice assistants — early acoustic models',
  'overview.lstm.intro':
    'The LSTM adds three learned gates (forget / input / output) and a cell-state "conveyor belt" to the RNN, fixing its fatal flaw: vanishing gradients that erase long-range memory. For nearly two decades before the Transformer, LSTMs ruled sequence modeling.',
  'overview.lstm.problems':
    'It solves long dependencies: the subject at the start of a sentence governs the verb at the end; context minutes ago shapes what a speech recognizer hears now. LSTM made long-term memory reliably trainable for the first time.',
  'overview.lstm.domains':
    'Speech recognition (the early core of Siri & Google Voice)\nMachine translation (pre-Transformer era)\nHandwriting recognition\nStock & demand forecasting\nVideo activity analysis',
  'overview.lstm.industries':
    'Smart speakers & voice assistants\nTranslation services\nQuantitative finance\nHealthcare — ECG / EEG analysis\nLogistics — demand forecasting',
  'overview.ae.intro':
    'The autoencoder forces data through a narrow bottleneck and asks it to rebuild itself — no labels anywhere. The network must discover the essential structure of the data on its own. It is the gateway to representation learning; the variational variant (VAE) turns the bottleneck into a samplable probability distribution, crossing from compression into generation.',
  'overview.ae.problems':
    'It solves "learning without labels" and "too many dimensions": compressing images, reducing data to a visualizable space, and flagging anomalies (whatever reconstructs badly is abnormal).',
  'overview.ae.domains':
    'Data compression & dimensionality reduction\nAnomaly detection\nImage denoising\nFeature learning for recommenders\nVAE: image generation & interpolation',
  'overview.ae.industries':
    'Industry — equipment anomaly detection\nCybersecurity — intrusion detection\nFinance — fraud identification\nPharma — molecule generation (VAE)\nRemote sensing — data compression',
  'overview.llm.intro':
    'The Transformer discards recurrence: self-attention lets every position look directly at every other, and with multi-head attention, residuals and LayerNorm it stacks extremely deep. It is the skeleton of every modern large language model — GPT, Claude, Gemini. The MoE variant (Mixtral, DeepSeek) extends capacity further with sparsely-routed experts.',
  'overview.llm.problems':
    "It cracked AI's central problem — understanding and generating human language: writing, translation, coding, question-answering. Attention's parallelism also solved the engineering bottleneck that kept RNNs from training at scale.",
  'overview.llm.domains':
    'Large language models (ChatGPT / Claude)\nMachine translation\nCode generation\nProtein structure prediction (AlphaFold)\nImage generation (DiT backbones)\nSpeech & music models',
  'overview.llm.industries':
    'Software — coding assistants\nEducation — personalized tutoring\nCustomer service — conversational agents\nLegal & finance — document analysis\nBiotech — drug discovery\nMedia — content creation',
  'overview.text.intro':
    'This is an "AI application" example: a real problem (which language is this text?) decomposed into feature engineering + a small model. It shows a path that quietly powers much of industry — not every problem needs a giant model.',
  'overview.text.problems':
    'It solves automatic language identification: the "Translate this page?" prompt in your browser, routing multilingual support tickets, indexing the web by language.',
  'overview.text.domains':
    'Browsers & input methods\nMultilingual content platforms\nSpam filtering (same features+classifier recipe)\nSearch engines',
  'overview.text.industries': 'Internet platforms\nCross-border e-commerce\nLocalization services',
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
  'ae.plainChip': 'Plain AE',
  'layer.sampleZ': 'Sample z',
  'vae.resample': 'Resample ε',
  'vae.resampleTip': 'Draw a fresh ε through the same input — the bottleneck is stochastic',
  'vae.generate': 'Generate',
  'vae.generateTip': 'Sample z straight from N(0,1) and decode — pure generation, no input',
  'vae.generatedBadge': 'Sampled from the latent prior — a brand-new image',
  'vae.reparamNote':
    'The reparameterization trick: randomness lives only in ε, so gradients can flow through μ and σ during training.',
  'explain.muSigma.what':
    'Instead of one code, the encoder outputs a distribution per latent dimension: a mean μ and a log-variance log σ². The input is now a fuzzy region of latent space, not a single point.',
  'explain.muSigma.why':
    'Making the code a distribution (pulled toward N(0,1) by the KL loss) organizes the latent space so that every region decodes to something sensible — the property that makes generation possible.',
  'explain.muSigma.simple':
    'The summary is no longer an exact address but a neighborhood: "somewhere around here".',
  'explain.sampleZ.what':
    'A concrete code is drawn from the distribution: z = μ + ε·σ with ε ~ N(0,1). The reparameterization trick keeps the randomness in ε so gradients can still flow through μ and σ.',
  'explain.sampleZ.why':
    'Sampling during training forces the decoder to tolerate noise around μ, smoothing the latent space. At generation time, sampling z from N(0,1) yields entirely new images.',
  'explain.sampleZ.simple': 'Roll the dice inside the neighborhood, then hand the resulting spot to the decoder.',
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
  'nav.language': '语言',
  'controls.fullscreen': '全屏',
  'controls.exitFullscreen': '退出全屏',
  'controls.draw': '绘制',
  'controls.clear': '清空',
  'cnn.hand': '手工核',
  'cnn.handFull': '经典手工边缘检测核（Sobel 等），只训练全连接头',
  'cnn.learned': '学习核',
  'cnn.learnedFull': '卷积核从随机噪声开始，通过卷积反向传播端到端学习',
  'llm.temp': '温度',
  'llm.tempTip': '采样温度——越低越保守，越高越放飞',
  'llm.loop': '自回归：输出 → 输入',
  'llm.denseChip': '密集 FFN',
  'llm.moeChip': 'MoE 专家×4',
  'llm.variantTip': '前馈类型——MoE 把每个 token 路由到 4 个专家中的 2 个（实时重训练）',
  'llm.gate': '门控权重',
  'llm.selected': '已选中（top-2）',
  'llm.notSelected': '未选中——该专家对这个 token 被跳过',
  'llm.notRouted': '该 token 未被路由到此专家——计算被真实跳过。这种稀疏性正是 MoE 的意义所在。',
  'layer.router': '路由器（4 选 2）',
  'layer.experts': '专家网络',
  'layer.combine': '加权合并',
  'explain.router.what':
    '一个小型门控网络为每个 token 给全部 4 个专家打分并做 softmax。每个 token 只激活得分前 2 的专家，其余完全跳过。',
  'explain.router.why':
    '路由让模型在不增加单 token 计算量的前提下扩大总容量——每个 token 只为 4 个专家中的 2 个付费。Mixtral / DeepSeek 这类 MoE 大模型正是这样扩展的。',
  'explain.router.simple': '像前台接待员：读一眼每个 token，把它送到最合适的两位专家那里。',
  'explain.experts.what':
    '四个相互独立的前馈网络。某一行是暗的，说明该 token 没有被路由到这里——它的计算被真实跳过（稀疏性）。',
  'explain.experts.why':
    '专家会分工：不同的 token 走不同的子网络，模型在同样的单 token 成本下能存下更多知识。',
  'explain.experts.simple': '一间有四位医生的诊所：每位病人只看和自己症状最匹配的两位。',
  'explain.combine.what': '每个 token 选中的专家按门控权重混合：yₜ = Σ gₑ·Eₑ(xₜ)。',
  'explain.combine.why': '门控加权保持了整体可微，路由器和专家因此能端到端一起训练。',
  'explain.combine.simple': '把两位专家的意见按前台的信任程度加权合成一个结论。',
  'llm.embedMap': '嵌入地图',
  'llm.embedMapNote':
    '28 个学习到的字符嵌入的 PCA 投影——模型认为相似的字符会聚在一起（元音为青色，空格/句号为橙色）。',
  'footer.blog': '作者博客',
  'footer.source': '开源代码',
  'overview.title': '模型简介',
  'overview.secIntro': '这是什么',
  'overview.secProblems': '解决了什么问题',
  'overview.secDomains': '作用领域',
  'overview.secIndustries': '支持的行业',
  'overview.tip': '左侧就是它的 3D 实况——点击任意节点看具体算式，点击层标题看模块讲解。',
  'overview.mlp.intro':
    '多层感知机是最基础的神经网络：若干层全连接神经元堆叠，每层做"加权求和 + 非线性激活"。它体现了一个朴素的思想——把足够多的简单单元连接起来，就能逼近几乎任何函数（万能逼近定理）。所有更大的架构里都有它的身影。',
  'overview.mlp.problems':
    '解决"从一组数值特征做出判断"的问题：根据体检指标估计健康风险、根据房屋参数估价、根据用户属性预测流失。任何能整理成表格一行的数据，MLP 都能学。',
  'overview.mlp.domains':
    '表格数据的分类与回归\n推荐系统的打分层\n信用与风险评分\n每个 Transformer 内部的 FFN 模块\n简单的控制与预测任务',
  'overview.mlp.industries':
    '金融——信用评分、反欺诈\n保险——核保定价\n零售——流失与需求预测\n医疗——风险分层\n几乎所有生产环境的深度模型内部',
  'overview.cnn.intro':
    '卷积网络让小滤波器在图像上滑动扫描，同一组权重全图复用。"图案出现在哪里都认得"（平移不变性）被直接刻进了结构里。CNN（LeNet → AlexNet → ResNet）点燃了深度学习在视觉领域的革命。',
  'overview.cnn.problems':
    '它教会了机器"看"：认出照片里的物体、找出 X 光片上的病灶、识别车牌和人脸。在 CNN 之前，这些任务依赖人工设计特征，效果差且脆弱。',
  'overview.cnn.domains':
    '图像分类与目标检测\n医学影像分析\n人脸识别与安防\n自动驾驶感知\n工业质检\nOCR 文字识别',
  'overview.cnn.industries':
    '医疗影像——癌症筛查\n汽车——自动驾驶感知\n制造业——缺陷检测\n农业——作物监测\n手机——相册搜索、人像模式\n安防与零售分析',
  'overview.rnn.intro':
    '循环网络逐个读取序列元素，用一个不断更新的隐状态携带"到目前为止的记忆"。同一组权重在每个时间步复用，因此能处理任意长度的序列。它是"让机器理解顺序"的第一个成功答案。',
  'overview.rnn.problems':
    '解决"顺序即含义"的数据：一句话取决于词序，股价取决于历史，语音是时间上的波形。RNN 让模型第一次能一边读、一边记。',
  'overview.rnn.domains':
    '早期机器翻译与语音识别\n时间序列预测\n文本生成\n音乐生成\n传感器流分析',
  'overview.rnn.industries':
    '输入法——下一词联想（早期）\n金融——时序模型\n工业物联网——传感器异常检测\n语音助手——早期声学模型',
  'overview.lstm.intro':
    'LSTM 给循环网络装上三扇可学习的门（遗忘/输入/输出）和一条细胞状态"传送带"，修复了普通 RNN 的致命缺陷——梯度消失导致记不住长距离信息。在 Transformer 出现之前的近二十年里，LSTM 统治着整个序列建模领域。',
  'overview.lstm.problems':
    '解决"长依赖"问题：句首的主语决定句尾的动词形式，几分钟前的上下文影响此刻的语音识别。LSTM 让"长期记忆"第一次变得可靠可训练。',
  'overview.lstm.domains':
    '语音识别（Siri、Google 语音的早期核心）\n机器翻译（前 Transformer 时代）\n手写识别\n股价与销量预测\n视频行为分析',
  'overview.lstm.industries':
    '智能音箱与语音助手\n翻译服务\n量化金融\n医疗——心电/脑电时序分析\n物流——需求预测',
  'overview.ae.intro':
    '自编码器强迫数据穿过狭窄的瓶颈再重建自己——全程没有任何标签，网络必须自己发现数据的本质结构。它是表示学习的入口；变分自编码器（VAE）进一步把瓶颈变成可采样的概率分布，从"压缩"跨入"生成"。',
  'overview.ae.problems':
    '解决"没有标签也要学习"和"维度太高"的问题：压缩图像、把数据降维到可视化空间、发现异常样本（重建得差的就是不正常的）。',
  'overview.ae.domains':
    '数据压缩与降维\n异常检测\n图像去噪\n推荐系统的特征学习\nVAE：图像生成与插值',
  'overview.ae.industries':
    '工业——设备异常检测\n网络安全——入侵检测\n金融——欺诈识别\n医药——分子生成（VAE）\n遥感——数据压缩',
  'overview.llm.intro':
    'Transformer 抛弃了循环：自注意力让序列中每个位置直接"看到"所有其他位置，配合多头注意力、残差和 LayerNorm 可以堆得极深。它是 GPT、Claude、Gemini 等所有现代大语言模型的骨架；MoE 变体（Mixtral、DeepSeek）进一步用稀疏路由的专家扩展容量。',
  'overview.llm.problems':
    '攻克了 AI 的核心难题——理解和生成人类语言：写作、翻译、编程、问答。注意力的可并行性同时解决了 RNN 无法大规模训练的工程瓶颈。',
  'overview.llm.domains':
    '大语言模型（ChatGPT / Claude）\n机器翻译\n代码生成\n蛋白质结构预测（AlphaFold）\n图像生成（DiT 骨干）\n语音与音乐模型',
  'overview.llm.industries':
    '软件——编程助手\n教育——个性化辅导\n客服——对话机器人\n法律与金融——文书分析\n生物医药——药物发现\n媒体——内容创作',
  'overview.text.intro':
    '这是一个"AI 应用"示例：把真实问题（这段文字是什么语言？）拆解成"特征工程 + 小模型"。它展示了在工业界默默支撑大量场景的路径——不是所有问题都需要大模型。',
  'overview.text.problems':
    '解决文本语言自动识别：浏览器的"翻译此页？"提示、多语言客服工单的自动分流、搜索引擎按语言建索引。',
  'overview.text.domains':
    '浏览器与输入法\n多语言内容平台\n垃圾邮件过滤（同一"特征+分类器"配方）\n搜索引擎',
  'overview.text.industries': '互联网平台\n跨境电商\n本地化服务',
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
  'ae.plainChip': '标准 AE',
  'layer.sampleZ': '采样 z',
  'vae.resample': '重采样 ε',
  'vae.resampleTip': '同一输入换一个 ε——瓶颈是随机的',
  'vae.generate': '随机生成',
  'vae.generateTip': '直接从 N(0,1) 采样 z 并解码——不需要任何输入的纯生成',
  'vae.generatedBadge': '从潜空间先验采样——一张全新的图',
  'vae.reparamNote': '重参数化技巧：随机性只存在于 ε 中，因此训练时梯度能够穿过 μ 和 σ。',
  'explain.muSigma.what':
    '编码器不再输出一个确定的编码，而是每个潜维度输出一个分布：均值 μ 和对数方差 log σ²。输入对应的不再是潜空间中的一个点，而是一片模糊的区域。',
  'explain.muSigma.why':
    '把编码变成分布（并用 KL 损失把它拉向 N(0,1)）会把潜空间整理得处处有意义——任何区域解码出来都是合理的图像，这正是"能生成"的前提。',
  'explain.muSigma.simple': '摘要不再是精确地址，而是一个街区："大概在这一带"。',
  'explain.sampleZ.what':
    '从分布中抽出一个具体编码：z = μ + ε·σ，其中 ε ~ N(0,1)。重参数化技巧把随机性留在 ε 里，梯度因此仍能穿过 μ 和 σ。',
  'explain.sampleZ.why':
    '训练时的采样迫使解码器容忍 μ 附近的噪声，从而平滑整个潜空间；生成时直接从 N(0,1) 采样 z，就能得到全新的图像。',
  'explain.sampleZ.simple': '在街区里掷一次骰子选定落点，再把这个落点交给解码器。',
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
  'nav.language': '言語',
  'controls.fullscreen': '全画面',
  'controls.exitFullscreen': '全画面終了',
  'controls.draw': '描画',
  'controls.clear': 'クリア',
  'cnn.hand': '手作りカーネル',
  'cnn.handFull': '古典的な手作りエッジ検出カーネル（Sobel など）。全結合ヘッドのみ学習',
  'cnn.learned': '学習カーネル',
  'cnn.learnedFull': 'カーネルはランダムノイズから畳み込み逆伝播でエンドツーエンドに学習',
  'llm.temp': '温度',
  'llm.tempTip': 'サンプリング温度——低いほど堅実、高いほど大胆',
  'llm.loop': '自己回帰：出力 → 入力',
  'llm.denseChip': '密なFFN',
  'llm.moeChip': 'MoE ×4',
  'llm.variantTip': 'FFNの種類——MoE は各トークンを 4 人中 2 人の専門家へルーティング（その場で再学習）',
  'llm.gate': 'ゲート重み',
  'llm.selected': '選択済み（top-2）',
  'llm.notSelected': '未選択——このトークンではこの専門家はスキップされます',
  'llm.notRouted':
    'このトークンはこの専門家にルーティングされず、計算は実際にスキップされています。この疎性こそ MoE の要点です。',
  'layer.router': 'ルーター（4 中 2）',
  'layer.experts': 'エキスパート',
  'layer.combine': '加重合成',
  'explain.router.what':
    '小さなゲートネットワークが各トークンについて 4 人の専門家を採点し softmax します。上位 2 人だけが起動され、残りは完全にスキップされます。',
  'explain.router.why':
    'ルーティングにより、トークンあたりの計算量を増やさずに総容量を拡大できます——各トークンは 4 人中 2 人分しか支払いません。Mixtral / DeepSeek 系の MoE LLM はこの方法でスケールしています。',
  'explain.router.simple':
    '受付係のようなもの：各トークンを一読し、最も適した 2 人の専門家へ案内します。',
  'explain.experts.what':
    '4 つの独立したフィードフォワードネットワーク。暗い行はそのトークンがここにルーティングされなかった印——計算は本当にスキップされています（疎性）。',
  'explain.experts.why':
    '専門家は分業します：トークンごとに別のサブネットワークが働くため、同じトークン単価でより多くの知識を蓄えられます。',
  'explain.experts.simple':
    '4 人の医師がいる診療所：患者は症状に合う 2 人だけを受診します。',
  'explain.combine.what':
    '各トークンの選ばれた専門家をゲート重みで混合します：yₜ = Σ gₑ·Eₑ(xₜ)。',
  'explain.combine.why':
    'ゲート加重の混合により全体が微分可能に保たれ、ルーターと専門家がエンドツーエンドで一緒に学習できます。',
  'explain.combine.simple':
    '受付係の信頼度に応じて、2 人の専門家の意見を重み付けして 1 つにまとめる作業。',
  'llm.embedMap': '埋め込みマップ',
  'llm.embedMapNote':
    '学習済み 28 文字埋め込みの PCA 射影——モデルが似ていると判断した文字は近くに集まります（母音は水色、空白/句点は橙色）。',
  'footer.blog': '作者ブログ',
  'footer.source': 'ソースコード',
  'overview.title': 'モデル概要',
  'overview.secIntro': 'これは何か',
  'overview.secProblems': '解決する課題',
  'overview.secDomains': '主な用途',
  'overview.secIndustries': '支える産業',
  'overview.tip':
    '左に見えているのが 3D の実働です——ノードをクリックで実際の数式、層タイトルでモジュール解説。',
  'overview.mlp.intro':
    '多層パーセプトロンは最も基礎的なニューラルネットワークです：全結合ニューロンの層を積み重ね、各層で「加重和＋非線形活性化」を行います。単純なユニットを十分つなげばほぼ任意の関数を近似できる（万能近似定理）という素朴な思想の体現で、より大きなアーキテクチャの内部にも必ず存在します。',
  'overview.mlp.problems':
    '「数値特徴から判断を下す」問題を解きます：健診数値から健康リスクを推定、物件属性から価格を予測、ユーザー属性から解約を予測。表の 1 行にできるデータなら MLP は学習できます。',
  'overview.mlp.domains':
    '表形式データの分類・回帰\n推薦システムのスコアリング層\n信用・リスクスコアリング\nすべての Transformer 内部の FFN\n単純な制御・予測タスク',
  'overview.mlp.industries':
    '金融——信用スコア、不正検知\n保険——引受・価格設定\n小売——解約・需要予測\n医療——リスク層別化\n実運用のほぼ全ての深層モデルの内部',
  'overview.cnn.intro':
    '畳み込みネットワークは小さな学習フィルタを画像上でスライドさせ、同じ重みを全面で使い回します。「どこに現れても同じパターンを認識する」（平行移動不変性）が構造そのものに刻まれています。CNN（LeNet → AlexNet → ResNet）は視覚分野の深層学習革命の起点でした。',
  'overview.cnn.problems':
    '機械に「見る」ことを教えました：写真の物体認識、X 線画像の病変検出、ナンバープレートや顔の識別。CNN 以前は手作り特徴が必要で、性能は脆弱でした。',
  'overview.cnn.domains':
    '画像分類・物体検出\n医用画像解析\n顔認識・セキュリティ\n自動運転の知覚\n工業検査\nOCR 文字認識',
  'overview.cnn.industries':
    '医用画像——がんスクリーニング\n自動車——自動運転\n製造——欠陥検出\n農業——作物モニタリング\nスマホ——写真検索、ポートレート\n防犯・小売分析',
  'overview.rnn.intro':
    'リカレントネットワークは系列を 1 要素ずつ読み、更新され続ける隠れ状態に「ここまでの記憶」を載せて運びます。全時刻で同じ重みを使い回すため任意長の系列を扱えます。「機械に順序を理解させる」最初の成功でした。',
  'overview.rnn.problems':
    '順序が意味を持つデータを扱います：文は語順に、株価は履歴に、音声は時間波形に依存します。RNN は「読みながら覚える」ことを初めて可能にしました。',
  'overview.rnn.domains':
    '初期の機械翻訳・音声認識\n時系列予測\nテキスト生成\n音楽生成\nセンサーストリーム解析',
  'overview.rnn.industries':
    'キーボード——次語予測（初期）\n金融——時系列モデル\n産業 IoT——センサー異常検知\n音声アシスタント——初期の音響モデル',
  'overview.lstm.intro':
    'LSTM は RNN に 3 つの学習ゲート（忘却/入力/出力）とセル状態の「ベルトコンベア」を加え、長距離の記憶を消してしまう勾配消失という致命的欠陥を修復しました。Transformer 登場までの約 20 年、系列モデリングを支配しました。',
  'overview.lstm.problems':
    '長距離依存を解決します：文頭の主語が文末の動詞を決め、数分前の文脈が今の音声認識を左右します。LSTM は「長期記憶」を初めて安定して学習可能にしました。',
  'overview.lstm.domains':
    '音声認識（初期 Siri・Google 音声の中核）\n機械翻訳（Transformer 以前）\n手書き認識\n株価・需要予測\n映像行動解析',
  'overview.lstm.industries':
    'スマートスピーカー・音声アシスタント\n翻訳サービス\nクオンツ金融\n医療——心電/脳波解析\n物流——需要予測',
  'overview.ae.intro':
    'オートエンコーダはデータを狭いボトルネックに通して自分自身を再構成させます——ラベルは一切なく、ネットワークは自力でデータの本質構造を発見するしかありません。表現学習の入口であり、変分版（VAE）はボトルネックをサンプリング可能な確率分布に変え、「圧縮」から「生成」へ踏み出します。',
  'overview.ae.problems':
    '「ラベルなしで学ぶ」「次元が多すぎる」という課題を解きます：画像圧縮、可視化のための次元削減、再構成の悪いサンプル＝異常の検出。',
  'overview.ae.domains':
    'データ圧縮・次元削減\n異常検知\n画像ノイズ除去\n推薦の特徴学習\nVAE：画像生成・補間',
  'overview.ae.industries':
    '産業——設備異常検知\nサイバーセキュリティ——侵入検知\n金融——不正検出\n製薬——分子生成（VAE）\nリモートセンシング——データ圧縮',
  'overview.llm.intro':
    'Transformer は再帰を捨てました：自己注意により各位置が他の全位置を直接「見る」ことができ、マルチヘッド・残差・LayerNorm と組み合わせて極めて深く積めます。GPT、Claude、Gemini などすべての現代 LLM の骨格です。MoE 変種（Mixtral、DeepSeek）は疎にルーティングされた専門家で容量をさらに拡張します。',
  'overview.llm.problems':
    'AI の中心課題——人間の言語の理解と生成（執筆・翻訳・コーディング・質問応答）を解きました。注意機構の並列性は、RNN が大規模学習できなかった工学的ボトルネックも同時に解決しました。',
  'overview.llm.domains':
    '大規模言語モデル（ChatGPT / Claude）\n機械翻訳\nコード生成\nタンパク質構造予測（AlphaFold）\n画像生成（DiT）\n音声・音楽モデル',
  'overview.llm.industries':
    'ソフトウェア——コーディング支援\n教育——個別指導\nカスタマーサポート——対話エージェント\n法務・金融——文書解析\nバイオ——創薬\nメディア——コンテンツ制作',
  'overview.text.intro':
    'これは「AI アプリ」の例です：現実の問題（このテキストは何語？）を「特徴量エンジニアリング＋小さなモデル」に分解しています。巨大モデルを使わずとも産業の多くを静かに支えている道筋を示します。',
  'overview.text.problems':
    '言語の自動判定を解決します：ブラウザの「このページを翻訳？」表示、多言語サポートの自動振り分け、言語別のウェブ索引。',
  'overview.text.domains':
    'ブラウザ・入力メソッド\n多言語コンテンツ基盤\nスパムフィルタ（同じ特徴量＋分類器の定石）\n検索エンジン',
  'overview.text.industries': 'インターネットプラットフォーム\n越境 EC\nローカライズサービス',
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
  'ae.plainChip': '標準AE',
  'layer.sampleZ': 'z をサンプル',
  'vae.resample': 'ε を再サンプル',
  'vae.resampleTip': '同じ入力で別の ε を引く——ボトルネックは確率的です',
  'vae.generate': '生成',
  'vae.generateTip': 'N(0,1) から z を直接サンプリングして復号——入力なしの純粋な生成',
  'vae.generatedBadge': '潜在事前分布からサンプリング——全く新しい画像',
  'vae.reparamNote':
    '再パラメータ化トリック：ランダム性は ε だけに存在するため、学習時に勾配が μ と σ を通過できます。',
  'explain.muSigma.what':
    'エンコーダは 1 つのコードではなく、潜在次元ごとに分布を出力します：平均 μ と対数分散 log σ²。入力は潜在空間の一点ではなく、ぼんやりした領域に対応します。',
  'explain.muSigma.why':
    'コードを分布にし（KL 損失で N(0,1) に引き寄せ）ることで、潜在空間のどこを復号しても意味のある画像になるよう整えられます——これが「生成できる」ための条件です。',
  'explain.muSigma.simple': '要約は正確な住所ではなく「だいたいこの辺り」という街区になります。',
  'explain.sampleZ.what':
    '分布から具体的なコードを 1 つ引きます：z = μ + ε·σ、ε ~ N(0,1)。再パラメータ化によりランダム性は ε に留まり、勾配は μ と σ を通過できます。',
  'explain.sampleZ.why':
    '学習中のサンプリングはデコーダに μ 周辺のノイズへの耐性を強い、潜在空間を滑らかにします。生成時は N(0,1) から z を引くだけで新しい画像が得られます。',
  'explain.sampleZ.simple': '街区の中でサイコロを振って落下点を決め、それをデコーダに渡します。',
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
