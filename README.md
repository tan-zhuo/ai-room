# AI Room

Interactive 3D environment for exploring **real** neural network computation node-by-node, running entirely in the browser. / 走进一个**正在真实运算**的神经网络的 3D 空间，全部在浏览器本地运行。

**🔗 Live: [airoom.run](https://www.airoom.run/)**

by [tanzhuo](https://tanzhuo.xyz) · Blog: **[tanzhuo.xyz](https://tanzhuo.xyz)** · Source: **[github.com/tan-zhuo/ai-room](https://github.com/tan-zhuo/ai-room)** · License: MIT

> **Not an animation — real computation.** Every network is genuinely trained in your browser at load time (deterministic seeds). Every value you see — activations, attention weights, softmax probabilities — is the true result of the forward pass. Click any node and check the arithmetic yourself; type your own text and watch the numbers change.

## Models · 模型架构

### MLP — Multi-Layer Perceptron · 多层感知机

Classifies 3 Gaussian clusters. Watch data-flow particles run through weighted connections, layer by layer.

![MLP forward pass](docs/mlp.gif)

### CNN — Convolutional Network · 卷积神经网络

Classifies patterns (vertical / horizontal / diagonal / ring). The receptive-field window slides across the input exactly in computation order; pooling windows animate the same way. Three sizes (S/M/L up to 16×16 input, 6 kernels) — switching **retrains the network live**.

Two kernel modes: **hand-crafted** Sobel-style edge detectors (interpretable, only the dense head trains) or **learned** — kernels start as random noise and are trained end-to-end through a hand-written conv backward pass. And a **Draw mode**: paint your own pattern on the input grid and watch the whole network classify it live, stroke by stroke.

![CNN convolution](docs/cnn.gif)

### RNN — Recurrent Neural Network · 循环神经网络（Elman）

Char-level next-character prediction on the same corpus as the transformer. The hidden-state sheet computes row by row — one timestep at a time, h_t = tanh(Wx·x_t + Wh·h_{t-1} + b) — with the recurrence arcs drawn between consecutive rows. Trained with hand-written BPTT.

### LSTM — Long Short-Term Memory · 长短期记忆网络

The classic gated recurrent cell, fully visualized: four gate sheets (forget / input / candidate / output), the additive cell-state conveyor belt c = f⊙c′ + i⊙g, and the exposed hidden state h = o⊙tanh(c). Every gate cell opens to its exact math. Same task as the RNN — compare how the two learn.

### Autoencoder · 自编码器

Self-supervised: the 8×8 input is squeezed through a 6-number latent bottleneck and reconstructed, trained by MSE with no labels. Draw your own pattern and watch it survive (or not) the round trip; inspect any output pixel to compare original vs reconstructed values.

Toggle to **VAE**: the bottleneck becomes a distribution (μ, log σ²) with reparameterized sampling z = μ + ε·σ and a KL loss — resample ε to see the stochastic bottleneck, or hit **Generate** to decode z drawn straight from N(0,1): brand-new images from the latent prior.

### Transformer — Tiny char-level LLM · 迷你 Transformer（字符级）

A structurally faithful character-level transformer block with **hand-written forward AND backward passes** (including LayerNorm and residual gradients), trained in-browser on a small corpus (~2.5s):

**Tokenizer → Embedding → Positional Encoding (sinusoidal) → Multi-Head Attention (2 heads, causal, with output projection W_O) → Residual + LayerNorm → Feed-Forward → Residual + LayerNorm → Output softmax**

Both heads' 8×8 attention matrices light up row by row; residual skip connections are drawn when you inspect an Add & Norm cell — down to μ, σ, γ, β. Type a prompt, watch it predict: `"the ai r"` → `o` (room), `"attentio"` → `n` (58%).

Hit **Generate · 连续生成** for true autoregressive decoding: the sampled character is appended to the context, the window slides, the whole pipeline re-runs — and the output streams onto the screen one character at a time, exactly how real LLMs write. A **temperature slider** (0.2–1.4) controls the sampling distribution live.

![Tiny transformer](docs/llm.gif)

## AI Applications · AI 应用

### Lang ID — Language detector · 语言识别（基于 MLP）

**Type anything.** The text becomes 8 interpretable statistics (Latin %, CJK %, kana %, …) and a trained MLP detects 中文 / English / 日本語 — proof the computation is real: your input, its numbers, its prediction.

![Text language detector](docs/text.gif)

## Inspect any node · 点开任意节点

Every neuron, feature-map pixel, attention cell shows its exact math: inputs × weights tables, Σ → bias → activation, kernels and receptive fields, q·k/√d products and softmax rows. The numbers add up — check them.

![Node inspection](docs/inspect.gif)

## Module explanations · 模块讲解

Click any layer title: what it does / why the network needs it / a plain-words analogy — in 中文, English and 日本語 — while the layer glows in 3D and everything else dims.

![Module explanation](docs/explain.gif)

## Controls · 操作

| Input | Action |
| --- | --- |
| Mouse drag / scroll / right-drag | Orbit / zoom / pan |
| Click node · 点击节点 | Inspect its computation |
| Click layer title · 点击层标题 | Module explanation + highlight |
| ☰ menu | Switch between models and AI apps |
| Text box (Lang ID / Transformer) | Run the network on your own input |
| Draw mode (CNN) | Paint the input, classify live |
| Kernel toggle (CNN) | Hand-crafted ↔ learned (end-to-end) kernels |
| Temp slider (Transformer) | Sampling temperature for generation |
| `Space` | Play / pause |
| `←` `→` | Previous / next step |
| `R` | Reset |
| `1` – `6` | Models: MLP / CNN / RNN / LSTM / Autoencoder / Transformer |
| `7` | AI apps: Lang ID |
| `S / M / L` buttons | Network scale (retrains live) |
| `L` | Cycle language 中文 / EN / 日本語 |
| `F` | Focus selected node / layer |
| `Esc` | Close panel / deselect |

## Simplifications vs production models · 与生产级模型的差异

Honest list of what is deliberately simplified — the math shown is real, the scale is not:

- One transformer block, 2 heads, d=12, char-level tokens (production: dozens of blocks, subword BPE, d in the thousands); post-LN as in the original paper (modern LLMs mostly use pre-LN); no dropout or weight decay (datasets are tiny and synthetic).
- Training is plain SGD, sample-by-sample (production: Adam, batches, schedulers).
- CNN: single conv+pool stage, stride 1, no padding; datasets are procedurally generated patterns rather than photos.
- Lang ID uses hand-crafted statistical features on purpose — it demonstrates the simplest form of text encoding, not modern embeddings.

## Development

```bash
npm install
npm run dev      # dev server
npm run build    # type-check + production build (dist/)
npm run sanity   # trains every net at every scale, verifies accuracy
```

Pure frontend, no backend, no runtime network calls. Stack: Vite + React + TypeScript + React Three Fiber + Drei + Zustand.

```
src/
  nn/           # engines: MLP fwd/backprop · CNN conv/pool · transformer fwd/backprop · tasks & training
  store.ts      # zustand: architecture, playback, selection, language, scale
  i18n/         # zh / en / ja dictionaries (UI + module explanations)
  scene/        # R3F: layouts, instanced nodes/connections/particles, slide & attention anims
  ui/           # HUD: transport, inspector, explanations, text entry, tooltip, help
```

## 中文说明（简要）

- **真实计算**：四个网络都在页面加载时用固定随机种子真实训练（`npm run sanity` 可验证准确率）；面板里的每个中间值都是前向传播的真实结果，可以手动对账。
- **MLP**：三类高斯簇分类，逐层粒子流动画对应真实计算顺序。
- **CNN**：Sobel 边缘卷积核 + 训练的全连接头识别图案；感受野滑窗动画与计算顺序一致；S/M/L 三档规模，切换时现场重新训练。
- **RNN / LSTM**：字符级下一字符预测，手写 BPTT 训练；RNN 隐状态逐时间步计算并画出循环连线，LSTM 完整展示 f/i/g/o 四门、细胞状态传送带与 h = o⊙tanh(c)。
- **自编码器**：8×8 图案压入 6 维潜向量再重建（MSE 自监督，无标签）；可手绘输入看重建效果，逐像素对比原值与重建值。
- **语言识别（AI 应用）**：输入任意文字 → 8 个可解释统计特征 → 训练好的 MLP 判断语言。导航中模型（MLP/CNN/RNN/LSTM/自编码器/Transformer）与 AI 应用分为两组。
- **LLM**：结构完整的字符级 Transformer 块——分词器 → 嵌入 → 正弦位置编码 → 多头因果注意力（2 头）→ 残差 + LayerNorm → 前馈 → 残差 + LayerNorm → 输出 softmax。前向与反向传播（含 LayerNorm/残差梯度）均为手写实现，浏览器内训练；两个头的注意力矩阵逐行点亮，点开 Add & Norm 格子可看到 μ、σ、γ、β 的完整算式，残差跳线直接画在 3D 里。点击「连续生成」进入自回归解码：采样的字符沿反馈回路飞回输入端、窗口滑动、流水线重跑——文字一个字一个字流出来。
- **三语界面**：所有 UI 与模块讲解均有中文 / English / 日本語。
