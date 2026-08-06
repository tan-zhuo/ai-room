# AI Room

Interactive 3D environment for exploring **real** neural network computation node-by-node, running entirely in the browser. / 走进一个**正在真实运算**的神经网络的 3D 空间，全部在浏览器本地运行。

> **Not an animation — real computation.** Every network is genuinely trained in your browser at load time (deterministic seeds). Every value you see — activations, attention weights, softmax probabilities — is the true result of the forward pass. Click any node and check the arithmetic yourself; type your own text and watch the numbers change.

## The four architectures · 四种架构

### MLP — Multi-Layer Perceptron · 多层感知机

Classifies 3 Gaussian clusters. Watch data-flow particles run through weighted connections, layer by layer.

![MLP forward pass](docs/mlp.gif)

### CNN — Convolutional Network · 卷积神经网络

Hand-crafted Sobel edge kernels + a trained dense head classify patterns (vertical / horizontal / diagonal / ring). The receptive-field window slides across the input exactly in computation order; pooling windows animate the same way. Three sizes (S/M/L up to 16×16 input, 6 kernels) — switching **retrains the network live**.

![CNN convolution](docs/cnn.gif)

### TEXT — Language detector · 文本语言识别

**Type anything.** The text becomes 8 interpretable statistics (Latin %, CJK %, kana %, …) and a trained MLP detects 中文 / English / 日本語 — proof the computation is real: your input, its numbers, its prediction.

![Text language detector](docs/text.gif)

### LLM — Tiny Transformer · 迷你 Transformer（字符级）

A 1-block, 1-head character-level transformer with **hand-written forward AND backward passes**, trained in-browser on a small corpus (~2s). Embeddings → Q/K/V → causal scaled-dot-product attention (the 8×8 matrix lights up row by row) → weighted sum → FFN → next-character prediction. Type a prompt, watch it predict: `"the ai r"` → `o` (room), `"hello wo"` → `r` (world).

![Tiny transformer](docs/llm.gif)

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
| Text box (TEXT / LLM) | Run the network on your own input |
| `Space` | Play / pause |
| `←` `→` | Previous / next step |
| `R` | Reset |
| `1` `2` `3` `4` | MLP / CNN / TEXT / LLM |
| `S / M / L` buttons | Network scale (retrains live) |
| `L` | Cycle language 中文 / EN / 日本語 |
| `F` | Focus selected node / layer |
| `Esc` | Close panel / deselect |

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
- **TEXT**：输入任意文字 → 8 个可解释统计特征 → 训练好的 MLP 判断语言。
- **LLM**：字符级单头 Transformer，前向与反向传播均为手写实现，浏览器内训练；注意力矩阵逐行点亮，可点开任意格子查看 q·k/√d 与 softmax 的完整算式；输入前缀实时预测下一个字符。
- **三语界面**：所有 UI 与模块讲解均有中文 / English / 日本語。
