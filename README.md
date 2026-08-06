# AI Room

Interactive 3D environment for exploring real-time neural network computations node-by-node, running entirely in the browser.

Walk inside a living neural network: orbit around its layers in 3D, click any neuron or feature-map pixel to see the exact arithmetic it just performed (inputs × weights → sum → bias → activation), and watch data flow through the network as it computes — for real, not as a canned animation.

## Highlights

- **Real computation** — both demo networks are genuinely trained in your browser at load time (deterministic seeds, a few tens of milliseconds), so the predictions you watch are correct: the MLP separates 3 Gaussian clusters, the CNN classifies 8×8 patterns (vertical / horizontal / diagonal / ring) using hand-crafted Sobel-style edge-detector kernels plus a trained dense head.
- **Two architectures, switchable live** — MLP (4→6→5→3) and CNN (conv 3×3 ×3 → maxpool 2×2 → flatten → dense 10 → softmax 4). The layer/slot system is designed to be extended (RNN / Transformer attention later).
- **Node inspection** — click anything: dense neurons show the full inputs×weights table; conv pixels show kernel, receptive field, and element-wise products; pool pixels show their window with the max highlighted; outputs show softmax probabilities.
- **Module explanations** — click any layer title to open a guide panel (what it does / why the network needs it / in plain words, in all 3 languages); the explained layer is highlighted in 3D with a pulsing bounding box while the rest of the scene dims.
- **Playback** — continuous, layer-by-layer, and step-by-step, with flow particles along connections and a sliding receptive-field animation for convolution and pooling.
- **Visual encoding** — connection thickness/brightness = |weight|, cyan = positive, orange = negative; node glow = activation strength (bloom post-processing).
- **i18n** — full UI in 中文 / English / 日本語.

## Controls

| Input | Action |
| --- | --- |
| Mouse drag / scroll / right-drag | Orbit / zoom / pan |
| Click node | Inspect its computation |
| Click layer title | Module explanation + 3D highlight |
| `Space` | Play / pause |
| `←` / `→` | Previous / next step |
| `R` | Reset |
| `1` / `2` | MLP / CNN (`3` reserved) |
| `L` | Cycle language |
| `F` | Focus selected node |
| `Esc` | Deselect / close panel |

## Development

```bash
npm install
npm run dev      # dev server
npm run build    # type-check + production build (dist/)
npm run sanity   # engine sanity check: trains both nets, verifies accuracy
```

Pure frontend — no backend, no network calls at runtime. Stack: Vite + React + TypeScript + React Three Fiber + Drei + Zustand.

## Structure

```
src/
  nn/         # engine: MLP forward/backprop, CNN conv/pool/flatten, model building + training
  store.ts    # zustand state: architecture, playback, selection, language
  i18n/       # zh / en / ja dictionaries
  scene/      # R3F: layouts, instanced nodes/connections/particles, slide anims, camera
  ui/         # HUD: top bar, transport, inspector panel, tooltip, legend, help
```
