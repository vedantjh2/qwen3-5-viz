
# Qwen3.5 Forward Pass Visualization

This fork adds an interactive `/qwen` tutorial for one dense Qwen3.5 four-layer quartet:
three Gated DeltaNet layers followed by one gated full-attention layer. It uses a small deterministic
teaching model so every computed matrix cell can be traced through its exact dot-product terms.

Run the dependency-free view:

```bash
python3 -m http.server 3002
```

Then open `http://localhost:3002/qwen/`.

The original GPT visualization remains available at `/llm` when running the full Next.js app. See
[`AGENTS.md`](AGENTS.md) for the Qwen3.5 equations, real-versus-mini dimensions, color semantics, and
file map.

## Upstream foundation

This project builds on Brendan Bycroft's
[`bbycroft/llm-viz`](https://github.com/bbycroft/llm-viz) renderer and tutorial structure. The
upstream personal-homepage content and portrait are not included here. The original MIT license is
preserved in [`LICENSE`](LICENSE); bundled third-party fonts and RISC-V assets retain their original
rights and licenses.

The copied upstream routes remain useful as implementation references:

- `/llm` — original nano-GPT visualization
- `/cpu` — CPU schematic project
- `/codec` — image-codec project
- `/fluid-sim` — fluid simulation

## Full Next.js development

The copied Next.js version requires Node.js 20.9 or newer:

```bash
yarn install
yarn dev
```

Then open `http://localhost:3002/qwen`.
