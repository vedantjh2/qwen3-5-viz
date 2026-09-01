
# Qwen3.5 Forward Pass Visualization

This fork adds an interactive `/qwen` tutorial for one dense Qwen3.5 four-layer quartet:
three Gated DeltaNet layers followed by one gated full-attention layer. It uses a small deterministic
teaching model so every computed matrix cell can be traced through its exact dot-product terms.

Run the dependency-free view:

```bash
python3 -m http.server 3002
```

Then open `http://localhost:3002/qwen/`.

The hosted visualization is available at:

**https://vedantjh2.github.io/qwen3-5-viz/**

GitHub Pages publishes the static `qwen/` directory directly. It does not install dependencies or
run Next.js.

## Reusable Copilot skill

This repository includes the project skill
[`model-forward-pass-visualizer`](.github/skills/model-forward-pass-visualizer/SKILL.md). It guides
Copilot through researching another model, preserving real architecture ratios, building a
deterministic mini forward pass, generating the same cell-traceable UI and long-form tutorial, and
validating or publishing the result.

From a Copilot CLI session in this repository, invoke it explicitly with:

```text
Use the /model-forward-pass-visualizer skill to create the same style of visualization for <model>.
```

To reuse it in another repository, copy the skill directory or add it with `copilot skill add`.

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

Next.js is optional for the Qwen visualization. It provides the multi-route application shell and
keeps the copied `/llm`, `/cpu`, `/codec`, and `/fluid-sim` projects runnable. The standalone Qwen
app in `qwen/` contains its own HTML, CSS, JavaScript, and deterministic model data.
