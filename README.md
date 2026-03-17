# Out of Sight but Not Out of Mind: Hybrid Memory for Dynamic Video World Models (Hybrid Memory)

Official repository for **Hybrid Memory** and **HyDRA**, and the **HM-World** dataset.

## Links

- Project page: `https://<YOUR_GITHUB_USERNAME>.github.io/<REPO_NAME>/` (GitHub Pages)
- Paper (arXiv): `https://arxiv.org/abs/XXXX.XXXXX`
- Dataset (Hugging Face): `https://huggingface.co/datasets/<ORG>/<HM-WORLD>`
- Models (Hugging Face): `https://huggingface.co/<ORG>/<MODEL_NAME>`

## Overview

When dynamic subjects leave the camera view and later re-enter, existing memory mechanisms often fail (frozen, distorted, or missing subjects). **Hybrid Memory** treats the scene as a combination of:

- **Static background**: preserved as a stable “canvas”
- **Dynamic subjects**: tracked to maintain identity and motion continuity through out-of-view intervals

We introduce **HM-World**, a large-scale dataset with exit-entry events, and propose **HyDRA** (Hybrid Dynamic Retrieval Attention) to retrieve motion- and identity-relevant memory cues for consistent re-entry.

## Repository Structure

- `index.html`: project page entry (for GitHub Pages)
- `assets/`: project page assets (css/js/images/videos)
- `CITATION.cff`: citation metadata (GitHub “Cite this repository”)
- `CITATION.bib`: BibTeX entry

## Code & Data

This repository currently hosts the **project page scaffold** and citation files. Training/evaluation code and dataset access instructions can be added here as they are finalized.

If you want, I can scaffold the following folders/files next (with placeholders filled after you provide links):

- `src/` (training/eval code)
- `configs/` (experiment configs)
- `scripts/` (download/preprocess/run scripts)
- `requirements.txt` or `environment.yml`

## How to Host the Project Page (GitHub Pages)

1. On GitHub: `Settings → Pages`
2. `Source`: **Deploy from a branch**
3. Select branch `main` and folder `/ (root)`
4. Wait for deployment, then open: `https://<YOUR_GITHUB_USERNAME>.github.io/<REPO_NAME>/`

## Citation

See `CITATION.bib` and the BibTeX section on the project page.
