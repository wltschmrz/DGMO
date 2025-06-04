<div style="display: flex; justify-content: center; align-items: center; gap: 100px;">
  <img src="assets/ku-logo.png" alt="Korea University" height="50" style="margin-right: 12px;">
  <img src="assets/miil-logo.png" alt="MIIL" height="50">
  <img src="assets/interspeech2025-logo.png" alt="Interspeech 2025" height="60">
</div>


# DGMO: Training-Free Audio Source Separation through Diffusion-Guided Mask Optimization (Interspeech 2025)

[[Paper]](https://arxiv.org/abs/2506.02858) [[Project Page]](https://wltschmrz.github.io/DGMO/)

by [Geonyoung Lee](https://wltschmrz.github.io/)\*, [Geonhee Han](https://chaksseu.github.io/)\*, [Paul Hongsuck Seo](https://phseo.github.io/)  
\*Equal contribution

This is the official repository for our Interspeech 2025 paper:  
**DGMO: Training-Free Audio Source Separation through Diffusion-Guided Mask Optimization**.

We propose a novel *training-free* framework that enables **zero-shot language-queried audio source separation** by repurposing pretrained text-to-audio diffusion models. DGMO refines magnitude spectrogram masks at test-time via guidance from diffusion-generated references.

---

## Overview

![DGMO Diagram](assets/figures/dgmo_0601.png)

DGMO consists of two key modules:

- **Reference Generation**: Uses DDIM inversion to generate query-conditioned audio references with pretrained diffusion models.
- **Mask Optimization**: Learns a spectrogram mask aligned to the reference, enabling faithful extraction of the target sound from the input mixture.

Unlike traditional LASS approaches, DGMO requires no training and generalizes across datasets with only test-time optimization.

---

## Installation

<!-- To be updated -->

## Inference

## Acknowledgement

Our implementation builds on several open-source projects including [AudioLDM](https://github.com/haoheliu/AudioLDM), [Auffusion](https://github.com/happylittlecat2333/Auffusion), and [Peekaboo](https://github.com/RyannDaGreat/Peekaboo). We sincerely thank the authors for their contributions.



---

## Citation

If you find our work useful in your research, please consider citing:

```bibtex
※ This BibTeX entry is a placeholder. Please refer to the official Interspeech 2025 proceedings for the final citation.
@inproceedings{lee2025dgmo,
  title={DGMO: Training-Free Audio Source Separation through Diffusion-Guided Mask Optimization},
  author={Geonyoung Lee and Geonhee Han and Paul Hongsuck Seo},
  booktitle={Proceedings of the Annual Conference of the International Speech Communication Association (INTERSPEECH)},
  year={2025}
}


