# Research - Semantic Segmentation of Retinal OCT image segmentation

## Acknowledgements

This project builds on the **DeepSet SimCLR** framework for self-supervised learning of set-based representations, as described in the paper [DeepSet SimCLR: Self-supervised deep sets for improved pathology representation learning]([https://arxiv.org/abs/2002.05709](https://arxiv.org/abs/2402.15598)) by [Richard Klein, David Torpey]. The implementation was adapted from the code provided by [DeepSet SimCLR GitHub Repository](https://github.com/[link-to-repository]).

### Modifications

The original DeepSet SimCLR framework utilizes a DeepSet architecture to model set-based inputs. In this project, I have modified the framework by:
- Creating a custom dataset preprocessing pipeline specifically designed for retinal Optical Coherence Tomography (OCT) images.
- Using the standard **SimCLR** architecture instead of the per-scan or DeepSet architecture for modeling the images.
- Applying the SimCLR augmentations but **without using the crop transformation**.

These modifications were made to adapt the framework for retinal OCT image analysis, ensuring it aligns with the specific structure and characteristics of medical imaging data.


For details on the original DeepSet SimCLR pre-training process, refer to the official repository at [[https://github.com/[link-to-repository]](https://github.com/DavidTorpey/deepset-simclr/tree/main)].

