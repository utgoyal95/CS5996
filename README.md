# CS5996: AI Result Replication: DCGAN Unsupervised Representation Learning

## Project Goal
Reproduce and explore the key findings from “Unsupervised Representation Learning with Deep Convolutional GANs” (DCGAN) by Radford et al. (2016). Specifically, we will:

1. **Re-implement a DCGAN** that learns to generate realistic 64×64 RGB images from noise—using only unlabeled data.  
2. **Verify interpretability** of learned features by visualizing:  
   - Convolutional filters that detect object parts  
   - Smooth semantic interpolations in the generator’s latent space  

---

## Key Concepts

- **Generative Adversarial Network (GAN)**  
  A two-player game:  
  - **Generator (G):** Learns to map random noise → realistic images.  
  - **Discriminator (D):** Learns to tell real images apart from generated ones.  
  They train together: G tries to “fool” D, while D gets better at spotting fakes.

- **Deep Convolutional GAN (DCGAN)**  
  A specific GAN architecture that uses:  
  - **All‐convolutional networks** (no pooling or fully-connected layers)  
  - **Strided convolutions** (to learn downsampling) and **transposed convolutions** (to learn upsampling)  
  - **Batch normalization** throughout (except generator’s output and discriminator’s input)  
  - **LeakyReLU/ReLU activations** and careful kernel sizing  

- **Latent Space & Interpolation**  
  We sample a random vector (noise) from a simple distribution (e.g., uniform). The generator transforms it into an image. By moving slowly through this latent space, we see smooth changes in the generated images (e.g., turning a frown into a smile).

- **Feature Visualization**  
  Inspecting early convolutional filters in the discriminator reveals that each filter learns to detect meaningful patterns (windows, wheels, textures), even though it was never given labels.

---