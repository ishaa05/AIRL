# AIRL 

## How to Run in Colab  
- Open the notebook in **Google Colab**.  
- Enable GPU runtime (`Runtime` > `Change runtime type` > `GPU`).  
- Run all cells sequentially from top to bottom.  

---

## Q1 Vision Transformer on CIFAR-10  
**Objective:**  
Implement and train a Vision Transformer (ViT) from scratch on CIFAR-10 to achieve the highest possible test accuracy.

**Implementation:**  
- Patchify CIFAR-10 images into non-overlapping tokens (4×4 patches).  
- Add learnable positional embeddings and a `[CLS]` token.  
- Stack Transformer encoder blocks (Multi-Head Self-Attention + MLP with residual connections + LayerNorm).  
- Classification from the `[CLS]` token.  
- Training pipeline:  
  - Optimizer: **AdamW**  
  - Scheduler: **CosineAnnealingLR** with warmup  
  - Regularization: Dropout, DropPath (stochastic depth), Label smoothing  
  - Strong data augmentation: RandomCrop + Flip, RandAugment, MixUp  

**Experiments Conducted:**  
- Patch size variations (2×2, 4×4, 8×8).  
- Depth/width trade-offs (6 vs. 12 layers, embedding dims 128 vs. 192).  
- Augmentation effects (baseline vs. RandAugment, MixUp, CutMix).  
- Optimizer/scheduler variants (Adam vs. AdamW, fixed LR vs. Cosine LR).  
- Overlapping vs. non-overlapping patches.  

**Results:**  
| Model Variant | Test Accuracy (%) |
|---------------|-------------------|
| Baseline (patch=4, depth=6, emb=128, simple aug) | 76.67 |
| Final (patch=4, depth=12, emb=192, RandAugment + MixUp + Label smoothing + DropPath) | **90.95** |

**Concise Analysis:**  
- **Patch Size:** 4×4 gave the best trade-off (64 tokens, good granularity without heavy compute). Larger patches (8×8) underfit; smaller patches (2×2) increased compute.  
- **Depth/Width:** A deeper (12-layer) but narrower (192-dim) ViT outperformed shallower or very wide models. Strong regularization was essential to prevent overfitting.  
- **Augmentation:** RandAugment + MixUp significantly boosted accuracy versus basic crop/flip. CutMix gave similar benefits but required tuning.  
- **Optimizer/Scheduler:** AdamW + weight decay + cosine decay with warmup converged smoothly and gave stable accuracy.  
- **Overlapping vs. Non-Overlapping:** Standard non-overlapping patches were used; overlapping patches could slightly improve locality but added extra cost.  

**Best Result:**  
**90.95% test accuracy** on CIFAR-10 using the final configuration.  

---

## Q2 Text-Driven Image Segmentation with SAM2  
- **Goal:** Perform segmentation of an object in an image given a text description.  

### Pipeline  
1. Load an input image.  
2. Accept a **text prompt** describing the target object.  
3. Use **GroundingDINO** to convert the text prompt into region proposals (bounding boxes).  
4. Pass bounding boxes into **SAM 2** (Segment Anything Model v2).  
5. Display the final segmentation mask overlayed on the image.  

### Dependencies  
The notebook installs the following:  
- `torch`, `torchvision`  
- `transformers`  
- `groundingdino` (for text-to-region grounding)  
- `segment-anything` / `sam2` (for segmentation)  
- `opencv-python`, `matplotlib`, `PIL` (for visualization)  

### How to Run  
1. Open `q2.ipynb` in **Colab**.  
2. Install all dependencies.  
3. Upload or load your test image.  
4. Enter a **text prompt** (e.g., `"cat"`, `"person"`, `"car"`).  
5. Run all cells to view the segmentation mask overlay.  

### Example Output  
- Input Image: User-provided image  
- Text Prompt: `"dog"`  
- Output: GroundingDINO generates bounding box → SAM2 refines it → Final segmented mask overlayed on image.

<img width="676" height="444" alt="image" src="https://github.com/user-attachments/assets/7b963fc8-d2cc-4f2b-b595-45316bc824b7" />


<img width="991" height="302" alt="image" src="https://github.com/user-attachments/assets/63a11657-5957-4d70-a88a-5e9218b161d2" />


### Limitations  
- GroundingDINO may struggle with vague or uncommon text prompts.  
- For overlapping/occluded objects, multiple regions may be detected.  
- Segmentation quality depends on both **GroundingDINO** localization and SAM2 refinement.  

### Bonus (in progress)  
- Extension to **video segmentation** possible:  
  - Use GroundingDINO + SAM2 on the first frame.  
  - Propagate masks to subsequent frames.  

---

## Results  
- **Q1:** Pending.  
- **Q2:** Successfully demonstrated text-driven image segmentation pipeline using GroundingDINO + SAM2.  
