# AIRL 

## How to Run in Colab  
- Open the notebook in **Google Colab**.  
- Enable GPU runtime (`Runtime` > `Change runtime type` > `GPU`).  
- Run all cells sequentially from top to bottom.  

---

## Q1 Vision Transformer on CIFAR-10  
- **Status:** In Progress   
- **Objective:** Implement and train a Vision Transformer (ViT) from scratch on CIFAR-10 to achieve the highest possible test accuracy.  
- **Planned Implementation:**  
  - Patchify CIFAR-10 images into tokens.  
  - Add learnable positional embeddings and CLS token.  
  - Stack Transformer encoder blocks (MHSA + MLP with residual + normalization).  
  - Train with AdamW optimizer + CosineAnnealingLR scheduler.  
- **Planned Experiments:**  
  - Patch size variations.  
  - Depth/width trade-offs.  
  - Data augmentation (RandAugment, Mixup, CutMix).  
  - Optimizer/scheduler variants.  
- **Results:** Pending.  
- A concise analysis and accuracy table will be updated after experiments are complete.  

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
