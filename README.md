# FairEnc: A Fair Vision-Language Model with Fair Vision and Text Encoders for Glaucoma Detection

This repository contains the official implementation of:

> **FairEnc: A Fair Vision-Language Model with Fair Vision and Text Encoders for Glaucoma Detection**

---

## 🏗️ FairEnc Framework

<p align="center">
  figs/FairEnc_Architecture.png
</p>

<p align="center">
  <em>
  Overview of FairEnc. The framework jointly debiases textual and visual representations through synthetic clinical note generation, contrastive text learning, mutual information regularization, and adversarial debiasing.
  </em>
</p>

---

## 📄 Abstract

Automated glaucoma detection is critical for preventing irreversible vision loss and reducing the burden on healthcare systems. However, ensuring fairness across diverse patient populations remains a significant challenge.

In this paper, we propose **FairEnc**, a fair pretraining method for vision-language models (VLMs) that enables simultaneous debiasing across multiple sensitive attributes.

FairEnc jointly mitigates biases in both textual and visual modalities with respect to multiple sensitive attributes, including **race, gender, ethnicity, and language**. Specifically, for the textual encoder, we leverage a large language model to generate synthetic clinical descriptions with varied sensitive attributes while preserving disease semantics, and employ a contrastive alignment objective to encourage demographic-invariant representations. For the visual encoder, we propose a dual-level fairness strategy that combines **mutual information regularization** to reduce statistical dependence between learned features and demographic groups with **multi-discriminator adversarial debiasing**.

Comprehensive experiments on the publicly available **Harvard-FairVLMed** dataset show that FairEnc reduces demographic disparities measured by **DPD** and **DEOdds** while achieving strong diagnostic performance under both **zero-shot** and **linear probing** evaluations. Additional experiments on the **FairFundus** dataset suggest that the fairness benefits of FairEnc generalize to cross-domain and cross-modality settings while maintaining diagnostic performance within a competitive range. These results highlight FairEnc's ability to generalize fairness under distribution shifts, supporting its potential for more equitable deployment in real-world clinical settings.

---

## Synthetic Clinical Notes

The repository includes a pre-generated CSV file:

**`qwen_synthetic_all_notes.csv`**

This file contains LLM-generated clinical notes used for fairness-aware training:

- `neutral_note`: demographic-neutral clinical description  
- `random_note_1` → `random_note_5`: synthetic descriptions with randomized demographic attributes  

📌 **Important:**  
This file **must be placed inside the Harvard-FairVLMed dataset directory**, as the training pipeline expects it to be co-located with the dataset when using mixed-note training.

---

## Main Repository Contents

- `finetune.py`  
  FairEnc fine-tuning for CLIP vision–language models.
- `evaluate.py`  
  Zero-shot evaluation of fine-tuned models.
- `linear_probe.py`  
  Linear probing (training a classifier on frozen representations).
- `evaluate_linear_probe.py`  
  Evaluation of trained linear probe models.
- `scripts/`  
  Shell scripts to run each task (fine-tuning, evaluation, linear probing).
- `requirements.txt`  
  Exact Python environment used in experiments.

---

## Dependencies

Install the required Python packages using:

```bash
pip install -r requirements.txt
```
---

## 🔧 Placeholders to Replace in the `.sh` Scripts

The following placeholders appear across the provided scripts and **must be replaced before execution**.

---

### Dataset and Code Paths

- **`DATASET_DIR`**  
  Path to the **Harvard-FairVLMed** dataset directory.

- **`</path/to/FairEnc_codebase>`**  
  Absolute path to the root directory of this repository.

---


### Output and Logging Paths

- **`RESULT_DIR`**  
  Directory where results, logs, and evaluation outputs will be saved.

- **`</linearprobe_checkpoint_saving/path>`**  
  Directory used to store linear probing checkpoints.

- **`</test_saving_and_logging/path>`**  
  Directory used for evaluation outputs.

---

### Model and Checkpoints

- **`MODEL_ARCH`**  
  CLIP backbone architecture. Supported options:
  - `vit-b16`
  - `vit-l14`

- **`</path/to/checkpoint_to_evaluate/.pth_file>`**  
  Path to a FairEnc-trained model checkpoint for zero-shot evaluation.

- **`</path/to/checkpoint_to_apply_linearprobing/.pth_file>`**  
  Path to a FairEnc checkpoint used as initialization for linear probing.

- **`</path/to/folder_containing/linearprobe_checkpoint_to_evaluate>`**  
  Directory containing the trained linear probe checkpoint (`clip_best.pth`).

---

### Comet.ml Logging (Required for Fine-Tuning)

The FairEnc fine-tuning script logs experiments to **Comet.ml**.  
The following placeholders must be set:

- **`<your_comet_api_key>`**  
  Your personal Comet.ml API key.

- **`<your_comet_project_name>`**  
  The Comet.ml project name to which experiments are logged.

📌 **Note:**  
You must create a **Comet.ml account and project** before running the fine-tuning script.