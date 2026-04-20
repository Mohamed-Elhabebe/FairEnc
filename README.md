# FairEnc: A Fair Vision-Language Model with Fair Vision and Text Encoders for Glaucoma Detection

This repository contains the official implementation for the paper:

> **FairEnc: A Fair Vision-Language Model with Fair Vision and Text Encoders for Glaucoma Detection**

---

## 📄 Abstract

Automated glaucoma detection is critical for preventing irreversible vision loss and reducing the burden on healthcare systems. However, ensuring fairness across diverse patient populations remains a significant challenge in medical AI. In this work, we propose **FairEnc**, a fairness-aware pretraining framework for vision–language models (VLMs) that enables *simultaneous debiasing across multiple sensitive attributes within a single model*.

FairEnc jointly mitigates bias in both textual and visual modalities with respect to sensitive attributes including **race, gender, ethnicity, and language**.  
For the textual encoder, we leverage a large language model (LLM) to generate synthetic clinical descriptions with varied demographic attributes while preserving disease semantics, and apply a contrastive alignment objective to encourage demographic-invariant representations.  
For the visual encoder, we introduce a dual-level fairness optimization strategy combining **mutual information regularization** and **multi-discriminator adversarial debiasing**.

Comprehensive experiments on the **Harvard-FairVLMed** dataset demonstrate that FairEnc substantially reduces demographic disparity (measured using **DPD** and **DEOdds**) while maintaining strong diagnostic performance under both **zero-shot** and **linear probing** evaluations. Additional experiments on the **FairFundus** dataset confirm strong generalization under cross-domain settings.

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

## Repository Contents

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