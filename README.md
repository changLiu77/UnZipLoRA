# UnZipLoRA: Separating Content and Style from a Single Image
[\[Paper\]](https://arxiv.org/abs/2412.04465) 
[\[Project Page\]](https://unziplora.github.io/)


__UnZipLoRA__, a method for decomposing an image into its constituent subject and style, represented as two distinct LoRAs (Low-Rank Adaptations) by training both the LoRAs simultaneously. UnZipLoRA ensures that the resulting LoRAs are compatible, i.e., they can be seamlessly combined using direct addition. UnZipLoRA enables independent manipulation and recontextualization of subject and style, including generating variations of each, applying the extracted style to new subjects, and recombining them to reconstruct the original image or create novel variations.

This is the official release of the __UnZipLoRA__ and more details can be found in our paper [UnZipLoRA: Separating Content and Style from a Single Image](https://arxiv.org/abs/2412.04465).

    
![image](cover_images/teaser.png)

## Requirements

Install dependencies:

```
conda create -n unziplora python=3.11
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
conda activate unziplora
pip install -r requirements.txt
pip install jupyter notebook
```

## Data

We provide the figures and corresponding prompts used for training and evaluation in the [data](instance_data).

## Train

To train the content and style model symotanenously, you need to provide:

* One or more reference images
* A prompt, a content prompt and a style prompt
* Output path to save the trained models

You can directly run the [training script](train.sh).  

__UnZipLoRA__ proposes three separation strategies. All hyperparameters in the script match the settings used in the paper, and can be adjusted for ablation and tuning.

**Core Hyperparameters**

| Flag                          | Description                              | Default        |
|------------------------------|------------------------------------------|----------------|
| `--pretrained_model_name_or_path` | Path to base model (e.g. SDXL)      | Required       |
| `--instance_data_dir`         | Path to training images                  | Required       |
| `--output_dir`                | Path to save outputs                     | Required       |
| `--instance_prompt`          | Prompt used for training                 | Required       |
| `--content_forward_prompt`   | Prompt used for training(content)         | Required       |
| `--style_forward_prompt`     | Prompt used for training(style)           | Required       |
| `--rank`                      | LoRA rank                                | `64`           |
| `--max_train_steps`           | Number of training steps                 | `600`         |
| `--content_learning_rate`     | LR for content LoRA                      | `5e-5`         |
| `--style_learning_rate`       | LR for style LoRA                        | `5e-5`         |
| `--weight_learning_rate`      | LR for weight fusion                     | `5e-3`         |

---

**Logging / Ablation Flags**

| Flag                          | Description                              | Default        |
|------------------------------|------------------------------------------|----------------|
| `--with_period_column_separation`| Apply column separation              | `True`        |
| `--with_freeze_unet`         | Apply block separation                   | `True`        |
| `--with_saved_per_validation`| Save model on every validation step      | `False`        |
| `--with_image_per_validation`| Generate images at every validation step    | `False`        |

---

**Experimental**

| Flag                          | Description                              | Default        |
|------------------------------|------------------------------------------|----------------|
| `--sample_times`             | Times of re-caliberate LoRA column masks | `3`            |
| `--column_ratio`             | Ratio of sampled columns for LoRA mask   | `0.1`          |


## Infer

After training, use [infer script](infer.sh) to generate images with your trained LoRAs. You can:

* Generate content / style recontextualizations with individual saved content / style models and given prompts
* Generate combined recontexualization by merging both content and style LoRAs with learned mask weights 
* Generate cross-image recontexualization by combining LoRAs and mask weights from different source figures

We also provide a [__notebook__](playground.ipynb) to play with these functionalities.

You can download two example models from [__Google Drive__](https://drive.google.com/file/d/1oaDBGntlg3yX3Nig1BaLEllLKp82-l21/view?usp=drive_link) and try out these functionalities using them.

## Citations
If you find this work helpful, please cite us as:
```
@misc{liu2024unziploraseparatingcontentstyle,
title={UnZipLoRA: Separating Content and Style from a Single Image},
author={Chang Liu and Viraj Shah and Aiyu Cui and Svetlana Lazebnik},
year={2024},
eprint={2412.04465},
archivePrefix={arXiv},
primaryClass={cs.CV},
url={https://arxiv.org/abs/2412.04465},
}
```
