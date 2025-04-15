# UnZipLoRA: Separating Content and Style from a Single Image
[\[Paper\]](https://arxiv.org/abs/2412.04465) 
[\[Project Page\]](https://unziplora.github.io/)


__UnZipLoRA__, a method for decomposing an image into its constituent subject and style, represented as two distinct LoRAs (Low-Rank Adaptations) by training both the LoRAs simultaneously. UnZipLoRA ensures that the resulting LoRAs are compatible, i.e., they can be seamlessly combined using direct addition. UnZipLoRA enables independent manipulation and recontextualization of subject and style, including generating variations of each, applying the extracted style to new subjects, and recombining them to reconstruct the original image or create novel variations.

This is the official release of the __UnZipLoRA__ and more details can be found in our paper [UnZipLoRA: Separating Content and Style from a Single Image](https://arxiv.org/abs/2412.04465).

    
![image](cover_images/teaser.png)

## Requirements

Install dependencies

```
conda create -n unziplora python=3.11
conda activate unziplora
pip install -r requirements.txt
pip install jupyter notebook
```

## Data

We have public the figures and corresponding prompts we used to train and test our models in [data](instance_data)

## Train
Street TryOn Dataset contains __unpaired__ __in-the-wild person images__ that can be used for virtual try-on tasks. Street TryOn Dataset consists of 12,364 and 2089 images filtered from [Deepfashion2 Dataset](https://github.com/switchablenorms/DeepFashion2) for training and validation.


We release all the annotations mentioned in [our paper](https://arxiv.org/pdf/2311.16094.pdf). Note for images: we provide scripts to extract them from DeepFashion2 dataset. Please follow the below steps to download the dataset into your datapath `$DATA`. 

## Infer

After training, use [`infer.py`](infer.py) to generate images with your trained LoRAs. Given prompts, content prompts, style prompts for combined generation, while give either content or style prompts.

We also provide a [notebook](playground.ipynb) to play with in-figure generation and cross-figure generation.

Note:
- The default dataset configuration is loading ATR segmentation for street images and loading the provided segmentation for VITON-HD. If you need either in different format, you can change the `segm_dir` or `garment_segm_dir` in the corresponding ``.yaml`` config file with your new datapath.


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
