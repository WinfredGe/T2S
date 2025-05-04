<div align="center">
  <!-- <h1><b> T2S </b></h1> -->
  <!-- <h2><b> T2S </b></h2> -->
  <h2><b> (IJCAI'25) <span style="color:rgb(185,5,14)">T</span><span style="color:rgb(19,175,85)">2</span><span style="color:rgb(46,96,179)">S</span>: High-resolution Time Series Generation with Text-to-Series Diffusion Models </b></h2>
</div>

<div align="center">

![](https://img.shields.io/github/last-commit/WinfredGe/T2S?color=green)
![](https://img.shields.io/github/stars/WinfredGe/T2S?color=yellow)
![](https://img.shields.io/github/forks/WinfredGe/T2S?color=lightblue)
![](https://img.shields.io/badge/PRs-Welcome-green)

</div>

<p align="center">
    <img src="./figures/logo.png" width="70">
</p>

> 1️⃣ T2S is the **first work** for text-to-time series generation with a domain-agnostic approach.

> 2️⃣ TSFragment-600K is the **first** fragment-level text-time series pairs dataset comprising across 6 classic domains.


---
>
>🙋 Please let us know if you find out a mistake or have any suggestions!
>
>🌟 If you find this resource helpful, please consider to star this repository and cite our research:

```
@inproceedings{ge2025t2s,
  title={{T2S}: High-resolution Time Series Generation with Text-to-Series Diffusion Models},
  author={Ge, Yunfeng and Li, Jiawei and Zhao, Yiji and Wen, Haomin and Li, Zhao and Qiu, Meikang and Li, Hongyan and Jin, Ming and Pan, Shirui},
  booktitle={International Joint Conference on Artificial Intelligence (IJCAI)},
  year={2025}
}
```

## Introduction
T2S is the first domain-agnostic model for text-to-time series generation. This allows ordinary people to describe temporal changes without requiring specialized expertise in a particular field. 

Application Scenarios:

 (1) Ordinary people can create time series data and engage with data-driven tools without needing advanced skills. This could encourage **broader participation in data analysis**.

 (2) Professionals can use simple textual descriptions to quickly generate time series data that simulate specific system behaviors. This capability supports **rapid prototyping** and analysis of system evolution under different conditions. 

(3) It can be used for **stress testing** systems, such as creating "an extreme surge in demand" to assess a database’s responsiveness or network elements’  carrying capacity under extreme cases. Note that traditional methods struggle to model these extreme cases because they rely on stationary source data distributions.

<p align="center">
<img src="./figures/method.png" height = "360" alt="" align=center />
</p>


- T2S comprises two key components: (1) T2S Diffusion Transformer and (2) Pretrained Length-Adaptive Variational Autoencoder,  to empower the capability of generating semantically aligned time series of arbitrary lengths.  
- *TSFragment-600K* comprising over 600,000 fragment-level text-time series pairs. Each captions captures fine-grained temporal morphological characteristics, offering a rich and nuanced representation of the underlying trends.

<p align="center">
<img src="./figures/dataset.png" height = "190" alt="" align=center />
</p>



## Requirements

Use python 3.10 from Conda

- torch==2.3.1
- datasets==2.21.0
- einops==0.7.0
- numpy==1.26.4
- pandas==1.5.3
- scipy==1.14.1
- seaborn==0.13.2
- tqdm==4.66.5
- transformers==4.47.0
- timm==1.0.11
- sentencepiece==0.2.0
- peft==0.10.0
- openai==1.35.9

To install all dependencies:

```shell
pip install -r requirements.txt
```

## 📚 Datasets

[TSFragment-600K dataset](https://huggingface.co/datasets/WinfredGe/TSFragment-600K) is available on 🤗 Hugging Face.

You can follow the following usage example to call TSFragment-600K dataset, or download all well pre-processed [[three levels datasets]](https://drive.google.com/file/d/1tV0xBd0ToWvuLpI5Ocd49uM3QcRkP4NT/view?usp=sharing), then place them under `./Data` directory.

```
from datasets import load_dataset
ds = load_dataset("WinfredGe/TSFragment-600K")
```

```
Data
├─ TSFragment-600K
│  ├─ embedding_cleaned_airquality_24.csv
│  ├─ embedding_cleaned_airquality_48.csv
│  ├─ embedding_cleaned_airquality_96.csv
│  ├─ embedding_cleaned_electricity_24.csv
│  ├─ embedding_cleaned_electricity_48.csv
│  ├─ embedding_cleaned_electricity_96.csv
│  ├─ embedding_cleaned_ETTh1s_24.csv
│  ├─ embedding_cleaned_ETTh1s_48.csv
│  ├─ embedding_cleaned_ETTh1s_96.csv
│  ├─ embedding_cleaned_ETTh1_24.csv
│  ├─ embedding_cleaned_ETTh1_48.csv
│  ├─ embedding_cleaned_ETTh1_96.csv
│  ├─ embedding_cleaned_ETTm1_24.csv
│  ├─ embedding_cleaned_ETTm1_48.csv
│  ├─ embedding_cleaned_ETTm1_96.csv
│  ├─ embedding_cleaned_exchangerate_24.csv
│  ├─ embedding_cleaned_exchangerate_48.csv
│  ├─ embedding_cleaned_exchangerate_96.csv
│  ├─ embedding_cleaned_traffic_24.csv
│  ├─ embedding_cleaned_traffic_48.csv
│  └─ embedding_cleaned_traffic_96.csv
├─ SUSHI
│  └─ embedding_cleaned_SUSHI.csv
└─ MMD
   ├─ embedding_cleaned_Agriculture_24.csv
   ├─ embedding_cleaned_Agriculture_48.csv
   ├─ embedding_cleaned_Agriculture_96.csv
   ├─ embedding_cleaned_Climate_24.csv
   ├─ embedding_cleaned_Climate_48.csv
   ├─ embedding_cleaned_Climate_96.csv
   ├─ embedding_cleaned_Economy_24.csv
   ├─ embedding_cleaned_Economy_48.csv
   ├─ embedding_cleaned_Economy_96.csv
   ├─ embedding_cleaned_Energy_24.csv
   ├─ embedding_cleaned_Energy_48.csv
   ├─ embedding_cleaned_Energy_96.csv
   ├─ embedding_cleaned_Health_US_24.csv
   ├─ embedding_cleaned_Health_US_48.csv
   ├─ embedding_cleaned_Health_US_96.csv
   ├─ embedding_cleaned_SocialGood_24.csv
   ├─ embedding_cleaned_SocialGood_48.csv
   └─ embedding_cleaned_SocialGood_96.csv
```

We also open source the dataset construction and evaluation pipeline in `./Dataset_Construction_Pipeline/` folder.

## Get Started

Core Structure Overview:

```
T2S-main
├─ pretrained_lavae_unified.py
├─ train.py
├─ infer.py
├─ evaluation.py
├─ datafactory
│  ├─ dataloader.py
│  └─ dataset.py
├─ model
│  ├─ pretrained
│  │  ├─ core.py
│  │  └─ vqvae.py
│  ├─ denoiser
│  │  ├─ mlp.py
│  │  └─ transformer.py
│  └─ backbone
│     ├─ DDPM.py
│     └─ rectified_flow.py
└─ evaluate
   ├─ feature_based_measures.py
   ├─ ts2vec.py
   └─ utils.py

```

1. Install Python 3.10, and then install the dependencies:

```shell
pip install -r requirements.txt
```

**Note: Time-MoE requires `torch==2.3.1` .**



1. Install Python 3.10 and PyTorch 2.3.1.
2. Download the [*TSFragment-600K* data](https://drive.google.com/file/d/1YEe66ptAl52yp17MXVO9xWOe5rS1yUyZ/view?usp=sharing) and [checkpoints](https://drive.google.com/file/d/1T-gjPMvnpSFpkkUSZpAeeIqALThOQydT/view?usp=sharing) from Google Drive to `./`
3. Train and perform inference with the model. We provide the experiment script under the  `./script.sh`. (optional)
4. You can evaluate the model using  `./scripts_validation_only.sh` directly.
`

## Detailed usage

Please refer to ```pretrained_lavae_unified.py```, ```train.py```, ```infer.py``` and ```evaluation.py``` for the detailed description of each hyperparameter.

## Further Reading
1, [**Position Paper: What Can Large Language Models Tell Us about Time Series Analysis**](https://arxiv.org/abs/2402.02713), in *ICML* 2024.

**Authors**: Ming Jin, Yifan Zhang, Wei Chen, Kexin Zhang, Yuxuan Liang*, Bin Yang, Jindong Wang, Shirui Pan, Qingsong Wen*

```bibtex
@inproceedings{jin2024position,
   title={Position Paper: What Can Large Language Models Tell Us about Time Series Analysis}, 
   author={Ming Jin and Yifan Zhang and Wei Chen and Kexin Zhang and Yuxuan Liang and Bin Yang and Jindong Wang and Shirui Pan and Qingsong Wen},
  booktitle={International Conference on Machine Learning (ICML 2024)},
  year={2024}
}
```
2, [**A Survey on Diffusion Models for Time Series and Spatio-Temporal Data**](https://arxiv.org/abs/2404.18886), in *arXiv* 2024.
[\[GitHub Repo\]](https://github.com/yyysjz1997/Awesome-TimeSeries-SpatioTemporal-Diffusion-Model/blob/main/README.md)

**Authors**: Yiyuan Yang, Ming Jin, Haomin Wen, Chaoli Zhang, Yuxuan Liang, Lintao Ma, Yi Wang, Chenghao Liu, Bin Yang, Zenglin Xu, Jiang Bian, Shirui Pan, Qingsong Wen

```bibtex
@article{yang2024survey,
  title={A survey on diffusion models for time series and spatio-temporal data},
  author={Yang, Yiyuan and Jin, Ming and Wen, Haomin and Zhang, Chaoli and Liang, Yuxuan and Ma, Lintao and Wang, Yi and Liu, Chenghao and Yang, Bin and Xu, Zenglin and others},
  journal={arXiv preprint arXiv:2404.18886},
  year={2024}
}
```
3, [**Foundation Models for Time Series Analysis: A Tutorial and Survey**](https://arxiv.org/pdf/2403.14735), in *KDD* 2024.

**Authors**: Yuxuan Liang, Haomin Wen, Yuqi Nie, Yushan Jiang, Ming Jin, Dongjin Song, Shirui Pan, Qingsong Wen*

```bibtex
@inproceedings{liang2024foundation,
  title={Foundation models for time series analysis: A tutorial and survey},
  author={Liang, Yuxuan and Wen, Haomin and Nie, Yuqi and Jiang, Yushan and Jin, Ming and Song, Dongjin and Pan, Shirui and Wen, Qingsong},
  booktitle={ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD 2024)},
  year={2024}
}
```

## Acknowledgement

Our implementation adapts [Time-Series-Library](https://github.com/thuml/Time-Series-Library), [TSGBench](https://github.com/YihaoAng/TSGBench), [TOTEM](https://github.com/SaberaTalukder/TOTEM) and [Meta (Scalable Diffusion Models with Transformers)](https://github.com/facebookresearch/DiT) as the code base and have extensively modified it to our purposes. We thank the authors for sharing their implementations and related resources.

## License

This project is licensed under the Apache-2.0 License.
