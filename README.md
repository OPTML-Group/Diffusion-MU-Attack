<div align="center">

# TO GENERATE OR NOT? SAFETY-DRIVEN UNLEARNED DIFFUSION MODELS ARE STILL EASY TO GENERATE UNSAFE IMAGES ... FOR NOW
</div>
Welcome to the official implementation of the paper TO GENERATE OR NOT? SAFETY-DRIVEN UNLEARNED DIFFUSION MODELS ARE STILL EASY TO GENERATE UNSAFE IMAGES ... FOR NOW. This work introduces one fast and efficient attack methods to generate toxic content for safety-driven diffusion models.
<table align="center">
  <tr>
    <td align="center"> 
      <img src="assests/overview.png" alt="Image 1" style="width: 1000px;"/> 
      <br>
    </td>
  </tr>
</table>
<div align="left">

## Abstract 
The recent advances in diffusion models (DMs) have revolutionized the generation of complex and
diverse images. However, these models also introduce potential safety hazards, such as the produc-
tion of harmful content and infringement of data copyrights. Although there have been efforts to
create safety-driven unlearning methods to counteract these challenges, doubts remain about their
capabilities. To bridge this uncertainty, we propose an evaluation framework built upon adversarial
attacks (also referred to as adversarial prompts), in order to discern the trustworthiness of these
safety-driven unlearned DMs. Specifically, our research explores the (worst-case) robustness of un-
learned DMs in eradicating unwanted concepts, styles, and objects, assessed by the generation of
adversarial prompts. We develop a novel adversarial learning approach called UnlearnDiff that
leverages the inherent classification capabilities of DMs to streamline the generation of adversarial
prompts, making it as simple for DMs as it is for image classification attacks. This technique stream-
lines the creation of adversarial prompts, making the process as intuitive for generative modeling as it
is for image classification assaults. Through comprehensive benchmarking, we assess the unlearning
robustness of five prevalent unlearned DMs across multiple tasks. Our results underscore the effec-
tiveness and efficiency of UnlearnDiff when compared to state-of-the-art adversarial prompting
methods

## Code Structure
```configs```: contains the default parameter for each methods

```prompts```: contains the prompts we selected for each experiments

```src```: contains the source code for the proposed methods

* ```attackers```: contains different attack methods (different discrete optimization methods)
* ```tasks```: contains different type of attacks (auxiliary model-based attacks P4D, and ours UnlearnDiff)
* ```execs```: contains the main execution files to run experiments
* ```loggers```: contains the logger codes for the experiments

## Usage
In this section, we provide the instructions to reproduce the results on nudity (ESD) in our paper. You can change the config file path to reproduce the results on other concepts or unlearned models.

### Requirements

```conda env create -n ldm --file environments/x86_64.yaml```

### Unlearned model preparation 
Here we provide different unlearned models from ESD and FMN. You can download them from [here](). We also provide Artist classifier for evaluating the style task. You can download it from [here](https://drive.google.com/file/d/1me_MOrXip1Xa-XaUrPZZY7i49pgFe1po/view?usp=share_link).

### Generate dataset

```python src/execs/generate_dataset.py --prompts_path prompts/nudity.csv --concept nudity --save-path files/dataset/i2p_nude```


### No attack

```python src/execs/attack.py --config-file configs/nudity/no_attack_esd_classifier.json```

### UnlearnDiff attack

```python src/execs/attack.py --config-file configs/nudity/text_grad_esd_nudity_classifier.json```

### Evaluation

For ```nudity/violence/illegal```:

```python scripts/analysis/check_asr.py --root-no-attack $path_to_no_attack_results --root $path_to_${P4D|UnlearnDiff}_results ```

For ```style```:

```python scripts/analysis/style_analysis.py --root $path_to_${P4D|UnlearnDiff}_results --top_k {1|3}```

## bib 
If you find this work useful, please cite following papers:
```
@article{hou2022textgrad,
  title={Textgrad: Advancing robustness evaluation in nlp by gradient-driven optimization},
  author={Hou, Bairu and Jia, Jinghan and Zhang, Yihua and Zhang, Guanhua and Zhang, Yang and Liu, Sijia and Chang, Shiyu},
  journal={arXiv preprint arXiv:2212.09254},
  year={2022}
}

@article{jia2023model,
  title={Model sparsification can simplify machine unlearning},
  author={Jia, Jinghan and Liu, Jiancheng and Ram, Parikshit and Yao, Yuguang and Liu, Gaowen and Liu, Yang and Sharma, Pranay and Liu, Sijia},
  journal={arXiv preprint arXiv:2304.04934},
  year={2023}
}
```
