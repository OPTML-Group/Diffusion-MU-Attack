<div align="center">

# To Generate or Not? <br> Safety-Driven Unlearned Diffusion Models <br> Are Still Easy To Generate Unsafe Images <br> ... For Now
</div>
Welcome to the official implementation of the paper: [To Generate or Not? Safety-Driven Unlearned Diffusion Models Are Still Easy To Generate Unsafe Images ... For Now](https://arxiv.org/abs/2310.11868). This work introduces one fast and efficient attack methods to generate toxic content for safety-driven diffusion models.
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
Here we provide different unlearned models from ESD and FMN. You can download them from [here](https://drive.google.com/file/d/1f4gncLqMXXdlbxpiFtY0WefDL1YUr9b7/view?usp=sharing). We also provide an Artist classifier for evaluating the style task. You can download it from [here](https://drive.google.com/file/d/1me_MOrXip1Xa-XaUrPZZY7i49pgFe1po/view?usp=share_link).

### Generate dataset

```python src/execs/generate_dataset.py --prompts_path prompts/nudity.csv --concept i2p_nude --save_path files/dataset```


### No attack

```python src/execs/attack.py --config-file configs/nudity/no_attack_esd_nudity_classifier.json --attacker.attack_idx $i --logger.name attack_idx_$i```

where ```i``` is from ```[0,142)```

### UnlearnDiff attack

```python src/execs/attack.py --config-file configs/nudity/text_grad_esd_nudity_classifier.json --attacker.attack_idx $i --logger.name attack_idx_$i```

where ```i``` is from ```[0,142)```

### Evaluation

For ```nudity/violence/illegal```:

```python scripts/analysis/check_asr.py --root-no-attack $path_to_no_attack_results --root $path_to_${P4D|UnlearnDiff}_results ```

For ```style```:

```python scripts/analysis/style_analysis.py --root $path_to_${P4D|UnlearnDiff}_results --top_k {1|3}```


### Citation

```
@article{zhang2023generate,
  title={To Generate or Not? Safety-Driven Unlearned Diffusion Models Are Still Easy To Generate Unsafe Images... For Now},
  author={Zhang, Yimeng and Jia, Jinghan and Chen, Xin and Chen, Aochuan and Zhang, Yihua and Liu, Jiancheng and Ding, Ke and Liu, Sijia},
  journal={arXiv preprint arXiv:2310.11868},
  year={2023}
}
```
