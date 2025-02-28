<div align="center">

# [ECCV 2024] <br> UnlearnDiffAtk:<br>To Generate or Not? <br> Safety-Driven Unlearned Diffusion Models <br> Are Still Easy To Generate Unsafe Images <br> ... For Now
</div>

###  [Project Website](https://www.optml-group.com/posts/mu_attack) | [Arxiv Preprint](https://arxiv.org/abs/2310.11868) | [Unlearned DM Benchmark](https://huggingface.co/spaces/Intel/UnlearnDiffAtk-Benchmark) | [Demo](https://huggingface.co/spaces/Intel/UnlearnDiffAtk)  | [Poster](https://damon-demon.github.io/links/ECCV24_UnlearnDiffAtk_poster.pdf)

Welcome to the official implementation of <strong>UnlearnDiffAtk</strong>, which capitalizes on the intrinsic classification abilities of Diffusion Models (DMs) to <strong>simplify the creation of adversarial prompts</strong>, thereby eliminating the need for auxiliary classification or diffusion models.Through extensive benchmarking, we <strong>evaluate the robustness</strong> of five widely-used safety-driven unlearned DMs (i.e., DMs after unlearning undesirable concepts, styles, or objects) across a variety of tasks.
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
safety-driven unlearned DMs. Specifically, our research explores the <strong>(worst-case) robustness of unlearned DMs</strong> in eradicating unwanted concepts, styles, and objects, assessed by the generation of
adversarial prompts. <strong>We develop a novel adversarial learning approach called UnlearnDiff that
leverages the inherent classification capabilities of DMs to streamline the generation of adversarial
prompts, making it as simple for DMs as it is for image classification attacks.</strong> This technique streamlines the creation of adversarial prompts, making the process as intuitive for generative modeling as it
is for image classification assaults. Through comprehensive benchmarking, we assess the unlearning
robustness of five prevalent unlearned DMs across multiple tasks. Our results underscore the effectiveness and efficiency of UnlearnDiff when compared to state-of-the-art adversarial prompting
methods

<br>

## [UnlearnDiffAtk: Unlearned Diffusion Model Benchmark](https://huggingface.co/spaces/Intel/UnlearnDiffAtk-Benchmark)

### We will evaluate your model on UnlearnDiffAtk Benchmark! 
Open a [github issue](https://github.com/OPTML-Group/Diffusion-MU-Attack/issues) or email us at zhan1853@msu.edu!

<strong>FULL UnlearnDiffAtk Unlearned DM benchmark</strong> can be found in [Huggingface](https://huggingface.co/spaces/Intel/UnlearnDiffAtk-Benchmark)!

<br>

#### DM Unlearning Tasks:
* <strong>NSFW</strong>: Nudity
* <strong>Style</strong>: Van Gogh
* <strong>Object</strong>: Church, Tench, Parachute, Garbage Truck

 #### Evaluation Metrics:
 * Pre-Attack Success Rate (<strong>Pre-ASR</strong>): lower is better;
 * Post-Attack Success Rate (<strong>Post-ASR</strong>): lower is better;
 * Fréchet inception distance(<strong>FID</strong>): evaluate distributional quality of image generations, lower is better;
 * <strong>CLIP Score</strong>: measure contextual alignment with prompt descriptions, higher is better.

Detailed instructions for performing utility evaluations (FID and CLIP score) are available in our DM Unlearning project, [AdvUnlearn](https://github.com/OPTML-Group/AdvUnlear). This repository includes guidance on downloading the COCO-10k dataset and provides scripts for calculating both FID and CLIP scores.


### Unlearned Concept: Nudity
| Unlearned Methods                                                                                            | Pre-ASR (%) | ASR (%) | FID   | CLIP-Score |
|--------------------------------------------------------------------------------------------------------------|---------|----------|-------|------------|
| [EraseDiff (ED)](https://github.com/JingWu321/EraseDiff)                                                     | 0.00    | 2.11     | 233   | 0.18       |
| [ScissorHands (SH)](https://github.com/JingWu321/Scissorhands)                                               | 0.00    | 7.04     | 128.53| 0.235      |
| [Saliency Unlearning (SalUn)](https://github.com/OPTML-Group/Unlearn-Saliency)                               | 1.41    | 11.27    | 33.62 | 0.287      |
| [Adversarial Unlearning (AdvUnlearn)](https://github.com/OPTML-Group/AdvUnlearn)                             | 7.75    | 21.13    | 19.34 | 0.290      |
| [Erased Stable Diffusion (ESD)](https://github.com/rohitgandikota/erasing)                                   | 20.42   | 76.05    | 18.18 | 0.302      |
| [Unified Concept Editing (UCE)](https://github.com/rohitgandikota/unified-concept-editing)                   | 21.83   | 79.58    | 17.10 | 0.309      |
| [Forget-Me-Not (FMN)](https://github.com/SHI-Labs/Forget-Me-Not)                                             | 88.03   | 97.89    | 16.86 | 0.308      |
| [concept-SemiPermeable Membrane (SPM)](https://github.com/Con6924/SPM)                                       | 54.93   | 91.55    | 17.48 | 0.310      |


<br>

## Code Structure
```configs```: contains the default parameter for each methods

```prompts```: contains the prompts we selected for each experiments

```src```: contains the source code for the proposed methods

* ```attackers```: contains different attack methods (different discrete optimization methods)
* ```tasks```: contains different type of attacks (auxiliary model-based attacks P4D, and ours UnlearnDiff)
* ```execs```: contains the main execution files to run experiments
* ```loggers```: contains the logger codes for the experiments

<br>

## Prepare

### Environment Setup
```
conda env create -n ldm --file environments/x86_64.yaml
```

### Unlearned model preparation 
We provide different unlearned models (ESD and FMN), and you can download them from [[Object](https://drive.google.com/file/d/1e5aX8gkC34YaHGR0S1-EQwBmUXiAPvpE/view?usp=sharing) , [Others](https://drive.google.com/file/d/1yeZNJ8MoHsisdZmt5lbnG_kSgl5xned0/view?usp=sharing)]. We also provide an Artist classifier for evaluating the style task. You can download it from [here](https://drive.google.com/file/d/1me_MOrXip1Xa-XaUrPZZY7i49pgFe1po/view?usp=share_link).

### Generate dataset (for unlearning robustness evaluation)

```python src/execs/generate_dataset.py --prompts_path prompts/nudity.csv --concept i2p_nude --save_path files/dataset```


<br>

## Code Implementation
In this section, we provide the instructions to reproduce the results on nudity (ESD) in our paper. You can change the config file path to reproduce the results on other concepts or unlearned models.

### No attack

```python src/execs/attack.py --config-file configs/nudity/no_attack_esd_nudity_classifier.json --attacker.attack_idx $i --logger.name attack_idx_$i```

where ```i``` is from ```[0,142)```

### UnlearnDiff attack

```python src/execs/attack.py --config-file configs/nudity/text_grad_esd_nudity_classifier.json --attacker.attack_idx $i --logger.name attack_idx_$i```

where ```i``` is from ```[0,142)```

### Evaluation

For ```nudity/violence/illegal/objects```:

```python scripts/analysis/check_asr.py --root-no-attack $path_to_no_attack_results --root $path_to_${P4D|UnlearnDiff}_results```

For ```style```:

```python scripts/analysis/style_analysis.py --root $path_to_${P4D|UnlearnDiff}_results --top_k {1|3}```


<br>

## Citation

```
@article{zhang2024generate,
  title={To generate or not? safety-driven unlearned diffusion models are still easy to generate unsafe images... for now},
  author={Zhang, Yimeng and Jia, Jinghan and Chen, Xin and Chen, Aochuan and Zhang, Yihua and Liu, Jiancheng and Ding, Ke and Liu, Sijia},
  journal={European Conference on Computer Vision (ECCV)},
  year={2024}
}
```
<br>

## Related Works - Machine Unlearning

* [Adversarial Unlearning (AdvUnlearn)](https://github.com/OPTML-Group/AdvUnlearn)
* [Sparse Unlearning (l1-sparse)](https://github.com/OPTML-Group/Unlearn-Sparse)
* [Saliency Unlearning (SalUn)](https://github.com/OPTML-Group/Unlearn-Saliency)
