# PlanLLM: Video Procedure Planning with Refinable Large Language Models


<!-- ## Updates
- 2024/12/16: Release `PlanLLM-Vicuna7B` code, checkpoints and dataset
- 2024/12/90: **PlanLLM** accepted by **AAAI2025** -->
## Overview
> Video procedure planning, *i.e.*, planning a sequence of action steps given the video frames of start and goal states, is an essential ability for embodied AI.
Recent works utilize Large Language Models (LLMs) to generate enriched action step description texts to guide action step decoding.
Although introducing LLMs, these methods decode the action steps into a close set of one-hot vectors, limiting the model's capability of generalizing to new steps or tasks.
Additionally, the fixed action step descriptions based on world-level commonsense may contain noise in specific samples of visual states.
In this paper, we propose PlanLLM, a cross-modal joint learning framework with LLMs for video procedure planning.
We propose LLM-Enhanced Planning module which fully use the generalization ability of LLMs to produce free form planning outputs and to enhance action step decoding.
We also propose Mutual Information Maximization module to connect world-level commonsense of step descriptions and sample-specific information of visual states, enabling LLMs to employ the reasoning ability to generate step sequence.
With the assistance of LLMs, our method can deal with both close set and open vocabulary procedure planning tasks.
Our PlanLLM achieves superior performance on three benchmarks, demonstrating the effectiveness of our designs.

## 1. Environment Setup

Either creating manually:

```bash
git clone https://github.com/idejie/PlanLLM.git
cd PlanLLM
export PYTHONPATH=<YOUR_PROJECT_PATH>
conda create -n PlanLLM python==3.12
conda activate PlanLLM
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
conda install tensorboardX pandas ftfy regex
pip install loguru timm peft opencv_python fvcore transformers imageio  wandb sentencepiece einops scipy
MAX_JOBS=16 pip install flash-attn --no-build-isolation # wait a long time to build the dependency...

```

## 2. Download Pretrained Models:
- [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased)
- [lmsys/vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5)

*Note*: (1) put them into `pretrained` directory; (2)You can use the `https://hf-mirror.com/` in China



## 3. Data Preparation


### 3.1 Download Datasets

Download pre-extracted HowTo100M features

```bash
# CrossTask
bash scripts/dataset/download_crosstask.sh
# COIN
bash scripts/dataset/download_coin.sh
# NIV
bash scripts/dataset/download_niv.sh
```

### 3.2(Optional) Generate Descriptions

The descriptions of actions and states have been already provided in this repo. The raw descriptions are saved as .json files in the "data" folder. The state and action description features extracted by CLIP language encoder are saved respectively in the "data/state_description_features" and "data/action_description_features" folders.

If you want to customize the prompts and generate new descriptions, please follow the steps below:

1. Modify line 9 of *generate_descriptors.py*, set the variable *openai_key* to your OpenAI key.
2. Modify the prompt starting from line 25 of *generate_descriptors.py*.
3. Download OpenAI package and generate description files:

   ```
   pip install openai
   python tools/generate_descriptors.py --dataset [DATASET]
   ```

   **Note: Replace the [DATASET] with a specific dataset: crosstask or coin or niv. (Same for the following steps)**
4. Extract description features:

   ```
   python tools/extract_description_feature.py --dataset [DATASET]
   ```

## 4. Model Train
- modify the variable **dataset**(`niv` or  `crosstask` or `coin`) and **max_traj_len**(time_horizon=3,4) in `scripts/planllm_vicuna/config_qformer.py`   
- run the training script:
   ```
   bash scripts/planllm_vicuna/run_qformer.py
   ```

## 5. (Optional)Evaluation
### 5.1 modify settings
- modify the variable **dataset**(`niv` or  `crosstask` or `coin`) and **max_traj_len**(time_horizon=3,4) in `scripts/planllm_vicuna/config_llm.py`    
- modify the variable **vit_blip_model_path** in `scripts/planllm_vicuna/config_llm.py` to your checkpoints path

### 5.2 results and checkpoints
| Dataset   | Success Rate | Checkpoints |
| --------- | ------------- | ----------- |
| niv(T=3)      | 30.63         |    [BaiduNetdisk](https://pan.baidu.com/s/1SxSNwxqI1WzfO7iJ83Wy6w?pwd=plan)     |
| niv(T=4)      | 24.81         |     [BaiduNetdisk](https://pan.baidu.com/s/1Wtow-gQP4xKPNEyDeFlWqg?pwd=plan)     |
| crosstask(T=3) | 40.01         |   [BaiduNetdisk]()  【uploading】        |
| crosstask(T=4) | 25.61           |   [BaiduNetdisk]()   【uploading】       |
| coin(T=3)   | 33.22         |    [BaiduNetdisk]()   【uploading】      |
| coin(T=4)      | 25.13         |  [BaiduNetdisk]()  【uploading】         |
### 5.3 Run
run the training script:
   ```
   bash scripts/planllm_vicuna/run_llm.py
   ```
## 6.Citing
Please consider citing our paper if it helps your research.
```
@inproceedings{PlanLLM,
 title={PlanLLM: Video Procedure Planning with Refinable Large Language Models},
 author={Dejie Yang, Zijing Zhao, and Yang Liu},
 year={2025},
 booktitle={AAAI},
}
```