# ClothTransformer: Unified Latent-Space Transformers for Scalable Cloth Simulation (NeurIPS 2026)

<p align="center">
  <a href="https://yucrazing.github.io/clothtransformer/"><img src="https://img.shields.io/badge/Project-Page-4c8bf5?logo=googlechrome&logoColor=white" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2605.27852"><img src="https://img.shields.io/badge/arXiv-2605.27852-b31b1b?logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://huggingface.co/datasets/YuCrazing/ClothTransformer-dataset"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-ffcc4d" alt="Dataset"></a>
  <a href="https://huggingface.co/YuCrazing/ClothTransformer-checkpoint"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Checkpoint-ffcc4d" alt="Checkpoint"></a>
  <a href="https://x.com/yucrazing/status/2064254772086280337"><img src="https://img.shields.io/badge/X-Post-000000?logo=x&logoColor=white" alt="X Post"></a>
</p>

![teaser](https://yucrazing.github.io/clothtransformer/static/images/teaser.webp)

## News

- **[2026-09-26]** ClothTransformer has been accepted to NeurIPS 2026!
- **[2026-07-17]** The dataset is available on [Hugging Face](https://huggingface.co/datasets/YuCrazing/ClothTransformer-dataset).

## Installation

```bash
git clone https://github.com/YuCrazing/ClothTransformer.git
cd ClothTransformer
conda create -n ct python=3.10 -y
conda activate ct
# Install PyTorch for your CUDA version first (https://pytorch.org); tested with torch 2.4.0 + CUDA 12.1
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Pretrained Model

Download the checkpoint from [Hugging Face](https://huggingface.co/YuCrazing/ClothTransformer-checkpoint)
to `checkpoints/`:

```bash
hf download YuCrazing/ClothTransformer-checkpoint ct-no-ccd-preview.pt --local-dir checkpoints
```

## Examples

[`examples/showcase.txt`](examples/showcase.txt) lists three test trajectories, one per scenario. [`examples/README.md`](examples/README.md) shows rollout videos of them. To reproduce:

```bash
python scripts/download_data.py --file-list examples/showcase.txt
python main.py infer.ckpt=checkpoints/ct-no-ccd-preview.pt infer.file_list=examples/showcase.txt \
    ccd_postprocess.enabled=true runtime.dtype=float16
```

## Inference

```bash
# A single trajectory
python main.py infer.ckpt=checkpoints/ct-no-ccd-preview.pt infer.sim=sims_garment_anim_all/sim_00005
```

Each trajectory is rolled out autoregressively for all 240 frames, starting from the ground-truth
first frame and driven by the collider motion. The rollouts are written as frame-aligned `.obj`
sequences of the prediction, the ground truth and the collider:

```
outputs/rollouts/<category>/<sim_id>/
├── pred/frame_0000.obj ... frame_0239.obj    # (pred_ccd_post/ with CCD post-processing)
├── gt/frame_0000.obj   ... frame_0239.obj
└── collision/frame_0000.obj ...
```

## Training

All settings are in [`configs/default.yaml`](configs/default.yaml). Any field can be overridden on
the command line, for example `training.batch_size=2`.

```bash
# Single GPU
python main.py runtime.run_name=my_run

# All GPUs of one machine
bash scripts/run.sh runtime.run_name=my_run

# Slurm (edit the #SBATCH lines for your cluster)
sbatch scripts/slurm_train.sbatch
```

Training proceeds in two stages:

1. **Pretraining** with the reconstruction and contact losses, using the default config.
2. **Finetuning** from the pretrained weights with the differentiable CCD self-collision loss:

   ```bash
   bash scripts/run.sh runtime.run_name=my_run_ccd \
       training.init_ckpt=outputs/my_run/checkpoints/step_0200000.pt \
       loss.ccd.enabled=true training.lr=1e-6 training.n_train_steps=40000
   ```

Each run writes `config.yaml`, `split.json`, TensorBoard logs (`tb/`) and checkpoints
(`checkpoints/step_XXXXXXX.pt`) to `outputs/<run_name>/`. Every `training.val_every` steps, the
trainer computes the validation loss on `training.val_max_batches` evenly spaced batches per process;
set it to `null` to use the whole validation set. Use `runtime.dry_run=true` to test a setup without
writing any files.

**Custom data.** Put your trajectories in `<dir>/<category>/sim_XXXXX.npz` using the same fields as
the dataset, then set `data.dir=<dir> data.split_file=""`. Every category is then split at random
with the ratio set by `data.val_ratio` and `data.test_ratio`.

## Code Structure

```
main.py                    entry point: training, or inference when infer.ckpt is set
configs/default.yaml       all options and their defaults
splits/ct_split.json       fixed train/val/test split
scripts/                   torchrun / Slurm launchers, dataset download
examples/                  showcase trajectories and their rollout videos
ct/
├── data.py                npz loading, splits, dataset and collate function
├── models/
│   ├── clothtransformer.py  spatial encoder -> temporal Transformer -> spatial decoder
│   ├── spatial.py           spatial encoder/decoder (latent set + mesh GNNs)
│   └── attention.py         Transformer encoder
├── ccd/
│   ├── primitives.py      reference CCD tests (used by the CCD loss)
│   ├── fast.py            batched GPU broad phase and CCD tests
│   ├── loss.py            differentiable CCD loss
│   └── postprocess.py     inference-time CCD post-processing
├── losses.py              reconstruction, contact and self-contact losses
├── rollout.py             autoregressive rollout
├── trainer.py             training loop
└── inference.py           rollouts and .obj export
```

## Citation

```bibtex
@inproceedings{zhang2026clothtransformer,
  title     = {ClothTransformer: Unified Latent-Space Transformers for Scalable Cloth Simulation},
  author    = {Zhang, Yu and Shao, Yidi and Ouyang, Wenqi and Lan, Yushi and Liang, Zhexin and Wu, Chengrui and Xu, Xudong and Pan, Xingang},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2605.27852}
}
```

## License

The code is released under the [MIT License](LICENSE). The dataset is released under
[CC BY 4.0](https://huggingface.co/datasets/YuCrazing/ClothTransformer-dataset).
