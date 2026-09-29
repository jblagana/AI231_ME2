#!/bin/bash
# v1g (GPU 1): 30 epochs, class-weighted CE, warm-start v1f, REAL MUSAN noise
# Change vs v1f: model.py pool AdaptiveAvgPool2d -> AdaptiveMaxPool2d
# (verb frames win instead of being averaged out). Same data as v1f.
cd /mnt/jfs_hpc/home/jan.rhey.lagana/vcm
P=/mnt/jfs_hpc/home/jan.rhey.lagana/.conda/envs/vcm/bin/python
CUDA_VISIBLE_DEVICES=1 $P src/train.py \
  --data data/raw_v1f \
  --epochs 30 --batch 32 \
  --class-weights \
  --real-noise data/noise16k \
  --init-from runs/v1f/vcm_v1.pt \
  --out runs/v1g \
  > train_v1g.log 2>&1
echo "=== v1g done ===" >> train_v1g.log
