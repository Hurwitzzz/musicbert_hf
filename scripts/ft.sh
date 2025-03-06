# #!/bin/bash

# python scripts/finetune.py \
#     --data_dir /zhome/stud/zhangya/yyang/datasets_debug/rnbert_h5\
#     --output_dir_base /zhome/stud/zhangya/yyang/output/musicbert_hf/finetune \
#     --targets key_pc_mode quality \
#     --checkpoint_path /home/stud/zhangya/yyang/BERT/ckpts/pretrained/checkpoint_last_musicbert_base.pt \
#     --eval_steps 2500 \
#     --save_steps 2500 \
#     --learning_rate 2.5e-5 \
#     --lr_scheduler_type linear \
#     --batch-size 4 \
#     --fp16 \
#     --freeze_layers 9 \
#     --max_steps 50000 \
#     --warmup_steps 2500 \
#     # --report_to wandb


#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python scripts/finetune.py \
    data_dir=/zhome/stud/zhangya/yyang/datasets/rnbert_h5 \
    output_dir_base=/zhome/stud/zhangya/yyang/output/musicbert_hf/finetune \
    targets="[key_pc_mode, quality, harmony_onset, inversion, primary_alteration_primary_degree_secondary_alteration_secondary_degree]" \
    checkpoint_path=/home/stud/zhangya/yyang/BERT/ckpts/pretrained/checkpoint_last_musicbert_base.pt \
    eval_steps=2500 \
    save_steps=2500 \
    learning_rate=2.5e-5 \
    lr_scheduler_type=linear \
    batch_size=4 \
    fp16=true \
    freeze_layers=9 \
    max_steps=50000 \
    warmup_steps=2500 \
    report_to=wandb
