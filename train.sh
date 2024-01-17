#!/bin/bash
#SBATCH --job-name allon                # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 5-0:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --mail-user 956347073@qq.com       # 这些程序开始、结束、异常突出的时候都发邮件告诉我
#SBATCH --mail-type ALL 
##SBATCH --nodelist air-node-02                            

# 任务 ID 通过 SLURM_ARRAY_TASK_ID 环境变量访问
# 上述行指定参数将传递给 sbatch 作为命令行参数
# 中间不可以有非 #SBATCH 开头的行

# 执行 sbatch 命令前先通过 conda activate [env_name] 进入环境

python train.py --cuda --pretrained ${SLURM_ARRAY_TASK_ID}