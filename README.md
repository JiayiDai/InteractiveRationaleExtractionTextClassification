# InteractiveRationaleExtractionForTextClassification
Imitating human interaction to improve the performance of selective rationale extraction models 

This repository covers the implementation for the papers to appear at TSRML@NeurIPS 2022 (https://tsrml2022.github.io) and ALTA 2022 (https://aclanthology.org/venues/alta/) by Jiayi Dai, Mi-Young Kim, Randy Goebel.

The base models' implementation is modifiled from https://github.com/yala/text_nn. Please follow the installation instructions from that repository.

Example run (with IMDB movie reviews):

Train the first base model:

scripts/main.py --batch_size 128 --cuda --dataset imdb --embedding glove --dropout 0.5 --weight_decay 5e-06\
--num_layers 1 --model_form cnn --hidden_dim 100 --epochs 20 --init_lr 0.001 --num_workers 0 --objective\
cross_entropy --patience 5 --save_dir snapshot --train --test --results_path logs/t1.results\
--gumbel_decay 1e-5 --get_rationales --selection_lambda 0.001 --continuity_lambda 0 --rand_seed 2022

Train the second base model:

scripts/main.py --batch_size 128 --cuda --dataset imdb --embedding glove --dropout 0.5 --weight_decay 5e-06\
--num_layers 1 --model_form cnn --hidden_dim 120 --epochs 20 --init_lr 0.001 --num_workers 0 --objective\
cross_entropy --patience 5 --save_dir snapshot --train --test --results_path logs/t2.results\
--gumbel_decay 1e-5 --get_rationales --selection_lambda 0.001 --continuity_lambda 0 --rand_seed 2022

Interaction:

scripts/main_combine.py --cuda --dataset imdb --test --num_workers 0 --get_rationales --init_lr 0.001\
--selection_lambda 0.001 --continuity_lambda 0 --rand_seed 2022
