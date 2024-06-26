python main.py --task asqp \
            --dataset Camera-COQE \
            --model_name_or_path t5-small \
            --n_gpu 0 \
            --do_train \
            --do_direct_eval \
            --do_inference \
            --train_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 16 \
            --learning_rate 3e-4 \
            --num_train_epochs 10 \
            --num_beams 5 \
            --weight_decay 0.0 \
            --seed 123 \
            --cont_loss 0.05 \
            --cont_temp 0.25 