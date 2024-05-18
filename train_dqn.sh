export DISPLAY=:10
python dqn.py --load_model ./checkpoint/dqn_v2/model_8192 \
 --num_env 8 \
 --eval_freq 128 \
 --save_freq 512 \
 --total_timesteps 1000000 \
 --batch_size 64 \
 --save_path ./checkpoint/dqn_v2 \
 --log_path ./dqn_sb3_log_v2 > dqn_train_log_v2.txt 2> dqn_train_error_log_v2.txt