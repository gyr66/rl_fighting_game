export DISPLAY=:10
python dqn.py --load_model ./checkpoint/dqn/bootstrap \
 --num_env 16 \
 --eval_freq 128 \
 --save_freq 256 \
 --total_timesteps 1000000 \
 --batch_size 64 \
 --save_path ./checkpoint/dqn \
 --log_path ./dqn_sb3_log > dqn_train_log.txt 2> dqn_train_error_log.txt