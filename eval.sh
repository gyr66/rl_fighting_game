export DISPLAY=:10
python eval.py --checkpoint ./checkpoint/dqn/model_471040 --num_episode 100 > eval_log.txt 2> eval_error_log.txt