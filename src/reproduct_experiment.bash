#!/usr/bin/bash 
# At the commit 288ea2093cb15237384c8f60cce921bd07f988a3 run this bash file.

nohup python run.py --agt 9 --usr 1 --max_turn 40 --movie_kb_path ./deep_dialog/data/movie_kb.1k.p --dqn_hidden_size 80 --experience_replay_pool_size 1000 --episodes 500 --simulation_epoch_size 100 --write_model_dir ./deep_dialog/checkpoints/rl_agent/clip --run_mode 3 --act_level 0 --slot_err_prob 0.05 --intent_err_prob 0.00 --batch_size 16 --goal_file_path ./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p --warm_start 1 --warm_start_epochs 120 > no_clip_288ea2093.log &
