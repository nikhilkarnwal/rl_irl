
c_t=$(date "+%d_%m_%Y_%H_%M_%S")
# python train_rl.py --gen=ppo \
# --trajs=/media/biswas/D/rl_irl/test_env/CartPole-v1/30_12_2021-17_58_15/trajs  --env=CartPole-v1 --irl=Gail1

# python train_rl.py --gen=ppo \
# --trajs=/media/biswas/D/rl_irl/test_env/CartPole-v1/30_12_2021-17_58_15/trajs  --env=CartPole-v1 --irl=Gail3

# python train_rl.py --gen=ppo \
# --trajs=/media/biswas/D/rl_irl/test_env/CartPole-v1/30_12_2021-17_58_15/trajs  --env=CartPole-v1 --irl=Gail4


# python train_rl.py --gen=sac \
# --resume_model=/media/biswas/D/rl_irl/test_env/Hopper-v3/02_01_2022-20_16_08/model.zip  --env=Hopper-v3 --gen_trajs \
# --config_file=config.yml

# python train_rl.py --gen=sac \
# --resume_model=/media/biswas/D/rl_irl/test_env/Hopper-v3/hopper_v3_irl_fhab/gen_model.pth  --env=Hopper-v3 \
# --config_file=config.yml --save_video --abs --num_trajs=2 --test_model

#Traing GAil on Hopper-v3
# python train_rl.py --gen=sac \
# --trajs=/media/biswas/D/rl_irl/test_env/Hopper-v3/03_01_2022-16_10_09/trajs  --env=Hopper-v3 \
# --config_file=config.yml --irl=Gail1 --abs > "$c_t.txt"

# python train_rl.py --gen=sac \
# --trajs=/media/biswas/D/rl_irl/test_env/Hopper-v3/03_01_2022-16_10_09/trajs  --env=Hopper-v3 \
# --config_file=config.yml --irl=Gail2 --abs

# python train_rl.py --gen=sac \
# --trajs=/media/biswas/D/rl_irl/test_env/Hopper-v3/03_01_2022-16_10_09/trajs  --env=Hopper-v3 \
# --config_file=config.yml --irl=Gail3 --abs

# python train_rl.py --gen=sac \
# --trajs=/media/biswas/D/rl_irl/test_env/Hopper-v3/03_01_2022-16_10_09/trajs  --env=Hopper-v3 \
# --config_file=config.yml --irl=Gail4 --abs

# Test traj
# python train_rl.py --gen=sac \
# --trajs=/media/biswas/D/rl_irl/test_env/Hopper-v3/03_01_2022-16_10_09/trajs  --env=Hopper-v3 \
# --config_file=config.yml --test_trajs

# Train on HalfCheetah-v3
# python train_rl.py --gen=sac \
# --env=HalfCheetah-v3 --gen_trajs --config_file=config.yml --save_video --irl=Gail6 --num_trajs=100 > "$c_t-HalfCheetah.txt"

# python train_rl.py --gen=sac \
# --env=HalfCheetah-v3 --gen_trajs --config_file=config.yml --save_video --abs --num_trajs=2 \
# --resume_model=/media/biswas/D/rl_irl/test_env/HalfCheetah-v3/rl_sac_irl_ts3e6/model.zip --irl=Gail6 --explore > "$c_t.txt"


# python train_rl.py --gen=sac \
# --env=HalfCheetah-v3 --config_file=config.yml \
# --trajs=/media/biswas/D/rl_irl/test_env/HalfCheetah-v3/rl_sac_irl_ts3e6/trajs --irl=Gail6 --explore > "$c_t.txt"

# Walker2d-v3
# python train_rl.py --gen=sac \
# --env=Walker2d-v3 --gen_trajs --config_file=config.yml --save_video --abs --irl=Gail6 --num_trajs=1000 > "$c_t-Walker2d.txt"

# c_t=$(date "+%d_%m_%Y_%H_%M_%S")
# python train_rl.py --gen=sac \
# --env=door-expert-v1 --config_file=config.yml \
# --trajs=/media/biswas/D/d4rl/door-expert-v1/door-expert-v1.hdf5 --irl=Gail7 --explore > "$c_t-adroit.txt"

# c_t=$(date "+%d_%m_%Y_%H_%M_%S")
# python train_rl.py --gen=sac \
# --env=door-expert-v1 --config_file=config.yml \
# --trajs=/media/biswas/D/d4rl/door-expert-v1/door-expert-v1.hdf5 --irl=Gail10 --explore > "$c_t-adroit.txt"

for itr in 1 2
do
    echo "Running $itr Iteration"

    # c_t=$(date "+%d_%m_%Y_%H_%M_%S")
    # python -m cProfile -s cumtime train_rl.py --gen=sac \
    # --env=door-expert-v1 --config_file=sac_config.yml \
    # --trajs=/media/biswas/D/d4rl/door-expert-v1/door-expert-v1.hdf5 --irl=Gail1 --gen_hp="door-expert-v1_1" --sh --explore --spec_norm --policy_kw > "$c_t-adroit.txt"

    # c_t=$(date "+%d_%m_%Y_%H_%M_%S")
    # python -m cProfile -s cumtime train_rl.py --gen=sac \
    # --env=door-expert-v1 --config_file=sac_config.yml \
    # --trajs=/media/biswas/D/d4rl/door-expert-v1/door-expert-v1.hdf5 --irl=Gail1 --gen_hp="door-expert-v1_1" --spec_norm --policy_kw > "$c_t-adroit.txt"
    
    c_t=$(date "+%d_%m_%Y_%H_%M_%S")
    python -m cProfile -s cumtime train_rl.py --gen=sac \
    --env=door-expert-v1 --config_file=sac_config.yml \
    --trajs=/media/biswas/D/d4rl/door-expert-v1/door-expert-v1.hdf5 --irl=Gail1_1 --gen_hp="door-expert-v1_2" --spec_norm --policy_kw > "$c_t-adroit.txt"

    # c_t=$(date "+%d_%m_%Y_%H_%M_%S")
    # python -m cProfile -s cumtime train_rl.py --gen=sac \
    # --env=door-expert-v1 --config_file=sac_config.yml \
    # --trajs=/media/biswas/D/d4rl/door-expert-v1/door-expert-v1.hdf5 --irl=Gail1 --gen_hp="door-expert-v1_1" --explore --spec_norm --policy_kw > "$c_t-adroit.txt"
    
    # c_t=$(date "+%d_%m_%Y_%H_%M_%S")
    # python train_rl.py --gen=sac \
    # --env=door-expert-v1 --config_file=config.yml \
    # --trajs=/media/biswas/D/d4rl/door-expert-v1/door-expert-v1.hdf5 --irl=Gail6 --sh --explore> "$c_t-adroit.txt"

    # c_t=$(date "+%d_%m_%Y_%H_%M_%S")
    # python train_rl.py --gen=sac \
    # --env=door-expert-v1 --config_file=config.yml \
    # --trajs=/media/biswas/D/d4rl/door-expert-v1/door-expert-v1.hdf5 --irl=Gail7 --sh --explore> "$c_t-adroit.txt"

    # c_t=$(date "+%d_%m_%Y_%H_%M_%S")
    # python train_rl.py --gen=sac \
    # --env=door-expert-v1 --config_file=sac_config.yml --gen_trajs --num_trajs=1000 --gen_hp="door-expert-v1" --explore \
    # --save_video --reward_file="/media/biswas/D/rl_irl/test_env/adroit/vae/logs/MLPVAE/version_19/checkpoints/epoch=13-step=10892.ckpt" > "$c_t-adroit.txt"
    
    # c_t=$(date "+%d_%m_%Y_%H_%M_%S")
    # python train_rl.py --gen=ppo \
    # --env=CartPole-v1 --config_file=ppo_config.yml --gen_trajs --num_trajs=1000 --gen_hp="CartPole-v1" --explore \
    # --save_video --reward_file="/media/biswas/D/rl_irl/test_env/cartpole/vae/logs/MLPVAE/version_7/checkpoints/epoch=24-step=1000.ckpt" > "$c_t-adroit.txt"

    # for gid in 1 2 3 4 5 6
    # do
    #     for pid in 1 2 3
    #     do
    #         c_t=$(date "+%d_%m_%Y_%H_%M_%S")
    #         python train_rl.py --gen=ppo \
    #         --env=door-expert-v1 --config_file=ppo_config.yml \
    #         --trajs=/media/biswas/D/d4rl/door-expert-v1/door-expert-v1.hdf5 --irl="Gail$gid" --gen_hp="ppo$pid" --sh --explore > "$c_t-adroit.txt" &
    #     done
    #     wait
    #     for pid in 4 5
    #     do
    #         c_t=$(date "+%d_%m_%Y_%H_%M_%S")
    #         python train_rl.py --gen=ppo \
    #         --env=door-expert-v1 --config_file=ppo_config.yml \
    #         --trajs=/media/biswas/D/d4rl/door-expert-v1/door-expert-v1.hdf5 --irl="Gail$gid" --gen_hp="ppo$pid" --sh --explore > "$c_t-adroit.txt" &
    #     done
    #     wait
    #     echo "Done $itr - $gid for all pid"
    # done
done



# c_t=$(date "+%d_%m_%Y_%H_%M_%S")
# python train_rl.py --gen=ppo \
# --env=door-expert-v1 --config_file=ppo_config.yml \
# --trajs=/media/biswas/D/d4rl/door-expert-v1/door-expert-v1.hdf5 --irl=Gail10 --explore > "$c_t-adroit.txt"

# c_t=$(date "+%d_%m_%Y_%H_%M_%S")
# python train_rl.py --gen=sac --env=door-expert-v1 --config_file=config.yml \
# --save_video --num_trajs=1000 --gen_trajs --explore> "$c_t-adroit.txt"


# c_t=$(date "+%d_%m_%Y_%H_%M_%S")
# python train_rl.py --gen=ppo --env=door-expert-v1 --config_file=ppo_config.yml \
# --save_video --num_trajs=1000 --gen_trajs > "$c_t-adroit.txt"
