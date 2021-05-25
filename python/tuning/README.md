Tuning was performed by runing the following commands:

`python train.py -optimize --n-trials 25 --env fishing-v1 -n 300000 --env-kwargs sigma:0.1 --algo ppo --study-name ppo_sigma_0.1 --seed 8970231`

`python train.py -optimize --n-trials 25 --env fishing-v1 -n 300000 --env-kwargs sigma:0.1 --algo a2c --study-name a2c_sigma_0.1 --seed 8970231`

`python train.py -optimize --n-trials 25 --env conservation-v6 -n 300000 --algo ppo --study-name conservation_v6_ppo --seed 8970231`

`python train.py -optimize --n-trials 25 --env conservation-v6 -n 300000 --algo a2c --study-name conservation_v6_a2c --seed 8970231`
