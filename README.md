我想参照HAT的文件结构将所有的AQA模型放到同一个文件夹下面。方便日后整理和复用

#### Script

** core-mlp **
python run_net.py --experiment single --approach core-mlp --dataset aqa7-pair --exp_name core-mlp-diving --action_id 1 --gpu 0,1,2,3,4,5,6,7