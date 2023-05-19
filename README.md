# Classification:
python main.py --batch_size 64  --learning_rate 0.05 --mode normal --dataset cifar10 --model_type "LeNET"
python main.py --batch_size 128 --learning_rate 0.01 --mode greedy --dataset cifar10 --model_type "LeNET"
python main.py --batch_size 512 --learning_rate 0.01 --mode normal
python main.py --batch_size 256 --learning_rate 0.01 --mode greedy


# Forgetting:
python forgetting_check_hard.py --batch_size 1024 --learning_rate 0.01 --mode normal
python forgetting_check_hard.py --batch_size 1024 --learning_rate 0.01 --mode greedy

# Depth:

these are hyperparameter tuned runs. You can reproduce hyperparameter tuning similarly. 

python main.py --batch_size 1024 --learning_rate 0.25 --mode normal --num_epochs 50 --model_layers 2000 --seed 1
python main.py --batch_size 1024 --learning_rate 0.05 --mode greedy --num_epochs 50 --model_layers 2000 --seed 1

python main.py --batch_size 1024 --learning_rate 0.25 --mode normal --num_epochs 50 --model_layers 2000,2000 --seed 1
python main.py --batch_size 1024 --learning_rate 0.05 --mode greedy --num_epochs 50 --model_layers 2000,2000 --seed 1

python main.py --batch_size 1024 --learning_rate 0.25 --mode normal --num_epochs 50 --model_layers 2000,2000,2000 --seed 1
python main.py --batch_size 1024 --learning_rate 0.01 --mode greedy --num_epochs 50 --model_layers 2000,2000,2000 --seed 1

python main.py --batch_size 1024 --learning_rate 0.25 --mode normal --num_epochs 50 --model_layers 2000,2000,2000,2000 --seed 1
python main.py --batch_size 1024 --learning_rate 0.01 --mode greedy --num_epochs 50 --model_layers 2000,2000,2000,2000 --seed 1

python main.py --batch_size 1024 --learning_rate 0.25 --mode normal --num_epochs 50 --model_layers 2000,2000,2000,2000,2000 --seed 1
python main.py --batch_size 1024 --learning_rate 0.01 --mode greedy --num_epochs 50 --model_layers 2000,2000,2000,2000,2000 --seed 1

python main.py --batch_size 1024 --learning_rate 0.25 --mode normal --num_epochs 50 --model_layers 2000,2000,2000,2000,2000,2000 --seed 1
python main.py --batch_size 1024 --learning_rate 0.01 --mode greedy --num_epochs 50 --model_layers 2000,2000,2000,2000,2000,2000 --seed 1

python main.py --batch_size 1024 --learning_rate 0.25 --mode normal --num_epochs 50 --model_layers 2000,2000,2000,2000,2000,2000,2000 --seed 1
python main.py --batch_size 1024 --learning_rate 0.002 --mode greedy --num_epochs 50 --model_layers 2000,2000,2000,2000,2000,2000,2000 --seed 1

python main.py --batch_size 1024 --learning_rate 0.25 --mode normal --num_epochs 50 --model_layers 2000,2000,2000,2000,2000,2000,2000,2000 --seed 1
python main.py --batch_size 1024 --learning_rate 0.002 --mode greedy --num_epochs 50 --model_layers 2000,2000,2000,2000,2000,2000,2000,2000 --seed 1

python main.py --batch_size 1024 --learning_rate 0.25 --mode normal --num_epochs 50 --model_layers 2000,2000,2000,2000,2000,2000,2000,2000,2000 --seed 1
python main.py --batch_size 1024 --learning_rate 0.002 --mode greedy --num_epochs 50 --model_layers 2000,2000,2000,2000,2000,2000,2000,2000,2000 --seed 1

python main.py --batch_size 1024 --learning_rate 0.25 --mode normal --num_epochs 50 --model_layers 2000,2000,2000,2000,2000,2000,2000,2000,2000,2000 --seed 1
python main.py --batch_size 1024 --learning_rate 0.002 --mode greedy --num_epochs 50 --model_layers 2000,2000,2000,2000,2000,2000,2000,2000,2000,2000 --seed 1

python main.py --batch_size 1024 --learning_rate 0.002 --mode greedy --num_epochs 50 --model_layers 2000,2000,2000,2000,2000,2000,2000,2000,2000,2000,2000 --seed 1

python main.py --batch_size 1024 --learning_rate 0.002 --mode greedy --num_epochs 50 --model_layers 2000,2000,2000,2000,2000,2000,2000,2000,2000,2000,2000,2000 --seed 1

# Performance:
python performance.py --mode normal --dataset cifar10 --model_type "LeNET"
python performance.py --mode greedy --dataset cifar10 --model_type "LeNET"
python performance.py --mode normal
python performance.py --mode greedy


