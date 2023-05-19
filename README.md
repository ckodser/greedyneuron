# Classification:
python performance.py --batch_size 1024 --mode normal --dataset cifar10 --model_type "LeNET"
python performance.py --batch_size 1024 --mode greedy --dataset cifar10 --model_type "LeNET"
python performance.py --batch_size 1024 --mode normal
python performance.py --batch_size 1024 --mode greedy


# Forgetting:
python performance.py --batch_size 1024 --mode normal --dataset cifar10 --model_type "LeNET"
python performance.py --batch_size 1024 --mode greedy --dataset cifar10 --model_type "LeNET"
python performance.py --batch_size 1024 --mode normal
python performance.py --batch_size 1024 --mode greedy

# Depth

these are hyperparameter tuned runs. You can reproduce hyperparameter tuning similarly. 

python performance.py --batch_size 1024 --learning_rate 0.25 --mode normal --epochs 50 --model_layers 2000
python performance.py --batch_size 1024 --learning_rate 0.05 --mode greedy --epochs 50 --model_layers 2000

python performance.py --batch_size 1024 --learning_rate 0.25 --mode normal --epochs 50 --model_layers 2000,2000
python performance.py --batch_size 1024 --learning_rate 0.05 --mode greedy --epochs 50 --model_layers 2000,2000

python performance.py --batch_size 1024 --learning_rate 0.25 --mode normal --epochs 50 --model_layers 2000,2000,2000
python performance.py --batch_size 1024 --learning_rate 0.01 --mode greedy --epochs 50 --model_layers 2000,2000,2000

python performance.py --batch_size 1024 --learning_rate 0.25 --mode normal --epochs 50 --model_layers 2000,2000,2000,2000
python performance.py --batch_size 1024 --learning_rate 0.01 --mode greedy --epochs 50 --model_layers 2000,2000,2000,2000

python performance.py --batch_size 1024 --learning_rate 0.25 --mode normal --epochs 50 --model_layers 2000,2000,2000,2000,2000
python performance.py --batch_size 1024 --learning_rate 0.01 --mode greedy --epochs 50 --model_layers 2000,2000,2000,2000,2000

python performance.py --batch_size 1024 --learning_rate 0.25 --mode normal --epochs 50 --model_layers 2000,2000,2000,2000,2000,2000
python performance.py --batch_size 1024 --learning_rate 0.01 --mode greedy --epochs 50 --model_layers 2000,2000,2000,2000,2000,2000

python performance.py --batch_size 1024 --learning_rate 0.25 --mode normal --epochs 50 --model_layers 2000,2000,2000,2000,2000,2000,2000
python performance.py --batch_size 1024 --learning_rate 0.002 --mode greedy --epochs 50 --model_layers 2000,2000,2000,2000,2000,2000,2000

python performance.py --batch_size 1024 --learning_rate 0.25 --mode normal --epochs 50 --model_layers 2000,2000,2000,2000,2000,2000,2000,2000
python performance.py --batch_size 1024 --learning_rate 0.002 --mode greedy --epochs 50 --model_layers 2000,2000,2000,2000,2000,2000,2000,2000

python performance.py --batch_size 1024 --learning_rate 0.25 --mode normal --epochs 50 --model_layers 2000,2000,2000,2000,2000,2000,2000,2000,2000
python performance.py --batch_size 1024 --learning_rate 0.002 --mode greedy --epochs 50 --model_layers 2000,2000,2000,2000,2000,2000,2000,2000,2000

python performance.py --batch_size 1024 --learning_rate 0.25 --mode normal --epochs 50 --model_layers 2000,2000,2000,2000,2000,2000,2000,2000,2000,2000
python performance.py --batch_size 1024 --learning_rate 0.002 --mode greedy --epochs 50 --model_layers 2000,2000,2000,2000,2000,2000,2000,2000,2000,2000

python performance.py --batch_size 1024 --learning_rate 0.002 --mode greedy --epochs 50 --model_layers 2000,2000,2000,2000,2000,2000,2000,2000,2000,2000,2000

python performance.py --batch_size 1024 --learning_rate 0.002 --mode greedy --epochs 50 --model_layers 2000,2000,2000,2000,2000,2000,2000,2000,2000,2000,2000,2000

# Performance:
python performance.py --mode normal --dataset cifar10 --model_type "LeNET"
python performance.py --mode greedy --dataset cifar10 --model_type "LeNET"
python performance.py --mode normal
python performance.py --mode greedy


