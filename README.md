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
python performance.py --batch_size 1024 --mode normal --dataset cifar10 --model_type "LeNET" --epochs 50
python performance.py --batch_size 1024 --mode greedy --dataset cifar10 --model_type "LeNET" --epochs 50
python performance.py --batch_size 1024 --mode normal --epochs 50
python performance.py --batch_size 1024 --mode greedy --epochs 50


# Performance:
python performance.py --mode normal --dataset cifar10 --model_type "LeNET"
python performance.py --mode greedy --dataset cifar10 --model_type "LeNET"
python performance.py --mode normal
python performance.py --mode greedy


