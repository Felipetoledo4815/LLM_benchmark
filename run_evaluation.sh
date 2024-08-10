conda activate spacellava


##############
### MODE 1 ###
##############

Run llava_1.5
python run.py --model llava_1.5 --log_folder logs/study/ --mode 1 --shot zero --nr_images 100

# Run llava_1.6_mistral
python run.py --model llava_1.6_mistral --log_folder logs/study/ --mode 1 --shot zero --nr_images 100
python run.py --model llava_1.6_mistral --log_folder logs/study/ --mode 1 --shot one --nr_images 100

# Run spacellava
python run.py --model spacellava --log_folder logs/study/ --mode 1 --shot zero --nr_images 100
python run.py --model spacellava --log_folder logs/study/ --mode 1 --shot one --nr_images 100


##############
### MODE 2 ###
##############

# Run llava_1.5
python run.py --model llava_1.5 --log_folder logs/study/ --mode 2 --shot zero --nr_images 100

# Run llava_1.6_mistral
python run.py --model llava_1.6_mistral --log_folder logs/study/ --mode 2 --shot zero --nr_images 100
python run.py --model llava_1.6_mistral --log_folder logs/study/ --mode 2 --shot one --nr_images 100

# Run spacellava
python run.py --model spacellava --log_folder logs/study/ --mode 2 --shot zero --nr_images 100
python run.py --model spacellava --log_folder logs/study/ --mode 2 --shot one --nr_images 100