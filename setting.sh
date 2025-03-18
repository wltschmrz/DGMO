sudo /opt/deeplearning/install-driver.sh
sudo chmod -R 777 /home/
sudo chmod -R 777 /home/rtrt5060/
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install ffmpeg wget git zip unzip gcc libopenmpi-dev libmpich-dev curl tar libjpeg-dev libpng-dev libgl1-mesa-glx libglib2.0-0 libsndfile1 ninja-build -y
sudo apt-get update
sudo apt-get upgrade -y
pip install -r requirements.txt
# pip uninstall -y torch torchvision torchaudio
# pip cache purge
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# conda create -n mmg python=3.10 -y
# conda init
# conda activate mmg
# conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
# pip install -U pip
# pip install -r requirements.txt
# pip install -U torchvision
# conda install -c conda-forge mpi4py -y
# conda install cuda-cudart cuda-version=12 -y
# conda install -c conda-forge moviepy -y