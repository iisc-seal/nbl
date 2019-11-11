echo y|conda create -n nbl python=2.7
source activate nbl
pip install subprocess32 pycparser==2.19 regex sklearn
pip install numpy==1.15.1 tensorflow-gpu==1.9.0 keras==2.2.3

tar xzf data.tar.gz
wget https://sites.google.com/site/csarahul/nbl-dataset.db.tar.gz -P data/
cd data
tar xzf nbl-dataset.db.tar.gz
cd ..
