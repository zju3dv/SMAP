# export CUDA_VISIBLE_DEVICES="0,1,2,3"
export PROJECT_HOME='/path/to/SMAP'
export PYTHONPATH=$PYTHONPATH:$PROJECT_HOME
python train.py
