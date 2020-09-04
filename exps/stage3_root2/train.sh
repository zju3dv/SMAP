export CUDA_VISIBLE_DEVICES="0"
export PROJECT_HOME='/path/to/SMAP'
export PYTHONPATH=$PYTHONPATH:$PROJECT_HOME
python config.py -log
python -m torch.distributed.launch --nproc_per_node=1 train.py