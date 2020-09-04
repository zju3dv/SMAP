export PROJECT_HOME='/path/to/SMAP'
export PYTHONPATH=$PYTHONPATH:$PROJECT_HOME
python test.py -p "/path/to/SMAP_model.pth" \
-t run_inference \
-d test \
-rp "/path/to/RefineNet.pth" \
--batch_size 16 \
--do_flip 1 \
--dataset_path "/path/to/custom/image_dir"
