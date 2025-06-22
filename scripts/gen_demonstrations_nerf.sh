# this script generate demonstrations for a given task, for training.
# example:
#       bash scripts/gen_demonstrations.sh coordinated_push_box
task=${1}
printf 'task = %s\n' "$task"

cd third_party/RLBench/tools

# # evaluation data
# xvfb-run -a python dataset_generator_bimanual.py --tasks=${task} \
#                             --save_path="../../../data2/test_data"  \
#                             --image_size=256x256 \
#                             --episodes_per_task=100 #\
#                             # --all_variations=True   # default is True
#                             # --processes=1 \
#                             # --renderer=opengl \

# training data
xvfb-run -a python nerf_dataset_generator_bimanual.py --tasks=${task} \
                            --save_path="../../../data_nerf/train_data" \
                            --image_size=256x256 \
                            --episodes_per_task=100  
                            # --all_variations=True
                            # --processes=1 \      
                            # --renderer=opengl \                     


cd ..