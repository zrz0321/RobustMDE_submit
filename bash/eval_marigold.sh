perturb_list=("color_shift" "geometric" "motion_blur" "banding")
model_name="marigold"
root_dir="/path/to/experiment_results/marigold/nyu_depth_test_full"
optimize_list=("original_direct" "pareto_direct" "direct" "dual_annealing")
dataset_list=("nyud" "kitti-eigen")
save_dir="./experiment_results/test/"
for dataset in ${dataset_list[@]}; do
    for perturb_type in ${perturb_list[@]}; do
        for optimize_method in ${optimize_list[@]}; do
            python3 -m src.scripts.marigold_eval \
                --model_name ${model_name} \
                --root_dir ${root_dir} \
                --dataset ${dataset} \
                --perturb_type ${perturb_type} \
                --optimize_method ${optimize_method} \
                --save_dir ${save_dir}
        done
    done
done