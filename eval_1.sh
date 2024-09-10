(CUDA_VISIBLE_DEVICES=0 python utils/infer.py \
 --model_path /mnt/data/zhiyuanhu/yuliang/easycontext_modify/EasyContext/output/llama3_8b-full-80-80000-0830_redo/stage_2 \
 --tag llama_3_8b_full_80k_0910_test \
 --ability LongBench \
 --eval_times 1) # &

# (CUDA_VISIBLE_DEVICES=1 python infer_no_longbench_loogle.py \
#  --model_path /mnt/data/zhiyuanhu/yuliang/easycontext_modify/EasyContext/output/llama3_8b-full-80-80000-0830_redo/stage_2 \
#  --tag llama_3_8b_full_80k_0831 \
#  --ability all \
#  --eval_times 1) 
# #  --eval_times 1)


# wait