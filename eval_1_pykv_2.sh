
python infer_loogle_score.py \
 --model_path /mnt/data/zhiyuanhu/yuliang/easycontext_modify/EasyContext/output/qwen2_7b-RSS_S-80-38400-0810/stage_2 \
 --tag qwen2_7b_LR_128_38_redo \
 --ability LooGLE \
 --eval_times 3

python infer_loogle_score.py \
 --model_path /mnt/data/zhiyuanhu/yuliang/easycontext_modify/EasyContext/output/qwen2_7b-RSS_S-80-24000-0813_feature/stage_2 \
 --ability LooGLE \
 --tag qwen2_7b_LR_80_24_redo \
 --eval_times 3

python infer_loogle_score.py \
 --model_path /mnt/data/zhiyuanhu/yuliang/easycontext_modify/EasyContext/output/qwen2_7b-R-80-38400-0813/stage_2 \
 --ability LooGLE \
 --tag qwen2_7b_R_128_38_redo \
 --eval_times 3

python infer_loogle_score.py \
 --model_path /mnt/data/zhiyuanhu/yuliang/easycontext_modify/EasyContext/output/qwen2_7b-R-80-24000-0813/stage_2 \
 --ability LooGLE \
 --tag qwen2_7b_R_80_24_redo \
 --eval_times 3

python infer_loogle_score.py \
 --model_path /mnt/data/zhiyuanhu/yuliang/easycontext_modify/EasyContext/output/qwen2_7b-pose-80-38400-0810/stage_2 \
 --ability LooGLE \
 --tag qwen2_7b_pose_128_38_redo \
 --eval_times 3

python infer_loogle_score.py \
 --model_path /mnt/data/zhiyuanhu/yuliang/easycontext_modify/EasyContext/output/qwen2_7b-pose-80-24000-0813/stage_2 \
 --ability LooGLE \
 --tag qwen2_7b_pose_80_24_redo \
 --eval_times 3

python infer_loogle_score.py \
 --model_path /mnt/data/zhiyuanhu/yuliang/models/qwen15_7b_llamafy \
 --ability LooGLE \
 --tag qwen15_7b_base \
 --eval_times 3

python infer_loogle_score.py \
 --model_path /mnt/data/zhiyuanhu/yuliang/models/qwen2_7b_llamafy \
 --ability LooGLE \
 --tag qwen2_7b_base \
 --eval_times 3
