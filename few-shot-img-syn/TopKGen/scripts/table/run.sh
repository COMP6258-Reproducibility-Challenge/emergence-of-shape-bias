python3 ../../train.py --path ../../../data/table \
--im_size 256 \
--batch_size 8 \
--iter 100000 \
--start_iter 0 \
--name topk \
--nonoise \
--eps_G 1e-4 \
--sparse_hw_info 32_15 \
--sp_hw_policy_name TopKMaskHW \
--fid_feature_extractor_name inception-v3-compat \
--ckpt restart \

