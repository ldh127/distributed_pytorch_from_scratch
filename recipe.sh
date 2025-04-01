#!/bin/bash

txt_data_path="pg100.txt"
json_data_path="pg100.json"
token_data_path="${json_data_path%.*}_tokens.json"

vocab=1024
tokenizer_path="./tokenizer/tokenizer.json"
root_ckpt_dir="./.checkpoints"  


step_idx=1
echo -e "Step ${step_idx}: Downloading txt data...\n"
if [ ! -f "${txt_data_path}" ]; then
    echo "Downloading ${txt_data_path}..."
    corpus_url="https://www.gutenberg.org/cache/epub/100/pg100.txt"
    wget "${corpus_url}"
else
    echo -e "txt data path exisits, skip downloading\n"
fi


step_idx=$((step_idx+1))
echo -e "Step ${step_idx}: Preprocessing data...\n"
if [ ! -f "${json_data_path}" ]; then
    echo "Preprocessing ${txt_data_path}..."
    python3 ./preprocess_data.py "${txt_data_path}" "${json_data_path}"
else
    echo -e "json data path exisits, skip preprocessing\n"
fi


step_idx=$((step_idx+1))
echo -e "Step ${step_idx}: Train tokenizers...\n"
if [ ! -f "${tokenizer_path}" ]; then
    echo "Training tokenizers..."
    python3 ./train_tokenizer.py --data_path "${json_data_path}" --vocab_size "${vocab}" --output_path "${tokenizer_path}"
else
    echo -e "tokenizers exisits, skip training\n"
fi

step_idx=$((step_idx+1))
echo -e "Step ${step_idx}: Tokenizing data...\n"
if [ ! -f "${token_data_path}" ]; then
    echo "Tokenizing ${json_data_path}..."
    python3 ./pre_tokenize.py -i "${json_data_path}" -o "${token_data_path}" -t "${tokenizer_path}"
else
    echo -e "token data path ${token_data_path} exists, skip tokenizing\n"
fi

exit 0


step_idx=$((step_idx+1))
echo -e "Step ${step_idx}: Train Vallina model...\n"
ckpt_dir="${root_ckpt_dir}/vallina"
if [ ! -d "${ckpt_dir}" ]; then
    echo "Training Vallina model..."
    CUDA_VISIBLE_DEVICES='1' python3 ./train.py --tp_size 1 --save_dir "${ckpt_dir}" --data_path "${token_data_path}" \
     --use_vallina_impl --master_port 23330 --bf16
else
    echo -e "Vallina model ckpt exisits, skip training\n"
fi


step_idx=$((step_idx+1))
echo -e "Step ${step_idx}: Test Vallina model...\n"
ckpt_dir="${root_ckpt_dir}/vallina"
save_dir="${ckpt_dir}/val"
if [ ! -d "${save_dir}" ]; then
    echo "Testing Vallina model..."
    CUDA_VISIBLE_DEVICES='3' python3 ./test.py --tp_size 1 --use_vallina_impl --ckpt_dir "${ckpt_dir}" \
     --data_path "${token_data_path}" --tokenizer_path "${tokenizer_path}"
else
    echo -e "Vallina model test result exisits, skip testing\n"
fi

# step_idx=$((step_idx+1))
# echo -e "Step ${step_idx}: Train TP-1 model...\n"
# ckpt_dir="${root_ckpt_dir}/tp1"
# if [ ! -d "${ckpt_dir}" ]; then
#     echo "Training TP-1 model..."
#     CUDA_VISIBLE_DEVICES='1' python3 ./train.py --tp_size 1 --save_dir "${ckpt_dir}" \
#      --data_path "${json_data_path}" --tokenizer_path "${tokenizer_path}" \
#      --master_port 23331
# else
#     echo -e "TP-1 model ckpt exisits, skip training\n"
# fi

# step_idx=$((step_idx+1))
# echo -e "Step ${step_idx}: Train TP-2 model...\n"
# ckpt_dir="${root_ckpt_dir}/tp2"
# if [ ! -d "${ckpt_dir}" ]; then
#     echo "Training TP-2 model..."
#     CUDA_VISIBLE_DEVICES='5,6' python3 ./train.py --tp_size 2 --save_dir "${ckpt_dir}" \
#      --data_path "${json_data_path}" --tokenizer_path "${tokenizer_path}" \
#      --master_port 23334
# else
#     echo -e "TP-2 model ckpt exisits, skip training\n"
# fi

# step_idx=$((step_idx+1))
# echo -e "Step ${step_idx}: Train TP-4 model...\n"
# ckpt_dir="${root_ckpt_dir}/tp4"
# if [ ! -d "${ckpt_dir}" ]; then
#     echo "Training TP-4 model..."
#     CUDA_VISIBLE_DEVICES='0,1,3,4' python3 ./train.py --tp_size 4 --save_dir "${ckpt_dir}" \
#      --data_path "${json_data_path}" --tokenizer_path "${tokenizer_path}" \
#      --master_port 23335
# else
#     echo -e "TP-4 model ckpt exisits, skip training\n"
# fi
