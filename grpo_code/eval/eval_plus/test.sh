mkdir -p results/humaneval
# export HF_ENDPOINT=https://hf-mirror.com
export PATH=./vllm/bin:$PATH
export PYTHONPATH=$PYTHONPATH:./eval_plus/evalplus
MODEL_DIR=${1}
# uv pip install datamodel_code_generator anthropic mistralai google-generativeai


MODEL_DIR=${MODEL_DIR:-"/path/to/pretrained_models/"}
TP=${2}
TP=${TP:-1}
OUTPUT_DIR=${3}
OUTPUT_DIR=${OUTPUT_DIR:-"/path/to/"}
mkdir -p ${OUTPUT_DIR}

echo "EvalPlus: ${MODEL_DIR}, OUTPUT_DIR ${OUTPUT_DIR}"

python generate.py \
  --model_type qwen2 \
  --model_size chat \
  --model_path ${MODEL_DIR} \
  --bs 1 \
  --temperature 0 \
  --n_samples 1 \
  --greedy \
  --root ${OUTPUT_DIR} \
  --dataset humaneval \
  --tensor-parallel-size ${TP}

echo "Generated samples: ${OUTPUT_DIR}/humaneval/qwen2_chat_temp_0.0"

echo "Sanitizing samples"
python -m evalplus.sanitize --samples ${OUTPUT_DIR}/humaneval/qwen2_chat_temp_0.0

echo "Evaluating humaneval raw"
evalplus.evaluate \
  --dataset humaneval \
  --samples ${OUTPUT_DIR}/humaneval/qwen2_chat_temp_0.0 > ${OUTPUT_DIR}/raw_humaneval_results.txt
echo "Finished evaluating humaneval file at ${OUTPUT_DIR}/raw_humaneval_results.txt"
echo "Evaluating humaneval sanitized"
evalplus.evaluate \
  --dataset humaneval \
  --samples ${OUTPUT_DIR}/humaneval/qwen2_chat_temp_0.0-sanitized > ${OUTPUT_DIR}/humaneval_results.txt
echo "Finished evaluating humaneval file at ${OUTPUT_DIR}/humaneval_results.txt"

python generate.py \
  --model_type qwen2 \
  --model_size chat \
  --model_path ${MODEL_DIR} \
  --bs 1 \
  --temperature 0 \
  --n_samples 1 \
  --greedy \
  --root ${OUTPUT_DIR} \
  --dataset mbpp \
  --tensor-parallel-size ${TP}

echo "Sanitizing mbpp"
python -m evalplus.sanitize --samples ${OUTPUT_DIR}/mbpp/qwen2_chat_temp_0.0

echo "Evaluating mbpp raw"
evalplus.evaluate \
  --dataset mbpp \
  --samples ${OUTPUT_DIR}/mbpp/qwen2_chat_temp_0.0 > ${OUTPUT_DIR}/raw_mbpp_results.txt
echo "Finished evaluating mbpp file at ${OUTPUT_DIR}/raw_mbpp_results.txt"
echo "Evaluating mbpp sanitized"
evalplus.evaluate \
  --dataset mbpp \
  --samples ${OUTPUT_DIR}/mbpp/qwen2_chat_temp_0.0-sanitized > ${OUTPUT_DIR}/mbpp_results.txt
echo "Finished evaluating mbpp file at ${OUTPUT_DIR}/mbpp_results.txt"