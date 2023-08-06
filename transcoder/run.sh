#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8
CURRENT_DIR=$(pwd)
CODE_DIR_HOME=$(realpath ..)

evaluator_script="${CODE_DIR_HOME}/evaluation"
codebleu_path="${CODE_DIR_HOME}/evaluation/CodeBLEU"
prog_test_case_dir="${CODE_DIR_HOME}/test_cases"

SOURCE=${1:-java}
TARGET=${2:-python}

export NGPU=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
echo "Source: $SOURCE Target: $TARGET"
pretrained_model=${CODE_DIR_HOME}/models/transcoder

if [[ $SOURCE == 'java' && $TARGET == 'python' ]]; then
    RELOAD_MODEL_PATH=${pretrained_model}/Online_ST_Java_Python.pth
    MT_STEPS="java-python"
    VALID_METRICS="valid_java-python_mt_bleu"
elif [[ $SOURCE == 'python' && $TARGET == 'java' ]]; then
    RELOAD_MODEL_PATH=${pretrained_model}/Online_ST_Python_Java.pth
    MT_STEPS="python-java"
    VALID_METRICS="valid_python-java_mt_bleu"
fi

EXP_NAME=transcoder-st-ft
EXP_ID="${MT_STEPS//,/_}"
TRAIN_SCRIPT=${CODE_DIR_HOME}/codegen/model/train.py

function train() {
    DATA_SRC=$1
    if [[ $DATA_SRC == 'program' ]]; then
        path_2_data=${CODE_DIR_HOME}/data
    elif [[ $DATA_SRC == 'function' ]]; then
        path_2_data=${CODE_DIR_HOME}/data/parallel_functions
    fi

    SAVE_DIR=${CURRENT_DIR}/${DATA_SRC}/${SOURCE}2${TARGET}
    mkdir -p $SAVE_DIR

    export PYTHONPATH=$CODE_DIR_HOME
    torchrun --nproc_per_node=$NGPU $TRAIN_SCRIPT \
        --exp_name $EXP_NAME \
        --exp_id $EXP_ID \
        --dump_path $SAVE_DIR \
        --data_path $path_2_data/transcoder-bin \
        --split_data_accross_gpu local \
        --max_len 512 \
        --mt_steps $MT_STEPS \
        --encoder_only False \
        --n_layers 0 \
        --n_layers_encoder 6 \
        --n_layers_decoder 6 \
        --emb_dim 1024 \
        --n_heads 8 \
        --lgs 'java-python' \
        --max_vocab 64000 \
        --roberta_mode false \
        --reload_model "$RELOAD_MODEL_PATH,$RELOAD_MODEL_PATH" \
        --lgs_mapping 'java:java_sa,python:python_sa' \
        --amp 2 \
        --fp16 True \
        --batch_size 2 \
        --accumulate_gradients 1 \
        --epoch_size 2000 \
        --max_epoch 20 \
        --optimizer 'adam_inverse_sqrt,warmup_updates=200,lr=0.0001,weight_decay=0.01' \
        --eval_bleu true \
        --eval_bleu_valid_only true \
        --validation_metrics ${VALID_METRICS} \
        --stopping_criterion "${VALID_METRICS},5" \
        2>&1 | tee ${SAVE_DIR}/finetune.log
}

function program_translation_ngram_evaluation() {
    path_2_data=${CODE_DIR_HOME}/data
    BPE_PATH=${CODE_DIR_HOME}/codegen/bpe/cpp-java-python/codes
    INPUT_FILE=${path_2_data}/test.java-python.${SOURCE}
    GROUND_TRUTH_PATH=${path_2_data}/test.jsonl

    SAVE_DIR=${CURRENT_DIR}/program/${SOURCE}2${TARGET}/$EXP_NAME/$EXP_ID
    MODEL_FILENAME=best-valid_${SOURCE}-${TARGET}_mt_bleu.pth
    MODEL_PATH=${SAVE_DIR}/${MODEL_FILENAME}

    FILE_PREF=${SAVE_DIR}/test
    RESULT_FILE=${SAVE_DIR}/ngram_eval.txt

    export PYTHONPATH=$CODE_DIR_HOME
    python translate.py \
        --model_path $MODEL_PATH \
        --src_lang $SOURCE \
        --tgt_lang $TARGET \
        --BPE_path $BPE_PATH \
        --input_file $INPUT_FILE \
        --output_file $FILE_PREF.output \
        --batch_size 2 \
        --beam_size 10

    python $evaluator_script/evaluator.py \
        --references $GROUND_TRUTH_PATH \
        --predictions $FILE_PREF.output \
        --language $TARGET \
        2>&1 | tee $RESULT_FILE

    export PYTHONPATH=$CODE_DIR_HOME
    python $codebleu_path/calc_code_bleu.py \
        --ref $GROUND_TRUTH_PATH \
        --hyp $FILE_PREF.output \
        --lang $TARGET \
        2>&1 | tee -a $RESULT_FILE

    python $evaluator_script/compile.py \
        --input_file $FILE_PREF.output \
        --language $TARGET \
        2>&1 | tee -a $RESULT_FILE

    count=$(ls -1 *.class 2>/dev/null | wc -l)
    [[ $count != 0 ]] && rm *.class

}

function program_translation_exec_evaluation() {
    SAVE_DIR=${CURRENT_DIR}/program/${SOURCE}2${TARGET}/$EXP_NAME/$EXP_ID
    EXEC_DIR=${SAVE_DIR}/executions
    mkdir -p $EXEC_DIR
    RESULT_FILE=$SAVE_DIR/exec_eval.txt

    export PYTHONPATH=$CODE_DIR_HOME
    python $prog_test_case_dir/compute_ca.py \
        --hyp_paths $SAVE_DIR/test.output \
        --ref_path ${CODE_DIR_HOME}/data/test.jsonl \
        --testcases_dir $prog_test_case_dir \
        --outfolder $EXEC_DIR \
        --source_lang $SOURCE \
        --target_lang $TARGET \
        2>&1 | tee $RESULT_FILE
}

function function_translation_ngram_evaluation() {
    path_2_data=${CODE_DIR_HOME}/data/parallel_functions
    BPE_PATH=${CODE_DIR_HOME}/codegen/bpe/cpp-java-python/codes
    INPUT_FILE=${path_2_data}/test.java-python.${SOURCE}
    GROUND_TRUTH_PATH=${path_2_data}/test.java-python.${TARGET}

    SAVE_DIR=${CURRENT_DIR}/function/${SOURCE}2${TARGET}/$EXP_NAME/$EXP_ID
    MODEL_FILENAME=best-valid_${SOURCE}-${TARGET}_mt_bleu.pth
    MODEL_PATH=${SAVE_DIR}/${MODEL_FILENAME}
    FILE_PREF=${SAVE_DIR}/test
    RESULT_FILE=${SAVE_DIR}/ngram_eval.txt

    export PYTHONPATH=$CODE_DIR_HOME
    python translate.py \
        --model_path $MODEL_PATH \
        --src_lang $SOURCE \
        --tgt_lang $TARGET \
        --BPE_path $BPE_PATH \
        --input_file $INPUT_FILE \
        --output_file $FILE_PREF.output \
        --batch_size 2 \
        --beam_size 10

    python $evaluator_script/evaluator.py \
        --references $GROUND_TRUTH_PATH \
        --txt_ref \
        --predictions $FILE_PREF.output \
        --language $TARGET \
        2>&1 | tee $RESULT_FILE

    export PYTHONPATH=$CODE_DIR_HOME
    python $codebleu_path/calc_code_bleu.py \
        --ref $GROUND_TRUTH_PATH \
        --txt_ref \
        --hyp $FILE_PREF.output \
        --lang $TARGET \
        2>&1 | tee -a $RESULT_FILE

}

function function_translation_exec_evaluation() {
    SAVE_DIR=${CURRENT_DIR}/function/${SOURCE}2${TARGET}/$EXP_NAME/$EXP_ID
    DATA_DIR=${CODE_DIR_HOME}/data/transcoder_test_gfg
    EXEC_DIR=${SAVE_DIR}/executions
    mkdir -p $EXEC_DIR
    RESULT_FILE=${SAVE_DIR}/exec_eval.txt

    export PYTHONPATH=$CODE_DIR_HOME
    python $evaluator_script/compute_ca.py \
        --src_path $DATA_DIR/test.java-python.${SOURCE} \
        --ref_path $DATA_DIR/test.java-python.${TARGET} \
        --hyp_paths $SAVE_DIR/test.output \
        --id_path $DATA_DIR/test.java-python.id \
        --split test \
        --outfolder $EXEC_DIR \
        --source_lang $SOURCE \
        --target_lang $TARGET \
        --retry_mismatching_types True \
        2>&1 | tee $RESULT_FILE

    python $evaluator_script/classify_errors.py \
        --logfile $EXEC_DIR/test_${SOURCE}-${TARGET}.log \
        --lang $TARGET \
        2>&1 | tee -a $RESULT_FILE
}

train 'program'
train 'function'

program_translation_ngram_evaluation
program_translation_exec_evaluation

function_translation_ngram_evaluation
function_translation_exec_evaluation
