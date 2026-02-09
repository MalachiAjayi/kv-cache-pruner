cd evaluation/longbench

MODEL=Llama-3.1-8B-Instruct-128k
TASKS=(narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique trec triviaqa samsum passage_count passage_retrieval_en lcc repobench-p)

SAVEDIR=results
mkdir -p $SAVEDIR

for task in "${TASKS[@]}"; do
    python -u pred.py \
            --model $MODEL --task $task \
            --precision bf16 --save_dir $SAVEDIR \
            --cache_type full \
            --enable_prefill_flash_attn
done

METHODS=(h2o kvcV kvcK kvcVK tova kvcV+tova kvcK+tova kvcVK+tova snapkv kvcV+maxpool kvcK+maxpool kvcVK+maxpool)
CRS=(0.05 0.07 0.1 0.2)

for cr in "${CRS[@]}"; do
    for task in "${TASKS[@]}"; do
        for method in "${METHODS[@]}"; do
            python -u pred.py \
                --model $MODEL --task $task \
                --precision bf16 --save_dir $SAVEDIR \
                --cache_type $method --cache_ratio $cr \
                --no_decode_evict \
                --enable_prefill_flash_attn
        done
    done
done


python eval.py --model $MODEL --save_dir $SAVEDIR
