# source /pasteur/u/jmhb/miniconda3/etc/profile.d/conda.sh
source $(dirname $(dirname $CONDA_EXE))/etc/profile.d/conda.sh
conda activate retriever 

corpus_file=data/pubmed.jsonl
save_dir=data/pubmed_test/
retriever_name=e5 # this is for indexing naming
retriever_model=intfloat/e5-base-v2

# Test with only 100 documents
CUDA_VISIBLE_DEVICES=0 python search_r1/search/index_builder.py \
    --retrieval_method $retriever_name \
    --model_path $retriever_model \
    --corpus_path $corpus_file \
    --save_dir $save_dir \
    --use_fp16 \
    --max_length 512 \
    --batch_size 512 \
    --pooling_method mean \
    --faiss_type Flat \
    --save_embedding \
    --test_100 