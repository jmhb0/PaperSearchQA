# source /pasteur/u/jmhb/miniconda3/etc/profile.d/conda.sh
source $(dirname $(dirname $CONDA_EXE))/etc/profile.d/conda.sh
conda activate retriever 

# Dataset configuration
dataset_name=pubmed_restricted_jmhb_bioasq_trainv0_n1609_test100
corpus_file=data/${dataset_name}.jsonl

# Retriever type: set to either "bm25" or "e5"
# retriever_type=bm25
retriever_type=e5

# Configure based on retriever type
if [ "$retriever_type" = "bm25" ]; then
    retriever_name=bm25
    retriever_model='none'
    save_dir=data/${dataset_name}_bm25/
elif [ "$retriever_type" = "e5" ]; then
    retriever_name=e5
    retriever_model=intfloat/e5-base-v2
    save_dir=data/${dataset_name}_e5/
else
    echo "Error: retriever_type must be either 'bm25' or 'e5'"
    exit 1
fi

echo "Using $retriever_type retriever with model: $retriever_model"
echo "Dataset: $dataset_name"
echo "Save directory: $save_dir"

# change faiss_type to HNSW32/64/128 for ANN indexing
# change retriever_name to bm25 for BM25 indexing
CUDA_VISIBLE_DEVICES=0,1,2,3 python search_r1/search/index_builder.py \
    --retrieval_method $retriever_name \
    --model_path $retriever_model \
    --corpus_path $corpus_file \
    --save_dir $save_dir \
    --use_fp16 \
    --max_length 512 \
    --batch_size 512 \
    --pooling_method mean \
    --faiss_type Flat \
    --save_embedding



