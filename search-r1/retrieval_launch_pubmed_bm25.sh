corpus_file=data/pubmed.jsonl
# retriever_name=e5
retriever_name=bm25
if [ "$retriever_name" = "e5" ]; then
    index_file=data/pubmed_e5/e5_Flat.index
    retriever_path=intfloat/e5-base-v2
elif [ "$retriever_name" = "bm25" ]; then
    index_file=data/pubmed_bm25/bm25
    retriever_path=none
fi

python search_r1/search/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --faiss_gpu &
