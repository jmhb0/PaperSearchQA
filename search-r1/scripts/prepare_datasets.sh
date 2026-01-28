#DATA_TRAIN="jmhb/PaperSearchRL_v5_gv3_n3000_test300_parav1pcnt50"
#DATA_TRAIN="jmhb/PaperSearchRL_v5_gv3_n20000_test5000_parav1pcnt50"
DATA_TRAIN="jmhb/papersearchr1"
DATA_TEST="$DATA_TRAIN,jmhb/bioasq_factoid"

WORK_DIR=.
LOCAL_DIR=$WORK_DIR/data/

## process multiple dataset search format train file
python $WORK_DIR/scripts/data_process/qa_search_train_merge.py --local_dir $LOCAL_DIR --data_sources $DATA_TRAIN

## process multiple dataset search format test file
python $WORK_DIR/scripts/data_process/qa_search_test_merge.py --local_dir $LOCAL_DIR --data_sources $DATA_TEST
