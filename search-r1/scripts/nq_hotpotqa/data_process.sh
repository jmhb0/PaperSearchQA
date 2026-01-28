# run in the project root
source $(dirname $(dirname $CONDA_EXE))/etc/profile.d/conda.sh
conda activate searchr1

WORK_DIR=.
LOCAL_DIR=$WORK_DIR/data/nq_hotpotqa_train

## process multiple dataset search format train file
DATA=nq,hotpotqa
python $WORK_DIR/scripts/data_process/qa_search_train_merge.py --local_dir $LOCAL_DIR --data_sources $DATA

## process multiple dataset search format test file
DATA=
python $WORK_DIR/scripts/data_process/qa_search_test_merge.py --local_dir $LOCAL_DIR --data_sources $DATA
