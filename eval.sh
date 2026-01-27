CHECKPOINT_PATH="./checkpoint_last.pth" # Path to the trained model checkpoint RS5M_Pretrain.pth
CONFIG_PATH="./configs/Retrieval_rsitmd_vit.yaml"                                                                                    

DEVICE="cuda"                                                 
SEED=0                                                        
BATCH_SIZE=64                                                 
NUM_WORKER=5                                                  

python evaluate_retrieval.py \
    --checkpoint ${CHECKPOINT_PATH} \
    --config ${CONFIG_PATH} \
    --device ${DEVICE} \
    --seed ${SEED} \
    --num_worker ${NUM_WORKER} \
    --batch_size ${BATCH_SIZE} 