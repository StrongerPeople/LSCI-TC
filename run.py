import os
import sys
import time
import random
import argparse
os.environ['HUGGINGFACE_HUB_MIRROR'] = 'https://hf-mirror.com'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from utils.hdfs_io import HADOOP_BIN, hexists, hmkdir, hcopy


def get_dist_launch(args):
    PYTHON_PATH = "/root/autodl-fs/env/bin/python"
    BASE_CMD_PREFIX = f"{PYTHON_PATH} -W ignore -m torch.distributed.launch"
    NNODES = "--nnodes=1 "

    dist_config = {
        'f4': ('0,1,2,3', 4, 9999, 4),
        'f2': ('0,1', 2, 9999, 2),
        'f3': ('0,1,2', 3, 9999, 3),
        'f12': ('1,2', 2, 9999, 2),
        'f02': ('0,2', 2, 9999, 2),
        'f03': ('0,3', 2, 9999, 2),
        'l2': ('2,3', 2, 9998, 2),
    }

    if args.dist in dist_config:
        devices, world_size, port, nproc = dist_config[args.dist]
        return (
            f"CUDA_VISIBLE_DEVICES={devices} WORLD_SIZE={world_size} {BASE_CMD_PREFIX} "
            f"--master_port {port} --nproc_per_node={nproc} {NNODES}"
        )
    elif args.dist.startswith('gpu'):
        num = int(args.dist[3:])
        assert 0 <= num <= 8
        return (
            f"CUDA_VISIBLE_DEVICES={num} WORLD_SIZE=1 {BASE_CMD_PREFIX} "
            f"--master_port 9995 --nproc_per_node=1 {NNODES}"
        )
    else:
        raise ValueError("Unsupported dist parameter: {}".format(args.dist))


def get_from_hdfs(file_hdfs):
    """
    compatible to HDFS path or local path
    """
    if file_hdfs.startswith('hdfs'):
        file_local = os.path.split(file_hdfs)[-1]

        if os.path.exists(file_local):
            print(f"rm existing {file_local}")
            os.system(f"rm {file_local}")

        hcopy(file_hdfs, file_local)

    else:
        file_local = file_hdfs
        assert os.path.exists(file_local)

    return file_local


def run_retrieval(args):
    dist_launch = get_dist_launch(args)

    os.system(f"{dist_launch} "
              f"--use_env Retrieval.py --config {args.config} "
              f"--output_dir {args.output_dir} --bs {args.bs} --checkpoint {args.checkpoint} {'--evaluate' if args.evaluate else ''}")


def run(args):
    if args.task == 'itr_rsicd_vit':
        run_retrieval(args)

    elif args.task == 'itr_rsitmd_vit':
        run_retrieval(args)
    elif args.task == 'itr_rsitmd_geo':
        run_retrieval(args)
    elif args.task == 'itr_rsicd_geo':
        run_retrieval(args)

    elif args.task == 'itr_coco':
        run_retrieval(args)

    elif args.task == 'itr_nwpu':
        run_retrieval(args)
    else:
        raise NotImplementedError(f"task == {args.task}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='itr_rsicd_vit') # itr_rsitmd_vit itr_rsicd_vit itr_rsitmd_geo itr_rsicd_geo itr_rsitmd_geo itr_rsicd_geo itr_rsitmd_vit itr_rsicd_vit
    parser.add_argument('--dist', type=str, default='gpu0', help="see func get_dist_launch for details")
    parser.add_argument('--config', default='configs/Retrieval_rsicd_vit.yaml', type=str, help="if not given, use default")
    parser.add_argument('--bs', default=-1, type=int, help="for each gpu, batch_size = bs // num_gpus; "
                                                  "this option only works for fine-tuning scripts.")
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--checkpoint', default='-1', type=str, help="for fine-tuning")
    parser.add_argument('--load_ckpt_from', default=' ', type=str, help="load domain pre-trained params")
    # write path: local or HDFS
    parser.add_argument('--output_dir', type=str, default='./outputs/test', help='for fine-tuning, local path; '
                                                                      'for pre-training, local and HDFS are both allowed.')
    parser.add_argument('--evaluate', action='store_true', default=False, help="evaluation on downstream tasks")
    parser.add_argument('--precheckpoint', type=str, required=True)
    args = parser.parse_args()
    run(args)

