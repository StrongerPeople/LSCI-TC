import os
import random
import numpy as np
import argparse
import time
import datetime
import torch
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import utils
from dataset import create_dataset, create_loader, dataset_collate
from models.model_retrieval import CLIPFusionModule
from ruamel.yaml import YAML
from models.open_clip import tokenizer
import datetime
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler 
from models.loss import Weight_soft_CEloss
from utils.eval_utils import evaluate_dataset,evaluate_dataset_ECE_error
scaler = GradScaler()
now = datetime.datetime.now()
time_dir = now.strftime("%Y-%m-%d_%H-%M-%S")
filename = now.strftime("%Y-%m-%d_%H-%M-%S-log.txt")

@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config, k=40):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'
    print('Computing features for evaluation...')
    start_time = time.time()
    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = config['batch_size_test_text']
    image_feas = []
    text_feas = []
    local_images = []
    texts_ids = []
    all_ = []
    print('_________________{}__________________'.format(len(data_loader)))
    for index, batch in enumerate(metric_logger.log_every(data_loader, 100, header)):
        torch.cuda.empty_cache()
        image, _ = batch
        image = image.to(device)
        t1 = time.time()
        image_fea = model.encode_image(image)
        image_feas.append(image_fea)
        local_image = model.encode_image(image, embeds=True)
        local_images.append(local_image)
        t2 = time.time()
        all_.append(t2 - t1)
    print("infer image time:{:.2f}".format(np.average(all_)))
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i + text_bs)]
        text_input = tokenizer.tokenize(text).to(device)
        text_fea = model.encode_text(text_input)
        text_feas.append(text_fea)
        texts_ids.append(text_input)
    
    image_feas = torch.cat(image_feas, dim=0).to(device)
    text_feas = torch.cat(text_feas, dim=0).to(device)
    texts_ids = torch.cat(texts_ids, dim=0).to(device)
    image_features = image_feas / image_feas.norm(dim=-1, keepdim=True)  
    text_features = text_feas / text_feas.norm(dim=-1, keepdim=True)
    sims_matrix = model.clip.logit_scale.exp() * image_features @ text_features.t() + model.clip.logit_bias
    
    score_matrix_i2t = sims_matrix.clone()
    score_matrix_t2i = sims_matrix.clone().t()
    # score_matrix_i2t = F.normalize(score_matrix_i2t, dim=1)
    # score_matrix_t2i = F.normalize(score_matrix_t2i, dim=1)
    # re-ranking
    local_image_feas = torch.cat(local_images, dim=0).to(device)
    image_to_text_mapping = model.get_image_to_text_mapping(image_features, text_features, k)
    text_to_image_mapping = model.get_text_to_image_mapping(text_features, image_features, k)
    for i, img_local in enumerate(local_image_feas):
        topk_text_idx = image_to_text_mapping[i]
        topk_text_ids = texts_ids[topk_text_idx]
        img_local_expanded = img_local.unsqueeze(0).repeat(k, 1, 1)
        match_prob = model.encode_weight_image(topk_text_ids, img_local_expanded)
        score_matrix_i2t[i, topk_text_idx] += match_prob
    for i, txt_local in enumerate(texts_ids):
        topk_image_idx = text_to_image_mapping[i]
        topk_img_fea = local_image_feas[topk_image_idx]
        txt_local_expanded = txt_local.repeat(k, 1)
        match_prob = model.encode_weight_image(txt_local_expanded, topk_img_fea)
        score_matrix_t2i[i, topk_image_idx] += match_prob
    score_matrix_i2t = F.normalize(score_matrix_i2t, dim=1)
    score_matrix_t2i = F.normalize(score_matrix_t2i, dim=1)
    if args.distributed:
        dist.barrier()   
        score_matrix_t2i = score_matrix_t2i.contiguous()
        score_matrix_i2t = score_matrix_i2t.contiguous()
        torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM) 
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)
    
    if utils.is_main_process():
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Evaluation time {}'.format(total_time_str))
    
    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy(),  F.normalize(sims_matrix,dim=1).cpu().numpy(), F.normalize(sims_matrix.t(),dim=1).cpu().numpy()


@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    # Images->Text
    ranks = np.zeros(scores_i2t.shape[0])

    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    
    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])

    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2
    

    eval_result = {'txt_r1': round(tr1, 8),
                   'txt_r5': round(tr5, 8),
                   'txt_r10': round(tr10, 8),
                   'img_r1': round(ir1, 8),
                   'img_r5': round(ir5, 8),
                   'img_r10': round(ir10, 8),
                   'r_mean': round(r_mean, 8)}
    
    return eval_result


def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    print("Creating model", flush=True)

    model = CLIPFusionModule(config=config)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint
    msg = model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    preprocess_train,preprocess_val = model.preprocess_train, model.preprocess_val
    print("Creating retrieval dataset", flush=True)
    _, _, test_dataset = create_dataset('re', config, args.evaluate, preprocess_train, preprocess_val)
    start_time = time.time()
    print("Start evaluating", flush=True)
    test_loader = create_loader([test_dataset], [None],
                                batch_size=[64],
                                num_workers=[0],
                                is_trains=[False],
                                collate_fns=[dataset_collate])[0]
    # test
    cross_score_test_i2t,cross_score_test_t2i,dual_score_test_i2t,dual_score_test_t2i = evaluation(model, test_loader, tokenizer, device, config,k=40)
    # Retrieval Metrics
    if utils.is_main_process():
        test_result = itm_eval(cross_score_test_i2t, cross_score_test_t2i, test_loader.dataset.txt2img, test_loader.dataset.img2txt)
        print("### cross test", test_result)
        test_result = itm_eval(dual_score_test_i2t, dual_score_test_t2i, test_loader.dataset.txt2img, test_loader.dataset.img2txt)
        print("###dual test", test_result)
    # Evaluation of the Dual-Tower Phase: Calibration Error Metrics
    image_text_ece, image_text_bin_dict,text_image_ece,text_image_bin_dict,image_text_meancalibration_gap,text_image_meancalibration_gap = evaluate_dataset_ECE_error(dual_score_test_i2t,dual_score_test_t2i,test_loader.dataset.img2txt,test_loader.dataset.txt2img,num_bins=config['num_bins'])
    mean_adaece = (image_text_ece.item() + text_image_ece.item()) / 2
    calibration_dict = {
        'mean_ece': mean_adaece,
        'image_text_meancalibration_gap': image_text_meancalibration_gap,
        'text_image_meancalibration_gap': text_image_meancalibration_gap,
        'image_text_ece': image_text_ece.item(),
        'text_image_ece': text_image_ece.item(),
    }
    print('###  Dual-Tower Calibration Evaluation ', calibration_dict)
    # Evaluation of the Cross-Tower Phase: Calibration Error Metrics
    image_text_ece, image_text_bin_dict,text_image_ece,text_image_bin_dict,image_text_meancalibration_gap,text_image_meancalibration_gap = evaluate_dataset_ECE_error(cross_score_test_i2t,cross_score_test_t2i,test_loader.dataset.img2txt,test_loader.dataset.txt2img,num_bins=config['num_bins'])
    mean_adaece = (image_text_ece.item() + text_image_ece.item()) / 2
    calibration_dict = {
        'mean_ece': mean_adaece,
        'image_text_meancalibration_gap': image_text_meancalibration_gap,
        'text_image_meancalibration_gap': text_image_meancalibration_gap,
        'image_text_ece': image_text_ece.item(),
        'text_image_ece': text_image_ece.item(),
    }
    print('###  Cross-Tower Calibration Evaluation ', calibration_dict)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('### Time {}'.format(total_time_str))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--bs', default=-1, type=int, help="for each gpu, batch_size = bs // num_gpus")
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--accumulation_steps', type=int, default=8)
    parser.add_argument("--num_worker", type=int,
                        default=5,
                        help='number of workers')
    parser.add_argument("--batch_size", type=int,
                        default=64,
                        help='batch size')
    parser.add_argument("--k", type=int,
                        default=40,
                        help='top-k value for fused features')

    args = parser.parse_args()
    yaml = YAML()
    print("Initialization successful")
    with open(args.config, 'r') as config_file:
        config = yaml.load(config_file)
    device = torch.device(args.device)
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True
    main(args, config)