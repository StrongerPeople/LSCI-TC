import argparse
import os
import sys
import math
import numpy as np
import random
import time
import datetime
import json
import torch
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import utils
from PIL import Image
import io
import random
import argparse
import os
import time
from dataset import create_dataset, create_sampler, create_loader,dataset_collate,rs5m_dataset_collate
from scheduler import create_scheduler
from optim import create_optimizer
from models.model_retrieval import CLIPFusionModule,create_and_load_pretrained
from ruamel.yaml import YAML
from models.open_clip import tokenizer
import datetime
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler 
from models.loss import Weight_soft_CEloss
from utils.eval_utils import evaluate_dataset,evaluate_dataset_ECE_error
import torch.multiprocessing as mp
scaler = GradScaler()
now = datetime.datetime.now()
time_dir = now.strftime("%Y-%m-%d_%H-%M-%S")
filename = now.strftime("%Y-%m-%d_%H-%M-%S-log.txt")
            
def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def check_grad(model):
    for name, param in model.named_parameters():
        if param.requires_grad == False:
            print(f'No gradient for {name}, skipping...')


def train(model, data_loader, optimizer, tokenizer,epoch, device, scheduler, config):
    metric_logger = utils.MetricLogger(delimiter="")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.7f}'))
    metric_logger.add_meter('TCloss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    # metric_logger.add_meter('sigmoid_loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 200
    print('_________________{}__________________'.format(len(data_loader)))
    lennum = len(data_loader)
    for i, (image, text) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        with autocast():
            images = image.to(device, non_blocking=True)
            texts = text.to(device, non_blocking=True)
            total_loss = model.module(images, texts,WeightsoftCEloss)
            loss = total_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scale = scaler.get_scale()
        scaler.update()
        skip_lr_sched = (scale > scaler.get_scale())
        if not skip_lr_sched:
            scheduler.step()
        optimizer.zero_grad()
        metric_logger.update(TCloss=loss.item())
        metric_logger.update(lr=scheduler.get_lr()[-1])
    torch.cuda.empty_cache()
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.10f} ".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

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
        del image_fea, local_image, image
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
    score_matrix_t2i = score_matrix_i2t.clone().t()
    local_image_feas = torch.cat(local_images, dim=0).to(device)
    if k != 0:
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
    
    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()


@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    # Images->Text
    ranks = np.zeros(scores_i2t.shape[0])
    total_recall_i2t = 0
    
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

    world_size = utils.get_world_size()
    if args.bs > 0:
        config['batch_size_train'] = args.bs // world_size

    seed = config['seed'] + utils.get_rank()
    # TODO seed everything but still not deterministic(± 1~1.5% difference in results), need to check
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True
    print("Creating model", flush=True)

    model = CLIPFusionModule(config=config)
    checkpoint = torch.load(args.precheckpoint, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint
    msg = model.load_state_dict(state_dict, strict=False)
    print("missing", msg.missing_keys)
    print("good")
    print("unexp", msg.unexpected_keys)
    model = model.to(device)
    model_without_ddp = model
    preprocess_train,preprocess_val = model.preprocess_train, model.preprocess_val
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],find_unused_parameters=True)
        model_without_ddp = model.module
    print("Creating retrieval dataset", flush=True)
    train_dataset, val_dataset, test_dataset = create_dataset('re', config, args.evaluate, preprocess_train, preprocess_val)
    start_time = time.time()
    print("### output_dir, ", args.output_dir, flush=True)
    model.train()
    print("Start training", flush=True)
    print(f"The trainable parameters are {count_trainable_parameters(model)}")
    train_dataset_size = len(train_dataset)
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = [None,None] + create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None] + [None, None] + [None]
    else:
        samplers = [None,None, None, None]

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    samplers =  create_sampler([train_dataset], [True], num_tasks, global_rank) + [None,None, None]
    train_loader, val_loader,test_loader = create_loader([train_dataset ,val_dataset,test_dataset], samplers,
                                                            batch_size=[
                                                             config['batch_size_train']] * 3,
                                                            num_workers=[6, 6, 6],
                                                            is_trains=[True,False,False], 
                                                            collate_fns=[dataset_collate,dataset_collate,dataset_collate])
    
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    arg_sche['step_per_epoch'] = math.ceil(train_dataset_size/(config['batch_size_train']*world_size))
    lr_scheduler = create_scheduler(arg_sche, optimizer)
    max_epoch = config['schedular']['epochs']
    best = 0
    best_epoch = 0
    for epoch in range(0, max_epoch):
        train_stats = train(model, train_loader, optimizer, tokenizer, epoch, device, lr_scheduler, config)
        with torch.no_grad():
            score_val_i2t, score_val_t2i = evaluation(model_without_ddp, val_loader, tokenizer, device, config, k=0)
            image_text_ece, image_text_bin_dict,text_image_ece,text_image_bin_dict,image_text_meancalibration_gap,text_image_meancalibration_gap = evaluate_dataset_ECE_error(score_val_i2t,score_val_t2i,val_loader.dataset.img2txt,val_loader.dataset.txt2img,num_bins=config['num_bins'])
        mean_adaece = (image_text_ece.item() + text_image_ece.item()) / 2
        calibration_dict = {
            'epoch': epoch,
            'mean_ece': mean_adaece,
            'image_text_meancalibration_gap': image_text_meancalibration_gap,
            'text_image_meancalibration_gap': text_image_meancalibration_gap,
            'image_text_ece': image_text_ece.item(),
            'text_image_ece': text_image_ece.item(),
        }
        if utils.is_main_process():
            with open(os.path.join(args.output_dir+time_dir, "val_calibration_dict.txt"), "a") as f:
                f.write(json.dumps(calibration_dict) + "\n")
        WeightsoftCEloss.updategamma(image_text_meancalibration_gap,text_image_meancalibration_gap)
        score_test_i2t, score_test_t2i = evaluation(model_without_ddp, test_loader, tokenizer, device, config, k=0)
        image_text_ece, image_text_bin_dict,text_image_ece,text_image_bin_dict,image_text_meancalibration_gap,text_image_meancalibration_gap = evaluate_dataset_ECE_error(score_test_i2t,score_test_t2i,test_loader.dataset.img2txt,val_loader.dataset.txt2img,num_bins=config['num_bins'])
        mean_adaece = (image_text_ece.item() + text_image_ece.item()) / 2
        calibration_dict = {
            'epoch': epoch,
            'mean_ece': mean_adaece,
            'image_text_meancalibration_gap': image_text_meancalibration_gap,
            'text_image_meancalibration_gap': text_image_meancalibration_gap,
            'image_text_ece': image_text_ece.item(),
            'text_image_ece': text_image_ece.item(),
        }
        if utils.is_main_process():
            with open(os.path.join(args.output_dir+time_dir, "test_calibration_dict.txt"), "a") as f:
                f.write(json.dumps(calibration_dict) + "\n")
        if epoch >= 25:
            score_test_i2t, score_test_t2i = evaluation(model_without_ddp, test_loader, tokenizer, device, config, k=40)
            if utils.is_main_process():
                test_result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img, test_loader.dataset.img2txt)
                # val_result = itm_eval(score_val_i2t, score_val_t2i, val_loader.dataset.txt2img, val_loader.dataset.img2txt)
                # print(f"rs5m :{rs5m_val_result}")
                print(test_result)
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                                    # **{f'val_{k}': v for k, v in val_result.items()},
                                    **{f'test_{k}': v for k, v in test_result.items()},
                                    'epoch': epoch}
                with open(os.path.join(args.output_dir+time_dir, filename), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")
                if test_result['r_mean'] > best:
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'config': config,
                    }
                    torch.save(save_obj, os.path.join(args.output_dir+time_dir, 'checkpoint_best.pth'))
                    best = test_result['r_mean']
                    best_epoch = epoch
                else:
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'config': config,
                        # 'epoch': epoch,
                    }
                    torch.save(save_obj, os.path.join(args.output_dir+time_dir, f'checkpoint_last.pth'))
        dist.barrier()
        torch.cuda.empty_cache()
    if utils.is_main_process():
        with open(os.path.join(args.output_dir+time_dir, filename), "a") as f:
            f.write("best epoch: %d" % best_epoch)

        os.system(f"cat {args.output_dir}{time_dir}/{filename}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('### Time {}'.format(total_time_str))
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--precheckpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_dir',type=str, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--world_size', default=2, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', action='store_false')
    parser.add_argument('--bs', default=-1, type=int, help="for each gpu, batch_size = bs // num_gpus")
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--accumulation_steps', type=int, default=8)
    parser.add_argument("--num_worker", type=int,
                        default=5,
                        help='number of workers')
    parser.add_argument("--batch_size", type=int,
                        default=64,
                        help='batch size')

    args = parser.parse_args()
    yaml = YAML()
    print("Init Successful")
    with open(args.config, 'r') as config_file:
        config = yaml.load(config_file)
    Path(args.output_dir+time_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    WeightsoftCEloss = Weight_soft_CEloss(imagegamma = config['weight_init_imagegamma'],textgamma = config['weight_init_textgamma'],maxgamma = config['themaxgamma'],mingamma = config['themingamma'])
    yaml.dump(config, open(os.path.join(args.output_dir+time_dir, 'config.yaml'), 'w'))    
    main(args, config)