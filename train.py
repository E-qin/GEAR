# QWC edit 2023/07/02 GMT+8 22:00

import argparse
import collections
import json
import torch
import numpy as np
from data_loader.data_loaders import build_loaders
import model.loss as module_loss
import model.metric as module_metric
from model.model import RankinglossModel_v6 as RankinglossModel 
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration, BertTokenizer
from preprocess.data_prepare import  PrepareData
from Index_tree.tree_structure import Node


SEED =  42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config, config_path):
    logger = config.get_logger('train')

    if config["model_path"].endswith('T5') or config["model_path"].endswith('Randeng-T5-77M-MultiTask-Chinese'):
        special_tokens = ["<extra_id_{}>".format(i) for i in range(100)]
        model_tokenizer  = T5Tokenizer.from_pretrained(
                                config["model_path"],
                                do_lower_case=True,
                                max_length= config["max_len"],
                                truncation=True,
                                additional_special_tokens=special_tokens,
                            )
    elif config["model_path"].endswith('chinese-cluecorpussmall'):
        model_tokenizer = BertTokenizer.from_pretrained(config["model_path"])
    else:
        raise ValueError("{}".format(config["model_path"]))

    
    ex_id = config_path[config_path.rfind('/')+1:config_path.rfind('.')]
    if ex_id == '':
        print("config_path", config_path)
        print("Cannot find ex_id in config_path")
        exit(0)
    
    dataset_name = config_path[config_path.rfind('/', 0, config_path.rfind('/'))+1:config_path.rfind('/')]
    data_provider = PrepareData(config=config, dataset_name=dataset_name, tree_pad_token_id=0)
    final_model_input,extra_input = data_provider.prepare_data_ELAM(config, ex_id)


    
    docid_ty = config["ELAM_data_prepare"]["docid_ty"] 
    if docid_ty == "atomic": 
        print("Adding atomic ids to tokenizer.........")
        model_tokenizer.add_tokens(list(data_provider.atomicId_docid_dict.values()))
    elif docid_ty == "original_atomic": 
        print("Adding atomic ids to tokenizer.........")
        model_tokenizer.add_tokens(list(data_provider.atomicId_docid_dict.values()))
    elif docid_ty == "CrimeTxt_random_exact": 
        with open("Seq_ids/CrimeTxt_random_exact/for_atomicRandomNum.json",'r')as f:
            model_tokenizer.add_tokens(list(json.load(f)))     
    elif docid_ty.startswith('atomic_'): 
        print("Adding atomic ids to tokenizer.........")
        model_tokenizer.add_tokens(list(data_provider.atomicId_docid_dict.values()))

    
    
    if config["model_path"].endswith('T5') or config["model_path"].endswith('Randeng-T5-77M-MultiTask-Chinese'):
        model_config = T5Config.from_pretrained(config["model_path"])
        if "nci" in config and config["nci"]:
            pretrain_params = dict(T5ForConditionalGeneration.from_pretrained(config["model_path"], config=model_config).named_parameters())
            model = T5ForConditionalGeneration(model_config)
            for name, param in model.named_parameters():
                if name.startswith(("shared.", "encoder.")):
                    with torch.no_grad():
                        param.copy_(pretrain_params[name])
            mode_model = model
            print("NCI")
        else:
            # model_config = T5Config.from_pretrained(config["model_path"])
            mode_model = T5ForConditionalGeneration.from_pretrained(config["model_path"], config=model_config)
        mode_model.resize_token_embeddings(len(model_tokenizer))
    elif config["model_path"].endswith('chinese-cluecorpussmall'):
        mode_model = T5ForConditionalGeneration.from_pretrained(config["model_path"])
        mode_model.resize_token_embeddings(len(model_tokenizer))
    else:
        raise ValueError("{}".format(config["model_path"]))    
    
    if config["gradient_ckpt"]:
        mode_model.config.gradient_checkpointing = True
    
    model = RankinglossModel(mode_model, model_tokenizer,  data_provider, 
                     config=config, dbg=config["model_dbg"])
        

    train_loader, test_loader, extra_train_loader, _ = build_loaders(config, ex_id, model_tokenizer, 
                                            data_provider=data_provider,
                                            final_model_input=final_model_input,
                                            extra_input=extra_input,
                                            # atomicId_labels_dict=atomicId_labels_dict, 
                                            # atomicId_docid_dict=atomicId_docid_dict,
                                        )

    
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        print("Let's use GPUs:", device_ids)
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics_ftns = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics_ftns, optimizer,
                      config=config,
                      device=device,
                      train_data_loader=train_loader,
                      test_data_loader=test_loader,
                      extra_train_loader=extra_train_loader,
                      lr_scheduler=lr_scheduler,
                      ex_id=ex_id,
                      data_provider=data_provider
    )

    
    trainer.train()


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default="./config.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options) 

    args = args.parse_args()


    main(config, args.config)
