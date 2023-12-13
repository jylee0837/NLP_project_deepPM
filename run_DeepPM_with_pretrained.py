import argparse
import torch
import torch.nn as nn
import torch.optim

import models
import train
import optim

from torch.utils.data import DataLoader
from utils import set_seeds, get_device
from DeepPM_utils import *
from experiments.experiment import Experiment
from models import BERT
from train import BERTTrainer
from data.data_cost import BERTNoMaskDataset
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer

from custom_models import BERTStackedDeepPMPadZero

def main():
    # type: () -> None
    parser = argparse.ArgumentParser()

    # data arguments
    parser.add_argument('--data', required=True, help='The data file to load from')
    parser.add_argument('--train_cfg', required=True, help='Configuration for train')
    parser.add_argument('--model_cfg', required=True, help='Configuration of model')
    parser.add_argument('--experiment-name', required=True, help='Name of the experiment to run')
    parser.add_argument('--experiment-time', required=True, help='Time the experiment was started at')
    #parser.add_argument("-c", "--train_dataset", required=True, type=str, help="train dataset for train bert")
    parser.add_argument("-t", "--test_dataset", type=str, default=None, help="test set for evaluate train set")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="number of batch_size")
    parser.add_argument("-w", "--num_workers", type=int, default=0, help="dataloader worker size")
    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--log_freq", type=int, default=50, help="printing loss every n iter: setting n")
    parser.add_argument("--corpus_lines", type=int, default=None, help="total number of lines in corpus")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
    parser.add_argument("--on_memory", type=bool, default=True, help="Loading on memory: true or false")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")

    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")
    parser.add_argument("--bert_path", required=True, type=str, help="ex)output/bert_weight.pth")
    parser.add_argument("--hot_index_mapping_path", required=True, type=str, help="ex)output/predictor.dump")


    args = parser.parse_args()
    print("parser done")

    #DeepPM용 data 생성
    data, _= load_data(args.data, args.hot_index_mapping_path) #data: 마스킹 되지 않은 DeepPM용 one-hot 벡터 변환됨., testestest: 마스킹 되지 않은 BERT용 one-hot 벡터 변환됨
    
    print('load data done')
    #data 전처리(masking) (=BERTDataset) + DataLoader -> 마스킹 되지 않은 기존 데이터만 받도록 수정함

    #DeepPM config 파일들 받아옴
    cfg = train.Config.from_json(args.train_cfg)
    print('cfg done')
    model_cfg = models.Config.from_json(args.model_cfg)
    print('model_cfg done')

    model_cfg.set_vocab_size(660)   #사전사이즈     #628 -> 660
    print(model_cfg)        #출력 결과: Config(vocab_size=None, dim=512, n_layers=224, n_heads=8, dim_ff=2048, p_drop_hidden=0.1, p_drop_attn=0.1, max_len=100, pad_idx=628)
    print('set vocab size done')
    set_seeds(cfg.seed)
    print('set seeds done')
    expt = Experiment(args.experiment_name, args.experiment_time)
    print('expt done')

    #BERT data만드는거(masking된 명령어 token data)
    pad_idx = data.token_to_hot_idx['<PAD>']
    end_idx = data.token_to_hot_idx['<END>']
    train_dataset = BERTNoMaskDataset(data.train, pad_idx=pad_idx)      #vocab 받아와야함
    print('train dataset done')
    print('train_dataset======================')

    test_dataset = BERTNoMaskDataset(data.test, pad_idx=pad_idx)     
    print('dataset build done')


    print("Creating Dataloader")
    using_dataset = train_dataset
    train_dl = DataLoader(using_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True,
                        collate_fn=using_dataset.collate_fn)

    using_dataset = test_dataset
    test_dl = DataLoader(using_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False,
                        collate_fn=using_dataset.collate_fn)
    
    print('dataloader done')



    ## Building BERT model

    bert = BERT(vocab_size=660, hidden=512, n_layers=12, attn_heads=8)
    pretrained_bert_weights = torch.load(args.bert_path, map_location=torch.device('cpu'))
    bert.load_state_dict(pretrained_bert_weights)

    # for longer positional encoding support
    bert.embedding.position = Summer(PositionalEncoding1D(bert.hidden))
    bert.eval()

    # freezing bert
    for param in bert.parameters():
        param.requires_grad = False
    print('Loaded pretrained BERT model')




    # Initialize the DeepPM model
    model = BERTStackedDeepPMPadZero.from_cfg(bert, pad_idx=pad_idx, end_idx=end_idx, cfg=model_cfg)
    model = model.to(get_device())  # Move the model to the appropriate device
    
    dump_model_and_data(model, data, os.path.join(expt.experiment_root_path(), 'predictor.dump'))
    print('dump_model_and_data done')
    
    optimizer = torch.optim.Adam(model.parameters(), cfg.lr)
    lr_sched = torch.optim.lr_scheduler.LinearLR(optimizer)
    #DeepPM train
    #trainer = train.Trainer(cfg, model, data, expt, optim.optim4GPU(cfg, model), get_device())
    trainer = train.Trainer(cfg, model, (train_dataset, test_dataset), expt, 
                            optimizer,
                            lr_sched,
                            get_device())
    print('trainer done')
    trainer.train()
    print('train done')

if __name__ == '__main__':
    main()
