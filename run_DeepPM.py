import argparse
import torch

import models
import train
import optim

from torch.utils.data import DataLoader
from utils import set_seeds, get_device
from DeepPM_utils import *
from experiments.experiment import Experiment
from models import BERT
from train import BERTTrainer
from data.data_cost import BERTDataset



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
    parser.add_argument("-o", "--output_path", required=True, type=str, help="ex)output/bert.model")


    args = parser.parse_args()
    print("parser done")

    #DeepPM용 data 생성
    data, testestest= load_data(args.data) #data: 마스킹 되지 않은 DeepPM용 one-hot 벡터 변환됨., testestest: 마스킹 되지 않은 BERT용 one-hot 벡터 변환됨
    print(data)
    #print(testestest)
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
    train_dataset = BERTDataset(testestest)      #vocab 받아와야함
    print('train dataset done')
    ##sample = train_dataset[0]
    ##print('-------------------------------------')
    ##for key, value in sample.items():
    ##    print(f"{key}: {value}")
    ##print('-------------------------------------')
    print('train_dataset======================')
    #print(len(train_dataset)) #815992 출력됨
    test_dataset = BERTDataset(testestest) \
        if args.test_dataset is not None else None
    print('dataset build done')

    ##train data랑 test data 나누기
    #def collate_fn(batch):
    #    bert_input = [torch.tensor(sample['bert_input']) for sample in batch]
    #    bert_label = [torch.tensor(sample['bert_label']) for sample in batch]
    #    
    #    bert_input = torch.nn.utils.rnn.pad_sequence(bert_input, batch_first=True)
    #    bert_label = torch.nn.utils.rnn.pad_sequence(bert_label, batch_first=True)

    #    return {'bert_input': bert_input, 'bert_label': bert_label}




    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)#, collate_fn = collate_fn)
    print('train_data_loader===========')
    print(train_data_loader)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False) \
        if test_dataset is not None else None
    print('dataloader done')



    ## Building BERT model

    bert = BERT(vocab_size=660, hidden=512, n_layers=12, attn_heads=8)
    print('BERT model build done')

    ##Creating BERT Trainer
    trainer = BERTTrainer(bert, 660, train_data_loader, test_data_loader,
                          lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                          with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq)
    print('creating BERT trainer done')
    ###여기까진됨

    saved_weights_path = os.path.join(args.output_path, 'bert_weights.pth')


    ##BERT Pretrain하고
    for epoch in range(args.epochs):
        trainer.train(epoch)
        trainer.save(epoch, args.output_path)
        torch.save(bert.state_dict(), saved_weights_path)   #save the BERT model weights after training for each epoch

        if test_data_loader is not None:
            trainer.test(epoch)

    print('BERT pretrain done')

##########################

    #DeepPM 모델 만들고
    #model = models.DeepPM(model_cfg)
    #print('DeepPM model done')

    #bert_state_dict = torch.load(saved_weights_path) #저장된 weight 불러와서
    #model.bert.load_state_dict(bert_state_dict)     #DeepPM model에 추가해줌

################################

    # Load the pretrained BERT model weights
    pretrained_bert_weights = torch.load(saved_weights_path)

    print('pretrainedbertweights------------------------')
    print(pretrained_bert_weights.keys())

    # Initialize the DeepPM model
    model = models.DeepPM(model_cfg)
    model = model.to(get_device())  # Move the model to the appropriate device




    # Load the BERT weights into the pre_blocks layers
    for i, block in enumerate(model.pre_blocks):
        block_prefix = f'transformer_blocks.{i}.'  # BERT 모델에서 사용된 키 구조에 맞게 변경
        for key, value in pretrained_bert_weights.items():
            if key.startswith(block_prefix):
                new_key = key.replace(block_prefix, '')
                block.state_dict()[new_key].copy_(value)

    print('Loaded pretrained BERT weights into DeepPM pre_blocks layers')



    model = model.to(get_device())
    print('model get_device done')
    
    dump_model_and_data(model, data, os.path.join(expt.experiment_root_path(), 'predictor.dump'))
    print('dump_model_and_data done')
    
    #DeepPM train
    #trainer = train.Trainer(cfg, model, data, expt, optim.optim4GPU(cfg, model), get_device())
    trainer = train.Trainer(cfg, model, data, expt, "1", get_device())
    print('trainer done')
    trainer.train()
    print('train done')

if __name__ == '__main__':
    main()
