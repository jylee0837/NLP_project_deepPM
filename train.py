# Copyright 2018 Dong-Hyun Lee, Kakao Brain.

""" Training Config & Helper Classes  """

import os
import json
from typing import NamedTuple
from models import BERTLM, BERT
from optim import ScheduledOptim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
import time
import losses as ls
import random
import csv

from utils import correct_regression

class Config(NamedTuple):
    """ Hyperparameters for training """
    seed: int = 3431 # random seed
    batch_size: int = 32
    lr: int = 5e-5 # learning rate
    n_epochs: int = 10 # the number of epoch
    # `warm up` period = warmup(0.1)*total_steps
    # linearly increasing learning rate from zero to the specified value(5e-5)
    warmup: float = 0.001
    clip_grad_norm: float = 0.2
    #save_steps: int = 100 # interval for saving model
    #total_steps: int = 100000 # total number of steps to train

    @classmethod
    def from_json(cls, file): # load config from json file
        return cls(**json.load(open(file, "r")))

class LossReporter(object):
    def __init__(self, experiment, n_data_points):
        # type: (Experiment, int, tr.Train) -> None

        self.experiment = experiment
        self.n_datapoints = n_data_points
        self.start_time = time.time()

        self.loss = 1.0
        self.avg_loss = 1.0
        self.epoch_no = 0
        self.total_processed_items = 0
        self.epoch_processed_items = 0
        self.accuracy = 0.0

        self.last_report_time = 0.0
        self.last_save_time = 0.0

        self.root_path = self.experiment.experiment_root_path()

        try:
            os.makedirs(self.root_path)
        except OSError:
            pass

        self.loss_report_file = open(os.path.join(self.root_path, 'loss_report.log'), 'w',1)
        self.pbar = tqdm(desc = self.format_loss(), total=self.n_datapoints)

    def format_loss(self):

        return 'Epoch {}, Loss: {:.2}, {:.2}, Accuracy: {:.2}'.format(
                self.epoch_no,
                self.loss,
                self.avg_loss,
                self.accuracy
        )

    def start_epoch(self, epoch_no):
        
        self.epoch_no = epoch_no
        self.epoch_processed_items = 0
        self.accuracy = 0.0

        self.pbar.close()
        self.pbar = tqdm(desc=self.format_loss(), total=self.n_datapoints)

    def report(self, n_items, loss, avg_loss, t_accuracy):

        self.loss = loss
        self.avg_loss = avg_loss
        self.accuracy = t_accuracy
        self.epoch_processed_items += n_items
        self.total_processed_items += n_items

        desc = self.format_loss()
        self.pbar.set_description(desc)
        self.pbar.update(n_items)

    def checkpoint(self, model, optimizer, lr_scheduler, file_name):

        state_dict = {
            'epoch': self.epoch_no,
            'model': model.state_dict(),
            'optimizer':optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict()
        }
            
        try: 
            os.makedirs(os.path.dirname(file_name))
        except OSError:
            pass

        torch.save(state_dict, file_name) 

    def end_epoch(self, model, optimizer, lr_scheduler, loss):
        
        self.loss = loss

        t = time.time()
        message = '\t'.join(map(str, (
            self.epoch_no,
            t - self.start_time,
            self.loss,
            self.accuracy,
        )))
        self.loss_report_file.write(message + '\n')

        file_name = os.path.join(self.experiment.checkpoint_file_dir(),'{}.mdl'.format(self.epoch_no))
        self.checkpoint(model,optimizer, lr_scheduler, file_name)


    def finish(self, model, optimizer, lr_scheduler):

        self.pbar.close()
        print("Finishing training")

        file_name = os.path.join(self.root_path, 'trained.mdl')
        self.checkpoint(model, optimizer, lr_scheduler, file_name)

    def save_best(self, model, optimizer, lr_scheduler):
        file_name = os.path.join(self.root_path, 'best.mdl')
        self.checkpoint(model,optimizer, lr_scheduler, file_name)

class Trainer:
    def __init__(self, cfg, model, ds, expt, optimizer, lr_scheduler, device):
        self.cfg = cfg # config for training : see class Config
        self.model = model
        self.train_ds, self.test_ds = ds
        self.expt = expt
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device # device name
        self.save_dir = self.expt.experiment_root_path()

        self.loss_fn = ls.mse_loss

        self.loss_reporter = LossReporter(expt, len(self.train_ds))
        self.clip_grad_norm = cfg.clip_grad_norm

    class BatchResult:
        def __init__(self):
            self.batch_len = 0

            self.measured = []
            self.prediction = []
            
            self.loss = 0
            self.loss_sum = 0
            self.mape = 0
            self.mape_sum = 0

        def __iadd__(self, other):
            self.batch_len += other.batch_len

            self.measured.extend(other.measured)
            self.prediction.extend(other.prediction)

            self.loss += other.loss
            self.loss_sum += other.loss_sum
            return self
        
        def __repr__(self):
            return f'''Batch len: {self.batch_len}
Loss: {self.loss}
'''

    def run_batch(self, batch, is_train=False):
        short = batch['short']
        long = batch['long']

        result = self.BatchResult()
        short_len = len(short['y'])
        long_len = len(long)
        result.batch_len = short_len + long_len
        
        if short_len > 0:
            loss_mod = short_len / result.batch_len if long_len > 0 else None

            x = short['x'].to(self.device)
            output = self.model(x)
            y = short['y'].to(self.device)
            loss = self.loss_fn(output, y)

            if loss_mod is not None:
                loss *= loss_mod

            if is_train:
                loss.backward()

            result.measured.extend(y.tolist())
            result.prediction.extend(output.tolist())

            result.loss += loss.item()
        
        if long_len > 0:
            for long_item in long:
                x = long_item['x'].to(self.device)
                output = self.model(x)
                y = long_item['y'].to(self.device)
                loss = self.loss_fn(output, y)

                if loss_mod is not None:
                    loss *= loss_mod

                if is_train:
                    loss.backward()

                result.measured.extend(y.tolist())
                result.prediction.extend(output.tolist())

                result.loss += loss.item()

        result.loss_sum = result.loss * result.batch_len
        return result

    def train(self):
        """ Train Loop """
        best_correct = -1
        resultfile = os.path.join(self.expt.experiment_root_path(), 'validation_results.txt')

        self.model.to(self.device)

        loader = DataLoader(self.train_ds, batch_size=self.cfg.batch_size, shuffle=True,
                        collate_fn=self.train_ds.collate_fn)
    
        epoch_len = len(loader)

        for epoch_no in range(self.cfg.n_epochs):
            epoch_loss_sum = 0.
            step = 0
            total_correct = 0
            total_cnts = 0
            print(f'using lr: {self.optimizer.param_groups[0]["lr"]}')
            self.loss_reporter.start_epoch(epoch_no + 1) 

            self.model.train()
            
            for idx, batch in enumerate(loader):
                self.optimizer.zero_grad()
                batch_result = self.run_batch(batch, is_train=True)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

                self.optimizer.step()

                step += 1
            
                epoch_loss_sum += batch_result.loss_sum
                total_cnts += batch_result.batch_len

                report_batch_len = batch_result.batch_len
                total_correct += correct_regression(batch_result.prediction, batch_result.measured, 25)
                self.loss_reporter.report(report_batch_len, 
                                    batch_result.loss, epoch_loss_sum/step, total_correct/total_cnts)   
              
            epoch_loss_avg = epoch_loss_sum / step
            self.loss_reporter.end_epoch(self.model, self.optimizer, self.lr_scheduler, epoch_loss_avg)

            cur_loss, correct = self.validate(resultfile, epoch_no + 1)

            if correct >= best_correct:
                best_correct = correct
                self.loss_reporter.save_best(self.model, self.optimizer, self.lr_scheduler)

            self.lr_scheduler.step()

        self.loss_reporter.finish(self.model, self.optimizer, self.lr_scheduler)

    def validate(self, resultfile, epoch):
        self.model.eval()
        self.model.to(self.device)

        f = open(resultfile,'w')

        loader = DataLoader(self.test_ds, shuffle=False,
            batch_size=self.cfg.batch_size, collate_fn=self.test_ds.collate_fn)
        epoch_result = self.BatchResult()

        with torch.no_grad():
            for batch in tqdm(loader):
                epoch_result += self.run_batch(batch, is_train=False)

       
        
        epoch_result.loss = epoch_result.loss_sum / epoch_result.batch_len 
        correct = correct_regression(epoch_result.prediction, epoch_result.measured, 25)
        f.write(f'loss - {epoch_result.loss}\n')
        f.write(f'{correct}, {epoch_result.batch_len}\n')
        f.close()


        print(f'Validate: loss - {epoch_result.loss}\n\t{correct}/{epoch_result.batch_len} = {correct / epoch_result.batch_len}\n')
        print()

        return epoch_result.loss, correct


class BERTTrainer:
    """
    BERTTrainer make the pretrained BERT model with two LM training method.

        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction

    please check the details on README.md with simple example.

    """

    def __init__(self, bert: BERT, vocab_size: int,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 80):
        """
        :param bert: BERT model which you want to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        #self.device = "cpu"
        self.device =torch.device("cuda:0" if cuda_condition else "cpu")

        # This BERT model will be saved every epoch
        self.bert = bert
        # Initialize the BERT Language Model, with BERT model
        self.model = BERTLM(bert, vocab_size).to(self.device)

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, self.bert.hidden, n_warmup_steps=warmup_steps)

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.criterion = nn.NLLLoss(ignore_index=0)

        self.log_freq = log_freq

        # CSV 파일에 저장될 로그의 경로 설정
        self.csv_file = os.path.join("output", "training_logs.csv")

        # CSV 파일의 헤더 작성
        with open(self.csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["epoch", "iter", "loss", "avg_loss", "avg_acc", "avg_nonpad_acc"])


        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_data)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)
    
    '''
    def save_to_csv(data, filename):
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)
            '''

    def iteration(self, epoch, data_loader, train=True):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        str_code = "train" if train else "test"

        print("Entering iteration function")  # Debug print 1
        print("data_loader:", data_loader)  # Debug print 2
        print("data_loader.dataset:", data_loader.dataset)  # Debug print 3
        print("len(data_loader.dataset):", len(data_loader.dataset))  # Debug print 4

        # Setting the tqdm progress bar
        data_iter = tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        total_correct = 0
        total_element = 0
        total_nonpadcorr = 0
        total_nonpadelem = 1

        #start_iter = 427100



        for i, data in data_iter:

        #    if i < start_iter:
        #        continue
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}

            # 1. forward the next_sentence_prediction and masked_lm model
            #print('----------------bert_input-----------')
            #print(data["bert_input"])
            #print('----------------bert_input printed-----------')
            mask_lm_output = self.model.forward(data["bert_input"])

            # 2-2. NLLLoss of predicting masked token word
            loss = self.criterion(mask_lm_output.transpose(1, 2), data["bert_label"])


            # 3. backward and optimization only in train
            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()

            # masked language model prediction accuracy
            #print('mask_lm_output===========================')      #예측결과(확률)
            #print(mask_lm_output)
            #print('"bert_label"===========================')        #원래 답
            #print(data["bert_label"])
            masked_preds = mask_lm_output.argmax(dim=-1)    #가장 높은 확률 가진 단어의 인덱스 선택. ignore index에 따라 달라짐
            #print('masked_preds===========================')
            #print(masked_preds)
            mask_lm_correct = masked_preds.eq(data["bert_label"]).sum().item()      #정답이랑 예측 비교하고 불리언 마스크 생성, 정확하게 예측된 경우 개수 계산
            #print('mask_lm_correct===========================')
            #print(mask_lm_correct)      #150개 전체에 대해 맞은 개수
            #print(data["bert_label"].nelement())    #150개
            avg_loss += loss.item()
            total_correct += mask_lm_correct        #150개 전체에 대해 맞은 개수 총합
            total_element += data["bert_label"].nelement()  ##패딩 포함, 150개 전체 총합



            non_padding_mask = data["bert_label"] != 0      #패딩 부분인 0 제외하고 마스킹된 부분만
            non_padding_corr = (masked_preds[non_padding_mask] == data["bert_label"][non_padding_mask]).sum().item()    #마스킹된 부분에 대해 정답 개수
            #print(non_padding_corr)
            #print(non_padding_mask.sum().item())
            total_nonpadcorr += non_padding_corr        #마스킹된 부분만 정답 개수 총합
            total_nonpadelem += non_padding_mask.sum().item()       #마스킹된 부분 개수 총합
            #print(total_element)
            #print(total_nonpadcorr)
            #print(total_nonpadelem)

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "loss": loss.item(),        #해당 iteration loss
                "avg_loss": avg_loss / (i + 1),     #loss average
                "avg_acc": total_correct / total_element * 100,     #150개에 대한 accuracy average
                "avg_nonpad_acc": total_nonpadcorr / total_nonpadelem * 100     #마스킹된 부분에 대한 accuracy average
            }

                        # 로그 빈도에 도달할 때마다 로그 정보를 CSV 파일에 추가
            if i % self.log_freq == 0:
                with open(self.csv_file, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([epoch, i, loss.item(), avg_loss / (i + 1), total_correct / total_element * 100, total_nonpadcorr / total_nonpadelem * 100])
                data_iter.write(str(post_fix))
                


        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter), "total_acc=",
              total_correct * 100.0 / total_element, "total_nonpad_acc=", total_nonpadcorr * 100.0 / total_nonpadelem)


    def save(self, epoch, file_path="output/bert_trained.model"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + ".ep%d" % epoch
        torch.save(self.bert.cpu(), output_path)
        self.bert.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path
    
    