import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import torch
from tqdm import tqdm
from .data import Data
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import statistics
import pandas as pd
import xml.etree.ElementTree as ET
import itertools

import sys
sys.path.append('..')

import utilities as ut


class DataItem:
    def __init__(self, x, y, block, code_id):
        self.x = x
        self.y = y
        self.block = block
        self.code_id = code_id
        #self.mask_label = mask_label

class Vocabulary:
    def __init__(self):
        self.token_to_hot_idx_m = {}
        self.hot_idx_to_token_m = {}
        self.pad_index_m = 0
        self.mask_index_m = 630

class DataInstructionEmbedding(Data):       #masking안된버전(forDeepPMdata)

    def __init__(self, token_to_hot_idx_file=None):
        super(DataInstructionEmbedding, self).__init__()
        self.token_to_hot_idx = {}
        self.hot_idx_to_token = {}
        self.data = []

    def dump_dataset_params(self):
        return (self.token_to_hot_idx, self.hot_idx_to_token)

    def load_dataset_params(self, params):
        (self.token_to_hot_idx, self.hot_idx_to_token) = params

    def prepare_data(self, progress=True, fixed=False):
        def hot_idxify(elem):
            if elem not in self.token_to_hot_idx:
                if fixed:
                    # TODO: this would be a good place to implement UNK tokens
                    raise ValueError('Ithemal does not yet support UNK tokens!')
                self.token_to_hot_idx[elem] = len(self.token_to_hot_idx)
                self.hot_idx_to_token[self.token_to_hot_idx[elem]] = elem
            return self.token_to_hot_idx[elem]
        
        self.token_to_hot_idx['<PAD>'] = 0  
        self.hot_idx_to_token[0] = '<PAD>'

        if progress:
            iterator = tqdm(self.raw_data)
        else:
            iterator = self.raw_data
        
        send_x = []
        for (code_id, timing, code_intel, code_xml) in iterator:
            
            #if timing > 1112:#1000:
            #    continue

            block_root = ET.fromstring(code_xml)
            instrs = []
            raw_instrs = []
            curr_mem = self.mem_start
            for _ in range(1): # repeat for duplicated blocks
                # handle missing or incomplete code_intel
                split_code_intel = itertools.chain((code_intel or '').split('\n'), itertools.repeat(''))
                for (instr, m_code_intel) in zip(block_root, split_code_intel):
                    raw_instr = []
                    opcode = int(instr.find('opcode').text)
                    raw_instr.extend([opcode, '<SRCS>'])
                    #raw_instr.append(opcode)
                    srcs = []
                    for src in instr.find('srcs'):
                        if src.find('mem') is not None:
                            raw_instr.append('<MEM>')
                            for mem_op in src.find('mem'):
                                raw_instr.append(int(mem_op.text))
                                srcs.append(int(mem_op.text))
                            raw_instr.append('</MEM>')
                            srcs.append(curr_mem)
                            curr_mem += 1
                        else:
                            raw_instr.append(int(src.text))
                            srcs.append(int(src.text))

                    raw_instr.append('<DSTS>')
                    dsts = []
                    for dst in instr.find('dsts'):
                        if dst.find('mem') is not None:
                            raw_instr.append('<MEM>')
                            for mem_op in dst.find('mem'):
                                raw_instr.append(int(mem_op.text))
                                # operands used to calculate dst mem ops are sources
                                srcs.append(int(mem_op.text))
                            raw_instr.append('</MEM>')
                            dsts.append(curr_mem)
                            curr_mem += 1
                        else:
                            raw_instr.append(int(dst.text))
                            dsts.append(int(dst.text))

                    raw_instr.append('<END>')
                    raw_instrs.append(list(map(hot_idxify, raw_instr)))
                    #print('-----------------raw_instr-------------------')
                    #print(raw_instr)
                    #print("---------------------------------------------")
                    #print("token_to_hot")
                    #index_17_value = self.token_to_hot_idx[4]
                    #print(len(self.token_to_hot_idx))
                    instrs.append(ut.Instruction(opcode, srcs, dsts, len(instrs)))
                    instrs[-1].intel = m_code_intel
            if len(raw_instrs) > 400:
                #print(len(raw_instrs))
                continue
            
            #print('raw_instrs') 
            #print(raw_instrs)
            
            block = ut.BasicBlock(instrs)
            block.create_dependencies()
            datum = DataItem(raw_instrs, timing, block, code_id)
            #print("-------datum.x-------")
            #print(datum.x)
            #print(datum.x)
            #testestest=datum.x
           # print(testestest)
            #print("-------datum.x-------")
            #print(datum.x) #[[51, 1, 5, 21, 7, 4, 15, 8], [29, 1, 66, 6, 4, 8]]같이 마스킹 되지 않고 one-hot으로 변환된 토큰 하나의 베이직 블록 단위로 나옴
            send_x.append(datum.x)
            #print('--------------------------sendx----------------------------')
            #print(send_x)   #누적돼서 쌓인 3차원 형태로 나옴
            #print(len(send_x))  #1씩 올라감
            self.data.append(datum)
        ##print("-------datum.x-------")
        #print(len(self.data))   #815992
        #print('--------------------------sendx----------------------------')
        #print(len(send_x))      #815992
        #exit()
        #print(send_x)

        return send_x


#위에서 datum.x 받아서 마스킹한 token(bert_input)이랑 label(bert_label) return함
class BERTDataset(Dataset):     #masking한 버전(forBERTdata)

    def __init__(self, testestest):
        self.vocab = Vocabulary()
        self.Dataembedding = DataInstructionEmbedding
        self.testestest = testestest
        #self.seq_len = seq_len

    def __len__(self):
        return len(self.testestest)

    def __getitem__(self, index):

        #print('----indexnum------')
        #print(index)
        #print('self.testestest[index]')
        #print(self.testestest[index])
        #print(len(self.testestest[index]))

        t1, t1_label = self.random_word(self.testestest[index])
        #print(t1_label)
        bert_input = t1[:150]    #[:self.seq_len]
        bert_label = t1_label[:150]

        #print('bert_input===========')
        ##print(bert_input)
        #print(len(bert_input))
        #print('bert_label')
        #print(bert_label)
        #print(len(bert_label))

        
        if len(bert_input) < 150:  # Check if the size is less than 2
            pad_length = 150 - len(bert_input)
            bert_input += [self.vocab.pad_index_m] * pad_length
            bert_label += [self.vocab.pad_index_m] * pad_length


        #padding = [self.vocab.pad_index_m for _ in range(20 - len(bert_input))]
        #bert_input.extend(padding)
        #bert_label.extend(padding) 


        output = {"bert_input": bert_input,
                  "bert_label": bert_label}
        
        #print('output=========')
        #print(output)   #딕셔너리로 들어가서 출력 잘됨

        #aa ={key: torch.tensor(value) for key, value in output.items()}
        aa = {key: torch.tensor(value) for key, value in output.items()}
        #print('----------aa--------------')
        #print(aa)   #tensor로 들어감
        #print(aa.tolist())





        #for key, value in aa.items():
        #    print(f"{key} size: {value.size()}")
        #print('aa')
        #print(aa)

        # 추가된 코드: 각 value의 크기를 출력 -> [n,15]로 나옴.
        #sizes = [value.size() for value in aa.values()]
        #print('Sizes of each value in `aa`:', sizes)

        # 추가된 코드: 모든 value의 크기가 같은지 확인
        #is_same_size = all(size == sizes[0] for size in sizes)
        #print('Are all values in `aa` the same size?:', is_same_size)


        return aa
    
    def random_word(self, sentence):        #랜덤 단어 마스킹하고, 해당 마스킹된 위치의 정답 단어 레이블 생성. sentence는 마스킹할 문장임
        tokens = np.concatenate(sentence).tolist()           #datum.x 받음
        #print('TOKENS')
        #print(tokens)
        output_label = []
        for i, token in enumerate(tokens):      #각 단어에 대해 랜덤한 확률(prob) 계산
            prob = random.random()              #prob 값에 따라 단어를 마스킹할지, 랜덤 단어로 대체할지, 현재 단어 그대로 사용할지 결정함
            if prob < 0.15:
                prob /= 0.15
                # 80% randomly change token to mask token 진짜 마스킹
                if prob < 0.8:
                    tokens[i] = self.vocab.mask_index_m
                # 10% randomly change token to random token 랜덤 단어로 대체
                elif prob < 0.9:
                    tokens[i] = random.randrange(627)
                # 10% randomly change token to current token    안바꾸는거
                else:
                    pass
                output_label.append(token)       #마스킹된 단어에 대해 정답 레이블 생성
            else:
                output_label.append(0)  #마스킹되지 않은 단어에 대해서는 0으로 레이블 설정
            #print('tokens')
            #print(token)
            #print('output_label')
            #print(output_label)
        return tokens, output_label     #마스킹된 문장, 정답 레이블 반환

class BERTNoMaskDataset(Dataset):     #masking한 버전(forBERTdata)

    def __init__(self, data, long_limit=512, pad_idx=0):
        self.long_limit = long_limit
        self.xs = [torch.cat([torch.tensor(tmp) for tmp in datum.x]) for datum in data]
        self.ys = [datum.y for datum in data]
        self.total_len = len(self.ys)
        self.pad_idx = pad_idx

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        return self.xs[index], self.ys[index]
    
    def collate_fn(self, batch):
        short_x = []
        short_y = []
        long_x = []
        long_y = []

        for x, y in batch:
            if len(x) <= self.long_limit:
                short_x.append(x)
                short_y.append(y)
            else:
                long_x.append(x)
                long_y.append(y)

        if len(short_x) > 0:
            short_max_len = max([len(x) for x in short_x])
            short_x = torch.stack(
                [
                    F.pad(item, (0, short_max_len-len(item)), value=self.pad_idx)
                        for item in short_x
                ]
            )
            short_y = torch.tensor(short_y)

        short_dict = {
            'x': short_x,
            'y': short_y
        }

        long_list = [
            {
                'x': x.unsqueeze(0),
                'y': torch.tensor([y]),
            } for x, y in zip(long_x, long_y)
        ]

        return {
            'short': short_dict,
            'long': long_list
        }
    
'''
    #2D로 다시 만드는거. seq를 basicblock단위로 생각
    def random_word(self, sentence):
        tokens = np.concatenate(sentence).tolist()

        output_label = []
        new_tokens = []
        new_output_label = []
        curr_index = 0

        for sublist in sentence:
            new_sublist = [] 
            new_output_sublist = []
            for token in sublist:
                prob = random.random()
                if prob < 0.15:
                    prob /= 0.15

                    # 80% randomly change token to mask token 진짜 마스킹
                    if prob < 0.8:
                        tokens[curr_index] = self.vocab.mask_index_m #self.vocab.mask_index

                    # 10% randomly change token to random token 랜덤 단어로 대체
                    elif prob < 0.9:
                        tokens[curr_index] = random.randrange(len(tokens))
                    
                    # 10% randomly change token to current token    안바꾸는거
                    else:
                        pass

                    output_label.append(token)
                    new_output_sublist.append(token)

                else:
                    output_label.append(0)
                    new_output_sublist.append(0)
                new_sublist.append(tokens[curr_index])
                curr_index += 1
            padding = [self.vocab.pad_index_m] * (15 - len(new_sublist))
            new_sublist.extend(padding)
            new_output_sublist.extend(padding)
            
            new_tokens.append(new_sublist)
            new_output_label.append(new_output_sublist)

            #padding = [self.vocab.pad_index_m for _ in range(20 - len(new_tokens))]
            #new_tokens.extend(padding)
            #new_output_label.extend(padding) 

            print('-----new_tokens----------')
            print(new_tokens)
            print('-----new_output_label----------')
            print(new_output_label)

        return new_tokens, new_output_label
'''

    



def load_dataset(data_savefile=None, token_to_hot_idx_file=None, arch=None, format='text'):
    print("****************load_dataset START********************")

    data = DataInstructionEmbedding()       #객체 생성
    if token_to_hot_idx_file is not None:
        hot_idx = torch.load(token_to_hot_idx_file, map_location=torch.device('cpu')).dataset_params
        data.load_dataset_params(hot_idx)

    data.raw_data = torch.load(data_savefile)       #raw_data 넣어주고
    #raw_data: 리스트 형태. 각 원소는 하나의 블록 나타내는 리스트. 
    #print(data.raw_data[0]) #하나의 블록 나타내는 리스트. 측정값, 원래 명령어, 토큰들 있음
    #print(data.raw_data[1])
    #print(data.raw_data[3])
    #print(data.raw_data)
    data.read_meta_data()       #여기서 함수 실행됨
    testestest = data.prepare_data()
    #testestest = prepare_data()
    #testestest = DataInstructionEmbedding()
    #print(testestest)
    #testestest = data.prepare_data()         #블록 토큰화, 원핫인코딩으로 변환
    print('-----------testestest----------')
    print(len(testestest))  #출력 결과: 815992
    #vocab = vocab
    data.generate_datasets()    #test, train으로 나눔(652793, 163198으로 나눠짐)
    print('generate dataset done')

    #print("data--------------")
    #print(data.data[0].x)   #출력 결과: [[0, 1, 2, 3, 4, 3, 5, 3, 6, 7, 8], [9, 1, 5, 10, 6, 7, 6, 4, 8]]
    #print(data.data[3].x)
    #print(data.data[4].x)
    #print("--------------")
    for instr, token_inputs in zip(data.data[0].block.instrs, data.data[0].x):
        print(instr, token_inputs)  #출력 결과: push   rbx [0, 1, 2, 3, 4, 3, 5, 3, 6, 7, 8] test   byte ptr [rdi+0x0e], 0x01 [9, 1, 5, 10, 6, 7, 6, 4, 8]


    return data, testestest #vocab




class DataInstructionEmbedding_fixed_notuse(Data):

    def __init__(self):
        super(DataInstructionEmbedding_fixed_notuse, self).__init__()

        self.vocab = Vocabulary()
        
        #self.token_to_hot_idx = {}  #vocab_dict
        #self.hot_idx_to_token = {}
        self.data = []
        
    def dump_dataset_params(self):
        return (self.vocab.token_to_hot_idx, self.vocab.hot_idx_to_token)

    def load_dataset_params(self, params):
        (self.vocab.token_to_hot_idx, self.vocab.hot_idx_to_token) = params
        #print("aa")
        #print(params)

    def prepare_data(self, progress=True, fixed=False):
        def hot_idxify(elem):
            
            #print(elem)
            if elem not in self.vocab.token_to_hot_idx:
                if fixed:
                    # TODO: this would be a good place to implement UNK tokens
                    raise ValueError('Ithemal does not yet support UNK tokens!')
                self.vocab.token_to_hot_idx[elem] = len(self.vocab.token_to_hot_idx)
                self.vocab.hot_idx_to_token[self.vocab.token_to_hot_idx[elem]] = elem
            return self.vocab.token_to_hot_idx[elem]
        

        if progress:
            iterator = tqdm(self.raw_data)
        else:
            iterator = self.raw_data

        for (code_id, timing, code_intel, code_xml) in iterator:
            
            #if timing > 1112:#1000:
            #    continue
            block_root = ET.fromstring(code_xml)
            instrs = []
            raw_instrs = []
            mask_label = []
            curr_mem = self.mem_start
            for _ in range(1): # repeat for duplicated blocks
                # handle missing or incomplete code_intel
                split_code_intel = itertools.chain((code_intel or '').split('\n'), itertools.repeat(''))
                for (instr, m_code_intel) in zip(block_root, split_code_intel):
                    raw_instr = []
                    opcode = int(instr.find('opcode').text)
                    raw_instr.extend([opcode, '<SRCS>'])
                    #raw_instr.append(opcode)
                    srcs = []
                    for src in instr.find('srcs'):
                        if src.find('mem') is not None:
                            raw_instr.append('<MEM>')
                            for mem_op in src.find('mem'):
                                raw_instr.append(int(mem_op.text))
                                srcs.append(int(mem_op.text))
                            raw_instr.append('</MEM>')
                            srcs.append(curr_mem) 
                            curr_mem += 1
                        else:
                            raw_instr.append(int(src.text))
                            srcs.append(int(src.text))

                    raw_instr.append('<DSTS>')
                    dsts = []
                    for dst in instr.find('dsts'):
                        if dst.find('mem') is not None:
                            raw_instr.append('<MEM>')
                            for mem_op in dst.find('mem'):
                                raw_instr.append(int(mem_op.text))
                                # operands used to calculate dst mem ops are sources
                                srcs.append(int(mem_op.text))
                            raw_instr.append('</MEM>')
                            dsts.append(curr_mem)
                            curr_mem += 1
                        else:
                            raw_instr.append(int(dst.text))
                            dsts.append(int(dst.text))

                    raw_instr.append('<END>')
                    raw_instrs.append(list(map(hot_idxify, raw_instr)))
                    print('~~~~~~~raw_instr~~~~~~~~~')
                    print(raw_instr)
                    #print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
                    #print("token_to_hot")
                    #print(self.vocab.token_to_hot_idx)
                    instrs.append(ut.Instruction(opcode, srcs, dsts, len(instrs)))
                    instrs[-1].intel = m_code_intel

            
            if len(raw_instrs) > 400:
                #print(len(raw_instrs))
                continue
            
            
            block = ut.BasicBlock(instrs)
            block.create_dependencies()
            datum = DataItem(raw_instrs, timing, block, code_id, mask_label)
            ##print("-------datum.x-------")
            ##print(datum.x)  #[[11,1,5,35,7,10,8],[14,1,21,4,10,8]]처럼 basic block이 토큰화된 형태
            #print("datum.x[1]")
            ###print(datum.x[1])       #[14,1,21,4,10,8] 출력됨
            ###print("datum.y----")
            ###print(datum.y)     #292.0 처럼 측정값
            #print(datum.code_id)    #1 계속 나옴
            #마스킹 여기서 ?
            gotoBERTtoken, = self.random_word(datum.x)       #마스킹
            ##print("---------datum.x after masking--------")
            ##print(datum.x)  #원핫벡터
            ##print("---------datum.mask_label after masking----------")
            ##print(datum.mask_label)     #마스킹된거??
           #print(datum.mask_label[0])
            self.data.append(datum)
        print("masking finish")
        print("token_to_hot")
        print(self.vocab.token_to_hot_idx)
        print("hot_to_token ")
        print(self.vocab.hot_idx_to_token)
        #print("len(self.vocab)")
        #print(len(self.vocab))
        #print("~~~~~~~~~~~datum.x")
        #print(datum.x)  #원핫벡터

    '''
    def random_word(self, sentence):        #랜덤 단어 마스킹하고, 해당 마스킹된 위치의 정답 단어 레이블 생성. sentence는 마스킹할 문장임
        tokens = np.concatenate(sentence).tolist()           #datum.x 받음
        print('TOKENS')
        print(tokens)
        output_label = []
        for i, token in enumerate(tokens):      #각 단어에 대해 랜덤한 확률(prob) 계산
            prob = random.random()              #prob 값에 따라 단어를 마스킹할지, 랜덤 단어로 대체할지, 현재 단어 그대로 사용할지 결정함
            if prob < 0.15:
                prob /= 0.15
                # 80% randomly change token to mask token 진짜 마스킹
                if prob < 0.8:
                    tokens[i] = self.vocab.mask_index
                # 10% randomly change token to random token 랜덤 단어로 대체
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(tokens))
                # 10% randomly change token to current token    안바꾸는거
                else:
                    pass
                output_label.append(token)       #마스킹된 단어에 대해 정답 레이블 생성
            else:
                output_label.append(0)  #마스킹되지 않은 단어에 대해서는 0으로 레이블 설정
        return tokens, output_label     #마스킹된 문장, 정답 레이블 반환
    '''

    #2D->1D->2D
    def random_word(self, sentence):
        tokens = np.concatenate(sentence).tolist()

        output_label = []
        new_tokens = []
        new_output_label = []
        curr_index = 0

        for sublist in sentence:
            new_sublist = [] 
            new_output_sublist = []
            for token in sublist:
                prob = random.random()
                if prob < 0.15:
                    prob /= 0.15

                    # 80% randomly change token to mask token 진짜 마스킹
                    if prob < 0.8:
                        tokens[curr_index] = self.vocab.mask_index

                    # 10% randomly change token to random token 랜덤 단어로 대체
                    elif prob < 0.9:
                        tokens[curr_index] = random.randrange(len(tokens))
                    
                    # 10% randomly change token to current token    안바꾸는거
                    else:
                        pass

                    output_label.append(token)
                    new_output_sublist.append(token)

                else:
                    output_label.append(0)
                    new_output_sublist.append(0)
                new_sublist.append(tokens[curr_index])
                curr_index += 1
            new_tokens.append(new_sublist)
            new_output_label.append(new_output_sublist)

        return new_tokens, new_output_label