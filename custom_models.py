import torch
import torch.nn as nn
import torch.nn.functional as F

from positional_encodings.torch_encodings import PositionalEncoding1D, Summer

class Seq(nn.Module):
    def __init__(self, dim, dim_ff, n_heads, n_layers, t_dropout=0.1, activation='relu'):
        super().__init__()

        block = nn.TransformerEncoderLayer(dim, n_heads, dim_feedforward=dim_ff,
                                                batch_first=True, dropout=t_dropout, activation=activation)
        self.tr = nn.TransformerEncoder(block, n_layers, enable_nested_tensor=False)

    def forward(self, x, mask, op_seq_mask):
        batch_size, inst_size, seq_size, _ = x.shape

        x = x.view(batch_size * inst_size, seq_size, -1)
        mask = mask.view(batch_size * inst_size, seq_size)
        op_seq_mask = op_seq_mask.view(batch_size * inst_size)

        x = x.masked_fill(op_seq_mask.unsqueeze(-1).unsqueeze(-1), 1)
        mod_mask = mask.masked_fill(op_seq_mask.unsqueeze(-1), False)
        x = self.tr(x, src_key_padding_mask=mod_mask)
        x = x.masked_fill(mask.unsqueeze(-1), 0)
        x = x.view(batch_size, inst_size, seq_size, -1)
        return x
    
class BasicBlock(nn.Module):
    def __init__(self, dim, dim_ff, n_heads, n_layers, t_dropout=0.1, activation='relu'):
        super().__init__()

        block = nn.TransformerEncoderLayer(dim, n_heads, dim_feedforward=dim_ff,
                                                batch_first=True, dropout=t_dropout, activation=activation)
        self.tr = nn.TransformerEncoder(block, n_layers, enable_nested_tensor=False)

    def forward(self, x, mask):
        batch_size, inst_size, seq_size, _ = x.shape

        x = x.view(batch_size, inst_size * seq_size, -1)
        mask = mask.view(batch_size, inst_size * seq_size)
        x = self.tr(x, src_key_padding_mask=mask)
        x = x.masked_fill(mask.unsqueeze(-1), 0)
        x = x.view(batch_size, inst_size, seq_size, -1)
        return x
    
class Op(nn.Module):
    def __init__(self, dim, dim_ff, n_heads, n_layers, t_dropout=0.1, activation='relu'):
        super().__init__()

        block = nn.TransformerEncoderLayer(dim, n_heads, dim_feedforward=dim_ff,
                                                batch_first=True, dropout=t_dropout, activation=activation)
        self.tr = nn.TransformerEncoder(block, n_layers, enable_nested_tensor=False)

    def forward(self, x, op_seq_mask):
        batch_size, inst_size, _ = x.shape

        x = self.tr(x, src_key_padding_mask=op_seq_mask)
        x = x.masked_fill(op_seq_mask.unsqueeze(-1), 0)
        return x
    
    
class BERTStackedDeepPMPadZero(nn.Module):
    """DeepPM model with Trasformer """
    def __init__(self, bert,
                dim=512, n_heads=8, dim_ff=2048, 
                pad_idx=0, end_idx=9,
                vocab_size=700, pred_drop=0.0,
                num_instruction_layer=2,
                num_op_layer=4):
        super().__init__()

        self.bert = bert

        self.instruction_layer = Seq(dim, dim_ff, n_heads, num_instruction_layer)
        self.op_layer = Op(dim, dim_ff, n_heads, num_op_layer)

        self.end_idx = end_idx
        self.pad_idx = pad_idx
        self.pos_embed = Summer(PositionalEncoding1D(dim))
        self.prediction = nn.Sequential(
            nn.Dropout(pred_drop),
            nn.Linear(dim, 1)
        )

    @classmethod
    def from_cfg(cls, bert, pad_idx, end_idx, cfg):
        return cls(bert, 
                   vocab_size=bert.embedding.token.num_embeddings,
                   pad_idx=pad_idx,
                   end_idx=end_idx,
                   dim=cfg.dim, 
                   n_heads=cfg.n_heads,
                   dim_ff=cfg.dim_ff)
    
    def train(self, mode=True):
        super().train(mode)
        self.bert.eval()

    def bert_output_to_deeppm_input(self, x, output):
        all_instrs = []
        all_tensors = []
        for x_row, output_row in zip(x, output):
            end_mask = x_row == self.end_idx
            end_idx_tensor = end_mask.nonzero(as_tuple=True)[0]
            lens = end_idx_tensor - torch.cat((torch.tensor([-1], dtype=torch.int, device=x.device), end_idx_tensor[:-1]), dim=0)
            max_seq = max(lens)
            without_padding_len = end_idx_tensor[-1].item() + 1

            x_row = x_row[:without_padding_len]
            instrs = x_row.split(lens.tolist())
            instrs = [F.pad(instr, (0, max_seq-instr.size(0)), value=self.pad_idx) for instr in instrs]
            instrs = torch.stack(instrs)
            all_instrs.append(instrs)

            output_row = output_row[:without_padding_len]
            tensors = output_row.split(lens.tolist())
            tensors = [F.pad(tn, (0, 0, 0, max_seq-tn.size(0)), value=self.pad_idx) for tn in tensors]
            tensors = torch.stack(tensors)
            all_tensors.append(tensors)

        max_instr = max([instr.size(0) for instr in all_instrs])
        max_seq = max([instr.size(1) for instr in all_instrs])
        all_instrs = torch.stack(
            [
                F.pad(instr, (0, max_seq-instr.size(1), 0, max_instr-instr.size(0)), value=self.pad_idx) 
                    for instr in all_instrs
            ]
        )

        all_tensors = torch.stack(
            [
                F.pad(tn, (0, 0, 0, max_seq-tn.size(1), 0, max_instr-tn.size(0)), value=self.pad_idx) 
                    for tn in all_tensors
            ]
        )
        return all_instrs, all_tensors
       
    def forward(self, x):
        output = self.bert(x)
        x, output = self.bert_output_to_deeppm_input(x, output)

        # Basic setup
        batch_size, inst_size, seq_size = x.shape
        mask = x == self.pad_idx
        op_seq_mask = mask.all(dim=-1)
        
        # Instruction layer
        output = self.instruction_layer(output, mask, op_seq_mask)

        #  Selecting Op
        output = output[:,:, 0, :]

        # Op layer
        output = self.pos_embed(output)
        output = self.op_layer(output, op_seq_mask)
        output = output.sum(dim = 1)
        out = self.prediction(output).squeeze(1)

        return out
    