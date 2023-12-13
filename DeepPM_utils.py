import sys
import os

from enum import Enum
import torch
from typing import Any, NamedTuple

import data.data_cost as dt
#import models.graph_models as md

def load_data(data_file, token_to_hot_idx_file=None):
    # type: (BaseParameters) -> dt.DataCost
    data, testestest = dt.load_dataset(data_file, token_to_hot_idx_file)
#    def filter_data(filt):
        # type: (Callable[[dt.DataItem], bool]) -> None
#        data.data = [d for d in data.data if filt(d)]
#        data.train = [d for d in data.train if filt(d)]
#        data.test = [d for d in data.test if filt(d)]

#    if params.no_mem:
#        filter_data(lambda d: not d.block.has_mem())

#    ablate_data(data, params.edge_ablation_types, params.random_edge_freq)

#    if params.linear_dependencies:
#        filter_data(lambda d: d.block.has_linear_dependencies())

#    if params.flat_dependencies:
#        filter_data(lambda d: d.block.has_no_dependencies())

    return data, testestest

"""
def load_model(params, data):
    # type: (BaseParameters, dt.DataCost) -> md.AbstractGraphModule

    if params.use_rnn:
        rnn_params = md.RnnParameters(
            embedding_size=params.embed_size,
            hidden_size=params.hidden_size,
            num_classes=1,
            connect_tokens=params.rnn_connect_tokens,
            skip_connections=params.rnn_skip_connections,
            hierarchy_type=params.rnn_hierarchy_type,
            rnn_type=params.rnn_type,
            learn_init=params.rnn_learn_init,
        )
        model = md.RNN(rnn_params)
    else:
        model = md.GraphNN(embedding_size=params.embed_size, hidden_size=params.hidden_size, num_classes=1,
                           use_residual=not params.no_residual, linear_embed=params.linear_embeddings,
                           use_dag_rnn=not params.no_dag_rnn, reduction=params.dag_reduction,
                           nonlinear_type=params.dag_nonlinearity, nonlinear_width=params.dag_nonlinearity_width,
                           nonlinear_before_max=params.dag_nonlinear_before_max,
        )

    model.set_learnable_embedding(mode=params.embed_mode, dictsize=628 or max(data.hot_idx_to_token) + 1)

    return model
"""
PredictorDump = NamedTuple('PredictorDump', [
    ('model', Any),
    ('dataset_params', Any),
])

def dump_model_and_data(model, data, fname):
    # type: (md.AbstractGraphMode, dt.DataCost, str) -> None
    try:
        os.makedirs(os.path.dirname(fname))
    except OSError:
        pass
    torch.save(PredictorDump(
        model=model,
        dataset_params=data.dump_dataset_params(),
    ), fname)

def load_model_and_data(fname):
    # type: (str) -> (md.AbstractGraphMode, dt.DataCost)
    dump = torch.load(fname)
    data = dt.DataInstructionEmbedding()
    data.read_meta_data()
    data.load_dataset_params(dump.dataset_params)
    return (dump.model, data)

