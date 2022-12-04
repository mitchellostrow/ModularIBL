from collections import OrderedDict
import networkx
import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.init as init

class RecurrentModel(nn.Module):

    def __init__(self,
                 model_architecture,
                 model_kwargs):

        super(RecurrentModel, self).__init__()
        self.model_str = model_architecture
        assert model_architecture in {'rnn', 'lstm', 'gru'}
        self.model_kwargs = model_kwargs
        self.input_size = model_kwargs['input_size']
        self.output_size = model_kwargs['output_size']

        # create and save core i.e. the recurrent operation
        self.core = self._create_core(
            model_architecture=model_architecture,
            model_kwargs=model_kwargs)

        masks = self._create_connectivity_masks(
            model_str=model_architecture,
            model_kwargs=model_kwargs)
        self.input_mask = masks['input_mask']
        self.recurrent_mask = masks['recurrent_mask']
        self.readout_mask = masks['readout_mask']

        self.description_str = create_description_str(model=self)

        self.core_hidden = None
        self.readout = nn.Linear(
            in_features=model_kwargs['core_kwargs']['hidden_size'],
            out_features=self.output_size,
            bias=True)

        if self.output_size == 1:
            self.prob_fn = nn.Sigmoid()
        elif self.output_size == 2:
            self.prob_fn = nn.Softmax(dim=2)

        # converts all weights into doubles i.e. float64
        # this prevents PyTorch from breaking when multiplying float32 * float64
        self.double()

        # TODO figure out why writing the model to tensorboard doesn't work
        # dummy_input = torch.zeros(size=(10, 1, 1), dtype=torch.double)
        # tensorboard_writer.add_graph(
        #     model=self,
        #     input_to_model=dict(stimulus=dummy_input))

    def _create_core(self, model_architecture, model_kwargs):
        if model_architecture == 'lstm':
            core_constructor = nn.LSTM
        elif model_architecture == 'rnn':
            core_constructor = nn.RNN
        elif model_architecture == 'gru':
            core_constructor = nn.GRU
        else:
            raise ValueError('Unknown core string')

        core = core_constructor(
            input_size=self.input_size,
            batch_first=True,
            **model_kwargs['core_kwargs'])

        param_init_str = model_kwargs['param_init']
        if param_init_str == 'default':
            return core
        elif param_init_str == 'eye':
            param_init_fn = init.eye_
        elif param_init_str == 'zeros':
            param_init_fn = init.zeros_
        elif param_init_str == 'ones':
            # TODO: breaks with error
            # ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
            param_init_fn = init.ones_
        elif param_init_str == 'uniform':
            param_init_fn = init.uniform
        elif param_init_str == 'normal':
            # TODO: breaks with error
            # ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
            param_init_fn = init.normal_
        elif param_init_str == 'xavier_uniform':
            param_init_fn = init.xavier_uniform_
        elif param_init_str == 'xavier_normal':
            param_init_fn = init.xavier_normal_
        else:
            raise NotImplementedError(f'Weight init function {param_init_str} unrecognized')

        if param_init_str != 'default':
            for weight in core.all_weights:
                for parameter in weight:
                    # some initialization functions e.g. eye only apply to 2D tensors
                    # skip the 1D tensors e.g. bias
                    try:
                        param_init_fn(parameter)
                    except ValueError:
                        continue

        return core

    def _create_connectivity_masks(self, model_str, model_kwargs):

        hidden_size = model_kwargs['core_kwargs']['hidden_size']

        # if mask not specifies, set to defaults
        for mask_str in ['input_mask', 'recurrent_mask', 'readout_mask']:
            if mask_str not in model_kwargs['connectivity_kwargs']:
                if mask_str == 'input_mask':
                    model_kwargs['connectivity_kwargs'][mask_str] = mask_str
                elif mask_str == 'readout_mask':
                    model_kwargs['connectivity_kwargs'][mask_str] = mask_str
                elif mask_str == 'recurrent_mask':
                    model_kwargs['connectivity_kwargs'][mask_str] = 'none'

        # determine how much to inflate
        if self.model_str == 'rnn':
            size_prefactor = 1
        elif self.model_str == 'gru':
            size_prefactor = 3
        elif self.model_str == 'lstm':
            size_prefactor = 4

        # create input-to-hidden, hidden-to-hidden, hidden-to-readout masks
        masks = dict()
        for mask_str, mask_type_str in model_kwargs['connectivity_kwargs'].items():

            if mask_str == 'input_mask':
                mask_shape = (size_prefactor * hidden_size, self.input_size)
            elif mask_str == 'recurrent_mask':
                mask_shape = (size_prefactor * hidden_size, hidden_size)
            elif mask_str == 'readout_mask':
                mask_shape = (self.output_size, hidden_size)
            else:
                raise ValueError(f'Unrecognized mask str: {mask_str}')

            mask = self._create_mask(
                mask_type_str=mask_type_str,
                output_shape=mask_shape[0],
                input_shape=mask_shape[1])

            masks[mask_str] = mask

        return masks

    def _create_mask(self, mask_type_str, output_shape, input_shape):

        if mask_type_str == 'none':
            connectivity_mask = np.ones(shape=(output_shape, input_shape))
        elif mask_type_str == 'input_mask':
            # special case for input - zeros except for first 30% of rows
            connectivity_mask = np.zeros(shape=(output_shape, input_shape))
            connectivity_mask[:int(0.3 * output_shape), :] = 1
        elif mask_type_str == 'readout_mask':
            # special case for output -
            connectivity_mask = np.zeros(shape=(output_shape, input_shape))
            connectivity_mask[:, -int(0.3 * input_shape):] = 1
        elif mask_type_str == 'diagonal':
            connectivity_mask = np.eye(N=output_shape, M=input_shape)
        elif mask_type_str == 'circulant':
            first_column = np.zeros(shape=output_shape)
            first_column[:int(0.2 * output_shape)] = 1.
            connectivity_mask = scipy.linalg.circulant(c=first_column)
        elif mask_type_str == 'toeplitz':
            first_column = np.zeros(shape=output_shape)
            first_column[:int(0.2 * output_shape)] = 1.
            connectivity_mask = scipy.linalg.toeplitz(c=first_column)
        elif mask_type_str == 'small_world':
            graph = networkx.watts_strogatz_graph(
                n=output_shape,
                k=int(0.2 * output_shape),
                p=0.1)
            connectivity_mask = networkx.to_numpy_matrix(G=graph)
        elif mask_type_str.endswith('_block_diag'):
            # extract leading integer
            num_blocks = int(mask_type_str.split('_')[0])
            subblock_size = output_shape // num_blocks
            # check output size is exactly divisible by number of blocks
            assert num_blocks * subblock_size == output_shape
            connectivity_mask = scipy.linalg.block_diag(
                *[np.ones((subblock_size, subblock_size))] * num_blocks)
        else:
            raise ValueError(f'Unrecognized mask type str: {mask_type_str}')

        connectivity_mask = torch.from_numpy(connectivity_mask).double()
        return connectivity_mask

    def forward(self, model_input):
        """
        Performs a forward pass through model.


        :param model_input: dictionary containing 4 keys:
            stimulus: Tensor with shape (batch size, 1 step, stimulus dimension)
            reward: Tensor with shape (batch size, 1 step)
            info: List of len batch size. Currently unused
            done: List of len batch size. Booleans indicating whether environment is done.
        :return forward_output: dictionary containing 4 keys:
            core_output: Tensor of shape (batch size, num steps, core dimension)
            core_hidden: Tensor of shape (batch size, num steps, core dimension)
            linear_output: Tensor of shape (batch size, num steps, output dimension)
            prob_output: Tensor of shape (batch size, num steps, output dimension)
        """

        core_input = torch.cat(
            [model_input['stimulus'],
             torch.unsqueeze(model_input['reward'], dim=2)],  # TODO: check that this change didn't break anything
            dim=2)

        core_output, self.core_hidden = self.core(
            core_input,
            self.core_hidden)

        # hidden state is saved as (Number of RNN layers, Batch Size, Dimension)
        # swap so that hidden states is (Batch Size, Num of RNN Layers, Dimension)
        if self.model_str == 'rnn' or self.model_str == 'gru':
            core_hidden = self.core_hidden.transpose(0, 1)
        elif self.model_str == 'lstm':
            # hidden state is 2-tuple of (h_t, c_t). need to save both
            # stack h_t, c_t using last dimension
            # shape: (Batch Size, Num of RNN Layers, Dimension, 2)
            core_hidden = torch.stack(self.core_hidden, dim=-1).transpose(0, 1)
        else:
            raise NotImplementedError

        linear_output = self.readout(core_output)

        # shape: (batch size, 1, output dim e.g. 1)
        prob_output = self.prob_fn(linear_output)

        # if probability function is sigmoid, add 1 - output to get 2D distribution
        if self.output_size == 1:
            prob_output = torch.cat([1 - prob_output, prob_output], dim=2)
            # TODO: implement linear output i.e. inverse sigmoid
            linear_output = None
            raise NotImplementedError

        forward_output = dict(
            core_output=core_output,
            core_hidden=core_hidden,
            linear_output=linear_output,
            prob_output=prob_output)

        return forward_output

    def reset_core_hidden(self):
        self.core_hidden = None

    def apply_connectivity_masks(self):

        self.readout.weight.data[:] = torch.mul(
            self.readout.weight, self.readout_mask)

        # if self.model_str == 'rnn':
        self.core.weight_ih_l0.data[:] = torch.mul(
            self.core.weight_ih_l0, self.input_mask)
        self.core.weight_hh_l0.data[:] = torch.mul(
            self.core.weight_hh_l0, self.recurrent_mask)
        # elif self.model_str == 'lstm':
        #     raise NotImplementedError('LSTM masking not yet implemented')
        # elif self.model_str == 'gru':
        #     raise NotImplementedError('GRU masking not yet implemented')
        # else:
        #     raise NotImplementedError('Unrecognized Model String')


def create_description_str(model):
    description_str = '{}'.format(model.model_str)
    for key, value in model.model_kwargs.items():
        if key == 'input_size' or key == 'output_size':
            continue
        if isinstance(value, dict):
            for nkey, nvalue in value.items():
                description_str += ', {}={}'.format(str(nkey), str(nvalue))
        else:
            description_str += ', {}={}'.format(str(key), str(value))
    print(description_str)
    return description_str
