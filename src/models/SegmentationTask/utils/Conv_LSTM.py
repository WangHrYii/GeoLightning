import torch.nn as nn
from torch.autograd import Variable
import torch


class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        
        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias        = bias
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        
        h_cur, c_cur = cur_state
        
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1) 
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        # the shape of h and c is (batch_size, self.hidden_dim, self.height, self.width)
        return h_next, c_next
    # init hidden layers
    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda(),
                Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda())


class ConvLSTM(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        """
        
        Parameters
        ----------
        input_size: 
            the spatial size of input tensor (height, width)
        input_dim:
            the dim of input tensor
        hidden_dim:
            the dim of hidden convlstm cell [num1, num2, ...], it also means the out_dim (dim of h and c)
        kernel_size:
            kernel size of convlstm (3, 3)
        num_layers:
            the number of ConvLSTM cells
            
        """       
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are list and their length == num_layers
        # if is not list, then extend these two params to list, which has the same length as num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim  = self._extend_for_multilayer(hidden_dim, num_layers)
        # make sure the length of kernel_size equal to num_layers
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        # size of feature map
        self.height, self.width = input_size

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        # build a list with ConvLSTM cells. the length of the list is num_layers, that means num_layers is the hidden number of ConvLSTM cells
        cell_list = []
        for i in range(0, self.num_layers):
            # if is the first cell, then the is no previous output, and the input dim is the dim of input_data
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]

            # append cells into al list
            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        # convert a normal list to modulelist
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        
        Parameters
        ----------
        input_tensor: todo 
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
            
        Returns
        -------
        last_state_list, layer_output
        """
        # if use the batch as the first dim, then convert dim, else the first dim is length of time series
        # t: the number of group in your data, each group means a 'moment' sent to ConvLSTM
        # b: batch size
        # c: channels of each group
        # h, w: the spatial size of each group
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list   = []

        # seq_len is t
        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        # layer_idx means the id of ConvLSTM cell in the cell list
        for layer_idx in range(self.num_layers):
            
            # h and c is two path of lstm
            h, c = hidden_state[layer_idx]
            # output of each lstm cell, each cell has an output h, these h will be stored with this list
            output_inner = []

            # t means curent 'time' of input tensor
            for t in range(seq_len):
                # update h and c, h and c will go through all cell in sequence, so that they can update parameters with t set value from 0-seq_len
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            # stack the output together, the shape of h is (b, hidden_dim, h, w), the shape of layer_output is (b, hidden_dim * seq_len, h, w)
            layer_output = torch.stack(output_inner, dim=1)
            # set the input of next cell
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])
        # import ipdb;ipdb.set_trace()
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1]
            last_state_list   = last_state_list[-1]

        return layer_output_list, last_state_list

    # return a list, each element of the list is an all zero tensor with the shape of (batch_size, num_channels, height, width)
    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                    (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
