
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import itertools

import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init

torch.manual_seed(1)


# # Generator Structure
# 
# Generator Class
# Goals:
#     


MAX_SEQ_LEN = 10
MONTE_CARLO_N = 5
class Encoder(nn.Module):
    r"""
    Embeds an input sequence (in our case consisting of input-output pairs)
    into a latent vector. This latent vector is then used by the decoder to
    produce "program" outputs

    Arguments:
        1. vocab_dim (int): size of the vocabulary
        2. embedding_dim (int): dimension of the embedding
        3. hidden_dim (int): dimensionality of the hidden vectors
        4. num_layers (int): number of hidden layers
        5. max_seq_len (int): maximum size of the sequence (*TODO - make unnecessary)

    TODO:
        1. Make batching neater
        2. Make variable-sequence length easier
        3. Implement Drop-Out (***) See Srivastava et al. 
                            http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf
        4. Make Bi-Directional LSTM
    """

    def __init__(self, vocab_dim, embedding_dim, hidden_dim, num_layers, max_seq_len=MAX_SEQ_LEN, rnn='LSTM'):

        super(Encoder, self).__init__() # blindly do this

        # make initializations
        self.vocab_dim = vocab_dim # size of input-output space (equiv. to vocab
                                                            # for speech tagging)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        # layers
        self.embedding = nn.Embedding(self.vocab_dim, self.embedding_dim)
        self.type = rnn
        if rnn=='LSTM':
            self.rnn = nn.LSTM(self.embedding_dim, self.hidden_dim, self.num_layers)
        elif rnn=='GRU':
            self.rnn = nn.GRU(self.embedding_dim, self.hidden_dim, self.num_layers)

        else:
            raise("not implemented")

    def forward(self, inputs, hidden, variable_length=False, batch_size=1, input_lengths=None):
        r"""
        Creates forward graph; returns an output (useless) and final hidden vectors/states. Can accept minibatches.
        Arguments:
            1. inputs (Variable: (seq_len, batch, input_dim))
                example:
                >>> inputs = Variable(torch.FloatTensor([[[1,2,3,4],[2,3,0,0]], [[2,3,3,4],[2,3,2,3]]]))
                >>> inputs.data.size()

                Out: torch.Size([2, 2, 4])
                            sequence_index; batch_index; input_dim_index
            2. hidden (Variable): hidden value of previous state
            3. variable_length: If variable length is set to true, input_lengths need to be provided.
                                NOTE: Both inputs and input_lengths need to be provided in descending order of the
                                number of elements in each batch desired. 

                                see: http://pytorch.org/docs/master/nn.html#torch.nn.utils.rnn.pack_padded_sequence
        """

        if variable_length==False:
            seq_len = inputs.size()[0]
            embedded = self.embedding(inputs).view(seq_len,batch_size,-1)
            print("Embedded", embedded)
            output, hidden = self.rnn(embedded, hidden)

        elif variable_length==True:
            seq_len = inputs.size()[0]
            assert(input_lengths!=None)
            embedded = self.embedding(inputs).view(seq_len, batch_size,-1)
            
            packed_inputs = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
            output, hidden = self.rnn(packed_inputs, hidden)

        return output, hidden

    def init_hidden(self):
        r"""
        Initialized hidden vector/cell-states. Given that weight initializations are crucial, we experiment with
        the Xavier Normal Initialization/Glorot Initialization.

        see: 1. http://pytorch.org/docs/0.1.12/_modules/torch/nn/init.html
             2. "Understanding the difficulty of training deep feedforward neural networks" 
                                                                                - Glorot, X. & Bengio, Y. (2010)
        """
        if self.type == 'LSTM':
            # dim = [num_layers*num_directions, batch, hidden_size]
            h0 = torch.Tensor(self.num_layers, 1, self.hidden_dim)
            c0 = torch.Tensor(self.num_layers, 1, self.hidden_dim)

            # initializations
            h0, c0 = init.xavier_normal(h0), init.xavier_normal(c0)

            return Variable(h0), Variable(c0)

        elif self.type == 'GRU':
            h = torch.Tensor(self.num_layers, 1, self.hidden_size)
            h = init.xavier_normal(h)

            return Variable(h)



class Decoder(nn.Module):
    """
    without attention and drop-out layers 
    """
    def __init__(self, vocab_dim, hidden_dim, num_layers=1, rnn='LSTM'):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # layers
        self.embedding = nn.Embedding(vocab_dim, hidden_dim)
        self.type = rnn
        if self.type=='LSTM':
            self.rnn = nn.LSTM(self.hidden_dim, self.hidden_dim, self.num_layers)
        elif self.type=='GRU':
            self.rnn = nn.GRU(self.hidden_dim, self.hidden_dim, self.num_layers)
        self.out = nn.Linear(hidden_dim, vocab_dim)
        
    def forward(self, input, hidden):
        r"""
        Creates forward graph.
        
        Arguments:
            1. input (1,1): if no teacher_forcing, this corresponds to the last output of the RNN
            2. hidden : previous hidden layers
        """
        output = self.embedding(input).view(1,1,-1)
        outputs_dict = {}
        for i in range(self.num_layers):
            #output = F.relu(inp) # not necessary
            output = F.leaky_relu(output)
            output, hidden = self.rnn(output, hidden)
            
        output = F.softmax(self.out(output[0]))
        self.hidden = hidden
        return output, hidden
    
    def init_hidden(self):
        if self.type == 'LSTM':
            # dim = [num_layers*num_directions, batch, hidden_size]
            h0 = torch.Tensor(self.num_layers, 1, self.hidden_dim)
            c0 = torch.Tensor(self.num_layers, 1, self.hidden_dim)

            # initializations
            h0, c0 = init.xavier_normal(h0), init.xavier_normal(c0)

            return Variable(h0), Variable(c0)

        elif self.type == 'GRU':
            h = torch.Tensor(self.num_layers, 1, self.hidden_size)
            h = init.xavier_normal(h)

            return Variable(h)



#learning_rate = 0.01
#encoder_optimizer = optim.Adam(encoder.parameters(), lr = learning_rate)
#decoder_optimizer = optim.Adam(decoder.parameters(), lr = learning_rate)
#teacher_forcing_ratio = 0.3
# create encoder outputs
# given some input sequence - input
def train(input_sequence, target_sequence, evaluator, max_seq_length=MAX_SEQ_LEN):
    """
    both input_sequence and target_sequence should be variables
    """
    input_length = input_sequence.size()[0]
    target_length = target_sequence.size()[0]
    encoder_outputs = Variable(torch.zeros(max_seq_length, encoder.hidden_size))
    inputs = Variable(torch.zeros(max_seq_length))
    inputs[:input_length] = inputs
    
    e = Evaluator()
    scores = []
    for i in range(num_interactions):
        
        # sequentially provide inputs to decoder
        for ei in range(input_length+i):
            encoder_output, encoder_hidden = encoder(inputs[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0][0]
        

        # init decoder hidden to encoder hidden
        decoder_hidden = encoder_hidden

        decoder_input = Variable(torch.LongTensor[[SOS]]) # starting token has to be SOS

        decoder_outputs = Variable(torch.zeros(max_length, decoder.hidden_size)) # set of output distributions
        decoder_sequence = Variable(torch.zeros(max_length, 1)) # max_length + 1 to account for [SOS_token]
        
        for di in range(target_length): # TODO: check with LU
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            decoder_outputs[i] = decoder_output # stores the "generator probability"
            
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            
            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_sequence[di,0] = ni
            if ni == EOS:
                decoder_sequence[di+1:,0] = ni
                break
        
        f = e.eval_init(target_sequence)
        pred_list = decoder_sequence.data.numpy()
        score, new_example = f(pred_list.tolist())
        
        encoder_hidden = decoder_hidden
        inputs[ei+1] = new_example
        scores.append(score)
        
        if i == num_interactions - 1 or score == 100000:
            J = 0
            for g in range(Ng):
                # generate an output sequence
                # that is, run through decoder network generating output and storing probabilities

                phi = Variable(torch.zeros(1, max_length), requires_grad=False)#state-action vector
                for di in range(target_length): # TODO: check with LU
                    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                    decoder_outputs[i] = decoder_output # stores the "generator probability"

                    topv, topi = decoder_output.data.topk(1)
                    ni = topi[0][0]

                    decoder_input = Variable(torch.LongTensor([[ni]]))
                    decoder_sequence[di,0] = ni
                    if ni == EOS:
                        decoder_sequence[di+1:,0] = ni
                        break
                for t in range(1,target_length):
                    phi[0, t-1] = state_action_function(decoder_sequence[:t], 
                                                  decoder_sequence[t], decoder_outputs)#decoder_outputs[t-1])

                for y in decoder_sequence:
                    J += decoder_outputs.mm(phi)[y]
            J = J/Ng
            J.backward()
            
        return J.data[0], scores


def gen_sequences(current_state_seq, policy, N=MONTE_CARLO_N, max_seq_len=MAX_SEQ_LEN):
    """
    Function that given the current state-sequence, and the policy, rolls
    out the sequence to produce N sequence of the required lenghth
    
    Arguments:
        1. current_state_seq (n, 1): (assuming no batches)
        2. policy: generator probability distributions
    """
    '''
    DON'T USE: 
    # seq-index, batch-index, vector_index
    sequences = torch.LongTensor(N, 1, max_seq_len)
    
    l = current_state_seq.size()[0]
    sequences[:, 0, :l] = current_state_seq.data.expand(N, 2)
    hidden = policy.hidden
    print(sequences)
    for n in range(N):
        for i in range(l, max_seq_len):
            output, hidden = policy.forward(torch.LongTensor(sequences[n,0,i-1]), hidden)
            
            output = torch.multinomial(torch.exp(out),1)
            sequences[n, 0, i] = output.data
    
    return sequences
    '''

    sequences = torch.LongTensor(N, 1, max_seq_len)
    
    l = current_state_seq.size()[0]
    # squeezing to make expansion easier
    cs = current_state_seq.data.squeeze(1)
    
    print(current_state_seq)
    #sequences[:, 0, :l] = current_state_seq.data.expand(N, l)
    sequences[:, 0, :l] = cs.expand(N, l)
    # print(current_state_seq.data.expand(N, 2))
    hidden = policy.hidden
    it = 0
    for n in range(N):
        for i in range(l, max_seq_len):
            it += 1
            #print("Sequences at iteration {}: {}".format(i, sequences[n,0, i-1]))
            
            output = torch.multinomial(policy[i,:], 1)
            #print("Iteration ", it)
            #print(output)
            sequences[n,0,i] = output.data.numpy()[0,0]
    return sequences


### NEW ###
def state_action_function(current_states, action, policy, N=5,  max_seq_length=10):
    """
    Arguments:
        1. current_states (LongTensor: Assume that the function takes in the current state as a LongTensor)
        2. action (LongTensor of size (1x1))
    """
    if current_states.size()[0] < max_seq_length-1:
        new_states = torch.LongTensor(current_states.size()[0]+1, 1)
        new_states[:current_states.size()[0], :] = current_states.data
        new_states[current_states.size()[0], :] = action
        sequences = gen_sequences(Variable(new_states, requires_grad=False), policy, N, max_seq_length)
        s = 0
        for n in range(N):
            print(n)
            s += discriminator(sequences[n,0, :])
        return s/N
    
    elif current_states.size()[0] == max_seq_length-1:
        new_states = torch.LongTensor(current_states.size()[0]+1, current_states.size()[1])
        new_states[:current_states.size()[0], :] = current_states
        new_states[current_states.size()[0], :] = action
        
        return discriminator(new_states)


# In[166]:

### OLD VERSION - DO NOT USE ###
def gen_sequences(current_state_seq, policy, N=MONTE_CARLO_N, max_seq_len=MAX_SEQ_LEN):
    """
    Function that given the current state-sequence, and the policy, rolls
    out the sequence to produce N sequence of the required lenghth
    
    Arguments:
        1. current_state_seq (n, 1): (assuming no batches)
        2. policy: Decoder RNN
    """
    '''
    DON'T USE: 
    # seq-index, batch-index, vector_index
    sequences = torch.LongTensor(N, 1, max_seq_len)
    
    l = current_state_seq.size()[0]
    sequences[:, 0, :l] = current_state_seq.data.expand(N, 2)
    hidden = policy.hidden
    print(sequences)
    for n in range(N):
        for i in range(l, max_seq_len):
            output, hidden = policy.forward(torch.LongTensor(sequences[n,0,i-1]), hidden)
            
            output = torch.multinomial(torch.exp(out),1)
            sequences[n, 0, i] = output.data
    
    return sequences
    '''

    sequences = torch.LongTensor(N, 1, max_seq_len)
    
    l = current_state_seq.size()[0]
    # squeezing to make expansion easier
    cs = current_state_seq.data.squeeze(1)
    
    print(current_state_seq)
    #sequences[:, 0, :l] = current_state_seq.data.expand(N, l)
    sequences[:, 0, :l] = cs.expand(N, l)
    # print(current_state_seq.data.expand(N, 2))
    hidden = policy.hidden
    it = 0
    for n in range(N):
        for i in range(l, max_seq_len):
            it += 1
            #print("Sequences at iteration {}: {}".format(i, sequences[n,0, i-1]))
            output, hidden = policy.forward(Variable(torch.LongTensor([sequences[n,0,i-1]]),requires_grad=False), hidden)

            output = torch.multinomial(torch.exp(output),1)
            
            
            #print("Iteration ", it)
            #print(output)
            sequences[n,0,i] = output.data.numpy()[0,0]
    return sequences

