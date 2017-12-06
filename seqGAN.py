
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


SOS = 0
EOS = 1


MAX_IN_SEQ_LEN = 150
MAX_OUT_SEQ_LEN = 40
MAX_INTERACTIONS = 10
MONTE_CARLO_N = 16
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

    def __init__(self, vocab_dim, embedding_dim, hidden_dim, num_layers, max_seq_len=MAX_IN_SEQ_LEN, rnn='LSTM'):

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
        # vocab_dim is the size of program output dimensions
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
            2. hidden : previous hidden layer state
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


# generate a program with max_length
# the argument "prefix" contains some program prefix: we produce the rest
# encoder intermediate states saved as desired, for rollout purposes
def gen_prog(start_hidden, start_input, decoder, prefix, hiddens, outputs, selected, choice_policy, max_length):
    # initialize decoder values
    decoder_hidden = start_hidden
    decoder_input = start_input

    for di in range(max_length - len(prefix)):
        # recall decoder hidden state for future rollout
        hiddens.append(decoder_hidden)
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        
        # select next decoder input
        selectv, selecti = choice_policy(decoder_output)
        next_i = selecti[0][0]
        
        decoder_input = Variable(torch.LongTensor([[next_i]]))

        # save decoder behaviors
        outputs.append(decoder_output)
        selected.append(next_i)
        prefix.append(next_i)

        if next_i == EOS:
            break
    else:
        # hit it with the fact that it's over---not the previous output
        hiddens.append(decoder_hidden)
        decoder_output, decoder_hidden = decoder(Variable(torch.LongTensor([[EOS]])), decoder_hidden)
        outputs.append(decoder_output)
        selected.append(EOS)
        prefix.append(EOS)

    return prefix, decoder_hidden

# rollout from some prefix
# rest_interactions is the number of interactions remaining
# scores contains the score values produced in all past interaction sessions
# prefix is the generated program symbols if we're in the middle of generating a program
# hiddens, output, selected will accumulate those values: don't use this for monte carlo
# inputs is the inputs we've accumulated in this rollout, for encoder use
# choice is the choice function: max or multinomial sampling, typically
def unroll(rest_interactions, f, scores_so_far, start_hidden, start_input, prefix, hiddens, outputs, inputs, selected, choice):
    if len(prefix) != 0:
        # in the middle of generating a program
        # finish up this round...

        # get generated program
        # recall decoder hidden state for next encoder round
        generate, encoder_hidden = gen_prog(start_hidden, start_input, decoder, prefix,
                    hiddens, outputs, selected,
                    choice, max_out_seq_len)

        # evaluator interaction
        score, new_example = f(generate)

        # correct program signalled
        if score == 100000:
            return scores_so_far
        
        scores_so_far.append(score)
        inputs += new_example
        #inputs[ei+1:ei+1+new_example.size()[0]] = new_example
        #input_length += new_example.size()[0]
        #lengths += [input_length]

        rest_interactions -= 1
    else:
        encoder_hidden = start_hidden

    for i in range(rest_interactions):
        
        # produce encoder output on current input sequence
        for ei in range(len(inputs)):
            encoder_output, encoder_hidden = encoder(inputs[ei], encoder_hidden)
            #encoder_outputs[ei] = encoder_output[0][0]
        
        # get generated program
        # recall decoder hidden state for next encoder round
        generate, encoder_hidden = gen_prog(
                    encoder_hidden, Variable(torch.LongTensor([[SOS]]), decoder, [],
                    hiddens, outputs, selected,
                    choice, max_out_seq_len)

        # evaluator interaction
        #pred_list = decoder_selected.data.numpy()
        score, new_example = f(generate) #f(pred_list.tolist())

        # correct program signalled
        if score == 100000:
            return scores_so_far
        
        scores_so_far.append(score)
        inputs += new_example
        #inputs[ei+1:ei+1+new_example.size()[0]] = new_example
        #input_length += new_example.size()[0]

    return scores_so_far

#learning_rate = 0.01
#encoder_optimizer = optim.Adam(encoder.parameters(), lr = learning_rate)
#decoder_optimizer = optim.Adam(decoder.parameters(), lr = learning_rate)
#teacher_forcing_ratio = 0.3
# create encoder outputs
# given some input sequence - input

# a single input sequence, a single target output sequence, and we run
def train_single(input_sequence, target_sequence, max_in_seq_length=MAX_IN_SEQ_LEN, max_out_seq_len=MAX_OUT_SEQ_LEN):
    """
    both input_sequence and target_sequence should be variables
    """
    #input_length = input_sequence.size()[0]
    #target_length = max_out_seq_len #target_sequence.size()[0]

    #encoder_outputs = Variable(torch.zeros(max_in_seq_length, encoder.hidden_size))
    #inputs = Variable(torch.zeros(max_seq_length))
    #inputs[:input_length] = input_sequence

    # initial encoder hidden input: zero
    encoder_hidden = Variable(torch.zeros(encoder.hidden_size))
    
    e = Evaluator()
    f = e.eval_init(target_sequence)

    # score prefix
    scores = []

    # save intermediate hidden states, output multinomials, selected outputs for future rollout
    rollout_hiddens = [] # rollout_hiddens[t] = Y_1:t-1
    rollout_outputs = [] # rollout_outputs[t] = G(y_t | Y_1:t-1)
    rollout_selected = [] # rollout_selected[t] = y_t
    rollout_inputs = input_sequence
    init_input_len = len(input_sequence)

    # perform a single full unroll
    final_sample_scores = unroll(MAX_INTERACTIONS, f, scores,
                                encoder_hidden, Variable(torch.LongTensor([[SOS]])), [],
                                rollout_hiddens, rollout_outputs, rollout_selected,
                                rollout_inputs,
                                lambda x: x.data.topk(1))
    final_sample_est = discriminator(final_sample_scores)

    assert len(rollout_hiddens) == len(rollout_outputs)
    assert len(rollout_outputs) == len(rollout_selected)

    # intermediate score array, intermediate prefix array
    #inter_scores = []
    inter_prefix = []
    interactions = 0 # count of interactions we've had so far in this rollout
    #tokens_already = 0 # if we're in the middle of an instance

    # perform MC rollout to estimate final RL reward, in the style of SeqGAN
    #J = rollout_outputs[-1] * final_sample_est # accounted for by last loop
    for t in range(len(rollout_hiddens)):

        inter_prefix += rollout_selected[t]

        if rollout_selected[t] == EOS:
            interactions += 1
            #tokens_already = 0
            #inter_scores.append(scores[interactions - 1])
            inter_prefix = []

        sample_est = 0.
        for g in range(MONTE_CARLO_N):
            # generate an output sequence
            # that is, run through decoder network generating output and storing probabilities
            # compute Q(rollout_hiddens[t], rollout_selected[t])

            sample_scores = unroll(MAX_INTERACTIONS - interactions, f, scores[:interactions],
                                    rollout_hiddens[t], rollout_outputs[t], inter_prefix[:],
                                    [], [], [],
                                    rollout_inputs[:init_input_len + interactions],
                                    lambda x: torch.multinomial(torch.exp(x), 1))
            sample_est += discriminator(sample_scores)

        sample_est = sample_est / MONTE_CARLO_N
        
        J += rollout_outputs[t] * sample_est

    J.backward()

    return J


