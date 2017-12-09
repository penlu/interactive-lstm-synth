
# coding: utf-8

from eval import Evaluator

import numpy as np
import matplotlib.pyplot as plt
import itertools
import time

import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init

import torch.multiprocessing as mp

torch.manual_seed(1)

torch.cuda.device(0) # let's go

e = Evaluator()

SOS = 0
EOS = 1

# preconstruct token table

dec_tokens = [Variable(torch.from_numpy(np.array([[i]])).cuda(), requires_grad=False) for i in range(14)]

MAX_IN_SEQ_LEN = 150
MAX_OUT_SEQ_LEN = 24
MAX_INTERACTIONS = 4
MONTE_CARLO_N = 2
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
            self.rnn = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers, dropout=0.3)
        #elif rnn=='GRU':
            #self.rnn = nn.GRU(self.embedding_dim, self.hidden_dim, self.num_layers, dropout=0.3)

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
            #seq_len = inputs.size()[0]
            embedded = self.embedding(inputs.view(inputs.size()[0], 1)).view(inputs.size()[0],1,-1)
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
            h0 = torch.Tensor(self.num_layers, 1, self.hidden_dim).cuda()
            c0 = torch.Tensor(self.num_layers, 1, self.hidden_dim).cuda()

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
            self.rnn = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers, dropout=0.3)
        #elif self.type=='GRU':
        #    self.rnn = nn.GRU(self.hidden_dim, self.hidden_dim, self.num_layers)
        self.out = nn.Linear(hidden_dim, vocab_dim)
        
    def forward(self, inputs, hidden):
        r"""
        Creates forward graph.
        
        Arguments:
            1. input (1,1): if no teacher_forcing, this corresponds to the last output of the RNN
            2. hidden : previous hidden layer state
        """
        embedding = self.embedding(inputs).view(1,1,-1)
        #for i in range(self.num_layers):
        #    output = F.leaky_relu(output)
        #    output, hidden = self.rnn(output, hidden)
        output, hidden = self.rnn(embedding, hidden)
        output = F.softmax(self.out(output[0]), dim=1)

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
        next_i = choice_policy(decoder_output)
        
        decoder_input = dec_tokens[next_i]

        # save decoder behaviors
        outputs.append(decoder_output)
        selected.append(next_i)
        prefix.append(next_i)

        if next_i == EOS:
            break
    else:
        # hit it with the fact that it's over---not the previous output
        hiddens.append(decoder_hidden)
        decoder_output, decoder_hidden = decoder(dec_tokens[EOS], decoder_hidden)

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
def unroll(encoder, decoder, rest_interactions, max_out_seq_len, f, scores_so_far, start_hidden, start_input, prefix, hiddens, outputs, selected, inlens, inputs, choice):
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

        # update inputs
        curlen = inlens[-1]
        newlen = curlen + new_example.size()[0]
        inlens += [newlen]
        inputs[curlen:newlen] = new_example

        rest_interactions -= 1
    else:
        encoder_hidden = start_hidden

    for i in range(rest_interactions):
        
        # produce encoder output on current input sequence
        encoder_output, encoder_hidden = encoder(Variable(inputs[:inlens[-1]]), encoder_hidden)

        # get generated program
        # recall decoder hidden state for next encoder round
        generate, encoder_hidden = gen_prog(
                    encoder_hidden,
                    dec_tokens[SOS],
                    decoder, [],
                    hiddens, outputs, selected,
                    choice, max_out_seq_len)

        # evaluator interaction
        #pred_list = decoder_selected.data.numpy()
        score, new_example = f(generate) #f(pred_list.tolist())

        # correct program signalled
        if score == 100000:
            return scores_so_far
        
        scores_so_far.append(score)

        # update inputs
        curlen = inlens[-1]
        newlen = curlen + new_example.size()[0]
        inlens += [newlen]
        inputs[curlen:newlen] = new_example

    return scores_so_far

# init target
target_sequence = "BCDCDCDC"
f = e.eval_init(target_sequence)

def q_est(prefix):
    global f
    interactions = prefix[0]
    pre = prefix[1]
    rollout_inlens = prefix[2]
    rollout_inputs = prefix[3]
    rollout_hiddens = prefix[4]
    rollout_output = prefix[5]
    long_zeros_in = prefix[6]
    t = prefix[7]
    scores = prefix[8]
    max_out_seq_len = prefix[9]

    # re-prepare input slice
    mc_inlen = rollout_inlens[interactions]
    mc_inputs = torch.cat([rollout_inputs[:mc_inlen],
                           long_zeros_in[mc_inlen:]], dim=0)
    sample_est = 0.
    print("carlo %s/%s" % (str(t), str(len(rollout_hiddens))))
    for g in range(MONTE_CARLO_N):
        # compute Q(rollout_hiddens[t], rollout_selected[t])

        select = samp(rollout_output)

        sample_scores = unroll(encoder, decoder, MAX_INTERACTIONS - interactions, max_out_seq_len,
                                f, scores[:interactions],
                                rollout_hiddens[t], dec_tokens[select],
                                pre, [], [], [],
                                [mc_inlen], mc_inputs,
                                samp)
        sample_est += discriminator(sample_scores)

    sample_est = sample_est / MONTE_CARLO_N

    return - torch.log(rollout_output[0][select]) * sample_est

def samp(x):
    v = torch.multinomial(x, 1).data.cpu().numpy()[0][0]
    return v

# a single input sequence, a single target output sequence, and we run
def train_single(encoder, decoder, input_sequence, max_in_seq_length=MAX_IN_SEQ_LEN, max_out_seq_len=MAX_OUT_SEQ_LEN):
    """
    both input_sequence and target_sequence should be variables
    """
    #encoder_outputs = Variable(torch.zeros(max_in_seq_length, encoder.hidden_size))
    #inputs = Variable(torch.zeros(max_in_seq_length))
    #inputs[:len(input_sequence)] = input_sequence



    # initialize encoder hidden state
    encoder_hidden = encoder.init_hidden()

    # reformat input
    long_zeros_in = torch.LongTensor(max_in_seq_length, 1).zero_().cuda()
    inputs = long_zeros_in.clone()
    inputs[:len(input_sequence)] = torch.stack([torch.LongTensor([x]).cuda() for x in input_sequence], dim = 0)

    # score prefix
    scores = []

    # save intermediate hidden states, output multinomials, selected outputs for future rollout
    rollout_hiddens = [] # rollout_hiddens[t] = Y_1:t-1
    rollout_outputs = [] # rollout_outputs[t] = G(y_t | Y_1:t-1)
    rollout_selected = [] # rollout_selected[t] = y_t
    rollout_inputs = inputs #input_sequence[:] # stuff what goes to the encoder
    rollout_inlens = [len(input_sequence)]

    # perform a single full unroll
    final_sample_scores = unroll(encoder, decoder, MAX_INTERACTIONS, max_out_seq_len,
                                f, scores,
                                encoder_hidden, dec_tokens[SOS], [],
                                rollout_hiddens, rollout_outputs, rollout_selected,
                                rollout_inlens, rollout_inputs,
                                samp) #lambda x: x.data.topk(1)[1][0][0])
    final_sample_est = discriminator(final_sample_scores)

    print("sample score %s" % str(final_sample_est))
    print(rollout_selected)

    assert len(rollout_hiddens) == len(rollout_outputs)
    assert len(rollout_outputs) == len(rollout_selected)


    # perform MC rollout to estimate final RL reward, in the style of SeqGAN

    # construct session prefixes
    prefixes = []

    # intermediate score array, intermediate prefix array
    interactions = 0 # count of interactions we've had so far in this rollout
    cur_prefix = []
    for t in range(len(rollout_hiddens)):
        prefixes.append((interactions, cur_prefix, rollout_inlens, rollout_inputs, rollout_hiddens, rollout_outputs[t], long_zeros_in, t, scores, max_out_seq_len))

        cur_prefix += [rollout_selected[t]]
        if rollout_selected[t] == EOS:
            interactions += 1
            cur_prefix = []

    #J = 0.
    #for t in range(len(rollout_hiddens)):
    #def q_est(t):
    #    interactions = prefixes[t][0]

    #    # re-prepare input slice
    #    mc_inlen = rollout_inlens[interactions]
    #    mc_inputs = torch.cat([rollout_inputs[:mc_inlen],
    #                           long_zeros_in[mc_inlen:]], dim=0)
    #    sample_est = 0.
    #    print("carlo %s/%s" % (str(t), str(len(rollout_hiddens))))
    #    for g in range(MONTE_CARLO_N):
    #        # compute Q(rollout_hiddens[t], rollout_selected[t])

    #        select = samp(rollout_outputs[t])

    #        sample_scores = unroll(encoder, decoder, MAX_INTERACTIONS - interactions, max_out_seq_len,
    #                                f, scores[:interactions],
    #                                rollout_hiddens[t], dec_tokens[select],
    #                                prefixes[t][1], [], [], [],
    #                                [mc_inlen], mc_inputs,
    #                                samp)
    #        sample_est += discriminator(sample_scores)

    #    sample_est = sample_est / MONTE_CARLO_N

        #print "Jvalue %s" % str(t)
        #print J
        #print rollout_outputs[t][0][select]
        #print sample_est

    #    return - torch.log(rollout_outputs[t][0][select]) * sample_est
        #J -= torch.log(rollout_outputs[t][0][select]) * sample_est

    J = sum(pool.imap_unordered(q_est, prefixes, chunksize=2))

    def clamp(message):
        def _internal(x):
            #print message
            m = max(torch.norm(x.data) / 100000, 1)
            #print x / m
            return x / m
        return _internal

    for i in range(len(rollout_hiddens)):
        rollout_outputs[i].register_hook(clamp("output %s grad" % str(i)))
        rollout_hiddens[i][0].register_hook(clamp("hidden h %s grad" % str(i)))
        rollout_hiddens[i][1].register_hook(clamp("hidden c %s grad" % str(i)))

    J.backward()

    return J, rollout_outputs, rollout_hiddens

def discriminator(scores):
    return sum(scores)


encoder = Encoder(22, 100, 100, 3).cuda()
decoder = Decoder(14, 100, num_layers=3).cuda()

learning_rate = 0.001
#teacher_forcing_ratio = 0.3
# create encoder outputs
# given some input sequence - input


# PRE-TRAIN
encoder_optimizer = optim.Adam(encoder.parameters(), lr = learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr = learning_rate)

# TODO


# encode single target input for seqGAN

inseq = [SOS, 2, 2, 18, 2, 7, 19,
              2, 3, 18, 2, 15, 19,
              2, 4, 18, 3, 7, 19,
              2, 5, 19, 3, 15, 19, EOS]

# seqGAN training step
if __name__ == "__main__":
    mp.set_start_method('forkserver')

    pool = mp.Pool(2)

    for i in range(4):
        print("EPOCH %s" % str(i))
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        j, outs, hids = train_single(encoder, decoder, inseq)
        print("reward: %s" % str(j))
        #print "outputs start"
        #for i in range(len(outs)):
        #  print "outputs %s" % str(i)
        #  print outs[i]
        #print "hiddens h start"
        #for i in range(len(hids)):
        #  print "hiddens h %s" % str(i)
        #  print hids[i][0]
        #print "hiddens c start"
        #for i in range(len(outs)):
        #  print "hiddens c %s" % str(i)
        #  print hids[i][1]
        #print "encoder params and grads"
        #for i in encoder.parameters():
        #  print i
        #  print i.grad
        #print "decoder params and grads"
        #for i in decoder.parameters():
        #  print i
        #  print i.grad
        #print decoder.parameters().next()
        #print decoder.parameters().next().grad
        encoder_optimizer.step()
        decoder_optimizer.step()

