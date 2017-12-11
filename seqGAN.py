
# coding: utf-8

from eval import Evaluator

import numpy as np
import matplotlib.pyplot as plt
import itertools
import random
import time
import sys

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

e = Evaluator()

torch.cuda.device(0) # let's go

# precompute input tokens
tokens = [torch.LongTensor([[x]]).cuda() for x in range(22)]


MAX_IN_SEQ_LEN = 150
MAX_OUT_SEQ_LEN = 6
MAX_INTERACTIONS = 6
MONTE_CARLO_N = 30
EXP_SUBSAMPLE = 1
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

    TODO:
        1. Make batching neater
        2. Make variable-sequence length easier
        3. Implement Drop-Out (***) See Srivastava et al. 
                            http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf
        4. Make Bi-Directional LSTM
    """

    def __init__(self, vocab_dim, embedding_dim, hidden_dim, num_layers, rnn='LSTM'):

        super(Encoder, self).__init__() # blindly do this

        # make initializations
        self.vocab_dim = vocab_dim # size of input-output space (equiv. to vocab
                                                            # for speech tagging)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

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
            embedded = self.embedding(inputs.squeeze())\
                                .view(inputs.size()[0],inputs.size()[1],-1)
            output, hidden = self.rnn(embedded, hidden)

        elif variable_length==True:
            seq_len = inputs.size()[0]
            assert(input_lengths!=None)
            embedded = self.embedding(inputs).view(seq_len, batch_size,-1)
            
            packed_inputs = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
            output, hidden = self.rnn(packed_inputs, hidden)

        return output, hidden

    def init_hidden(self, batches=1):
        r"""
        Initialized hidden vector/cell-states. Given that weight initializations are crucial, we experiment with
        the Xavier Normal Initialization/Glorot Initialization.

        see: 1. http://pytorch.org/docs/0.1.12/_modules/torch/nn/init.html
             2. "Understanding the difficulty of training deep feedforward neural networks" 
                                                                                - Glorot, X. & Bengio, Y. (2010)
        """
        if self.type == 'LSTM':
            # dim = [num_layers*num_directions, batch, hidden_size]
            h0 = torch.Tensor(self.num_layers, batches, self.hidden_dim).cuda()
            c0 = torch.Tensor(self.num_layers, batches, self.hidden_dim).cuda()

            # initializations
            h0, c0 = init.xavier_normal(h0), init.xavier_normal(c0)

            return Variable(h0), Variable(c0)

        elif self.type == 'GRU':
            h = torch.Tensor(self.num_layers, batches, self.hidden_size)
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
        self.softmax = nn.LogSoftmax()
        
    def forward(self, inputs, hidden):
        r"""
        Creates forward graph.
        
        Arguments:
            1. input (1,1): if no teacher_forcing, this corresponds to the last output of the RNN
            2. hidden : previous hidden layer state
        """
        embedded = self.embedding(inputs.view(1, -1))\
                            .view(1, -1, self.hidden_dim)
        #for i in range(self.num_layers):
        #    output = F.leaky_relu(output)
        #    output, hidden = self.rnn(output, hidden)
        output, hidden = self.rnn(embedded, hidden)
        output = self.softmax(self.out(output[0]))

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

    next_i = start_input.data.cpu().numpy()[0][0]
    for di in range(max_length - len(prefix)):
        # recall decoder hidden state for future rollout
        hiddens.append(decoder_hidden)
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        
        # select next decoder input
        next_i = choice_policy(decoder_output)
        
        decoder_input = Variable(tokens[next_i], requires_grad=False)

        # save decoder behaviors
        outputs.append(decoder_output)
        selected.append(next_i)
        prefix.append(next_i)

        if next_i == SOS:
            break
    else:
        hiddens.append(decoder_hidden)
        decoder_output, decoder_hidden = decoder(Variable(tokens[next_i], requires_grad=False),
                                                    decoder_hidden)

        outputs.append(decoder_output)
        selected.append(SOS)
        prefix.append(SOS)

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
        inlens.append(newlen)
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
                    Variable(tokens[SOS], requires_grad=False).cuda(),
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
        inlens.append(newlen)
        inputs[curlen:newlen] = new_example

    return scores_so_far

def samp(x):
    v = torch.multinomial(torch.exp(x), 1).data.cpu().numpy()[0][0]
    return v

# a single input sequence, a single target output sequence, and we run
def train_single(encoder, decoder, input_sequence, f, max_in_seq_length=MAX_IN_SEQ_LEN, max_out_seq_len=MAX_OUT_SEQ_LEN):
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
    inputs[:len(input_sequence)] = torch.cat([tokens[x] for x in input_sequence], dim = 0)

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
                                encoder_hidden, Variable(tokens[SOS], requires_grad=False), [],
                                rollout_hiddens, rollout_outputs, rollout_selected,
                                rollout_inlens, rollout_inputs,
                                samp) #lambda x: x.data.topk(1)[1][0][0])
    final_sample_est = discriminator(final_sample_scores)

    print("sample score %s" % str(final_sample_est))
    print(rollout_selected)

    assert len(rollout_hiddens) == len(rollout_outputs)
    assert len(rollout_outputs) == len(rollout_selected)


    # perform MC rollout to estimate final RL reward, in the style of SeqGAN

    J = 0.

    # construct session prefixes
    prefixes = []

    # intermediate score array, intermediate prefix array
    interactions = 0 # count of interactions we've had so far in this rollout
    cur_prefix = []
    for t in range(len(rollout_hiddens)):
        prefixes.append((interactions, cur_prefix))

        cur_prefix.append(rollout_selected[t])
        if rollout_selected[t] == SOS:
            interactions += 1
            cur_prefix = []

    for t in range(len(rollout_hiddens)):
        interactions = prefixes[t][0]

        # re-prepare input slice
        mc_inlen = rollout_inlens[interactions]
        mc_inputs = torch.cat([rollout_inputs[:mc_inlen],
                               long_zeros_in[mc_inlen:]], dim=0)
        print("carlo %s/%s" % (str(t), str(len(rollout_hiddens))))

        distro = torch.exp(rollout_outputs[t])
        select = torch.multinomial(distro, EXP_SUBSAMPLE, replacement=False).data.cpu().numpy()[0]
        tot_est = 0.
        tot_density = 0.
        for e in range(EXP_SUBSAMPLE):

            sample_est = 0.
            for g in range(MONTE_CARLO_N):
                # compute Q(rollout_hiddens[t], rollout_selected[t])
                sample_scores = unroll(encoder, decoder, MAX_INTERACTIONS - interactions, max_out_seq_len,
                                        f, scores[:interactions],
                                        rollout_hiddens[t], Variable(tokens[select[e]], requires_grad=False),
                                        prefixes[t][1], [], [], [],
                                        [mc_inlen], mc_inputs,
                                        samp)

                sample_est += discriminator(sample_scores)
                #print "    DEBUG %s %s" % (str(g), str(e))
                #print rollout_outputs[t][0, select[e]]
                #print discriminator(sample_scores)
                #print density

            density = distro[0][select[e]].data.cpu().numpy()[0].item() # yes, we screen the gradient

            tot_est += density * rollout_outputs[t][0, select[e]] * sample_est / MONTE_CARLO_N
            tot_density += density

        J -= tot_est / tot_density
        #print "Jvalue %s" % str(t)
        #print J
        #print rollout_outputs[t][0][select]
        #print sample_est

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

    return J, rollout_outputs, rollout_hiddens

def discriminator(scores):
    return sum(scores) + 50 * (MAX_INTERACTIONS - len(scores))


encoder = Encoder(22, 128, 512, 3).cuda()
decoder = Decoder(14, 512, num_layers=3).cuda()

learning_rate = 0.001
#teacher_forcing_ratio = 0.3
# create encoder outputs
# given some input sequence - input


# load data
data = []

dat = open(sys.argv[1])
for l in dat:
    p = l.split()

    targ_prog = p[0]
    targ_prog_f = e.eval_init(targ_prog)

    in_data = [int(p[1][2*b:2*b+2], base=16) for b in range(256)]

    data.append((targ_prog, in_data, targ_prog_f))

dat.close()


# shuffled lists for data sampling
in_sample = range(256)
data_sample = range(len(data))


# PRE-TRAIN
NSAMPLES=5

encoder_optimizer = optim.Adam(encoder.parameters(), lr = learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr = learning_rate)

pre_loss = torch.nn.NLLLoss(ignore_index=14)

PRE_BATCHSIZE=300
for epoch in range(200):
    print("PRE EPOCH %s" % str(epoch + 1))

    # zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # prepare data for epoch
    epoch_Xs = []
    epoch_Yi = []
    epoch_Ys = []
    lookup = {'@': 1,         'A': 2, 'B': 3, 'C': 4, 'D': 5, 'E': 6,
              'F': 7, 'G': 8, 'H': 9, 'I':10, 'J':11, 'K':12, 'L': 13}
    for d in range(len(data)):
        random.shuffle(in_sample)
        inseq = [tokens[SOS]]
        for i in range(NSAMPLES):
            samp_in = in_sample[i]
            samp_out = data[data_sample[d]][1][samp_in]
            inseq.extend([tokens[samp_in /16+2], tokens[samp_in %16+2], tokens[18],
                          tokens[samp_out/16+2], tokens[samp_out%16+2], tokens[19]])
        inseq.append(tokens[EOS])
        epoch_Xs.append(torch.cat(inseq, dim=0))

        outseq = [tokens[lookup[c]] for c in data[data_sample[d]][0]]
        epoch_Yi.append(torch.cat([tokens[SOS]] + outseq + [tokens[0]] * (MAX_OUT_SEQ_LEN - len(outseq)), dim=0))
        epoch_Ys.append(torch.cat(outseq + [tokens[SOS]] + [tokens[14]] * (MAX_OUT_SEQ_LEN - len(outseq)), dim=0))

    epoch_Xs = torch.stack(epoch_Xs, dim=1)
    epoch_Yi = torch.stack(epoch_Yi, dim=1)
    epoch_Ys = torch.stack(epoch_Ys, dim=1)

    # model forward compute
    loss = 0.
    for b in range(epoch_Xs.size()[1] / PRE_BATCHSIZE + 1):
        start, end = b * PRE_BATCHSIZE, min((b + 1) * PRE_BATCHSIZE, epoch_Xs.size()[1])
        bsize = end - start

        # 1. initialize encoder hidden state
        encoder_hidden = encoder.init_hidden(batches=bsize)

        # 2. encoder forward compute
        _, decoder_hidden = encoder(Variable(epoch_Xs[:, start:end, :], requires_grad=False), encoder_hidden)
        #decoder_input = Variable(tokens[SOS].repeat(1, bsize, 1))

        # 3. decoder recurrent loop
        outputs = []
        for i in range(MAX_OUT_SEQ_LEN + 1):
            decoder_output, decoder_hidden = decoder(Variable(epoch_Yi[i, start:end ,:], requires_grad=False), decoder_hidden)

            outputs.append(decoder_output)

            # TODO implement teacher forcing
            # sample next input
            #decoder_input = torch.multinomial(torch.exp(decoder_output), 1).view(1, -1, 1)

        # 4. finish off outputs
        outputs = torch.stack(outputs, dim=0)

        loss += sum([pre_loss(outputs[:, bi, :], Variable(epoch_Ys[:, start+bi, 0], requires_grad=False)) for bi in range(bsize)])

    loss.backward()
    print "    loss %s" % str(loss.data.cpu().numpy()[0])

    encoder_optimizer.step()
    decoder_optimizer.step()

# seqGAN training step
RL_BATCHSIZE=4
for epoch in range(20):
    print("RL EPOCH %s" % str(epoch + 1))

    # zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # sample data
    random.shuffle(data_sample)

    J = 0.
    for d in range(RL_BATCHSIZE):
        print("RL EPOCH %s DATA %s/%s" % (str(epoch + 1), str(d + 1), RL_BATCHSIZE))
        print data[data_sample[d]][0]

        # prepare inputs
        random.shuffle(in_sample) # pick which I/O pairs to provide
        inseq = [SOS]
        for i in range(NSAMPLES):
            samp_in = in_sample[i]
            samp_out = data[data_sample[d]][1][samp_in]
            inseq.extend([samp_in /16+2,samp_in %16+2, 18,
                          samp_out/16+2,samp_out%16+2, 19])
        inseq.append(EOS)
        print "  DATA %s" % str(inseq)

        # get rewards on each input
        J_single, outs, hids = train_single(encoder, decoder, inseq, data[data_sample[d]][2])

        print("  PUNISH %s" % str(J_single.data.cpu().numpy()[0]))

        J += J_single

    # grad descent
    J.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

