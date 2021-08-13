import torch
import torch.nn as nn
import torch.nn.functional as f
import math
from misc_backup.model.model import RNNLM
import numpy as np


class InferenceMachine(nn.Module):
    def __init__(self, model_path, model_name, embed_size, hidden_size, num_layers, batch_size, test_iter, vocab_size,
                 rnn_type, logger, dictionary, device, path, drop_out, dropouth, dropouti, dropoute, wdrop,
                 tie_weights, number_of_top_scoring_words, seq_length):

        super(InferenceMachine, self).__init__()
        self.model = model_path + model_name
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.rnn_type = rnn_type
        self.device = device
        self.logger = logger
        self.drop_out = drop_out
        self.dropouth = dropouth
        self.dropouti = dropouti
        self.dropoute = dropoute
        self.wdrop = wdrop
        self.tie_weights = tie_weights
        self.path = path
        self.test_iter = test_iter
        self.vocab_size = vocab_size
        self.text_inv_vocab = dictionary['inv_vocab']
        self.words_to_keep = number_of_top_scoring_words
        self.seq_length = seq_length

        ### house keeping #####
        self.hidden_states = torch.zeros(self.seq_length + 1, self.hidden_size)
        self.cell_states = torch.zeros(self.seq_length + 1, self.hidden_size)
        self.top_words = torch.zeros(self.seq_length + 1, number_of_top_scoring_words)
        self.max_score_sequences = torch.zeros(number_of_top_scoring_words, self.seq_length)
        self.back_pointers = torch.zeros(number_of_top_scoring_words, self.seq_length)

    """
    The function creates an instance of RNN and loads the model. This model is then used to make inference. 
    """

    def reset_house_keepers(self, number_of_top_scoring_words):
        self.hidden_states = torch.zeros(self.seq_length + 1, self.hidden_size)
        self.cell_states = torch.zeros(self.seq_length + 1, self.hidden_size)
        self.top_words = torch.zeros(self.seq_length + 1, number_of_top_scoring_words)
        self.max_score_sequences = torch.zeros(number_of_top_scoring_words, self.seq_length)
        self.back_pointers = torch.zeros(number_of_top_scoring_words, self.seq_length)

    def load_model(self):
        model = RNNLM(self.rnn_type, self.vocab_size, self.embed_size, self.hidden_size, self.num_layers,
                           self.drop_out, self.dropouth, self.dropouti, self.dropoute, self.wdrop,
                           self.tie_weights).to(self.device)

        model_dict = torch.load(self.path + self.model, map_location=self.device)
        model.load_state_dict(model_dict, strict=True)
        self.logger.info("Model has been initialsed with correct weights")
        return model

    """
    Runs the given sentence through the model and create fuzzy words for the blanks 
    """
    def preprocess(self, model, inputs, targets, word_idx_before_blank_idx, batch_size=1):
        counter = 0  #count for sequnce
        hidden_state_index = 1  #index for saving hidden states. The first one is all zero
        flag = True  #for reshaping input initially

        states = model.reset_hidden_states(batch_size)

        for input_idx, target_idx in zip(inputs, targets):

            if flag:
                input = input_idx.reshape(1, 1)
                self.top_words[counter][0] = input
                flag = False

            # generate initial hidden state
            output, states = model(input, states)

            self.hidden_states[hidden_state_index] = states[0][0]
            self.cell_states[hidden_state_index] = states[0][1]

            if counter in word_idx_before_blank_idx:
                softmax_output, softmax_idx = torch.sort(f.softmax(output, dim=1)[0], descending=True)
                top_scoring_words_score = softmax_output[:self.words_to_keep]
                top_scoring_words_index = softmax_idx[0:self.words_to_keep]

                if target_idx[0] not in top_scoring_words_index.long():
                    top_scoring_words_score[-1] = softmax_output[target_idx[0]]
                    top_scoring_words_index[-1] = target_idx[0]

                self.top_words[hidden_state_index] = top_scoring_words_index
                output_idx = torch.abs((torch.dot(top_scoring_words_score, top_scoring_words_index.float()) // torch.sum(top_scoring_words_score)).long())
                input.fill_(output_idx)
            else:
                input.fill_(target_idx[0])
                #self.top_words[hidden_state_index][0] = target_idx[0]
                t = torch.zeros(self.words_to_keep)
                for i in range(self.words_to_keep):
                    t[i] = target_idx[0]
                self.top_words[hidden_state_index] = t #[target_idx[0] for _ in range(self.words_to_keep)]

            hidden_state_index += 1
            counter += 1

        # check the hidden state and yop words
        #np.savetxt(self.path + 'language_modelling/tests/hidden_states.csv', self.hidden_states.numpy(),delimiter=',')
        #np.savetxt(self.path + 'language_modelling/tests/top_words.csv', self.top_words.numpy(), delimiter=',')
        #self.logger.info("Preprocessing has been completed")

    def generate_factor(self, model, col):
        hidden_state = self.hidden_states[col-1]
        cell_state = self.cell_states[col-1]
        word_list = self.top_words[col-1]
        target_list = self.top_words[col]

        top_words_size = len(word_list)
        factor_table = torch.zeros(top_words_size, top_words_size)

        with torch.no_grad():
            for row, word in enumerate(word_list):
                if word != 0:
                    h = hidden_state.reshape(1, 1, self.hidden_size).to(self.device)
                    c = cell_state.reshape(1, 1, self.hidden_size).to(self.device)
                    output, states = model(word.long().reshape(1, 1).to(self.device), [(h,c)])
                    for i in range(top_words_size):
                        if target_list[i] != 0:
                            y = torch.index_select(-1 * f.log_softmax(output, dim=1),
                                                   1, target_list[i].long().to(self.device))
                            factor_table[row][i] = y
        return factor_table

    def generate_sentence(self, model):
        #a = torch.zeros([self.words_to_keep])
        self.max_score_sequences = torch.zeros(self.words_to_keep, self.seq_length)
        self.back_pointers = torch.zeros(self.words_to_keep, self.seq_length)

        for col in range(1, self.seq_length):  # this is column
            factor = self.generate_factor(model, col)
            for i in range(0, self.words_to_keep):
                messages = torch.zeros([self.words_to_keep])
                for j in range(self.words_to_keep):
                    messages[j] = self.max_score_sequences[j, col-1] + factor[j, i]
                self.max_score_sequences[i, col] = max(messages)
                #if not torch.equal(messages, a):
                self.back_pointers[i, col] = torch.argmax(messages)

        np.savetxt(self.path + 'language_modelling/tests/max_score_sequences.csv', self.max_score_sequences.numpy(),delimiter=',')
        np.savetxt(self.path + 'language_modelling/tests/back_pointers.csv', self.back_pointers.numpy(), delimiter=',')

        # back track sequence
        best_sequence = []
        best_sequence_score = float(max(self.max_score_sequences[:, -1]))
        best_sequence.append(int(torch.argmax(self.max_score_sequences[:, -1])))
        for col in range(self.seq_length - 2, -1, -1):
            t = self.back_pointers[int(best_sequence[-1]), col]
            best_sequence.append(int(t))
        return best_sequence[::-1], best_sequence_score

    def update_hidden_states(self, model, sentence):
        model.eval()
        with torch.no_grad():
            inputs = torch.tensor(sentence)
            batch_size = 1
            states = model.reset_hidden_states(batch_size)
            self.hidden_states[0] = states[0][0]
            self.cell_states[0] = states[0][1]
            c = 1
            for input_idx in inputs:
                input = input_idx.reshape(1, 1).to(self.device)
                output, states = model(input, states)
                self.hidden_states[c] = states[0][0]
                self.cell_states[c] = states[0][1]
                c += 1

    def fill_in_the_blank(self, save_path, mask_rate, number_of_blanks):
        file_name = self.path + save_path + "fill_in_the_blanks_dynamic.txt"

        ### load model ##
        rnn_model = self.load_model()

        total_blanks = 0
        number_of_correct_blanks = 0

        ### open file for outputs ###
        with open(file_name, 'w') as outf:
            with torch.no_grad():
                for step, batch in enumerate(self.test_iter):

                    if len(batch.text) != self.seq_length:
                        pass

                    inputs = batch.text.to(self.device)
                    targets = batch.target.to(self.device)

                    # Generate places where blanks are inserted
                    blank_idx = self.generate_blank_indexes(inputs, mask_rate, number_of_blanks)
                    word_idx_before_blank_idx = [i - 1 for i in blank_idx]

                    total_blanks += len(blank_idx)
                    input_sentence = ""
                    blank_sentence = ""

                    for i, input_idx in enumerate(inputs):
                        input_sentence += self.text_inv_vocab[input_idx] + " "
                        if i in blank_idx:
                            blank_sentence += '<blank>' + " "
                        else:
                            blank_sentence += self.text_inv_vocab[input_idx] + " "

                    outf.write(input_sentence)
                    outf.write('\n')
                    outf.write(blank_sentence)
                    outf.write('\n')

                    #############################---Preprocessing----###########################
                    self.reset_house_keepers(self.words_to_keep)
                    self.preprocess(rnn_model, inputs, targets, word_idx_before_blank_idx)

                    #############---generate sentence----########################################
                    best_sequence = None
                    sentence_score = 0
                    for _ in range(2):
                        best_sequence, sentence_score = self.generate_sentence(rnn_model)
                        # recalculate hidden states
                        self.update_hidden_states(rnn_model, best_sequence)

                    ############################################################################
                    output_sentence = ""
                    for i, word_idx in enumerate(best_sequence):
                        output_sentence += self.text_inv_vocab[self.top_words[i][word_idx].long()] + ' '

                    outf.write(output_sentence)
                    outf.write('\n')
                    outf.write("score " + str(sentence_score))
                    outf.write('\n')
                    outf.write("--------------------------------------------")
                    outf.write('\n')

                    a = output_sentence.split()
                    b = input_sentence.split()

                    for i in blank_idx:
                        try:
                            if a[i] == b[i]:
                                number_of_correct_blanks += 1
                        except IndexError:
                            pass

            outf.write(str(number_of_correct_blanks / total_blanks * 100))

    def generate_blank_indexes(self, sentence, mask_rate, no_of_blanks):
        length_of_sentence = len(sentence)
        number_of_words_to_mask = math.ceil((mask_rate / 100) * length_of_sentence)

        each_blank_length = []

        while each_blank_length == []:
            each_blank_length_list = np.random.multinomial(int(number_of_words_to_mask),
                                                           np.ones(no_of_blanks) / no_of_blanks,
                                                           size=int(number_of_words_to_mask))

            for i in each_blank_length_list:
                if 0 in i:
                    continue
                else:
                    each_blank_length = i
                    break

        ######  Generate indexes for blanks  ##########
        blank_list = []

        if no_of_blanks == 1:
            end_blank = np.ceil(
                np.random.uniform(each_blank_length[0], length_of_sentence - 1))
            start_blank = end_blank - each_blank_length[0] + 1
            blank_list.append((start_blank, end_blank))

        elif no_of_blanks == 2:
            for i in range(no_of_blanks):
                if i == 0:
                    end_blank = np.ceil(
                        np.random.uniform(each_blank_length[i], length_of_sentence - each_blank_length[i + 1] - 1))
                    start_blank = end_blank - each_blank_length[i] + 1
                    blank_list.append((start_blank, end_blank))
                else:
                    start_blank = np.ceil(
                        np.random.uniform(end_blank + 2, length_of_sentence - each_blank_length[i] + 1))
                    end_blank = start_blank + each_blank_length[i] - 1
                    blank_list.append((start_blank, end_blank))

        elif no_of_blanks == 3:
            for i in range(no_of_blanks):
                if i == 0:
                    end_blank = np.ceil(np.random.uniform(each_blank_length[i],
                                                          length_of_sentence - each_blank_length[i + 1] -
                                                          each_blank_length[i + 2]
                                                          - no_of_blanks))
                    start_blank = end_blank - each_blank_length[i] + 1
                    blank_list.append((start_blank, end_blank))
                elif i == 1:
                    end_blank = np.ceil(np.random.uniform(end_blank + each_blank_length[i] + 1,
                                                          length_of_sentence - each_blank_length[
                                                              i + 1] - no_of_blanks - 1))
                    start_blank = end_blank - each_blank_length[i] + 1
                    blank_list.append((start_blank, end_blank))
                else:
                    start_blank = np.ceil(
                        np.random.uniform(end_blank + 2, length_of_sentence - each_blank_length[i] + 1))
                    end_blank = start_blank + each_blank_length[i] - 1
                    blank_list.append((start_blank, end_blank))
        else:
            print("Implement other blanks")

        blank_index_list = []
        for t in blank_list:
            blank_index_list += range(int(t[0]), int(t[1]) + 1)

        return blank_index_list


