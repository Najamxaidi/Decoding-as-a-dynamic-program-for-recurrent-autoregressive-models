from model.model import RNNLM
import torch
import math
import numpy as np


class Decode:
    def __init__(self, model_path, model_name, test_iter, dictionary, logger,  device):
        self.saved_model = model_path + model_name
        self.device = device
        self.logger = logger
        self.test_iter = test_iter
        self.text_inv_vocab = dictionary['inv_vocab']

    def initialise_model(self, rnn_type, vocab_size, embed_size, hidden_size, num_layers, drop_out, tie_weights):
        model = RNNLM(rnn_type, vocab_size, embed_size, hidden_size, num_layers, drop_out, tie_weights).to(self.device)
        model_dict = torch.load(self.saved_model, map_location=self.device)
        model.load_state_dict(model_dict)
        return model

    @ staticmethod
    def lookup_words(x, vocab):
        x = [vocab[i] for i in x]
        return [str(t) for t in x]

    @staticmethod
    def generate_blank_indexes(sentence, mask_rate, no_of_blanks):
        length_of_sentence = len(sentence)
        number_of_words_to_mask = math.ceil((mask_rate / 100) * length_of_sentence)

        each_blank_length = []

        while each_blank_length == []:
            each_blank_length_list = np.random.multinomial(int(number_of_words_to_mask),
                                                           np.ones(no_of_blanks) / no_of_blanks,
                                                           size=int(number_of_words_to_mask))
            # print(each_blank_length_list)

            for i in each_blank_length_list:
                if 0 in i:
                    continue
                else:
                    each_blank_length = i
                    break

        ######  Generate indexes for blanks  ##########
        blank_list = []
        if no_of_blanks == 2:
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

    def fill_in_the_blank(self, model, save_path, save_file, mask_rate, number_of_blanks):
        model.eval()
        batch_size = 1
        total_blanks = 0
        number_of_correct_blanks = 0
        file_name = save_path + save_file

        with open(file_name, 'w') as outf:
            with torch.no_grad():
                for step, batch in enumerate(self.test_iter):
                    inputs = batch.text.to(self.device)
                    targets = batch.target.to(self.device)
                    states = model.reset_hidden_states(batch_size)

                    # Generate places where blanks are inserted
                    blank_idx = self.generate_blank_indexes(inputs, mask_rate, number_of_blanks)
                    word_idx_before_blank_idx = [i - 1 for i in blank_idx]
                    total_blanks += len(blank_idx)

                    #
                    # flag = True
                    #
                    # for i, input_idx in enumerate(inputs):
                    #     if i in blank_idx:
                    #         blank_sentence += '<blank>' + " "
                    #     else:
                    #         blank_sentence += self.text_inv_vocab[input_idx] + " "
                    #
                    # counter = 0
                    # sentence_score = 0
                    #
                    # for input_idx, target_idx in zip(inputs, targets):
                    #     input_sentence += self.text_inv_vocab[input_idx] + " "
                    #
                    #     if flag:
                    #         input = input_idx.reshape(1, 1)
                    #         flag = False
                    #
                    #     output, states = self.model(input, states)
                    #
                    #     if counter in word_idx_before_blank_idx:
                    #         word_weights = output.squeeze().div(1).exp().cpu()
                    #         output_idx = np.argmax(word_weights)
                    #         #output_idx = torch.multinomial(word_weights, 1)[0]
                    #         input.fill_(output_idx)
                    #         sentence_score += np.log(word_weights[output_idx])
                    #         output_sentence += self.text_inv_vocab[output_idx] + " "
                    #
                    #         if self.text_inv_vocab[output_idx] == self.text_inv_vocab[target_idx[0]]:
                    #             number_of_correct_blanks += 1
                    #     else:
                    #         input.fill_(target_idx[0])
                    #         word_weights = output.squeeze().div(1).exp().cpu()
                    #         sentence_score += np.log(word_weights[target_idx[0]])
                    #
                    #     counter += 1
                    src_sent = " ".join(self.lookup_words(inputs, vocab=self.text_inv_vocab))
                    trg_sent = " ".join(self.lookup_words(targets, vocab=self.text_inv_vocab))

                    input_sentence = "INPUT-" + str(step) + ': ' + src_sent
                    #blank_sentence = "BLANK-" + str(step) + ': '
                    target_sentence = "TARGET-" + str(step) + ': ' + trg_sent
                    #output_sentence = "OUTPUT-" + str(step) + ': '

                    outf.write(input_sentence)
                    outf.write('\n')
                    # outf.write(blank_sentence)
                    # outf.write('\n')
                    outf.write(target_sentence)
                    outf.write('\n')
                    # outf.write(output_sentence)
                    # outf.write('\n')
                    # outf.write("Sentence_score: " + str(sentence_score))
                    # outf.write('\n')
                    # outf.write("--------------------------------------------")
                    # outf.write('\n')

            accuracy = str(number_of_correct_blanks/total_blanks * 100)
            outf.write(str(accuracy))
        return accuracy



