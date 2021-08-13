from model.model import RNNLM
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import numpy as np
import time


class Training:
    def __init__(self, train_iter, valid_iter, clip, lrate, save_path, save_model_name, logger, device):

        # iterators
        self.train_iter = train_iter
        self.valid_iter = valid_iter

        # some values needed throughout model
        self.clip = clip
        self.lrate = lrate

        # misc
        self.logger = logger
        self.device = device
        self.save_path = save_path
        self.save_model_name = save_model_name

        # house keeping
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.states = None
        self.old_perplexity = None

    def initialise_model(self, rnn_type, vocab_size, embed_size, hidden_size, num_layers, drop_out, tie_weights,
                         optimizer):
        model = RNNLM(rnn_type, vocab_size, embed_size, hidden_size, num_layers, drop_out, tie_weights).to(self.device)

        criterion = nn.CrossEntropyLoss().to(self.device)

        if optimizer == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lrate)
        else:
            self.logger.info('The optimiser has not been implemented. The default adam optimiser will be used')
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lrate)

        params = list(model.parameters())
        total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.logger.info('Model has been initialised')
        self.logger.info('Optimiser has been created')
        self.logger.info('Total parameters in the model: ' + str(total_params))

    @ staticmethod
    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def detach(self, states):
        if isinstance(states, torch.Tensor):
            return states.detach()
        else:
            return tuple(self.detach(v) for v in states)

    def step(self, batch, train):
        inputs = batch.text.to(self.device)
        targets = batch.target.to(self.device)
        # Forward pass
        # states need to be detached because we don't need the gradient for them
        self.states = self.detach(self.states)
        outputs, self.states = self.model(inputs, self.states)
        loss = self.criterion(outputs, targets.reshape(-1))
        # Backward and optimize
        if train:
            self.model.zero_grad()
            # accumulates gradients
            loss.backward()
            # avoid exploding gradients
            clip_grad_norm_(self.model.parameters(), self.clip)
            for p in self.model.parameters():
                p.data.add_(-1 * self.lrate, p.grad.data)
            # update parameters
            self.optimizer.step()
        return loss.item()

    def validate_model(self, batch_size):
        self.model.eval()
        dev_loss = 0
        total_sequence_length = 0
        self.states = self.model.reset_hidden_states(batch_size)
        with torch.no_grad():
            for step, batch in enumerate(self.valid_iter):
                dev_loss += len(batch.text) * self.step(batch, train=False)
                total_sequence_length += len(batch.text)
        return dev_loss/total_sequence_length

    def save_model(self, new_perplexity, save_path, name):
        if self.old_perplexity is None:
            torch.save(self.model.state_dict(), save_path + name)
            self.old_perplexity = new_perplexity
        elif new_perplexity < self.old_perplexity:
            torch.save(self.model.state_dict(), save_path + name)
            self.old_perplexity = new_perplexity
        else:
            self.logger.info("previous model is better")

    def train_model(self, batch_size, num_epochs, monitor_loss, checkpoint):

        for epoch in range(num_epochs):
            start_time = time.time()
            self.model.train()
            train_loss = 0
            # Set initial hidden and cell states
            self.states = self.model.reset_hidden_states(batch_size)

            for step, batch in enumerate(self.train_iter):
                train_loss += self.step(batch, train=True)
                if step % monitor_loss == 0:
                    if step != 0:
                        curr_loss = train_loss / monitor_loss      #step
                    else:
                        curr_loss = train_loss
                    self.logger.info('Epoch [{}/{}], Batch[{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}'
                                     .format(epoch + 1, num_epochs, step, len(self.train_iter), curr_loss,
                                             np.exp(curr_loss)))
                    train_loss = 0

                if step != 0 and step % checkpoint == 0:
                    dev_loss = self.validate_model(batch_size)
                    new_perplexity = np.exp(dev_loss)
                    self.logger.info('Perplexity on DEV set after batch [{}/{}] in epoch [{}] is {:5.2f} '.format(step, len(self.train_iter), epoch + 1, new_perplexity))
                    self.save_model(new_perplexity, self.save_path, self.save_model_name)
                    self.model.train()

            end_time = time.time()
            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)
            dev_loss = self.validate_model()
            new_perplexity = np.exp(dev_loss)
            self.logger.info('After complete epoch {} | Time: {}m {}s '.format(epoch + 1, epoch_mins, epoch_secs))
            self.logger.info('Perplexity on DEV set after complete epoch: {:5.2f} '.format(new_perplexity))
            self.save_model(new_perplexity, self.save_path, self.save_model_name)
