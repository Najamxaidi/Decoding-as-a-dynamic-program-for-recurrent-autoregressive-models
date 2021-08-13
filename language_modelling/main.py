from data_utils import DataPreprocessor
from model.train import Training
from model.decode import Decode
# from model.inference_machine import InferenceMachine

import torch
import argparse
import logging
import json


def read_config(file_path):
    """Read JSON config."""
    json_object = json.load(open(file_path, 'r'))
    return json_object


def hyperparam_string(config):
    exp_name = ''
    exp_name += 'model_%s_' % (config['management']['task'])
    if config['training']['train']:
        exp_name += "_training"
    else:
        exp_name += "_inference"
    return exp_name


if __name__ == "__main__":
    try:
        if __name__ == '__main__':

            # Get the path to the json file according to the operating system
            # The json file contains the configuration of the model
            file_handle = open('paths_to_jason.txt')
            paths = file_handle.readlines()

            # Open the json file and set initial parameters
            parser = argparse.ArgumentParser()
            parser.add_argument("--machine_type", help="0: mac, 1: linux, 2: M3",  type=int)
            args = parser.parse_args()
            machine_type = args.machine_type

            if machine_type == 0:
                path_to_jason_file = paths[0].strip()
            elif machine_type == 1:
                path_to_jason_file = paths[1].strip()
            else:
                path_to_jason_file = paths[2].strip()

            config = read_config(path_to_jason_file)
            experiment_name = hyperparam_string(config)

            if machine_type == 0:
                path = config['paths']['Mac']
            elif machine_type == 1:
                path = config['paths']['Linux']
            else:
                path = config['paths']['m3']

            # Set-up the logger
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                                filename='log/%s' % experiment_name, filemode='w')
            # define a new Handler to log to console as well
            console = logging.StreamHandler()
            # optional, set the logging level
            console.setLevel(logging.INFO)
            # set a format which is the same for console use
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            # tell the handler to use this format
            console.setFormatter(formatter)
            # add the handler to the root logger
            logging.getLogger('').addHandler(console)

            # set seed and cuda for reproducability #
            torch.manual_seed(config['management']['seed'])
            if torch.cuda.is_available():
                if not config['management']['cuda']:
                    device = torch.device("cpu")
                    logging.warning("WARNING: You have a CUDA device, so you should probably run with cuda")
                else:
                    device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            # Set-up data iterator
            logging.info('[!] preparing dataset...')
            obj = DataPreprocessor(device, logging)

            # This is a pre-processing step and pre-process data as is required by the model
            # This needs not to be done every time
            preprocess = config['management']['preprocess']
            if preprocess:
                obj.preprocess(path + config['data']['preprocess']['raw_data_path'],
                               config['data']['preprocess']['train_file_name'],
                               config['data']['preprocess']['dev_file_name'],
                               config['data']['preprocess']['test_file_name'],
                               'train', 'dev', 'test',
                               path + config['data']['preprocess']['save_dir_for_model_data'],
                               config['data']['general']['min_freq'])

            train_dataset, val_dataset, test_dataset, vocabs = obj.load_data('train', 'dev', 'test',
                                                                             path + config['data']['preprocess']
                                                                             ['save_dir_for_model_data'])
            train_iter, val_iter, test_iter = obj.return_iterators(train_dataset, val_dataset, test_dataset,
                                                                   config['data']['general']['batch_size'],
                                                                   config['data']['general']['bptt'],
                                                                   config['data']['general']['min_freq'])
            vocab_size = len(vocabs['vocab'])

            # log this info so far
            logging.info('Data statistics : ')
            logging.info('Task : %s ' % (config['management']['task']))
            logging.info('Batch Size : %d ' % (config['data']['general']['batch_size']))
            logging.info('Found %d batches in train text ' % (len(train_iter)))
            logging.info('Found %d batches in valid text ' % (len(val_iter)))
            logging.info('Found %d batches in test text ' % (len(test_iter)))
            logging.info('Found %d words in vocab ' % (vocab_size))
            logging.info('Model statistics : ')
            logging.info('embed_size_%d' % (config['model']['emb_size']))
            logging.info('hidden_size_%d' % (config['model']['hidden_size']))
            logging.info('num_layers_%d' % (config['model']['layers']))
            logging.info('Training statistics : ')
            logging.info('optimizer_%s' % (config['training']['general']['optimizer']))
            logging.info('epoch_%d' % (config['training']['general']['num_epochs']))

            if config['management']['train']:
                trainer = Training(train_iter, val_iter, config['training']['general']['clip_c'],
                                   config['training']['general']['lrate'],
                                   path + config['training']['train']['save_model_dir'],
                                   config['training']['train']['model_name'],logging, device)
                trainer.initialise_model(config['model']['rnn_type'], vocab_size, config['model']['emb_size'],
                                         config['model']['hidden_size'], config['model']['layers'],
                                         config['model']['drop_out'], config['training']['general']['tie_weights'],
                                         config['training']['general']['optimizer'])
                trainer.train_model(config['data']['general']['batch_size'], config['training']['general']['num_epochs']
                                    ,config['management']['monitor_loss'],config['management']['checkpoint_freq'])
            elif config['management']['infer']:
                if config['inference']['general']['type'] == 'greedy':
                    greedy_infer = Decode(path + config['training']['train']['save_model_dir'],
                                          config['training']['train']['model_name'],test_iter, vocabs, logging, device)
                    lm_model = greedy_infer.initialise_model(config['model']['rnn_type'], vocab_size,
                                                          config['model']['emb_size'], config['model']['hidden_size'],
                                                          config['model']['layers'], config['model']['drop_out'],
                                                          config['training']['general']['tie_weights'])

                    accuracy = greedy_infer.fill_in_the_blank(lm_model, path + config['inference']['general']['save_path'],
                                                              config['inference']['general']['save_file_name'],
                                                              config['inference']['general']['mask_rate'],
                                                              config['inference']['general']['number_of_blanks'])
                    logging.info('accuracy_%f' % accuracy)
            else:
                logging.info('Either one of train or infer should be true')
            # elif config['inference']['general']['type'] == 'dynamic':
            #     IM = InferenceMachine(config['inference']['model_path'], config['inference']['model_name'],
            #                                config['model']['embed_size'], config['model']['hidden_size'],
            #                                config['model']['num_layers'],config['data']['batch_size'], test_iter,
            #                                vocab_size, config['model']['rnn_type'], logging, vocabs, device, path,
            #                                config['model']['drop_out'], config['model']['dropouth'],
            #                                config['model']['dropouti'], config['model']['dropoute'],
            #                                config['model']['wdrop'], config['training']['tie_weights'],
            #                                config['inference']['number_of_top_scoring_words'],
            #                                config['data']['seq_length'])
            #
            #     IM.fill_in_the_blank(config['inference']['save_path'],
            #                              config['inference']['mask_rate'],
            #                              config['inference']['number_of_blanks'])
            # else:
            #     logging.info('Inference type is not implemented')

            exit()

    except KeyboardInterrupt as e:
        logging.info('Operation has been interrupted from keyboard')
        print("[STOP]", e)
        exit()
