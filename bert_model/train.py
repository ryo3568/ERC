from solver import *
from data_loader import get_loader
from configs import get_config
from util import Vocab
import os
import pickle


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    config = get_config(mode='train')
    val_config = get_config(mode='valid')
    test_config = get_config(mode='test')

    _RUNS = config.runs

    _best_test_loss, _best_test_f1_w, _best_test_f1_m, _best_epoch = [], [], [], []

    for run in range(_RUNS):

        print(config)

        # No. of videos to consider
        training_data_len = int(config.training_percentage * \
            len(load_pickle(config.sentences_path)))


        train_data_loader = get_loader(
            sentences=load_pickle(config.sentences_path)[:training_data_len],
            labels=load_pickle(config.label_path)[:training_data_len],
            #直前の感情系列を追加
            before_labels=load_pickle(config.before_label_path)[:training_data_len],
            #話者情報を追加
            speakers=load_pickle(config.speaker_path)[:training_data_len],
            conversation_length=load_pickle(config.conversation_length_path)[:training_data_len],
            sentence_length=load_pickle(config.sentence_length_path)[:training_data_len],
            batch_size=config.batch_size)

        eval_data_loader = get_loader(
            sentences=load_pickle(val_config.sentences_path),
            labels=load_pickle(val_config.label_path),
            #直前の感情系列を追加
            before_labels=load_pickle(val_config.before_label_path),
            #話者情報を追加
            speakers=load_pickle(val_config.speaker_path),
            conversation_length=load_pickle(val_config.conversation_length_path),
            sentence_length=load_pickle(val_config.sentence_length_path),
            batch_size=val_config.eval_batch_size,
            shuffle=False)
        
        test_data_loader = get_loader(
            sentences=load_pickle(test_config.sentences_path),
            labels=load_pickle(test_config.label_path),
            #直前の感情系列を追加
            before_labels=load_pickle(test_config.before_label_path),
            #話者情報を追加
            speakers=load_pickle(test_config.speaker_path),
            conversation_length=load_pickle(test_config.conversation_length_path),
            sentence_length=load_pickle(test_config.sentence_length_path),
            batch_size=test_config.eval_batch_size,
            shuffle=False)




        # for testing
        solver = Solver

        solver = solver(config, train_data_loader,
                        eval_data_loader, test_data_loader, is_train=True)

        solver.build()

        best_test_loss, best_test_f1_w, best_epoch = solver.train()

        print(f"Current RUN: {run+1}")

        print("\n\nBest test loss")
        print(best_test_loss)
        #出力を変更
        if (config.data == "dailydialog"):
            print("Best test micro f1")
        else:
            print("Best test f1 weighted")
        print(best_test_f1_w)
        print("Best epoch")
        print(best_epoch)

        _best_test_loss.append(best_test_loss)
        _best_test_f1_w.append(best_test_f1_w)
        _best_epoch.append(best_epoch)


    # Print final
    print(f"\n\nAverage across runs:")

    print("Best epoch")
    print(_best_epoch)

    print("\n\nBest test loss")
    print(np.mean(np.array(_best_test_loss), axis=0))

    #出力を変更
    if(config.data == "dailydialog"):
        print("Overall test micro f1")
    else:
        print("Overall test f1 weighted")
    _best_test_f1_w = np.array(_best_test_f1_w) * 100
    _best_test_f1_w = np.round(_best_test_f1_w, decimals=2)
    print(np.array(_best_test_f1_w))
    
    #出力を変更
    if (config.data == "dailydialog"):
        print("Best test micro f1")
    else:
        print("Best test f1 weighted")
    _best_test_f1_w_mean = np.mean(np.array(_best_test_f1_w), axis=0)
    _best_test_f1_w_mean = round(_best_test_f1_w_mean,2)
    print(_best_test_f1_w_mean)