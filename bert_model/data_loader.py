import random
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import numpy as np

from pytorch_pretrained_bert.tokenization import BertTokenizer

SEQ_LEN = 30

class DialogDataset(Dataset):
    #話者情報を追加
    #直前の感情系列を追加
    def __init__(self, conversations, labels, before_labels, speakers, conversation_length, sentence_length, data=None):

        # [total_data_size, max_conversation_length, max_sentence_length]
        # tokenized raw text of sentences
        self.conversations = conversations
        self.labels = labels

        #直前の感情系列を追加
        self.before_labels = before_labels
        #話者情報を追加
        self.speakers = speakers

        # conversation length of each batch
        # [total_data_size]
        self.conversation_length = conversation_length

        # list of length of sentences
        # [total_data_size, max_conversation_length]
        self.sentence_length = sentence_length
        self.data = data
        self.len = len(conversations)

        # Prepare for BERT
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        self.prepare_BERT()

    def prepare_BERT(self,):

        new_conversations=[]
        new_sentence_lengths=[]
        new_type_id = []
        new_masks = []

        for idx in range(len(self.conversations)):
            conversation = self.conversations[idx]
            sentence_lengths = self.sentence_length[idx]
            assert(len(conversation)==len(sentence_lengths))

            local_sentences, local_sentence_length, local_type_id, local_masks = [],[],[],[] 
            for sentence, length in zip(conversation, sentence_lengths):
                line = " ".join(sentence[:sentence.index("<eos>")])
                tokens_a = self.tokenizer.tokenize(line)
                
                tokens = []
                input_type_ids = []
                tokens.append("[CLS]")
                input_type_ids.append(0)
                for token in tokens_a:
                    tokens.append(token)
                    input_type_ids.append(0)
                tokens.append("[SEP]")
                input_type_ids.append(0)

                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                input_mask = [1] * len(input_ids)

                # Zero-pad up to the sequence length.
                while len(input_ids) < SEQ_LEN:
                    input_ids.append(0)
                    input_mask.append(0)
                    input_type_ids.append(0)

                input_ids = input_ids[:SEQ_LEN]
                input_mask = input_mask[:SEQ_LEN]
                input_type_ids = input_type_ids[:SEQ_LEN]

                assert len(input_ids) == SEQ_LEN
                assert len(input_mask) == SEQ_LEN
                assert len(input_type_ids) == SEQ_LEN


                local_sentences.append(input_ids)
                local_sentence_length.append(len(input_ids))
                local_type_id.append(input_type_ids)
                local_masks.append(input_mask)
            new_conversations.append(local_sentences.copy())
            new_sentence_lengths.append(local_sentence_length.copy())
            new_type_id.append(local_type_id.copy())
            new_masks.append(local_masks.copy())

        self.conversations = new_conversations
        self.sentence_length = new_sentence_lengths
        self.type_ids = new_type_id
        self.masks = new_masks


    def __getitem__(self, index):
        """Return Single data sentence"""
        # [max_conversation_length, max_sentence_length]
        conversation = self.conversations[index]
        labels = self.labels[index]
        
        #直前の感情系列を追加
        before_labels = self.before_labels[index]

        #話者情報を追加
        speakers = self.speakers[index]
        conversation_length = self.conversation_length[index]
        sentence_length = self.sentence_length[index]
        type_id = self.type_ids[index]
        masks = self.masks[index]

        #話者情報を追加
        #直前の感情系列を追加
        return conversation, labels, before_labels, speakers, conversation_length, sentence_length, type_id, masks

    def __len__(self):
        return self.len



#話者情報を追加
def get_loader(sentences, labels, before_labels, speakers, conversation_length, sentence_length, batch_size=100, data=None, shuffle=True):
    """Load DataLoader of given DialogDataset"""

    #話者情報を追加
    #直前の感情系列を追加
    dataset = DialogDataset(sentences, labels, before_labels, speakers, conversation_length,
                            sentence_length, data=data)

    #話者情報の数を検査
    #直前の感情系列の数を検査
    for sentence, label, before_label,speaker in zip(sentences, labels, before_labels, speakers):
        assert(np.array(sentence).shape[0] == np.array(label).shape[0] and np.array(sentence).shape[0] == np.array(speaker).shape[0] and np.array(sentence).shape[0] == np.array(before_label).shape[0])





    def collate_fn(data):
        """
        Collate list of data in to batch

        Args:
            data: list of tuple(source, target, conversation_length, source_length, target_length)
        Return:
            Batch of each feature
            - source (LongTensor): [batch_size, max_conversation_length, max_source_length]
            - target (LongTensor): [batch_size, max_conversation_length, max_source_length]
            - conversation_length (np.array): [batch_size]
            - source_length (LongTensor): [batch_size, max_conversation_length]
        """
        # Sort by conversation length (descending order) to use 'pack_padded_sequence'
        data.sort(key=lambda x: x[2], reverse=True)

        # Separate
        #話者情報を追加
        #直前の感情系列を追加
        sentences, labels, before_labels, speakers, conversation_length, sentence_length, type_id, mask = zip(*data)

        # return sentences, conversation_length, sentence_length.tolist()
        #話者情報を追加
        return sentences, labels, before_labels, speakers, conversation_length, sentence_length, type_id, mask


    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn)

    return data_loader
