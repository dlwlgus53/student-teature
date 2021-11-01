from transformers import AutoModelForQuestionAnswering, AutoTokenizer, AdamW
import torch
import pickle
from tqdm import tqdm
import pdb
import os
import json

import ontology
# here, squad means squad2

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, type, tokenizer):
        self.tokenizer = tokenizer

        try:
            print("Load processed data")
            with open(f'data/preprocessed_{type}.pickle', 'rb') as f:
                encodings = pickle.load(f)
        except:
            print("preprocessing data...")
            raw_dataset = json.load(open(data_path, "r"))
            context, question, answer = self._preprocessing_dataset(raw_dataset)
            assert len(context) == len(question) == len(answer['answer_start']) == len(answer['answer_end'])
            print("Encoding dataset (it will takes some time)")
            
            encodings = tokenizer(context, question, truncation='only_second', padding=True) # [CLS] context [SEP] question
            print("add token position")
            encodings = self._add_token_positions(encodings, answer)

            with open(f'data/preprocessed_{type}.pickle', 'wb') as f:
                pickle.dump(encodings, f, pickle.HIGHEST_PROTOCOL)

        self.encodings = encodings



    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

    def _preprocessing_dataset(self, dataset):
        
        context = []
        question = []
        answer = {'answer_start' : [], 'answer_end' : []}
        print(f"preprocessing data")
        pdb.set_trace()
        
        for key in dataset.keys():
            dialogue = dataset[key]['log']
            dialouge_text = ""
            for turn in dialogue:
                dialouge_text += turn['user']
                c = dialouge_text[-128:] # get from tail. TODO for 128
                for key in turn['belief']:
                    if key in ontology.QA['extract-domain']:
                        q = ontology.QA[key]['description']
                        a = turn['belief'][key]
                        
                        context.append(c)
                        question.append(q)
                        answer['answer_start'].append(c.find(a)) # 여기서 아마 문제가 생길걸??
                        answer['answer_end'].append(c.find(a) + len(a))                

                
                dialouge_text += turn['response']
        
        return context, question, answer


    def _char_to_token_with_possible(self, i, encodings, char_position, type):
        if type == 'start':
            possible_position = [0,-1,1]
        else:
            possible_position = [-1,-2,0]

        for pp in possible_position:
            position = encodings.char_to_token(i, char_position + pp)
            if position != None:
                break
        return position

    def _add_token_positions(self, encodings, answers):
        start_positions = []
        end_positions = []
        for i in range(len(answers['answer_start'])):
            # char의 index로 되어있던것을 token의 index로 찾아준다.
            if  answers['answer_start'][i] != -1: # for case of mrq
                start_char = answers['answer_start'][i] 
                end_char = answers['answer_end'][i]
                start_position = self._char_to_token_with_possible(i, encodings, start_char,'start')
                end_position = self._char_to_token_with_possible(i, encodings, end_char,'end')
                start_positions.append(start_position)
                end_positions.append(end_position)
            else:
                start_positions.append(None)
                end_positions.append(None)
            
            if start_positions[-1] is None:
                start_positions[-1] = self.tokenizer.model_max_length
            if end_positions[-1] is None:
                end_positions[-1] = self.tokenizer.model_max_length

        return encodings.update({'start_positions': start_positions, 'end_positions': end_positions})
        

if __name__ == '__main__':
    data_path = '../../data/MultiWOZ_2.1/dev_data.json'
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
    type = 'dev'
    dd = Dataset(data_path, type, tokenizer,)
    pdb.set_trace()
    for i in range(10):
        print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(dd[i]['input_ids'])))
        try:
            print(tokenizer.convert_ids_to_tokens(dd[i]['input_ids'])[dd[i]['start_positions']:dd[i]['end_positions']+1][0])
        except:
            print(tokenizer.convert_ids_to_tokens(dd[i]['input_ids'])[dd[i]['start_positions']:dd[i]['end_positions']][0])
            
        


