import os
import torch
import argparse

from dataset import Dataset
from utils import compute_F1, compute_exact_match
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm
from trainer import train, valid
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, AdamW
from torch.utils.tensorboard import SummaryWriter
from knockknock import email_sender

import datetime
now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
writer = SummaryWriter()
parser = argparse.ArgumentParser()

parser.add_argument('--patience' ,  type = int, default=3)
parser.add_argument('--batch_size' , type = int, default=8)
parser.add_argument('--max_epoch' ,  type = int, default=2)
parser.add_argument('--base_trained_model', type = str, default = 'bert-base-uncased', help =" pretrainned model from 🤗")
parser.add_argument('--pretrained_model' , type = str,  help = 'pretrainned model')
parser.add_argument('--gpu_number' , type = int,  default = 0, help = 'which GPU will you use?')
parser.add_argument('--debugging' , type = bool,  default = False, help = "Don't save file")
parser.add_argument('--log_file' , type = str,  default = f'logs/log_{now_time}.txt', help = 'Is this debuggin mode?')
parser.add_argument('--dataset_name' , required= True, type = str,  help = 'mrqa|squad|coqa')
# parser.add_argument('--max_length' , type = int,  default = 512, help = 'max length')
parser.add_argument('--do_train' , default = True, help = 'do train or not', action=argparse.BooleanOptionalAction)


args = parser.parse_args()

def makedirs(path): 
   try: 
        os.makedirs(path) 
   except OSError: 
       if not os.path.isdir(path): 
           raise
       

args = parser.parse_args()

@email_sender(recipient_emails=["jihyunlee@postech.ac.kr"], sender_email="knowing.deep.clean.water@gmail.com")
def main():
    makedirs("./data"); makedirs("./logs"); makedirs("./model");
    
    
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_trained_model, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(args.base_trained_model)
    train_dataset = Dataset(args.dataset_name, tokenizer, "train")
    val_dataset = Dataset(args.dataset_name, tokenizer,  "validation") 

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    dev_loader = DataLoader(val_dataset, args.batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    log_file = open(args.log_file, 'w')
    device = torch.device(f'cuda:{args.gpu_number}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device) # change allocation of current GPU
    torch.cuda.empty_cache()


    if args.pretrained_model:
        print("use trained model")
        log_file.write("use trained model")
        model.load_state_dict(torch.load(args.pretrained_model))
    
    log_file.write(str(args))
    model.to(device)
    penalty = 0
    min_loss = float('inf')

    for epoch in range(args.max_epoch):
        print(f"Epoch : {epoch}")
        if args.do_train:
            train(model, train_loader, optimizer, device)

        pred_texts, ans_texts, loss = valid(model, dev_loader, device, tokenizer,log_file)
        
        EM, F1 = 0, 0
        for iter, (pred_text, ans_text) in enumerate(zip(pred_texts, ans_texts)):
            EM += compute_exact_match(pred_text, ans_text)
            F1 += compute_F1(pred_text, ans_text)
        
        print("Epoch : %d, EM : %.04f, F1 : %.04f, Loss : %.04f" % (epoch, EM/iter, F1/iter, loss))
        log_file.writelines("Epoch : %d, EM : %.04f, F1 : %.04f, Loss : %.04f" % (epoch, EM/iter, F1/iter, loss))

        writer.add_scalar("EM", EM/iter, epoch)
        writer.add_scalar("F1", F1/iter, epoch)
        writer.add_scalar("loss",loss, epoch)


        if loss < min_loss:
            print("New best")
            min_loss = loss
            penalty = 0
            if not args.debugging:
                torch.save(model.state_dict(), f"model/{args.dataset_name}.pt")
        else:
            penalty +=1
            if penalty>args.patience:
                print(f"early stopping at epoch {epoch}")
                break
    writer.close()
    log_file.close()
    
    
    return {'EM' : EM/iter, 'F1' : F1/iter}

    
if __name__ =="__main__":
    main()

    