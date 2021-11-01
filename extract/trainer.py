
import torch
from tqdm import tqdm

def train(model, train_loader, optimizer, device):
        model.train()
        loss_sum = 0
        t_train_loader = tqdm(train_loader)
        for batch in t_train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            t_train_loader.set_description("Loss %.04f" % (loss))



def valid(model, dev_loader, device, tokenizer, log_file):

    model.eval()
    pred_texts = []
    ans_texts = []
    loss_sum = 0
    print("Validation start")
    with torch.no_grad():
        log_file.write("\n")
        t_dev_loader = tqdm(dev_loader)
        for iter,batch in enumerate(t_dev_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)

            pred_start_positions = torch.argmax(outputs['start_logits'], dim=1).to('cpu')
            pred_end_positions = torch.argmax(outputs['end_logits'], dim=1).to('cpu')
            for b in range(len(pred_start_positions)):
                ans_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[b][start_positions[b]:end_positions[b]+1]))
                pred_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[b][pred_start_positions[b]:pred_end_positions[b]+1]))
                ans_texts.append(ans_text)
                pred_texts.append(pred_text)
            if iter%100 ==0:
                log_file.write(f"ans text : {ans_text}\n pred_text : {pred_text}\n")         
            loss = outputs[0].to('cpu')
            loss_sum += loss
            t_dev_loader.set_description("Loss %.04f  | step %d" % (loss, iter))
    
    return pred_texts, ans_texts, loss_sum/iter
        
        

            