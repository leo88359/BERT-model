from core import convert_data_to_feature, make_dataset, split_dataset, compute_accuracy, use_model
from torch.utils.data import DataLoader
import torch
from transformers import AdamW
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter   #for visulization
writer = SummaryWriter()
import pickle

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



if __name__ == "__main__":    
    # BERT
    model_setting = {
        "model_name":"bert", 
        "config_file_path":"bert-base-chinese", 
        "model_file_path":"bert-base-chinese", 
        "vocab_file_path":"bert-base-chinese-vocab.txt",
        "num_labels":2  # 分幾類 
    }    


    ## RoBERTa-wwm-ext-large-chinese
    #model_setting = {
    #    "model_name":"bert", 
    #    "config_file_path":r"C:\Users\e7789520\Desktop\HO TSUNG TSE\TaipeiCityFood\chinese_roberta_wwm_large_ext_pytorch\bert_config.json", 
    #    "model_file_path":r"C:\Users\e7789520\Desktop\HO TSUNG TSE\TaipeiCityFood\chinese_roberta_wwm_large_ext_pytorch\pytorch_model.bin", 
    #    "vocab_file_path":r"C:\Users\e7789520\Desktop\HO TSUNG TSE\TaipeiCityFood\chinese_roberta_wwm_large_ext_pytorch\vocab.txt",
    #    "num_labels":18  # 分幾類 
    #}    



    # ALBERT
    # model_setting = {
    #     "model_name":"albert", 
    #     "config_file_path":"albert/albert_tiny/config.json", 
    #     "model_file_path":"albert/albert_tiny/pytorch_model.bin", 
    #     "vocab_file_path":"albert/albert_tiny/vocab.txt",
    #     "num_labels":149 # 分幾類
    # }    

    #

    model, tokenizer = use_model(**model_setting)
    
    # setting device  
    ##使用multi-GPU時，設定torch.device('cuda:要跑的GPU編號')，設定nn.DataParallel(model,device_ids=[要跑的GPU編號])  
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print("using device",device)
    model = nn.DataParallel(model, device_ids=[0])  #run multi-GPU時要加的部分，意思為把model建在不同的GPU上去跑
    model.to(device)
    




    #資料擷取
    data_feature = convert_data_to_feature(tokenizer,'Diet_Health_Need_combine_withfeature.txt')
    input_ids = data_feature['input_ids']
    input_masks = data_feature['input_masks']
    input_segment_ids = data_feature['input_segment_ids']
    answer_lables = data_feature['answer_lables']
    clinical_feature_ids = data_feature['clinical_feature_ids']
    

    
    #訓練時的training跟testing的設定
    full_dataset = make_dataset(input_ids = input_ids, input_masks = input_masks, input_segment_ids = input_segment_ids, answer_lables = answer_lables, clinical_feature_ids = clinical_feature_ids)
    train_dataset, test_dataset = split_dataset(full_dataset, 0.8)  #區分training跟testing dataset嗎?
    train_dataloader = DataLoader(train_dataset,batch_size=16,shuffle=True)
    test_dataloader = DataLoader(test_dataset,batch_size=16,shuffle=True)    


    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-6, eps=1e-8)
    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    #呼叫pkl檔(pkl檔案儲存的資料為word/position/segment embedding的資料，在training一開始時完成BERT input representation就會生成這個檔)
    #把pkl檔call進來後，擷取裡面的answer label作為id轉標註內容的dictionary
    pkl_file = open('trained_model/data_features.pkl', 'rb')
    data_features = pickle.load(pkl_file)
    answer_dic = data_features['answer_dic']
    question_dic = data_features['question_dic']
    clinical_feature_dic = data_features['clinical_feature_dic']

    model.zero_grad()
    for epoch in range(150):       #30 in Taipei_QA_BERT
        running_loss_val = 0.0
        running_acc = 0.0
        for batch_index, batch_dict in enumerate(train_dataloader):
            model.train()
            batch_dict = tuple(t.to(device) for t in batch_dict)
            outputs = model(
                batch_dict[0],
                clinical_feature_embeddings = batch_dict[4],
                # attention_mask=batch_dict[1],
                labels = batch_dict[3]
                )
            loss,logits = outputs[:2]
            
            #把loss回傳到optimizer中計算，然後調整model
            loss.sum().backward()
            optimizer.step()
            # scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            
            
            #loss.sum()在single GPU時沒有功用,但在multi-GPU時，因為分了n張GPU跑，所以會有n個loss回傳
            #但是進入loss計算時只能有一個tensor，所以要用sum來做合併
            #這邊做宣告方便下方loss計算
            loss_sum=loss.sum()
            
           
            # compute the loss            #single-GPU時:loss.item()  #muti-GPU時:loss_sum.item()
            loss_t = loss_sum.item()      
            running_loss_val += (loss_t - running_loss_val) / (batch_index + 1)

            # compute the accuracy
            acc_t = compute_accuracy(logits, batch_dict[3])
            running_acc += (acc_t - running_acc) / (batch_index + 1)

            # log
            print("epoch:%2d batch:%4d train_loss:%2.4f train_acc:%3.4f"%(epoch+1, batch_index+1, running_loss_val, running_acc))
            
            #這裡進行畫圖表。在cmd中，進到runs檔案夾的前一層中，下: tensorboard --logdir runs --port=4100
            #在瀏覽器下localhost:4100，就可以看到圖表       
            writer.add_scalar("training loss", running_loss_val, epoch)
            writer.add_scalar("training acc", running_acc, epoch)

            writer.close()
            
        
        running_loss_val = 0.0
        running_acc = 0.0
        for batch_index, batch_dict in enumerate(test_dataloader):
            model.eval()
            batch_dict = tuple(t.to(device) for t in batch_dict)
            outputs = model(
                batch_dict[0],
                # attention_mask=batch_dict[1],
                labels = batch_dict[3]
                )
            loss,logits = outputs[:2]
            loss.sum()
            
            
            #loss.sum()在single GPU時沒有功用,但在multi-GPU時，因為分了n張GPU跑，所以會有n個loss回傳
            #但是進入loss計算時只能有一個tensor，所以要用sum來做合併
            #這邊做宣告方便下方loss計算
            loss_sum=loss.sum()
            
            
            # compute the loss            #single-GPU時:loss.item()  #muti-GPU時:loss_sum.item()
            loss_t = loss_sum.item()     
            running_loss_val += (loss_t - running_loss_val) / (batch_index + 1)

            # compute the accuracy
            acc_t = compute_accuracy(logits, batch_dict[3])
            running_acc += (acc_t - running_acc) / (batch_index + 1)

            # log
            print("epoch:%2d batch:%4d test_loss:%2.4f test_acc:%3.4f"%(epoch+1, batch_index+1, running_loss_val, running_acc))
            print(" ")
            
            #把每個epoch中的每個batch所算的testing set的loss和acc寫到txt檔中
            f=open("show_every_testing_Diet_Health_Need_combine_batch_16.txt","a")
            f.write("epoch:%2d batch:%4d test_loss:%2.4f test_acc:%3.4f \n"%(epoch+1, batch_index+1, running_loss_val, running_acc))
            
            test_predict=torch.argmax(logits,1)
            a=-1
            for c in range(len(batch_dict[3])):
                a+=1
                print("測試的問題: ",question_dic.answers[input_ids.index(batch_dict[0][a].tolist())])
                f.write("\n測試的問題: ")
                f.write(question_dic.answers[input_ids.index(batch_dict[0][a].tolist())])
                print("預測的答案: ",answer_dic.to_text(test_predict[a]))
                f.write("\n預測的答案: ")
                f.write(answer_dic.to_text(test_predict[a]))
                print("實際的答案: ",answer_dic.to_text(batch_dict[3][a]))
                f.write("\n實際的答案: ")
                f.write(answer_dic.to_text(batch_dict[3][a]))
                print('  ')
                f.write("\n")
                f.write("\n")
            print("---------------------------------------------")
            f.write("---------------------------------------------")
            f.write("\n")
            f.close()
            
            
            
            #這裡進行畫圖表。在cmd中，進到runs檔案夾中，下: tensorboard --logdir runs --port=4100
            #在瀏覽器下localhost:4100，就可以看到圖表
            writer.add_scalar("testing loss", running_loss_val, epoch)
            writer.add_scalar("testing acc", running_acc, epoch)
            
            writer.close()
            
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained('trained_model')
