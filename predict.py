import torch
import pickle
from core import to_bert_ids, use_model

if __name__ == "__main__":    
    ## load and init  (call其他result的路徑)
    #pkl_file = open(r'C:/Users/e7789520/Desktop/HO TSUNG TSE/AI_AAC_for_ICU/results/Health_Dite_Need_batch 64/trained_model/data_features.pkl', 'rb')
    
    # load and init  (做實驗的路徑)
    pkl_file = open('trained_model/data_features.pkl', 'rb')
    
    data_features = pickle.load(pkl_file)
    answer_dic = data_features['answer_dic']
        
    ## BERT (call其他result的路徑)
    #model_setting = {
    #    "model_name":"bert", 
    #    "config_file_path":r"C:/Users/e7789520/Desktop/HO TSUNG TSE/AI_AAC_for_ICU/results/Health_Dite_Need_batch 64/trained_model/config.json", 
    #    "model_file_path":r"C:/Users/e7789520/Desktop/HO TSUNG TSE/AI_AAC_for_ICU/results/Health_Dite_Need_batch 64/trained_model/pytorch_model.bin", 
    #    "vocab_file_path":"bert-base-chinese-vocab.txt",
    #    "num_labels":71  # 分幾類 
    #}    

    # BERT (做實驗的路徑)
    model_setting = {
        "model_name":"bert", 
        "config_file_path":"trained_model/config.json", 
        "model_file_path":"trained_model/pytorch_model.bin", 
        "vocab_file_path":"bert-base-chinese-vocab.txt",
        "num_labels":2  # 分幾類 
    }    


    # ALBERT
    # model_setting = {
    #     "model_name":"albert", 
    #     "config_file_path":"trained_model/config.json", 
    #     "model_file_path":"trained_model/pytorch_model.bin", 
    #     "vocab_file_path":"albert/albert_tiny/vocab.txt",
    #     "num_labels":149 # 分幾類
    # }
    
    #
    model, tokenizer = use_model(**model_setting)
    model.eval()

    #
    q_inputs = ['那裡、那裡、那裡']
    for q_input in q_inputs:
        bert_ids = to_bert_ids(tokenizer,q_input)
        assert len(bert_ids) <= 512
        input_ids = torch.LongTensor(bert_ids).unsqueeze(0)
        
    cf_inputs = ['骨科']
    for cf_input in cf_inputs:
        feature_ids = to_bert_ids(tokenizer,cf_input)
        assert len(feature_ids) <= 512
        clinical_feature_ids = torch.LongTensor(feature_ids).unsqueeze(0)

        # predict
        outputs = model(input_ids)
        predicts = outputs[:2]
        predicts = predicts[0]
        max_val = torch.max(predicts)
        label = (predicts == max_val).nonzero().numpy()[0][1]
        ans_label = answer_dic.to_text(label)
        # predict num 2~4
        predicts_sort=torch.sort(predicts, descending=True) #將predict值依大小降序排列
        predicts_second=predicts_sort[1][0][1] #predict值第二大的id
        predicts_third=predicts_sort[1][0][2]  #predict值第三大的id
        predicts_fourth=predicts_sort[1][0][3] #predict值第三大的id
        
        #predict all (num5~71)-------------------------------------------------
        predicts_fifth=predicts_sort[1][0][4]
        predicts_sixth=predicts_sort[1][0][5]
        predicts_seventh=predicts_sort[1][0][6]
        predicts_eighth=predicts_sort[1][0][7]
        predicts_ninth=predicts_sort[1][0][8]
        predicts_tenth=predicts_sort[1][0][9]
        predicts_eleven=predicts_sort[1][0][10]
        predicts_twelfth=predicts_sort[1][0][11]
        predicts_thirteen=predicts_sort[1][0][12]
        predicts_fourteen=predicts_sort[1][0][13]
        predicts_fifteen=predicts_sort[1][0][14]
        predicts_sixteen=predicts_sort[1][0][15]
        predicts_seventeen=predicts_sort[1][0][16]
        predicts_eighteen=predicts_sort[1][0][17]
        predicts_ninteen=predicts_sort[1][0][18]
        predicts_twenty=predicts_sort[1][0][19]
        predicts_twenty_first=predicts_sort[1][0][20]
        predicts_twenty_second=predicts_sort[1][0][21]
        predicts_twenty_third=predicts_sort[1][0][22]
        predicts_twenty_fourth=predicts_sort[1][0][23]
        predicts_twenty_fifth=predicts_sort[1][0][24]
        predicts_twenty_sixth=predicts_sort[1][0][25]
        predicts_twenty_seventh=predicts_sort[1][0][26]
        predicts_twenty_eighth=predicts_sort[1][0][27]
        predicts_twenty_ninth=predicts_sort[1][0][28]
        predicts_thirty=predicts_sort[1][0][29]
        predicts_thirty_first=predicts_sort[1][0][30]
        predicts_thirty_second=predicts_sort[1][0][31]
        predicts_thirty_third=predicts_sort[1][0][32]
        predicts_thirty_fourth=predicts_sort[1][0][33]
        predicts_thirty_fifth=predicts_sort[1][0][34]
        predicts_thirty_sixth=predicts_sort[1][0][35]
        predicts_thirty_seventh=predicts_sort[1][0][36]
        predicts_thirty_eighth=predicts_sort[1][0][37]
        predicts_thirty_ninth=predicts_sort[1][0][38]
        predicts_forty=predicts_sort[1][0][39]
        predicts_forty_first=predicts_sort[1][0][40]
        predicts_forty_second=predicts_sort[1][0][41]
        predicts_forty_third=predicts_sort[1][0][42]
        predicts_forty_fourth=predicts_sort[1][0][43]
        predicts_forty_fifth=predicts_sort[1][0][44]
        predicts_forty_sixth=predicts_sort[1][0][45]
        predicts_forty_seventh=predicts_sort[1][0][46]
        predicts_forty_eighth=predicts_sort[1][0][47]
        predicts_forty_ninth=predicts_sort[1][0][48]
        predicts_fifty=predicts_sort[1][0][49]
        predicts_fifty_first=predicts_sort[1][0][50]
        predicts_fifty_second=predicts_sort[1][0][51]
        predicts_fifty_third=predicts_sort[1][0][52]
        predicts_fifty_fourth=predicts_sort[1][0][53]
        predicts_fifty_fifth=predicts_sort[1][0][54]
        predicts_fifty_sixth=predicts_sort[1][0][55]
        predicts_fifty_seventh=predicts_sort[1][0][56]
        predicts_fifty_eighth=predicts_sort[1][0][57]
        predicts_fifty_ninth=predicts_sort[1][0][58]
        predicts_sixty=predicts_sort[1][0][59]
        predicts_sixty_first=predicts_sort[1][0][60]
        predicts_sixty_second=predicts_sort[1][0][61]
        predicts_sixty_third=predicts_sort[1][0][62]
        predicts_sixty_fourth=predicts_sort[1][0][63]
        predicts_sixty_fifth=predicts_sort[1][0][64]
        predicts_sixty_sixth=predicts_sort[1][0][65]
        predicts_sixty_seventh=predicts_sort[1][0][66]
        predicts_sixty_eighth=predicts_sort[1][0][67]
        predicts_sixty_ninth=predicts_sort[1][0][68]
        predicts_seventy=predicts_sort[1][0][69]
        predicts_seventy_first=predicts_sort[1][0][70]
        #---------------------------------------------------------------------
        
        
        
        #把2~4的predict id回去查表
        ans_labelsecond=answer_dic.to_text(predicts_second) #predict第二順位答案
        ans_labelthird=answer_dic.to_text(predicts_third)   #predict第三順位答案
        ans_labelfourth=answer_dic.to_text(predicts_fourth) #predict第四順位答案
        
        #查表: num5~71---------------------------------------------------------
        ans_label_5=answer_dic.to_text(predicts_fifth)
        ans_label_6=answer_dic.to_text(predicts_sixth)   
        ans_label_7=answer_dic.to_text(predicts_seventh)
        ans_label_8=answer_dic.to_text(predicts_eighth)
        ans_label_9=answer_dic.to_text(predicts_ninth)   
        ans_label_10=answer_dic.to_text(predicts_tenth)
        ans_label_11=answer_dic.to_text(predicts_eleven)
        ans_label_12=answer_dic.to_text(predicts_twelfth)   
        ans_label_13=answer_dic.to_text(predicts_thirteen)
        ans_label_14=answer_dic.to_text(predicts_fourteen)
        ans_label_15=answer_dic.to_text(predicts_fifteen)   
        ans_label_16=answer_dic.to_text(predicts_sixteen)
        ans_label_17=answer_dic.to_text(predicts_seventeen)
        ans_label_18=answer_dic.to_text(predicts_eighteen)   
        ans_label_19=answer_dic.to_text(predicts_ninteen)
        ans_label_20=answer_dic.to_text(predicts_twenty)
        ans_label_21=answer_dic.to_text(predicts_twenty_first)   
        ans_label_22=answer_dic.to_text(predicts_twenty_second)
        ans_label_23=answer_dic.to_text(predicts_twenty_third)
        ans_label_24=answer_dic.to_text(predicts_twenty_fourth)   
        ans_label_25=answer_dic.to_text(predicts_twenty_fifth)
        ans_label_26=answer_dic.to_text(predicts_twenty_sixth)
        ans_label_27=answer_dic.to_text(predicts_twenty_seventh)   
        ans_label_28=answer_dic.to_text(predicts_twenty_eighth)
        ans_label_29=answer_dic.to_text(predicts_twenty_ninth)
        ans_label_30=answer_dic.to_text(predicts_thirty)
        ans_label_31=answer_dic.to_text(predicts_thirty_first)   
        ans_label_32=answer_dic.to_text(predicts_thirty_second)
        ans_label_33=answer_dic.to_text(predicts_thirty_third)
        ans_label_34=answer_dic.to_text(predicts_thirty_fourth)   
        ans_label_35=answer_dic.to_text(predicts_thirty_fifth)
        ans_label_36=answer_dic.to_text(predicts_thirty_sixth)
        ans_label_37=answer_dic.to_text(predicts_thirty_seventh)   
        ans_label_38=answer_dic.to_text(predicts_thirty_eighth)
        ans_label_39=answer_dic.to_text(predicts_thirty_ninth)
        ans_label_40=answer_dic.to_text(predicts_forty)
        ans_label_41=answer_dic.to_text(predicts_forty_first)   
        ans_label_42=answer_dic.to_text(predicts_forty_second)
        ans_label_43=answer_dic.to_text(predicts_forty_third)
        ans_label_44=answer_dic.to_text(predicts_forty_fourth)   
        ans_label_45=answer_dic.to_text(predicts_forty_fifth)
        ans_label_46=answer_dic.to_text(predicts_forty_sixth)
        ans_label_47=answer_dic.to_text(predicts_forty_seventh)   
        ans_label_48=answer_dic.to_text(predicts_forty_eighth)
        ans_label_49=answer_dic.to_text(predicts_forty_ninth)
        ans_label_50=answer_dic.to_text(predicts_fifty)
        ans_label_51=answer_dic.to_text(predicts_fifty_first)   
        ans_label_52=answer_dic.to_text(predicts_fifty_second)
        ans_label_53=answer_dic.to_text(predicts_fifty_third)
        ans_label_54=answer_dic.to_text(predicts_fifty_fourth)   
        ans_label_55=answer_dic.to_text(predicts_fifty_fifth)
        ans_label_56=answer_dic.to_text(predicts_fifty_sixth)
        ans_label_57=answer_dic.to_text(predicts_fifty_seventh)   
        ans_label_58=answer_dic.to_text(predicts_fifty_eighth)
        ans_label_59=answer_dic.to_text(predicts_fifty_ninth)
        ans_label_60=answer_dic.to_text(predicts_sixty)
        ans_label_61=answer_dic.to_text(predicts_sixty_first)   
        ans_label_62=answer_dic.to_text(predicts_sixty_second)
        ans_label_63=answer_dic.to_text(predicts_sixty_third)
        ans_label_64=answer_dic.to_text(predicts_sixty_fourth)   
        ans_label_65=answer_dic.to_text(predicts_sixty_fifth)
        ans_label_66=answer_dic.to_text(predicts_sixty_sixth)
        ans_label_67=answer_dic.to_text(predicts_sixty_seventh)   
        ans_label_68=answer_dic.to_text(predicts_sixty_eighth)
        ans_label_69=answer_dic.to_text(predicts_sixty_ninth)
        ans_label_70=answer_dic.to_text(predicts_seventy)
        ans_label_71=answer_dic.to_text(predicts_seventy_first)
        #---------------------------------------------------------------------
        
        
        
        print("醫護人員問的話： ",q_input)
        print('病人想要的回答(最佳解答)： ',ans_label)
        print("病人想要的回答(第2~4順位): ",ans_labelsecond,ans_labelthird,ans_labelfourth)
        
        # print num 5~71-------------------------------------------------------
        print("病人想要的回答(第5~10順位): ",ans_label_5,ans_label_6,ans_label_7,ans_label_8,ans_label_9,ans_label_10)
        print("病人想要的回答(第11~20順位): ",ans_label_11,ans_label_12,ans_label_13,ans_label_14,ans_label_15,ans_label_16,ans_label_17,ans_label_18,ans_label_19,ans_label_20)
        print("病人想要的回答(第21~30順位): ",ans_label_21,ans_label_22,ans_label_23,ans_label_24,ans_label_25,ans_label_26,ans_label_27,ans_label_28,ans_label_29,ans_label_30)
        print("病人想要的回答(第31~40順位): ",ans_label_31,ans_label_32,ans_label_33,ans_label_34,ans_label_35,ans_label_36,ans_label_37,ans_label_38,ans_label_39,ans_label_40)
        print("病人想要的回答(第41~50順位): ",ans_label_41,ans_label_42,ans_label_43,ans_label_44,ans_label_45,ans_label_46,ans_label_47,ans_label_48,ans_label_49,ans_label_50)
        print("病人想要的回答(第51~60順位): ",ans_label_51,ans_label_52,ans_label_53,ans_label_54,ans_label_55,ans_label_56,ans_label_57,ans_label_58,ans_label_59,ans_label_60)
        print("病人想要的回答(第61~71順位): ",ans_label_61,ans_label_62,ans_label_63,ans_label_64,ans_label_65,ans_label_66,ans_label_67,ans_label_68,ans_label_69,ans_label_70,ans_label_71)
        
        #f=open("all_outside_testing_results_Diet_Health_Need_combine_batch_64.txt","a")
        #f.write("醫護人員問的話： ")
        #f.write(q_input)
        #f.write("\n病人想要的回答(第1~10順位): ")
        #f.write(ans_label,ans_labelsecond,ans_labelthird,ans_labelfourth,ans_label_5,ans_label_6,ans_label_7,ans_label_8,ans_label_9,ans_label_10)
        #f.write("\n病人想要的回答(第11~20順位): ")
        #f.write(ans_label_11,ans_label_12,ans_label_13,ans_label_14,ans_label_15,ans_label_16,ans_label_17,ans_label_18,ans_label_19,ans_label_20)
        #f.write("\n病人想要的回答(第21~30順位): ")
        #f.write(ans_label_21,ans_label_22,ans_label_23,ans_label_24,ans_label_25,ans_label_26,ans_label_27,ans_label_28,ans_label_29,ans_label_30)
        #f.write("\n病人想要的回答(第31~40順位): ")
        #f.write(ans_label_31,ans_label_32,ans_label_33,ans_label_34,ans_label_35,ans_label_36,ans_label_37,ans_label_38,ans_label_39,ans_label_40)
        #f.write("\n病人想要的回答(第41~50順位): ")
        #f.write(ans_label_41,ans_label_42,ans_label_43,ans_label_44,ans_label_45,ans_label_46,ans_label_47,ans_label_48,ans_label_49,ans_label_50)
        #f.write("\n病人想要的回答(第51~60順位): ")
        #f.write(ans_label_51,ans_label_52,ans_label_53,ans_label_54,ans_label_55,ans_label_56,ans_label_57,ans_label_58,ans_label_59,ans_label_60)
        #f.write("\n病人想要的回答(第61~71順位): ")
        #f.write(ans_label_61,ans_label_62,ans_label_63,ans_label_64,ans_label_65,ans_label_66,ans_label_67,ans_label_68,ans_label_69,ans_label_70,ans_label_71)
        #f.close()
        
        
        
        
        
        
        print()


