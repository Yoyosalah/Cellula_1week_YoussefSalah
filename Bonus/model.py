import torch
from transformers import DistilBertForSequenceClassification
from peft import LoraConfig,get_peft_model,TaskType

def get_model(num_labels,device='cuda'):
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',num_labels=num_labels)
    lora_config = LoraConfig(r=8,lora_alpha=16,target_modules=["q_lin","k_lin","v_lin"], lora_dropout=0.1,bias="none",task_type=TaskType.SEQ_CLS) # scale = rank/alpha
    model = get_peft_model(model,lora_config)
    model.print_trainable_parameters()
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model