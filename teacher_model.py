import torch.nn as nn
import torch


class CNNTeacherModel(nn.Module):
    def __init__(self, shared_model, tokenizer, num_labels, args, hidden_size):
        super().__init__()
        self.shared_model = shared_model
        self.category_head = nn.Linear(hidden_size, num_labels)
        self.class_head = nn.Linear(hidden_size, num_labels)
        self.variant_head = nn.Linear(hidden_size, num_labels)
        self.base_head = nn.Linear(hidden_size, num_labels)
        self.deprecated_head = nn.Linear(hidden_size, num_labels)
        # Note. pillar has only one label, so no need to train a head
        self.tokenizer = tokenizer
        self.args = args

    def forward(self, input_ids, groups, labels, return_prob=False, return_logit=False):
        # size: batch_size, num_labels
        hidden_state = self.shared_model(input_ids=input_ids, return_hidden_state=True) #教师的中间层特征
        category_logits = self.category_head(hidden_state)
        class_logits = self.class_head(hidden_state)
        variant_logits = self.variant_head(hidden_state)
        base_logits = self.base_head(hidden_state)
        deprecated_logits = self.deprecated_head(hidden_state)
        # iter batch
        logits = torch.empty(category_logits.shape[0], category_logits.shape[1]).float().to(self.args.device)
        #print(len(groups))
        for i in range(len(groups)):
            if groups[i].item() == 0:
                logits[i, :] = category_logits[i]#表示 logits 矩阵的第 i 行的所有列。: 是切片符号，表示选择该行的所有列。
            elif groups[i].item() == 1:
                logits[i, :] = class_logits[i]
            elif groups[i].item() == 2:
                logits[i, :] = variant_logits[i]
            elif groups[i].item() == 3:
                logits[i, :] = base_logits[i]
            elif groups[i].item() == 4:
                logits[i, :] = deprecated_logits[i]
            elif groups[i].item() == 5:
                logits[i, :] = labels[i]
        if return_prob:
            prob = torch.softmax(logits, dim=-1) #对 logits 的最后一个维度进行 softmax 变换，得到每个样本属于每个类别的概率分布
            return prob
        elif return_logit:# 对于学生训练中返回两个logits
            #修改行
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            # print("---------------------------------------------------")
            # print(loss)
            # print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
            return logits,loss
        else:  #对于自己训练
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)#教师预测和ground truth
            return loss
