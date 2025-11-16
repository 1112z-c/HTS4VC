# import torch
# import torch.nn as nn
# from torch.nn import CrossEntropyLoss
# import torch.nn.functional as F
#
#
# class RobertaClassificationHead(nn.Module):
#     def __init__(self, config, num_labels):
#         super().__init__()
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         # layers for [CLS]
#         self.cls_dense = nn.Linear(config.hidden_size, config.hidden_size)
#         self.cls_out_proj = nn.Linear(config.hidden_size, num_labels)
#         # layers for [SOFT DIS]
#         self.dis_dense = nn.Linear(config.hidden_size, config.hidden_size)
#         self.dis_out_proj = nn.Linear(config.hidden_size, num_labels)
#
#     def forward(self, features, dis_locations, args):
#         # predict based on [CLS]
#         x_cls = features[:, 0, :]  # take <s> token (equiv. to [CLS])
#         x_cls = self.dropout(x_cls)
#         x_cls = self.cls_dense(x_cls)
#         x_cls = torch.tanh(x_cls)
#         x_cls = self.dropout(x_cls)
#         x_cls = self.cls_out_proj(x_cls)
#         # predict based on [DIS]
#         x_dis = []
#         for i in range(len(dis_locations)): # take <dis>
#             x_dis_slice = features[i, dis_locations[i], :]
#             x_dis.append(x_dis_slice.tolist())
#         x_dis = torch.tensor(x_dis).to(args.device)
#         x_dis = self.dropout(x_dis)
#         x_dis = self.dis_dense(x_dis)
#         x_dis = torch.tanh(x_dis)
#         x_dis = self.dropout(x_dis)
#         x_dis = self.dis_out_proj(x_dis)
#         return x_cls, x_dis
#
# class StudentBERT(nn.Module):
#     def __init__(self, encoder, config, tokenizer, args, num_labels):
#         super().__init__()
#         self.encoder = encoder
#         self.tokenizer = tokenizer
#         self.classifier = RobertaClassificationHead(config, num_labels)#共用主干，分类器不一样
#         #self.classifier1 = RobertaClassificationHead(config, num_labels)
#         self.args = args
#
#     def forward(self,
#                 input_ids=None,
#                 labels=None,
#                 soft_label=None,
#                 hard_label=None,
#                 best_beta=None):
#         DISTIL_TOKEN_ID = self.tokenizer.dis_token_id
#         # soft dis token location
#         locs = (input_ids == DISTIL_TOKEN_ID).nonzero(as_tuple=True)
#         locs = locs[1].tolist()
#        #学生的中间层特征
#         last_hidden_state = self.encoder(input_ids, attention_mask=input_ids.ne(self.tokenizer.pad_token_id)).last_hidden_state
#         logits_cls, logits_dis = self.classifier(last_hidden_state, dis_locations=locs, args=self.args)
#         # logits_dis----Zdis     logits_cls=Zcls
#         prob_cls = torch.softmax(logits_cls, dim=-1) #Hcls
#         prob_dis = nn.functional.log_softmax(logits_dis, dim=-1)#Hdis
#
#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             kl_loss_fct = nn.KLDivLoss(reduction="batchmean", log_target=True)#
#             loss_cls = loss_fct(logits_cls, labels)
#             # KL Loss with log softmax input and target
#             if soft_label is not None:
#                 loss_dis = kl_loss_fct(prob_dis, soft_label)
#                 return loss_cls, loss_dis
#             elif hard_label is not None:
#                 loss_dis = loss_fct(logits_dis, hard_label)
#                 return loss_cls, loss_dis
#         #做预测，Test
#         else:
#             #beta = self.args.beta
#             #prob = beta * prob_cls + (1-beta) * prob_dis
#             #return prob
#             #
#             if best_beta is not None:
#                 prob = best_beta * prob_cls + (1-best_beta) * prob_dis ##公式（5）
#                 return prob
#             betas = [0.5, 0.6, 0.7, 0.8, 0.9]
#             probs = []
#             for beta in betas:
#                 prob = beta * prob_cls + (1-beta) * prob_dis
#                 probs.append(prob)
#             return probs, betas
#
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F


class RobertaClassificationHead(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # layers for [CLS]
        self.cls_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.cls_out_proj = nn.Linear(config.hidden_size, num_labels)
        # layers for [SOFT DIS]
        self.dis_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dis_out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, dis_locations, args):
        # predict based on [CLS]
        x_cls = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x_cls = self.dropout(x_cls)
        x_cls = self.cls_dense(x_cls)
        x_cls = torch.tanh(x_cls)
        x_cls = self.dropout(x_cls)
        x_cls = self.cls_out_proj(x_cls)
        # predict based on [DIS]
        x_dis = []
        for i in range(len(dis_locations)):  # take <dis>
            x_dis_slice = features[i, dis_locations[i], :]
            x_dis.append(x_dis_slice.tolist())
        x_dis = torch.tensor(x_dis).to(args.device)
        x_dis = self.dropout(x_dis)
        x_dis = self.dis_dense(x_dis)
        x_dis = torch.tanh(x_dis)
        x_dis = self.dropout(x_dis)
        x_dis = self.dis_out_proj(x_dis)
        return x_cls, x_dis


class StudentBERT(nn.Module):
    def __init__(self, encoder, config, tokenizer, args, num_labels):
        super().__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.classifier1 = RobertaClassificationHead(config, num_labels)  # 对应 group 1, 2, 3
        #self.classifier2 = RobertaClassificationHead(config, num_labels)  # 对应 group 4, 5
        self.args = args
        #以下 3 行为添加
        self.conv1 = nn.Conv1d(in_channels=512, out_channels=768, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=768, out_channels=512, kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool1d(kernel_size=1, stride=1)


    def extract_feature_conv(self, x):
        x = F.relu(self.conv1(x))
        x = self.max_pool(x)
        x = F.relu(self.conv2(x))
        x = self.max_pool(x)
        return x

    def forward(self, input_ids=None, labels=None, soft_label=None, hard_label=None, groups=None, best_beta=None):
        DISTIL_TOKEN_ID = self.tokenizer.dis_token_id
        # soft dis token location
        locs = (input_ids == DISTIL_TOKEN_ID).nonzero(as_tuple=True)
        locs = locs[1].tolist()

        # 学生的中间层特征
        #last_hidden_state = self.encoder(input_ids,
        #                                attention_mask=input_ids.ne(self.tokenizer.pad_token_id)).last_hidden_state #torch.Size([1, 512, 768])

        #以下 5 行为添加
        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(self.tokenizer.pad_token_id), output_hidden_states=True, return_dict=True)
        cls_first = outputs.hidden_states[1]
        cls_last = outputs.hidden_states[-1]
        last_hidden_state = cls_first + cls_last
        last_hidden_state = self.extract_feature_conv(last_hidden_state)

        # 使用不同的分类器
        logits_cls, logits_dis = self.classifier1(last_hidden_state, dis_locations=locs, args=self.args)
        #logits_cls2, logits_dis2 = self.classifier2(last_hidden_state, dis_locations=locs, args=self.args)

        # # 根据 groups 选择合适的 logits
        # logits_cls = torch.empty_like(logits_cls1)
        # logits_dis = torch.empty_like(logits_dis1)
        #print(len(groups))
        # for i in range(len(groups)):
        #     if groups[i].item() in [0, 1, 3]:
        #         logits_cls[i, :] = logits_cls1[i]
        #         logits_dis[i, :] = logits_dis1[i]
        #     elif groups[i].item() in [2, 4, 5]:
        #         logits_cls[i, :] = logits_cls2[i]
        #         logits_dis[i, :] = logits_dis2[i]

        prob_cls = torch.softmax(logits_cls, dim=-1)  # Hcls
        prob_dis = nn.functional.log_softmax(logits_dis, dim=-1)  # Hdis

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            kl_loss_fct = nn.KLDivLoss(reduction="batchmean", log_target=True)
            loss_cls = loss_fct(logits_cls, labels)

            # KL Loss with log softmax input and target
            if soft_label is not None:
                loss_dis = kl_loss_fct(prob_dis, soft_label)
                return loss_cls, loss_dis
            elif hard_label is not None:#
                loss_dis = loss_fct(logits_dis, hard_label)
                return loss_cls, loss_dis
        else:
            if best_beta is not None:
                prob = best_beta * prob_cls + (1 - best_beta) * prob_dis  # 公式（5）
                return prob
            betas = [0.5, 0.6, 0.7, 0.8, 0.9]
            probs = []
            for beta in betas:
                prob = beta * prob_cls + (1 - beta) * prob_dis
                probs.append(prob)
            return probs, betas

