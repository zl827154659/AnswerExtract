import torch
from transformers import BertModel, BertPreTrainedModel
import torch.nn as nn
from torchcrf import CRF


class BertNER(BertPreTrainedModel):
    def __init__(self, config):
        super(BertNER, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(config.num_labels, batch_first=True)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, inputs_embeds=None, head_mask=None):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        # 得到判别值
        logits = self.classifier(sequence_output)
        outputs = (logits,)
        if labels is not None:
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(-100).type_as(labels)
                )  # [-100, -100, 1, 0...]

            else:
                active_logits = logits.view(-1, self.num_labels)
                active_labels = labels.view(-1)
            select_index = []
            final_labels = []
            for index, label in enumerate(active_labels):
                if label != -100:
                    final_labels.append(label)
                    select_index.append(index)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            active_logits.to(device)
            select_index = torch.tensor(select_index, device=active_logits.device)
            final_labels = torch.tensor(final_labels, device=active_logits.device).unsqueeze(0)
            final_logits = active_logits.index_select(0, select_index).unsqueeze(0)
            loss = self.crf(final_logits, final_labels) * (-1)
            outputs = (loss,) + outputs

        # contain: (loss), scores
        return outputs
