from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights, vit_b_16, ViT_B_16_Weights
from transformers import BertModel, BertTokenizer, BertForSequenceClassification
import torch

def sigmoid(x):
    return 1/(1+torch.exp(-x))

class BertWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(
            input_ids=x[:, :, 0],
            attention_mask=x[:, :, 1],
            token_type_ids=x[:, :, 2]).logits


def get_model(pretrained, model, num_classes):
    weights = {"resnet18": ResNet18_Weights,
               "resnet50": ResNet50_Weights,
               "ViT": ViT_B_16_Weights,
               "BERT": None}
    if pretrained:
        weights = weights[model]
    else:
        weights = None

    if model == 'resnet18':
        net = resnet18(weights=weights)
        net.fc = torch.nn.Linear(net.fc.in_features, num_classes)
    elif model == 'resnet50':
        net = resnet50(weights=weights)
        net.fc = torch.nn.Linear(net.fc.in_features, num_classes)
    elif model == "BERT":
        # Do not support non-pretrained BERT since it does not make sense
        net = BertWrapper(BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_classes))
    else:
        print('choose a valid model - resnet18, resnet50, ViT, BERT', flush=True)
        raise ValueError

    return net
