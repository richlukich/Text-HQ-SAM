import torch
import cv2
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoTokenizer, BertModel, BertTokenizer, RobertaModel, RobertaTokenizerFast
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch import Tensor, nn
from torchvision.ops.boxes import nms
from transformers import BertConfig, BertModel, BertPreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from PIL import Image
import numpy as np
import os

class ImageCaption():
    def __init__(self, device):
        self.device = device
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to(device)
        self.k = 0
    def forward(self, image, save = False):
        if save:
            cv2.imwrite(f'/home/vetoshkin_ln/text_sam_hq/test/test_{self.k}.png', image)
            self.k += 1
        image1 = Image.fromarray(np.uint8(image))
        inputs = self.processor(image1, return_tensors="pt").to(self.device, torch.float16)
        out = self.model.generate(**inputs)
        text = self.processor.decode(out[0], skip_special_tokens=True)
        return text
    

class Tokenizer():
    def __init__(self,
                 device,
                 text_encoder_type = 'bert-base-uncased',
                 max_text_len = 15):
        self.tokenizer = get_tokenlizer(text_encoder_type)
        self.device = device
        self.max_text_len = max_text_len
    def tokenize(self, caption):
        inputs = self.tokenizer.encode_plus(
            caption ,
            None,
            pad_to_max_length=True,
            add_special_tokens=True,
            return_attention_mask=True,
            max_length=self.max_text_len,
            return_tensors="pt"
        ).to(self.device)
        
        ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        mask = inputs["attention_mask"]
        return {'ids': ids,
                'token_type_ids': token_type_ids,
                'mask': mask}
    

def get_tokenlizer(text_encoder_type):
    if not isinstance(text_encoder_type, str):
        # print("text_encoder_type is not a str")
        if hasattr(text_encoder_type, "text_encoder_type"):
            text_encoder_type = text_encoder_type.text_encoder_type
        elif text_encoder_type.get("text_encoder_type", False):
            text_encoder_type = text_encoder_type.get("text_encoder_type")
        elif os.path.isdir(text_encoder_type) and os.path.exists(text_encoder_type):
            pass
        else:
            raise ValueError(
                "Unknown type of text_encoder_type: {}".format(type(text_encoder_type))
            )
    print("final text_encoder_type: {}".format(text_encoder_type))

    tokenizer = AutoTokenizer.from_pretrained(text_encoder_type)
    return tokenizer


def get_pretrained_language_model(text_encoder_type):
    if text_encoder_type == "bert-base-uncased" or (os.path.isdir(text_encoder_type) and os.path.exists(text_encoder_type)):
        return BertModel.from_pretrained(text_encoder_type)
    if text_encoder_type == "roberta-base":
        return RobertaModel.from_pretrained(text_encoder_type)

    raise ValueError("Unknown text_encoder_type {}".format(text_encoder_type))
