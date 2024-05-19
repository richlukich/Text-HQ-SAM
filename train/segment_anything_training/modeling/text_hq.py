import torch
import cv2
from transformers import BlipProcessor, BlipForConditionalGeneration

import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch import Tensor, nn
from torchvision.ops.boxes import nms
from transformers import BertConfig, BertModel, BertPreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from PIL import Image
import numpy as np
from segment_anything_training.modeling.bertwarper import (
    generate_masks_with_special_tokens,
    generate_masks_with_special_tokens_and_transfer_map,
)
from transformers import AutoTokenizer, BertModel, BertTokenizer, RobertaModel, RobertaTokenizerFast
from segment_anything_training.utils.get_tokenizer import get_tokenlizer, get_pretrained_language_model
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
class BertEmbedding(nn.Module):
    def __init__(self,
                 hidden_dim = 256,
                 text_encoder_type = 'bert-base-uncased',
                 max_text_len = 64,
                 sub_sentence_present = True,
                 device = 'cuda:0'):
        super().__init__()
        
        self.bert = get_pretrained_language_model(text_encoder_type).to(device)
        #self.bert = BertModelWarper(bert_model=self.bert).to(device)
        self.bert_token = nn.Linear(768, hidden_dim, bias=True).to(device)
        self.max_text_len = max_text_len
        self.sub_sentence_present = sub_sentence_present

        nn.init.constant_(self.bert_token.bias.data, 0)
        nn.init.xavier_uniform_(self.bert_token.weight.data)

        #self.specical_tokens = self.tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]","[PAD]", ".", "?"])
        self.device = device
    def forward(self, tokenized):
        
        ids = tokenized['ids']
        mask = tokenized['mask'],
        token_type_ids = tokenized['token_type_ids']
        
       

        bert_out = self.bert(ids,attention_mask = mask[0], token_type_ids = token_type_ids)
        last_hidden_state_cls = bert_out[0][:, 0, :]
        
        return last_hidden_state_cls








    