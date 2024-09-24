#
# Pyserini: Reproducible IR research with sparse and dense representations
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os
import torch
from collections import OrderedDict

from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer, AutoModel, AutoTokenizer

from pyserini.encode import DocumentEncoder, QueryEncoder


class DprDocumentEncoder(DocumentEncoder):
    def __init__(self, model_name, tokenizer_name=None, device='cuda:0'):
        self.device = device
        self.from_local = False
        if os.path.exists(model_name):
            self.from_local = True
            self.load_from_checkpoint(model_name)
        else:
            self.model = DPRContextEncoder.from_pretrained(model_name)
            self.tokenizer = DPRContextEncoderTokenizer.from_pretrained(tokenizer_name or model_name)

        self.model.to(self.device)

    def encode(self, texts, titles=None,  max_length=256, **kwargs):
        if titles:
            inputs = self.tokenizer(
                titles,
                text_pair=texts,
                max_length=max_length,
                padding='longest',
                truncation=True,
                add_special_tokens=True,
                return_tensors='pt'
            )
        else:
            inputs = self.tokenizer(
                texts,
                max_length=max_length,
                padding='longest',
                truncation=True,
                add_special_tokens=True,
                return_tensors='pt'
            )
        
        inputs.to(self.device)
        
        if self.from_local:
            return self.model(**inputs)[0][:, 0, :].detach().cpu().numpy()
        else:
            return self.model(inputs["input_ids"]).pooler_output.detach().cpu().numpy()
    
    def load_from_checkpoint(self, model_name):
        self.model = AutoModel.from_pretrained('bert-base-uncased')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        checkpoint_dict = torch.load(model_name, map_location=torch.device('cpu'))
        state = checkpoint_dict['state_dict']

        print('Loading from a dual model...')
        new_state = OrderedDict((key.replace('context_encoder.transformer.', ''), value) for key, value in state.items() if key.startswith('context_encoder.transformer.'))
 
        self.model.load_state_dict(new_state, strict=False)


class DprQueryEncoder(QueryEncoder):
    def __init__(self, model_name: str, tokenizer_name: str = None, device: str = 'cpu'):
        self.device = device
        self.from_local = False
        if os.path.exists(model_name):
            self.from_local = True
            self.load_from_checkpoint(model_name)
        else:
            self.model = DPRQuestionEncoder.from_pretrained(model_name)
            self.tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(tokenizer_name or model_name)

        self.model.to(self.device)

    def encode(self, query: str, **kwargs):
        input_ids = self.tokenizer(query, return_tensors='pt')
        input_ids.to(self.device)

        if self.from_local:
            embeddings = self.model(**input_ids)[0][:, 0, :].detach().cpu().numpy()
        else:
            embeddings = self.model(input_ids["input_ids"]).pooler_output.detach().cpu().numpy()
        return embeddings.flatten()

    def load_from_checkpoint(self, model_name):
        self.model = AutoModel.from_pretrained('bert-base-uncased')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        checkpoint_dict = torch.load(model_name, map_location=torch.device('cpu'))
        state = checkpoint_dict['state_dict']
        new_state = OrderedDict((key.replace('query_encoder.transformer.', ''), value) for key, value in state.items() if key.startswith('query_encoder.transformer.'))
        self.model.load_state_dict(new_state, strict=False)

