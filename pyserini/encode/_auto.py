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
import numpy as np
from sklearn.preprocessing import normalize
from transformers import AutoModel, AutoTokenizer
from collections import OrderedDict

from pyserini.encode import DocumentEncoder, QueryEncoder

from sentence_transformers import SentenceTransformer

class AutoDocumentEncoder(DocumentEncoder):
    def __init__(self, model_name, tokenizer_name=None, device='cuda:0', pooling='cls', l2_norm=False):
        self.device = device
        self.from_local = False
        if os.path.exists(model_name):
            self.from_local = True
            self.load_from_checkpoint(model_name)
        else:
            self.model = AutoModel.from_pretrained(model_name)
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name)
            except:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name, use_fast=False)
        
        self.model.to(self.device)

        self.has_model = True
        self.pooling = pooling
        self.l2_norm = l2_norm

    def encode(self, texts, titles=None, max_length=256, add_sep=False, **kwargs):
        shared_tokenizer_kwargs = dict(
            max_length=max_length,
            truncation=True,
            padding='longest',
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt',
            add_special_tokens=True,
        )
        input_kwargs = {}
        if not add_sep:
            input_kwargs["text"] = [f'{title} {text}' for title, text in zip(titles, texts)] if titles is not None else texts
        else:
            if titles is not None:
                input_kwargs["text"] = titles
                input_kwargs["text_pair"] = texts
            else:
                input_kwargs["text"] = texts

        inputs = self.tokenizer(**input_kwargs, **shared_tokenizer_kwargs)
        inputs.to(self.device)
        outputs = self.model(**inputs)
        if self.pooling == "mean":
            embeddings = self._mean_pooling(outputs[0], inputs['attention_mask']).detach().cpu().numpy()
        else:
            embeddings = outputs[0][:, 0, :].detach().cpu().numpy()
        if self.l2_norm:
            embeddings = normalize(embeddings, axis=1)
        return embeddings
    
    def load_from_checkpoint(self, model_name):
        self.model = AutoModel.from_pretrained('bert-base-uncased')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        checkpoint_dict = torch.load(model_name, map_location=torch.device('cpu'))
        state = checkpoint_dict['state_dict']
        new_state = OrderedDict((key.replace('context_encoder.transformer.', ''), value) for key, value in state.items() if key.startswith('context_encoder.transformer.'))
        self.model.load_state_dict(new_state, strict=False)


class AutoQueryEncoder(QueryEncoder):
    def __init__(self, model_name: str, tokenizer_name: str = None, device: str = 'cpu',
                 pooling: str = 'cls', l2_norm: bool = False, prefix=None):
        self.device = device
        self.from_local = False
        if os.path.exists(model_name):
            self.from_local = True
            self.load_from_checkpoint(model_name)
        else:
            self.model = AutoModel.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name)

        self.model.to(self.device)
        self.pooling = pooling
        self.l2_norm = l2_norm
        self.prefix = prefix

    def encode(self, query: str, **kwargs):
        if self.prefix:
            query = f'{self.prefix} {query}'
        inputs = self.tokenizer(
            query,
            add_special_tokens=True,
            return_tensors='pt',
            truncation='only_first',
            padding='longest',
            return_token_type_ids=False,
        )
        inputs.to(self.device)
        outputs = self.model(**inputs)[0].detach().cpu().numpy()
        if self.pooling == "mean":
            embeddings = np.average(outputs, axis=-2)
        else:
            embeddings = outputs[:, 0, :]
        if self.l2_norm:
            embeddings = normalize(outputs, norm='l2')
        return embeddings
    
    def load_from_checkpoint(self, model_name):
        self.model = AutoModel.from_pretrained('bert-base-uncased')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        checkpoint_dict = torch.load(model_name, map_location=torch.device('cpu'))
        state = checkpoint_dict['state_dict']
        new_state = OrderedDict((key.replace('query_encoder.transformer.', ''), value) for key, value in state.items() if key.startswith('query_encoder.transformer.'))
        self.model.load_state_dict(new_state, strict=False)



class SentenceTransformerDocumentEncoder(DocumentEncoder):
    def __init__(self, model_name, device='cuda:0'):
        self.device = device
        self.from_local = False
        if os.path.exists(model_name):
            self.from_local = True
            self.load_from_checkpoint(model_name)
        else:
            self.model = SentenceTransformer(model_name, device=self.device)

        self.model.to(self.device)
    
    
    def encode(self, texts, titles=None, max_length=None, batch_size=32, **kwargs):
        if max_length is not None:
            self.model.max_seq_length = max_length
        sentences = [f'{title} {text}' for title, text in zip(titles, texts)] if titles is not None else texts
        embeddings = self.model.encode(sentences, batch_size=batch_size)
        return embeddings

    def load_from_checkpoint(self, model_name):
        if 'gtr.large' in model_name.lower(): # gtr-large
            self.model = SentenceTransformer('gtr-t5-large')
        elif 'gtr' in model_name.lower(): # gtr-base
            self.model = SentenceTransformer('gtr-t5-base')

        checkpoint_dict = torch.load(model_name, map_location=torch.device('cpu'))
        state = checkpoint_dict['state_dict']
        new_state = OrderedDict((key.replace('query_encoder.transformer.', 'auto_model.'), value) for key, value in state.items() if key.startswith('query_encoder.transformer.'))
        """
        SentenceTransformer(
            (0): Transformer({'max_seq_length': 512, 'do_lower_case': False}) with Transformer model: T5EncoderModel 
            (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False})
            (2): Dense({'in_features': 768, 'out_features': 768, 'bias': False, 'activation_function': 'torch.nn.modules.linear.Identity'})
            (3): Normalize()
        )
        """
        
        self.model[0].load_state_dict(new_state)


class InstructorDocumentEncoder(DocumentEncoder):
    def __init__(self, model_name, device='cuda:0'):
        self.device = device
        from InstructorEmbedding import INSTRUCTOR
        self.model = INSTRUCTOR(model_name)
        self.model.to(self.device)

    def encode(self, texts, titles=None, max_length=None, batch_size=32, **kwargs):
        sentences = [f'{title} {text}' for title, text in zip(titles, texts)] if titles is not None else texts
        sentences = [["Represent the document for retrieval: ", sent] for sent in sentences]
        embeddings = self.model.encode(sentences, batch_size=batch_size)
        return embeddings


class SentenceTransformerQueryEncoder(QueryEncoder):
    def __init__(self, model_name: str, tokenizer_name: str = None, device: str = 'cpu', prefix=None):
        assert prefix is None, "SentenceTransformerQueryEncoder does not support prefix"
        assert tokenizer_name is None, "SentenceTransformerQueryEncoder does not support tokenizer_name"
        self.device = device
        self.model_name = model_name
        self.from_local = False
        if os.path.exists(model_name):
            self.from_local = True
            self.load_from_checkpoint(model_name)
        else:
            if 'qwen' in model_name.lower(): # Alibaba-NLP/gte-Qwen2-1.5B-instruct
                self.model = SentenceTransformer(model_name, device=self.device, trust_remote_code=True)
            else:
                self.model = SentenceTransformer(model_name, device=self.device)
    
    def encode(self, query):
        if 'qwen' in self.model_name.lower(): # Alibaba-NLP/gte-Qwen2-1.5B-instruct
            embeddings = self.model.encode(query, prompt_name='query')
        elif 'e5' in self.model_name.lower(): # intfloat/e5-mistral-7b-instruct
            embeddings = self.model.encode(query, prompt_name='web_search_query')
        else:
            embeddings = self.model.encode(query)
        return embeddings


    def load_from_checkpoint(self, model_name):
        if 'gtr.large' in model_name.lower(): # gtr-large
            self.model = SentenceTransformer('gtr-t5-large')
        elif 'gtr' in model_name.lower(): # gtr-base
            self.model = SentenceTransformer('gtr-t5-base')
        
        checkpoint_dict = torch.load(model_name, map_location=torch.device('cpu'))
        state = checkpoint_dict['state_dict']
        new_state = OrderedDict((key.replace('query_encoder.transformer.', 'auto_model.'), value) for key, value in state.items() if key.startswith('query_encoder.transformer.'))
        self.model[0].load_state_dict(new_state)


class InstructorQueryEncoder(QueryEncoder):
    def __init__(self, model_name: str, tokenizer_name=None, device: str = 'cuda:0', prefix=None):
        self.device = device
        from InstructorEmbedding import INSTRUCTOR
        self.model = INSTRUCTOR(model_name)
        self.model.to(self.device)

    def encode(self, query):
        query_chunks = query.split('[SEP]')
        assert len(query_chunks) == 2, "InstructorQueryEncoder requires a query with '[SEP]' separator"
        embeddings = self.model.encode([[component.strip() for component in query_chunks]])
        return embeddings.flatten()