import torch
from transformers import AutoTokenizer, AutoModel

sent_token = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
sent_model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')

# Sentence Bert apply function

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
  token_embeddings = model_output[0] #First element of model_output contains all token embeddings
  # print('token size:',token_embeddings.shape)

  
  input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
  return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def sentbert(text):
  special_tokens_dict = {'additional_special_tokens': ['chinavirus', 'cherry picker', 'china virus','coronaviruschina', 'ccpvirus',
                                                       'kungflu','chinese virus','wuhanvirus', 'wuhan virus', 'maskless', 'womensuch', 'walkaway',
                                                       'antimask','antivaccine', 'novaccine', 'maskoff', 'boomer', 'maskfree', 'babyboomer',
                                                       'boomerremover', 'boomer remover', 'wuflu']}
  num_added_toks = sent_token.add_special_tokens(special_tokens_dict)
  sent_model.resize_token_embeddings(len(sent_token))
  tokens = sent_token(text, padding=True, truncation=True, return_tensors='pt')
  with torch.no_grad():
    output = sent_model(**tokens)

  sentence_embeddings = mean_pooling(output, tokens['attention_mask'])

  return sentence_embeddings