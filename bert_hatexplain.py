import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain")
model = AutoModelForSequenceClassification.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain")

# Get the embeddings generate from HateXPlain
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output
    return hook

def hatexplian(text):
  special_tokens_dict = {'additional_special_tokens': ['chinavirus', 'cherry picker', 'china virus','coronaviruschina', 'ccpvirus',
                                                       'kungflu','chinese virus','wuhanvirus', 'wuhan virus', 'maskless', 'womensuch', 'walkaway',
                                                       'antimask','antivaccine', 'novaccine', 'maskoff', 'boomer', 'maskfree', 'babyboomer',
                                                       'boomerremover', 'boomer remover', 'wuflu']}
  num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
  model.resize_token_embeddings(len(tokenizer))
  model.bert.register_forward_hook(get_activation('bert'))

  tokens = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
  with torch.no_grad():
    output = model(**tokens)

  hatexbert = activation['bert'][1]

  return hatexbert