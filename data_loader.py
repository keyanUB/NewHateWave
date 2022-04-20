import torch
from torch.utils.data import DataLoader, Dataset

class TheData(Dataset):
  def __init__(self, dataset):
    self.data = dataset
  
  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    '''data keys:
      'label', 'tweet', 'category', 
      'freshness', 'sentiment', 'target', 'othering', 'derogatory', 'threat',
      'sent_emb'
    '''
    # read the data
    text = self.data[index]['tweet']
    category = self.data[index]['category']
    label = torch.LongTensor([self.data[index]['label']])
    sent_emb = torch.FloatTensor(self.data[index]['sent_emb'][0])

    # combine the attributes
    attributes = torch.FloatTensor([self.data[index]['freshness'], self.data[index]['sentiment'], self.data[index]['target'], self.data[index]['othering'], 
                                    self.data[index]['derogatory'], self.data[index]['threat']
                                    ])



    return {'text': text, 'category': category, 'label':label, 'attributes': attributes, 'sent_emb': sent_emb}

def get_data_loaders(dataset, batch_size, num_worker):
  data = TheData(
      dataset
  )
  
  data_loader = DataLoader(
    data,
    batch_size = 16,
    shuffle = True,
    num_workers = 2
  )

  return {'data_loader': data_loader, 'dataset': data} 
