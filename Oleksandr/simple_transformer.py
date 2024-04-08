import random
import torch
import math


torch.manual_seed(42)


class Tokenizer:
  def __init__(self):
    f = open('/root/tiny-transformer/names.txt', 'r')
    names = f.read().splitlines()
    vocab = ['<pad>', '<eos>', '<sos>'] + sorted(set(''.join(names)))
    self.stoi = {c:i for i, c in enumerate(vocab)}
    self.itos = {i:c for i, c in enumerate(vocab)}
    self.vocab_size = len(vocab)
    f.close()

  def encode(self, name):
    return [self.stoi[c] for c in name]

  def decode(self, tokens):
    return ''.join([self.itos[t] for t in tokens])


tokenizer = Tokenizer()
# print(tokenizer.vocab_size)  # 29
foo = tokenizer.encode('john') # [2, 12, 17, 10, 16, 1]
bar = tokenizer.decode(foo)    # john


class Dataset(torch.utils.data.Dataset):
  def __init__(self):
    f = open('/root/tiny-transformer/names.txt', 'r')
    self.names = f.read().split('\n')
    self.tokenizer = Tokenizer()
    f.close()

  def __len__(self):
    return len(self.names)

  def __getitem__(self, idx):
    name = self.names[idx]
    return torch.tensor(self.tokenizer.encode(name))


ds = Dataset()
dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)


class BesSimpleTransformer(torch.nn.Module):
  def __init__(self):
    super(BesSimpleTransformer, self).__init__()
    # Embedding part of the model
    self.embedding    = torch.nn.Embedding(29, 7)
    self.pos_emb      = self.get_pos_matrix()
    # Mask tensor trick
    self.register_buffer('mask', torch.tril(torch.ones(19, 19)))
    # First decoder block
    self.layer_00_key = torch.nn.Linear(7, 11)
    self.layer_00_qry = torch.nn.Linear(7, 11)
    self.layer_00_val = torch.nn.Linear(7, 11)
    self.layer_00_ffw = torch.nn.Linear(11, 7)
    # Second decoder block
    self.layer_01_key = torch.nn.Linear(7, 11)
    self.layer_01_qry = torch.nn.Linear(7, 11)
    self.layer_01_val = torch.nn.Linear(7, 11)
    self.layer_01_ffw = torch.nn.Linear(11, 7)
    # Output of the model
    self.map_to_vocab = torch.nn.Linear(7, 29)

  def forward(self, x):
    emb = self.embedding(x)
    pos = self.pos_emb[0:x.shape[0], :]
    emb = emb + pos

    key = self.layer_00_key(emb)
    qry = self.layer_00_qry(emb)
    val = self.layer_00_val(emb)
    att = torch.mm(qry, key.t())
    msk = self.mask[0:x.shape[0], 0:x.shape[0]]
    att = att.masked_fill(msk == 0, float('-inf'))
    att = torch.nn.functional.softmax(att, dim=1)
    res = torch.mm(att, val)
    res = self.layer_00_ffw(res)

    key = self.layer_01_key(res)
    qry = self.layer_01_qry(res)
    val = self.layer_01_val(res)
    att = torch.mm(qry, key.t())
    msk = self.mask[0:x.shape[0], 0:x.shape[0]]
    att = att.masked_fill(msk == 0, float('-inf'))
    att = torch.nn.functional.softmax(att, dim=1)
    res = torch.mm(att, val)
    res = self.layer_01_ffw(res)

    out = self.map_to_vocab(res)
    return out

  def get_pos_matrix(self):
    store = torch.zeros(19, 7)
    for pos in range(19):
      for i in range(0, 7, 2):
        denominator = 10000 ** (2 * i / 7)
        store[pos, i] = math.sin(pos / denominator)
        if i + 1 < 7: store[pos, i + 1] = math.cos(pos / denominator)
    return store


m = BesSimpleTransformer()
opt = torch.optim.SGD(m.parameters(), lr=0.01)


for epoch in range(10):
  for idx, batch in enumerate(dl):

    sos = torch.tensor([2])
    eos = torch.tensor([1])

    x = batch[0]
    x = torch.cat([sos, x])
    y = torch.cat([x[1:], eos])

    p = m(x)
    l = torch.nn.functional.cross_entropy(p, y)
    if idx % 1000 == 0: print("Loss:", l.item())
    l.backward()
    opt.step()
    opt.zero_grad()

  x = tokenizer.decode([random.randint(3, 25)])
  x = torch.cat([sos, torch.tensor(tokenizer.encode([x]))])
  while True:
    p = m(x)
    p = torch.nn.functional.softmax(p, dim=1)
    p = torch.argmax(p, dim=1)
    x = torch.cat([x, p[-1].unsqueeze(0)])
    if p[-1] == 1 or len(p.tolist()) == 17: break
  print("Generate:", tokenizer.decode(x.tolist()))