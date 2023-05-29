import pandas as pd
from Bio.SeqUtils import molecular_weight
import gentrl
df = pd.read_csv('peptides-complete.csv')
# df['mol_weight'] = df['SEQUENCE'].apply(molecular_weight, args=("protein"))
# df.to_csv('peptides_mol_weight.csv', index=None)


valid_seqs = set()
for seq in df['SEQUENCE']:
    try:
        if len(seq.split(",")) == 1:
            # print(seq)
            if "X" in seq:
                print("X present")
                # valid_seqs.add(seq.upper())
            else:
                if "x" not in seq:
                    if len(seq) > 4:
                        seq = seq.replace(" ", "")
                        valid_seqs.add(seq.upper())
    except AttributeError as e:
        pass

# print(valid_seqs)

mol_weights = []
seqs = []
for se in valid_seqs:
    seqs.append(se)
    weight = molecular_weight(se, 'protein')
    mol_weights.append(weight)

d = {'SEQUENCES': seqs, 'Mol_weight': mol_weights}
df_1 = pd.DataFrame(d)
df_1.to_csv('./peptides_mol_weight.csv', index=None)


enc = gentrl.RNNEncoder(latent_size=50)
dec = gentrl.DilConvDecoder(latent_input_size=50)
model = gentrl.GENTRL(enc, dec, 50 * [('c', 20)], [('c', 20)], beta=0.001)


md = gentrl.MolecularDataset(sources=[
    {'path': './peptides_mol_weight.csv',
     'sequences': 'SEQUENCES',
     'prob': 1,
     'mol_weight' : 'Mol_weight',
    }],
    props=['mol_weight'])


from torch.utils.data import DataLoader
train_loader = DataLoader(md, batch_size=50, shuffle=True, drop_last=True)
model.train_as_vaelp(train_loader, lr=1e-4)
model.save('./saved_gentrl/')