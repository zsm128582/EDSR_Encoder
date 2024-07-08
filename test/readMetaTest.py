import scipy.io

meta = scipy.io.loadmat('/home/zengshimao/code/Super-Resolution-Neural-Operator/test/meta.mat')
print(meta.keys())
synsets = meta['synsets']
print(synsets)
first_synset = synsets[0][0]
print(f"ILSVRC2012_ID: {first_synset[0][0]}")
print(f"WNID: {first_synset[1][0]}")
print(f"words: {first_synset[2][0]}")
print(f"gloss: {first_synset[3][0]}")
print(f"num_children: {first_synset[4][0]}")
print(f"children: {first_synset[5]}")
print(f"wordnet_height: {first_synset[6][0]}")
print(f"num_train_images: {first_synset[7][0]}")