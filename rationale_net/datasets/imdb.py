import gzip
import re
import tqdm
from rationale_net.utils.embedding import get_indices_tensor
from rationale_net.datasets.factory import RegisterDataset
from rationale_net.datasets.abstract_dataset import AbstractDataset
from torchtext.datasets import IMDB
import random
import string

CATEGORIES = ["pos", "neg"]

#IMDB() loads neg and pos with uncertain execution order
#to be reproducible, always get neg first
def prep_data(raw_data):
    pos_data = []
    neg_data = []
    for row in raw_data:
        label = row[0]
        text = row[1]
        text = text.lower()
        text = "".join([char for char in text if char not in string.punctuation])
        if label == "neg":
            label_index = 1
            neg_data.append((text, label_index, label))
    for row in raw_data:
        label = row[0]
        text = row[1]
        text = text.lower()
        text = "".join([char for char in text if char not in string.punctuation])
        if label == "pos":
            label_index = 0
            pos_data.append((text, label_index, label))
    return(pos_data, neg_data)

def split_train_dev(pos, neg, half_dev_size=2500):
    pos_data = pos
    neg_data = neg
    random.seed(2021)
    random.shuffle(pos_data)
    random.shuffle(neg_data)
    dev = pos[:half_dev_size] + neg[:half_dev_size]
    train = pos[half_dev_size:] + neg[half_dev_size:]
    random.shuffle(train)
    return(train, dev)


@RegisterDataset('imdb')
class IMDBDataset(AbstractDataset):
    def __init__(self, args, word_to_indx, name, max_length=200):
        self.args = args
        self.args.num_class = len(CATEGORIES)
        self.name = name
        self.dataset = []
        self.word_to_indx  = word_to_indx
        self.max_length = max_length
        self.class_balance = {}

        

        if name in ["train", "dev"]:
            pos, neg = prep_data(IMDB(split="train"))
            train, dev = split_train_dev(pos, neg, half_dev_size=2500)

            if name == "train":
                data = train
                print("train data size:", len(data))
            else:
                data = dev
                print("dev data size:", len(data))
        else:
            pos, neg = prep_data(IMDB(split="test"))
            data = pos+neg
            print("test data size:", len(data))
            
        for indx, _sample in tqdm.tqdm(enumerate(data)):
            sample = self.processLine(_sample)
            if not sample['y'] in self.class_balance:
                self.class_balance[ sample['y'] ] = 0
            self.class_balance[ sample['y'] ] += 1
            self.dataset.append(sample)
        print ("Class balance", self.class_balance)

        if args.class_balance:
            raise NotImplementedError("Dataset doesn't support balanced sampling")
        if args.objective == 'mse':
            raise NotImplementedError("Does not support Regression objective")

    ## Convert one line from beer dataset to {Text, Tensor, Labels}
    def processLine(self, row):
        text, label, label_name = row
        text = " ".join(text.split()[:self.max_length])
        x =  get_indices_tensor(text.split(), self.word_to_indx, self.max_length)
        sample = {'text':text,'x':x, 'y':label, 'y_name': label_name}
        return sample
