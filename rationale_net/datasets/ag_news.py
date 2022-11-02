import gzip
import re
import tqdm
from rationale_net.utils.embedding import get_indices_tensor
from rationale_net.datasets.factory import RegisterDataset
from rationale_net.datasets.abstract_dataset import AbstractDataset
from torchtext.datasets import AG_NEWS
import random
import string

CATEGORIES = ["1", "2", "3", "4"]

def prep_data(raw_data):
    data = []
    for row in raw_data:
        label = row[0]
        text = row[1]
        text = text.lower()
        text = "".join([char for char in text if char not in string.punctuation])
        """
        if label == "1":
            label_name = "some_label"
        elif label == "2":
            label_name = "some_label"
        elif label == "3":
            label_name = "some_label"
        elif label == "4":
            label_name = "some_label"
        """
        data.append((text, label-1, label-1))#labels: 1,2,3,4 to 0,1,2,3
    return(data)

def get_balanced_dev(train, label_amount = 6000):
    home = []
    not_home = []
    for i in train:
        if i[1] == 0:
            home.append(i)
        else:
            not_home.append(i)
    to_dev = home[:half_size] + not_home[:half_size]
    to_train = home[half_size:] + not_home[half_size:]
    random.seed(2021)
    random.shuffle(to_dev)
    random.shuffle(to_train)
    return(to_train, to_dev)


@RegisterDataset('ag_news')
class AGNEWSDataset(AbstractDataset):
    def __init__(self, args, word_to_indx, name, max_length=80):
        self.args = args
        self.args.num_class = len(CATEGORIES)
        self.name = name
        self.dataset = []
        self.word_to_indx  = word_to_indx
        self.max_length = max_length
        self.class_balance = {}
        train_data, test_data = AG_NEWS()
        
        
        data_dev = []
        random.seed(2021)
        

        if name in ['train', 'dev']:
            train_data = prep_data(train_data)
            random.shuffle(train_data)
            if name == 'train':
                data = train_data
            else:
                data = test_data[:1000]
            
        else:
            test_data = prep_data(test_data)
            data = test_data
            
        for indx, _sample in tqdm.tqdm(enumerate(data)):
            sample = self.processLine(_sample)
            if not sample['y'] in self.class_balance:
                self.class_balance[ sample['y'] ] = 0
            self.class_balance[ sample['y'] ] += 1
            self.dataset.append(sample)
        print ("Class balance", self.class_balance)

        if args.class_balance:
            raise NotImplementedError("NewsGroup dataset doesn't support balanced sampling")
        if args.objective == 'mse':
            raise NotImplementedError("News Group does not support Regression objective")

    ## Convert one line from beer dataset to {Text, Tensor, Labels}
    def processLine(self, row):
        text, label, label_name = row
        text = " ".join(text.split()[:self.max_length])
        x =  get_indices_tensor(text.split(), self.word_to_indx, self.max_length)
        sample = {'text':text,'x':x, 'y':label, 'y_name': label_name}
        return sample
