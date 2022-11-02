import gzip
import re
import tqdm
from rationale_net.utils.embedding import get_indices_tensor
from rationale_net.datasets.factory import RegisterDataset
from rationale_net.datasets.abstract_dataset import AbstractDataset
import random
import csv, string



#SMALL_TRAIN_SIZE = 800
CATEGORIES = ["home", "not_home"]

def preprocess_data(data):
    processed_data = []
    for indx, sample in enumerate(data['data']):
        text, label = sample, data['target'][indx]
        label_name = data['target_names'][label]
        text = re.sub('\W+', ' ', text).lower().strip()
        processed_data.append( (text, label, label_name) )
    return processed_data

def get_balanced_dev(train, half_size = 2000):
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


def read_data(path):
    f = open(path, encoding="utf-8")
    csvreader = csv.reader(f, delimiter='\t')
    data = []
    for row in csvreader:
        text = row[0]
        label = row[1]
        text = text.lower()
        text = "".join([char for char in text if char not in string.punctuation])
        if label == "home":
            label_num = 0
        elif label == "not_home":
            label_num = 1
        else:
            print(label)
        data.append((text, label_num, label))
    f.close()
    return(data)

@RegisterDataset('healthlink')
class MoviesDataset(AbstractDataset):
    def __init__(self, args, word_to_indx, name, max_length=80):
        self.args = args
        self.args.num_class = len(CATEGORIES)
        self.name = name
        self.dataset = []
        self.word_to_indx  = word_to_indx
        self.max_length = max_length
        self.class_balance = {}
        
        
        
        #num_train = int(len(data)*.8)
        if name in ['train', 'dev']:
            data_train = read_data(R"D:\Datasets\health link\train.csv")
            if name == 'train':
                data, _ = get_balanced_dev(data_train, 2000)
                print("first train data:", data[0])
            else:
                _, data = get_balanced_dev(data_train, 2000)
            
        else:
            data_test = read_data(R"D:\Datasets\health link\test.csv")
            data = data_test
            
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
