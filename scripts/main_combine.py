from os.path import dirname, realpath
import sys
sys.path.append(dirname(dirname(realpath(__file__))))
import argparse
import rationale_net.datasets.factory as dataset_factory
import rationale_net.utils.embedding as embedding
import rationale_net.utils.model as model_factory
import rationale_net.utils.generic as generic
import rationale_net.learn.train as train
import os
import torch
import datetime
import pickle
import pdb
torch.manual_seed(2021)

gen_path1 = os.path.join("snapshot", "t1.pt.gen")
enc_path1 = os.path.join("snapshot", "t1.pt")
gen_path2 = os.path.join("snapshot", "t2.pt.gen")
enc_path2 = os.path.join("snapshot", "t2.pt")



def load_models(gen_path1, enc_path1, gen_path2, enc_path2):

    gen1 = torch.load(gen_path1)
    enc1 = torch.load(enc_path1)
    gen2 = torch.load(gen_path2)
    enc2 = torch.load(enc_path2)
    return(gen1, enc1, gen2, enc2)
    


if __name__ == '__main__':
    # update args and print
    args = generic.parse_args()
    #torch.manual_seed(args.rand_seed)
    embeddings, word_to_indx = embedding.get_embedding_tensor(args)
    train_data, dev_data, test_data = dataset_factory.get_dataset(args, word_to_indx)
    
    results_path_stem = args.results_path.split('/')[-1].split('.')[0]
    args.model_path = '{}.pt'.format(os.path.join(args.save_dir, results_path_stem))

    gen1, enc1, gen2, enc2 = load_models(gen_path1, enc_path1, gen_path2, enc_path2)

    # test
    if args.test :
        #test_stats = train.test_model(test_data, model, gen, args)
        #args.test_stats = test_stats

        test_stats1, test_stats2, test_stats1_other, test_stats2_other, accuracy =\
                     train.test_model_combine(test_data, enc1, gen1, enc2, gen2, args)
        max_accuracy = max(test_stats1["test_accuracy"][0], test_stats2["test_accuracy"][0])

        print("best_accuracy:", max_accuracy)
        print("combined_accuracy:", accuracy)
        print("improvement:", accuracy - max_accuracy)
        f = open('results.txt', 'a')
        f.write("hyperparameters: "+str(args.selection_lambda)+", "+str(args.continuity_lambda)+"\n")
        f.write("base_accuracy: "+str(test_stats1["test_accuracy"][0])+", "+str(test_stats2["test_accuracy"][0])+"\n")
        f.write("improvement:"+str(accuracy - max_accuracy)+"\n")
        f.write("combined_accuracy: "+str(accuracy)+"\n")
        f.close()
        

        args.train_data = train_data
        args.test_data = test_data
        
        args.test_stats = test_stats1
        args_dict = vars(args)
        pickle.dump(args_dict, open("logs/t1_test.results",'wb') )
        
        args.test_stats = test_stats2
        args_dict = vars(args)
        pickle.dump(args_dict, open("logs/t2_test.results",'wb') )


            
