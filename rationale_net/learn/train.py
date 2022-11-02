import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import rationale_net.utils.generic as generic
import rationale_net.utils.metrics as metrics
import tqdm
import numpy as np
import pdb
import sklearn.metrics
import rationale_net.utils.learn as learn
import math, random

def train_model(train_data, dev_data, model, gen, args):
    '''
    Train model and tune on dev set. If model doesn't improve dev performance within args.patience
    epochs, then halve the learning rate, restore the model to best and continue training.

    At the end of training, the function will restore the model to best dev version.

    returns epoch_stats: a dictionary of epoch level metrics for train and test
    returns model : best model from this call to train
    '''

    if args.cuda:
        model = model.cuda()
        gen = gen.cuda()

    args.lr = args.init_lr
    optimizer = learn.get_optimizer([model, gen], args)

    num_epoch_sans_improvement = 0
    epoch_stats = metrics.init_metrics_dictionary(modes=['train', 'dev'])
    step = 0
    tuning_key = "dev_{}".format(args.tuning_metric)
    best_epoch_func = min if tuning_key == 'loss' else max

    train_loader = learn.get_train_loader(train_data, args)
    dev_loader = learn.get_dev_loader(dev_data, args)




    for epoch in range(1, args.epochs + 1):

        print("-------------\nEpoch {}:\n".format(epoch))
        for mode, dataset, loader in [('Train', train_data, train_loader), ('Dev', dev_data, dev_loader)]:
            train_model = mode == 'Train'
            print('{}'.format(mode))
            key_prefix = mode.lower()
            epoch_details, step, _, _, _, _, _ = run_epoch(
                data_loader=loader,
                train_model=train_model,
                model=model,
                gen=gen,
                optimizer=optimizer,
                step=step,
                args=args)

            epoch_stats, log_statement = metrics.collate_epoch_stat(epoch_stats, epoch_details, key_prefix, args)
            
            # Log  performance
            print(log_statement)


        # Save model if beats best dev
        
        best_func = min if args.tuning_metric == 'loss' else max
        if best_func(epoch_stats[tuning_key]) == epoch_stats[tuning_key][-1]:
            num_epoch_sans_improvement = 0
            if not os.path.isdir(args.save_dir):
                os.makedirs(args.save_dir)
            # Subtract one because epoch is 1-indexed and arr is 0-indexed
            epoch_stats['best_epoch'] = epoch - 1
            torch.save(model, args.model_path)
            torch.save(gen, learn.get_gen_path(args.model_path))
        else:
            num_epoch_sans_improvement += 1

        if not train_model:
            print('---- Best Dev {} is {:.4f} at epoch {}'.format(
                args.tuning_metric,
                epoch_stats[tuning_key][epoch_stats['best_epoch']],
                epoch_stats['best_epoch'] + 1))

        if num_epoch_sans_improvement >= args.patience:
            print("Reducing learning rate")
            num_epoch_sans_improvement = 0
            model.cpu()
            gen.cpu()
            model = torch.load(args.model_path)
            gen = torch.load(learn.get_gen_path(args.model_path))

            if args.cuda:
                model = model.cuda()
                gen   = gen.cuda()
            args.lr *= .5
            optimizer = learn.get_optimizer([model, gen], args)


    f = open('results.txt', 'a')
    f.write("==============================================="+"\n")
    f.write(str(epoch_stats["dev_accuracy"][epoch_stats['best_epoch']]))
    f.close()
    # Restore model to best dev performance
    if os.path.exists(args.model_path):
        model.cpu()
        model = torch.load(args.model_path)
        gen.cpu()
        gen = torch.load(learn.get_gen_path(args.model_path))

    return epoch_stats, model, gen


def test_model(test_data, model, gen, args):
    '''
    Run model on test data, and return loss, accuracy.
    '''
    if args.cuda:
        model = model.cuda()
        gen = gen.cuda()

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False)

    test_stats = metrics.init_metrics_dictionary(modes=['test'])

    mode = 'Test'
    train_model = False
    key_prefix = mode.lower()
    print("-------------\nTest")
    epoch_details, _, losses, preds, golds, rationales, _ = run_epoch(
        data_loader=test_loader,
        train_model=train_model,
        model=model,
        gen=gen,
        optimizer=None,
        step=None,
        args=args)

    test_stats, log_statement = metrics.collate_epoch_stat(test_stats, epoch_details, 'test', args)
    test_stats['losses'] = losses
    test_stats['preds'] = preds
    test_stats['golds'] = golds
    test_stats['rationales'] = rationales

    print(log_statement)

    return test_stats


def test_model_combine(test_data, model1, gen1, model2, gen2, args):
    '''
    Run model on test data, and return loss, accuracy.
    '''
    if args.cuda:
        model1 = model1.cuda()
        gen1 = gen1.cuda()
        model2 = model2.cuda()
        gen2 = gen2.cuda()

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False)

    test_stats1 = metrics.init_metrics_dictionary(modes=['test'])
    test_stats1_other = metrics.init_metrics_dictionary(modes=['test'])
    test_stats2 = metrics.init_metrics_dictionary(modes=['test'])
    test_stats2_other = metrics.init_metrics_dictionary(modes=['test'])

    mode = 'Test'
    train_model = False
    key_prefix = mode.lower()
    print("-------------\nTest")

    
    #return epoch_stat, step, losses, preds, golds, rationales
    #f1(r1)
    epoch_details1, _, losses1, preds1, golds1, rationales1, logits1 = run_epoch(
        data_loader=test_loader,
        train_model=train_model,
        model=model1,
        gen=gen1,
        optimizer=None,
        step=None,
        args=args)
    #f1(r2)
    epoch_details1_other, _, losses1_other, preds1_other, golds1_other, rationales1_other, logits1_other = run_epoch(
        data_loader=test_loader,
        train_model=train_model,
        model=model1,
        gen=gen2,
        optimizer=None,
        step=None,
        args=args)
    #f2(r2)
    epoch_details2, _, losses2, preds2, golds2, rationales2, logits2 = run_epoch(
        data_loader=test_loader,
        train_model=train_model,
        model=model2,
        gen=gen2,
        optimizer=None,
        step=None,
        args=args)
    #f2(r1)
    epoch_details2_other, _, losses2_other, preds2_other, golds2_other, rationales2_other, logits2_other = run_epoch(
        data_loader=test_loader,
        train_model=train_model,
        model=model2,
        gen=gen1,
        optimizer=None,
        step=None,
        args=args)

    
    test_stats1, log_statement1 = metrics.collate_epoch_stat(test_stats1, epoch_details1, 'test', args)
    test_stats1['losses'] = losses1
    test_stats1['preds'] = preds1
    test_stats1['golds'] = golds1
    test_stats1['rationales'] = rationales1

    test_stats1_other, log_statement1_other = metrics.collate_epoch_stat(test_stats1_other, epoch_details1_other, 'test', args)
    test_stats1_other['losses'] = losses1_other
    test_stats1_other['preds'] = preds1_other
    test_stats1_other['golds'] = golds1_other

    test_stats2, log_statement2 = metrics.collate_epoch_stat(test_stats2, epoch_details2, 'test', args)
    test_stats2['losses'] = losses2
    test_stats2['preds'] = preds2
    test_stats2['golds'] = golds2
    test_stats2['rationales'] = rationales2

    test_stats2_other, log_statement2_other = metrics.collate_epoch_stat(test_stats2_other, epoch_details2_other, 'test', args)
    test_stats2_other['losses'] = losses2_other
    test_stats2_other['preds'] = preds2_other
    test_stats2_other['golds'] = golds2_other
    print(log_statement1)
    print(log_statement2)
    #print(log_statement1_other)
    #print(log_statement2_other)

    
    #combine
    count = 0
    labels = []
    wrong_cases = []
    agreement = []
    disagreement = []
    one_changes = []
    two_changes = []
    confidence_one = []
    confidence_two = []
    avg_cls = {"cl1":0, "cl2":0, "cl1_other":0, "cl2_other":0}
    for i in range(len(golds1)):
        pred1 = preds1[i]
        pred1_other = preds1_other[i]
        pred2 = preds2[i]
        pred2_other = preds2_other[i]

        cl1 = logit2cl(logits1[i])
        cl2 = logit2cl(logits2[i])
        cl1_other = logit2cl(logits1_other[i])
        cl2_other = logit2cl(logits2_other[i])
        
        avg_cls["cl1"] = avg_cls["cl1"] + cl1
        avg_cls["cl2"] = avg_cls["cl2"] + cl2
        avg_cls["cl1_other"] = avg_cls["cl1_other"] + cl1_other
        avg_cls["cl2_other"] = avg_cls["cl2_other"] + cl2_other
        
        label, deal = select(cl1, cl1_other, cl2, cl2_other, \
                              pred1, pred1_other, pred2, pred2_other)

        labels.append(label)
        #select
        if deal == "agreement":
            agreement.append(i)
        else:
            disagreement.append(i)
            if deal == "1_changes":
                one_changes.append(i)
            elif deal == "2_changes":
                two_changes.append(i)
            elif deal == "confidence1":
                confidence_one.append(i)
            elif deal == "confidence2":
                confidence_two.append(i)
            else:
                print("unknown interaction case")
        if label == golds1[i]:
            count += 1
        else:
            wrong_cases.append(i)
    
    for key in avg_cls:
        avg_cls[key] = avg_cls[key]/len(golds1)
    #print("avg_cls:", avg_cls)
    #print("count:", count)
    accuracy = count/len(golds1)
    #print("accuracy:", accuracy)
    

    #print("total cases:", len(golds1))
    
    #print(agreement)
    #print("agreement:", len(agreement))

    #print(disagreement)
    print("disagreement:", len(disagreement))
    
    #print(one_changes)
    print("one_changes:", len(one_changes))
    print()

    #print(two_changes)
    print("two_changes:", len(two_changes))
    print()

    #print(uncertain)
    print("confidence_one:", len(confidence_one))
    print()

    #print(stubborn)
    print("confidence_two:", len(confidence_two))
    print()

    #print("wrong cases:", wrong_cases)
    print("wrong count:", len(wrong_cases))
    print()

    #record results
    f = open('results.txt', 'a')
    
    f.write("dataset = " + args.dataset +"\n")
    #f.write("rand_seed = " + str(args.rand_seed) +"\n")
    f.write("LR = " + str(args.init_lr) +"\n")
    f.write("r1,2 length: "+str(epoch_details1['k_selection_loss'])+", "+ str(epoch_details2['k_selection_loss']) +"\n")
    f.write("r1,2 contig: "+str(epoch_details1['k_continuity_loss'])+", "+ str(epoch_details2['k_continuity_loss']) +"\n")
    f.write("one_changes")
    f.write(str(one_changes)+"\n")
    
    f.write("two_changes")
    f.write(str(two_changes)+"\n")
    
    f.write("confidence_one")
    f.write(str(confidence_one)+"\n")
    
    f.write("confidence_two")
    f.write(str(confidence_two)+"\n")
    
    f.write("wrong_cases")
    f.write(str(wrong_cases)+"\n")
    f.close()

    
    error_analysis(agreement, disagreement, one_changes, two_changes, confidence_one, confidence_two, wrong_cases)

    """
    for i in wrong_cases:
        if preds1[i] != preds2[i]:
            print("index:", i, "gold:", golds1[i])
            print(rationales1[i])
            print("1:", preds1[i])
            print(rationales2[i])
            print("2:", preds2[i])
            print()
    print()
    """
    
    return test_stats1, test_stats2, test_stats1_other, test_stats2_other, accuracy

select_one = 0
select_two = 0

#consistency + cl
def select(cl1, cl1_other, cl2, cl2_other, \
           pred1, pred1_other, pred2, pred2_other):
    global select_one, select_two
    if pred1 == pred2:
        return(pred1, "agreement")
    else:#disagreement    
        if pred1 == pred2_other and pred1_other != pred2:
            select_one += 1
            return(pred1, "2_changes")
        elif pred1 != pred2_other and pred1_other == pred2:
            select_two += 1
            return(pred2, "1_changes")
        else:#no r gives more consistent predictions
            if cl1 >= cl2:
                select_one += 1
                return(pred1, "confidence1")
            else:
                select_two += 1
                return(pred2, "confidence2")

def logit2cl(logit):#softmax for cl
    return(math.exp(max(logit))/sum([math.exp(x) for x in logit]))

def error_analysis(agreement, disagreement, one_changes, two_changes, conf1, conf2, wrong_cases):
    count_agreed_wrong = 0
    count_disagreed_wrong = 0
    
    count_one_changes_wrong = 0
    count_two_changes_wrong = 0
    count_conf1_wrong = 0
    count_conf2_wrong = 0
    
    for i in agreement:
        if i in wrong_cases:
            count_agreed_wrong += 1
            
    for i in disagreement:
        if i in wrong_cases:
            count_disagreed_wrong += 1
            
    for i in one_changes:
        if i in wrong_cases:
            count_one_changes_wrong += 1
            
    for i in two_changes:
        if i in wrong_cases:
            count_two_changes_wrong += 1
            
    for i in conf1:
        if i in wrong_cases:
            count_conf1_wrong += 1
            
    for i in conf2:
        if i in wrong_cases:
            count_conf2_wrong += 1
            
    if len(agreement) != 0:
        print("agreed_accuracy:", 1-count_agreed_wrong/len(agreement))
    else:
        print("len(agreement) = 0")
    if len(disagreement) != 0:
        print("disagree_accuracy:", 1-count_disagreed_wrong/len(disagreement))
    else:
        print("len(disagreement) = 0")

    if len(one_changes) != 0:
        print("one_changes_accuracy:", 1-count_one_changes_wrong/len(one_changes))
    else:
        print("len(one_changes) = 0")
    if len(two_changes) != 0:
        print("two_changes_accuracy:", 1-count_two_changes_wrong/len(two_changes))
    else:
        print("len(two_changes) = 0")
    if len(conf1) != 0:
        print("conf1_accuracy:", 1-count_conf1_wrong/len(conf1))
    else:
        print("len(conf1) = 0")
    if len(conf2) != 0:
        print("conf1_accuracy:", 1-count_conf2_wrong/len(conf2))
    else:
        print("len(conf1) = 0")
    f = open('results.txt', 'a')
    if len(one_changes) != 0:
        f.write("one_changes_accuracy: "+str(len(one_changes))+": "+str(1-count_one_changes_wrong/len(one_changes))+"\n")
    if len(two_changes) != 0:
        f.write("two_changes_accuracy: "+str(len(two_changes))+": "+str(1-count_two_changes_wrong/len(two_changes))+"\n")
    if len(conf1) != 0:
        f.write("conf1_accuracy: "+str(len(conf1))+": "+str(1-count_conf1_wrong/len(conf1))+"\n")
    if len(conf2) != 0:
        f.write("conf2_accuracy: "+str(len(conf2))+": "+str(1-count_conf2_wrong/len(conf2))+"\n")
    f.close()
    print()
    print("select_one:", select_one)
    print("select_two:", select_two)


def run_epoch(data_loader, train_model, model, gen, optimizer, step, args):
    '''
    Train model for one pass of train data, and return loss, acccuracy
    '''
    eval_model = not train_model
    data_iter = data_loader.__iter__()

    losses = []
    obj_losses = []
    k_selection_losses = []
    k_continuity_losses = []
    
    preds = []
    golds = []
    losses = []
    texts = []
    rationales = []

    selection_losses_list = []
    contiguity_losses_list = []
    logits = []
    if train_model:
        model.train()
        gen.train()
    else:
        gen.eval()
        model.eval()

    num_batches_per_epoch = len(data_iter)
    if train_model:
        num_batches_per_epoch = min(len(data_iter), 10000)

    for _ in tqdm.tqdm(range(num_batches_per_epoch)):
        batch = data_iter.next()
        if train_model:
            step += 1
            if  step % 100 == 0 or args.debug_mode:
                args.gumbel_temprature = max( np.exp((step+1) *-1* args.gumbel_decay), .05)

        x_indx = learn.get_x_indx(batch, args, eval_model)
        text = batch['text']
        y = autograd.Variable(batch['y'])#, volatile=eval_model

        if args.cuda:
            x_indx, y = x_indx.cuda(), y.cuda()

        if train_model:
            optimizer.zero_grad()

        if args.get_rationales:
            mask, z = gen(x_indx)
        else:
            mask = None

        logit, _ = model(x_indx, mask=mask)

        for i in logit.cpu().detach().numpy():
            logits.append(i.tolist())        
        if args.use_as_tagger:
            logit = logit.view(-1, 2)
            y = y.view(-1)
        loss = get_loss(logit, y, args)
        
        obj_loss = loss.detach().clone()
        
        if args.get_rationales:
            selection_cost, continuity_cost = gen.loss(mask, x_indx)
            loss += args.selection_lambda * selection_cost
            loss += args.continuity_lambda * continuity_cost

        if train_model:
            loss.backward()
            optimizer.step()

        if args.get_rationales:
            
            k_selection_losses.append( generic.tensor_to_numpy(selection_cost))
            k_continuity_losses.append( generic.tensor_to_numpy(continuity_cost))

        obj_losses.append(generic.tensor_to_numpy(obj_loss))
        losses.append( generic.tensor_to_numpy(loss) )
        batch_softmax = F.softmax(logit, dim=-1).cpu()
        preds.extend(torch.max(batch_softmax, 1)[1].view(y.size()).data.numpy())

        texts.extend(text)
        rationales.extend(learn.get_rationales(mask, text))

        if args.use_as_tagger:
            golds.extend(batch['y'].view(-1).numpy())
        else:
            golds.extend(batch['y'].numpy())



    epoch_metrics = metrics.get_metrics(preds, golds, args)

    epoch_stat = {
        'loss' : np.mean(losses),
        'obj_loss': np.mean(obj_losses)
    }

    for metric_k in epoch_metrics.keys():
        epoch_stat[metric_k] = epoch_metrics[metric_k]

    if args.get_rationales:
        epoch_stat['k_selection_loss'] = np.mean(k_selection_losses)
        epoch_stat['k_continuity_loss'] = np.mean(k_continuity_losses)

    return epoch_stat, step, losses, preds, golds, rationales, logits


def get_loss(logit,y, args):
    if args.objective == 'cross_entropy':
        if args.use_as_tagger:
            loss = F.cross_entropy(logit, y, reduce=False)
            neg_loss = torch.sum(loss * (y == 0).float()) / torch.sum(y == 0).float()
            pos_loss = torch.sum(loss * (y == 1).float()) / torch.sum(y == 1).float()
            loss = args.tag_lambda * neg_loss + (1 - args.tag_lambda) * pos_loss
        else:
            loss = F.cross_entropy(logit, y)
    elif args.objective == 'mse':
        loss = F.mse_loss(logit, y.float())
    else:
        raise Exception(
            "Objective {} not supported!".format(args.objective))
    return loss
