from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import random
import time
import os
import traceback

import opts
import models
from dataloaders.dataloader import *
from misc import eval_utils
import misc.utils as utils
from models.loss_wrapper import LossWrapper

# reproducibility
random_seed = 2019 
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = False  
torch.backends.cudnn.benchmark = True 

def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)

def save_checkpoint(model, infos, optimizer, histories=None, append=''):
    if len(append) > 0:
        append = '-' + append
    # if checkpoint_path doesn't exist
    if not os.path.isdir(opt.checkpoint_path):
        os.makedirs(opt.checkpoint_path)
    checkpoint_path = os.path.join(opt.checkpoint_path, 'model%s.pth' %(append))
    torch.save(model.state_dict(), checkpoint_path)
    print("model saved to {}".format(checkpoint_path))
    if append == '':
        optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer%s.pth' %(append))
        torch.save(optimizer.state_dict(), optimizer_path)
    with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'%s.pkl' %(append)), 'wb') as f:
        utils.pickle_dump(infos, f)
    if histories:
        with open(os.path.join(opt.checkpoint_path, 'histories_'+opt.id+'%s.pkl' %(append)), 'wb') as f:
            utils.pickle_dump(histories, f)

def train(opt):
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    tb_summary_writer = SummaryWriter(opt.checkpoint_path)

    infos = {}
    histories = {}
    if opt.start_from is not None:  # resume training
        # open old infos and check if models are compatible
        with open(os.path.join(opt.start_from, 'infos_'+opt.id+'.pkl'), 'rb') as f:
            infos = utils.pickle_load(f)
            saved_model_opt = infos['opt']
            need_be_same=["caption_model", "rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved model disagree on '%s' " % checkme

        if os.path.isfile(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')):
            with open(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl'), 'rb') as f:
                histories = utils.pickle_load(f)
    else:
        infos['iter'] = 0
        infos['epoch'] = 0
        infos['iterators'] = loader.iterators
        infos['split_ix'] = loader.split_ix
        infos['vocab'] = loader.get_vocab()

    infos['opt'] = opt
    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)
    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})
    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_ix = infos.get('split_ix', loader.split_ix)
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)
    
    # setup model and optimizer
    model = models.setup(opt).cuda()
    dp_model = torch.nn.DataParallel(model)
    lw_model = LossWrapper(model, opt)
    dp_lw_model = torch.nn.DataParallel(lw_model)
    dp_lw_model.train()
    optimizer = utils.build_optimizer(filter(lambda p: p.requires_grad, model.parameters()), opt)
    if vars(opt).get('start_from', None) is not None:  # Load the optimizer
        optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer.pth')))

    done_flag = True # True when the first iteration, warm-up done and epoch done
    try:
        while True:
            warmup_n = opt.warmup_n
            if iteration <= warmup_n:
                opt.current_lr = iteration * opt.learning_rate / warmup_n
                utils.set_lr(optimizer, opt.current_lr)
                if iteration == warmup_n:
                    done_flag = True

            if done_flag and iteration >= warmup_n:
                # Assign the learning rate
                if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                    frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                    decay_factor = opt.learning_rate_decay_rate  ** frac
                    opt.current_lr = opt.learning_rate * decay_factor
                else:
                    opt.current_lr = opt.learning_rate
                utils.set_lr(optimizer, opt.current_lr) # set the decayed rate
                if iteration == warmup_n:
                    done_flag = False

            if done_flag:
                # Assign the scheduled sampling prob
                if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                    frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                    opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
                    model.ss_prob = opt.ss_prob                
                done_flag = False
            
            start = time.time()
            data = loader.get_batch('train')
            if iteration % 5 == 0:
                print('Read data:', time.time() - start)
            if iteration % 5 == 0:
                print('learning rate: {}'.format(opt.current_lr))
            torch.cuda.synchronize()

            start = time.time()
            tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks'], data['trip_pred'],\
                  data['obj_dist'], data['obj_box'], data['rel_ind'], data['pred_fmap'], data['pred_dist'],\
                  data['gpn_obj_ind'], data['gpn_pred_ind'], data['gpn_nrel_ind'], data['gpn_pool_mtx']]
            tmp = [_ if _ is None else _.cuda() for _ in tmp]
            fc_feats, att_feats, labels, masks, att_masks, trip_pred, obj_dist, obj_box, rel_ind, pred_fmap, pred_dist,\
            gpn_obj_ind, gpn_pred_ind, gpn_nrel_ind, gpn_pool_mtx = tmp

            optimizer.zero_grad()
            model_out = dp_lw_model(fc_feats, att_feats, labels, masks, att_masks, data['gts'], torch.arange(0, len(data['gts'])), trip_pred,\
                                    obj_dist, obj_box, rel_ind, pred_fmap, pred_dist, gpn_obj_ind, gpn_pred_ind, gpn_nrel_ind, gpn_pool_mtx)
            
            gpn_loss = model_out['gpn_loss'].mean() if model_out['gpn_loss'] is not None else None
            if model_out['lang_loss'] is not None:
                lang_loss = model_out['lang_loss'].mean()
                if gpn_loss is not None:
                    loss = lang_loss + gpn_loss
                else:
                    loss = lang_loss  # no gpn module

            loss.backward()
            utils.clip_gradient_norm(optimizer, 10.)
            optimizer.step()

            gpn_l = gpn_loss.item() if gpn_loss is not None else 0
            lang_l = lang_loss.item() if lang_loss is not None else 0
            train_loss = loss.item()
            torch.cuda.synchronize()
            
            end = time.time()
            if iteration % 5 == 0:
                print("iter {} (ep {}), gpn_loss = {:.3f}, lang_loss = {:.3f}, loss = {:.3f}, time/b = {:.3f}" \
                    .format(iteration, epoch, gpn_l, lang_l, train_loss, end - start))

            # Update the iteration and epoch
            iteration += 1
            if data['bounds']['wrapped']:
                epoch += 1
                done_flag = True
            
            # Write the training loss summary
            if (iteration % opt.losses_log_every == 0):
                add_summary_value(tb_summary_writer, 'train_loss', train_loss, iteration)
                add_summary_value(tb_summary_writer, 'gpn_loss', gpn_l, iteration)
                add_summary_value(tb_summary_writer, 'lang_loss', lang_l, iteration)
                add_summary_value(tb_summary_writer, 'learning_rate', opt.current_lr, iteration)
                add_summary_value(tb_summary_writer, 'scheduled_sampling_prob', model.ss_prob, iteration)

                loss_history[iteration] = train_loss
                lr_history[iteration] = opt.current_lr
                ss_prob_history[iteration] = model.ss_prob

            # update infos
            infos['iter'] = iteration
            infos['epoch'] = epoch
            infos['iterators'] = loader.iterators
            infos['split_ix'] = loader.split_ix
            
            # make evaluation on validation set, and save model
            if (iteration % opt.save_checkpoint_every == 0) or (epoch >= opt.max_epochs and opt.max_epochs != -1):
                # eval model
                eval_kwargs = {'split': 'val', 'dataset': opt.input_json}
                eval_kwargs.update(vars(opt))

                val_loss = eval_utils.eval_split(dp_model, lw_model.crit, loader, eval_kwargs, opt=opt, val_model=model)

                # Write validation result into summary
                add_summary_value(tb_summary_writer, 'language validation loss', val_loss, iteration)
                val_result_history[iteration] = {'loss': val_loss}

                # Save model if is improving on validation result
                current_score = - val_loss # still using the language validation loss

                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score

                # Dump miscalleous informations
                infos['best_val_score'] = best_val_score
                histories['val_result_history'] = val_result_history
                histories['loss_history'] = loss_history
                histories['lr_history'] = lr_history
                histories['ss_prob_history'] = ss_prob_history

                save_checkpoint(model, infos, optimizer, histories)
                if opt.save_history_ckpt:
                    save_checkpoint(model, infos, optimizer, append=str(iteration))

                # Stop if reaching max epochs
                if epoch >= opt.max_epochs and opt.max_epochs != -1:
                    #save_checkpoint(model, infos, optimizer, append='last')
                    break
    except (RuntimeError, KeyboardInterrupt):
        stack_trace = traceback.format_exc()
        print(stack_trace)

if __name__ == '__main__':
    # train model
    opt = opts.parse_opt()
    train(opt)
