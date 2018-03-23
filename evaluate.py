import tensorflow as tf
import os, sys
import numpy as np
import time

from dataprovider import dataprovider
from model import ground_model
from util.iou import calc_iou
from util.iou import calc_iou_by_reg_feat
from util.nms import nms
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_name", type=str, default='qrc')
parser.add_argument("-r", "--reward_con", type=float, default=0.2)
parser.add_argument("-g", "--gpu", type=str, default='0')
parser.add_argument("--restore_id", type=int, default=0)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

class Config(object):
    batch_size = 40
    img_feat_dir = './feature'
    sen_dir = './annotation'
    train_file_list = 'flickr30k_train_val.lst'
    test_file_list = 'flickr30k_test.lst'
    log_file = './log/ground_supervised'
    save_path = './model/ground_supervised'
    vocab_size = 17869    
    num_epoch = 3
    max_step = 40000
    optim='adam'
    dropout = 0.5
    lr = 0.0001
    weight_decay=0.0
    lstm_dim = 500

def update_feed_dict(dataprovider, model, is_train):
    img_feat, sen_feat, gt_reg, bbx_label, reward_batch, pos_all, pos_reg_all = dataprovider.get_next_batch_reg()
    feed_dict = {
                model.sen_data: sen_feat,
                model.vis_data: img_feat,
                model.bbx_label: bbx_label,
                model.gt_reg: gt_reg,
                model.reward: reward_batch,
                model.is_train: is_train}
    if dataprovider.multi_reg:
        feed_dict[model.pos_all] = pos_all
        feed_dict[model.pos_reg_all] = pos_reg_all
        feed_dict[model.num_reg] = float(pos_all.shape[0])
    return feed_dict

def eval_cur_batch(gt_label, cur_logits, 
    is_train=True, num_sample=0, pos_or_reg=None,
    bbx_loc=None, gt_loc_all=None, ht = 1.0, wt = 1.0):
    accu = 0.0
    if is_train:
        res_prob = cur_logits[:, :, 0]
        res_label = np.argmax(res_prob, axis=1)        
        accu = float(np.sum(res_label == gt_label)) / float(len(gt_label))
    else:
        num_bbx = len(bbx_loc)
        res_prob = cur_logits[:, :num_bbx, 0]
        res_label = np.argmax(res_prob, axis=1)        
        for gt_id in range(len(pos_or_reg)):
            cur_gt_pos = gt_label[gt_id]
            success = False
            
            cur_gt = gt_loc_all[gt_id]
            if np.any(cur_gt):
                cur_bbx = bbx_loc[res_label[gt_id]]
                cur_reg = cur_logits[gt_id, res_label[gt_id], 1:]
                #print 'IOU Stats: ', cur_gt, cur_bbx, cur_reg
                iou, _ = calc_iou_by_reg_feat(cur_gt, cur_bbx, cur_reg, ht, wt)
                if iou > 0.5:
                    success = True
            if success:
                accu += 1.0

        accu /= float(num_sample)
    return accu

def load_img_id_list(file_list):
    img_list = []
    with open(file_list) as fin:
        for img_id in fin.readlines():
            img_list.append(int(img_id.strip()))
    img_list = np.array(img_list).astype('int')    
    return img_list

def run_eval(sess, dataprovider, model, eval_op, feed_dict):
    num_test = 0.0
    num_corr_all = 0.0
    num_cnt_all = 0.0
    for img_ind, img_id in enumerate(dataprovider.test_list):
        img_feat_raw, sen_feat_batch, bbx_gt_batch, gt_loc_all, \
        bbx_loc, num_sample_all, pos_or_reg, ht, wt = dataprovider.get_test_feat_reg(img_id)

        if num_sample_all > 0:
            num_test += 1.0
            num_corr = 0
            num_sample = len(bbx_gt_batch)
            img_feat = feed_dict[model.vis_data]
            for i in range(num_sample):
                img_feat[i] = img_feat_raw
            sen_feat = feed_dict[model.sen_data]
            sen_feat[:num_sample] = sen_feat_batch

            eval_feed_dict = {
                model.sen_data: sen_feat,
                model.vis_data: img_feat,
                model.is_train: False}

            cur_att_logits = sess.run(eval_op, feed_dict=eval_feed_dict)
            cur_att_logits = cur_att_logits[:num_sample]
            # print cur_att_logits
            cur_accuracy = eval_cur_batch(bbx_gt_batch, cur_att_logits, False, 
                num_sample_all, pos_or_reg, bbx_loc, gt_loc_all, ht , wt)

            num_valid = np.sum(np.all(gt_loc_all, 1))
            print '%d/%d: %d/%d, %.4f'%(img_ind, len(dataprovider.test_list), num_valid, num_sample, cur_accuracy)
            num_corr_all += cur_accuracy*num_sample_all
            num_cnt_all += float(num_sample_all)

    accu = num_corr_all/num_cnt_all
    print 'Accuracy = %.4f'%(accu)
    return accu

def run_evaluate():
    train_list = []
    test_list = []
    config = Config()
    train_list = load_img_id_list(config.train_file_list)
    test_list = load_img_id_list(config.test_file_list)

    config.save_path = config.save_path + '_' + args.model_name
    assert(os.path.isdir(config.save_path))

    config.hidden_size = 500
    config.is_multi = True
    config.reward_con = args.reward_con
    restore_id = args.restore_id
    assert(restore_id > 0)

    cur_dataset = dataprovider(train_list, test_list, config.img_feat_dir, config.sen_dir, config.vocab_size,
                                reward_con=config.reward_con, batch_size=config.batch_size)

    model = ground_model(config)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33)

    with tf.Graph().as_default():
        loss, loss_vec, logits, rwd_pred, loss_rwd = model.build_model()
        # Create a session for running Ops on the Graph.
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        # Run the Op to initialize the variables.
        saver = tf.train.Saver(max_to_keep=20)
        feed_dict = update_feed_dict(cur_dataset, model, False)

        print 'Restore model_%d'%restore_id
        cur_dataset.is_save = False
        saver.restore(sess, './model/%s/model_%d.ckpt'%(config.save_path, restore_id))   

        print "-----------------------------------------------"
        eval_accu = run_eval(sess, cur_dataset, model, logits, feed_dict)
        print "-----------------------------------------------"

def main(_):
    run_evaluate()

if __name__ == '__main__':
    tf.app.run()
