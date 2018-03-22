from __future__ import division

import numpy as np
import cPickle as pickle
import os, sys
import scipy.io

class dataprovider(object):
	def __init__(self, train_list, test_list, img_feat_dir, sen_dir, vocab_size,
		reward_con=0.2, multi_reg=True, val_list='', phrase_len=19, batch_size=20, seed=1):
		self.train_list = train_list
		self.val_list = val_list
		self.test_list = test_list
		self.img_feat_dir = img_feat_dir
		self.sen_dir = sen_dir
		self.phrase_len = phrase_len
		self.multi_reg = multi_reg
		self.cur_id = 0
		self.epoch_id = 0
		self.num_prop = 100
		self.img_feat_size = 4096
		self.num_test = 1000
		self.reward_con = reward_con
		self.batch_size = batch_size
		self.vocab_size = vocab_size
		self.is_save = False
		np.random.seed(seed)
		self.train_id_list = np.random.permutation(len(train_list))

	def _reset(self):
		self.cur_id = 0
		self.train_id_list = np.random.permutation(len(self.train_list))
		self.is_save = False

	def _get_reward(self, sen_feat, sen_id):
		reward = np.zeros(self.num_prop, dtype=float)
		cur_sen_ids = np.array(sen_feat['sen_ids']).astype('int')
		cur_sen_id = sen_feat['sen_ids'][sen_id]
		
		cur_sen_set = np.where(cur_sen_ids == cur_sen_id)[0]
		for cur_sen_ind in cur_sen_set:
			reward[sen_feat['gt_pos_all'][cur_sen_ind]] = self.reward_con
		reward[sen_feat['gt_pos_all'][sen_id]] = 1.0
		return reward

	def _read_single_feat(self, img_id):
		# img_id = self.train_list[self.train_id_list[self.cur_id]]

		sen_feat = np.load('%s/%d.pkl'%(self.sen_dir, img_id))
		pos_ids = np.array(sen_feat['pos_id']).astype('int')
		pos_ind = np.where(pos_ids != -1)[0]
		bbx_loc = np.array(sen_feat['ss_box'].astype('float'))	# [xmin, ymin, xmax, ymax]
		h = float(sen_feat['height'])
		w = float(sen_feat['width'])

		if len(pos_ind) > 0:
			img_feat = np.zeros((self.num_prop, self.img_feat_size+5))
			cur_feat = np.load('%s/%d.npy'%(self.img_feat_dir, img_id))
			img_feat[:cur_feat.shape[0], :-5] = cur_feat

			sens = sen_feat['sen_lang_token']
			sen_id = np.random.randint(len(pos_ind))
			# print img_id, sen_id
			sen = sens[pos_ind[sen_id]]
			reward = self._get_reward(sen_feat, pos_ind[sen_id])
			# pad sen tokens to phrase_len with UNK token as (self.vocab_size-1)
			sen_token = np.ones(self.phrase_len)*(self.vocab_size-1)	
			sen_token = sen_token.astype('int')
			sen_token[:len(sen)] = sen
			sen_token = sen_token[::-1]		# reverse tokens to adapt to encodeer-decoder structure

			bbx_data = np.zeros((cur_feat.shape[0], 5))
			bbx_data[:, 0] = bbx_loc[:, 0] / w
			bbx_data[:, 1] = bbx_loc[:, 1] / h
			bbx_data[:, 2] = bbx_loc[:, 2] / w
			bbx_data[:, 3] = bbx_loc[:, 3] / h
			bbx_data[:, 4] = (bbx_loc[:, 2]-bbx_loc[:, 0]+1.0)*(bbx_loc[:, 3]-bbx_loc[:, 1]+1.0)/w/h
			img_feat[:cur_feat.shape[0], -5:] = bbx_data
			img_feat = img_feat.astype('float')

			gt_reg = sen_feat['bbx_reg'][pos_ind[sen_id]].astype('float32') 
			y = pos_ids[pos_ind[sen_id]]			
			if self.multi_reg:
				pos_all = np.array(sen_feat['gt_pos_all'][pos_ind[sen_id]]).astype('int')
				pos_reg_all = sen_feat['bbx_reg_all'][pos_ind[sen_id]]
				return img_feat, sen_token, gt_reg, y, pos_all, pos_reg_all, reward
			else:
				return img_feat, sen_token, gt_reg, y, reward
		else:
			if self.multi_reg:
				return None, None, None, -1, None, None, None
			else:
				return None, None, None, -1, None

	def get_next_batch_reg(self):
		img_feat_batch = np.zeros((self.batch_size, self.num_prop, self.img_feat_size+5)).astype('float')
		token_batch = np.zeros((self.batch_size, self.phrase_len)).astype('int')
		y_batch = np.zeros(self.batch_size).astype('int')
		bbx_reg_batch = np.zeros((self.batch_size, 4)).astype('float')
		reward_batch = np.zeros((self.batch_size, self.num_prop), dtype=float)

		num_cnt = 0
		pos_all_batch = []
		pos_reg_all_batch = []
		while num_cnt < self.batch_size:
			if self.cur_id == len(self.train_list):
				self._reset()
				self.epoch_id += 1
				self.is_save = True
				print('Epoch %d complete'%(self.epoch_id))
			img_id = self.train_list[self.train_id_list[self.cur_id]]	
			if self.multi_reg:
				img_feat, sen_token, bbx_reg, y, pos_all, pos_reg_all, reward = self._read_single_feat(img_id)
			else:	
				img_feat, sen_token, bbx_reg, y, reward = self._read_single_feat(img_id)
			if y != -1:
				img_feat_batch[num_cnt] = img_feat
				token_batch[num_cnt] = sen_token
				bbx_reg_batch[num_cnt] = bbx_reg
				y_batch[num_cnt] = y
				reward_batch[num_cnt] = reward
				if self.multi_reg:
					for pos_id in pos_all:
						pos_all_batch.append([num_cnt, pos_id])
					if num_cnt == 0:
						pos_reg_all_batch = pos_reg_all
					else:
						pos_reg_all_batch = np.concatenate([pos_reg_all_batch, pos_reg_all], axis=0)
				num_cnt += 1
			# else:
			# 	print('No positive samples for %d'%(self.train_list[self.train_id_list[self.cur_id]]))
			self.cur_id += 1	
		if self.multi_reg:
			pos_all_batch = np.array(pos_all_batch).astype('int')
			return img_feat_batch, token_batch, bbx_reg_batch, y_batch, reward_batch, pos_all_batch, pos_reg_all_batch
		else:	
			return img_feat_batch, token_batch, bbx_reg_batch, y_batch, reward_batch		

	def get_test_feat_reg(self, img_id):
		sen_feat = np.load('%s/%d.pkl'%(self.sen_dir, img_id))
		pos_ids = np.array(sen_feat['pos_id']).astype('int')
		# pos_ind = np.where(pos_ids != -1)[0]
		gt_pos_all = sen_feat['gt_pos_all'] # proposal box id which overlaps gt > 0.5
		gt_bbx_all = sen_feat['gt_box']		# ground truth bbx for query: [xmin, ymin, xmax, ymax]
		num_sample = len(pos_ids)
		bbx_loc = np.array(sen_feat['ss_box'].astype('float'))	# 100 proposal bbx: [xmin, ymin, xmax, ymax]
		h = float(sen_feat['height'])
		w = float(sen_feat['width'])		

		img_feat = np.zeros((self.num_prop, self.img_feat_size+5)).astype('float')
		cur_feat = np.load('%s/%d.npy'%(self.img_feat_dir, img_id)).astype('float')
		img_feat[:cur_feat.shape[0], :-5] = cur_feat

		bbx_data = np.zeros((cur_feat.shape[0], 5))
		bbx_data[:, 0] = bbx_loc[:, 0] / w
		bbx_data[:, 1] = bbx_loc[:, 1] / h
		bbx_data[:, 2] = bbx_loc[:, 2] / w
		bbx_data[:, 3] = bbx_loc[:, 3] / h
		bbx_data[:, 4] = (bbx_loc[:, 2]-bbx_loc[:, 0]+1.0)*(bbx_loc[:, 3]-bbx_loc[:, 1]+1.0)/w/h
		img_feat[:cur_feat.shape[0], -5:] = bbx_data

		sen_feat_batch = np.zeros((len(pos_ids), self.phrase_len)).astype('int')

		gt_loc_all = []	# record ground truth bbx location for query phrase
		pos_or_reg = []	# record whether current query phrase has corresponding proposal

		sens = sen_feat['sen_lang_token']
		for sen_ind in range(len(pos_ids)):
			cur_sen = sens[sen_ind]
			sen_token = np.ones(self.phrase_len)*(self.vocab_size-1)
			sen_token = sen_token.astype('int')
			sen_token[:len(cur_sen)] = cur_sen
			sen_feat_batch[sen_ind] = sen_token[::-1]

			if pos_ids[sen_ind] != -1:
				gt_loc_all.append(gt_bbx_all[sen_ind])
				pos_or_reg.append(True)
			else:
				pos_or_reg.append(False)
				if np.any(gt_bbx_all[sen_ind]):
					gt_loc_all.append(gt_bbx_all[sen_ind])
				else:
					# there are phrases which do not have corresponding gt bbx
					gt_loc_all.append([0.0, 0.0, 0.0, 0.0])
					num_sample -= 1

		return img_feat, sen_feat_batch, gt_pos_all, gt_loc_all, bbx_loc, num_sample, pos_or_reg, h, w


if __name__ == '__main__':
	train_list = []
	test_list = []
	img_feat_dir = '~/dataset/flickr30k_img_bbx_ss_vgg_cls'
	sen_dir = '~/dataset/flickr30k_img_sen_feat'
	vocab_size = 17150
	with open('flickr30k_train.lst') as fin:
		for img_id in fin.readlines():
			train_list.append(int(img_id.strip()))
	train_list = np.array(train_list).astype('int')
	cur_dataset = dataprovider(train_list, test_list, img_feat_dir, sen_dir, vocab_size)
	for i in range(10000):
		img_feat_batch, token_batch, bbx_reg_batch, y_batch, reward_batch, pos_all_batch, pos_reg_all_batch = cur_dataset.get_next_batch_reg()
		print img_feat_batch.shape
		print pos_all_batch.shape[0]
		print y_batch
		print '%d/%d'%(cur_dataset.cur_id, len(cur_dataset.train_list))





