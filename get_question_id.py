'''
Transform the user/item id to int between [1,N], which is convenient for analysis tasks later.
The Mapping is a original_category:new_id dict, and save as pickle format. 
Note that index start with 1 (0 is preserve for Embedding's padding)
By default, the pickles will save in 'pkls/'
'''
import pickle
import pandas as pd
import os

df = pd.read_csv('cathay_test/inbound_question.csv')
if not os.path.isdir('pkls'):
    os.mkdir('pkls')

def get_question_user_id_map():
	'save question and user"s id mapping as dic pickle'
	question_id_map = dict()
	user_id_map = dict()
	for question,uid in zip(df.inbound_question,df.user_id):
	    question_id_map.setdefault(question,len(question_id_map)+1)
	    user_id_map.setdefault(uid,len(user_id_map)+1)
	#save pickle
	with open('pkls/question_id_map.pkl','wb') as f:
	    pickle.dump(question_id_map,f)
	with open('pkls/user_id_map.pkl','wb') as f:
	    pickle.dump(user_id_map,f)

	return question_id_map,user_id_map

def get_sequence(question_id_map:dict,
			     user_id_map:dict):
	'get uid:question_sequence format dict and save pickle'
	uid_seq = dict()
	for uid,qid in zip(df.user_id,df.inbound_question):
	    uid = user_id_map[uid]
	    qid = question_id_map[qid]
	    if uid not in uid_seq:
	        uid_seq[uid] = [qid]
	    else:
	        uid_seq[uid].append(qid)
	with open('pkls/uid_seq.pkl','wb') as f:
	    pickle.dump(uid_seq,f)
	return uid_seq

def main():
	question_id_map,user_id_map = get_question_user_id_map()
	get_sequence(question_id_map, user_id_map)


if __name__ == '__main__':
	main()