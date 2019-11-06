echo 'now prepare data...'
python get_question_id.py
python split_train_test.py
echo 'now training word2vec embedding...'
python train_word2vec.py
echo 'time to train model...'
python train_cnn.py