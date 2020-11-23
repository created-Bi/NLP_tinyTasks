'''
    Subsystem4: UnderstandingOfNLP
    Time: 2020/10/1
    Name: Kyle 
'''

'''
    IF-IDF = log(count(t,d)+1) * log(N/d_ft)
'''

'''
    Text-Rank: WS(V_i) = (1-d)+d*sum_{V_i}(W_{ji} / sum_{VK∈Out(V_j)*W_{jk}} * WS(V_j)
'''
import Config
import jieba
import numpy as np
from collections import Counter, defaultdict

# Tool1: timer
def timer(func):
    def func_wrapper(*args, **kwargs):
        from time import time
        time_start = time()
        result = func(*args, **kwargs)
        time_end = time()
        time_spend = time_end - time_start
        print('TIME-CONSUMING : [[FUNCTION %s]] cost time: %.3f s' % (func.__name__, time_spend))
        return result
    return func_wrapper

# Tool2: cut the words
@timer
def cut_the_words(sentences):
    cut = lambda string: list(jieba.cut(string))
    return list(map(cut, sentences))

# Step 1: load the news data
@timer
def load_news(newsFile_url, pre_news_num, other_news_num):
    with open(newsFile_url, 'r', encoding='UTF-8') as f:
        pre_news_data, other_news_data = [], []
        for i, data in enumerate(f.readlines()):
            if i < pre_news_num: pre_news_data.append(data[:-1])  # load the first 100 news to calculate TF-IDF
            elif i > pre_news_num and i < other_news_num: other_news_data.append(data[:-1])
    original_pre_data = cut_the_words(pre_news_data)
    length_of_pre_news_data, length_of_other_news_data = len(pre_news_data), len(other_news_data)
    print('--The numbers of the news: {}'.format(len(pre_news_data)))
    print('--The numbers of the others: {}'.format(length_of_other_news_data))
    pre_news_data = [x for new in original_pre_data for x in new]  # put all the sentences into one doc
    other_news_data = [x for new in cut_the_words(other_news_data) for x in new]  # put all the sentences into one doc
    print('--The first {} doc have [{}] single-words, while the last {} doc have [{}] single-words'.format(length_of_pre_news_data, len(pre_news_data),
                                                                                           length_of_other_news_data, len(other_news_data)))
    return original_pre_data, pre_news_data, other_news_data, length_of_other_news_data# numbers of other doc

# Step 2: clean data
@timer
def clean_data_by_delete_stopWords_from_knowledge(stopWords_url):
    with open(stopWords_url, 'r', encoding='UTF-8') as f:
        common_stop_words = [x[:-1] for x in f.readlines()]
    print('--The numbers of the news: {}'.format(len(common_stop_words)))
    print('--The first 10 stopWords data: {}'.format(common_stop_words[:10]))
    return common_stop_words

# Step 3: TF-IDF
@timer
def get_tf_idf(pre_news_data, other_news_data, odoc_num):
    # Counter for the first 100 doc
    word_count = Counter(pre_news_data)
    # Counter for other doc
    word_doc_count = Counter(other_news_data)
    # result
    tf, idf = [], []
    for word in pre_news_data:
        tf.append(np.log10(word_count[word] + 1))  # +1 is to avoid 0 happening
        idf.append(np.log10(odoc_num / (word_doc_count[word] + 1)))
    words_tfidf = np.array([tf[i] * idf[i] for i in range(len(tf))])
    return words_tfidf

# Step 4: Obtain the stopWords from tf-idf
@timer
def get_stopWords_from_tfidf(pre_news_data, words_tfidf):
    words_dic = dict(zip(pre_news_data, words_tfidf))
    words_dic_sort = sorted(words_dic.items(), key=lambda x: x[1], reverse=True)
    print('--Len of tf-idf: {}'.format(len(words_dic_sort)))
    tfidf_stop_words = [k for k, v in words_dic_sort][len(words_dic_sort)-500:]
    return tfidf_stop_words

class Text_Rank(object):
    def __init__(self, stride):
        self.stride = stride
        self.words_table = defaultdict(set)
        self.words_num = len(self.words_table)
        self.ws_table = {}

    def create_words_table(self, news_data):
        for news in news_data:
            for i, word in enumerate(news):
                # put left words far from this word in stride into words_table
                if i >= self.stride:
                    left = news[i - self.stride:i]
                else:
                    left = news[:i]
                # put right words far from this word in stride into words_table
                right = news[i + 1:i + self.stride + 1]
                # print(left+right)

                # all the left and right words should be in words_table
                for y in left + right:
                    self.words_table[word].add(y)

    def iteration_of_textRank(self, iter_times):
        d = 0.85
        # create a ws table for all the words in words_table to restore the ws scores
        self.ws_table = dict(zip(list(self.words_table.keys()), [1 for _ in range(len(self.words_table))]))
        # compute the ouput value of every in_node for the sepecific node
        calculate_frac = lambda in_node: 1 / len(self.words_table[in_node])
        # The iteration
        for i in range(iter_times):
            if i % 100 == 0:
                print("--{}th iteration.".format(i))

            for k, v in self.words_table.items():
                self.ws_table[k] = (1 - d) + d * sum([calculate_frac(node) * self.ws_table[node] for node in v])
        # After the iteration, we should sort the ws_of_words(WS Table) to extract the first N most important words in the news
        return sorted(self.ws_table.items(), key=lambda x: x[1], reverse=True)

    def extract_keyWords(self, news_data, iter_times, top_k):
        key_words = []  # List to restore the keywords
        # create the ws table
        self.create_words_table(news_data)
        # restore the sorted result
        sorted_result = self.iteration_of_textRank(iter_times)
        for i, (word, value) in enumerate(sorted_result):
            if i < top_k: key_words.append(word)
        return ', '.join(key_words)

if __name__ == '__main__':

    # Extract the first 100 news keywords
    conf = Config.Config('./article_9k.txt', 'stopWords.txt',
                         pre_news_num=100, other_news_num=3000,
                         stride=4, iter_times=1000, top_k=10)
    # load the stopWords
    print('Step 1: LOAD COMMON STOPWORDS: ')
    common_stop_words = clean_data_by_delete_stopWords_from_knowledge(conf.stopWords_url)

    # prepare for tfidf
    print('Step 2: LOAD NEWS for TF-IDF: ')
    original_pre_data, pre_news_data_for_tfidf, other_news_data_for_tfidf, odoc_num = load_news(conf.newsFile_url, conf.pre_news_num, conf.other_news_num)

    # calculate the value of tfidf
    print('Step 3: COMPUTE TF-IDF: ')
    words_tfidf = get_tf_idf(pre_news_data_for_tfidf, other_news_data_for_tfidf, odoc_num)

    # extract the stopWords from tfidf
    print('Step 4: OBTAIN STOPWORDS FROM TF-IDF: ')
    tfidf_stop_words = get_stopWords_from_tfidf(pre_news_data_for_tfidf, words_tfidf)

    print('Step 5: CLEAN NEWS DATA: ')
    clean_news_data = [[word for word in news if word not in common_stop_words and word not in tfidf_stop_words]
                       for news in original_pre_data]

    print('Step 6: CREATE TEXT RANK CLASS: ')
    text_rank = Text_Rank(stride=conf.stride)

    print('Step 7: EXTRACT KEYWORDS: ')
    key_words = text_rank.extract_keyWords(clean_news_data, conf.iter_times, conf.top_k)

    print('---- The KEY WORDS: ----')
    print(key_words)