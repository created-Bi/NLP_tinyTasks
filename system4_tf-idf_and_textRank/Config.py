
class Config(object):
    def __init__(self, newsFile_url, stopWords_url, pre_news_num, other_news_num, stride=4,
                 iter_times=500, top_k=10):
        self.newsFile_url = newsFile_url
        self.stopWords_url = stopWords_url
        self.pre_news_num = pre_news_num
        self.other_news_num = other_news_num
        self.stride = stride
        self.iter_times = iter_times
        self.top_k = top_k