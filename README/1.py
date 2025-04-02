from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np


class TextFeatureSelector:
    """
    文本特征选择器，支持高频词和TF-IDF两种特征提取方式

    参数:
        method (str): 特征提取方法，'frequency'或'tfidf'（默认'frequency'）
        max_features (int): 保留的最大特征数量（默认None，保留全部）
        ngram_range (tuple): n-gram范围（默认(1,1)只使用单词）
        stop_words (str/list): 停用词（默认None，可选'english'或自定义列表）
        min_df (int/float): 最小文档频率（默认1，整数表示绝对次数，小数表示比例）
        max_df (int/float): 最大文档频率（默认1.0）
        binary (bool): 是否使用二进制特征（仅frequency方法有效，默认False）
        use_idf (bool): 是否使用逆文档频率（仅tfidf方法有效，默认True）
        norm (str): 归一化方法（仅tfidf方法有效，默认'l2'）
    """

    def __init__(self,
                 method='frequency',
                 max_features=None,
                 ngram_range=(1, 1),
                 stop_words=None,
                 min_df=1,
                 max_df=1.0,
                 binary=False,
                 use_idf=True,
                 norm='l2'):

        self.method = method.lower()
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.stop_words = stop_words
        self.min_df = min_df
        self.max_df = max_df
        self.binary = binary
        self.use_idf = use_idf
        self.norm = norm
        self.vectorizer = None
        self.feature_names = None

        # 验证方法参数
        if self.method not in ['frequency', 'tfidf']:
            raise ValueError("method参数必须是'frequency'或'tfidf'")

    def fit(self, texts):
        """拟合特征提取器"""
        if self.method == 'frequency':
            self.vectorizer = CountVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                stop_words=self.stop_words,
                min_df=self.min_df,
                max_df=self.max_df,
                binary=self.binary
            )
        else:  # tfidf
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                stop_words=self.stop_words,
                min_df=self.min_df,
                max_df=self.max_df,
                use_idf=self.use_idf,
                norm=self.norm
            )

        self.vectorizer.fit(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        return self

    def transform(self, texts):
        """转换文本为特征矩阵"""
        if self.vectorizer is None:
            raise RuntimeError("请先调用fit方法拟合数据")
        return self.vectorizer.transform(texts).toarray()

    def fit_transform(self, texts):
        """拟合并转换数据"""
        return self.fit(texts).transform(texts)

    def get_feature_names(self):
        """获取特征名称"""
        if self.feature_names is None:
            raise RuntimeError("请先拟合数据")
        return list(self.feature_names)


# 示例数据
corpus = [
    'This is the first document about Python programming.',
    'This document is the second document about machine learning.',
    'And this is the third one about data science and Python.',
    'Is this the first document about artificial intelligence?',
    'Python is popular for data science and machine learning.'
]

# 1. 使用高频词特征（默认参数）
print("=== 高频词特征 ===")
freq_selector = TextFeatureSelector(method='frequency', max_features=10)
freq_features = freq_selector.fit_transform(corpus)

print("\n特征名称:")
print(freq_selector.get_feature_names())

print("\n特征矩阵:")
print(freq_features)

# 2. 使用TF-IDF特征
print("\n=== TF-IDF特征 ===")
tfidf_selector = TextFeatureSelector(
    method='tfidf',
    max_features=10,
    stop_words='english',
    ngram_range=(1, 2)  # 包含unigram和bigram
)
tfidf_features = tfidf_selector.fit_transform(corpus)

print("\n特征名称:")
print(tfidf_selector.get_feature_names())

print("\n特征矩阵:")
print(np.round(tfidf_features, 3))  # 保留3位小数便于查看