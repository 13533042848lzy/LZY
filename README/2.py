import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from enum import Enum


# 1. 算法基础：多项式朴素贝叶斯分类器
class NaiveBayesClassifier:
    """
    多项式朴素贝叶斯分类器实现
    核心公式：
    P(c|d) ∝ P(c) * ∏ P(w_i|c)^x_i
    其中x_i是词w_i在文档d中的出现次数
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha  # 平滑参数
        self.class_probs = None
        self.feature_probs = None

    def fit(self, X, y):
        # 计算先验概率P(c)
        classes, counts = np.unique(y, return_counts=True)
        self.class_probs = counts / counts.sum()

        # 计算条件概率P(w|c)
        self.feature_probs = []
        for c in classes:
            class_samples = X[y == c]
            total_count = class_samples.sum(axis=0) + self.alpha
            total_words = total_count.sum() + self.alpha * X.shape[1]
            self.feature_probs.append(total_count / total_words)

    def predict(self, X):
        log_probs = []
        for c in range(len(self.class_probs)):
            # 对数空间计算避免下溢
            log_prob = np.log(self.class_probs[c]) + \
                       np.sum(np.log(self.feature_probs[c]) * X, axis=1)
            log_probs.append(log_prob)
        return np.argmax(np.array(log_probs).T, axis=1)


# 2. 数据处理流程
class TextPreprocessor:
    """文本预处理流水线"""

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def preprocess(self, text):
        # 1. 清洗文本
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())

        # 2. 分词和过滤
        tokens = [w for w in word_tokenize(text)
                  if w not in self.stop_words and len(w) > 2]

        # 3. 词形还原
        return [self.lemmatizer.lemmatize(w) for w in tokens]


# 3. 特征构建与模式切换
class FeatureMode(Enum):
    FREQUENCY = 1
    TFIDF = 2


class FeatureBuilder:
    """支持双模式的特征构建器"""

    def __init__(self, mode=FeatureMode.TFIDF, max_features=1000):
        self.mode = mode
        self.vectorizer = self._init_vectorizer(max_features)

    def _init_vectorizer(self, max_features):
        common_params = {
            'max_features': max_features,
            'ngram_range': (1, 2),
            'stop_words': 'english'
        }

        if self.mode == FeatureMode.FREQUENCY:
            return CountVectorizer(**common_params)
        return TfidfVectorizer(**common_params)

    def build_features(self, texts):
        # 将分词列表转换为空格连接的字符串
        processed_texts = [' '.join(text) for text in texts]
        return self.vectorizer.fit_transform(processed_texts)

    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out()


# 4. 完整流程示例
if __name__ == "__main__":
    # 示例数据
    emails = [
        "Claim your free prize now!",
        "Meeting about project timeline",
        "Win a million dollars today",
        "Quarterly report preparation",
        "Special offer just for you",
        "Team building activity schedule"
    ]
    labels = np.array([1, 0, 1, 0, 1, 0])  # 1=spam, 0=ham

    # 文本预处理
    preprocessor = TextPreprocessor()
    processed = [preprocessor.preprocess(email) for email in emails]

    # 对比两种特征模式
    for mode in FeatureMode:
        print(f"\n=== {mode.name} ===")

        # 特征构建
        builder = FeatureBuilder(mode=mode, max_features=5)
        X = builder.build_features(processed)

        # 输出特征信息
        print("Feature Names:", builder.get_feature_names())
        print("Feature Matrix:\n", X.toarray().round(3))

        # 训练分类器
        clf = NaiveBayesClassifier()
        clf.fit(X, labels)
        preds = clf.predict(X)

        # 评估结果
        accuracy = np.mean(preds == labels)
        print(f"Accuracy: {accuracy:.1%}")