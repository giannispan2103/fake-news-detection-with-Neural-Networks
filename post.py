import re


def clean_en(text, lower=True):
    """
    Tokenization/string cleaning for all datasets.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    :param text: The string to be cleaned
    :param lower: If True text is converted to lower case
    :return: The clean string
    """
    text = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", text)
    text = re.sub(r"\'s", " \'s", text)
    text = re.sub(r"\'ve", " \'ve", text)
    text = re.sub(r"n\'t", " n\'t", text)
    text = re.sub(r"\'re", " \'re", text)
    text = re.sub(r"\'d", " \'d", text)
    text = re.sub(r"\'ll", " \'ll", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\(", " \( ", text)
    text = re.sub(r"\)", " \) ", text)
    text = re.sub(r"\?", " \? ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip().lower().encode('utf-8') if lower else text.strip().encode('utf-8')


class Post(object):
    """
    this class encapsulates data from https://www.kaggle.com/c/fake-news/data
    """
    def __init__(self, post_id,  title, author, text, label=None):
        """
        :param post_id: the id of the post
        :param title: the title of the post
        :param author: the author of the post if exists
        :param text: the body of the post
        :param label: 0 if post is categorized as reliable, 1 otherwise
        """
        self.post_id = post_id
        self.author = author
        self.label = label
        self.title = title
        self.cleaned_title = clean_en(self.title, False)
        self.text = text
        self.cleaned_text = clean_en(self.text, False)
        init_tokens = self.cleaned_text.split()
        self.uppercase_indicators = [x.isupper() for x in init_tokens]
        self.tokens = [x.lower() for x in init_tokens]
        title_init_tokens = self.cleaned_title.split()
        self.title_uppercase_indicators = [x.isupper() for x in title_init_tokens]
        self.title_tokens = [x.lower() for x in title_init_tokens]
        self.text_size = self.get_text_size()
        self.title_size = self.get_title_size()

    def get_text_size(self):
        return len(self.tokens)

    def get_title_size(self):
        return len(self.title_tokens)

    def get_text_uppercase_score(self):
        if self.text_size == 0:
            return self.get_title_uppercase_score()
        else:
            return sum(self.uppercase_indicators) / float(self.text_size)

    def get_title_uppercase_score(self):
        if self.title_size == 0:
            return self.get_text_uppercase_score()
        else:
            return sum(self.title_uppercase_indicators) / float(self.title_size)

    def get_uppercase_scores(self):
        return [self.get_text_uppercase_score(), self.get_title_uppercase_score()]

    def __eq__(self, other):
        return self.text == other.text

    def __ne__(self, other):
        return self.text != other.text

    def __hash__(self):
        return hash(self.text)



