import numpy as np
import pandas as pd
from post import Post

PAD_TOKEN = "*$*PAD*$*"
UNK_TOKEN = "*$*UNK*$*"
UNK_AUTHOR = "Nan"


def create_batches(posts, w2i, a2i, pad_tnk=PAD_TOKEN, unk_tkn=UNK_TOKEN, unk_author=UNK_AUTHOR, batch_size=128,
                   max_len=100, title_max_len=10, sort_data=True):
    """
    :param posts: a list of Post objects
    :param w2i: a word-to-index dictionary with all embedded words that will be used in training
    :param pad_tnk: the pad token
    :param unk_tkn: the unknown token
    :param batch_size: haow many posts will be in every batch
    :param max_len: the padding size for the texts
    :param title_max_len: the padding size for the title
    :param sort_data: boolean indicating if the list of posts  will be sorted by the size of the text
    :param a2i: a author-to-index dictionary
    :unk_author: unk author
    :return: a list of batches
    """
    if sort_data:
        posts.sort(key=lambda x: -len(x.tokens))
    offset = 0
    batches = []
    while offset < len(posts):
        batch_texts = []
        batch_titles = []
        batch_authors = []
        batch_labels = []
        start = offset
        end = min(offset + batch_size, len(posts))
        for i in range(start, end):
            batch_max_size = posts[start].text_size if sort_data else max(list(map(lambda x: x.text_size, posts[start:end])))
            batch_texts.append(get_indexed_text(w2i, pad_text(posts[i].tokens, max(min(max_len, batch_max_size), 1), pad_tkn=pad_tnk), unk_token=unk_tkn))
            batch_titles.append(get_indexed_text(w2i, pad_text(posts[i].title_tokens,
                                                               title_max_len, pad_tkn=pad_tnk), unk_token=unk_tkn))
            batch_authors.append(get_indexed_value(a2i, posts[i].author, unk_author))
            batch_labels.append(posts[i].label)
        batches.append({'text':np.array(batch_texts),
                        'title':np.array(batch_titles),
                        'author':np.array(batch_authors),
                        'label':np.array(batch_labels, dtype='float32')})
        offset += batch_size
    return batches


def get_embeddings(path='../input/embeddings/glove.6B.%dd.txt', size=50):
    """
    :param path: the directory where all glove embeddings are stored.
    glove embeddings can be downloaded from https://nlp.stanford.edu/projects/glove/
    :param size: the size of the embeddings. Must be in [50, 100, 200, 300]
    :return: a word-to-list dictionary with the embedded words and their corresponding embedding
    """
    embeddings_dict = {}
    f_path = path % size
    with open(f_path) as f:
        for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_dict[word] = coefs
    return embeddings_dict


def create_freq_vocabulary(tokenized_texts):
    """
    :param tokenized_texts: a list of lists of tokens
    :return: a word-to-integer dictionary with the value representing the frequency of the word in data
    """
    token_dict = {}
    for text in tokenized_texts:
        for token in text:
            try:
                token_dict[token] += 1
            except KeyError:
                token_dict[token] = 1
    return token_dict


def get_frequent_words(token_dict, min_freq):
    """
    :param token_dict: a word-to-integer dictionary with the value representing the frequency of the word in data
    :param min_freq: the minimum frequency
    :return: the list with tokens having frequency >= min_freq
    """
    return [x for x in token_dict if token_dict[x] >= min_freq]


def create_final_dictionary(posts,
                            min_freq,
                            unk_token,
                            pad_token,
                            embeddings_dict):
    """
    :param posts: a list of Post objects
    :param min_freq: the min times a word must be found in data in order not to be considered as unknown
    :param unk_token: the unknown token
    :param pad_token: the pad token
    :param embeddings_dict: a word-to-list dictionary with the embedded words and their corresponding embedding
    :return: a word-to-index dictionary with all the words that will be used in training
    """
    tokenized_texts = [x.tokens for x in posts] + [x.title_tokens for x in posts]
    voc = create_freq_vocabulary(tokenized_texts)
    print("tokens found in training data set:", len(voc))
    freq_words = get_frequent_words(voc, min_freq)
    print("tokens with frequency >= %d: %d" % (min_freq, len(freq_words)))
    words = list(set(freq_words).intersection(embeddings_dict.keys()))
    print("embedded tokens with frequency >= %d: %d" % (min_freq,len(words)))
    words = [pad_token, unk_token] + words
    return {w: i for i, w in enumerate(words)}


def create_author_dictionary(posts, min_freq):
    authors = [x.author for x in posts]
    author_dict = {}
    for author in authors:
        try:
            author_dict[author] += 1
        except KeyError:
            author_dict[author] = 1
    final_authors = [x for x in author_dict if author_dict[x] >= min_freq]
    return {a: i for i, a in enumerate(final_authors)}


def get_embeddings_matrix(word_dict, embeddings_dict, size):
    """
    :param word_dict: a word-to-index dictionary with the tokens found in data
    :param embeddings_dict: a word-to-list dictionary with the embedded words and their corresponding embedding
    :param size: the size of the word embedding
    :return: a matrix with all the embeddings that will be used in training
    """
    embs = np.zeros(shape=(len(word_dict), size))
    for word in word_dict:
        try:
            embs[word_dict[word]] = embeddings_dict[word]
        except KeyError:
            print('no embedding for: ', word)
    embs[1] = np.mean(embs[2:])

    return embs


def get_indexed_value(w2i, word, unk_token):
    """
    return the index of a token in a word-to-index dictionary
    :param w2i: the word-to-index dictionary
    :param word: the token
    :param unk_token: to unknown token
    :return: an integer
    """
    try:
        return w2i[word]
    except KeyError:
        return w2i[unk_token]


def get_indexed_text(w2i, words, unk_token):
    """
    return the indices of the all the tokens in a list in a word-to-index dictionary
    :param w2i: the word-to-index dictionary
    :param words: a list of tokens
    :param unk_token: to unknown token
    :return: a list of integers
    """
    return [get_indexed_value(w2i, word, unk_token) for word in words]


def pad_text(tokenized_text, maxlen, pad_tkn):
    """
    fills a list of tokens with pad tokens if the length of the list is larger than maxlen
    or return the maxlen last tokens of the list
    :param tokenized_text: a list of tokens
    :param maxlen: the max length
    :param pad_tkn: the pad token
    :return: a list of tokens
    """
    if len(tokenized_text) < maxlen:
        return [pad_tkn] * (maxlen - len(tokenized_text)) + tokenized_text
    else:
        return tokenized_text[len(tokenized_text) - maxlen:]


def load_posts(csv_file):
    """
    loads a csv file with posts and replaces the empty titles and texts
    :param csv_file:
    :return:
    """
    data = pd.read_csv(csv_file)
    data = data.fillna({'title': "nan", 'text': "nan", 'author': UNK_AUTHOR})
    return data


def get_posts(df):
    """
    returns a list of Posts. Posts with the same text we be considered as duplicates
    :param df: a dataframe  - the source of the data
    :return: all the posts in this dataset
    """
    posts = set()
    for i, d in df.iterrows():
        post = Post(*d)
        posts.add(post)
    return list(posts)


def split_data(data, split_point):
    """
    splits the data (for train and test)
    :param data: a list of posts
    :param split_point: the point of splitting
    :return: two lists of posts
    """
    return data[0:split_point], data[split_point:]


def generate_data(split_point, emb_size, min_freq=1, min_author_freq=3,
                  maxlen=1000, title_maxlen=10):
    """
    generates all neccesary components for training and evaluation (posts, embedding matrix, dictionaries and batches
    :param split_point: how many data will be used for training. the rest, will be used for evaluation
    :param emb_size: to size of word embeddings
    :param min_freq: how many times a word must be found in data in order t not being considered as unknown
    :param min_author_freq: least number of posts of the author in the dataset in order not to be consider unk
    even if its embedding is available
    :param maxlen: the padding size of text
    :param title_maxlen: the padding size of title
    :return: train_posts, test_posts, w2i, emb_matrix, train_batches, test_batches
    """
    df = load_posts("../input/fake_news_data/train.csv")
    posts = get_posts(df)
    train_posts, test_posts = split_data(posts, split_point)
    print('posts for training:', len(train_posts))
    print('posts for testing:', len(test_posts))
    embeddings_dict = get_embeddings(size=emb_size)
    w2i = create_final_dictionary(posts=posts, min_freq=min_freq, unk_token=UNK_TOKEN, pad_token=PAD_TOKEN,
                                  embeddings_dict=embeddings_dict)

    emb_matrix = get_embeddings_matrix(w2i, embeddings_dict, size=emb_size)
    a2i = create_author_dictionary(posts, min_author_freq)
    train_batches = create_batches(train_posts, w2i, a2i, max_len=maxlen, title_max_len=title_maxlen)
    test_batches = create_batches(test_posts, w2i, a2i, max_len=maxlen, title_max_len=title_maxlen)

    return {'train_posts': train_posts, 'test posts': test_posts, 'w2i': w2i,
            'a2i': a2i, 'emb_matrix': emb_matrix,
            'train_batches': train_batches, 'test_batches': test_batches}



