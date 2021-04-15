from data_utils import *
import regex
import string
from utils import *

from Levenshtein import ratio
from multiprocessing import Pool
import multiprocessing
from functools import partial

import collections
import os.path as osp

PREFIX = r'http(s)?://[a-z\.]+/[^/]+/'


def get_punctuations():
    # zh = zhon.hanzi.punctuation
    en = string.punctuation
    zh = ""
    puncs = set()
    for i in (zh + en):
        puncs.add(i)
    return puncs


PUNC = get_punctuations()


def remove_punc(str, punc=None):
    if punc is None:
        punc = PUNC
    if punc == '':
        return str
    return ''.join([' ' if i in punc else i for i in str])


def remove_prefix_to_list(entity_dict: {}, prefix=PREFIX) -> []:
    # punc = get_punctuations()
    punc = ''

    tmp_dict = {}
    entity_list = []
    p = regex.compile(prefix)
    for ent in entity_dict.keys():
        res = p.search(ent)
        if res is None:
            entity_list.append(remove_punc(ent, punc))
        else:
            _, end = res.span()
            entity_list.append(remove_punc(ent[end:], punc))

        tmp_dict[entity_list[-1]] = entity_dict[ent]
        # print(ent, entity_list[-1])
    entity_list = sorted(entity_list, key=lambda x: entity_dict[x] if x in entity_dict else tmp_dict[x])
    return entity_list


def normalize_vectors(embeddings, center=False):
    if center:
        embeddings -= torch.mean(embeddings, dim=0, keepdim=True)
    embeddings /= torch.linalg.norm(embeddings, dim=1, keepdim=True)
    return embeddings


def get_count(words, ent_lists, binary=True):
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(vocabulary=words, lowercase=False, tokenizer=lambda x: x.split(), binary=binary)
    return vectorizer.fit_transform(ent_lists)


def get_tf_idf(words, ent_lists, bert_tokenizer=None):
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    if bert_tokenizer is None:
        tokenizer = lambda x: x.split()
    else:
        tokenizer = lambda x: bert_tokenizer.tokenize(x)
    vectorizer = CountVectorizer(vocabulary=words, lowercase=False, tokenizer=tokenizer, binary=False)
    transformer = TfidfTransformer()
    X = vectorizer.fit_transform(ent_lists)
    tfidf = transformer.fit_transform(X)
    return tfidf


def get_fasttext_aligned_vectors(words, device, lang):
    embs = {}
    with open(osp.join("aligned_vectors", 'wiki.{}.align.vec'.format(lang)), 'r') as f:
        for i, line in enumerate(f):
            info = line.strip().split(' ')
            if len(info) > 300:
                embs[info[0]] = torch.tensor([float(x) for x in info[1:]])
            else:
                embs['**UNK**'] = torch.tensor([float(x) for x in info])
    # for word in words:
    #     word = word.replace('#', '')
    #     if word not in embs:
    #         print("fasttext unk:", word)
    word_embeds = [embs.get(word.replace('#', '').lower(), embs['**UNK**']) for word in words]

    return torch.stack(word_embeds, dim=0).to(device)


def tokenize(sent, tokenizer):
    if tokenizer is None:
        return sent.split()
    else:
        return tokenizer.tokenize(sent)


def get_name_feature_map(sents, embedding_loader=None, device='cuda',
                         batch_size=2048, use_fasttext=False, lang=None,
                         **kwargs):
    word_id_map = {}
    entity2word = []
    word2entity = collections.defaultdict(set)
    # device = torch.device(kwargs.get('device', 'cuda'))
    # batch_size = kwargs.get('batch_size', 2048)
    tokenizer = None if embedding_loader is None else embedding_loader.tokenizer

    for ent_id, sent in enumerate(sents):
        entity2word.append([])
        for word in tokenize(sent, tokenizer):
            word_id_map, word_id = add_cnt_for(word_id_map, word)
            entity2word[-1].append(word_id)
            word2entity[word_id].add(ent_id)
    word2entity = [word2entity[i] for i in range(len(word_id_map))]
    words = mp2list(word_id_map)
    if use_fasttext:
        if isinstance(lang, str):
            embeddings = get_fasttext_aligned_vectors(words, device, lang)
        else:
            embeddings = torch.cat([get_fasttext_aligned_vectors(words, device, lang) for lang in lang], dim=1)
    else:
        i = 0
        all_embed = []
        embed_size = 0
        lens = []
        while i < len(sents):
            embed, length = embedding_loader.get_embed_list(sents[i:min(i + batch_size, len(sents))], True)
            i += batch_size
            embed_size = embed.size(-1)
            lens.append(length.cpu().numpy())
            all_embed.append(embed.cpu().numpy())
        vectors = [emb for batch in all_embed for emb in batch]
        lens = [l for batch in lens for l in batch]
        vectors = [vectors[i][:lens[i]] for i in range(len(vectors))]
        embeddings = torch.zeros([len(words), embed_size], device=device, dtype=torch.float)
        for i, ent in enumerate(entity2word):
            index = torch.tensor(ent, device=device, dtype=torch.long)
            embeddings[index] += torch.tensor(vectors[i]).to(device)[:len(ent)]
            if i % 5000 == 0:
                print("average token embed --", i, "complete")
        if kwargs.get('size_average', True):
            sizes = torch.tensor([len(i) for i in word2entity]).to(device)
            embeddings /= sizes.view(-1, 1)

    if kwargs.get('normalize', True):
        embeddings = normalize_vectors(embeddings, kwargs.get('center', True))
    return words, embeddings.to(device), word2entity, entity2word


# CPM
def gen_mean(vals, p):
    p = float(p)
    return np.power(
        np.mean(
            np.power(
                np.array(vals, dtype=complex),
                p),
            axis=0),
        1 / p
    )


def reduce(tensor, reduction='mean', dim=0):
    if reduction == 'mean':
        return torch.mean(tensor, dim)
    if reduction == 'max':
        return torch.max(tensor, dim)[0]
    if reduction == 'min':
        return torch.min(tensor, dim)[0]
    if reduction == 'sum':
        return torch.sum(tensor, dim)
    if 'p_mean_' in reduction:
        p = float(reduction.split('_')[-1])
        vals = tensor.cpu().numpy()
        return torch.from_numpy(gen_mean(vals, p).real).to(tensor.device)


def embed_word2entity(ent2word, word_emb, reduction='max') -> Tensor:
    ent_emb = []
    for ent in ent2word:
        ent_emb.append(reduce(word_emb[torch.tensor(ent, device=word_emb.device)], reduction).squeeze())
    return torch.stack(ent_emb, dim=0)


def cpm_embedding(ent2word, words, cpm_types, models=('en', 'fr')):
    word_vec = [get_fasttext_aligned_vectors(words, 'cuda', lang) for lang in models]
    cpms = torch.cat([embed_word2entity(ent2word, vec, ty) for vec in word_vec for ty in cpm_types], dim=1)
    return cpms.to('cpu')


def pairwise_edit_distance(sent0, sent1, to_tensor=True):
    x = np.empty([len(sent0), len(sent1)], np.float)
    print(multiprocessing.cpu_count())
    pool = Pool(processes=multiprocessing.cpu_count())
    for i, s0 in enumerate(sent0):
        if i % 5000 == 0:
            print("edit distance --", i, "complete")
        x[i, :] = pool.map(partial(ratio, s0), sent1)
        # for j, s1 in enumerate(sent1):
        #     x[i, j] = distance(s0, s1)

    if to_tensor:
        return (torch.from_numpy(x).to(torch.float))
