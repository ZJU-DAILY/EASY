from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.io import read_txt_array
from torch_geometric.utils import sort_edge_index

from eval import *
from graph_utils import get_mask_with_sim
from text_sim import *
from transformer_helper import BERT, EmbeddingLoader


# REMAINS = [
#     "http://www.w3.org/1999/02/22-rdf-syntax-ns#langString",
#     "http://www.w3.org/2001/XMLSchema#integer",
#     "http://www.w3.org/2001/XMLSchema#double",
# ]


class DBP15k(InMemoryDataset):
    r"""The DBP15K dataset from the
    `"Cross-lingual Entity Alignment via Joint Attribute-Preserving Embedding"
    <https://arxiv.org/abs/1708.05045>`_ paper, where Chinese, Japanese and
    French versions of DBpedia were linked to its English version.
    Node features are given by pre-trained and aligned monolingual word
    embeddings from the `"Cross-lingual Knowledge Graph Alignment via Graph
    Matching Neural Network" <https://arxiv.org/abs/1905.11605>`_ paper.

    Args:
        root (string): Root directory where the dataset should be saved.
        pair (string): The pair of languages (:obj:`"en_zh"`, :obj:`"en_fr"`,
            :obj:`"en_ja"`, :obj:`"zh_en"`, :obj:`"fr_en"`, :obj:`"ja_en"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    file_id = '1dYJtj1_J4nYJdrDY95ucGLCuZXDXI7PL'

    def __init__(self, root, pair, device='cuda',
                 transform=None, pre_transform=None):
        assert pair in DBP15K_PAIRS
        self.pair = pair
        self.device = device
        super(DBP15k, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return DBP15K_PAIRS

    @property
    def processed_file_names(self):
        return '{0}.pt'.format(self.pair)

    def download(self):
        pass

    @torch.no_grad()
    def process(self):

        g1_path = osp.join(self.raw_dir, self.pair, 'triples_1')
        g2_path = osp.join(self.raw_dir, self.pair, 'triples_2')
        x1_path = osp.join(self.raw_dir, self.pair, 'id_features_1')
        x2_path = osp.join(self.raw_dir, self.pair, 'id_features_2')

        attrgnn_name2id_path = [osp.join(self.raw_dir, self.pair, "entity2id_" + lang + ".txt")
                                for lang in self.pair.split("_")]
        n2i_attrgnn = tuple([self.read_name2id_file(path) for path in attrgnn_name2id_path])

        edge_index1, rel1, assoc1, name_words1 = self.process_graph(
            g1_path, x1_path)
        edge_index2, rel2, assoc2, name_words2 = self.process_graph(
            g2_path, x2_path)

        name_paths = [osp.join(self.raw_dir, self.pair, "ent_ids_" + str(i)) for i in range(1, 3)]

        name2id1 = self.get_name2id(name_paths[0], assoc1)
        name2id2 = self.get_name2id(name_paths[1], assoc2)

        n2i_dataset = (name2id1, name2id2)
        saveobj(n2i_dataset, osp.join(self.raw_dir, self.pair, 'ent_name2id'))
        # translates = self.read_translate(n2i_dataset)
        # name_words1, name_words2 = translates
        total_y = self.process_y_n2id(osp.join(self.raw_dir, self.pair, "entity_seeds.txt"),
                                      n2i_attrgnn, n2i_dataset)
        hard_y, hard_val_y, hard_train_y = tuple(
            [self.process_y_n2id(osp.join(self.raw_dir, self.pair, ty + "_entity_seeds.txt"),
                                 n2i_attrgnn, n2i_dataset) for ty in ["test", "valid", "train"]])
        train_y, test_y, val_y = random_split(total_y, device=self.device)

        x1, x2 = self.bert_encode_maxpool([name_words1, name_words2])
        ground_truths = [total_y]  # , test_y, hard_y]

        embedding_loader = EmbeddingLoader('bert-base-cased')

        g1_words, g1_emb, g1_w2e, g1_e2w = \
            get_name_feature_map(name_words1, embedding_loader, device="cuda", normalize=False)
        g2_words, g2_emb, g2_w2e, g2_e2w = \
            get_name_feature_map(name_words2, embedding_loader, device="cuda", normalize=False)
        tokenizer = embedding_loader.tokenizer
        print(len(g1_words), len(g2_words))
        g1_tfidf, g2_tfidf = self.get_tf_idf(g1_words, name_words1, tokenizer), \
                             self.get_tf_idf(g2_words, name_words2, tokenizer)
        del embedding_loader

        sim_x2y = token_level_similarity(g1_tfidf, g2_tfidf, g1_emb, g2_emb, 1)
        sim_y2x = token_level_similarity(g2_tfidf, g1_tfidf, g2_emb, g1_emb, 1)

        x1, x2, g1_emb, g2_emb = apply(lambda x: x.to(self.device), x1, x2, g1_emb, g2_emb)

        print(sim_x2y._values().size(), sim_y2x._values().size())

        lev_x2y = pairwise_edit_distance(name_words1, name_words2)
        lev_y2x = lev_x2y.t()
        sims = save_similarity_matrix(False, lev_x2y=lev_x2y, lev_y2x=lev_y2x, sim_x2y=sim_x2y, sim_y2x=sim_y2x)

        # analyse
        for y in ground_truths:
            print("NEAP")
            evaluate_sim_matrix(y, sim_x2y, sim_y2x)
            print("EDIT-DIST")
            evaluate_sim_matrix(y, lev_x2y.to(self.device), lev_y2x.to(self.device))
            print("MAXPOOL")
            evaluate_embeds(x1, x2, y)
        # end
        lens = (x1.size(0), x2.size(0))
        data = Data(x1=x1, edge_index1=edge_index1, rel1=rel1, x2=x2,
                    edge_index2=edge_index2, rel2=rel2,
                    test_y=test_y, train_y=train_y, val_y=val_y,
                    total_y=total_y, hard_y=hard_y, hard_val_y=hard_val_y, hard_train_y=hard_train_y,
                    **sims)

        torch.save(self.collate([data]), self.processed_paths[0])

    def bert_encode_maxpool(self, sent_list):
        '''
         use BERT to encode sentences
        '''
        bert = BERT()
        bert.to("cuda")
        return [bert.pooled_encode_batched(lst, save_gpu_memory=True) for lst in sent_list]

    def str2id_map_to_list(self, mp):
        return sorted(list(mp.keys()), key=lambda x: mp[x])

    def get_assoc(self, n2i_curr, n2i_dataset):
        assoc = {}
        mx_id = 0
        for name, curr_id in n2i_curr.items():
            mx_id = max(curr_id, mx_id)
            assoc[curr_id] = n2i_dataset[name]

        x = np.zeros([mx_id + 1], np.long)
        for k, v in assoc.items():
            x[k] = v

        return torch.from_numpy(x)

    def process_y_n2id(self, link_path, n2i_curr, n2i_dataset):
        curr_reverse = self.pair[:2] == "en"
        assoc0, assoc1 = tuple([self.get_assoc(n2i_curr[i], n2i_dataset[i]) for i in range(2)])
        g1, g2 = read_txt_array(link_path, "\t", dtype=torch.long).t()
        if curr_reverse:
            g1, g2 = g2, g1
        g1 = assoc0[g1]
        g2 = assoc1[g2]
        return torch.stack([g1, g2], dim=0).to(self.device)

    def read_name2id_file(self, path, name_place=0, id_place=1, split='\t', skip=1, assoc=None):
        name2id = {}
        with open(path, "r") as f:
            for line in f:
                if skip > 0:
                    skip -= 1
                    continue
                info = line.strip().split(split)
                now = int(info[id_place])
                if assoc is not None:
                    now = assoc[now]
                name2id[info[name_place]] = now

        return name2id

    def get_count(self, words, entity_list):
        raw_count = get_count(words, entity_list)
        return to_torch_sparse(raw_count, device=self.device).to(float).t()

    def get_tf_idf(self, words, entity_list, tokenizer):
        raw_tf_idf = get_tf_idf(words, entity_list, tokenizer)
        return to_torch_sparse(raw_tf_idf, device=self.device).to(float).t()

    def get_name2id(self, name_path, assoc: Tensor):
        name2id = {}
        assoc = assoc.cpu().tolist()
        with open(name_path, 'r') as f:
            for line in f:
                info = line.strip().split('\t')
                name2id[info[1]] = assoc[int(info[0])]

        return name2id

    def process_graph(self, triple_path, feature_path):
        g1 = read_txt_array(triple_path, sep='\t', dtype=torch.long)
        subj, rel, obj = g1.t()
        name_dict = {}
        with open(feature_path, 'r') as f:
            for line in f:
                info = line.strip().split('\t')
                info = info if len(info) == 2 else info + ['']
                seq_str = remove_punc(info[1]).strip()
                if seq_str == "":
                    seq_str = '<unk>'
                name_dict[int(info[0])] = seq_str

        idx = torch.tensor(list(name_dict.keys()))
        assoc = torch.full((idx.max().item() + 1,), -1, dtype=torch.long)
        assoc[idx] = torch.arange(idx.size(0))

        subj, obj = assoc[subj], assoc[obj]
        edge_index = torch.stack([subj, obj], dim=0)
        edge_index, rel = sort_edge_index(edge_index, rel)

        # xs = [None for _ in range(idx.size(0))]
        names = [None for _ in range(idx.size(0))]
        for i in name_dict.keys():
            names[assoc[i]] = name_dict[i]
        # x = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)

        return edge_index, rel, assoc, names

    def process_y(self, path, assoc1, assoc2):
        row, col, mask = read_txt_array(path, sep='\t', dtype=torch.long).t()
        mask = mask.to(torch.bool)
        return torch.stack([assoc1[row[mask]], assoc2[col[mask]]], dim=0)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.pair)
