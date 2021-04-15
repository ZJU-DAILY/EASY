from torch_geometric.data import Data, InMemoryDataset

from eval import *
from graph_utils import get_mask_with_sim
from text_sim import *
from transformer_helper import BERT, EmbeddingLoader

SRPRS_REL1 = '/triples_1'
SRPRS_REL2 = '/triples_2'
CPM_TYPES = ('min', 'mean', 'max')


class SRPRS(InMemoryDataset):
    def __init__(self, root, pair, device="cuda",
                 use_fasttext=False,
                 cpm_types=None,
                 multilingual_bert=False,
                 mean_pool=False):
        self.pair = pair
        self.device = device
        self.use_fasttext = use_fasttext
        self.mul_bert = multilingual_bert
        self.mean_pool = mean_pool
        # self.lang = get_lang_name(root)
        self.cpm = bool(cpm_types is not None and len(cpm_types) > 0)
        self.cpm_types = cpm_types
        assert pair in SRPRS_PAIRS
        super(SRPRS, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [ATTR1, ATTR2, ENT_LINKS, SRPRS_REL1, SRPRS_REL2]

    @staticmethod
    def bool_suffix(name, val):
        return ['', '_' + name][val]

    @property
    def processed_file_names(self):
        return '{0}{1}{2}{3}{4}.pt'.format(self.pair,
                                           self.bool_suffix('ft', self.use_fasttext),
                                           self.bool_suffix('cpm', self.cpm),
                                           self.bool_suffix('mul', self.mul_bert),
                                           self.bool_suffix('mean_pool', self.mean_pool))

    def download(self):
        pass

    def process_one_graph(self, rel_pos: str):

        ei = [[], []]
        et = []

        rel_idx = {}
        ent_idx = {}

        with open(rel_pos, "r") as f:
            for line in f.readlines():
                now = line.strip().split('\t')
                ent_idx, s = add_cnt_for(ent_idx, now[0])
                rel_idx, p = add_cnt_for(rel_idx, now[1])
                ent_idx, o = add_cnt_for(ent_idx, now[2])
                ei[0].append(s)
                ei[1].append(o)
                et.append(p)

        rel_graph = torch.tensor(ei)
        rel_graph_edge_type = torch.tensor(et)

        rel_graph = {"edge_index": rel_graph, "edge_type": rel_graph_edge_type}
        return rel_graph, rel_idx, ent_idx

    @staticmethod
    def process_link(links_pos, ent1, ent2):
        link_index = [[], []]
        ent1_sz = len(ent1)
        p = regex.compile(r'http(s)?://[a-z\.]+/[^/]+/')

        with open(links_pos, "r") as f:
            for line in f.readlines():
                now = line.strip().split('\t')
                link_index[0].append(ent1[now[0]])
                link_index[1].append(ent2[now[1]])
        link_graph = torch.tensor(link_index)

        return link_graph

    def get_tf_idf(self, words, entity_list, tokenizer):
        raw_tf_idf = get_tf_idf(words, entity_list, tokenizer)
        return to_torch_sparse(raw_tf_idf, device=self.device).to(float).t()

    @staticmethod
    def bert_encode(sent_list, bert):
        '''
         use BERT to encode sentences
        '''
        return [bert.pooled_encode_batched(lst, save_gpu_memory=True, layer=1) for lst in sent_list]

    @staticmethod
    def load_id2name(path, name2id):
        mp = {}
        with open(path) as f:
            for line in f.readlines():
                info = line.strip().split()
                mp[info[0]] = info[1]
        return mp2list({mp[k]: v for k, v in name2id.items()})

    @torch.no_grad()
    def process(self):
        root = osp.join(self.raw_dir, self.pair + "_15k_V1")
        bert_model = 'bert-base-multilingual-cased' if self.mul_bert else 'bert-base-cased'

        self.graph1, self.rel1, self.ent1 = self.process_one_graph(root + SRPRS_REL1)
        self.graph2, self.rel2, self.ent2 = self.process_one_graph(root + SRPRS_REL2)
        total_y = self.process_link(root + ENT_LINKS, self.ent1, self.ent2).to(self.device)
        saveobj((self.ent1, self.ent2), osp.join(root, "ent_name2id"))
        saveobj((self.rel1, self.rel2), osp.join(root, "rel_name2id"))

        train_y, test_y, val_y = random_split(total_y, device=self.device)

        name_words1, name_words2 = remove_prefix_to_list(self.ent1), remove_prefix_to_list(self.ent2)
        if self.pair == 'dbp_wd':
            name_words2 = self.load_id2name(osp.join(root, 'id2name'), self.ent2)

        if self.use_fasttext:
            g1_lang, g2_lang = self.pair.split('_')
            if g1_lang == 'dbp':
                g1_lang = g2_lang = 'en'
            embedding_loader = tokenizer = None
            name_words1 = [remove_punc(sent) for sent in name_words1]
            name_words2 = [remove_punc(sent) for sent in name_words2]
        else:
            g1_lang = g2_lang = ''
            embedding_loader = EmbeddingLoader(bert_model, self.device, layer=1)
            tokenizer = embedding_loader.tokenizer
        g1_token, g1_emb, g1_w2e, g1_e2w = \
            get_name_feature_map(name_words1, embedding_loader, normalize=False,
                                 use_fasttext=self.use_fasttext, lang=(g1_lang, g2_lang))
        g2_token, g2_emb, g2_w2e, g2_e2w = \
            get_name_feature_map(name_words2, embedding_loader, normalize=False,
                                 use_fasttext=self.use_fasttext, lang=(g1_lang, g2_lang))

        g1_tfidf, g2_tfidf = self.get_tf_idf(g1_token, name_words1, tokenizer), \
                             self.get_tf_idf(g2_token, name_words2, tokenizer)
        del embedding_loader

        rawsim_x2y = token_level_similarity(g1_tfidf, g2_tfidf, g1_emb, g2_emb, do_sinkhorn=False)
        rawsim_y2x = token_level_similarity(g2_tfidf, g1_tfidf, g2_emb, g1_emb, do_sinkhorn=False)

        sim_x2y, sim_y2x = apply(sinkhorn_process, rawsim_x2y, rawsim_y2x)

        cpm_x1, cpm_x2 = None, None
        pool = 'mean' if self.mean_pool else 'max'
        if self.use_fasttext:
            if self.cpm:
                cpm_x1, cpm_x2 = cpm_embedding(g1_e2w, g1_token, self.cpm_types, (g1_lang, g2_lang)), \
                                 cpm_embedding(g2_e2w, g2_token, self.cpm_types, (g1_lang, g2_lang))
            x1, x2 = embed_word2entity(g1_e2w, g1_emb, pool), embed_word2entity(g2_e2w, g2_emb, pool)
        else:
            bert = BERT(model=bert_model, pool=pool)
            bert.to(self.device)
            x1, x2 = self.bert_encode([name_words1, name_words2], bert)
            del bert
        name_words1 = [remove_punc(i, '_') for i in name_words1]
        name_words2 = [remove_punc(i, '_') for i in name_words2]
        ground_truths = [total_y]
        print("sim cnt", sim_x2y._values().numel(), sim_y2x._values().numel())
        lev_x2y = pairwise_edit_distance(name_words1, name_words2)
        lev_y2x = lev_x2y.t()
        # lev_x2y, lev_y2x = remain_topk_sim(lev_x2y), remain_topk_sim(lev_y2x)
        if self.cpm:
            cpm_x2y = cosine_sim(cpm_x1, cpm_x2)
            cpm_y2x = cosine_sim(cpm_x2, cpm_x1)
        else:
            cpm_x2y = None
            cpm_y2x = None
        sims = save_similarity_matrix(False, lev_x2y=lev_x2y, lev_y2x=lev_y2x,
                                      # cosine_x2y=cosine_sim(x1, x2),
                                      # cosine_y2x=cosine_sim(x2, x1),
                                      cpm_x2y=cpm_x2y,
                                      cpm_y2x=cpm_y2x)

        for y in ground_truths:
            print("NEAP-wo sinkhorn")
            evaluate_sim_matrix(y, rawsim_x2y, rawsim_y2x)
            print('NEAP')
            evaluate_sim_matrix(y, sim_x2y, sim_y2x)
            print("EDIT-DIST")
            evaluate_sim_matrix(y, lev_x2y, lev_y2x)
            print("MAX/MEAN")
            cos_x2y = cosine_sim(x1, x2)
            cos_y2x = cosine_sim(x2, x1)
            cos_x2y, cos_y2x = remain_topk_sim(cos_x2y), \
                               remain_topk_sim(cos_y2x)

            evaluate_sim_matrix(y, cos_x2y, cos_y2x)
            if self.cpm:
                print("CPM")
                evaluate_sim_matrix(y, cpm_x2y, cpm_y2x)

        for g in [self.graph1, self.graph2]:
            if isinstance(g, dict):
                for k, i in g.items():
                    g[k] = i.to(device=self.device)
            else:
                g.to(device=self.device)
        lens = (x1.size(0), x2.size(0))
        ei1, ei2 = get_edge_index(self)
        et1, et2 = get_edge_type(self)
        x2y_mask, y2x_mask = get_mask_with_sim(sim_x2y, sim_y2x, lens)

        graph = Data(x1=x1, x2=x2, sim_x2y=sim_x2y, sim_y2x=sim_y2x, x2y_mask=x2y_mask, y2x_mask=y2x_mask,
                     edge_index1=ei1, edge_index2=ei2, rel1=et1, rel2=et2, total_y=total_y,
                     test_y=test_y, train_y=train_y, val_y=val_y, **sims)

        torch.save(self.collate([graph]), self.processed_paths[0])

    def get_len(self):

        self.triple1_len = len(self.graph1)
        self.triple2_len = len(self.graph2)
        self.ent1_len = len(self.ent1)
        self.ent2_len = len(self.ent2)
        self.rel1_len = len(self.SRPRS_REL1)
        self.rel2_len = len(self.SRPRS_REL2)

