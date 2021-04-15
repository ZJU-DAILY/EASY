DATASET_FOLDER = "dataset/"
ATTR1 = "attr_triples_1"
ATTR2 = "attr_triples_2"
ENT_LINKS = "/ent_links"
REL1 = "/rel_triples_1"
REL2 = "/rel_triples_2"
DBP15K_PAIRS = ['ja_en', 'zh_en', 'fr_en']
SRPRS_PAIRS = ['en_fr', 'en_de']


def get_edge_index(dataset):
    if hasattr(dataset, "graph1"):
        return dataset.graph1["edge_index"], dataset.graph2["edge_index"]
    return dataset.edge_index1, dataset.edge_index2


def get_edge_type(dataset):
    if hasattr(dataset, "graph1"):
        return dataset.graph1["edge_type"], dataset.graph2["edge_type"]
    # return None, None
    return dataset.rel1, dataset.rel2
