from fml.functional import sinkhorn

from dists import *
from utils import *


def matrix_sinkhorn(pred_or_m, expected=None, a=None, b=None):
    device = pred_or_m.device
    if expected is None:
        M = view3(pred_or_m).to(torch.float32)
        m, n = tuple(pred_or_m.size())
    else:
        m = pred_or_m.size(0)
        n = expected.size(0)
        M = cosine_distance(pred_or_m, expected)
        M = view3(M)

    if a is None:
        a = torch.ones([1, m], requires_grad=False, device=device)
    else:
        a = a.to(device)

    if b is None:
        b = torch.ones([1, n], requires_grad=False, device=device)
    else:
        b = b.to(device)
    P = sinkhorn(a, b, M, 1e-3, max_iters=300, stop_thresh=1e-3)
    return view2(P)


def token_level_similarity(src_w2e: Tensor, trg_w2e: Tensor, src_word_x: Tensor, trg_word_x: Tensor, sparse_k=1,
                           dense_mm=False, do_sinkhorn=True):
    # sim: Tensor = cosine_sim(src_word_x, trg_word_x)
    sim = cosine_sim(src_word_x, trg_word_x)
    if sparse_k is None:
        # print(src_w2e.size(), sim.size(), trg_w2e.size())
        tgm = spmm(src_w2e.t(), sim)
        tgm = spmm(trg_w2e.t(), tgm.t()).t()
    else:
        # sim_val, sim_id = torch.topk(sim, sparse_k)
        # id_x = torch.arange(src_word_x.size(0), dtype=torch.long).to(sim_id.device).view(-1, 1).expand_as(sim_id)
        # ind = torch.stack([id_x.view(-1), sim_id.view(-1)], dim=0)
        # sim = ind2sparse(ind, sim.size(), values=sim_val.view(-1)).to(float)
        # # src_w2e = rebuild_with_indices(src_w2e).to(float)
        # del sim_val, sim_id, id_x, ind
        sim = remain_topk_sim(sim, k=sparse_k).to(float)
        if dense_mm:
            tgm = src_w2e.t().to_dense().mm(sim.to_dense())
            tgm = tgm.mm(trg_w2e.to_dense())
        else:
            tgm = spspmm(src_w2e.t(), sim)
            tgm = spspmm(tgm, trg_w2e)
    if do_sinkhorn:
        tgm = sinkhorn_process(tgm)
    return dense_to_sparse(tgm)


def sinkhorn_process(M: Tensor):
    if M.is_sparse:
        M = M.to_dense()
    return dense_to_sparse(matrix_sinkhorn(1 - masked_minmax(M)))
