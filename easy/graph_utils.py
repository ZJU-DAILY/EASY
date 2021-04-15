from utils import *


def get_mask_with_sim(x2y, y2x, lens, link=None, logical_not=False) -> Tuple[Tensor, Tensor]:
    x2y = matrix_argmax(x2y, 1)
    y2x = matrix_argmax(y2x, 1)
    masks = get_bi_mapping(x2y, y2x, lens), \
            get_bi_mapping(y2x, x2y, (lens[1], lens[0]))
    if link is not None:
        from eval import get_hit_k
        link = link.to(x2y.device)
        not_masks = apply(torch.logical_not, *masks)
        print("Analyse mask acc")
        get_hit_k(x2y.view(-1, 1), link, 0, ignore=not_masks[0])
        get_hit_k(y2x.view(-1, 1), link, 1, ignore=not_masks[1])
    if logical_not:
        masks = apply(torch.logical_not, *masks)
    return masks


def get_bi_mapping(src2trg, trg2src, lens) -> Tensor:
    srclen, trglen = lens
    with torch.no_grad():
        i = torch.arange(srclen, device=src2trg.device).to(torch.long)
        return trg2src[src2trg[i]] == i


def filter_mapping(src2trg: Tensor, trg2src: Tensor, lens, values: Tuple[Tensor, Tensor], th):
    src2trg, trg2src, vals2t, valt2s = apply(lambda x: x.cpu().numpy(), src2trg, trg2src, *values)
    added_s = np.zeros(lens[0], dtype=np.int)
    added_t = np.zeros(lens[1], dtype=np.int)
    pair_s, pair_t = [], []
    for i in range(lens[0]):
        j = src2trg[i, 0]
        if added_t[j] == 1 or i != trg2src[j, 0]:
            continue
        gap_x2y = vals2t[i, 0] - vals2t[i, 1]
        gap_y2x = valt2s[j, 0] - valt2s[j, 1]
        if gap_y2x < th or gap_x2y < th:
            continue
        added_s[i] = 1
        added_t[j] = 1
        pair_s.append(i)
        pair_t.append(j)

    return torch.tensor([pair_s, pair_t])


def iterative_completion(src2trg: Tensor, trg2src: Tensor, eis: Tuple[Tensor, Tensor], lens: Tuple[int, int],
                         values: Optional[Tuple[Tensor, Tensor]] = None, th=0.15) -> \
        Tuple[Tensor, Tensor]:
    src_len, trg_len = lens
    if src_len > trg_len:
        ei0, ei1 = eis
        val0, val1 = values
        ei1, ei0 = iterative_completion(trg2src, src2trg,
                                        (ei1, ei0), (trg_len, src_len)
                                        , (val1, val0))
        return ei0, ei1
    with torch.no_grad():
        if values is None:
            i = torch.arange(src_len, device=src2trg.device).to(torch.long)
            ok_s2t = trg2src[src2trg[i]] == i
            trans_x, trans_y = i[ok_s2t], src2trg[ok_s2t]
            trans = torch.cat((trans_x[None, :], trans_y[None, :]), dim=0)
        else:
            trans = filter_mapping(src2trg, trg2src, lens, values, th).to(src2trg.device)
        print("src2trg selected", trans.size(1))
        ei_src, ei_trg = eis
        ei_src = ind2sparse(ei_src, src_len)
        ei_trg = ind2sparse(ei_trg, trg_len)
        trans = ind2sparse(trans, src_len, trg_len)

        trg_added = spspmm(spspmm(trans.t(), ei_src), trans)
        src_added = spspmm(spspmm(trans, ei_trg), trans.t())

        return src_added, trg_added


def graph_matching_distance(gs, gt, perm, sparse=True, argmax=True, eps=0.01):
    # gs, gt : sparse matrix
    # P      : [2, N] or [N, 2] indicating pairs
    #       or [N] indicating pairs with first col being torch.arange
    #       or [N, N] indicating similarity matrix(sparse/dense)
    if sparse:
        if perm.size(0) == 2:
            xy = perm
        elif perm.dim() == 2 and perm.size(1) == 2:
            xy = perm.t()
        else:
            y = matrix_argmax(perm, 1) if argmax else perm
            x = torch.arange(y.numel(), device=y.device).to(torch.long)
            xy = torch.stack((x, y), dim=0)
        perm = ind2sparse(xy, gs.size(0), gt.size(0))
        gsP = spspmm(gs, perm)
        PgsP = spspmm(perm.t(), gsP)
        dist = PgsP - gt
        dist = apply_on_sparse(torch.abs, dist)
        dist = scatter_op(dist, "sum", dim=1, dim_size=dist.size(0))
        deno = scatter_op(PgsP + gt, 'sum', dim=1, dim_size=dist.size(0))
        dist = dist / (deno + eps)
    else:
        gsP = spmm(gs, perm)
        PgsP = perm.t().mm(gsP)
        dist = torch.abs(PgsP - gt).sum(dim=1)
        dist = dist / (PgsP.sum(dim=1) + gt.to_dense().sum(dim=1) + 1)
    return dist
