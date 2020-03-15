import torch
import numpy as np

def rowSoftMax(MunkresOut, scale=100.0, threshold=0.7):
    
    clutter = torch.ones(MunkresOut.size(0), MunkresOut.size(1), 1).cuda() * threshold
    return F.softmax(torch.cat([MunkresOut, clutter], dim=2)*scale,  dim=2)


def colSoftMax(MunkresOut, scale=100.0, threshold=0.7):
    
    missed = torch.ones(MunkresOut.size(0), 1, MunkresOut.size(2)).cuda() * threshold
    return F.softmax(torch.cat([MunkresOut, missed], dim=1)*scale,  dim=1)

def missedObjectPerframe(colsoftmaxed):
    
    fn = torch.sum(colsoftmaxed[:, -1, :])
    return fn

def falsePositivePerFrame(rowsoftmax):
    
    fp = torch.sum(rowsoftmax[:, :, -1])
    return fp

def missedMatchErrorV3(prev_id, gt_ids, hypo_ids, colsoftmaxed, states, toUpdate):
    
    pre_id = copy.deepcopy(prev_id)
    updated_gt_ids = updateCurrentListV3(colsoftmaxed, gt_ids)
    id_switching = torch.zeros(1).float().cuda()
    softmaxed = colsoftmaxed[:, :-1, :]

    toputOne_h = []
    toputOne_w = []

    # to record hypo ids needed to switch target images ex. [1,2, 3,4, 5,6....]
    toswitch = []

    for w in range(len(updated_gt_ids)):
        _, idx = torch.max(softmaxed[0, :, w], dim=0)
        if updated_gt_ids[w] == -1 or (gt_ids[w] not in pre_id.keys()):  # lost object or new object or both
            # print("gt id is lost:", gt_ids[w])
            if gt_ids[w] in pre_id.keys():  # not new object but lost

                if pre_id[gt_ids[w]] in hypo_ids:
                    tmp = list(range(len(hypo_ids)))
                    tmp.pop(hypo_ids.index(pre_id[gt_ids[w]]))
                    id_switching = id_switching + torch.sum(softmaxed[0, tmp, w])
                    # print("mm is here")
                    # print(w)
                    # print(tmp)

                else:
                    # print("i am here")
                    id_switching = id_switching + torch.sum(softmaxed[0, :, w])

            elif updated_gt_ids[w] != -1:  # new object but not lost
                # add object id to prev_id, update prev id to current
                toputOne_w.append(w)
                toputOne_h.append(int(idx))
                pre_id[updated_gt_ids[w]] = hypo_ids[int(idx)] + 0

            else:  # new object and lost
                id_switching = id_switching + torch.sum(softmaxed[0, :, w])

        # if object w is not assigned to an target int(hypo_ids[idx])
        # same as previous target pre_id[int(updated_gt_ids[w])]
        elif pre_id[updated_gt_ids[w]] != hypo_ids[int(idx)]:
            if pre_id[updated_gt_ids[w]] in hypo_ids:
                tmp_idx = hypo_ids.index(pre_id[updated_gt_ids[w]])  # index of previous hypo id to this gt_id
                toputOne_w.append(w)
                toputOne_h.append(tmp_idx)  # we minimize old hypo track
                tmp = list(range(len(hypo_ids)))
                tmp.pop(tmp_idx)
                id_switching = id_switching + torch.sum(softmaxed[0, tmp, w])
                # switch target templates
                if toUpdate and pre_id[updated_gt_ids[w]] not in toswitch:  # if pair not yet switched

                    toswitch.append(pre_id[updated_gt_ids[w]])
                    toswitch.append(hypo_ids[int(idx)])
                    state_to_switch = states[pre_id[updated_gt_ids[w]]]
                    states[pre_id[updated_gt_ids[w]]] = states[hypo_ids[int(idx)]]
                    states[hypo_ids[int(idx)]] = state_to_switch
                    del state_to_switch

            else:
                id_switching = id_switching + torch.sum(softmaxed[0, :, w])  # todo wrong
                toputOne_w.append(w)
                toputOne_h.append(int(idx))  # todo

            # update prev id to current
            pre_id[updated_gt_ids[w]] = hypo_ids[int(idx)] + 0

        else:  # no id switching, no missed, prev_id[updated_gt_ids[w]] == hypo_ids[int(idx)]
            tmp = list(range(len(hypo_ids)))
            tmp.pop(int(idx))
            id_switching = torch.sum(softmaxed[0, tmp, w]) + id_switching
            toputOne_w.append(w)
            toputOne_h.append(int(idx))

    mask_for_matrix = createMatrix(toputOne_h, toputOne_w, softmaxed.size(1), softmaxed.size(2))

    return [id_switching, mask_for_matrix, pre_id]