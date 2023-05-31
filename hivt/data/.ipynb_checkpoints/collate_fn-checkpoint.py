from typing import Dict, List, Tuple

from trajpred.tptensor import TPTensor


class Collate_fn:

    def __init__(self, name: str = None):
        self.__name = name

    def __call__(self, batch: List[Tuple[str, TPTensor, TPTensor]]) -> Tuple[Dict[str, TPTensor], TPTensor]:
        return_batch = {}
        batch_size = len(batch)

        # Batching by use a list for non-fixed size
        for key in batch[0][1].keys():
            try: \
                return_batch[key] = torch.stack([x[key] for _, x, _ in batch])
            except:
                try:
                    if type(batch[0][1][key]) == dict:
                        return_batch[key] = [x[key] for _, x, _ in batch]
                    else:
                        return_batch[key] = torch.stack([x[key][0] for _, x, _ in batch])
                except:
                    breakpoint()
                    print('***** batch size mismatch, exceeding max num of lanes *****')
                    return_batch[key] = torch.stack([x[key][:batch[0][1][key].shape[0]] for _, x, _ in batch])
        # for key in batch[0][2].keys():
        ground_truth = torch.stack([x for _, _, x in batch])
        name = self.__name or batch[0][0]

        # return return_batch, ground_truth
        return {name: return_batch
                }, ground_truth
