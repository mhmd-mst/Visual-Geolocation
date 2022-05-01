import faiss
from sklearn import neighbors
import torch
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from sklearn.neighbors import NearestNeighbors


def test(args, eval_ds, model, rerank=False):
    """Compute descriptors of the given dataset and compute the recalls."""

    model = model.to(args.device).eval()
    with torch.no_grad():
        logging.debug("Extracting database descriptors for evaluation/testing")
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                         batch_size=args.infer_batch_size, pin_memory=(args.device == "cuda"))
        all_descriptors = np.empty((len(eval_ds), args.fc_output_dim), dtype="float32")
        for images, indices in tqdm(database_dataloader, ncols=100):
            descriptors = model(images.to(args.device))
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors

        logging.debug("Extracting queries descriptors for evaluation/testing using batch size 1")
        queries_infer_batch_size = 1
        queries_subset_ds = Subset(eval_ds,
                                   list(range(eval_ds.database_num, eval_ds.database_num + eval_ds.queries_num)))
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                        batch_size=queries_infer_batch_size, pin_memory=(args.device == "cuda"))
        for images, indices in tqdm(queries_dataloader, ncols=100):
            descriptors = model(images.to(args.device))
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors

    queries_descriptors = all_descriptors[eval_ds.database_num:]
    database_descriptors = all_descriptors[:eval_ds.database_num]

    # Use a kNN to find predictions
    faiss_index = faiss.IndexFlatL2(args.fc_output_dim)
    faiss_index.add(database_descriptors)
    del database_descriptors, all_descriptors
    args.recall_values = [1, 5, 10, 20]
    logging.debug("Calculating recalls")
    _, predictions = faiss_index.search(queries_descriptors, max(args.recall_values))

    # Re-Ranking
    if rerank:
        predictions = reranking(predictions, eval_ds)

    #### For each query, check if the predictions are correct
    positives_per_query = eval_ds.get_positives()
    # args.recall_values by default is [1, 5, 10, 20]
    recalls = np.zeros(len(args.recall_values))
    for query_index, pred in enumerate(predictions):
        for i, n in enumerate(args.recall_values):
            if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break
    # Divide by queries_num and multiply by 100, so the recalls are in percentages
    recalls = recalls / eval_ds.queries_num * 100
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])
    return recalls, recalls_str


def reranking(predictions, eval_ds):
    reranked_preds = list()
    for preds in predictions:
        database = list()
        for idx in preds:
            database.append(eval_ds.database_utms[idx])
        knn_re = NearestNeighbors(n_jobs=-1)
        knn_re.fit(database)
        tmp = knn_re.radius_neighbors(database, radius=5, return_distance=False)
        for z in range(len(tmp)):
            tmp[z] = tmp[z].tolist()
        out = []
        while len(tmp) > 0:
            first, *rest = tmp
            first = set(first)

            lf = -1
            while len(first) > lf:
                lf = len(first)

                rest2 = []
                for r in rest:
                    if len(first.intersection(set(r))) > 0:
                        first |= set(r)
                    else:
                        rest2.append(r)
                rest = rest2

            out.append(list(first))
            tmp = rest

        out.sort(key=len, reverse=True)
        cluster_list = out.copy()
        for i in range(len(out)):
            for j in range(len(out[i])):
                cluster_list[i][j] = preds[out[i][j]]
        l = cluster_list.copy()
        reranked_preds.append(sum(l, []))
    return reranked_preds


if __name__ == '__main__':
    from parser import parse_arguments
    from datasets import test_dataset
    from model import network

    args = parse_arguments()
    val_ds = test_dataset.TestDataset(args.val_set_folder, positive_dist_threshold=args.positive_dist_threshold)
    model = network.GeoLocalizationNet(args.backbone, args.fc_output_dim)
    x, y = test(args, val_ds, model)
