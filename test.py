import gc

import faiss
import torch
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from patch_matcher import compare_two_spatial_approximation, compare_two_ransac
from functools import partial



def test(args, eval_ds, model):
    """Compute descriptors of the given dataset and compute the recalls."""
    
    model = model.eval()
    with torch.no_grad():
        logging.debug("Extracting database descriptors for evaluation/testing")
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                        batch_size=args.infer_batch_size, pin_memory=(args.device=="cuda"))
        all_descriptors = np.empty((len(eval_ds), args.fc_output_dim), dtype="float32")
        for images, indices in tqdm(database_dataloader, ncols=100):
            descriptors = model.embedd(images.to(args.device))
            descriptors = model.global_pool(descriptors)
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors
        
        logging.debug("Extracting queries descriptors for evaluation/testing using batch size 1")
        queries_infer_batch_size = 1
        queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, eval_ds.database_num+eval_ds.queries_num)))
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                        batch_size=queries_infer_batch_size, pin_memory=(args.device=="cuda"))
        for images, indices in tqdm(queries_dataloader, ncols=100):
            descriptors = model.embedd(images.to(args.device))
            descriptors = model.global_pool(descriptors)
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


def test_with_postprocessing(args, eval_ds, model, method="rapid", patch_size=3, stride=1):
    if method =="rapid":
        compare_two = partial(compare_two_spatial_approximation, stride=stride)
    elif method=="ransac":
        compare_two = partial(compare_two_ransac, stride=stride)


    model = model.eval()
    with torch.no_grad():
        logging.debug("Extracting database descriptors for evaluation/testing")
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                        batch_size=args.infer_batch_size, pin_memory=(args.device=="cuda"))

        all_global_descriptors = np.empty((len(eval_ds), args.fc_output_dim), dtype="float32")
        all_local_descriptors = [None for _ in range(len(eval_ds))]

        for images, indices in tqdm(database_dataloader, ncols=100):
            descriptors = model.embedd(images.to(args.device))
            global_descriptors, local_descriptor = model.local_pool(descriptors, patch_sizes_list=(patch_size,))
            local_descriptor = local_descriptor[0]

            global_descriptors = global_descriptors.cpu().numpy()
            local_descriptor = local_descriptor.cpu().numpy()

            all_global_descriptors[indices.numpy(), :] = global_descriptors

            indices = indices.numpy()
            for ind in range(len(indices)):
                all_local_descriptors[indices[ind]] = local_descriptor[ind]

        output_shape_db = (descriptors.shape[2], descriptors.shape[3])


        logging.debug("Extracting queries descriptors for evaluation/testing using batch size 1")
        queries_infer_batch_size = 1
        queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, eval_ds.database_num+eval_ds.queries_num)))
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                        batch_size=queries_infer_batch_size, pin_memory=(args.device=="cuda"))

        all_output_shape_q = []

        for images, indices in tqdm(queries_dataloader, ncols=100):
            descriptors = model.embedd(images.to(args.device))
            global_descriptors, local_descriptor = model.local_pool(descriptors, patch_sizes_list=(patch_size,))
            local_descriptor = local_descriptor[0]
            global_descriptors = global_descriptors.cpu().numpy()
            local_descriptor = local_descriptor.cpu().numpy()

            all_global_descriptors[indices.numpy(), :] = global_descriptors

            indices = indices.numpy()
            for ind in range(len(indices)):
                all_local_descriptors[indices[ind]] = local_descriptor[ind]
                all_output_shape_q.append((descriptors.shape[2], descriptors.shape[3]))

    queries_global_descriptors = all_global_descriptors[eval_ds.database_num:]
    database_global_descriptors = all_global_descriptors[:eval_ds.database_num]

    queries_local_descriptors = all_local_descriptors[eval_ds.database_num:]
    database_local_descriptors = all_local_descriptors[:eval_ds.database_num]

    del all_global_descriptors, all_local_descriptors
    gc.collect()

    # Use a kNN to find predictions
    faiss_index = faiss.IndexFlatL2(args.fc_output_dim)
    faiss_index.add(database_global_descriptors)

    assert args.top_knn_value >= max(args.recall_values)
    logging.debug("Calculating recalls")
    _, predictions = faiss_index.search(queries_global_descriptors, args.top_knn_value)

    positives_per_query = eval_ds.get_positives()

    recalls = np.zeros(len(args.recall_values))
    for query_index, pred in enumerate(predictions):
        for i, n in enumerate(args.recall_values):
            if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break
    # Divide by queries_num and multiply by 100, so the recalls are in percentages
    recalls0 = recalls / eval_ds.queries_num * 100
    recalls_str0 = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls0)])

    query_to_scores = [[] for q in range(len(queries_global_descriptors))]
    for query_index, pred in enumerate(predictions):
        for p in pred:
            qfeats = queries_local_descriptors[query_index]
            dbfeats = database_local_descriptors[p]

            score = compare_two(qfeats, dbfeats, output_shape_db=output_shape_db, output_shape_q=all_output_shape_q[query_index])
            query_to_scores[query_index].append((p, score))


    predictions = []
    for qi, q_s in enumerate(query_to_scores):
        q_s.sort(key=lambda p: p[1])
        q, s = zip(*q_s)
        predictions.append(q)

    # args.recall_values by default is [1, 5, 10, 20]
    recalls = np.zeros(len(args.recall_values))
    for query_index, pred in enumerate(predictions):
        for i, n in enumerate(args.recall_values):
            if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break
    # Divide by queries_num and multiply by 100, so the recalls are in percentages
    patch_refined_recalls = recalls / eval_ds.queries_num * 100
    patch_refined_recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, patch_refined_recalls)])

    return recalls0, recalls_str0, patch_refined_recalls, patch_refined_recalls_str









