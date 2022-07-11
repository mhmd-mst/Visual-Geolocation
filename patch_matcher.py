import numpy as np
import torch
import cv2


STRIDE = 1


def torch_nn(x, y):
    mul = torch.matmul(x.T, y)

    dist = 2 - 2 * mul

    fw_inds = torch.argmin(dist, 0)
    bw_inds = torch.argmin(dist, 1)

    return fw_inds, bw_inds

def calc_keypoint_centers_from_patches(output_shape, stride):
        H = output_shape[0]
        W = output_shape[1]

        num_regions = H * W

        k = 0
        keypoints = np.zeros((2, num_regions), dtype=int)
        # Assuming sensible values for stride here, may get errors with large stride values
        for i in range(0, H, stride):
            for j in range(0, W, stride):
                keypoints[0, k] = j
                keypoints[1, k] = i
                k += 1

        return keypoints


def compare_two_spatial_approximation(qfeat, dbfeats, output_shape_q, output_shape_db, stride=STRIDE):
        indices_q = calc_keypoint_centers_from_patches(output_shape_q, stride)
        indices_db = calc_keypoint_centers_from_patches(output_shape_db, stride)

        qfeat = torch.FloatTensor(qfeat)
        dbfeats = torch.FloatTensor(dbfeats)

        fw_inds, bw_inds = torch_nn(qfeat, dbfeats)
        fw_inds = fw_inds.cpu().numpy()
        bw_inds = bw_inds.cpu().numpy()

        mutuals = np.atleast_1d(np.argwhere(bw_inds[fw_inds] == np.arange(len(fw_inds))).squeeze())

        if len(mutuals) > 0:
            index_keypoints = indices_db[:, mutuals]
            query_keypoints = indices_q[:, fw_inds[mutuals]]

            spatial_dist = index_keypoints - query_keypoints # manhattan distance works reasonably well and is fast
            mean_spatial_dist = np.mean(spatial_dist, axis=1)

            # residual between a spatial distance and the mean spatial distance. Smaller is better
            s_dists_x = spatial_dist[0, :] - mean_spatial_dist[0]
            s_dists_y = spatial_dist[1, :] - mean_spatial_dist[1]
            s_dists_x = np.absolute(s_dists_x)
            s_dists_y = np.absolute(s_dists_y)

            # anchor to the maximum x and y axis index for the patch "feature space"
            xmax = max([np.max(indices_q[0, :]), np.max(indices_db[0, :])])
            ymax = max([np.max(indices_q[1, :]), np.max(indices_db[1, :])])

            # find second-order residual, by comparing the first residual to the respective anchors
            # after this step, larger is now better
            # add non-linearity to the system to excessively penalise deviations from the mean
            s_score = (xmax - s_dists_x)**2 + (ymax - s_dists_y)**2
            s_score = - s_score.sum()/qfeat.shape[1]

        else:
            s_score = 0.

        return s_score


def compare_two_ransac(qfeat, dbfeats, output_shape_q, output_shape_db, stride=1):
        keypoints_q = calc_keypoint_centers_from_patches(output_shape_q, stride)
        keypoints_db = calc_keypoint_centers_from_patches(output_shape_db, stride)

        qfeat = torch.FloatTensor(qfeat)
        dbfeats = torch.FloatTensor(dbfeats)

        fw_inds, bw_inds = torch_nn(qfeat, dbfeats)
        fw_inds = fw_inds.cpu().numpy()
        bw_inds = bw_inds.cpu().numpy()

        mutuals = np.atleast_1d(np.argwhere(bw_inds[fw_inds] == np.arange(len(fw_inds))).squeeze())

        if len(mutuals) > 0:
            index_keypoints = keypoints_db[:, mutuals]
            query_keypoints = keypoints_q[:, fw_inds[mutuals]]

            index_keypoints = np.transpose(index_keypoints)
            query_keypoints = np.transpose(query_keypoints)

            _, mask = cv2.findHomography(index_keypoints, query_keypoints, cv2.FM_RANSAC, ransacReprojThreshold=stride*1.5)

            inlier_index_keypoints = index_keypoints[mask.ravel() == 1]
            inlier_count = inlier_index_keypoints.shape[0]
            s_score = -inlier_count / qfeat.shape[1]

        else:
            s_score = 0.

        return s_score



