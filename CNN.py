
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
import torch
from utils import *


    
class LaneDetector:
    DEFAULT_IMAGE_SIZE = (512, 256)

    def __init__(self, enet, hnet=None, with_projection=True):
        self._enet = enet
        self._hnet = hnet
        self._default_homography = torch.tensor(
            [[-2.0484e-01, -1.7122e+01,  3.7991e+02],
             [ 0.0000e+00, -1.6969e+01,  3.7068e+02],
             [ 0.0000e+00, -4.6739e-02,  1.0000e+00]],
            dtype=torch.float32
        )
        self._eps = 1.0
        self._with_projection = with_projection

    def __call__(self, image, y_positions=None):
        image = self._preprocess_image(image)
        if y_positions is None:
            y_positions = np.linspace(50, image.shape[2], 30)
        #print(image.shape)
        binary_logits, instance_embeddings = self._enet(image)
        segmentation_map = binary_logits.squeeze().argmax(dim=0)
        instances_map = self._cluster(segmentation_map, instance_embeddings)
        lanes = self._extract_lanes(instances_map)
        if len(lanes) ==0:
           return
        print(f"Detected {len(lanes)} lanes")
        
        #print(lanes)
        if self._with_projection:
            projected_lanes = self._project_lanes(lanes)
            # print(projected_lanes)
            coefs = self._fit(projected_lanes)
            #print(lanes[0])
            #print(projected_lanes[0])
            # print(coefs)
            y_positions_projected = self._project_y(y_positions)
            #print(y_positions_projected)
            fitted_lanes = self._predict_lanes(coefs, y_positions_projected)
            #print(fitted_lanes)
            reprojected_lanes = self._reproject(fitted_lanes)
            #print(reprojected_lanes)
            predicted_lanes = reprojected_lanes
        else:
            coefs = self._fit(lanes)
            #print(coefs)
            fitted_lanes = self._predict_lanes(coefs, y_positions)
            predicted_lanes = fitted_lanes

        predicted_lanes = self._postprocess_result(predicted_lanes)

        return instances_map.cpu().numpy(), predicted_lanes.cpu().numpy()

    def _cluster(self, segmentation_map, instance_embeddings):
        segmentation_map = segmentation_map.flatten()
        # print(segmentation_map.shape)
        instance_embeddings = instance_embeddings.squeeze().permute(1, 2, 0).reshape(segmentation_map.shape[0], -1)
        #print(instance_embeddings.shape)
        # print(instance_embeddings[:5, :])
        assert segmentation_map.shape[0] == instance_embeddings.shape[0]

        mask_indices = segmentation_map.nonzero().flatten()
        #print(mask_indices.shape)
        cluster_data = instance_embeddings[mask_indices].detach().cpu()
        #print(cluster_data.shape)

        clusterer = DBSCAN(eps=self._eps)
        labels = clusterer.fit_predict(cluster_data)
        labels = torch.tensor(labels, dtype=instance_embeddings.dtype)
        #print(labels.unique())

        instances_map = torch.zeros(instance_embeddings.shape[0], dtype=instance_embeddings.dtype)
        instances_map[mask_indices] = labels
        instances_map = instances_map.reshape(self.DEFAULT_IMAGE_SIZE[::-1])
        #print(instances_map.shape)

        return instances_map

    def _extract_lanes(self, instances_map, scale=False):
        lanes = []
        lane_indices = instances_map.unique()[1:]
        #print(lane_indices)
        for index in lane_indices:
            coords = (instances_map == index).nonzero(as_tuple=True)
            if scale:
                coords = [c / 4 for c in coords]
            coords = coords[::-1] # from (y, x) to (x, y)
            coords = torch.stack(coords).to(instances_map.dtype)
            lanes.append(coords)

        return lanes

    def _fit(self, lanes):
        coefs = []
        #print(len(lanes))
        for lane in lanes:
            x = lane[0, :].unsqueeze(dim=1)
            y = lane[1, :]
            Y = torch.stack((y, torch.ones(y.shape[0]))).T
            #print(x.shape, Y.shape)
            w = torch.linalg.inv(Y.T @ Y) @ Y.T @ x
            coefs.append(w)

        return coefs

    def _postprocess_result(self, lanes):
        processed = []
        for i, lane in enumerate(lanes):
            lane = lane.T
            lane[:, 2] = i
            ind1 = lane[:, 0] >= 0
            ind2 = lane[:, 0] <= 512
            index = torch.logical_and(ind1, ind2)
            lane = lane[index, :]
            processed.append(lane)

        return torch.cat(processed, dim=0)

    def _predict_lanes(self, coefs, y_positions):
        lanes = []

        for coef in coefs:
            c, d = coef
            lane = []
            for y in y_positions:
                x = c * y + d
                lane.append((x, y, 1))
            lanes.append(torch.tensor(lane).T)

        return lanes

    def _preprocess_image(self, image):
        image = cv2.resize(image, self.DEFAULT_IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image[..., None]
        image = torch.from_numpy(image).float().permute((2, 0, 1)).unsqueeze(dim=0)

        return image

    def _project_lanes(self, lanes):
        projected = []
        for lane in lanes:
            ones = torch.ones((1, lane.shape[1]))
            P = torch.cat((lane, ones), dim=0)
            #print(P)
            P_projected = self._default_homography @ P

            P_projected = P_projected / P_projected[2, :]
            #print(P_projected)
            projected.append(P_projected)

        return projected

    def _project_y(self, y_positions):
        y_positions = torch.from_numpy(y_positions).to(torch.float32)
        Y = torch.stack((
            torch.zeros(y_positions.shape[0]),
            y_positions,
            torch.ones(y_positions.shape[0])
        ))
        Y_projected = self._default_homography @ Y
        Y_projected = Y_projected / Y_projected[2, :]
        y_positions_projected = Y_projected[1, :]

        return y_positions_projected

    def _reproject(self, lanes):
        reprojected = []
        for lane in lanes:
            lane_reprojected = torch.linalg.inv(self._default_homography) @ lane
            lane_reprojected = lane_reprojected / lane_reprojected[2, ]
            reprojected.append(lane_reprojected)

        return reprojected

my_model = torch.load('model.pt', map_location=torch.device('cpu'))
my_model.eval()

detector = LaneDetector(enet=my_model)
print("done")

# image = cv2.imread(r"D:\Lane Detection\lane_detection-master\lane_detection-master\img\test4.jpg")
image = cv2.resize(cv2.imread(r"D:\Lane Detection\lane_detection-master\lane_detection-master\img\pic0.jpg"), (1080,720))

res = detector(image)

if res is None: 
    print("no line")
else:
    new = cv2.resize(res[0], [1080, 720])
    # new = cv2.cvtColor(new, cv2.COLOR_GRAY2RGB)

    # line_image = hough_lines(new, rho, theta, threshold, min_line_length, max_line_gap)
    # result = cv2.addWeighted(image, 0.8, new, 1, 0, dtype=cv2.CV_32F).astype(np.uint8)
    cv2.imshow("image", image)

    cv2.waitKey(0)
 
cv2.destroyAllWindows()
# plt.imshow(cv2.resize(image, (512, 256)))
# plt.imshow(res[0], alpha=0.5)

# plt.scatter(x=res[1][:,0], y=res[1][:, 1], c=res[1][:, 2])