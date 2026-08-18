import cv2
import numpy as np
from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.kmeans = None

    def get_clustering_model(self, image):
        # Reshape image to 2D array of pixels
        image_2d = image.reshape(-1, 3)

        # Fit KMeans with 2 clusters (foreground vs background / dominant colors)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(image_2d)
        return kmeans

    def get_player_color(self, frame, bbox):
        # Extract player image crop using bounding box
        x1, y1, x2, y2 = map(int, bbox)
        player_crop = frame[y1:y2, x1:x2]

        # Focus on upper half (jersey area)
        height = player_crop.shape[0]
        top_half = player_crop[0:int(height / 2), :]

        # Cluster colors in upper body
        kmeans = self.get_clustering_model(top_half)
        labels = kmeans.labels_

        # Reshape labels back to crop shape
        clustered_image = labels.reshape(top_half.shape[0], top_half.shape[1])

        # Get corner clusters to exclude grass/background
        corner_clusters = [
            clustered_image[0, 0],
            clustered_image[0, -1],
            clustered_image[-1, 0],
            clustered_image[-1, -1]
        ]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]
        return player_color

    def assign_team_color(self, frame, player_detections):
        player_colors = []
        for bbox in player_detections.xyxy:
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, bbox):
        player_color = self.get_player_color(frame, bbox)
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        return team_id + 1