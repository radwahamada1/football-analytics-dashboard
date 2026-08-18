import numpy as np

class BallAssigner:
    def __init__(self):
        # Maximum distance threshold (in pixels) to consider a player in possession
        self.max_player_ball_distance = 70

    def assign_ball_to_player(self, players, ball_bbox):
        if ball_bbox is None or len(ball_bbox) == 0:
            return -1

        # Calculate center of the ball
        ball_center = [(ball_bbox[0] + ball_bbox[2]) / 2, (ball_bbox[1] + ball_bbox[3]) / 2]

        minimum_distance = 99999
        assigned_player = -1

        for player_id, player_data in players.items():
            player_bbox = player_data['bbox']

            # Distance between ball and player's feet (bottom center of bbox)
            player_feet = [(player_bbox[0] + player_bbox[2]) / 2, player_bbox[3]]
            
            distance = np.linalg.norm(np.array(player_feet) - np.array(ball_center))

            if distance < self.max_player_ball_distance and distance < minimum_distance:
                minimum_distance = distance
                assigned_player = player_id

        return assigned_player