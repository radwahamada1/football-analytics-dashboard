import pandas as pd

class BallTracker:
    def __init__(self):
        pass

    def interpolate_ball_positions(self, ball_positions):
        # ball_positions is a list of dicts: [{'bbox': [x1, y1, x2, y2]} or {}]
        
        # Convert list of bboxes into DataFrame
        ball_positions_list = []
        for pos in ball_positions:
            if pos and 'bbox' in pos:
                ball_positions_list.append(pos['bbox'])
            else:
                ball_positions_list.append([None, None, None, None])

        df_ball_positions = pd.DataFrame(ball_positions_list, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolate missing values (Linear Interpolation)
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()  # Handle leading NaNs

        # Reconstruct updated ball positions list
        interpolated_positions = []
        for _, row in df_ball_positions.iterrows():
            interpolated_positions.append({'bbox': [row['x1'], row['y1'], row['x2'], row['y2']]})

        return interpolated_positions