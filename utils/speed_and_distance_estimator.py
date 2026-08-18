import cv2
import numpy as np

class SpeedAndDistanceEstimator:
    def __init__(self):
        # Frame rate of the video (approx 30 fps)
        self.fps = 30
        self.frame_window = 5

    def add_speed_and_distance_to_tracks(self, tracks):
        total_distance = {}

        for object, object_tracks in tracks.items():
            if object == "ball":
                continue
            
            number_of_frames = len(object_tracks)
            for frame_num in range(0, number_of_frames, self.frame_window):
                last_frame = min(frame_num + self.frame_window, number_of_frames - 1)

                for track_id, track_info in object_tracks[frame_num].items():
                    if track_id not in object_tracks[last_frame]:
                        continue

                    start_position = track_info.get('position_transformed')
                    end_position = object_tracks[last_frame][track_id].get('position_transformed')

                    if start_position is None or end_position is None:
                        continue

                    # Calculate distance in meters
                    distance_covered = np.linalg.norm(np.array(start_position) - np.array(end_position))
                    time_elapsed = (last_frame - frame_num) / self.fps

                    if time_elapsed == 0:
                        continue

                    # Calculate speed in km/h
                    speed_meters_per_second = distance_covered / time_elapsed
                    speed_km_per_hour = speed_meters_per_second * 3.6

                    if object not in total_distance:
                        total_distance[object] = {}
                    if track_id not in total_distance[object]:
                        total_distance[object][track_id] = 0

                    total_distance[object][track_id] += distance_covered

                    # Assign speed and distance to frame tracks
                    for frame_num_batch in range(frame_num, last_frame + 1):
                        if track_id in object_tracks[frame_num_batch]:
                            object_tracks[frame_num_batch][track_id]['speed'] = speed_km_per_hour
                            object_tracks[frame_num_batch][track_id]['distance'] = total_distance[object][track_id]