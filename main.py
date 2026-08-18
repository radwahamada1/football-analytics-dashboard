import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from utils.color_assigner import TeamAssigner
from utils.ball_assigner import BallAssigner
from utils.ball_tracker import BallTracker
from utils.view_transformer import ViewTransformer
from utils.speed_and_distance_estimator import SpeedAndDistanceEstimator
from utils.tactical_dashboard import TacticalDashboard

def main():
    model = YOLO("yolov8n.pt")
    tracker = sv.ByteTrack()
    team_assigner = TeamAssigner()
    ball_assigner = BallAssigner()
    ball_tracker_util = BallTracker()
    view_transformer = ViewTransformer()
    speed_estimator = SpeedAndDistanceEstimator()
    
    sidebar_w = 260
    dashboard_builder = TacticalDashboard(sidebar_width=sidebar_w)

    video_path = "data/video/sample.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    frames = []
    detections_list = []
    ball_positions_raw = []

    print("1/3: Reading frames & Object Detection...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[(detections.class_id == 0) | (detections.class_id == 32)]
        detections_list.append(detections)

        ball_dets = detections[detections.class_id == 32]
        ball_positions_raw.append({'bbox': ball_dets.xyxy[0]} if len(ball_dets) > 0 else {})

    cap.release()

    print("2/3: Tracking & Interpolating Ball Positions...")
    interpolated_ball_positions = ball_tracker_util.interpolate_ball_positions(ball_positions_raw)

    COLOR_TEAM_1 = (220, 180, 50)
    COLOR_TEAM_2 = (50, 50, 220)
    COLOR_BALL = (0, 255, 255)

    team_assigned = False
    possession_team_1 = 0
    possession_team_2 = 0
    tracks = {"players": []}

    output_dir = "output_videos"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "tactical_dashboard_final.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (orig_w + sidebar_w, orig_h))
    if not out_writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(output_path, fourcc, fps, (orig_w + sidebar_w, orig_h))

    print("3/3: Rendering HD Video Output...")
    window_name = "Football Analytics - Tactical Dashboard"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, orig_w + sidebar_w, orig_h)

    for frame_idx, frame in enumerate(frames):
        detections = detections_list[frame_idx]
        player_detections = detections[detections.class_id == 0]

        detections = tracker.update_with_detections(detections)
        player_detections = tracker.update_with_detections(player_detections)

        if not team_assigned and len(player_detections) > 0:
            team_assigner.assign_team_color(frame, player_detections)
            team_assigned = True

        frame_player_tracks = {}
        players_dict = {}
        if player_detections.tracker_id is not None:
            for bbox, tracker_id in zip(player_detections.xyxy, player_detections.tracker_id):
                players_dict[tracker_id] = {'bbox': bbox}
                x1, y1, x2, y2 = bbox
                position_transformed = view_transformer.transform_point([(x1 + x2) / 2, y2])
                frame_player_tracks[tracker_id] = {'bbox': bbox, 'position_transformed': position_transformed}

        tracks["players"].append(frame_player_tracks)

        ball_bbox = interpolated_ball_positions[frame_idx].get('bbox')
        assigned_player_id = ball_assigner.assign_ball_to_player(players_dict, ball_bbox)

        # حساب الاستحواذ الإحصائي الفعلي
        if assigned_player_id != -1 and assigned_player_id in players_dict:
            player_bbox = players_dict[assigned_player_id]['bbox']
            current_possessor_team = team_assigner.get_player_team(frame, player_bbox)
            if current_possessor_team == 1:
                possession_team_1 += 1
            elif current_possessor_team == 2:
                possession_team_2 += 1

        total_possession_frames = possession_team_1 + possession_team_2
        if total_possession_frames > 0:
            pct_team_1 = (possession_team_1 / total_possession_frames) * 100.0
            pct_team_2 = (possession_team_2 / total_possession_frames) * 100.0
        else:
            pct_team_1, pct_team_2 = 50.0, 50.0

        speed_estimator.add_speed_and_distance_to_tracks(tracks)
        current_tracks = tracks["players"][frame_idx]

        annotated_frame = frame.copy()

        # رسم الشارات والبيانات فوق كل لاعب
        if detections.tracker_id is not None:
            for bbox, class_id, tracker_id in zip(detections.xyxy, detections.class_id, detections.tracker_id):
                x1, y1, x2, y2 = map(int, bbox)
                
                if class_id == 0 and (y2 - y1) > 26:
                    team_id = team_assigner.get_player_team(frame, bbox)
                    bg_color = COLOR_TEAM_1 if team_id == 1 else COLOR_TEAM_2
                    x_center = int((x1 + x2) / 2)

                    player_info = current_tracks.get(tracker_id, {})
                    speed = player_info.get('speed', 0)
                    speed_val = f"{speed:.1f}k/h" if speed is not None else "0.0k/h"

                    badge_text = f"#{tracker_id} {speed_val}"
                    font_scale = 0.38
                    (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

                    bx1, by1 = x_center - int(tw / 2) - 4, y2 + 2
                    bx2, by2 = x_center + int(tw / 2) + 4, y2 + th + 6

                    cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), (12, 12, 12), -1)
                    cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), bg_color, 1, cv2.LINE_AA)
                    cv2.putText(annotated_frame, badge_text, (bx1 + 4, by1 + th + 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

                    if tracker_id == assigned_player_id:
                        triangle_pts = np.array([[x_center, y1 - 4], [x_center - 5, y1 - 12], [x_center + 5, y1 - 12]], np.int32)
                        cv2.drawContours(annotated_frame, [triangle_pts], 0, (0, 255, 0), -1, cv2.LINE_AA)

        if ball_bbox is not None and not np.isnan(ball_bbox[0]):
            bx1, by1, bx2, by2 = map(int, ball_bbox)
            ball_center_x, ball_top_y = int((bx1 + bx2) / 2), by1
            ball_triangle = np.array([[ball_center_x, ball_top_y - 3], [ball_center_x - 5, ball_top_y - 11], [ball_center_x + 5, ball_top_y - 11]], np.int32)
            cv2.drawContours(annotated_frame, [ball_triangle], 0, COLOR_BALL, -1, cv2.LINE_AA)

        # تم تمرير team_assigner هنا لرسم النقاط بالوان الفريقين في الـ Dashboard
        dashboard_frame = dashboard_builder.draw_dashboard(
            annotated_frame, 
            ball_bbox, 
            players_dict, 
            pct_team_1, 
            pct_team_2, 
            team_assigner=team_assigner
        )

        out_writer.write(dashboard_frame)
        cv2.imshow(window_name, dashboard_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out_writer.release()
    cv2.destroyAllWindows()
    print(f"DONE! Video saved at: {output_path}")

if __name__ == "__main__":
    main()