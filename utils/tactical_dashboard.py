import cv2
import numpy as np

class TacticalDashboard:
    def __init__(self, sidebar_width=260):
        self.sidebar_width = sidebar_width
        self.ball_trail_points = []

    def draw_dashboard(self, frame, ball_pos, players_dict, pos_team1_pct, pos_team2_pct, team_assigner=None):
        h, w, _ = frame.shape

        sidebar = np.zeros((h, self.sidebar_width, 3), dtype=np.uint8)
        
        possession_h = 90
        remaining_h = h - possession_h
        panel_h = remaining_h // 2

        COLOR_TEAM_1 = (220, 180, 50)  # أزرق فاتح / أصفر حسب ألوان التيم الأول
        COLOR_TEAM_2 = (50, 50, 220)   # أحمر / داكن للتيم الثاني

        # -------------------------------------------------------------
        # 1. TOP PANEL: POSSESSION STATS
        # -------------------------------------------------------------
        cv2.rectangle(sidebar, (0, 0), (self.sidebar_width, possession_h), (18, 18, 18), -1)
        cv2.putText(sidebar, "POSSESSION", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

        t1_str = f"T1: {pos_team1_pct:.1f}%"
        t2_str = f"T2: {pos_team2_pct:.1f}%"
        cv2.putText(sidebar, t1_str, (12, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEAM_1, 1, cv2.LINE_AA)
        cv2.putText(sidebar, t2_str, (self.sidebar_width - 85, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEAM_2, 1, cv2.LINE_AA)

        bar_x1, bar_y1 = 12, 58
        bar_w, bar_h = self.sidebar_width - 24, 12
        cv2.rectangle(sidebar, (bar_x1, bar_y1), (bar_x1 + bar_w, bar_y1 + bar_h), (40, 40, 40), -1)

        if (pos_team1_pct + pos_team2_pct) > 0:
            t1_w = int(bar_w * (pos_team1_pct / 100.0))
            if t1_w > 0:
                cv2.rectangle(sidebar, (bar_x1, bar_y1), (bar_x1 + t1_w, bar_y1 + bar_h), COLOR_TEAM_1, -1)
            if bar_w - t1_w > 0:
                cv2.rectangle(sidebar, (bar_x1 + t1_w, bar_y1), (bar_x1 + bar_w, bar_y1 + bar_h), COLOR_TEAM_2, -1)

        # -------------------------------------------------------------
        # 2. MIDDLE PANEL: LIVE POSITIONS (Multi-Color Players)
        # -------------------------------------------------------------
        heat_y_start = possession_h
        cv2.rectangle(sidebar, (0, heat_y_start), (self.sidebar_width, heat_y_start + panel_h), (12, 15, 12), -1)
        cv2.putText(sidebar, "LIVE POSITIONS", (12, heat_y_start + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 120), 1, cv2.LINE_AA)

        pitch_x1, pitch_y1 = 12, heat_y_start + 30
        pitch_w, pitch_h = self.sidebar_width - 24, panel_h - 40
        cv2.rectangle(sidebar, (pitch_x1, pitch_y1), (pitch_x1 + pitch_w, pitch_y1 + pitch_h), (20, 70, 25), -1)
        cv2.rectangle(sidebar, (pitch_x1, pitch_y1), (pitch_x1 + pitch_w, pitch_y1 + pitch_h), (180, 180, 180), 1, cv2.LINE_AA)
        cv2.line(sidebar, (pitch_x1 + pitch_w // 2, pitch_y1), (pitch_x1 + pitch_w // 2, pitch_y1 + pitch_h), (180, 180, 180), 1, cv2.LINE_AA)

        # رسم كل لاعب بلون فريقه المخصص
        for p_id, p_data in players_dict.items():
            bbox = p_data['bbox']
            norm_x = int(pitch_x1 + (bbox[0] / w) * pitch_w)
            norm_y = int(pitch_y1 + (bbox[1] / h) * pitch_h)
            
            p_color = (0, 255, 255) # اللون الافتراضي (أصفر)
            if team_assigner is not None:
                team_id = team_assigner.get_player_team(frame, bbox)
                if team_id == 1:
                    p_color = COLOR_TEAM_1
                elif team_id == 2:
                    p_color = COLOR_TEAM_2

            cv2.circle(sidebar, (norm_x, norm_y), 3, p_color, -1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 3. BOTTOM PANEL: BALL TRAIL
        # -------------------------------------------------------------
        trail_y_start = possession_h + panel_h
        cv2.rectangle(sidebar, (0, trail_y_start), (self.sidebar_width, h), (10, 12, 15), -1)
        cv2.putText(sidebar, "BALL TRAIL", (12, trail_y_start + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

        trail_x1, trail_y1 = 12, trail_y_start + 30
        trail_w, trail_h = self.sidebar_width - 24, (h - trail_y_start) - 40
        cv2.rectangle(sidebar, (trail_x1, trail_y1), (trail_x1 + trail_w, trail_y1 + trail_h), (20, 22, 25), -1)

        if ball_pos is not None and not np.isnan(ball_pos[0]):
            bx = int((ball_pos[0] + ball_pos[2]) / 2)
            by = int((ball_pos[1] + ball_pos[3]) / 2)
            norm_bx = int(trail_x1 + (bx / w) * trail_w)
            norm_by = int(trail_y1 + (by / h) * trail_h)
            self.ball_trail_points.append((norm_bx, norm_by))

        for i in range(1, len(self.ball_trail_points)):
            cv2.line(sidebar, self.ball_trail_points[i-1], self.ball_trail_points[i], (0, 255, 255), 1, cv2.LINE_AA)

        return np.hstack((frame, sidebar))