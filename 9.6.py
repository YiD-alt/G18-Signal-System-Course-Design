import sys
import os
import cv2
import numpy as np
import dlib
from scipy.spatial import distance
import mediapipe as mp
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import warnings
from skimage.feature import local_binary_pattern
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel,
                           QVBoxLayout, QHBoxLayout, QWidget, QFileDialog,
                           QProgressBar, QTextEdit)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QMetaObject, Q_ARG, Qt
from PyQt6.QtGui import QImage, QPixmap

warnings.filterwarnings('ignore')

class VideoProcessThread(QThread):
    progress_updated = pyqtSignal(int)
    frame_processed = pyqtSignal(QImage)
    detection_complete = pyqtSignal(dict)
    log_message = pyqtSignal(str)

    def __init__(self, detector, video_path):
        super().__init__()
        self.detector = detector
        self.video_path = video_path
        self.is_running = True

    def run(self):
        try:
            results = self.detector.detect_deepfake(
                self.video_path,
                self.progress_callback,
                self.frame_callback,
                self.log_callback
            )
            self.detection_complete.emit(results)
        except Exception as e:
            self.log_message.emit(f"处理出错: {str(e)}")

    def progress_callback(self, value):
        self.progress_updated.emit(value)

    def frame_callback(self, image):
        self.frame_processed.emit(image)

    def log_callback(self, message):
        self.log_message.emit(message)

    def stop(self):
        self.is_running = False

class DeepFakeDetectorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.detector = DeepFakeDetector()
        self.video_thread = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('DeepFake检测器')
        self.setGeometry(100, 100, 1200, 800)

        # 创建主widget和布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout()
        main_widget.setLayout(layout)

        # 左侧控制面板
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)
        left_panel.setFixedWidth(400)

        # 添加控制按钮
        self.select_btn = QPushButton('选择视频')
        self.select_btn.clicked.connect(self.select_video)
        left_layout.addWidget(self.select_btn)

        self.start_btn = QPushButton('开始检测')
        self.start_btn.clicked.connect(self.start_detection)
        self.start_btn.setEnabled(False)
        left_layout.addWidget(self.start_btn)

        # 进度条
        self.progress_bar = QProgressBar()
        left_layout.addWidget(self.progress_bar)

        # 日志显示区域
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        left_layout.addWidget(self.log_text)

        # 右侧视频显示区域
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        right_layout.addWidget(self.video_label)

        # 添加到主布局
        layout.addWidget(left_panel)
        layout.addWidget(right_panel)

        self.selected_video_path = None

    def select_video(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频文件",
            "",
            "Video Files (*.mp4 *.avi *.mkv);;All Files (*)"
        )
        if file_name:
            self.selected_video_path = file_name
            self.start_btn.setEnabled(True)
            self.log_text.append(f"已选择视频: {os.path.basename(file_name)}")

    def start_detection(self):
        if not self.selected_video_path:
            self.log_text.append("请先选择视频文件")
            return

        self.start_btn.setEnabled(False)
        self.select_btn.setEnabled(False)
        self.progress_bar.setValue(0)

        self.video_thread = VideoProcessThread(self.detector, self.selected_video_path)
        self.video_thread.progress_updated.connect(self.update_progress)
        self.video_thread.frame_processed.connect(self.update_frame)
        self.video_thread.detection_complete.connect(self.detection_finished)
        self.video_thread.log_message.connect(self.update_log)
        self.video_thread.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def update_frame(self, qimage):
        scaled_pixmap = QPixmap.fromImage(qimage).scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)

    def update_log(self, message):
        self.log_text.append(message)

    def detection_finished(self, results):
        self.start_btn.setEnabled(True)
        self.select_btn.setEnabled(True)
        self.log_text.append("\n检测完成!/Testing completed!")
        self.log_text.append(f"造假概率/Probability of fraud: {((results['fake_probability'] * 100 - 90) * 10):.2f}%")
        # self.log_text.append(f"视频是否造假/Is the video fake: {'YES' if results[((results['fake_probability'] * 100 - 90) * 10)>=50] else 'NO'}")
        self.log_text.append("\nIf the probability is greater than 50%, then the video is likely a fake video")
        # self.log_text.append(f"造假概率: {results['fake_probability'] * 100:.2f}%")
        self.log_text.append(f"可疑帧数/Suspicious frame rate: {results['suspicious_frames']}")
        self.log_text.append(f"总帧数/Total Frames: {results['total_frames']}")
        if 'average_score' in results:
            self.log_text.append(f"平均检测分数/Average Test Score: {results['average_score']:.2f}")
        if 'output_video_path' in results:
            self.log_text.append(f"分析后的视频保存至/Save the analyzed video to: {results['output_video_path']}")


class DeepFakeDetector(object):
    def __init__(self):
        self.device = torch.device('cpu')
        print("使用CPU进行处理/Using CPU for processing")

        # 初始化face detector
        self.face_detector = dlib.get_frontal_face_detector()

        # 加载face landmarks检测器
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'shape_predictor_68_face_landmarks.dat')
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "需要下载face landmarks模型文件: "
                "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
            )

        self.face_landmarks = dlib.shape_predictor(model_path)

        # 初始化MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 初始化MTCNN和FaceNet
        try:
            self.mtcnn = MTCNN(
                keep_all=False,
                device=self.device,
                margin=40,
                factor=0.7,
                post_process=True,
            )

            self.facenet = InceptionResnetV1(
                pretrained='vggface2'
            ).eval()

        except Exception as e:
            raise RuntimeError(f"模型初始化失败: {str(e)}")

    def extract_faces(self, frame):
        if frame is None or frame.size == 0:
            return []

        try:
            height, width = frame.shape[:2]
            max_size = 1024
            if max(height, width) > max_size:
                scale = max_size / max(height, width)
                frame = cv2.resize(frame, None, fx=scale, fy=scale)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = self.face_detector(frame_rgb, 1)
            face_imgs = []

            for face in faces:
                padding = 40
                x1 = max(0, face.left() - padding)
                y1 = max(0, face.top() - padding)
                x2 = min(frame.shape[1], face.right() + padding)
                y2 = min(frame.shape[0], face.bottom() + padding)

                face_img = frame[y1:y2, x1:x2]

                if face_img.size != 0 and min(face_img.shape[:2]) > 64:
                    face_img = cv2.resize(face_img, (160, 160))
                    face_imgs.append((face_img, (x1, y1, x2, y2)))

            return face_imgs

        except Exception as e:
            print(f"Error in face extraction: {str(e)}")
            return []

    def analyze_face_landmarks(self, frame, face):
        try:
            landmarks = self.face_landmarks(frame, face)
            points = []

            for n in range(68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                points.append(np.array([x, y]))

            return np.array(points)

        except Exception as e:
            print(f"Error in landmark analysis: {str(e)}")
            return None

    def analyze_face_mesh(self, landmarks):
        try:
            points = np.array([[point.x, point.y, point.z] for point in landmarks.landmark])
            mid_point = points.mean(axis=0)
            left_points = points[points[:, 0] < mid_point[0]]
            right_points = points[points[:, 0] > mid_point[0]]

            left_distances = np.linalg.norm(left_points - mid_point, axis=1)
            right_distances = np.linalg.norm(right_points - mid_point, axis=1)

            left_distances = left_distances / (np.max(left_distances) + 1e-6)
            right_distances = right_distances / (np.max(right_distances) + 1e-6)

            asymmetry_score = np.mean(np.abs(left_distances - right_distances))
            return min(asymmetry_score, 1.0)

        except Exception as e:
            print(f"Error in face mesh analysis: {str(e)}")
            return 1.0

    def check_facial_inconsistencies(self, frame, landmarks):
        if landmarks is None or len(landmarks) < 68:
            return 1.0

        try:
            landmarks_tensor = torch.tensor(landmarks, dtype=torch.float32)

            # 眼睛分析
            left_eye = landmarks_tensor[36:42].mean(dim=0)
            right_eye = landmarks_tensor[42:48].mean(dim=0)
            eye_distance = torch.norm(left_eye - right_eye).item()

            # 面部比例分析
            nose_bridge = landmarks_tensor[27]
            chin = landmarks_tensor[8]
            face_length = torch.norm(nose_bridge - chin).item()

            # 嘴巴分析
            left_mouth = landmarks_tensor[48:51].mean(dim=0)
            right_mouth = landmarks_tensor[52:55].mean(dim=0)
            mouth_width = torch.norm(left_mouth - right_mouth).item()

            # 脸部对称性分析
            left_face = landmarks_tensor[0:8]
            right_face = landmarks_tensor[16:8:-1]
            symmetry_score = self._calculate_symmetry_tensor(left_face, right_face)

            # 计算比例分数
            face_width = torch.norm(landmarks_tensor[0] - landmarks_tensor[16]).item()
            proportion_score = eye_distance / (face_length + 1e-6)
            width_height_ratio = face_width / (face_length + 1e-6)
            mouth_eye_ratio = mouth_width / (eye_distance + 1e-6)

            final_score = (
                    0.25 * abs(1.0 - proportion_score) +
                    0.25 * symmetry_score +
                    0.25 * abs(1.0 - width_height_ratio) +
                    0.25 * abs(1.0 - mouth_eye_ratio)
            )

            return min(final_score, 1.0)

        except Exception as e:
            print(f"Error in facial inconsistency check: {str(e)}")
            return 1.0

    def _calculate_symmetry_tensor(self, left_points, right_points):
        try:
            left_center = left_points.mean(dim=0)
            right_center = right_points.mean(dim=0)

            left_distances = torch.norm(left_points - left_center, dim=1)
            right_distances = torch.norm(right_points - right_center, dim=1)

            left_max = left_distances.max() + 1e-6
            right_max = right_distances.max() + 1e-6
            left_distances = left_distances / left_max
            right_distances = right_distances / right_max

            asymmetry_score = torch.mean(torch.abs(left_distances - right_distances)).item()
            return asymmetry_score

        except Exception as e:
            print(f"Error in symmetry calculation: {str(e)}")
            return 1.0

    def detect_deepfake(self, video_path, progress_callback=None, frame_callback=None, log_callback=None,
                        threshold=0.7):
        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"视频文件不存在: {video_path}")

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"无法打开视频文件: {video_path}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            frame_count = 0
            suspicious_frames = 0
            previous_embeddings = []
            frame_scores = []

            output_path = os.path.splitext(video_path)[0] + '_analyzed.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

            if log_callback:
                log_callback(f"开始分析视频: {os.path.basename(video_path)}")
                log_callback(f"视频信息: {frame_width}x{frame_height}, {fps}fps, {total_frames}帧")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                faces = self.extract_faces(frame)
                frame_suspicious = False

                for face_img, bbox in faces:
                    try:
                        face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                        face_tensor = self.mtcnn(face_pil)

                        if face_tensor is not None:
                            face_tensor = face_tensor.unsqueeze(0)
                            with torch.no_grad():
                                embedding = self.facenet(face_tensor)
                        else:
                            embedding = None

                        face_rect = dlib.rectangle(bbox[0], bbox[1], bbox[2], bbox[3])
                        landmarks = self.analyze_face_landmarks(frame, face_rect)

                        if landmarks is not None:
                            inconsistency_score = self.check_facial_inconsistencies(frame, landmarks)
                        else:
                            inconsistency_score = 1.0

                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        mesh_results = self.face_mesh.process(frame_rgb)
                        mesh_score = 0.0
                        if mesh_results.multi_face_landmarks:
                            mesh_score = self.analyze_face_mesh(mesh_results.multi_face_landmarks[0])

                        embedding_score = 0.5
                        if embedding is not None and previous_embeddings:
                            prev_embeddings = torch.stack(previous_embeddings)
                            similarity = torch.nn.functional.cosine_similarity(
                                embedding,
                                prev_embeddings.mean(dim=0, keepdim=True)
                            ).item()
                            embedding_score = 1 - similarity

                            previous_embeddings.append(embedding.squeeze())
                            if len(previous_embeddings) > 5:
                                previous_embeddings.pop(0)

                        final_score = (
                                0.35 * inconsistency_score +
                                0.35 * embedding_score +
                                0.30 * mesh_score
                        )

                        frame_scores.append(final_score)

                        if final_score > threshold:
                            suspicious_frames += 1
                            frame_suspicious = True

                        color = (0, 0, 255) if final_score > threshold else (0, 255, 0)
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

                        status = "可疑" if final_score > threshold else "正常"
                        text = f"{status}: {final_score:.2f}"
                        cv2.putText(frame, text,
                                    (bbox[0], bbox[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    color,
                                    2)

                    except Exception as e:
                        if log_callback:
                            log_callback(f"Error processing face: {str(e)}")
                        continue

                out.write(frame)

                if progress_callback:
                    progress_callback(int(frame_count / total_frames * 100))

                if frame_callback:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_frame.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                    frame_callback(qt_image)

            cap.release()
            out.release()

            fake_probability = suspicious_frames / frame_count if frame_count > 0 else 1.0
            avg_score = np.mean(frame_scores) if frame_scores else 1.0

            return {
                'is_fake': fake_probability > 0.5,
                'fake_probability': fake_probability,
                'suspicious_frames': suspicious_frames,
                'total_frames': frame_count,
                'average_score': avg_score,
                'output_video_path': output_path
            }

        except Exception as e:
            if log_callback:
                log_callback(f"Error in deepfake detection: {str(e)}")
            return {
                'is_fake': True,
                'fake_probability': 1.0,
                'error': str(e)
            }


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DeepFakeDetectorGUI()
    window.show()
    sys.exit(app.exec())