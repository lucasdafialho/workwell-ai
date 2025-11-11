"""Detector simplificado de fadiga com suporte opcional a visão computacional."""

from __future__ import annotations

import logging
from collections import deque
from typing import Dict, List, Optional

import numpy as np

try:
    import cv2
    import mediapipe as mp

    _VISION_AVAILABLE = True
except ImportError:
    cv2 = None  # type: ignore
    mp = None  # type: ignore
    _VISION_AVAILABLE = False

logger = logging.getLogger(__name__)


class FatigueDetector:
    """Detector de fadiga usando análise facial."""
    
    def __init__(self):
        self.enabled = _VISION_AVAILABLE
        if not self.enabled:
            logger.warning("OpenCV/MediaPipe não disponíveis. Detector usará fallback heurístico.")
            self.mp_face_mesh = None
            self.mp_drawing = None
            self.face_mesh = None
        else:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.mp_drawing = mp.solutions.drawing_utils
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        
        # Pontos de referência para olhos (MediaPipe)
        self.LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Histórico para análise temporal
        self.blink_history = deque(maxlen=30)
        self.head_pose_history = deque(maxlen=30)
        self.eye_closure_history = deque(maxlen=30)
        
        # Modelo CNN para classificação
        self.fatigue_model = None
        
    def calculate_eye_aspect_ratio(self, landmarks, eye_indices: List[int]) -> float:
        """Calcula Eye Aspect Ratio (EAR) para detectar piscadas."""
        eye_points = np.array([[landmarks[i].x, landmarks[i].y] for i in eye_indices])
        
        # Distâncias verticais
        vertical_1 = np.linalg.norm(eye_points[1] - eye_points[5])
        vertical_2 = np.linalg.norm(eye_points[2] - eye_points[4])
        
        # Distância horizontal
        horizontal = np.linalg.norm(eye_points[0] - eye_points[3])
        
        # EAR
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear
    
    def calculate_head_pose(self, landmarks) -> Dict[str, float]:
        """Calcula pose da cabeça (inclinação, rotação)."""
        # Pontos de referência para pose
        nose_tip = landmarks[1]
        chin = landmarks[175]
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        
        # Vetores
        eye_vector = np.array([right_eye.x - left_eye.x, right_eye.y - left_eye.y])
        face_vector = np.array([chin.x - nose_tip.x, chin.y - nose_tip.y])
        
        # Ângulos
        eye_angle = np.arctan2(eye_vector[1], eye_vector[0]) * 180 / np.pi
        face_angle = np.arctan2(face_vector[1], face_vector[0]) * 180 / np.pi
        
        return {
            'eye_angle': eye_angle,
            'face_angle': face_angle,
            'head_tilt': abs(eye_angle)
        }
    
    def detect_blinks(self, landmarks) -> Dict:
        """Detecta piscadas e calcula métricas."""
        left_ear = self.calculate_eye_aspect_ratio(landmarks, self.LEFT_EYE_INDICES)
        right_ear = self.calculate_eye_aspect_ratio(landmarks, self.RIGHT_EYE_INDICES)
        
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Threshold para piscada (ajustável)
        blink_threshold = 0.25
        is_blinking = avg_ear < blink_threshold
        
        self.blink_history.append(avg_ear)
        
        # Calcular frequência de piscadas
        if len(self.blink_history) >= 10:
            recent_blinks = [ear < blink_threshold for ear in list(self.blink_history)[-10:]]
            blink_rate = sum(recent_blinks) / len(recent_blinks)
        else:
            blink_rate = 0.0
        
        return {
            'ear': avg_ear,
            'is_blinking': is_blinking,
            'blink_rate': blink_rate,
            'left_ear': left_ear,
            'right_ear': right_ear
        }
    
    def analyze_facial_expressions(self, landmarks) -> Dict:
        """Analisa expressões faciais indicativas de fadiga."""
        # Pontos para boca (bocejo)
        mouth_points = [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
        mouth_landmarks = np.array([[landmarks[i].x, landmarks[i].y] for i in mouth_points])
        
        # Abertura da boca
        mouth_height = np.linalg.norm(mouth_landmarks[2] - mouth_landmarks[10])
        mouth_width = np.linalg.norm(mouth_landmarks[0] - mouth_landmarks[6])
        mouth_ratio = mouth_height / (mouth_width + 1e-6)
        
        # Detecção de bocejo
        yawn_threshold = 0.5
        is_yawning = mouth_ratio > yawn_threshold
        
        return {
            'mouth_ratio': mouth_ratio,
            'is_yawning': is_yawning
        }
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Processa um frame de vídeo e retorna métricas de fadiga.
        
        Args:
            frame: Frame BGR do OpenCV
            
        Returns:
            Dicionário com métricas de fadiga
        """
        if not self.enabled or self.face_mesh is None or cv2 is None:
            return {
                'face_detected': False,
                'fatigue_level': 'unknown'
            }

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return {
                'face_detected': False,
                'fatigue_level': 'unknown'
            }
        
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Análises
        blink_data = self.detect_blinks(landmarks)
        head_pose = self.calculate_head_pose(landmarks)
        expressions = self.analyze_facial_expressions(landmarks)
        
        # Calcular score de fadiga
        fatigue_score = self._calculate_fatigue_score(
            blink_data, head_pose, expressions
        )
        
        # Classificar nível de fadiga
        fatigue_level = self._classify_fatigue_level(fatigue_score)
        
        return {
            'face_detected': True,
            'fatigue_score': fatigue_score,
            'fatigue_level': fatigue_level,
            'blink_data': blink_data,
            'head_pose': head_pose,
            'expressions': expressions,
            'landmarks': landmarks
        }
    
    def _calculate_fatigue_score(
        self,
        blink_data: Dict,
        head_pose: Dict,
        expressions: Dict
    ) -> float:
        """Calcula score composto de fadiga (0-100)."""
        score = 0.0
        
        # Fator 1: Frequência de piscadas (baixa = fadiga)
        blink_rate = blink_data.get('blink_rate', 0.5)
        if blink_rate < 0.1:  # Muito poucas piscadas
            score += 30
        elif blink_rate < 0.2:
            score += 15
        
        # Fator 2: EAR médio (baixo = olhos fechados)
        avg_ear = blink_data.get('ear', 0.3)
        if avg_ear < 0.2:
            score += 25
        elif avg_ear < 0.25:
            score += 15
        
        # Fator 3: Inclinação da cabeça
        head_tilt = head_pose.get('head_tilt', 0)
        if head_tilt > 15:  # Cabeça muito inclinada
            score += 20
        
        # Fator 4: Bocejo
        if expressions.get('is_yawning', False):
            score += 25
        
        return min(score, 100.0)
    
    def _classify_fatigue_level(self, score: float) -> str:
        """Classifica nível de fadiga baseado no score."""
        if score < 25:
            return 'baixo'
        elif score < 50:
            return 'medio'
        elif score < 75:
            return 'alto'
        else:
            return 'critico'
    
    def process_video_stream(self, video_source: int = 0, callback=None):
        """
        Processa stream de vídeo em tempo real.
        
        Args:
            video_source: Índice da câmera ou caminho do vídeo
            callback: Função callback para processar resultados
        """
        if not self.enabled or cv2 is None:
            logger.warning("Processamento de vídeo indisponível neste ambiente.")
            return

        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            logger.error("Não foi possível abrir a câmera")
            return
        
        logger.info("Processando stream de vídeo...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Processar frame
            result = self.process_frame(frame)
            
            # Desenhar landmarks (opcional, para debug)
            if result.get('face_detected') and result.get('landmarks'):
                mp_drawing = mp.solutions.drawing_utils
                mp_drawing.draw_landmarks(
                    frame,
                    self.mp_face_mesh.FaceMesh.create_from_landmarks_list(
                        [result['landmarks']]
                    ),
                    self.mp_face_mesh.FACEMESH_CONTOURS,
                    None,
                    self.mp_drawing.DrawingSpec(
                        thickness=1, circle_radius=1, color=(0, 255, 0)
                    )
                )
            
            # Adicionar texto com nível de fadiga
            if result.get('face_detected'):
                fatigue_level = result.get('fatigue_level', 'unknown')
                fatigue_score = result.get('fatigue_score', 0)
                
                cv2.putText(
                    frame,
                    f"Fadiga: {fatigue_level} ({fatigue_score:.1f})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0) if fatigue_level == 'baixo' else (0, 165, 255) if fatigue_level == 'medio' else (0, 0, 255),
                    2
                )
            
            # Callback para processar resultado
            if callback:
                callback(result)
            
            # Mostrar frame (opcional)
            cv2.imshow('Fatigue Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Exemplo de uso
    detector = FatigueDetector()
    
    # Processar webcam
    def process_result(result):
        if result.get('face_detected'):
            print(f"Fadiga: {result['fatigue_level']} (Score: {result['fatigue_score']:.2f})")
    
    detector.process_video_stream(video_source=0, callback=process_result)

