import cv2, threading, time, os, requests, datetime, io
import numpy as np
from pathlib import Path
import json

# Intentar cargar configuración desde config.py
try:
    import config as user_config
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False

class DetectorConfig:
    def __init__(self, rtsp_url, telegram_bot_token, telegram_chat_id,
                 notify_start_hour=0, notify_end_hour=24, min_confidence=0.45):
        self.rtsp_url = rtsp_url
        self.telegram_bot_token = telegram_bot_token
        self.telegram_chat_id = telegram_chat_id
        self.notify_start_hour = notify_start_hour
        self.notify_end_hour = notify_end_hour
        self.min_confidence = min_confidence

class VideoCamera:
    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg
        self.capture = None
        self.thread = None
        self.running = False
        self.lock = threading.Lock()
        self.last_notify_time = 0
        # Usar cooldown del config.py si existe
        self.notify_cooldown = user_config.COOLDOWN_SECONDS if HAS_CONFIG else 15
        self._last_frame = None
        self.consecutive_detections = 0
        self.required_consecutive = 2  # requiere 2 detecciones consecutivas para notificar
        
        # Cargar máscara de detección si existe
        self.detection_mask = None
        self.mask_points = None
        self._load_detection_mask()
        
        # Load YOLOv8 for detection
        try:
            from ultralytics import YOLO
            model_name = user_config.YOLO_MODEL if HAS_CONFIG else "yolov8n.pt"
            self.model = YOLO(model_name)
            self.use_yolo = True
        except Exception as e:
            print(f"[Detector] ERROR: Could not load YOLO: {e}")
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self.use_yolo = False
    
    def _load_detection_mask(self):
        """Load detection zone mask if exists"""
        mask_file = Path("detection_mask.json")
        if mask_file.exists():
            try:
                with open(mask_file, 'r') as f:
                    mask_data = json.load(f)
                    self.mask_points = mask_data.get('points', [])
                    if len(self.mask_points) < 3:
                        self.mask_points = None
            except Exception as e:
                self.mask_points = None
    
    def _create_mask(self, frame_shape):
        """Crea una máscara binaria basada en los puntos definidos"""
        if not self.mask_points or len(self.mask_points) < 3:
            return None
        
        h, w = frame_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array(self.mask_points, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        return mask
    
    def _is_in_detection_zone(self, bbox):
        """Verifica si el bounding box está dentro de la zona de detección"""
        if not self.mask_points or self.detection_mask is None:
            return True  # Si no hay máscara, todo es válido
        
        x1, y1, x2, y2 = map(int, bbox)
        # Calcular centro del bounding box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Verificar si el centro está dentro de la máscara
        if (0 <= center_y < self.detection_mask.shape[0] and 
            0 <= center_x < self.detection_mask.shape[1]):
            return self.detection_mask[center_y, center_x] > 0
        
        return False

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.capture:
            try:
                self.capture.release()
            except:
                pass
            self.capture = None

    def _open_capture(self):
        url = self.cfg.rtsp_url.strip()
        if url in ("", "0"):
            idx = 0
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            return None
            
        return cap

    def _run(self):
        reconnect_attempts = 0
        max_reconnect = 5
        
        while self.running:
            self.capture = self._open_capture()
            if not self.capture or not self.capture.isOpened():
                reconnect_attempts += 1
                if reconnect_attempts >= max_reconnect:
                    self.running = False
                    return
                time.sleep(5)
                continue
            
            reconnect_attempts = 0
            frame_count = 0
            
            while self.running:
                ret, frame = self.capture.read()
                if not ret or frame is None:
                    time.sleep(1)
                    break
                
                frame_count += 1
                # Procesar cada N frames según config
                frame_skip = user_config.FRAME_SKIP if HAS_CONFIG else 3
                if frame_count % frame_skip != 0:
                    continue
                
                # Frame original para mostrar (buena resolución)
                h0, w0 = frame.shape[:2]
                display_size = user_config.DISPLAY_SIZE if HAS_CONFIG else 640
                scale_display = display_size / max(w0, h0)
                display_w, display_h = int(w0*scale_display), int(h0*scale_display)
                frame_display = cv2.resize(frame, (display_w, display_h))
                
                # Crear máscara para este tamaño de frame si existe
                if self.mask_points and self.detection_mask is None:
                    self.detection_mask = self._create_mask(frame_display.shape)
                
                # Frame pequeño solo para detección (rápido)
                detect_size = user_config.DETECTION_SIZE if HAS_CONFIG else 320
                scale_detect = detect_size / max(w0, h0)
                detect_w, detect_h = int(w0*scale_detect), int(h0*scale_detect)
                frame_detect = cv2.resize(frame, (detect_w, detect_h))
                
                # Detectar en el frame pequeño
                detections = self.detect(frame_detect)
                
                # Escalar las coordenadas al tamaño de display y filtrar por zona
                scaled_detections = []
                for det in detections:
                    x1, y1, x2, y2 = det['xyxy']
                    # Escalar coordenadas
                    scale_ratio = scale_display / scale_detect
                    scaled_bbox = [x1*scale_ratio, y1*scale_ratio, x2*scale_ratio, y2*scale_ratio]
                    
                    # Verificar si está en la zona de detección
                    if self._is_in_detection_zone(scaled_bbox):
                        scaled_det = {
                            'xyxy': scaled_bbox,
                            'conf': det['conf']
                        }
                        scaled_detections.append(scaled_det)
                
                detections = scaled_detections
                
                # Dibujar detecciones en el frame
                frame_with_boxes = self._draw_detections(frame_display, detections)
                
                # Lógica de notificación mejorada
                if detections:
                    self.consecutive_detections += 1
                    
                    # Require multiple consecutive detections to avoid false positives
                    if (self.consecutive_detections >= self.required_consecutive and 
                        self._in_notify_window()):
                        now = time.time()
                        if now - self.last_notify_time > self.notify_cooldown:
                            self.last_notify_time = now
                            print(f"[Detector] 🚨 PERSON DETECTED!")
                            # Save and notify in separate thread to avoid freezing video
                            notify_thread = threading.Thread(
                                target=self._save_and_notify, 
                                args=(frame.copy(), detections),
                                daemon=True
                            )
                            notify_thread.start()
                else:
                    self.consecutive_detections = 0
                
                # Codificar a JPEG con calidad del config
                jpeg_quality = user_config.JPEG_QUALITY if HAS_CONFIG else 75
                ret2, jpeg = cv2.imencode('.jpg', frame_with_boxes, 
                                         [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
                if ret2:
                    with self.lock:
                        self._last_frame = jpeg.tobytes()
                
                # Sin sleep para máxima velocidad
            
            # Liberar captura antes de reintentar
            try:
                self.capture.release()
            except:
                pass

    def _draw_detections(self, frame, detections):
        """Dibuja cajas alrededor de las personas detectadas"""
        frame_copy = frame.copy()
        
        # Dibujar zona de detección si existe
        if self.mask_points and len(self.mask_points) >= 3:
            # Escalar puntos al tamaño actual del frame
            h, w = frame.shape[:2]
            overlay = frame_copy.copy()
            pts = np.array(self.mask_points, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (0, 255, 255))
            cv2.addWeighted(overlay, 0.2, frame_copy, 0.8, 0, frame_copy)
            
            # Draw zone border
            cv2.polylines(frame_copy, [pts], True, (0, 255, 255), 2)
            cv2.putText(frame_copy, "DETECTION ZONE", (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        for det in detections:
            x1, y1, x2, y2 = map(int, det['xyxy'])
            conf = det['conf']
            
            # Green for high confidence, yellow for medium
            color = (0, 255, 0) if conf > 0.6 else (0, 255, 255)
            thickness = 2
            
            # Draw rectangle
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, thickness)
            
            # Label with confidence
            label = f'Person {conf:.2f}'
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            y1_label = max(y1, label_size[1] + 10)
            
            # Text background
            cv2.rectangle(frame_copy, 
                         (x1, y1_label - label_size[1] - 10),
                         (x1 + label_size[0], y1_label + baseline - 10),
                         color, cv2.FILLED)
            
            # Text
            cv2.putText(frame_copy, label, (x1, y1_label - 7),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Add counter in corner
        if detections:
            count_text = f'Persons detected: {len(detections)}'
            cv2.rectangle(frame_copy, (5, 5), (300, 35), (0, 0, 0), cv2.FILLED)
            cv2.putText(frame_copy, count_text, (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame_copy
    
    def _in_notify_window(self):
        """Check if we are in notification time window"""
        # Get current time in configured UTC offset
        now_utc = datetime.datetime.utcnow()
        
        if HAS_CONFIG and hasattr(user_config, 'UTC_OFFSET'):
            offset_hours = user_config.UTC_OFFSET
            now = now_utc + datetime.timedelta(hours=offset_hours)
        else:
            # Fallback to local time
            now = datetime.datetime.now()
        
        start = int(self.cfg.notify_start_hour) % 24
        end = int(self.cfg.notify_end_hour) % 24
        h = now.hour
        if start <= end:
            return (h >= start and h < end)
        else:
            return (h >= start or h < end)

    def detect(self, frame):
        """Detect persons in frame. Returns list of detections with confidence."""
        if self.use_yolo:
            try:
                device = 'cuda' if (HAS_CONFIG and user_config.USE_GPU) else 'cpu'
                results = self.model(
                    frame, 
                    imgsz=416,
                    conf=self.cfg.min_confidence,
                    classes=[0],  # class 0 = person
                    verbose=False,
                    device=device,
                    half=False
                )
                
                persons = []
                for r in results:
                    boxes = r.boxes
                    if boxes is None or len(boxes) == 0:
                        continue
                    
                    for box in boxes:
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        
                        if cls == 0 and conf >= self.cfg.min_confidence:
                            xyxy = box.xyxy[0].cpu().numpy().tolist()
                            persons.append({
                                "conf": conf,
                                "xyxy": xyxy
                            })
                
                return persons
            except Exception as e:
                return []
        else:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rects, weights = self.hog.detectMultiScale(
                    gray, 
                    winStride=(4, 4),
                    padding=(8, 8),
                    scale=1.05
                )
                
                persons = []
                for (x, y, w, h), wt in zip(rects, weights):
                    if float(wt) >= self.cfg.min_confidence:
                        persons.append({
                            "conf": float(wt),
                            "xyxy": [x, y, x+w, y+h]
                        })
                return persons
            except Exception as e:
                return []

    def _save_and_notify(self, frame, detections=None):
        """Save frame and send Telegram notification"""
        temp_dir = Path("temp_captures")
        temp_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            if detections and len(detections) > 0:
                for idx, det in enumerate(detections):
                    x1, y1, x2, y2 = map(int, det['xyxy'])
                    conf = det['conf']
                    
                    # Add margin around bounding box (10% extra)
                    h, w = frame.shape[:2]
                    margin_x = int((x2 - x1) * 0.1)
                    margin_y = int((y2 - y1) * 0.1)
                    
                    x1 = max(0, x1 - margin_x)
                    y1 = max(0, y1 - margin_y)
                    x2 = min(w, x2 + margin_x)
                    y2 = min(h, y2 + margin_y)
                    
                    # Crop person
                    person_crop = frame[y1:y2, x1:x2]
                    
                    if person_crop.size > 0:
                        fname = temp_dir / f"person_{timestamp}_conf{int(conf*100)}.jpg"
                        cv2.imwrite(str(fname), person_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                        print(f"[Notify] 💾 Person saved: {fname}")
                        
                        # Send only first detected person
                        if idx == 0 and self.cfg.telegram_bot_token and self.cfg.telegram_chat_id:
                            self._send_telegram_photo(str(fname))
            else:
                fname = temp_dir / f"capture_{timestamp}.jpg"
                cv2.imwrite(str(fname), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                print(f"[Notify] 💾 Capture saved: {fname}")
                
                if self.cfg.telegram_bot_token and self.cfg.telegram_chat_id:
                    self._send_telegram_photo(str(fname))
            
            max_captures = user_config.MAX_CAPTURES if HAS_CONFIG else 50
            self._cleanup_old_captures(temp_dir, keep_last=max_captures)
            
        except Exception as e:
            pass
    
    def _cleanup_old_captures(self, directory, keep_last=50):
        """Delete old captures to save disk space"""
        try:
            files = sorted(directory.glob("*.jpg"), key=lambda x: x.stat().st_mtime)
            if len(files) > keep_last:
                for old_file in files[:-keep_last]:
                    old_file.unlink()
        except Exception as e:
            pass

    def _send_telegram_photo(self, image_path):
        """Send photo via Telegram with detailed information"""
        if not self.cfg.telegram_bot_token or not self.cfg.telegram_chat_id:
            return
            
        url = f"https://api.telegram.org/bot{self.cfg.telegram_bot_token}/sendPhoto"
        
        try:
            # Get current time in configured UTC offset
            now_utc = datetime.datetime.utcnow()
            
            if HAS_CONFIG and hasattr(user_config, 'UTC_OFFSET'):
                offset_hours = user_config.UTC_OFFSET
                now = now_utc + datetime.timedelta(hours=offset_hours)
            else:
                now = datetime.datetime.now()
            
            caption = (
                f"🚨 PERSON DETECTED\n"
                f"📅 Date: {now.strftime('%d/%m/%Y')}\n"
                f"🕐 Time: {now.strftime('%H:%M:%S')}"
            )
            
            print("[Telegram] 📤 Sending photo...")
            with open(image_path, 'rb') as photo:
                files = {'photo': photo}
                data = {
                    'chat_id': self.cfg.telegram_chat_id,
                    'caption': caption,
                    'parse_mode': 'HTML'
                }
                resp = requests.post(url, files=files, data=data, timeout=10)
                
                if resp.status_code == 200:
                    print("[Telegram] ✓ Photo sent successfully")
                else:
                    print(f"[Telegram] ⚠️ Error {resp.status_code}: {resp.text[:100]}")
                    
        except requests.exceptions.Timeout:
            pass
        except Exception as e:
            pass

    def get_frame(self):
        with self.lock:
            return getattr(self, "_last_frame", None)
