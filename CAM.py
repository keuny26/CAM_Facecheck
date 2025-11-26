import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import cv2
import threading
import numpy as np
import time

class FaceReaderApp:
    def __init__(self, master):
        self.master = master
        master.title("🧐 AI 관상 프로그램 (CAM)")
        master.geometry("800x700") # 창 크기 조절
        master.resizable(False, False) # 창 크기 변경 불가

        self.cap = None  # 카메라 객체
        self.is_running = False # 카메라 스트리밍 상태
        self.current_frame = None # 현재 프레임 저장

        # OpenCV 얼굴 및 눈 감지기를 미리 로드 (haarcascade는 OpenCV 설치 시 함께 제공)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        # 코와 입은 별도의 haarcascade가 없거나 정확도가 낮아, 여기서는 얼굴 영역 내에서 추정합니다.
        
        # UI 요소 생성
        self.create_widgets()

    def create_widgets(self):
        # 타이틀
        tk.Label(self.master, text="웹캠으로 당신의 관상을 확인하세요!", font=('Helvetica', 16, 'bold')).pack(pady=10)

        # 비디오 피드를 표시할 Canvas
        self.video_canvas = tk.Canvas(self.master, width=640, height=480, bg="black")
        self.video_canvas.pack(pady=10)

        # 버튼 프레임
        button_frame = tk.Frame(self.master)
        button_frame.pack(pady=10)

        # 시작 버튼
        self.start_btn = tk.Button(button_frame, text="웹캠 시작", command=self.start_camera, 
                                   font=('Helvetica', 12), bg='green', fg='white')
        self.start_btn.pack(side=tk.LEFT, padx=10, ipadx=10, ipady=5)

        # 캡처 및 분석 버튼
        self.capture_btn = tk.Button(button_frame, text="관상 분석", command=self.capture_and_analyze, 
                                     font=('Helvetica', 12), bg='orange', fg='white', state=tk.DISABLED)
        self.capture_btn.pack(side=tk.LEFT, padx=10, ipadx=10, ipady=5)

        # 정지 버튼
        self.stop_btn = tk.Button(button_frame, text="웹캠 정지", command=self.stop_camera, 
                                  font=('Helvetica', 12), bg='red', fg='white', state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=10, ipadx=10, ipady=5)
        
        # 결과 표시 영역 (스크롤 가능)
        tk.Label(self.master, text="--- 관상 분석 결과 ---", font=('Helvetica', 12, 'underline')).pack(pady=5)
        self.result_text = tk.Text(self.master, height=5, width=70, font=('Helvetica', 10), wrap=tk.WORD)
        self.result_text.pack(pady=5, padx=10)
        self.result_text.config(state=tk.DISABLED) # 기본적으로 수정 불가

    def start_camera(self):
        """웹캠 스트리밍을 시작합니다."""
        if not self.is_running:
            self.cap = cv2.VideoCapture(0) # 0은 기본 웹캠
            if not self.cap.isOpened():
                messagebox.showerror("오류", "웹캠을 열 수 없습니다. 카메라가 연결되어 있는지 확인하세요.")
                return

            self.is_running = True
            self.start_btn.config(state=tk.DISABLED)
            self.capture_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.NORMAL)
            
            # 스레드를 사용하여 카메라 피드를 비동기적으로 업데이트
            self.video_thread = threading.Thread(target=self._update_video_feed, daemon=True)
            self.video_thread.start()

    def _update_video_feed(self):
        """웹캠에서 프레임을 읽어와 Canvas에 표시합니다."""
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            self.current_frame = frame # 현재 프레임 저장 (분석용)
            
            # OpenCV는 BGR 순서, Tkinter/PIL은 RGB 순서이므로 변환 필요
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(image=img)

            self.video_canvas.create_image(0, 0, image=img_tk, anchor=tk.NW)
            self.video_canvas.image = img_tk # Tkinter가 이미지 가비지 컬렉션하지 않도록 참조 유지
            
            time.sleep(0.01) # 짧은 딜레이로 CPU 과부하 방지

        if self.cap:
            self.cap.release()

    def stop_camera(self):
        """웹캠 스트리밍을 정지합니다."""
        if self.is_running:
            self.is_running = False
            # 스레드가 종료될 때까지 잠시 기다릴 수 있음 (선택 사항)
            # if self.video_thread.is_alive():
            #     self.video_thread.join() 
            
            if self.cap:
                self.cap.release()
                self.cap = None
            
            self.start_btn.config(state=tk.NORMAL)
            self.capture_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.DISABLED)
            self.video_canvas.delete("all") # 캔버스 지우기
            self.update_result_text("웹캠이 정지되었습니다.", clear_previous=True)


    def capture_and_analyze(self):
        """현재 프레임을 캡처하여 관상 분석을 수행합니다."""
        if self.current_frame is None:
            messagebox.showwarning("경고", "캡처할 이미지가 없습니다. 웹캠을 먼저 시작해주세요.")
            return

        captured_frame = self.current_frame.copy()
        
        # 얼굴 감지 (흑백 이미지로 변환하여 감지 정확도 높임)
        gray = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        analysis_result = "분석 중...\n"
        if len(faces) == 0:
            analysis_result += "얼굴을 찾을 수 없습니다. 다시 시도해 주세요.\n"
        else:
            for (x, y, w, h) in faces:
                # 얼굴 영역 표시 (파란색 사각형)
                cv2.rectangle(captured_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # 얼굴 영역에서 눈 감지 (관상 분석을 위한 특징점)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = captured_frame[y:y+h, x:x+w]
                
                eyes = self.eye_cascade.detectMultiScale(roi_gray)
                
                eye_count = 0
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2) # 눈 표시 (초록색)
                    eye_count += 1

                # 코와 입은 haarcascade로 정확히 찾기 어려움. 여기서는 얼굴 영역 내에서 추정
                # 간단한 추정: 얼굴 하단 중앙에 코, 그 아래에 입
                nose_x = x + int(w * 0.5)
                nose_y = y + int(h * 0.6)
                mouth_x = x + int(w * 0.5)
                mouth_y = y + int(h * 0.8)
                
                cv2.circle(captured_frame, (nose_x, nose_y), 5, (0, 255, 255), -1) # 코 위치 (노란색)
                cv2.circle(captured_frame, (mouth_x, mouth_y), 5, (0, 0, 255), -1) # 입 위치 (빨간색)

                # --- 아주 기본적인 관상 분석 로직 (예시) ---
                analysis_result += f"얼굴이 감지되었습니다! (감지된 눈 개수: {eye_count})\n"
                
                if w > 300: # 얼굴 크기가 큰 경우 (카메라와의 거리 또는 실제 얼굴 크기)
                    analysis_result += "- 넓고 큰 얼굴: 포용력과 지도력이 있을 수 있습니다.\n"
                else:
                    analysis_result += "- 갸름한 얼굴: 섬세하고 예술적인 기질이 있을 수 있습니다.\n"
                
                if eye_count >= 2: # 눈이 두 개 이상 감지되면
                    if eyes[0][2] > 40: # 첫 번째 눈의 너비가 40픽셀 이상이면 (눈이 큰 편)
                        analysis_result += "- 큰 눈: 감성적이고 호기심이 많습니다.\n"
                    else:
                        analysis_result += "- 작은 눈: 신중하고 통찰력이 있습니다.\n"
                else:
                    analysis_result += "- 눈 감지가 어렵거나 불분명합니다.\n"

                # 코 위치에 따른 간단한 분석
                if nose_y - y < h * 0.4: # 코가 얼굴의 위쪽에 위치 (추정)
                    analysis_result += "- 높은 코: 자존심이 강하고 이상을 추구합니다.\n"
                else:
                    analysis_result += "- 적당한 코: 현실적이고 균형 감각이 좋습니다.\n"
                
                # 입 위치에 따른 간단한 분석
                if h - mouth_y < h * 0.15: # 입이 턱에 가까운 경우 (추정)
                    analysis_result += "- 얇은 입술: 냉정하고 이성적인 편입니다.\n"
                else:
                    analysis_result += "- 두툼한 입술: 정이 많고 따뜻한 마음을 가졌습니다.\n"

        # 분석된 프레임을 새 창에 표시
        self.show_analysis_result(captured_frame, analysis_result)
        
        # 결과 텍스트 업데이트
        self.update_result_text(analysis_result, clear_previous=True)


    def show_analysis_result(self, frame, analysis_text):
        """캡처된 이미지와 분석 결과를 새 창에 표시합니다."""
        
        result_window = tk.Toplevel(self.master)
        result_window.title("🔮 관상 분석 결과")
        result_window.geometry("700x600")
        result_window.resizable(False, False)

        tk.Label(result_window, text="분석된 이미지", font=('Helvetica', 14, 'bold')).pack(pady=10)

        # 이미지 표시
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        
        # 이미지 크기 조절 (너무 크면 창에 안 들어갈 수 있으므로)
        img.thumbnail((400, 300)) 
        
        img_tk = ImageTk.PhotoImage(image=img)
        
        img_canvas = tk.Canvas(result_window, width=img.width, height=img.height, bg="black")
        img_canvas.pack(pady=5)
        img_canvas.create_image(0, 0, image=img_tk, anchor=tk.NW)
        img_canvas.image = img_tk # 참조 유지

        tk.Label(result_window, text="--- 상세 분석 ---", font=('Helvetica', 12, 'underline')).pack(pady=10)
        
        # 분석 텍스트 표시
        text_widget = tk.Text(result_window, height=10, width=60, font=('Helvetica', 10), wrap=tk.WORD)
        text_widget.insert(tk.END, analysis_text)
        text_widget.config(state=tk.DISABLED)
        text_widget.pack(pady=5, padx=10)

        tk.Label(result_window, text="* 이 분석은 오락용이며, 과학적 근거가 없습니다.", fg='gray', font=('Helvetica', 8)).pack(pady=5)

    def update_result_text(self, text, clear_previous=False):
        """메인 창의 결과 텍스트 영역을 업데이트합니다."""
        self.result_text.config(state=tk.NORMAL) # 수정 가능 상태로 변경
        if clear_previous:
            self.result_text.delete(1.0, tk.END) # 이전 내용 모두 삭제
        self.result_text.insert(tk.END, text + "\n")
        self.result_text.config(state=tk.DISABLED) # 다시 수정 불가 상태로 변경
        self.result_text.see(tk.END) # 스크롤을 가장 아래로 이동

    def on_closing(self):
        """Tkinter 창이 닫힐 때 호출될 함수."""
        if messagebox.askokcancel("종료", "프로그램을 종료하시겠습니까?"):
            self.is_running = False # 카메라 스레드 종료 신호
            if self.cap:
                self.cap.release() # 카메라 자원 해제
            self.master.destroy()

# 메인 실행
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceReaderApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing) # 창 닫기 버튼에 종료 함수 연결
    root.mainloop()