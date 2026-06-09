"""
gui_app.py — Giao diện chính cho Face Mask Detection & Tracking
Chạy: python gui_app.py
Yêu cầu: pip install torch torchvision opencv-python facenet-pytorch deep-sort-realtime Pillow
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import time
import os
import sys
import cv2

# ── Màu sắc & font ──────────────────────────────────────────────────────────
BG_DARK     = "#0D0F14"
BG_CARD     = "#161A22"
BG_PANEL    = "#1C2130"
ACCENT      = "#00E5FF"
ACCENT2     = "#FF4D6D"
ACCENT3     = "#39FF14"
TEXT_MAIN   = "#E8ECF4"
TEXT_DIM    = "#6B7A99"
BORDER      = "#252D3D"
BTN_HOVER   = "#00B8CC"

FONT_TITLE  = ("Courier New", 22, "bold")
FONT_SUB    = ("Courier New", 10)
FONT_BTN    = ("Courier New", 13, "bold")
FONT_LOG    = ("Courier New", 10)
FONT_LABEL  = ("Courier New", 11)
FONT_SMALL  = ("Courier New", 9)


class FaceMaskApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Face Mask Detection System")
        self.geometry("820x640")
        self.resizable(True, True)
        self.configure(bg=BG_DARK)
        self.minsize(700, 560)

        self._running = False
        self._thread  = None

        # Path vars
        _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.model_path = tk.StringVar(value=os.path.join(_root, "models", "weights(saugiam)", "best_model_adam.pth"))
        self.image_path  = tk.StringVar(value="")
        self.video_path  = tk.StringVar(value="")
        self.output_path = tk.StringVar(value="output.mp4")
        self.alert_sec   = tk.IntVar(value=10)
        self.cam_index   = tk.IntVar(value=0)
        self.active_mode = tk.StringVar(value="")  # image / video / webcam

        self._build_ui()

    # ────────────────────────────────────────────────────────────────────────
    #  BUILD UI
    # ────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Header ──────────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=BG_DARK, pady=0)
        hdr.pack(fill="x", padx=24, pady=(20, 4))

        tk.Label(hdr, text="[ FACE MASK DETECTION ]",
                 font=FONT_TITLE, fg=ACCENT, bg=BG_DARK).pack(side="left")

        # Model path mini bar
        mdl_row = tk.Frame(hdr, bg=BG_DARK)
        mdl_row.pack(side="right", padx=4)
        tk.Label(mdl_row, text="MODEL:", font=FONT_SMALL,
                 fg=TEXT_DIM, bg=BG_DARK).pack(side="left", padx=(0, 4))
        tk.Entry(mdl_row, textvariable=self.model_path,
                 font=FONT_SMALL, width=32,
                 bg=BG_PANEL, fg=TEXT_MAIN, insertbackground=ACCENT,
                 relief="flat", bd=4).pack(side="left")
        self._mkbtn(mdl_row, "...", lambda: self._browse_file(
            self.model_path, [("PyTorch model", "*.pth *.pt")]),
            small=True).pack(side="left", padx=(4, 0))

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=24, pady=(6, 14))

        # ── Mode selector ───────────────────────────────────────────────────
        mode_frame = tk.Frame(self, bg=BG_DARK)
        mode_frame.pack(fill="x", padx=24)
        tk.Label(mode_frame, text="CHỌN CHẾ ĐỘ:", font=FONT_SUB,
                 fg=TEXT_DIM, bg=BG_DARK).pack(anchor="w", pady=(0, 8))

        cards = tk.Frame(mode_frame, bg=BG_DARK)
        cards.pack(fill="x")
        cards.columnconfigure((0, 1, 2), weight=1, uniform="cols")

        self._btn_image  = self._mode_card(cards, 0, "🖼", "ẢNH", "Nhận diện từ file ảnh", "image")
        self._btn_video  = self._mode_card(cards, 1, "🎬", "VIDEO", "Tracking từ file video", "video")
        self._btn_webcam = self._mode_card(cards, 2, "📷", "WEBCAM", "Real-time qua camera", "webcam")

        # ── Config panel ────────────────────────────────────────────────────
        self._config_frame = tk.Frame(self, bg=BG_CARD,
                                      relief="flat", bd=0)
        self._config_frame.pack(fill="x", padx=24, pady=(14, 0))
        self._show_config_placeholder()

        # ── Action bar ──────────────────────────────────────────────────────
        action = tk.Frame(self, bg=BG_DARK)
        action.pack(fill="x", padx=24, pady=12)

        self._run_btn = self._mkbtn(action, "▶  CHẠY", self._on_run,
                                    color=ACCENT3, fg=BG_DARK, big=True)
        self._run_btn.pack(side="left")

        self._stop_btn = self._mkbtn(action, "■  DỪNG", self._on_stop,
                                     color=ACCENT2, fg="#fff", big=True)
        self._stop_btn.pack(side="left", padx=(10, 0))
        self._stop_btn.config(state="disabled")

        # Status dot
        self._status_var = tk.StringVar(value="● SẴNSÀNG")
        self._status_lbl = tk.Label(action, textvariable=self._status_var,
                                    font=FONT_SMALL, fg=ACCENT3, bg=BG_DARK)
        self._status_lbl.pack(side="right")

        # ── Log ─────────────────────────────────────────────────────────────
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=24)

        log_hdr = tk.Frame(self, bg=BG_DARK)
        log_hdr.pack(fill="x", padx=24, pady=(6, 2))
        tk.Label(log_hdr, text="OUTPUT LOG", font=FONT_SMALL,
                 fg=TEXT_DIM, bg=BG_DARK).pack(side="left")
        self._mkbtn(log_hdr, "XÓA", self._clear_log, small=True).pack(side="right")

        log_wrap = tk.Frame(self, bg=BG_PANEL, bd=0)
        log_wrap.pack(fill="both", expand=True, padx=24, pady=(0, 20))
        log_wrap.rowconfigure(0, weight=1)
        log_wrap.columnconfigure(0, weight=1)

        self._log = tk.Text(log_wrap, font=FONT_LOG, bg=BG_PANEL,
                            fg=TEXT_MAIN, insertbackground=ACCENT,
                            relief="flat", bd=0, padx=10, pady=8,
                            state="disabled", wrap="word")
        sb = tk.Scrollbar(log_wrap, command=self._log.yview,
                          bg=BG_PANEL, troughcolor=BG_PANEL,
                          activebackground=ACCENT, relief="flat", bd=0)
        self._log.config(yscrollcommand=sb.set)
        self._log.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")

        self._log_tag_config()
        self._write_log("Hệ thống khởi động. Chọn chế độ và nhấn CHẠY.\n", "info")

    # ────────────────────────────────────────────────────────────────────────
    #  WIDGETS HELPERS
    # ────────────────────────────────────────────────────────────────────────
    def _mkbtn(self, parent, text, cmd, color=ACCENT, fg=BG_DARK,
               big=False, small=False):
        size  = FONT_BTN if big else (FONT_SMALL if small else FONT_LABEL)
        padx_ = 18 if big else (6 if small else 12)
        pady_ = 10 if big else (3 if small else 6)
        btn = tk.Button(parent, text=text, command=cmd,
                        font=size, bg=color, fg=fg,
                        activebackground=BTN_HOVER, activeforeground=fg,
                        relief="flat", bd=0, cursor="hand2",
                        padx=padx_, pady=pady_)
        btn.bind("<Enter>", lambda e: btn.config(bg=BTN_HOVER))
        btn.bind("<Leave>", lambda e: btn.config(bg=color))
        return btn

    def _mode_card(self, parent, col, icon, title, sub, mode):
        frame = tk.Frame(parent, bg=BG_CARD, cursor="hand2",
                         padx=12, pady=14, relief="flat", bd=0)
        frame.grid(row=0, column=col, sticky="nsew",
                   padx=(0, 8) if col < 2 else 0)

        tk.Label(frame, text=icon, font=("Segoe UI Emoji", 26),
                 bg=BG_CARD, fg=TEXT_MAIN).pack()
        tk.Label(frame, text=title, font=FONT_BTN,
                 bg=BG_CARD, fg=ACCENT).pack(pady=(4, 0))
        tk.Label(frame, text=sub, font=FONT_SMALL,
                 bg=BG_CARD, fg=TEXT_DIM, wraplength=160).pack(pady=(2, 0))

        for w in [frame] + frame.winfo_children():
            w.bind("<Button-1>", lambda e, m=mode: self._select_mode(m))
            w.bind("<Enter>",    lambda e, f=frame: self._card_hover(f, True))
            w.bind("<Leave>",    lambda e, f=frame: self._card_hover(f, False))
        return frame

    def _card_hover(self, frame, hovered):
        color = BG_PANEL if hovered else BG_CARD
        frame.config(bg=color)
        for w in frame.winfo_children():
            try:
                w.config(bg=color)
            except Exception:
                pass

    def _card_activate(self, frame, active):
        border_color = ACCENT if active else BG_CARD
        frame.config(highlightbackground=border_color,
                     highlightthickness=2 if active else 0,
                     highlightcolor=border_color)

    # ────────────────────────────────────────────────────────────────────────
    #  CONFIG PANEL PER MODE
    # ────────────────────────────────────────────────────────────────────────
    def _clear_config(self):
        for w in self._config_frame.winfo_children():
            w.destroy()

    def _show_config_placeholder(self):
        self._clear_config()
        tk.Label(self._config_frame,
                 text="← Chọn chế độ ở trên để bắt đầu cấu hình",
                 font=FONT_SMALL, fg=TEXT_DIM, bg=BG_CARD,
                 pady=14).pack()

    def _row(self, label_text):
        """Tạo 1 hàng label+control trong config frame."""
        row = tk.Frame(self._config_frame, bg=BG_CARD)
        row.pack(fill="x", padx=14, pady=5)
        tk.Label(row, text=label_text, font=FONT_LABEL,
                 fg=TEXT_DIM, bg=BG_CARD, width=18,
                 anchor="w").pack(side="left")
        return row

    def _path_row(self, label, var, filetypes, save=False):
        row = self._row(label)
        tk.Entry(row, textvariable=var, font=FONT_SMALL,
                 bg=BG_PANEL, fg=TEXT_MAIN, insertbackground=ACCENT,
                 relief="flat", bd=4).pack(side="left", fill="x", expand=True)
        cmd = (lambda: self._save_file(var, filetypes)) if save else \
              (lambda: self._browse_file(var, filetypes))
        self._mkbtn(row, "Chọn..." if not save else "Lưu...",
                    cmd, small=True).pack(side="left", padx=(6, 0))

    def _show_config_image(self):
        self._clear_config()
        tk.Label(self._config_frame, text="CẤU HÌNH — ẢNH",
                 font=FONT_SUB, fg=ACCENT, bg=BG_CARD,
                 padx=14, pady=8).pack(anchor="w")
        self._path_row("File ảnh đầu vào:", self.image_path,
                       [("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")])
        self._alert_row()

    def _show_config_video(self):
        self._clear_config()
        tk.Label(self._config_frame, text="CẤU HÌNH — VIDEO",
                 font=FONT_SUB, fg=ACCENT, bg=BG_CARD,
                 padx=14, pady=8).pack(anchor="w")
        self._path_row("File video đầu vào:", self.video_path,
                       [("Video files", "*.mp4 *.avi *.mov *.mkv")])
        self._path_row("Lưu video output:", self.output_path,
                       [("MP4", "*.mp4")], save=True)
        self._alert_row()

    def _show_config_webcam(self):
        self._clear_config()
        tk.Label(self._config_frame, text="CẤU HÌNH — WEBCAM",
                 font=FONT_SUB, fg=ACCENT, bg=BG_CARD,
                 padx=14, pady=8).pack(anchor="w")
        row = self._row("Camera index:")
        tk.Spinbox(row, from_=0, to=10, width=6,
                   textvariable=self.cam_index,
                   font=FONT_LABEL, bg=BG_PANEL, fg=TEXT_MAIN,
                   buttonbackground=BG_PANEL, relief="flat",
                   insertbackground=ACCENT).pack(side="left")
        tk.Label(row, text="(0 = mặc định)", font=FONT_SMALL,
                 fg=TEXT_DIM, bg=BG_CARD).pack(side="left", padx=8)
        self._alert_row()

    def _alert_row(self):
        row = self._row("Cảnh báo sau (s):")
        sc = tk.Scale(row, from_=3, to=60, orient="horizontal",
                      variable=self.alert_sec,
                      bg=BG_CARD, fg=TEXT_MAIN, troughcolor=BG_PANEL,
                      highlightthickness=0, activebackground=ACCENT,
                      sliderrelief="flat", length=180)
        sc.pack(side="left")
        tk.Label(row, textvariable=self.alert_sec, font=FONT_LABEL,
                 fg=ACCENT, bg=BG_CARD, width=4).pack(side="left")
        tk.Label(row, text="giây", font=FONT_SMALL,
                 fg=TEXT_DIM, bg=BG_CARD).pack(side="left")

    # ────────────────────────────────────────────────────────────────────────
    #  MODE SELECTION
    # ────────────────────────────────────────────────────────────────────────
    def _select_mode(self, mode):
        self.active_mode.set(mode)
        # highlight active card
        card_map = {"image": self._btn_image,
                    "video": self._btn_video,
                    "webcam": self._btn_webcam}
        for m, card in card_map.items():
            self._card_activate(card, m == mode)

        if mode == "image":
            self._show_config_image()
        elif mode == "video":
            self._show_config_video()
        else:
            self._show_config_webcam()

    # ────────────────────────────────────────────────────────────────────────
    #  FILE DIALOGS
    # ────────────────────────────────────────────────────────────────────────
    def _browse_file(self, var, filetypes):
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            var.set(path)

    def _save_file(self, var, filetypes):
        path = filedialog.asksaveasfilename(filetypes=filetypes,
                                            defaultextension=".mp4")
        if path:
            var.set(path)

    # ────────────────────────────────────────────────────────────────────────
    #  LOG
    # ────────────────────────────────────────────────────────────────────────
    def _log_tag_config(self):
        self._log.tag_config("info",  foreground=TEXT_MAIN)
        self._log.tag_config("ok",    foreground=ACCENT3)
        self._log.tag_config("warn",  foreground="#FFD700")
        self._log.tag_config("error", foreground=ACCENT2)
        self._log.tag_config("dim",   foreground=TEXT_DIM)

    def _write_log(self, msg, tag="info"):
        self._log.config(state="normal")
        ts = time.strftime("%H:%M:%S")
        self._log.insert("end", f"[{ts}] {msg}\n", tag)
        self._log.see("end")
        self._log.config(state="disabled")

    def _clear_log(self):
        self._log.config(state="normal")
        self._log.delete("1.0", "end")
        self._log.config(state="disabled")

    # ────────────────────────────────────────────────────────────────────────
    #  REDIRECT STDOUT TO LOG
    # ────────────────────────────────────────────────────────────────────────
    def _redirect_stdout(self):
        app = self
        class LogWriter:
            def write(self, msg):
                if msg.strip():
                    tag = "warn" if "⚠" in msg or "CẢNH BÁO" in msg else \
                          "ok"   if "✅" in msg else \
                          "error" if "❌" in msg or "Lỗi" in msg.lower() else "info"
                    app.after(0, app._write_log, msg.strip(), tag)
            def flush(self): pass
        sys.stdout = LogWriter()

    def _restore_stdout(self):
        sys.stdout = sys.__stdout__

    # ────────────────────────────────────────────────────────────────────────
    #  RUN / STOP
    # ────────────────────────────────────────────────────────────────────────
    def _set_running(self, running: bool):
        self._running = running
        if running:
            self._run_btn.config(state="disabled")
            self._stop_btn.config(state="normal")
            self._status_var.set("● ĐANG CHẠY")
            self._status_lbl.config(fg="#FFD700")
        else:
            self._run_btn.config(state="normal")
            self._stop_btn.config(state="disabled")
            self._status_var.set("● SẴNSÀNG")
            self._status_lbl.config(fg=ACCENT3)

    def _on_run(self):
        mode = self.active_mode.get()
        if not mode:
            messagebox.showwarning("Chưa chọn chế độ",
                                   "Vui lòng chọn một chế độ: Ảnh / Video / Webcam.")
            return
        if not os.path.exists(self.model_path.get()):
            messagebox.showerror("Không tìm thấy model",
                                 f"Không tìm thấy file:\n{self.model_path.get()}\n\n"
                                 "Hãy chỉnh lại đường dẫn model ở trên.")
            return
        if mode == "image" and not self.image_path.get():
            messagebox.showwarning("Chưa chọn ảnh", "Vui lòng chọn file ảnh đầu vào.")
            return
        if mode == "video" and not self.video_path.get():
            messagebox.showwarning("Chưa chọn video", "Vui lòng chọn file video đầu vào.")
            return

        self._set_running(True)
        self._redirect_stdout()
        self._write_log(f"Bắt đầu chế độ: {mode.upper()}", "ok")

        self._thread = threading.Thread(target=self._run_task,
                                        args=(mode,), daemon=True)
        self._thread.start()

    def _on_stop(self):
        self._running = False
        self._write_log("Đã gửi tín hiệu dừng...", "warn")

    def _run_task(self, mode):
        try:
            if mode == "image":
                self._task_image()
            elif mode == "video":
                self._task_video()
            else:
                self._task_webcam()
        except Exception as e:
            self.after(0, self._write_log, f"❌ Lỗi: {e}", "error")
        finally:
            self._restore_stdout()
            self.after(0, self._set_running, False)

    # ────────────────────────────────────────────────────────────────────────
    #  TASKS (gọi code từ app.py của bạn)
    # ────────────────────────────────────────────────────────────────────────
    def _load_model_and_detector(self):
        """Import và load model + detector."""
        import sys, os

        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if root not in sys.path:
            sys.path.insert(0, root)
        try:
            from models.model    import load_model
            from src.detection.detectior import FaceDetector
        except ImportError as e:
                raise ImportError(f"LỖI THẬT SỰ: {e}") 
        mask_model, device = load_model(self.model_path.get())
        detector = FaceDetector(device)
        return mask_model, device, detector

    def _task_image(self):
        import cv2
        print("Đang load model...")
        mask_model, device, detector = self._load_model_and_detector()

        img_path = self.image_path.get()
        print(f"Đọc ảnh: {img_path}")
        frame = cv2.imread(img_path)
        if frame is None:
            print("❌ Không đọc được ảnh!")
            return

        dets = detector.detect_and_classify(frame, mask_model)
        print(f"✅ Phát hiện {len(dets)} khuôn mặt")

        from src.detection.detectior import CLASS_NAMES, COLORS
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            cls_id = d["cls_id"]
            conf   = d["conf"]
            color = COLORS[cls_id]

            label = (
                f"{CLASS_NAMES[cls_id]} "
                f"({conf*100:.1f}%)"
            )
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            print(f"  Face: {CLASS_NAMES[cls_id]}  conf={conf:.1%}  bbox=[{x1},{y1},{x2},{y2}]")

        cv2.imshow("Face Mask — Ảnh  (nhấn phím bất kỳ để đóng)", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("✅ Hoàn thành nhận diện ảnh.")

    def _task_video(self):
        import cv2, time as _time
        print("✅ Đang load model...")
        mask_model, device, detector = self._load_model_and_detector()

        from src.tracking.tracker  import PersonTracker
        from src.detection.detectior import CLASS_NAMES, COLORS

        tracker = PersonTracker(alert_seconds=self.alert_sec.get())
        vid_in  = self.video_path.get()
        vid_out = self.output_path.get()

        cap = cv2.VideoCapture(vid_in)
        if not cap.isOpened():
            print(f"❌ Không mở được video: {vid_in}"); return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"✅ Video: {w}x{h}  {fps:.1f}fps  {total_frames} frames")

        out = cv2.VideoWriter(vid_out,
                              cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        frame_n = 0
        while cap.isOpened() and self._running:
            ret, frame = cap.read()
            if not ret: break
            now      = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            dets     = detector.detect_and_classify(frame, mask_model)
            frame_rgb = frame[:, :, ::-1].copy()
            results  = tracker.update(dets, frame_rgb, now)

            for r in results:
                x1, y1, x2, y2 = r["bbox"]
                cls_id = r["cls_id"]; alert = r["alert"]
                color  = COLORS[cls_id]
                thick  = 4 if alert else 2
                label  = f"ID:{r['track_id']} {CLASS_NAMES[cls_id]} {r['duration']:.1f}s "
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
                cv2.putText(frame, label, (x1, y1-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                if alert:
                    cv2.putText(frame, "⚠ CANH BAO", (x1, y2+20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            out.write(frame)
            frame_n += 1
            if frame_n % 30 == 0:
                pct = frame_n / max(total_frames, 1) * 100
                print(f"  Tiến độ: {frame_n}/{total_frames} frame ({pct:.0f}%)")
            cv2.imshow("Face Mask — Video  (Q=thoát)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release(); out.release(); cv2.destroyAllWindows()
        print(f"Lưu video output: {vid_out}")

    # def _task_webcam(self):
    #     import cv2, time as _time
    #     print(" Đang load model...")
    #     mask_model, device, detector = self._load_model_and_detector()

    #     from src.tracking.tracker  import PersonTracker
    #     from src.detection.detectior import CLASS_NAMES, COLORS

    #     tracker = PersonTracker(alert_seconds=self.alert_sec.get())
    #     cam_idx = self.cam_index.get()
    #     cap = cv2.VideoCapture(cam_idx)
    #     if not cap.isOpened():
    #         print(f"Không mở được camera index={cam_idx}"); return

    #     print(f" Webcam {cam_idx} mở thành công. Nhấn Q trong cửa sổ video để thoát.")
    #     fps_arr = []
    #     while self._running:
    #         t0 = _time.time()
    #         ret, frame = cap.read()
    #         if not ret: break
    #         now       = _time.time()
    #         dets      = detector.detect_and_classify(frame, mask_model)
    #         frame_rgb = frame[:, :, ::-1].copy()
    #         results   = tracker.update(dets, frame_rgb, now)

    #         for r in results:
    #             x1, y1, x2, y2 = r["bbox"]
    #             cls_id = r["cls_id"]; alert = r["alert"]
    #             color  = COLORS[cls_id]
    #             thick  = 4 if alert else 2
    #             label  = f"ID:{r['track_id']} {CLASS_NAMES[cls_id]} {r['duration']:.1f}s"
    #             cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
    #             cv2.putText(frame, label, (x1, y1-8),
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    #             if alert:
    #                 cv2.putText(frame, "⚠ CANH BAO!", (x1, y2+20),
    #                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    #         # FPS overlay
    #         elapsed = _time.time() - t0
    #         fps_arr.append(1 / max(elapsed, 1e-6))
    #         if len(fps_arr) > 30: fps_arr.pop(0)
    #         fps_now = sum(fps_arr) / len(fps_arr)
    #         cv2.putText(frame, f"FPS:{fps_now:.1f}", (10, 24),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    #         cv2.imshow("Face Mask — Webcam  (Q=thoát)", frame)
    #         if cv2.waitKey(1) & 0xFF == ord("q"):
    #             break

    #     cap.release(); cv2.destroyAllWindows()
    #     print(" Đã tắt webcam.")
    def _task_webcam(self):
        import cv2
        import time as _time
        import os

        print("Đang load model...")
        mask_model, device, detector = self._load_model_and_detector()

        from src.tracking.tracker import PersonTracker
        from src.detection.detectior import CLASS_NAMES, COLORS

        tracker = PersonTracker(alert_seconds=self.alert_sec.get())

        # Thư mục lưu ảnh vi phạm
        snapshot_dir = "snapshots"
        os.makedirs(snapshot_dir, exist_ok=True)

        cam_idx = self.cam_index.get()
        cap = cv2.VideoCapture(cam_idx)

        if not cap.isOpened():
            print(f"❌ Không mở được camera index={cam_idx}")
            return

        print(
            f"✅ Webcam {cam_idx} mở thành công. "
            f"Nhấn Q trong cửa sổ video để thoát."
        )

        fps_arr = []

        while self._running:
            t0 = _time.time()

            ret, frame = cap.read()
            if not ret:
                break

            now = _time.time()

            dets = detector.detect_and_classify(frame, mask_model)

            frame_rgb = frame[:, :, ::-1].copy()

            results = tracker.update(
                dets,
                frame_rgb,
                now
            )

            for r in results:
                x1, y1, x2, y2 = r["bbox"]

                cls_id = r["cls_id"]
                alert  = r["alert"]

                color = COLORS[cls_id]
                thick = 4 if alert else 2

                label = (
                    f"ID:{r['track_id']} "
                    f"{CLASS_NAMES[cls_id]} "
                    f"{r['duration']:.1f}s"
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    thick
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

                if alert:
                    cv2.putText(
                        frame,
                        "⚠ CANH BAO!",
                        (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

                # ===== CHỤP ẢNH KHI VỪA VI PHẠM =====
                if r.get("captured", False):

                    crop = frame[
                        max(0, y1):min(frame.shape[0], y2),
                        max(0, x1):min(frame.shape[1], x2)
                    ]

                    timestamp = _time.strftime(
                        "%Y%m%d_%H%M%S"
                    )

                    filename = os.path.join(
                        snapshot_dir,
                        f"ID_{r['track_id']}_{timestamp}.jpg"
                    )

                    cv2.imwrite(filename, frame)

                    print(
                        f"📸 Đã lưu ảnh vi phạm: {filename}"
                    )

            # FPS
            elapsed = _time.time() - t0

            fps_arr.append(
                1 / max(elapsed, 1e-6)
            )

            if len(fps_arr) > 30:
                fps_arr.pop(0)

            fps_now = sum(fps_arr) / len(fps_arr)

            cv2.putText(
                frame,
                f"FPS:{fps_now:.1f}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.imshow(
                "Face Mask — Webcam (Q=thoat)",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        print(" Đã tắt webcam.")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = FaceMaskApp()
    app.mainloop()
