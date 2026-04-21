import streamlit as st
from PIL import Image
import time

# cau hinh trang
st.set_page_config(
    page_title="Hệ thống Phát hiện Khẩu trang - CNN",
    layout="wide"
)

st.markdown("""
<style>
    video { transform: scaleX(1) !important; }
</style>
""", unsafe_allow_html=True)

st.title("HỆ THỐNG PHÁT HIỆN KHẨU TRANG KHUÔN MẶT")

# Khởi tạo session state
if 'current_frame' not in st.session_state:
    st.session_state.current_frame = None
if 'matrix_images' not in st.session_state:
    st.session_state.matrix_images = []

# sidebar settings
with st.sidebar:
    st.header("CÀI ĐẶT")
    st.toggle("Kích hoạt CNN", value=True, disabled=True)
    st.divider()

# cau hinh giao dien
st.divider()

col1, col2 = st.columns([1.2, 1], gap="large")

# WEBCAM
with col1:
    st.subheader("WEBCAM TRỰC TIẾP", divider="blue")
    
    cam_on = st.toggle("Bật Webcam", value=True)
    
    if cam_on:
        photo = st.camera_input("Chụp khung hình", key="cam")
        if photo:
            st.session_state.current_frame = Image.open(photo)
            st.success("Đã chụp!")
            
            # Thêm vào matrix
            st.session_state.matrix_images.append(st.session_state.current_frame)
            if len(st.session_state.matrix_images) > 6:
                st.session_state.matrix_images.pop(0)
        
        if st.session_state.current_frame:
            st.image(st.session_state.current_frame, caption="Khung hình từ Webcam", use_container_width=True)
            w, h = st.session_state.current_frame.size
            st.caption(f" {w}×{h} px | {time.strftime('%H:%M:%S')}")
        else:
            st.info("Bật webcam và chụp khung hình")
    else:
        st.warning("Webcam đã tắt")
        st.session_state.current_frame = None

# CNN (chỉ hiển thị)
with col2:
    st.subheader("KẾT QUẢ CNN", divider="green")
    st.success("ĐANG HOẠT ĐỘNG")
    
    if st.session_state.current_frame:
        st.image(st.session_state.current_frame, caption="Ảnh từ Webcam", use_container_width=True)
    else:
        st.info("Chụp từ webcam để xem ảnh tại đây")

# upload video
st.divider()
st.subheader("TẢI VIDEO TỪ MÁY LÊN ", divider="gray")

uploaded_video = st.file_uploader(
    "Chọn file video từ máy tính (mp4, avi, mov, mkv)",
    type=['mp4', 'avi', 'mov', 'mkv']
)

if uploaded_video:
    st.video(uploaded_video)
    st.success(" Video đã tải lên và sẵn sàng xem!")
else:
    st.info("Tải video từ máy lên để xem tại đây")

# matrix hình ảnh
st.divider()
st.subheader("MATRIX HÌNH ẢNH (Ảnh bạn đã chụp từ webcam)", divider="violet")

if st.button("Xóa toàn bộ Matrix"):
    st.session_state.matrix_images = []
    st.rerun()

if st.session_state.matrix_images:
    cols = st.columns(3)
    for i, img in enumerate(st.session_state.matrix_images):
        with cols[i % 3]:
            st.image(img, caption=f"Ảnh {i+1}", use_container_width=True)
else:
    st.info("Chụp từ webcam để ảnh tự động thêm vào matrix")
