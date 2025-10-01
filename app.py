import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps, ImageFilter
import matplotlib.pyplot as plt
from datetime import datetime
import hashlib
from io import BytesIO

# Load the trained CNN model
MODEL_PATH = 'fashion_cnn_trained.h5'
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.info("💡 Please run the training notebook first to generate the model file.")
    st.stop()

# Class names for the Fashion MNIST dataset (10 classes)
class_names = ['Áo thun', 'Quần dài', 'Áo len', 'Váy', 'Áo khoác',
               'Guốc', 'Áo sơ mi', 'Giày sneaker', 'Túi', 'Ủng']

# ============================
# FMNIST Preprocess Utilities
# ============================
def fmnist_preprocess(image, out_size=28, obj_max=22, inv=True, normalize='unit'):
    """
    Tiền xử lý 1 ảnh để mô phỏng phân phối Fashion-MNIST.
    - image: PIL Image object (RGB hoặc L)
    - out_size: kích thước khung cuối (28)
    - obj_max: kích thước tối đa đặt vật thể vào (để lại viền giống MNIST)
    - inv: True -> vật thể sáng trên nền đen
    - normalize: 'unit' -> /255
    Trả về: mảng numpy float32 shape (28,28)
    """
    # Convert to grayscale if needed
    if image.mode != 'L':
        img = image.convert("L")
    else:
        img = image
    
    arr = np.array(img)

    # Tách nền trắng: chọn vùng tối hơn nền
    mask = arr < 240
    if mask.sum() == 0:
        mask = arr < 250

    # Nếu không tìm thấy vật thể (toàn bộ là nền trắng)
    if mask.sum() == 0:
        # Sử dụng toàn bộ ảnh
        cropped = img
    else:
        ys, xs = np.where(mask)
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        cropped = Image.fromarray(arr[y0:y1, x0:x1])

    # Làm mượt nhẹ để giảm răng cưa khi resize
    cropped = cropped.filter(ImageFilter.GaussianBlur(radius=0.5))

    # Invert cho giống FMNIST (foreground sáng, background tối)
    if inv:
        cropped = ImageOps.invert(cropped)

    # Resize giữ tỉ lệ và pad vào khung 28x28
    cropped.thumbnail((obj_max, obj_max), Image.Resampling.LANCZOS)
    w, h = cropped.size
    canvas = Image.new("L", (out_size, out_size), color=0)  # nền đen
    canvas.paste(cropped, ((out_size - w) // 2, (out_size - h) // 2))

    # Tăng tương phản nhẹ
    canvas = ImageOps.autocontrast(canvas, cutoff=1)

    x = np.array(canvas).astype("float32")
    if normalize == 'unit':
        x = x / 255.0
    return x  # (28,28)

# Function to preprocess the uploaded image
def preprocess_image(image, use_advanced=True):
    """
    Tiền xử lý ảnh upload với 2 phương pháp:
    - Simple: resize đơn giản
    - Advanced: sử dụng fmnist_preprocess để xử lý ảnh nền trắng tốt hơn
    """
    if use_advanced:
        # Sử dụng advanced preprocessing
        processed_array = fmnist_preprocess(image)
        processed_image = Image.fromarray((processed_array * 255).astype(np.uint8))
        image_array = processed_array.reshape(1, 28, 28, 1)
    else:
        # Simple preprocessing (phương pháp cũ)
        image = image.resize((28, 28))  # Resize to 28x28
        image = image.convert('L')  # Convert to grayscale
        image_array = np.array(image) / 255.0  # Normalize pixel values
        image_array = image_array.reshape(1, 28, 28, 1)  # Add batch and channel dimensions
        processed_image = image
    
    return image_array, processed_image

# Streamlit app
st.title("🔍 Ứng dụng Nhận dạng Thời trang")
st.write("Tải lên một hình ảnh để kiểm tra dự đoán của mô hình.")

# Preprocessing method selection
st.sidebar.title("⚙️ Tùy chọn Tiền xử lý")
use_advanced = st.sidebar.checkbox("Sử dụng Tiền xử lý Nâng cao", value=True, 
                                   help="Tiền xử lý nâng cao hoạt động tốt hơn với ảnh có nền trắng")

# File uploader for image input
uploaded_file = st.file_uploader("Chọn một tệp hình ảnh", type=["png", "jpg", "jpeg"])

# Khởi tạo session state cho lịch sử dự đoán
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []  # list các dict
if 'processed_hashes' not in st.session_state:
    st.session_state.processed_hashes = set()
if 'label_buffer' not in st.session_state:
    st.session_state.label_buffer = None  # tạm lưu nhãn chọn trước khi lưu

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 🖼️ Hình ảnh Gốc:")
        st.image(image, caption="Hình ảnh được tải lên", use_container_width=True)

    # Preprocess the image
    preprocessed_image, processed_image = preprocess_image(image, use_advanced)

    # Display the preprocessed image
    with col2:
        st.write("### ⚡ Hình ảnh Đã xử lý:")
        method_text = "Nâng cao" if use_advanced else "Đơn giản"
        st.image(processed_image, caption=f"Hình ảnh đã xử lý - {method_text} (28x28)", use_container_width=True)

    # Make prediction
    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    # Display the prediction
    st.write("### 🎯 Dự đoán:")
    st.write(f"Mô hình dự đoán đây là: **{class_names[predicted_class]}**")
    st.write(f"Độ tin cậy: **{confidence:.2%}**")

    # Thêm vào lịch sử (tránh thêm lặp khi Streamlit rerun)
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()
    if file_hash not in st.session_state.processed_hashes:
        st.session_state.processed_hashes.add(file_hash)
        st.session_state.prediction_history.append({
            'file_hash': file_hash,
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'predicted_index': int(predicted_class),
            'predicted_name': class_names[predicted_class],
            'confidence': float(confidence),
            'image_bytes': file_bytes,  # lưu ảnh gốc
            'true_index': None,
            'true_name': None,
            'is_correct': None,
            'prob_chart_bytes': None  # sẽ lưu ảnh biểu đồ xác suất
        })

    # Display the prediction probabilities as a bar chart
    st.write("### 📊 Xác suất Dự đoán:")
    fig, ax = plt.subplots(figsize=(12, 6))  # Tăng kích thước để chứa 10 classes
    bars = ax.bar(range(len(class_names)), prediction[0], 
                  color=['green' if i == predicted_class else 'skyblue' for i in range(len(class_names))])
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylabel("Xác suất")
    ax.set_title("Xác suất các Lớp")
    ax.set_ylim(0, 1)
    
    # Add percentage labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.1%}', ha='center', va='bottom')
    
    plt.tight_layout()
    st.pyplot(fig)
    # Lưu biểu đồ xác suất vào lịch sử (PNG bytes)
    try:
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        for h in st.session_state.prediction_history:
            if h['file_hash'] == file_hash:
                h['prob_chart_bytes'] = buf.getvalue()
                break
    except Exception as e:
        st.warning(f"Không thể lưu biểu đồ xác suất: {e}")
    
    # Show preprocessing comparison if advanced is used
    if use_advanced:
        st.write("### 🔄 So sánh Tiền xử lý:")
        st.write("**Tiền xử lý nâng cao** giúp với:")
        st.write("- ✅ Hình ảnh có nền trắng")
        st.write("- ✂️ Tự động cắt đối tượng")
        st.write("- 🔄 Đảo ngược màu sắc để phù hợp với phong cách Fashion-MNIST (nền tối, đối tượng sáng)")
        st.write("- ⚡ Cải thiện độ tương phản và căn giữa tốt hơn")

  
    # ===== Biểu đồ/Tổng hợp Độ chính xác =====
    st.write("---")
    st.subheader("📈 Biểu đồ Tỷ lệ Chính xác")
    labeled = [h for h in st.session_state.prediction_history if h['is_correct'] is not None and h['true_index'] is not None]
    if labeled:
        total = len(labeled)
        correct = sum(1 for h in labeled if h['is_correct'])
        accuracy = correct / total if total else 0
        colm1, colm2, colm3 = st.columns(3)
        with colm1:
            st.metric("Số ảnh đã gán nhãn", total)
        with colm2:
            st.metric("Đúng", correct)
        with colm3:
            st.metric("Độ chính xác", f"{accuracy:.2%}")

        # Biểu đồ đường tích lũy accuracy
        acc_values = []
        correct_so_far = 0
        for i, h in enumerate(labeled, start=1):
            if h['is_correct']:
                correct_so_far += 1
            acc_values.append(correct_so_far / i)
        fig_acc, ax_acc = plt.subplots(figsize=(6,3))
        ax_acc.plot(range(1, len(acc_values)+1), acc_values, marker='o', color='green')
        ax_acc.set_xlabel('Số mẫu đã gán nhãn')
        ax_acc.set_ylabel('Độ chính xác tích lũy')
        ax_acc.set_ylim(0,1)
        ax_acc.grid(alpha=0.3)
        st.pyplot(fig_acc)

        # Bar chart đúng vs sai
        fig_cs, ax_cs = plt.subplots(figsize=(4,3))
        ax_cs.bar(['Đúng','Sai'], [correct, total-correct], color=['green','red'])
        ax_cs.set_ylabel('Số lượng')
        st.pyplot(fig_cs)
    else:
        st.info("Chưa có ảnh nào được xác nhận nhãn để tính độ chính xác. Hãy chọn nhãn thực tế phía trên.")

    # Nếu chưa có accuracy thì hiển thị phân bố dự đoán (tỷ lệ dự đoán theo lớp)
    if not labeled and st.session_state.prediction_history:
        st.write("### 📊 Phân bố Lớp đã Dự đoán (theo lịch sử)")
        counts = {name:0 for name in class_names}
        for h in st.session_state.prediction_history:
            counts[h['predicted_name']] += 1
        fig_dist, ax_dist = plt.subplots(figsize=(8,3))
        ax_dist.bar(counts.keys(), counts.values(), color='skyblue')
        ax_dist.set_ylabel('Số lần dự đoán')
        ax_dist.set_xticklabels(counts.keys(), rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig_dist)

    # ===== Lịch sử dự đoán =====
    st.write("---")
    st.subheader("🧾 Lịch sử Dự đoán")
    if st.session_state.prediction_history:
        for idx, item in enumerate(reversed(st.session_state.prediction_history), start=1):
            colh1, colh2, colh3, colh4 = st.columns([1,2,2,2])
            with colh1:
                st.image(item['image_bytes'], caption=item['timestamp'], use_container_width=True)
            with colh2:
                st.write(f"**Dự đoán:** {item['predicted_name']}")
                st.write(f"Độ tin cậy: {item['confidence']:.1%}")
            with colh3:
                if item['true_name']:
                    st.write(f"**Thực tế:** {item['true_name']}")
                else:
                    st.write("Thực tế: _Chưa gán_")
            with colh4:
                if item['is_correct'] is True:
                    st.success("Đúng")
                elif item['is_correct'] is False:
                    st.error("Sai")
                else:
                    st.write("-")
            # Expander hiển thị biểu đồ xác suất đã lưu
            with st.expander(f"Biểu đồ xác suất #{idx} - {item['predicted_name']}"):
                if item.get('prob_chart_bytes'):
                    st.image(item['prob_chart_bytes'], use_container_width=True)
                    st.download_button(
                        "⬇️ Tải biểu đồ",
                        data=item['prob_chart_bytes'],
                        file_name=f"prob_chart_{item['timestamp'].replace(':','')}.png",
                        mime="image/png",
                        key=f"dl_{item['file_hash']}"
                    )
                else:
                    st.write("(Chưa lưu biểu đồ cho mục này)")
    else:
        st.write("Chưa có dự đoán nào.")



