import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import plotly.graph_objects as go
import time
import os
from tensorflow.keras.applications.efficientnet import preprocess_input

# ─────────────────────────────────────────────
#  KONFIGURASI HALAMAN
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Klasifikasi Motif Batik",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  KELAS MOTIF (20 kelas — sesuaikan urutannya
#  dengan urutan training dataset kamu)
# ─────────────────────────────────────────────
CLASS_NAMES = [
    "Aceh_Pintu_Aceh",
    "Bali_Barong",
    "Bali_Merak",
    "DKI_Ondel_Ondel",
    "JawaBarat_Megamendung",
    "JawaTimur_Pring",
    "Kalimantan_Dayak",
    "Lampung_Gajah",
    "Madura_Mataketeran",
    "Maluku_Pala",
    "NTB_Lumbung",
    "Papua_Asmat",
    "Papua_Cendrawasih",
    "Papua_Tifa",
    "Solo_Parang",
    "SulawesiSelatan_Lontara",
    "SumateraBarat_Rumah_Minang",
    "SumateraUtara_Boraspati",
    "Yogyakarta_Kawung",
    "Yogyakarta_Parang"
]

# Deskripsi singkat tiap motif
CLASS_DESC = {
    "Aceh_Pintu_Aceh"           :  "Motif khas Aceh yang terinspirasi dari bentuk arsitektur Pintu Aceh, melambangkan identitas budaya dan kebesaran daerah.",
    "Bali_Barong"               :  "Motif Bali yang menggambarkan Barong sebagai simbol kebaikan dan pelindung dari kekuatan jahat.",
    "Bali_Merak"                :  "Motif burung merak khas Bali yang melambangkan keindahan, kemegahan, dan kebanggaan.",
    "DKI_Ondel_Ondel"           :  "Motif Betawi yang menampilkan ikon Ondel-Ondel sebagai simbol pelindung masyarakat dan budaya Jakarta.",
    "JawaBarat_Megamendung"     :  "Motif awan khas Cirebon yang melambangkan ketenangan, kesabaran, dan keluasan pikiran.",
    "JawaTimur_Pring"           :  "Motif bambu (pring) khas Jawa Timur yang melambangkan pertumbuhan, keteguhan, dan kehidupan.",
    "Kalimantan_Dayak"          :  "Motif etnik Dayak dengan ornamen khas yang mencerminkan kekuatan, keberanian, dan hubungan dengan alam.",
    "Lampung_Gajah"             :  "Motif gajah yang menjadi simbol kekuatan, kebijaksanaan, dan kekayaan budaya Lampung.",
    "Madura_Mataketeran"        :  "Motif khas Madura dengan pola tegas dan warna kontras yang melambangkan keberanian masyarakat Madura.",
    "Maluku_Pala"               :  "Motif pala yang menggambarkan komoditas rempah unggulan Maluku dan sejarah perdagangan nusantara.",
    "NTB_Lumbung"               :  "Motif lumbung khas Nusa Tenggara Barat yang melambangkan kemakmuran dan ketahanan pangan.",
    "Papua_Asmat"               :  "Motif ukiran Asmat yang mencerminkan seni tradisional, spiritualitas, dan identitas masyarakat Papua.",
    "Papua_Cendrawasih"         :  "Motif burung Cendrawasih sebagai simbol keindahan alam dan kekayaan hayati Papua.",
    "Papua_Tifa"                :  "Motif alat musik Tifa yang melambangkan kebersamaan, budaya, dan tradisi masyarakat Papua.",
    "Solo_Parang"               :  "Motif Parang khas Solo dengan pola diagonal yang melambangkan kekuatan, semangat, dan kesinambungan.",
    "SulawesiSelatan_Lontara"   :  "Motif yang terinspirasi dari aksara Lontara sebagai simbol ilmu pengetahuan dan warisan budaya Bugis-Makassar.",
    "SumateraBarat_Rumah_Minang":  "Motif Rumah Gadang yang melambangkan kearifan lokal, persatuan keluarga, dan adat Minangkabau.",
    "SumateraUtara_Boraspati"   :  "Motif Boraspati khas Batak yang melambangkan perlindungan, kesejahteraan, dan keberuntungan.",
    "Yogyakarta_Kawung"         :  "Motif Kawung berbentuk lingkaran simetris yang melambangkan kesucian, keadilan, dan pengendalian diri.",
    "Yogyakarta_Parang"         :  "Motif Parang khas Yogyakarta yang melambangkan kekuatan, keteguhan, dan perjuangan tanpa henti."

}

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background utama gradient */
.stApp {
    background: linear-gradient(
        135deg,
        #121358 0%,
        #232F72 30%,
        #2F578A 65%,
        #36ADA3 100%
    );
    color: #f0f4ff;
}

/* Header */
.main-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
}
.main-header h1 {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    color: #ffffff;
}
.main-header p {
    color: #cbd5ff;
    font-size: 1rem;
}

/* Upload zone */
.upload-zone {
    border: 2px dashed #36ADA3;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    background: rgba(255,255,255,1);
}

/* Result card */
.result-card {
    background: linear-gradient(
        135deg,
        rgba(18,19,88,0.9),
        rgba(35,47,114,0.9),
        rgba(47,87,138,0.9)
    );
    border: 1px solid rgba(54,173,163,0.3);
    border-radius: 20px;
    padding: 1.8rem 2rem;
    margin-top: 1rem;
}

/* Text hasil */
.result-class {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    color: #ffffff;
}
.result-confidence {
    font-size: 1rem;
    color: #cbd5ff;
}

/* Confidence bar */
.confidence-bar-bg {
    background: rgba(255,255,255,0.1);
    border-radius: 100px;
    height: 10px;
}
.confidence-bar-fill {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #36ADA3, #2F578A);
}

/* Deskripsi */
.desc-text {
    color: #dbe6ff;
    border-left: 3px solid #36ADA3;
    padding-left: 1rem;
}

/* Badge */
.badge {
    display: inline-block;
    background: rgba(54,173,163,0.15);
    color: #36ADA3;
    border: 1px solid rgba(54,173,163,0.4);
    border-radius: 100px;
    padding: 0.25rem 0.85rem;
    font-size: 0.75rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #121358,
        #232F72
    );
}
section[data-testid="stSidebar"] .stMarkdown {
    color: #cbd5ff;
}

/* Metric boxes */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-top: 1rem;
}
.metric-box {
    flex: 1;
    background: rgba(18,19,88,0.6);
    border: 1px solid rgba(54,173,163,0.3);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.metric-box .val {
    font-size: 1.6rem;
    color: #ffffff;
}
.metric-box .lbl {
    font-size: 0.75rem;
    color: #aab6ff;
}

/* Plotly transparan */
.js-plotly-plot .plotly {
    background: transparent !important;
}

/* Hide branding */
#MainMenu, footer { visibility: hidden; }
header { visibility: hidden; }

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  LOAD MODEL (cache agar tidak reload)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    model_path = "model_fixed.h5"
    if not os.path.exists(model_path):
        return None, "Model tidak ditemukan. Pastikan file ada di direktori yang sama."
    model = tf.keras.models.load_model(model_path, compile=False)
    return model, None

# ─────────────────────────────────────────────
#  FUNGSI PREPROCESSING & PREDIKSI
# ─────────────────────────────────────────────
IMG_SIZE = (224, 224)

def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)            # (1, 224, 224, 3)
    return arr

def predict(model, image: Image.Image):
    arr   = preprocess(image)
    preds = model.predict(arr, verbose=0)[0]     # (20,)
    top5  = np.argsort(preds)[::-1][:5]
    return preds, top5

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎨 Info Model")
    st.markdown("""
**Arsitektur**  
EfficientNetB0 + Transfer Learning

**Input**  
224 × 224 × 3 (RGB)

**Output**  
20 kelas motif batik

**Framework**  
TensorFlow / Keras 2.15
    """)
    st.divider()
    st.markdown("### 📋 20 Kelas Motif")
    for i, name in enumerate(CLASS_NAMES):
        st.markdown(f"`{i+1:02d}` {name}")
    st.divider()
    st.markdown(
        "<small style='color:#4a4030'>Batik adalah warisan budaya Indonesia yang diakui UNESCO (2009).</small>",
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎨 Klasifikasi Motif Batik</h1>
    <p>Upload gambar batik — AI akan mengenali motifnya menggunakan EfficientNetB0</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────
with st.spinner("Memuat model EfficientNetB0..."):
    model, err = load_model()

if err:
    st.error(err)
    st.stop()

# ─────────────────────────────────────────────
#  UPLOAD GAMBAR
# ─────────────────────────────────────────────
col_up, col_result = st.columns([1, 1.4], gap="large")

with col_up:
    st.markdown(
    "<h4 style='color:white;'>📤 Upload Gambar Batik</h4>",
    unsafe_allow_html=True
    )
    uploaded = st.file_uploader(
        "Pilih file gambar (JPG / PNG / WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption=uploaded.name)

        # Metadata gambar
        w, h = image.size
        st.markdown(f"""
<div class="metric-row">
  <div class="metric-box"><div class="val">{w}</div><div class="lbl">Lebar (px)</div></div>
  <div class="metric-box"><div class="val">{h}</div><div class="lbl">Tinggi (px)</div></div>
  <div class="metric-box"><div class="val">{uploaded.size // 1024}</div><div class="lbl">Ukuran (KB)</div></div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  PREDIKSI
# ─────────────────────────────────────────────
with col_result:
    if uploaded is not None:
        st.markdown("#### 🔍 Hasil Klasifikasi")

        with st.spinner("Menganalisis motif batik..."):
            t0    = time.time()
            probs, top5 = predict(model, image)
            elapsed = time.time() - t0

        top1_idx  = top5[0]
        top1_name = CLASS_NAMES[top1_idx]
        top1_conf = probs[top1_idx] * 100

        # Warna confidence
        if top1_conf >= 80:
            badge_color, badge_text = "#2e7d32", "Tinggi"
        elif top1_conf >= 50:
            badge_color, badge_text = "#e65100", "Sedang"
        else:
            badge_color, badge_text = "#b71c1c", "Rendah"

        desc = CLASS_DESC.get(top1_name, "Motif batik khas Indonesia.")

        st.markdown(f"""
<div class="result-card">
    <div class="badge">Motif Terdeteksi</div>
    <div class="result-class">{top1_name}</div>
    <div class="result-confidence">
        Kepercayaan: <strong style="color:#e8c98a">{top1_conf:.1f}%</strong>
        &nbsp;·&nbsp;
        <span style="color:{badge_color};font-weight:500">{badge_text}</span>
        &nbsp;·&nbsp;
        <span style="color:#4a4030">{elapsed*1000:.0f}ms</span>
    </div>
    <div class="confidence-bar-bg">
        <div class="confidence-bar-fill" style="width:{top1_conf:.1f}%"></div>
    </div>
    <div class="desc-text">{desc}</div>
</div>
""", unsafe_allow_html=True)

        # ── Chart Top-5 ──────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        top5_names  = [CLASS_NAMES[i] for i in top5]
        top5_probs  = [probs[i] * 100 for i in top5]
        bar_colors  = ["#e8c98a" if i == 0 else "#3d3020" for i in range(5)]
 
        fig = go.Figure(go.Bar(
            x          = top5_probs,
            y          = top5_names,
            orientation= "h",
            marker_color = bar_colors,
            text       = [f"{p:.1f}%" for p in top5_probs],
            textposition = "inside",
            textfont = dict(size=12),
        ))
        fig.update_layout(
            title     = dict(text="Top-5 Prediksi", font=dict(color="#9c9080", size=13)),
            paper_bgcolor = "rgba(0,0,0,0)",
            plot_bgcolor  = "rgba(0,0,0,0)",
            font      = dict(family="DM Sans", color="#9c9080"),
            xaxis     = dict(
                range=[0, 105], showgrid=False,
                tickfont=dict(color="#4a4030"), title="",
                zeroline=False
            ),
            yaxis     = dict(
                autorange="reversed",
                tickfont=dict(color="#c4b89a", size=12),
                title=""
            ),
            margin    = dict(l=10, r=20, t=40, b=10),
            height    = 260,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    else:
        # Placeholder saat belum ada gambar
        st.markdown("""
<div style="
    background:#1a1713;
    border:2px dashed #2d2820;
    border-radius:20px;
    padding:3rem 2rem;
    text-align:center;
    color:#4a4030;
    margin-top:2rem;
">
    <div style="font-size:3rem;margin-bottom:1rem">🎨</div>
    <div style="font-family:'Playfair Display',serif;font-size:1.2rem; color:#ffffff">
        Upload gambar untuk mulai
    </div>
    <div style="font-size:0.85rem;margin-top:0.5rem;color:#ffffff">
        Mendukung JPG, PNG, WEBP
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;color:#2d2820;font-size:0.8rem;padding:1rem 0">
    Klasifikasi Motif Batik Indonesia · EfficientNetB0 · TensorFlow 2.15
</div>
""", unsafe_allow_html=True)
