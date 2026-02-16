import streamlit as st
import torch
import timm
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import pandas as pd

# --- 1. AYARLAR VE MODEL BİLGİLERİ ---
MODEL_PATH = "yatak_yarasi_rexnet.pth"
MODEL_ARCH = 'rexnet_150'
NUM_CLASSES = 7

# DÜZELTİLMİŞ SINIF LİSTESİ (Alfabetik Sıraya Göre)
# Modelin eğitim sırasında klasörleri A'dan Z'ye sıraladığı varsayılarak düzeltilmiştir.
# 0: Deep Tissue (D)
# 1: Healthy (H) -> Daha önce burası karışıktı
# 2: Stage 1 (S)
# 3: Stage 2 (S)
# 4: Stage 3 (S)
# 5: Stage 4 (S)
# 6: Unstageable (U)
CLASS_NAMES = [
    "Derin Doku Hasarı (Deep Tissue)",  # Index 0
    "Sağlıklı Doku (Healthy)",          # Index 1
    "Evre 1 (Stage 1)",                 # Index 2
    "Evre 2 (Stage 2)",                 # Index 3
    "Evre 3 (Stage 3)",                 # Index 4
    "Evre 4 (Stage 4)",                 # Index 5
    "Evrelemez / Nekrotik (Unstageable)" # Index 6
]

# --- 2. SAYFA TASARIMI ---
st.set_page_config(
    page_title="Yara Analiz Pro", 
    page_icon="🩺", 
    layout="wide"
)

# Özel CSS ile temiz görünüm
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; font-weight: bold; }
    .stMetric { background: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

st.title("🩺 Yatak Yarası ve Diyabetik Ayak Analizi")
st.markdown("**Model:** ReXNet-150 | **Durum:** Web Tabanlı Canlı Analiz")
st.markdown("---")

# --- 3. MODEL YÜKLEME FONKSİYONU ---
@st.cache_resource
def load_model():
    try:
        # ReXNet mimarisini kur (1.5x ölçekli)
        model = timm.create_model(MODEL_ARCH, pretrained=False, num_classes=NUM_CLASSES)
        
        # Ağırlıkları yükle (CPU uyumlu modda)
        state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except FileNotFoundError:
        st.error(f"🚨 HATA: '{MODEL_PATH}' dosyası bulunamadı. Lütfen dosya adını kontrol edin.")
        st.stop()
    except Exception as e:
        st.error(f"🚨 Model Yükleme Hatası: {e}")
        st.stop()

model = load_model()

# --- 4. GÖRÜNTÜ İŞLEME VE TAHMİN ---
def predict(image, model):
    # ImageNet standart normalizasyonu
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Görseli tensöre çevir ve boyut ekle (Batch size: 1)
    input_tensor = preprocess(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        # Softmax ile olasılığa çevir
        probs = F.softmax(output, dim=1)[0]
        conf, pred_idx = torch.max(probs, 0)
        
    return conf.item(), pred_idx.item(), probs

# --- 5. ANA ARAYÜZ ---
col_sol, col_sag = st.columns([1, 1.2])

with col_sol:
    st.subheader("1. Fotoğraf Yükle")
    uploaded_file = st.file_uploader("Analiz edilecek bölgenin fotoğrafı", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        # Görseli RGB'ye çevirerek aç (Renk hatasını önler)
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Yüklenen Görsel", use_container_width=True)

with col_sag:
    st.subheader("2. Analiz Sonucu")
    
    if uploaded_file and st.button("ANALİZİ BAŞLAT", type="primary"):
        with st.spinner("Yapay zeka doku katmanlarını inceliyor..."):
            confidence, index, all_probs = predict(image, model)
            
            result_label = CLASS_NAMES[index]
            
            # --- SONUÇ KARTI ---
            if confidence > 0.50:
                if "Sağlıklı" in result_label:
                    st.success(f"✅ SONUÇ: **{result_label}**")
                else:
                    st.error(f"⚠️ TESPİT: **{result_label}**")
            else:
                st.warning(f"❓ SONUÇ: **{result_label}** (Düşük Güven)")
            
            # Metrikler
            c1, c2 = st.columns(2)
            c1.metric("Güven Skoru", f"%{confidence*100:.1f}")
            c2.metric("Sınıf İndeksi", f"{index}")

            # --- DETAYLI GRAFİK ---
            st.markdown("### 📊 Detaylı Olasılıklar")
            
            # Veriyi tabloya dök
            probs_df = pd.DataFrame({
                "Durum": CLASS_NAMES,
                "Olasılık (%)": [p.item() * 100 for p in all_probs]
            })
            # Olasılığa göre sırala
            probs_df = probs_df.sort_values(by="Olasılık (%)", ascending=False)
            
            # Grafik çiz
            st.bar_chart(probs_df.set_index("Durum"))
            
            # Tablo göster
            with st.expander("Sayısal Verileri Göster"):
                st.table(probs_df)

# Alt bilgi
st.markdown("---")
st.caption("Bu sistem ReXNet-150 mimarisi kullanılarak geliştirilmiştir. Sonuçlar klinik karar destek amaçlıdır.")