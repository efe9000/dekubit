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

# YENİ DÜZELTİLMİŞ SINIF LİSTESİ (İsteklerinize göre güncellendi)
# ÖNEMLİ: Modelin eğitim aşamasındaki klasör sırası bu indekslerle birebir aynı olmalıdır.
CLASS_NAMES = [
    "Yara Objesi Yok / Sağlıklı",       # Index 0
    "Derin Doku Hasarı (Deep Tissue)",  # Index 1
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

# Özel CSS ile klinik görünüm
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3.5em; font-weight: bold; background-color: #007bff; color: white; }
    .stMetric { background: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    .result-box { padding: 20px; border-radius: 10px; margin-top: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🩺 Diyabetik Ayak ve Yara Evreleme Sistemi")
st.markdown(f"**Uzman Sistemi:** {MODEL_ARCH} | **Klinik Hedef:** İç Hastalıkları Hemşireliği Karar Destek")
st.markdown("---")

# --- 3. MODEL YÜKLEME FONKSİYONU ---
@st.cache_resource
def load_model():
    try:
        # ReXNet mimarisini kur
        model = timm.create_model(MODEL_ARCH, pretrained=False, num_classes=NUM_CLASSES)
        
        # Ağırlıkları yükle
        state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except FileNotFoundError:
        st.error(f"🚨 HATA: '{MODEL_PATH}' dosyası bulunamadı.")
        st.stop()
    except Exception as e:
        st.error(f"🚨 Model Yükleme Hatası: {e}")
        st.stop()

model = load_model()

# --- 4. GÖRÜNTÜ İŞLEME VE TAHMİN ---
def predict(image, model):
    # Modelin eğitimde gördüğü standart ImageNet ön işlemesi
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = preprocess(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)[0]
        conf, pred_idx = torch.max(probs, 0)
        
    return conf.item(), pred_idx.item(), probs

# --- 5. ANA ARAYÜZ ---
col_sol, col_sag = st.columns([1, 1.2])

with col_sol:
    st.subheader("📸 Görüntü Yükleme")
    uploaded_file = st.file_uploader("Yara bölgesinin net fotoğrafını yükleyin", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Analiz Edilecek Görüntü", use_container_width=True)

with col_sag:
    st.subheader("🔍 Yapay Zeka Analizi")
    
    if uploaded_file and st.button("DOKU ANALİZİNİ BAŞLAT"):
        with st.spinner("Katmanlar ve nekrotik alanlar inceleniyor..."):
            confidence, index, all_probs = predict(image, model)
            
            result_label = CLASS_NAMES[index]
            
            # --- SONUÇ GÖSTERİMİ ---
            if index == 0:  # Yara Objesi Yok / Sağlıklı
                st.success(f"### SONUÇ: {result_label}")
                st.balloons()
            else:
                st.error(f"### TESPİT: {result_label}")
            
            # Güven Skoru Metrikleri
            c1, c2 = st.columns(2)
            c1.metric("Tahmin Güveni", f"%{confidence*100:.1f}")
            c2.metric("Sınıf Kodu", f"Index {index}")

            # --- OLASILIK DAĞILIMI ---
            st.markdown("### 📊 Olasılık Dağılımı")
            
            probs_df = pd.DataFrame({
                "Kategori": CLASS_NAMES,
                "Olasılık (%)": [p.item() * 100 for p in all_probs]
            }).sort_values(by="Olasılık (%)", ascending=False)
            
            st.bar_chart(probs_df.set_index("Kategori"))
            
            with st.expander("Tüm Sınıf Olasılıklarını Listele"):
                st.table(probs_df)

# Alt Bilgi
st.markdown("---")
st.info("💡 **Bilgi:** Eğer sonuç 'Düşük Güven' veriyorsa, lütfen ışığı ve çekim açısını değiştirerek tekrar deneyiniz.")
st.caption("Bu uygulama bir hemşirelik tez çalışması kapsamında geliştirilmiştir. Tanı koyma amacı taşımaz, karar destek aracıdır.")
