import streamlit as st
import openai, requests
from PIL import Image
from io import BytesIO
import numpy as np
import onnxruntime as ort

# ------------------------------------------------------------------------------
# Load ONNX model with onnxruntime (caches session)
# ------------------------------------------------------------------------------
@st.cache(allow_output_mutation=True)
def load_onnx_model():
    # Load the ONNX file we just created
    sess = ort.InferenceSession(
        r"C:\Users\dheer\Downloads\Detecting-AI-Generated-Fake-Images-main\mobilemodel_v2.onnx",
        providers=["CPUExecutionProvider"]
    )
    return sess

def generate_image(input_image):
    buf = BytesIO()
    input_image.convert("RGB").save(buf, format="PNG")
    resp = openai.Image.create_variation(
        image=buf.getvalue(), n=1, size="1024x1024"
    )
    data = requests.get(resp['data'][0]['url']).content
    return Image.open(BytesIO(data))

def get_prediction(arr, sess):
    # 1) Resize & normalize in HWC order
    img = Image.fromarray(arr.astype('uint8'),'RGB').resize((224,224))
    x = np.array(img).astype(np.float32) / 255.0
    # 2) Transpose to CHW (3,224,224), then batch -> (1,3,224,224)
    x = np.transpose(x, (2,0,1))[None,:,:,:]
    # 3) Run ONNX
    inputs = { sess.get_inputs()[0].name: x }
    out = sess.run(None, inputs)[0]
    p = float(out[0,0])
    return ("Real Human Face", p) if p>0.5 else ("AI Generated Face", 1-p)

# ------------------------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------------------------
st.set_page_config("AI Face Gen & Fake‐Image Detector")

if "openai_api_key" not in st.session_state:
    key = st.text_input("OpenAI API Key", type="password")
    if st.button("Load Key") and key.strip():
        st.session_state.openai_api_key = key.strip()
        st.experimental_rerun()
else:
    openai.api_key = st.session_state.openai_api_key
    sess = load_onnx_model()

    mode = st.sidebar.selectbox("Mode", ["Generator","Detector"])
    if mode=="Generator":
        st.title("AI Human Face Generator")
        f = st.file_uploader("Upload PNG/JPG", ["png","jpg"])
        if f:
            img = Image.open(f)
            c1,c2 = st.columns(2)
            c1.image(img, caption="Input")
            if st.button("Generate"):
                c2.image(generate_image(img), caption="Variation")
    else:
        st.title("Real vs AI Face Detection")
        f = st.file_uploader("Upload PNG/JPG", ["png","jpg"])
        if f:
            arr = np.array(Image.open(f).convert("RGB"))
            c1,c2 = st.columns([2,1])
            c1.image(Image.fromarray(arr), caption="Input")
            label, prob = get_prediction(arr, sess)
            c2.markdown(f"### {label}")
            c2.write(f"Confidence: {prob:.2%}")
