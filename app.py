from joblib import load
import streamlit as st
from streamlit_mnist_canvas import st_mnist_canvas

# Carregar modelo treinado
modelo_sgd = load("models/modelo_sgd.pkl")

# Interface
st.markdown('<div class="drawImage">', unsafe_allow_html=True)
st.subheader("Desenhe um número (0 a 9):")
canvas_result = st_mnist_canvas()
st.markdown("</div>", unsafe_allow_html=True)

# Se o usuário desenhou e enviou
if canvas_result.is_submitted and canvas_result.resized_grayscale_array is not None:
    img = canvas_result.resized_grayscale_array
    img_reshaped = img.reshape(1, -1)  # -1 adapta automaticamente (784 no MNIST)

    # Fazer previsão
    prediction = modelo_sgd.predict(img_reshaped)
    predicted_number = prediction[0]

    # Mostrar resultado
    st.success(f"De acordo com a IA, o número desenhado é **{predicted_number}**")

# Aviso
st.caption("⚠️ A IA pode cometer erros. Verifique os resultados com cuidado.")
