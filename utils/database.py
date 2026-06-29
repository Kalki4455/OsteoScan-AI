# -----------------------------
# Grad-CAM Heatmap
# -----------------------------
img_array = preprocess_image(image)

last_conv_layer = None

for layer in reversed(model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer = layer.name
        break

if last_conv_layer is not None:

    heatmap = make_gradcam_heatmap(
        img_array,
        model,
        last_conv_layer
    )

    original = np.array(image.convert("RGB"))

    heatmap_image = overlay_heatmap(
        original,
        heatmap
    )

    st.markdown("---")

    st.subheader("🔥 AI Attention Heatmap")

    col1, col2 = st.columns(2)

    with col1:
        st.image(original, caption="Original Image")

    with col2:
        st.image(heatmap_image, caption="Grad-CAM Heatmap")