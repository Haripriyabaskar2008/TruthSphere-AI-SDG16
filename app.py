import streamlit as st
import pickle


model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))


st.set_page_config(
    page_title="TruthSphere AI",
    page_icon="🛡️",
    layout="centered"
)


st.title("🛡️ TruthSphere AI")
st.subheader("AI-Powered Misinformation Detection and Trust Analysis")

st.write(
    "Enter a news article or statement below and let AI analyze whether it is likely to be real or fake."
)


news = st.text_area(
    "Enter News Content",
    height=200,
    placeholder="Paste a news article or statement here..."
)


if st.button("Analyze News"):

    if news.strip() == "":
        st.warning("⚠️ Please enter some news content.")
    else:

        
        news_vector = vectorizer.transform([news])

      
        prediction = model.predict(news_vector)

    
        confidence = model.predict_proba(news_vector)

        if prediction[0] == 0:
            st.error("🚨 Prediction: Fake News")
            st.metric(
                label="Confidence Score",
                value=f"{confidence[0][0]*100:.2f}%"
            )
            st.warning("Trust Score: Low")
            st.info("Recommendation: Verify information using trusted sources.")
        else:
            st.success("✅ Prediction: True News")
            st.metric(
                label="Confidence Score",
                value=f"{confidence[0][1]*100:.2f}%"
            )
            st.success("Trust Score: High")
            st.info("Recommendation: Information appears reliable.")


st.markdown("---")
st.caption(
    "TruthSphere AI - AI-Powered Misinformation Detection and Trust Analysis for a Sustainable Digital Society (SDG 16)"
)