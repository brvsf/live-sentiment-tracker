import streamlit as st
import requests
from st_keyup import st_keyup
import hydralit as hy
from PIL import Image

app = hy.HydraApp(title='Live sentiment tracker', use_loader=False)

img_pos = Image.open('live-sentiment-tracker/project/images/positive.png')
img_neg = Image.open('live-sentiment-tracker/project/images/negative.png')
img_neu = Image.open('live-sentiment-tracker/project/images/neutral.png')


@app.addapp()
def home():
    hy.markdown('Test home page')
    value = st_keyup("", key="0")

    url = "http://127.0.0.1:8000/predict?sentence="

    url_full = url + value

    req = requests.get(url = url_full)
    result = req.json()

    pos_size = int((result['pos']*1.2)*80)
    if pos_size < 30:
        pos_size = 30

    neg_size = int((result['neg']*1.2)*80)
    if neg_size < 30:
        neg_size = 30

    neu_size = int((result['neu']*1.2)*80)
    if neu_size < 30:
        neu_size = 30

    col1, col2, col3, col4, col5, col6, col7, col8, col9, \
    col10, col11, col12, col13, col14, col15 = st.columns(15)

    with col1:
        # st.header("A cat")
        st.image(img_pos.resize((pos_size, pos_size)))

    with col2:
        # st.header("A dog")
        st.image(img_neu.resize((neu_size, neu_size)))

    with col3:
       # st.header("An owl")
       st.image(img_neg.resize((neg_size, neg_size)))

    st.markdown(f'positive: {pos_size}, P: {result["pos"]}')
    st.markdown(f'neutral: {neu_size}, P: {result["neu"]}')
    st.markdown(f'negative: {neg_size}, P: {result["neg"]}')

    # st.image(image,caption = 'Success')

@app.addapp()
def about_us():
 # hy.info('Hello from app 2')
 hy.markdown('Page for group about info')


#Run the whole lot, we get navbar, state management and app isolation, all with this tiny amount of work.
app.run()
