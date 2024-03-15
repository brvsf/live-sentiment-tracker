import streamlit as st
import requests
from st_keyup import st_keyup
import hydralit as hy
from PIL import Image
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

app = hy.HydraApp(title='Live sentiment tracker', use_loader=False)

img_pos = Image.open('live-sentiment-tracker/project/images/positive.png')
img_neg = Image.open('live-sentiment-tracker/project/images/negative.png')
img_neu = Image.open('live-sentiment-tracker/project/images/neutral.png')


@app.addapp(is_home=True)
def home():
    hy.header('Live sentiment tracking:')

    col1, col2 = st.columns([5, 2])

    with col1:

        st.markdown(' ')
        st.markdown(' ')
        st.markdown('**Make sure to pay attention to the emojis while you write!**')
        st.markdown(' ')
        st.markdown('''You can also toggle the pie chart* to see a more precise
                    breakdown of each emotion. We recommend turning it on after
                    the message is written.''')
        st.markdown(' ')
        st.markdown('''*Please note that the pie chart may cause the app to run
                    slower and no longer update the emojis live.''')



        value = st_keyup("", key="0", placeholder= 'Write here...', )

    #url = "http://127.0.0.1:8000/scores?sentence="
    url = "https://live-sentiment-tracker-xi2puhfsaq-ew.a.run.app/scores?sentence="

    url_full = url + value

    req = requests.get(url = url_full)
    result = req.json()

    with col2:

        pie_toggle = st.toggle('Toggle pie chart', value=False, key=None,
                               help=None, on_change=None, args=None, kwargs=None,
                               disabled=False, label_visibility="visible")

        if pie_toggle:

            if value:
                labels = 'Neutral', 'Positive', 'Negative'
                sizes = [float(result['neu'])*100, float(result['pos'])*100, float(result['neg'])*100]  # only "explode" the 2nd slice (i.e. 'Hogs')
                fig1, ax1 = plt.subplots()
                ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
                shadow=False, startangle=90)
                ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                st.pyplot(fig1)
            else:
                labels = 'Neutral', 'Positive', 'Negative'
                sizes = [1, 1, 1]  # only "explode" the 2nd slice (i.e. 'Hogs')
                fig2, ax2 = plt.subplots()
                ax2.pie(sizes, labels=labels, autopct='%1.1f%%',
                shadow=False, startangle=90)
                ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                st.pyplot(fig2)

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
        st.image(img_pos.resize((pos_size, pos_size)))

    with col2:
        st.image(img_neu.resize((neu_size, neu_size)))

    with col3:
       st.image(img_neg.resize((neg_size, neg_size)))



    # st.markdown(f'positive: {pos_size}, P: {result["pos"]}')
    # st.markdown(f'neutral: {neu_size}, P: {result["neu"]}')
    # st.markdown(f'negative: {neg_size}, P: {result["neg"]}')

@app.addapp(title = ' Project' , icon = '🚀')
def app2():

    st.markdown("<h1 style='text-align: center; color: black;'>Presentation</h1>",
                 unsafe_allow_html=True)
    cols = st.columns([1, 1, 1, 1, 1, 1])

    with cols[1]:
        components.iframe("https://docs.google.com/presentation/d/e/2PACX-1vShfnJ"
                      "JYuL9t4OGCmVJz5JKO7fi1BUyFzcjwqiHBom77xHYzBv5aC_7-fE46kuf"
                      "YQcNu4EpDlVDUxsE/embed?start=false&loop=false&delayms=60000"
                      , height=560, width = 900)


@app.addapp(title=' The crew', icon = '👨‍🔧')
def app3():


 col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(9)

 with col2:

     isaac_pfp = Image.open('live-sentiment-tracker/project/images/isaac_pfp.jpeg')
     st.image(isaac_pfp.resize((150,150)))

     st.markdown("""
        <style>
            .st-emotion-cache-1v0mbdj {
              width: 150px;
              height: 150px;
              border-radius: 50%;
              overflow: hidden;
              box-shadow: 0 0 5px rgba(0, 0, 0, 0.3);
          }

            .st-emotion-cache-1v0mbdj img {
              width: 100%;
              height: 100%;
              object-fit: cover;
          }
        </style>
        """, unsafe_allow_html=True)

 col1, col2, col3 = st.columns(3)

 with col1:


     st.markdown("<h2 style='text-align: center; color: black;'>Isaac </h2>",
                 unsafe_allow_html=True)


     st.markdown('**Pets-**  None 😔')
     st.markdown('**Food-** Pho 🍜')
     st.markdown('**Age-** 24 ')
     st.markdown('**Season-** Winter ⛄')
     st.markdown('**Movie-** Saving Private Ryan 🎖️')
     st.markdown('**Beverage-** Pureza 🥤')
     st.markdown('**Fruit-** Watermelon 🍉')


 with col2:
     st.markdown("<h2 style='text-align: center; color: black;'>Bruno </h2>",
                 unsafe_allow_html=True)
 with col3:
     st.markdown("<h2 style='text-align: center; color: black;'>Rachid </h2>",
                 unsafe_allow_html=True)


#Run the whole lot, we get navbar, state management and app isolation, all with this tiny amount of work.
app.run()
