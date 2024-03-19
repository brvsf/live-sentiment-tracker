import streamlit as st
import requests
from st_keyup import st_keyup
import hydralit as hy
from PIL import Image
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
from nltk.tokenize import word_tokenize
from preprocess import SlangTranslation
from model import vader_scores
import seaborn as sns
import matplotlib.pyplot as plt
import nltk

nltk.download('punkt')

#app = hy.HydraApp(title='Live sentiment tracker', use_loader=False, favicon= '😎',
#                 navbar_theme = {'txc_inactive': '#FFFFFF','menu_background':'#FA5765','txc_active':'#FA5765','option_active':'white'})

st.set_page_config(page_title="Live sentiment tracker", layout="wide", page_icon = "😎")



selected = option_menu(
    menu_title=None,  # required
    options=["Home", "Project", "The crew"],  # required
    icons=["house", "rocket-takeoff", "person-badge"],  # optional
    menu_icon="cast",  # optional
    default_index=0,  # optional
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"},
        "nav-link": {
            "font-size": "25px",
            "text-align": "left",
            "margin": "0px",
            "--hover-color": "#eee",
        },
        "nav-link-selected": {"background-color": "#FA5765"},
    },
)




st.write('<style>div.block-container{padding-top:3.5rem;}</style>', unsafe_allow_html=True)


img_pos = Image.open('live-sentiment-tracker/project/images/positive.png')
img_neg = Image.open('live-sentiment-tracker/project/images/negative.png')
img_neu = Image.open('live-sentiment-tracker/project/images/neutral.png')


#@app.addapp(is_home=True)
#def home():
if selected == "Home":
    hy.header('Live sentiment tracking:')

    col1, col2 = st.columns([5, 2])

    with col1:

        st.markdown(' ')
        st.markdown(' ')
        st.markdown(' ')
        st.markdown(' ')
        st.markdown(' ')
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


    tokenized_sentence = word_tokenize(SlangTranslation(value).apply_translator(value))
    sentences = []
    for i in range(1, len(tokenized_sentence) + 1):
        new_sentence = ' '.join(tokenized_sentence[:i])
        sentences.append([new_sentence])

    # Create a list of dicts with the vader_scores for each section of the sentence
    progressive_scores = []
    for i in sentences:
        progressive_scores.append(vader_scores(i[0]))

    # Calculating the compound value for each word added in the sentence
    compound_values = [score['compound'] for score in progressive_scores]

    # Seaborn config
    sns.set_style("darkgrid")
    fig_prog = plt.figure(figsize=(10, 6))
    sns.lineplot(x=[i+1 for i in range(len(compound_values))], y=compound_values, marker=None, color="black")

    # Adding title and labels
    plt.title('Sentiment Progression')
    plt.xlabel('Impact of Each word')

    plt.text(0.5, 1, 'Very Positive', ha='right')
    plt.text(0.5, 0.5, 'Positive', ha='right')
    plt.text(0.5, 0, 'Neutral', ha='right')
    plt.text(0.5, -0.5, 'Negative', ha='right')
    plt.text(0.5, -1, 'Very Negative', ha='right')

    plt.ylim(-1, 1)
    plt.yticks([])


    st.markdown('You can also check how each word impacted the results of your sentence!')

    graph_toggle = st.toggle('Show sentiment progression graph')
    if graph_toggle:
        if len(value.split(' ')) >= 5:
            st.pyplot(fig_prog)
        else:
            st.markdown('**⚠️Please make sure to write at least 5 words.⚠️**')


    # st.markdown(f'positive: {pos_size}, P: {result["pos"]}')
    # st.markdown(f'neutral: {neu_size}, P: {result["neu"]}')
    # st.markdown(f'negative: {neg_size}, P: {result["neg"]}')

#@app.addapp(title = ' Project' , icon = '🚀')
#def app2():
if selected == "Project":

    #construction = Image.open('live-sentiment-tracker/project/images/construction.jpeg')

    #st.markdown("<h1 style='text-align: center; color: black;'>Oops, you're here early</h1>",
    #             unsafe_allow_html=True)
    #cols = st.columns([1, 1, 1, 1, 1, 1, 1])

    #with cols[3]:
    #    st.image(construction.resize((4000,4000)))

    #st.markdown("<h4 style='text-align: center; color: black;'>Don't mind us :)</h4>",
    #             unsafe_allow_html=True)



    st.markdown("<h2 style='text-align: center; color: black; text-decoration:underline'>🚀 Live sentiment tracker project</h2>",
                 unsafe_allow_html=True)
    st.markdown('''Live sentiment tracker is a project that uses a pre-trained model to extract
             sentiment from text as the user types. This can be very
             useful to help people write a text or message while receiving
             feedback on the overall emotion that the text expresses. More information
             about the people who made this happen can be find in our **The crew** tab.''')

    st.markdown("<h2 style='text-align: center; color: black; text-decoration:underline'>🔧 Technical stuff</h2>",
                 unsafe_allow_html=True)

    st.markdown('''Our app uses the [vaderSentiment](https://github.com/cjhutto/vaderSentiment/tree/master/vaderSentiment)
                pre-trained model. This model is widely used to work with sentiment
                analysis and has around 88% accuracy, classifying texts between
                three emotions: Positive, negative and neutral.''')
    st.markdown('''The model is trained by understanding the “weight” of each
                word and how this affects the overall sentiment of a text. (read:
                [VADER: A Parsimonious Rule-based Model for Sentiment Analysis of
                Social Media Text](https://ojs.aaai.org/index.php/ICWSM/article/view/14550/14399)
                ). The word cloud graph above illustrates how
                much a word can affect the sentiment of a whole phrase.''')
    word_cloud = Image.open('live-sentiment-tracker/project/images/word_cloud.JPG')
    cols = st.columns((1, 1, 1))
    with cols[1]:

        st.image(word_cloud)

    st.markdown('''Even though the model is already trained on some of the most
                common abbreviations (example: lol, omg), we noticed during implementation
                that they’re not interpreted with the same weights as the full
                phrases. For example, “laughing out loud” had a much higher positive
                impact than “lol” did. With that in mind, we decided to level out the
                playing field and convert them to their extended formats.''')

    st.markdown('''It’s important to note that this is not the same as retraining the
                model. This is done during the preprocessing stage and will be
                done before the phrase is given to the model for processing.''')

    st.markdown('''The bar graph below is an example of how Vader can be used
                to classify texts, in this case it's classifying a sample of 1000
                [movie reviews from IMDB](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).''')

    imdb_reviews = Image.open('live-sentiment-tracker/project/images/imdb_reviews.JPG')
    cols = st.columns((1, 1, 1))
    with cols[1]:

        st.image(imdb_reviews.resize((2500,2000)))

    st.markdown("<h2 style='text-align: center; color: black; text-decoration:underline'>🔗 API</h2>",
                 unsafe_allow_html=True)

    st.markdown('''Our API is hosted in Google Cloud and can be accessed by anyone
                using this [link](https://live-sentiment-tracker-xi2puhfsaq-ew.a.run.app/).
                Here’s a quick tutorial on how to use it effectively:''')

    st.markdown('''1. If you want to make a query to receive the overall sentiment
                of a text you should use **"/predict"**, with the following syntax:  \n
                https://live-sentiment-tracker-xi2puhfsaq-ew.a.run.app/predict?sentence=<TEXT>''')
    st.markdown('''2. If you want to receive the scores of each sentiment in a
                text you should use **"/scores"**, with the following syntax:  \n
                https://live-sentiment-tracker-xi2puhfsaq-ew.a.run.app/scores?sentence=<TEXT>''')
    st.markdown('''**PS: Make sure you write your text as you normally would, with regular spacing.
                The URL formatting will be taken care of automatically**''')






#@app.addapp(title=' The crew', icon = '👨‍🔧')
#def app3():
if selected == "The crew":

 cols = st.columns((1,1,1,1,1,1))

 with cols[1]:
     isaac_pfp = Image.open('live-sentiment-tracker/project/images/isaac_pfp.jpg')
     st.image(isaac_pfp.resize((200,200)))

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
     st.markdown("<h2 style='text-align: center; color: black;'>Isaac  🤔</h2>",
                 unsafe_allow_html=True)

 with cols[-2]:
    bruno_pfp = Image.open('live-sentiment-tracker/project/images/bruno_pfp.jpeg')
    st.image(bruno_pfp.resize((200,200)))

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
    st.markdown("<h2 style='text-align: center; color: black;'>Bruno 🧐</h2>",
                unsafe_allow_html=True)

 col1, col2 = st.columns(2)

 with col1:


     st.markdown('**Pets-**  None 😔')
     st.markdown('**Food-** Pho 🍜')
     st.markdown('**Age-** 24 ')
     st.markdown('**Season-** Winter ⛄')
     st.markdown('**Movie-** Saving Private Ryan 🎖️')
     st.markdown('**Beverage-** Pureza 🥤')
     st.markdown('**Fruit-** Watermelon 🍉')
     st.link_button("Check Isaac's Linkedin",
                    "https://www.linkedin.com/in/isaac-pereira-474a9117b")

 with col2:

    st.markdown('**Pets-**  Horus 🐶')
    st.markdown('**Food-** Sushi 🍣')
    st.markdown('**Age-** 24 ')
    st.markdown('**Season-** Winter ⛄')
    st.markdown('**Movie-** The Lord of The Rings: The Return of the King :ring:')
    st.markdown('**Beverage-** Coffee :coffee:')
    st.markdown('**Fruit-** Strawberry :strawberry:')
    st.link_button("Check Bruno's Linkedin",
                   "https://www.linkedin.com/in/bruno-vieira-2b0817238")

 col1, col2, col3, col4, col5, col6 = st.columns(6)
 #with col2:

 #with col3:


 #Run the whole lot, we get navbar, state management and app isolation, all with this tiny amount of work.
 #app.run()
