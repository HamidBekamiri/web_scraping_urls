import streamlit as st
import numpy as np
import pandas as pd
import xmltodict
import urllib.request
import seaborn as sns
import matplotlib
from matplotlib.figure import Figure
from PIL import Image
import requests
st.set_page_config(layout="wide", page_title="Element Solutions Recommender System", page_icon="🐞")

st.header("🐞 Web Scraping Based on Urls!")
st.subheader('AI-Growth-Lab AAU')

matplotlib.use("agg")


sns.set_style('darkgrid')
import nltk
nltk.download('punkt')

def get_txt_urls (urls):
    listofdf_urls = []
    list_urls = urls.url.to_list()
    import pandas as pd
    df_url_all = pd.DataFrame(columns= ['person', 'sentences'])

    for url in list_urls:
        import requests
        # get() method downloads the entire HTML of the provided url
        response = requests.get(url)
        # Get the text from the response object
        # response_text = response.text
        from bs4 import BeautifulSoup
        # Create a BeautifulSoup object
        # response_text -> The downloaded webpage
        # lxml -> Used for processing HTML and XML pages
        soup = BeautifulSoup(response.content,features="html.parser")
        # rating_list_q = soup.find_all(class_ = 'wide')
        # with open("output1.html", "w") as file:
        #         file.write(str(soup.find_all(class_ = 'wide')))
        # st.write(soup.find(class_ = 'wide').get_text())
        ratings = []
        ratings.append(soup.find(class_ = 'wide').get_text())
        # As we saw the rating's class name was "ratings-bar" 
        # we prefix "." since its a class
        # rating_class_selector = ".wide"
        # Extract the all the ratings class
        # rating_list = soup.select(rating_class_selector)
        # st.write(rating_list[1])
        # This List will store all the ratings
        # Iterate through all the ratings object
        # st.write(soup.select(rating_class_selector))
        # rating_text = soup.select(rating_class_selector).find('').get_text()
        # for rating_object in rating_list_q:
        #     # st.write(rating_object)
        #     # Find the <strong> tag and get the Text
        #     rating_text = rating_object.find('').get_text()
        #     # rating_text = rating_text_element if rating_text_element else "No Description"
        #     # rating_text = rating_object.find('').getText() 
        #     # Append the rating to the list
        #     ratings.append(rating_text)
        list_p = ratings[0].split('\n\n')
        matches = []
        for match in list_p:
            if " \n \n \n \n" in match:
                matches.append(match)
        clean_list_speaker = []
        for mach in matches:
            mach = mach[8:]
            clean_list_speaker.append(mach)
        list_clean_list_speaker = []
        for item in clean_list_speaker:
            item_clean_list_speaker = item.split(':\n')
            list_clean_list_speaker.append(item_clean_list_speaker)
        person = []
        sentences = []
        organ = []
        for item in list_clean_list_speaker:
            if len(item) == 2:
                if item[0][:1] == '\n':
                    item[0] = item[0][1:]
                if item[0][-1:] == ')':
                    organ = item[0]
                    item[0] = item[0].split('(')[0][:-1]
                person.append(item[0])
                sentences.append(item[1])
        import pandas as pd
        df_url = pd.DataFrame({'person': person, 'sentences': sentences})
        i = list_urls.index(url)
        locals()["df_url_"+str(i)] = df_url.copy()
        listofdf_urls.append("df_url_"+str(i))
        df_url_all = pd.concat([df_url_all, locals()["df_url_"+str(i)]], ignore_index=True)
    return df_url_all

def get_tabel_content(df_url_all):
    from nltk.tokenize import word_tokenize
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    from nltk.stem import PorterStemmer
    X = df_url_all['sentences']
    print(X)
    corpus = [word_tokenize(token) for token in X]
    lowercase_train = [[token.lower() for token in doc] for doc in corpus]
    alphas = [[token for token in doc if token.isalpha()] for doc in lowercase_train]
    train_no_stop = [[token for token in doc if token not in stop_words] for doc in alphas]
    stemmer = PorterStemmer()
    stemmed = [[stemmer.stem(token) for token in doc] for doc in train_no_stop]
    train_clean_str = [ ' '.join(doc) for doc in stemmed]
    nb_words = [len(tokens) for tokens in alphas]
    alphas_unique = [set(doc) for doc in alphas]
    nb_words_unique = [len(doc) for doc in alphas_unique]
    train_str = [ ' '.join(doc) for doc in lowercase_train]
    nb_characters = [len(doc) for doc in train_str]
    train_stopwords = [[token for token in doc if token in stop_words] for doc in alphas]
    nb_stopwords = [len(doc) for doc in train_stopwords]
    non_alphas = [[token for token in doc if token.isalpha() == False] for doc in lowercase_train]
    nb_punctuation = [len(doc) for doc in non_alphas]
    train_title = [[token for token in doc if token.istitle() == True] for doc in corpus]
    nb_title = [len(doc) for doc in train_title]
    df_clean = pd.DataFrame(data={'text_clean': train_clean_str})
    df_clean.head()
    nb_words = pd.Series(nb_words)
    nb_words_unique = pd.Series(nb_words_unique)
    nb_characters = pd.Series(nb_characters)
    nb_stopwords = pd.Series(nb_stopwords)
    nb_punctuation = pd.Series(nb_punctuation)
    nb_title = pd.Series(nb_title)
    nb_total = nb_punctuation + nb_words + nb_stopwords
    df_show = pd.concat([df_clean, nb_total, nb_words, nb_words_unique, nb_characters, nb_stopwords, nb_punctuation, nb_title], axis=1).rename(columns={
        0: "Totall tokens", 1: "Number of words", 2: 'Number of unique words', 3: 'Number of characters', 4: 'Number of stopwords', 5: 'Number of punctuations',
        6: 'Number of titlecase words'
    })
    return df_show

@st.cache
def get_user_data(user_id, key='ZRnySx6awjQuExO9tKEJXw', v='2', shelf='read', per_page='200'):
    api_url_base = 'https://www.goodreads.com/review/list/'
    final_url = api_url_base + user_id + '.xml?key=' + key + \
        '&v=' + v + '&shelf=' + shelf + '&per_page=' + per_page
    contents = urllib.request.urlopen(final_url).read()
    return(contents)

def convert_df(df):
   return df.to_csv().encode('utf-8')
st.write("A Sample of the CSV")
df_sample = pd.read_csv('urls_web_scraping.csv')

st.dataframe(df_sample.head(2))
uploaded_file = st.file_uploader("Upload CSV file of Urls")
if uploaded_file is not None:    
    #read csv
    df_urls = pd.read_csv(uploaded_file)
    df_urls.head()
    txt = get_txt_urls(df_urls)
    df_cnt = get_tabel_content(txt)
    df_urls_all_report = pd.concat([txt, df_cnt], axis=1)
    st.dataframe(df_urls_all_report)
    csv = convert_df(df_urls_all_report)

    st.download_button("Press to Download", csv,"file.csv","text/csv",key='download-csv')
    df_urls_all_report = df_urls_all_report.reset_index()
    df_urls_all_report_person_text = df_urls_all_report.groupby('person')['text_clean'].transform(lambda x : ' '.join(x)).reset_index()
    df_urls_all_report_person_text = df_urls_all_report_person_text.rename(columns={'text_clean':'text_clean_all'})
    df_urls_all_report_person_set = pd.merge(df_urls_all_report_person_text, df_urls_all_report, right_on='index', left_on='index')
    # st.write(df_urls_all_report_person_set)

    df_urls_all_report_person_set_new = df_urls_all_report_person_set.drop(columns=['index','sentences', 'text_clean','Totall tokens', 'Number of words', 'Number of unique words', 'Number of characters', 'Number of stopwords', 'Number of punctuations','Number of titlecase words'])

    df_urls_all_report_person_set_f = df_urls_all_report_person_set_new.drop_duplicates()

    person_set = set(df_urls_all_report['person'].to_list())
    df_urls_all_report_person = df_urls_all_report.groupby('person').sum()['Totall tokens'].sort_values(ascending=False)
    df_urls_all_report_person = df_urls_all_report_person.reset_index()
    df_urls_all_report_person_set = pd.concat([df_urls_all_report, df_urls_all_report_person_text], axis=1)

    st.write(df_urls_all_report_person)
    csv_1 = convert_df(df_urls_all_report_person)

    st.download_button("Press to Download", csv_1,"file.csv","text/csv",key='download-csv')

    df_urls_all_report_person = pd.DataFrame(df_urls_all_report_person).reset_index()
    df_urls_all_report_person_top = df_urls_all_report_person[df_urls_all_report_person['Totall tokens']>1000]
    import seaborn as sns
    fig = Figure()
    ax = fig.subplots()
    sns.barplot(x=df_urls_all_report_person_top['Totall tokens'], y=df_urls_all_report_person_top['person'], data=df_urls_all_report_person_top, ax=ax)
    # ax.title('Top 30 Speaker by word count value', fontsize=16)
    # ax.xlabel('Word count')
    # ax.set_ylabel('Percentage')
    ax.set_xlabel('Word count')
    ax.set_title('Top 30 Speaker by word count value', fontsize=16)
    st.pyplot(fig)
    from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
    import matplotlib.pyplot as plt
    import seaborn as sns


# stopwords = set(STOPWORDS)
# stopwords = stopwords.union(["ha", "thi", "now", "onli", "im", "becaus", "wa", "will", "even", "go", "realli", "didnt", "abl"])
# wordcl = WordCloud(stopwords = stopwords, background_color='white', max_font_size = 50, max_words = 5000).generate(df_urls_all_report_person_text['text_clean_all'][0])
# plt.figure(figsize=(14, 12))
# plt.imshow(wordcl, interpolation='bilinear')
# fig_2 = Figure()
# fig_2 = plt.subplots()
# Create some sample text
    st.write(df_urls_all_report['person'][0])

    text = 'Fun, fun, awesome, awesome, tubular, astounding, superb, great, amazing, amazing, amazing, amazing'
    # Create and generate a word cloud image:
    wordcloud = WordCloud().generate(df_urls_all_report_person_text['text_clean_all'][0])

    # Display the generated image:
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    st.pyplot()







