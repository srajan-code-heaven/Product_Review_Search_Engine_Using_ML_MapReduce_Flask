import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
import bs4
from bs4 import BeautifulSoup
import requests
import time

app = Flask(__name__)

def getdata(url):
    HEADERS = ({'User-Agent':
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
                    AppleWebKit/537.36 (KHTML, like Gecko) \
                    Chrome/90.0.4430.212 Safari/537.36',
                'Accept-Language': 'en-US, en;q=0.5'})
    r = requests.get(url, headers=HEADERS)
    return r.text


def html_code(url):
    # pass the url
    # into getdata function
    htmldata = getdata(url)
    soup = BeautifulSoup(htmldata, 'html.parser')

    # display html code
    return (soup)


def cus_rev(soup):
    # find the Html tag
    # with find()
    # and convert into string
    data_str = ""

    product_name_lt = soup.find_all("h1", class_="a-size-large a-text-ellipsis")
    product_name = product_name_lt[0].get_text()
    for item in soup.find_all("a",
                              "a-size-base a-link-normal review-title a-color-base review-title-content a-text-bold"):
        data_str = data_str + item.get_text()

    rev_data = data_str.split("\n")
    rev_result = []
    for i in rev_data:
        if i == "":
            pass
        else:
            rev_result.append(i)
    return (product_name, rev_result)


def findtop10uniquesentences(mat, xtest_df):
    df = mat.replace(0, np.NaN)
    df1 = df.mean(axis=1, skipna=True).sort_values().to_frame()
    merged = xtest_df[['Sentence']].merge(df1, left_index=True, right_index=True, how='inner')
    merged = merged.loc[merged.astype(str).drop_duplicates().index]
    return merged.sort_values(0, ascending=False)[:10]['Sentence']


def crawldata(curr_url):
    cus_res_all = []
    rev_res_all = []
    product_name = ""
    # curr_url="https://www.amazon.com/Feethit-Lightweight-Breathable-Comfortable-Sneakers/dp/B089CWT4CH?pd_rd_w=vMy7L&pf_rd_p=669284d1-0739-4910-99f9-d216bdde93da&pf_rd_r=885HEE90YA1F4MMCYC5G&pd_rd_r=1f4dd7ee-d97e-41a3-a1f4-73c39fae5a17&pd_rd_wg=PJ84U&pd_rd_i=B089CWT4CH&psc=1&ref_=pd_bap_d_rp_1_i"
    for j in range(1,4):
        url = "https://www.amazon.com/product-reviews/" + curr_url.split("dp/")[1].split("/")[0]+"/ref=cm_cr_dp_d_show_all_btm?ie=UTF8"
        url = url+"&reviewerType=all_reviews&pageNumber=" + str(j)
        time.sleep(3)
        soup = html_code(url)
        print(url)
        product_name, cus_rev_currpage = cus_rev(soup)
        cus_rev_currpage = [x for x in cus_rev_currpage if "The media could not be loaded" not in x and x.strip() != ""]
        rev_res_all.extend(cus_rev_currpage)
        if product_name != "":
            product_name = product_name
    data = {'Sentence': rev_res_all}
    # Create DataFrame
    xtest_df = pd.DataFrame(data)
    xtrain_df = pd.read_csv('x_train.csv')[['Sentence']]
    vectorizer = TfidfVectorizer(min_df=2)
    train_term = vectorizer.fit_transform(xtrain_df['Sentence'].astype('U'))
    test_term = vectorizer.transform(xtest_df['Sentence'].astype('U'))
    with open('LogisticRegression.pkl', 'rb') as f:
        model = pickle.load(f)
    predictions_test = model.predict(test_term)
    predictions_test_df = pd.DataFrame(predictions_test)
    final_df = pd.concat([xtest_df, predictions_test_df], axis=1)
    final_df.to_csv("finalpred.csv")
    negative = final_df[0].value_counts().sort_index()[0]
    positive = final_df[0].value_counts().sort_index()[1]
    output = f'Pecentage of Positive Reviews are "{round(positive*100/(positive+negative),2)}"\n\n'
    output += f'Pecentage of Negative Reviews are "{round(negative*100/(positive+negative),2)}"\n\n\n'

    vect = CountVectorizer(min_df=2,stop_words='english')
    train_term = vect.fit_transform(xtrain_df['Sentence'].astype('U'))
    test_term = vect.transform(xtest_df['Sentence'].astype('U'))
    feature_names = vect.get_feature_names()
    tfidf_mat = pd.DataFrame(test_term.toarray(), columns=feature_names)
    top10 = findtop10uniquesentences(tfidf_mat, xtest_df)
    output += f'Top 10 Reviews are: \n\n'
    for t in top10:
        output += f'"{t}"\n'
    output+=f'\n\n'
    return output

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    text = request.form['experience']
    output = crawldata(text)

    return render_template('index.html', prediction_text='{}'.format(output))

@app.route('/predict1', methods=['POST'])
def predict1():
    '''
    For rendering results on HTML GUI
    '''
    xtrain_df=pd.read_csv('x_train.csv')[['Sentence']]
    vectorizer = TfidfVectorizer(min_df=2)
    train_term = vectorizer.fit_transform(xtrain_df['Sentence'].astype('U'))
    with open('LogisticRegression.pkl', 'rb') as f:
        model = pickle.load(f)

    text = request.form['experience1']

    arr = np.array([text])

    ser = pd.Series(arr)
    ser_test = vectorizer.transform(ser)
    ser_test

    prediction = model.predict(ser_test)
    output=""
    if(prediction[0]==1):
       output="Positive"
    else:
       output="Negative"
    return render_template('index.html', prediction_text1='Your entered review is {}'.format(output))



@app.route('/predict_api', methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)