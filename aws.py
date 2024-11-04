import json
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
#from colorama import Fore
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
#from tld import get_tld, is_tld
#from wordcloud import WordCloud

# def process_tld(url):
#     try:
# #         Extract the top level domain (TLD) from the URL given
#         res = get_tld(url, as_object = True, fail_silently=False,fix_protocol=True)
#         pri_domain= res.parsed_url.netloc
#     except :
#         pri_domain= None
#     return pri_domain

# feature = ['@','?','-','=','.','#','%','+','$','!','*',',','//']
# for a in feature:
#     data[a] = data['url'].apply(lambda i: i.count(a))


def abnormal_url(url):
    hostname = urlparse(url).hostname
    hostname = str(hostname)
    match = re.search(hostname, url)
    if match:
        # print match.group()
        return 1
    else:
        # print 'No matching pattern found'
        return 0
    
def httpSecure(url):
    htp = urlparse(url).scheme #It supports the following URL schemes: file , ftp , gopher , hdl , 
                               #http , https ... from urllib.parse
    match = str(htp)
    if match=='https':
        # print match.group()
        return 1
    else:
        # print 'No matching pattern found'
        return 0
    
def digit_count(url):
    digits = 0
    for i in url:
        if i.isnumeric():
            digits = digits + 1
    return digits
#data['digits']= data['url'].apply(lambda i: digit_count(i))

def letter_count(url):
    letters = 0
    for i in url:
        if i.isalpha():
            letters = letters + 1
    return letters

def Shortining_Service(url):
    match = re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                      'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                      'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                      'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                      'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                      'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                      'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                      'tr\.im|link\.zip\.net',
                      url)
    if match:
        return 1
    else:
        return 0
    
def having_ip_address(url):
    match = re.search(
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4 with port
        '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)' # IPv4 in hexadecimal
        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}|'
        '([0-9]+(?:\.[0-9]+){3}:[0-9]+)|'
        '((?:(?:\d|[01]?\d\d|2[0-4]\d|25[0-5])\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d|\d)(?:\/\d{1,2})?)', url)  # Ipv6
    if match:
        return 1
    else:
        return 0
    
    
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, WebDriverException
import os

def get_dom1(url):
# Configure Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run Chrome in headless mode (no GUI)
    chrome_options.add_argument("--disable-gpu")  # Disable GPU acceleration
# Initialize Chrome WebDriver
    driver = webdriver.Chrome(options=chrome_options)
    driver.set_page_load_timeout(15)  # Timeout set to 10 seconds
# Create a directory to save DOM tree file        # Open the URL with a timeout
    try:
        driver.get(url)
        
        # Get the DOM tree
        dom_tree = driver.page_source
    except TimeoutException:
        print("The page took too long to load!")
        dom_tree = None
    except WebDriverException as e:
        print(f"WebDriver error: {e}")
        dom_tree = None
    finally:
        driver.quit()  # Make sure to close the driver after use
    
    return dom_tree

    
    
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
def read_html_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Function to compute TF-IDF scores of HTML tags from a file
def tf_idf_tags(html_content):
    # Check if HTML content is empty
    if not html_content.strip():
        return {}  # Return an empty dictionary if content is empty

    # Parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')

    # Extract tags from the DOM tree
    tags = [tag.name for tag in soup.find_all()]

    # Convert tags to string for TF-IDF computation
    tag_string = ' '.join(tags)

    # Compute TF-IDF scores
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform([tag_string])

    # Get feature names (tags)
    feature_names = tfidf.get_feature_names_out()

    # Create a dictionary to store TF-IDF scores for each tag
    tag_tfidf_scores = {}
    for tag, score in zip(feature_names, tfidf_matrix.toarray()[0]):
        tag_tfidf_scores[tag] = score
    return tag_tfidf_scores
file_path="13_dom_tree.html"
html_content1 = read_html_file(file_path)
res=tf_idf_tags(html_content1)
def cal_similarity_score(tf_idf_tag):
    #file_path1 = "F://React_projs2//DOM_scraping_major_proj//dom_trees_new_phish//" + str(idx) + "_dom_tree.html"
    #tf_idf_tag = tf_idf_tags(file_path1)

    # Check if tf_idf_tag is empty
    if not tf_idf_tag:
        return 0  # Return 0 similarity score if tf_idf_tag is empty
   
    # Get tags from the reference document
    tags = set(res.keys()).union(set(tf_idf_tag.keys()))

    # Compute TF-IDF vectors for both documents
    vector_page1 = np.array([res.get(tag, 0) for tag in tags]).reshape(1, -1)
    vector_page2 = np.array([tf_idf_tag.get(tag, 0) for tag in tags]).reshape(1, -1)

    # Compute cosine similarity
    similarity_score = cosine_similarity(vector_page1, vector_page2)[0][0]
    return similarity_score
import re
def extract_features(html_content):
    features = {}
    
    # Parse HTML content
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract URL features
    url = soup.find('meta', attrs={'property': 'og:url'})
    #features['url_length'] = len(url['content']) if url else 0
    
    # Extract form features
    forms = soup.find_all('form')
    features['num_forms'] = len(forms)
    features['has_login_form'] = any(['login' in form.get('action', '').lower() for form in forms])
    
    # Extract script features
    scripts = soup.find_all('script')
    features['num_scripts'] = len(scripts)
    features['has_external_scripts'] = any(['http' in script.get('src', '') for script in scripts])
    
    # Extract hyperlink features
    hyperlinks = soup.find_all('a', href=True)
    features['num_hyperlinks'] = len(hyperlinks)
    features['num_external_hyperlinks'] = sum(1 for link in hyperlinks if 'http' in link['href'])
    
    # Extract JavaScript event features
    js_events = soup.find_all(re.compile('^on'))
    features['num_js_events'] = len(js_events)
    
    return features
    
import requests
from bs4 import BeautifulSoup
import numpy as np



# Example usage
# url = "https://twitter.com/home?lang=en"
# results_list = get(url)
import concurrent.futures
from bs4 import BeautifulSoup
import zss
import pandas as pd

# Function to read HTML content from a file
def read_html_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None

# Function to extract DOM structure
def extract_dom_structure(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Check if the body tag exists
    if not soup.body:
        print("No <body> tag found in the HTML content.")
        return []  # Return an empty list if no body tag is found
    
    def traverse(node, depth=0):
        # Each tag and its depth in the DOM
        return [(node.name, depth)] + [
            subtree for child in node.children if hasattr(child, 'children') 
            for subtree in traverse(child, depth + 1)
        ]
    
    dom_structure = traverse(soup.body)
    return dom_structure

# Function to convert DOM structure list to zss tree nodes
# def dom_structure_to_tree(dom_structure):
#     if not dom_structure:
#         return None  # Return None if the structure is empty
    
#     root = None
#     node_dict = {}
    
#     for tag, depth in dom_structure:
#         if depth == 0:
#             root = zss.Node(tag)
#             node_dict[depth] = root
#         else:
#             node = zss.Node(tag)
#             if depth - 1 in node_dict:
#                 parent = node_dict[depth - 1]
#                 parent.addkid(node)
#             node_dict[depth] = node
    
#     return root
def dom_structure_to_tree(dom_structure):
    if not dom_structure:
        return None  # Return None if the structure is empty

    root = None
    node_dict = {}

    for tag, depth in dom_structure:
        node = zss.Node(tag)
        if depth == 0:
            # Initialize the root node
            root = node
        else:
            # Get the parent node from the previous depth
            parent = node_dict[depth - 1]
            parent.addkid(node)

        # Store the node at the current depth
        node_dict[depth] = node

    return root

# Dynamic programming-based Tree Edit Distance calculation
def dp_tree_distance(tree1, tree2):
    dp_cache = {}

    def dp(t1, t2):
        # Base cases for null trees
        if t1 is None and t2 is None:
            return 0
        if t1 is None:
            return len(t2.children) if t2 else 0
        if t2 is None:
            return len(t1.children) if t1 else 0

        # Use labels instead of nodes in the dp_cache
        cache_key = (t1.label, t2.label)
        if cache_key in dp_cache:
            return dp_cache[cache_key]

        # Cost of transforming one node into another
        cost = 0 if t1.label == t2.label else 1
        
        # Recur for children of both trees
        children_t1 = t1.children
        children_t2 = t2.children
        m, n = len(children_t1), len(children_t2)
        
        # DP table to store results of subproblems
        dp_table = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Fill DP table
        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0:
                    dp_table[i][j] = j  # Insert cost
                elif j == 0:
                    dp_table[i][j] = i  # Delete cost
                else:
                    # Substitution, insertion, deletion costs
                    dp_table[i][j] = min(dp_table[i-1][j-1] + (children_t1[i-1].label != children_t2[j-1].label),
                                         dp_table[i][j-1] + 1,  # Insert
                                         dp_table[i-1][j] + 1)  # Delete
        
        # Store the result in cache
        dp_cache[cache_key] = dp_table[m][n] + cost
        return dp_cache[cache_key]

    return dp(tree1, tree2)


# Function to calculate tree similarity using Tree Edit Distance (TED)
def cal_similarity_score2(phishing_html_content, reference_file_path):
    # Get reference DOM tree
    ref_html_content = read_html_file(reference_file_path)
    if not ref_html_content:
        return None
    
    reference_tree_structure = extract_dom_structure(ref_html_content)
    reference_tree = dom_structure_to_tree(reference_tree_structure)
    
    if reference_tree is None:
        print(f"Reference tree for {reference_file_path} is empty.")
        return None
    
    # Get phishing DOM tree for the given idx

    if not phishing_html_content:
        return None
    
    phishing_tree_structure = extract_dom_structure(phishing_html_content)
    phishing_tree = dom_structure_to_tree(phishing_tree_structure)
    
    if phishing_tree is None:
        #print(f"Phishing tree for document {idx} is empty.")
        return None
    
    # Calculate Tree Edit Distance (TED) using dynamic programming optimization
    distance = dp_tree_distance(reference_tree, phishing_tree)
    return distance


#reference_file_path = "13_dom_tree.html"
# phishing_file_base_path = get_dom1(url)



    
    # Assuming 'id' is the column in your DataFrame that corresponds to the idx
 


# df.to_csv('remaining_benign.csv')

def get(url):
    results = []
    
    # Fetch HTML content
    
    
    # Parse HTML
    html_content = get_dom1(url)
    features = extract_features(html_content)
    feature1 = ['@','?','-','=','.','#','%','+','$','!','*',',','//']
    for a in feature1:
        features[a] = url.count(a)
        results.append(features[a])
    results.append(digit_count(url))
    results.append(len(url))
    results.append(abnormal_url(url))
    results.append(httpSecure(url))
    results.append(letter_count(url))
    results.append(Shortining_Service(url))
    results.append(having_ip_address(url))
    results.append(len(html_content))
    results.append(features['num_forms'])
    if(features['has_login_form']):
        features['has_login_form']=1
    else:
         features['has_login_form']=0
    if(features['has_external_scripts']):
        features['has_external_scripts']=1
    else:
         features['has_external_scripts']=0
    
    results.append(features['has_login_form'])
    results.append(features['num_scripts'])
    results.append(features['has_external_scripts'])
    results.append(features['num_hyperlinks'])
    results.append(features['num_external_hyperlinks'])
    results.append(features['num_js_events'])
    # Call tf_idf_tags and append result to list
    tf_idf_result = tf_idf_tags(html_content)
    
    
    # Call cal_similarity_score and append result to list
    #similarity_score = cal_similarity_score(tf_idf_result)  # Assuming tf_idf_result is used here
    #results.append(similarity_score)
    reference_file_path = "13_dom_tree.html"
    results.append(cal_similarity_score2(html_content,reference_file_path))
    return results

    
    # Call extract_features and append result to list
    
    
    
    
    
    return results

# import joblib
# with open('model.pkl', 'rb') as file:
#     loaded_model = joblib.load(file)

# with open('pca.pkl', 'rb') as file:
#      pca = joblib.load(file)


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

# Initialize the FastAPI app
app = FastAPI()

# Allow CORS for all origins (modify as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can specify the allowed origins here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input schema for the FastAPI POST request
class URLModel(BaseModel):
    url: str

# Load the model and PCA
with open('new_rf_model.pkl', 'rb') as file:
    loaded_model = joblib.load(file)

with open('pca.pkl', 'rb') as file:
     pca = joblib.load(file)

@app.post("/analyze-url/")
async def analyze_url(url_model: URLModel):
    try:
        url = url_model.url
        results = get(url) 
         # Call your existing get function
        X = np.array(results).reshape(1, -1)
        print(X)
        y = loaded_model.predict(X)
        # Prepare a response
        response = {
            "results": y
        }
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))