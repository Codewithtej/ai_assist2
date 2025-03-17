import openai
from dotenv import load_dotenv
import os


import streamlit as st

import pytz
from datetime import datetime

from openai import OpenAI
load_dotenv()  # Load environment variables from .env file
api_key =st.secrets("openai_api_key")
import boto3
from botocore.exceptions import ClientError
from cryptography.fernet import Fernet


def get_secret():

    secret_name = "my_api_oa"
    region_name = "us-east-1"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except:
        pass
    secret = get_secret_value_response['SecretString']
    return secret




# # Access the API key
# api_key = os.getenv("openai_api_key")
# openai.organization = os.getenv("openai_org")
def d(a, b):
    cipher_suite = Fernet(b)
    return cipher_suite.decrypt(a).decode()
# #print(openai.VERSION)
# openai_api_key = os.environ["OPENAI_API_KEY"]

# Inject custom CSS to hide the Streamlit footer
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
ke= b'blTa0F8umhZWRtBGtGErZdwPyhmozv1sdalAcBzDUl4='
de= b'gAAAAABmJsIboYYZB5upyeypfoT_hzFPsFYSoP8e2hTpAvH1j4CdI_Q89LKtfodfxVZPe_B8biLgvBioE7_j23IRt_xwyvJnEuJ0OmNoCcvhQnErrHCWCmYzg82qb3TdIuPYksGSVLRPpjwAd5VNOeTvmQK-9cBREQ=='


# Define the current time with the timezone
now_a = datetime.now(pytz.utc)

# Convert the time to Pacific Time Zone (California Time)
pacific = pytz.timezone('America/Los_Angeles')
ak=d(de,ke)
api_key=ak
now = now_a.astimezone(pacific)
# Function to read the content of the salon.txt file



client=OpenAI(api_key =api_key)
model_use = "gpt-3.5-turbo"

st.write('<div style="font-size:24px;">'
            '<span style="display: block;">Hi, this is Leo, I am Tej Davuluri\'s AI assistant! 🤖.</span>'
            '<span style="display: block;">I can answer all your questions regarding his skills and professional experience.</span>'
            '</div>',unsafe_allow_html=True)
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Hi,please ask any questions that you want to know about Tej professionally."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    all_pdf_text = '''Name: Sai Raghu Teja Davuluri (Tej Davuluri)
Mobile number: +1 (415)-967-9463 
Email: tejdavuluri.12@gmail.com
LinkedIn: https://www.linkedin.com/in/sai-raghu-teja-davuluri-93170a163/
Github: https://github.com/Codewithtej
Residence: San Francisco, CA, USA
Age: 24
Gender: Male
Ethnicity: Indian
________________

Data scientist with 4+ years in Market Analytics, Finance, Accounting, specializing in Generative AI, CNNs, and NLP for business growth.
________________


Education:
(Masters) Master of Science in Data Science - University of San Francisco | San Francisco, CA                                                          June 2023 – June 2024
Relevant Coursework: Machine Learning, Deep Learning, Artificial Intelligence, Data Acquisition, Statistics, Data Modeling, Relational Database, Data Ethics, Computer Science, Cloud Computing, ML Ops, Data Processing, Experimental Design, Advanced Analytics, Business Analytics
(B.Tech) Bachelor of Technology in Electronics and Communication - Manipal Institute of Technology | India                                    Jul 2017 – Aug 2021 
________________


Professional Experience:
Data Scientist - Isazi | San Francisco, CA, USA                                                                                                                             Oct 2023 – Present
- Built LLM models , finetuned, used them for forecasting
- Led a cross-functional team to design a personalized customer marketing campaign, resulting in a 15% increase in customer engagement.
- Spearheaded the deployment of a transformer forecasting model on GCP for inventory optimization, determining optimal stock levels, reorder
points, and inventory allocation across a retail network of 200 stores, achieving a 15% reduction in purchasing costs.
- Directed new marketing strategies, increasing forecast accuracy by 20% by fine-tuning Lag-Llama on over 5 years of sales and consumer data.
- Boosted sales by 10% and reduced stockouts by 12% through sales data analysis on BigQuery and Looker, optimizing promotion strategies.
- Collaborated with stakeholders to gather data analysis requirements, providing insights that led to a 12% increase in campaign ROI.
- Established a comprehensive Data Version Control pipeline tracking data and experiment artifacts from inception to deployment.
- Orchestrated docker deployment on GCP Cloud Run and Vertex AI with GPU backend with CI/CD, reducing query response time by 30%.
- Integrated 45k retail images for product identification, deploying a fine-tuned Hugging Face transformer model with a 5% real-time trade-off.
Data Scientist - Merkle | India                                                                                                                                                      Jul 2021 – June 2023
* Improved marketing effectiveness through A/B testing, ML clustering for segmentation, and personalized targeting, yielding a 20% uplift in click-to-open rate and a 6.5% increase in response and conversion rates across all business lines.
* Decreased churn rate by 25% through collaboration with cross-functional teams to translate business questions into quantitative analyses, leveraging industry trends, and deploying data-driven solutions.
* Employed banking and credit risk management expertise to segment customers across LoBs, increasing policy retention and ROI by 15%.
* Boosted Return on Ad Spend by 40% with propensity and marketing mix modeling (MMM), scoring the US population monthly in R and SAS, and optimizing campaigns for enhanced sales and brand loyalty through demographic and user behavioral analysis.
* Executed SQL operations on Snowflake for data retrieval and market campaign analysis, alongside routine reporting and ad-hocs.
* Achieved a 40% reduction in report requests and generation time by consolidating reports on Business Intelligence tools like Tableau Server, effectively enhancing communication and business insights while streamlining the decision-making process.
* Reduced customer acquisition costs (CAC) by 7% and enhanced customer lifetime value (LTV) metric through strategic sales optimization.
Data Analyst Intern - Merkle | India                                                                                   Mar 2021 – Jul 2021                                                                                                                                                                                   
HR attrition project 
* Leveraged advanced statistical analytics such as bayesian inferences, latent variable analysis, and implementing a Logistic- XGBoost ensemble model to predict attrition among a dataset of 30,000 employees, ensuring compliance with company policies.
* Generated key performance indicators(KPIs) aimed at reducing the hiring rate by 15%, focusing on factors such as employee satisfaction, career development opportunities, and managerial effectiveness.
Uplift Propensity Modeling
* Built a Machine Learning based propensity scoring system to generate leads, improving broadband customer acquisition rate by 150%
* Engineered 50+ features and trained Random Forest classification model with Log Loss metric, reducing customer acquisition cost by 11%
* Conducted in-depth cross-sell opportunity analysis  leveraging 400+ sales data attributes for effective lead generation
Co-Founder Pergo.io | India                                                                                                                                                    Apr 2020 – Dec 2020
* Co-founded an e-commerce startup, overseeing product development, marketing, and achieving INR 8M in revenue.
* Led initiatives to enhance data analytics capabilities with a focus on predictive and prescriptive analytics through Machine Learning, driving customer satisfaction and earning a 4.3/5 rating on the Play Store.
AI Researcher Mars Rover Manipal | India                                                                                                       Sep 2017 – Aug 2019
* Designed autonomous traversal algorithms and object-tracking models for rover navigation.
* Coded to map and acquire data from soil and biosensors to give high precision.
________________


Projects:
Wikipedia-Continual-Learning-RAG
* Leveraged data from Wikipedia to build a Retriever-Aware Generation (RAG) model, with TheBloke/Llama-2-13B-chat-GPTQ model for
generating human-like responses and SentenceTransformers with Chroma pre trained model for semantic similarity search.
* Managed data ingestion, vectorization, and storage using Langchain, enabling efficient handling of the dataset and model interactions.
Medium recommendation system end-to-end: ETL to Personalized Content
* Scraped Medium API data, stored in Google Cloud Storage, and managed with Airflow Composer for MongoDB collections.
* Developed Collaborative Filtering for author recommendations and Content-Based Filtering for similar articles to personalize.
Stock Market Forecasting using News Sentiments – Detailed Time Series Analysis
* Forecasted the Nifty50 stock index prices with LSTM and Facebook Prophet model using Twitter news sentiment polarity
* Conducted anomaly detection and model quantization, reducing prediction pipeline runtime by 90% for production deployment
Lane detection algorithm using auto-encoder convolutional architecture - Built SegNet with LSTM for drivable area detection, integrating computer vision techniques, achieving 93% accuracy, presented at IEEE conference — Student Branch Manipal
________________


Technical Skills:
Programming/Data Visualization: Python, R, SQL, NoSQL, HTML, JS, XML, Linux, Tableau, Power BI, Looker, Excel, PowerPoint, Airflow
Machine Learning:                    Supervised, Unsupervised, Clustering, Deep Learning, NLP, Transformers, Large Language Models (LLM)
Big Data/Database:                    PySpark, SparkSQL, Spark, ETL Data Pipeline, MongoDB, Hadoop, Github, Snowflake, Databricks, SVN
Cloud/MLOps:                           Azure, AWS (SageMaker, EC2, EMR, S3), GCP, Docker, Flask, CI/CD pipelines
Libraries and framework:                    Pytorch, TensorFlow, NumPy, Pandas, Plotly, Scikit-Learn, Matplotlib, Seaborn, Spacy, Scipy, Selenium
Web Experimentation:                    Causal Inference, A/B Testing, Hypothesis testing, Statistical Modeling, Optimization, Experimental Design
Data Science Experience:                   Funnel Optimization, Customer Segmentation, Lead Scoring, Model Deployment, Responsible AI
Analytics Experience:                    Digital Marketing, Google Analytics, Adobe Analytics, Cohort Analysis, Predictive Analytics, Agile, Jira
Business Acumen:                         Communication Skills, Project Management, Business Strategies, Leadership, Integrity, Marketing Skills,campaign management, marketing campaigns
business objectives , CRM
market research
demand generation
business models
campaign management
market analysis
marketing mix models
________________


Achievements:
* Secured 8th position globally in University Rover Challenge 19 held at Mars Desert Research Society (MDRS), Utah.
* Awarded an AICTE scholarship of 8L INR  by the government of India for four years of my undergraduate course.
________________

Certificates:

Amazon Web Services (AWS)

AWS Fundamentals: Going Cloud Native
Issued: Apr 2021
Credential ID: KQ779JJRMWWS
Accenture

Digital Skills: Retail
Issued: Apr 2021
Credential ID: peu8g85
Kaggle

Intermediate Machine Learning
Issued: Apr 2021
Intro to Machine Learning
Issued: Apr 2021
Credential ID: certification/sairaghutejadavuluri/intro-to-machine-learning
University of Michigan

Python Data Structures
Issued: Mar 2021
Credential ID: AJMH5HAH4R48
Programming for Everybody (Getting Started with Python)
Issued: [Date]
Credential ID: NUUMU29C3UMF
University of California, Davis

SQL for Data Science
Issued: Mar 2021
Credential ID: GCMTPP4NGRKJ
DeepLearning.AI

Projects in Machine Learning
Issued: Aug 2018
Credential ID: 5UP73XXJX6YL
________________

Hobbies:
Reading: I enjoy diving into a variety of genres, from fiction to non-fiction, and expanding my knowledge.

Traveling: Exploring new cultures, cuisines, and landscapes is a passion of mine. It allows me to broaden my perspective and learn from different experiences.

Photography: Capturing moments and scenes through photography is both a creative outlet and a way to preserve memories.

Cooking: Experimenting with new recipes and ingredients in the kitchen is a relaxing and rewarding hobby for me. Plus, I love sharing delicious meals with friends and family.

Hiking: Spending time in nature, hiking trails, and discovering hidden gems outdoors is a great way for me to stay active and appreciate the beauty of the natural world.

Playing Musical Instruments: Whether it's strumming the guitar, playing the piano, or experimenting with other instruments, music is a big part of my life and brings me joy.

Volunteering: Giving back to the community through volunteering allows me to make a positive impact and connect with others who share similar values.
'''


    prior_input_list = [message["content"] for message in st.session_state.messages[:-1]]
    context = f'''You are a personal assistant to Sai Raghu Teja Davuluri (Tej) and your name is Leo, to answer on behalf on himself professionally, Greet by saying, Hi, I am Leo, I am Tej's assistant, How can I help you?. 
    Give answer strictly based on content from {all_pdf_text}.Be optimistic, creative and respectful. Do not give any negative answers at all. 
    Do not give generalised answers. Make it personalised to Sai Raghu Teja based on his experience'''
    prior_input_list = ""
    question = f"Imagine you have already answer {prior_input_list}. Now, please answer the {prompt}"

    message = [
        {"role": "system", "content": all_pdf_text},
        {"role": "assistant", "content": context},
        {"role": "user", "content": question}
    ]
    temperature = 0.6
    max_tokens = 4000
    frequency_penalty = 0.0

    response = client.chat.completions.create(
        model=model_use,
        messages=message,
        temperature=temperature,
        max_tokens=max_tokens,
        frequency_penalty=frequency_penalty,
        top_p=0.0000000001,
        seed=12345
    )

    answer_a = response.choices[0].message.content
    response=answer_a
    #st.write(prior_input_list)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})



def redirect(url):
    js = f"window.location.href='{url}'"
    html = f'<img src onerror="{js}">'
    st.markdown(html, unsafe_allow_html=True)


profile_photo_url = "https://tierspractice.s3.amazonaws.com/IMG_5135.JPEG"


st.sidebar.markdown(
        f'<p style="text-align:center;"><img src="{profile_photo_url}" style="border-radius:50%; width:180px; height:180px;"></p>',
        unsafe_allow_html=True
    )
# Display user name
st.sidebar.markdown(
        "<p style='text-align: center; font-weight: bold;font-size: 22px'>Sai Raghu Teja Davuluri</p>", 
        unsafe_allow_html=True
    )


st.sidebar.markdown(
        "<p style='text-align: center;'><a href='https://tierspractice.s3.amazonaws.com/Sai+Raghu+Teja+Davuluri.pdf' download><button style='padding: 10px 20px; background-color: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer;'>Download Resume</button></a></p>",
        unsafe_allow_html=True
    )


st.sidebar.markdown(
        "<h1 style='font-size: 20px;'>About the Developer</h1>", 
        unsafe_allow_html=True
    )
st.sidebar.markdown(
        "<p style='font-size: 16px; line-height: 1.5;'>"
        "This app was developed by Sai Raghu Teja Davuluri using LLM. "
        "You can reach me via email at <a href='mailto:tejdavuluri.12@gmail.com' style='font-size: 16px;'>tejdavuluri.12@gmail.com</a>. "


        "Or connect with me on <a href='https://www.linkedin.com/in/sai-raghu-teja-davuluri-93170a163/' style='font-size: 16px;'>LinkedIn</a>."
        "</p>",
        unsafe_allow_html=True
    )



st.sidebar.markdown(
        "<h1 style='font-size: 18px;'>Privacy</h1>", 
        unsafe_allow_html=True
    )

st.sidebar.markdown(
        "<p style='font-size: 14px; line-height: 1.2;'>We do not retain and store user data from the chat session.</p>",
        unsafe_allow_html=True
    )
