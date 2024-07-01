## importing all the libraries

from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain.prompts import PromptTemplate
from flask import Flask, request, jsonify

from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
import re
import langchain

from langchain.agents import create_react_agent
from langchain_community.llms import Ollama
from langchain_community.tools import DuckDuckGoSearchRun

import pandas as pd
import pymongo
from pymongo import MongoClient
from datetime import datetime
import matplotlib.pyplot as plt

from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.memory import ConversationBufferMemory

from langchain.agents import initialize_agent
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.agents import AgentExecutor
from langchain_core.messages import HumanMessage

def get_stock_price(input):
    ''' Python Function which gets price of a stock on a particular date'''

    stock_name= input.split(',')[0]
    date= input.split(',')[1]
    stock_id = ""
    for i in stock_name:
        if i.islower():
            stock_id += i.upper()
        elif i.isupper():
            stock_id += i
        else:
            pass
    y,m,d = date.split('-')
    client = MongoClient('mongodb://intelliinvest:intelliinvest@67.211.219.52:27017/intelliinvest')
    db = client['intelliinvest']
    collection = db['STOCK_FUNDAMENTALS']
    target_date = datetime(int(y), int(m),int(d), 18, 30, 0)  # Example target date (replace with your date)

    # Query the collection for documents with the specified date
    query_result = collection.find({'todayDate': target_date,'securityId':stock_id})

    for document in query_result:
        return document['closePrice']

def find_top_stocks(industry):
    '''Python function to find top stocks in a particular industry'''

    s = ""
    for i in industry:
        if i.islower():
            s += i.upper()
        elif i.isupper():
            s += i
        else:
            pass
    industry_mapping = {
        'ABRASIVES':'Abrasives',
        'AGRI':'Agri',
        'AGRICULTURE':'Agri',
        'ALCOHOL':'Alcohol',
        'AUTOMOBILE' :'Automobile & Ancillaries',
        'AUTO' :'Automobile & Ancillaries',
        'AVIATION':'Aviation',
        'BANK':'Bank',
        'BANKING':'Bank',
        'BUSINESSSERVICES':'Business Services',
        'CAPITALGOODS':'Capital Goods',
        'CHEMICAL': 'Chemicals',
        'CHEMICALS': 'Chemicals',
        'CONSTRUCTION': 'Construction',
        'CONSUMERDURABLES':'Consumer Durables',
        'CRUDEOIL':'Crude Oil',
        'OIL':'Crude Oil',
        'DIVERSIFIED': 'Diversified',
        'DIAMOND':'Diamond  &  Jewellery',
        'EDUCATION':'Education & Training',
        'EDTECH':'Education & Training',
        'ELECTRICAL':'Electricals',
        'ELECTRICALS':'Electricals',
        'ETF':'ETF',
        'FMCG':'FMCG',
        'FINANCE':'Finance',
        'FORESTMATERIALS':'Forest Materials',
        'GASTRANSMISSION':'Gas Transmission',
        'HEALTHCARE': 'Healthcare',
        'HOSPITALITY':'Hospitality',
        'INFRASTRUCTURE':'Infrastructure',
        'INSURANCE':'Insurance',
        'INFORMATIONTECHNOLOGY':'IT',
        'IT':'IT',
        'JEWELLERY':'Diamond  &  Jewellery',
        'LOGISTICS':'Logistics',
        'LOGISTIC':'Logistics',
        'MEDIA': 'Media & Entertainment',
        'ENTERTAINMENT': 'Media & Entertainment',
        'METAL': 'Iron & Steel',
        'METALS': 'Iron & Steel',
        'MINING':'Mining',
        'IRON': 'Iron & Steel',
        'STEEL': 'Iron & Steel',
        'FUELS': 'Inds. Gases & Fuels',
        'GAS': 'Inds. Gases & Fuels',
        'PAPER':'Paper',
        'PLASTIC':'Plastic Products',
        'PHOTOGRAPHIC':'Photographic Product',
        'POWER': 'Power',
        'RATING':'Ratings',
        'RATINGS':'Ratings',
        'REALTY': 'Realty',
        'SHIP':'Ship Building',
        'SHIPBUILDING':'Ship Building',
        'SERVICES': 'Services',
        'RETAIL':'Retailing',
        'RETAILS':'Retailing',
        'TRADING':'Trading',
        'TELECOMMUNICATION': 'Telecom',
        'TELECOM':'Telecom',
        'TEXTILES': 'Textile',
        'TEXTILE': 'Textile'
    }

    client = MongoClient('mongodb://intelliinvest:intelliinvest@67.211.219.52:27017/intelliinvest')
    db = client['intelliinvest']
    collection = db['STOCK_FUNDAMENTALS']
    ## this date should be changed for production data
    target_date = datetime(2024, 6, 6, 18, 30, 0)

    query = {"industry":industry_mapping[s],'todayDate':target_date}
    results = collection.find(query).sort("alMarketCap", -1)
    unique_ids = []
    i = 0
    for result in results:
        unique_ids.append(result['securityId'])
        i+=1
        if i == 20:
            break


    collection = db['INTELLII_SIGNAL']

    query = {"securityId": {"$in": list(unique_ids)},'signalDate':target_date}

    documents = collection.find(query).sort("buyWeightage", -1)

    li = []
    i = 0
    for doc in documents:
        li.append(doc['securityId'])
        i+=1
        if i == 3:
            break
    return li

def get_stock_parameter(input):
    '''Python function to answer a particular detail about the stock'''

    param_mapping = {
        "MARKETCAP":'alMarketCap',
        'BOOKVALUEPERSHARE':'alBookValuePerShare',
        'EARNINGPERSHARE':'alEarningPerShare',
        'PRICETOEARNINGRATIO':'alPriceToEarning',
        'PERATIO':'alPriceToEarning',
        'PE':'alPriceToEarning',
        'CASHTODEBTRATIO':'alCashToDebtRatio',
        'EQUITYTOASSETRATIO':'alEquityToAssetRatio',
        'DEBTTOCAPITALRATIO':'alDebtToCapitalRatio',
        'RETURNONEQUITY':'alReturnOnEquity',
        'EBIDTAMARGIN':'qrEBIDTAMargin',
        'EBIDTA':'qrEBIDTAMargin',
        'OPERATINGMARGIN':'qrOperatingMargin',
        'NETMARGIN':'qrNetMargin',
        'DIVIDENTPERCENT':'qrDividendPercent',
        'DIVIDENT':'qrDividendPercent'

    }
    stock_name = input.split(',')[0]
    param = input.split(',')[1]
    stock_id = ""
    for i in stock_name:
        if i.islower():
            stock_id += i.upper()
        elif i.isupper():
            stock_id += i
        else:
            pass
    param_id = ""
    for i in param:
        if i.islower():
            param_id += i.upper()
        elif i == " " or i == '_':
            pass
        else:
            param_id += i

    client = MongoClient('mongodb://intelliinvest:intelliinvest@67.211.219.52:27017/intelliinvest')
    db = client['intelliinvest']
    collection = db['STOCK_FUNDAMENTALS']
    ## this date should be changed for production data
    target_date = datetime(2024, 4, 1, 18, 30, 0)

    # Query the collection for documents with the specified date
    query_result = collection.find({'todayDate': target_date,'securityId':stock_id})
    ans = param_mapping[param_id]
    for document in query_result:
        return document[ans]

def get_stock_fundamentals(input):
    ''' Python function which shows the fundamental details of a stock '''

    param_mapping = {
        "MARKET CAP":'alMarketCap',
        'BOOK VALUE PERSHARE':'alBookValuePerShare',
        'EARNING PER SHARE':'alEarningPerShare',
        'PRICE TO EARNING RATIO':'alPriceToEarning',
        'CASH TO DEBT RATIO':'alCashToDebtRatio',
        'EQUITY TO ASSET RATIO':'alEquityToAssetRatio',
        'DEBT TO CAPITAL RATIO':'alDebtToCapitalRatio',
        'RETURN ON EQUITY':'alReturnOnEquity',
        'EBIDTA MARGIN':'qrEBIDTAMargin',
        'OPERATING MARGIN':'qrOperatingMargin',
        'NET MARGIN':'qrNetMargin',
        'DIVIDENT PERCENT':'qrDividendPercent'

    }

    stock_name = input

    stock_id = ""
    for i in stock_name:
        if i.islower():
            stock_id += i.upper()
        elif i.isupper():
            stock_id += i
        else:
            pass

    client = MongoClient('mongodb://intelliinvest:intelliinvest@67.211.219.52:27017/intelliinvest')
    db = client['intelliinvest']
    collection = db['STOCK_FUNDAMENTALS']
    ## this date should be changed for production data
    target_date = datetime(2024, 4, 1, 18, 30, 0)

    # Query the collection for documents with the specified date
    query_result = collection.find({'todayDate': target_date,'securityId':stock_id})
    li=[]
    for document in query_result:
        for i in param_mapping:
            li.append(f"{i} : {document[param_mapping[i]]}")

    return li



def get_price_visualizations(input):
    ''' Python function to visualize a stock over a period of time '''
    if ',' in input:
        stock_name= input.split(',')[0]
        n= input.split(',')[1]
        n_records = int(n)
    else:
        stock_name = input
        n_records = 30
    stock_id = ""
    for i in stock_name:
        if i.islower():
            stock_id += i.upper()
        elif i.isupper():
            stock_id += i
        else:
            pass
    print("Visualization of",stock_id,'of last',n_records,'days')
    client = MongoClient('mongodb://intelliinvest:intelliinvest@67.211.219.52:27017/intelliinvest')
    db = client['intelliinvest']
    collection = db['STOCK_FUNDAMENTALS']


    cursor = collection.find({'securityId': stock_id}).sort('_id', -1).limit(n_records)

    li = []
    for document in cursor:

        li.append([str(document['todayDate']).split(' ')[0],document['closePrice']])

    df = pd.DataFrame(li,columns=['Date','closePrice'])
    df['Date'] = pd.to_datetime(df['Date'])

    df =df.sort_values(by='Date', ascending=True)
    if len(df) == 0:
        return "No data found for this"
    plt.figure(figsize=(10, 6))
    plt.plot(df['Date'], df['closePrice'], marker='o', linestyle='-')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('Close Price Over Time')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    return plt.show()


def find_top_performing_industries(input):
    '''Python function to find top performing industries'''

    client = MongoClient('mongodb://intelliinvest:intelliinvest@67.211.219.52:27017/intelliinvest')
    db = client['intelliinvest']
    collection = db['INDUSTRY_FUNDAMENTAL_ANALYSIS']
    ## this date should be changed for production data
    target_date = datetime(2024, 4, 17, 18, 30, 0)
    query = {'signalDate':target_date}

    results = collection.find(query)
    sorted_data = sorted(results, key=lambda x: x['quality'], reverse=True)

    top_performing_industry = sorted_data[:5]
    li = []
    for industry in top_performing_industry:
        li.append(industry['name'])

    return li



def get_intellii_signal(input):
    '''Python function to get buy/sell signal'''

    client = MongoClient('mongodb://intelliinvest:intelliinvest@67.211.219.52:27017/intelliinvest')
    db = client['intelliinvest']
    collection = db['INTELLII_SIGNAL']
    stock_id = ""
    for i in input:
        if i.islower():
            stock_id += i.upper()
        elif i.isupper():
            stock_id += i
        else:
            pass

    ## this date should be changed for production data
    target_date = datetime(2024, 4, 15, 18, 30, 0)
    query = {'securityId':stock_id,'signalDate':target_date}

    # Use the find() method to fetch data based on the query

    results = collection.find(query)
    for r in results:
        return r['intelliiSignal'] + ' becuase of buyweightage'



def merge(li):
    '''python function to answer merge answer of strategies '''
    ans = ''
    for i in li:
        ans += i + " "
    return ans

def get_merged_strategy_dataframe():
    '''python function to merge strategy dataframes'''

    client = MongoClient('mongodb://intelliinvest:intelliinvest@67.211.219.52:27017/intelliinvest',datetime_conversion='DATETIME_AUTO')
    db = client['intelliinvest']
    collection_1 = db['STRATEGY_CONFIGURATION']
    collection_2 = db['PORTFOLIO_SUMMARY']
    # Specify the desired columns to fetch
    columns_to_fetch = ['userId', "portfolioName", 'visibility' ,'riskType']
    # Use the find() method with projection to efficiently retrieve specific columns
    projection = {column: 1 for column in columns_to_fetch}  # Include only desired columns
    results = collection_1.find({'visibility': 'PUBLIC'}, projection=projection)
    # Efficiently convert results to a pandas DataFrame
    df_1 = pd.DataFrame(list(results))

    desired_columns = ["userId", "portfolioName", "pnlPercent","displayName"]

    # Use the find() method with projection to select specific columns
    cursor = collection_2.find({}, projection={col: 1 for col in desired_columns})

    # Convert the cursor to a pandas DataFrame
    df_2 = pd.DataFrame(list(cursor))
    final_df = pd.merge(df_1,df_2,on = ['userId','portfolioName'],how='inner')
    result_df = pd.DataFrame(final_df)
    return result_df

def get_strategy_recommendation(input):
    '''Python function to recommend strategy'''

    df = get_merged_strategy_dataframe()
    df_0 = df[df['riskType'] == '0'].sort_values(by='pnlPercent', ascending=False)
    df_1 = df[df['riskType'] == '1'].sort_values(by='pnlPercent', ascending=False)
    df_2 = df[df['riskType'] == '2'].sort_values(by='pnlPercent', ascending=False)
    ans_0 = str(df_0.iloc[0:1]['displayName']).split()[1:4]
    ans_1 = str(df_1.iloc[0:1]['displayName']).split()[1:4]
    ans_2 = str(df_2.iloc[0:1]['displayName']).split()[1:4]

    a1 = merge(ans_0)
    a2 = merge(ans_1)
    a3 = merge(ans_2)


    return f"High Risk Strategy : {a1} , Medium Risk Strategy : {a2} , Low Risk Strategy : {a2}"

def get_high_risk_strategy(input):
    '''Python function to find best performing high risk strategies'''
    df = get_merged_strategy_dataframe()
    df_0 = df[df['riskType'] == '0'].sort_values(by='pnlPercent',ascending=False)
    return merge(str(df_0.iloc[0:1]['displayName']).split()[1:4]) +" , "+ merge(str(df_0.iloc[1:2]['displayName']).split()[1:4])

def get_medium_risk_strategy(input):
    '''Python function to find best performing medium risk strategies'''
    df = get_merged_strategy_dataframe()
    df_1 = df[df['riskType'] == '1'].sort_values(by='pnlPercent', ascending=False)
    return merge(str(df_1.iloc[0:1]['displayName']).split()[1:4]) +" , "+merge(str(df_1.iloc[1:2]['displayName']).split()[1:4])

def get_low_risk_strategy(input):
    '''Python function to find best performing low risk strategies'''
    df = get_merged_strategy_dataframe()
    df_2 = df[df['riskType'] == '2'].sort_values(by='pnlPercent', ascending=False)
    return merge(str(df_2.iloc[0:1]['displayName']).split()[1:4]) +" , "+ merge(str(df_2.iloc[1:2]['displayName']).split()[1:4])

def format_stock_name(input):
    ''' python function to get a proper format of stock name'''
    stock_id = ""
    for i in input:
        if i.islower():
            stock_id += i.upper()
        elif i.isupper():
            stock_id += i
        else:
            pass

    return stock_id

def compare_two_stocks(input):
    '''Python function to compare two stocks '''
    client = MongoClient('mongodb://intelliinvest:intelliinvest@67.211.219.52:27017/intelliinvest')
    db = client['intelliinvest']
    collection = db['INTELLII_SIGNAL']
    s1,s2 = input.split(',')
    s1 = format_stock_name(s1)
    s2 = format_stock_name(s2)
    ## this date should be changed for production data
    target_date = datetime(2024, 6, 6, 18, 30, 0)

    query_1 = {'securityId':s1,'signalDate':target_date}
    query_2 = {'securityId':s2,'signalDate':target_date}

    results_1 = collection.find(query_1)
    for r in results_1:
        ans1 = r['buyWeightage']

    results_2 = collection.find(query_2)
    for r in results_2:
        ans2 = r['buyWeightage']

    if ans1 > ans2:
        return s1 + ',' + 'as it has higher buy weightage that is ' + ans1
    else:
        return s2 + ',' + 'as it has higher buy weightage that is ' + ans2



'''prompt_template="""
You are very smart AI assistant of IntelliInvest who is very good in explaining and expressing answer related to Indian Stock Market
user will ask question as {question}
you must give answer with detailed explanation with bullets, heading , subheading etc. based on the {context} only
dont use anything else other than the given {context}
if no related information found from the {context} just reply with "I dont know", this is very important
answer:
answer:
"""
'''
prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant of Intelliinvest. You may not to use tools for every query - the user may just want to chat!",
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
search = DuckDuckGoSearchRun()

ollama = Ollama(model='llama3')
memory = ConversationBufferMemory(max_length=200)


tools=[
    Tool(
        name="find whether to buy or sell",
        func=get_intellii_signal,
        description="Use this to answer whether to buy or purchase or sell a stock. Use the stock name as input.Give a proper output with prompt example if buy the say yes it is a good time or else say no it is not."
    ),
    Tool(
        name="get stock price",
        func=get_stock_price,
        description=" pass stock name and date in format YYYY-MM-DD as input and return the value from the function"
    ),
    Tool(
        name="top industry stocks",
        func=find_top_stocks,
        description="Use this to answer the top stocks in a particular industry. Use industry name as input. Give a proper answer with good prompt"
    ),
    Tool(
        name="get fundamental parameter",
        func=get_stock_parameter,
        description="Use this to answer any specific detail except price realted to stock. Use stock name , parameter as a single string input."
    ),
    Tool(
        name="suggest high risk strategy",
        func=get_high_risk_strategy,
        description="Use this to answer high risk strategies . do not pass any input."
    ),
    Tool(
        name="suggest meduim risk strategy",
        func=get_medium_risk_strategy,
        description="Use this to answer medium risk strategies . do not pass any input."
    ),
    Tool(
        name="suggest low risk strategy",
        func=get_low_risk_strategy,
        description="Use this to answer low risk strategies . do not pass any input."
    ),
    Tool(
        name="which stock to buy out of two stocks",
        func=compare_two_stocks,
        description="Use this to answer which stock to buy from two given stocks .In prompt use the answer returned by the function. Pass the two stock names as input."
    ),
    Tool(
        name="get stock fundamentals",
        func=get_stock_fundamentals,
        description="Use this to answer fundamental details of stock. Use stock name as string input."
    ),
    Tool(
        name="get stock price visualization",
        func=get_price_visualizations,
        description="Use this for showing performance or visualization of a stock over a particular period of time.Do not use date,only use stock name , number of days as a single string input and return ."
    ),
    Tool(
        name="find top performing industries",
        func=find_top_performing_industries,
        description="Use this to answer top performing industries.Do not pass anything as input only call function."
    ),
    Tool(
        name="find top performing strategies",
        func=get_strategy_recommendation,
        description="Use this to answer top performing strategies.Do not pass anything as input only call function."
    ),
    Tool(
        name="DuckDuckGo Search",
        func = search.run,
        description="Use this to answer the question related to an individual person or company."
    )
]

zero_shot_agent=initialize_agent(
    llm=ollama,
    agent="chat-zero-shot-react-description",
    tools=tools,
    return_intermediate_steps=True,
    handle_parsing_errors=True,
    prompt=prompt,
    memory = memory,
    verbose = True,
    max_iteration = 1,
    output_key='intermediate_steps'

)
#agent_executor = AgentExecutor(agent=zero_shot_agent,tools = tools,verbose=True)

def get_zero_shot_response(prompt: str):
    result = zero_shot_agent(prompt)
    return result['output']

app = Flask(__name__)

@app.route("/query", methods=["POST"])
def query():
    try:
        data = request.get_json()
        prompt = data.get("prompt")
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400
        response = get_zero_shot_response(prompt)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6000)
                                                                               
