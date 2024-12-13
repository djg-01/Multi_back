from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import warnings
warnings.filterwarnings("ignore")
import autogen
from autogen import AssistantAgent
from typing import Dict, List, Optional
from chromadb import PersistentClient
from langchain_huggingface import HuggingFaceEmbeddings
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator
import os
import re
from diskcache import Cache
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from openai import AzureOpenAI
from dotenv import load_dotenv
load_dotenv()
from io import BytesIO


client = AzureOpenAI(
    api_key = os.environ['AZURE_OPEN_AI_KEY'],
    api_version=os.environ['AZURE_API_VERSION'],
    azure_endpoint=os.environ['AZURE_ENDPOINT'], 
)

app = FastAPI()


# Define paths
# base_static_path = os.path.join(os.path.dirname(__file__), "static/dist")
base_static_path = r"./static/dist"
# assets_static_path = os.path.join(base_static_path, "assets")
assets_static_path = os.path.join(base_static_path, "assets")
os.makedirs(assets_static_path, exist_ok=True)

# Mount static assets
app.mount("/assets", StaticFiles(directory=assets_static_path), name="assets")


cache_dir = "./cache_directory"
cache = Cache(cache_dir)



# CORS settings
origins = ["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:8000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session store for conversation history
session_store: Dict[str, List[Dict]] = {}

# Data model for chat requests
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    history: Optional[List[Dict]] = None
    suggestion: Optional[str] = None

# Resumable Group Chat Manager
class ResumableGroupChatManager(autogen.GroupChatManager):
    def __init__(self, groupchat, history=None, **kwargs):
        self.groupchat = groupchat
        if history:
            self.groupchat.messages = history
        super().__init__(groupchat, **kwargs)
        if history:
            self.restore_from_history(history)

    @property
    def groupchat(self):
        return self._groupchat

    @groupchat.setter
    def groupchat(self, value):
        self._groupchat = value

    def restore_from_history(self, history) -> None:
        for message in history:
            for agent in self.groupchat.agents:
                if agent.name != message.get("speaker"):
                    self.send(message, agent, request_reply=False, silent=True)


def suggest_question(user_qt):

    prompt = f"""
        This is current user question : {user_qt}

        Following is the Probable list of question in order :
            1.	Hi
            2.	What can you do for me and Who are you ?
            3.	Ok, so, I’m having an issue with the conveyor 1, Could you please check ?
            4.	Could you help us to understand what is this code F07801 Drive: Motor overcurrent about and also share the remedies?
            5.	Could you also share the snippet of the VFD motor M501 specifications from the drawing
            6.	All looks fine, Could you look into other possible issues ?
            7.	I think Mr. Anil was in that shift, what did he observe ?
            8.	Could you please share the parameter trends for each sensor ?
            9.	Could you please create a maintenance work order in SAP ?
            10. Could you please check the production data & estimate how many billets we are producing by the end of the day today ? 
            11. Please check all the logs relevant to conveyer-1 motor

        ***Your task is to see the current user question and just reply with the next probable question as is in the list, nothing extra***

    """



    completion = client.chat.completions.create(
        model="gpt-4",  # e.g. gpt-35-instant
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.7,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return completion.choices[0].message.content


CSV_PROMPT_PREFIX = """
** KEEP IN MIND THAT YOU CAN ONLY SELECT ONE FLOW/SPEAKER BY ANALYSING USERS SENTIMENT AS DEFINED BELOW. STRICTLY FOLLOW THE SAME**
If :
    The question is asking for 
        Logs (Example - Who was the operator in that shift ?, What did Mr. Anil do in his shift?) 
        explaining Alarm codes (Example - Could you help us to understand what this code F07801 in Scada is about and also share the remedies ?), 
        Cause (Example - Why No valid pressure actual value available ?), 
        Remedies (Example - Could you help us to understand what is this code about and also share the remedies ?), 
        Specifications (Example -  Could you also share the snippet of the VFD motor M501 specifications from the drawing ?), 
        Features, etc.
        
        ** Then start with speaker - 'researcher'. **
Else :
    The question is asking for 
        Example 1 - Is there any issue in conveyor 1?
        Example 2 - Issue in dryer 2? 
        Example 3 - Hi, I am having an issue with the conveyor 1 , Could please check
        Example 4 - Could you look into other possible issues?
        Example 5 - Could you please share the parameter trends for vibration, current, and bearing temperature
         
        Or question related to sensor readings, any visualizations , trends, plots

        ** Then start with speaker - 'coder'. **
"""


df1 = os.path.join(os.path.dirname(__file__), '..', 'Events_Alarms.csv')
df2 = os.path.join(os.path.dirname(__file__), '..', 'Threshold.csv')
df3 = os.path.join(os.path.dirname(__file__), '..', 'Sensor_Readings.csv')
maintain = os.path.join(os.path.dirname(__file__), '..', 'Maintainance_order')





# df1 = r'C:\DJ\Udemy\Gen_AI\OpenInterpreter\data\Events_Alarms.csv'
# df2 = r'C:\DJ\Udemy\Gen_AI\OpenInterpreter\data\Threshold.csv'
# df3 = r'C:\DJ\Udemy\Gen_AI\OpenInterpreter\data\Sensor_Readings.csv'

# maintain = r'C:\DJ\Udemy\Gen_AI\Auto_New_2_Frontend\project\Maintainance_order'
# images = r'C:\DJ\Udemy\Gen_AI\Auto_New_2_Frontend\project\Images'

CSV_PROMPT_SUFFIX = f"""

If you are not sure, try another method.
FORMAT 4 FIGURES OR MORE WITH COMMAS.
- If the methods tried do not give the same result,reflect and
try again until you have two methods that have the same result.
- If you still cannot arrive to a consistent result, say that
you are not sure of the answer.
- If you are sure of the correct answer, create a beautiful 
and thorough response.
- **DO NOT MAKE UP AN ANSWER OR USE PRIOR KNOWLEDGE,
ONLY USE THE RESULTS OF THE CALCULATIONS YOU HAVE DONE**.
- **ALWAYS** try Flexible Keyword Search, and When accessing data, do not assume column names or any data; dynamically
reference the column headers by inspecting the dataframe structure.
and always try Flexible Keyword Search:


{df1}: Contains event and alarm logs with columns like Timestamp, Tag, and Event.
{df2}: Contains threshold values for sensor readings, including columns such as Tag,
High Alarm, High Warning, Low Alarm, and Low Warning.
{df3}: Contains real-time sensor readings with columns like Timestamp and tags
representing various sensors.

2) Flexible Keyword Search:
    If the question pertains to machine failures or specific events, search {df1} using
    keyword matching:
    Perform flexible keyword searches that look for partial matches (e.g., "conveyor 1
    issue" should match "conveyor 1").
    **Search across all columns in {df1} without assuming exact string matches or casing.**


3) For questions involving threshold comparisons, generate code that:

a) Loads {df2} and {df3}, replacing 'NA' values in {df2} with None.
b) Iterates through each sensor tag in {df3}, comparing readings to thresholds from {df2}.
c) Detects if readings exceed any threshold and records the relevant timestamp, sensor tag, reading, and threshold exceeded.

*An example code structure for such comparisons is:*
    # Convert NA values in threshold data to None for easier comparison
    df_threshold = df_threshold.replace('NA', None)
    # Initialize an empty list to store the results
    out_of_range_sensors = []
    # Iterate through each row in the sensor readings
    for _, row in df_sensor_readings.iterrows():
    timestamp = row['Timestamp']
    for tag in df_threshold['Tag']:
    # Get the sensor reading and threshold values
    sensor_value = row[tag]
    high_alarm = df_threshold[df_threshold['Tag'] == tag]['High Alarm'].values[0]
    high_warn = df_threshold[df_threshold['Tag'] == tag]['High Warning'].values[0]
    low_alarm = df_threshold[df_threshold['Tag'] == tag]['Low Alarm'].values[0]
    low_warn = df_threshold[df_threshold['Tag'] == tag]['Low Warning'].values[0]
    # Check if the sensor value is outside the threshold limits
    if high_alarm is not None and sensor_value > float(high_alarm):
    out_of_range_sensors.append((timestamp, tag, sensor_value, 'High Alarm'))
    elif high_warn is not None and sensor_value > float(high_warn):
    out_of_range_sensors.append((timestamp, tag, sensor_value, 'High Warning'))
    elif low_alarm is not None and sensor_value < float(low_alarm):
    out_of_range_sensors.append((timestamp, tag, sensor_value, 'Low Alarm'))
    elif low_warn is not None and sensor_value < float(low_warn):
    out_of_range_sensors.append((timestamp, tag, sensor_value, 'Low Warning'))
    # Convert the results to a DataFrame for easier viewing
    df_out_of_range = pd.DataFrame(out_of_range_sensors, columns=['Timestamp', 
    'Sensor', 'Value', 'Threshold Exceeded'])
    print(df_out_of_range)


4) For questions involving asking for any trends, patterns, anamolies or any abonormal trend/issues use {df3}.

5) Pay Close attention to Timestamps, when communicating between files restrict 
answers around particular Timestamps. 
6) If Question is not related to data then act as you are.

7) For Question like 'Could you please create a maintenance work order in SAP?' - Write a python code to Save a .txt file at {maintain}

"""

CSV_PROMPT_SUFFIX += """
Irrespective of previous questions always generate code.
8) **ALWAYS** For all visualizations (bar, line, charts), never generate plots, rather than generating plots filter data based on user request and transform result in the **EXACT** following way, check timestamps and handle them carefully while generate code accordingly if there is no date mentioned in user request then just try to check time within timestamp:
    [   { Timestamp: 'DD-MM-YYYY HH:MM:SS', 'TT-M501': 25.4, 'CT-M501': 15.2, 'VT-M501': 12.3, 'ST-M501': 20.1 },
        { Timestamp: '', 'TT-M501': 28.4, 'CT-M501': 18.2, 'VT-M501': 14.3, 'ST-M501': 22.1 },
        { Timestamp: '', 'TT-M501': 30.1, 'CT-M501': 20.5, 'VT-M501': 16.0, 'ST-M501': 25.3 },
    ]

    Sample code for transforming:

        json_data = filtered_data.to_dict(orient='records')


    **ALWAYS INCLUDE COMPLETE TIMESTAMP, including the seconds** Directly filter on the timestamp column, make no assumptions. 
    For 

        
        start_time = pd.to_datetime('2023-11-04 15:37:10')
        end_time = pd.to_datetime('2023-11-04 15:39:15')

        Do not use today, get the date from the timestamp itself.

        def convert_timestamp(obj):
            Custom JSON encoder to handle Pandas Timestamp objects
            if isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
            
        df_sensor_readings['Timestamp'] = pd.to_datetime(df_sensor_readings['Timestamp'], dayfirst=True, errors='coerce') 

        # Check for parsing issues
        if df_sensor_readings['Timestamp'].isnull().any():
            print("Warning: Some timestamps could not be parsed:")
            print(df_sensor_readings[df_sensor_readings['Timestamp'].isnull()])

        # Define the time range for filtering
        start_time = pd.to_datetime('2023-11-04 15:37:10')
        end_time = pd.to_datetime('2023-11-04 15:39:25')

        # Filter data within the specified time range
        filtered_df = df_sensor_readings[(df_sensor_readings['Timestamp'] >= start_time) & (df_sensor_readings['Timestamp'] <= end_time)]

        # Transform the filtered data to JSON format for visualization
        # Using json.dumps with a custom encoder to handle Timestamp


        # Convert Timestamp column to string format
        filtered_df['Timestamp'] = filtered_df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

        # Convert the filtered DataFrame to a list of dictionaries
        json_data = filtered_df.to_dict(orient='records')
    """
        # # Convert 'Timestamp' column to datetime
        # df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=True, errors='coerce')

        # # Check for parsing issues
        # if df['Timestamp'].isnull().any():
        #     print("Warning: Some timestamps could not be parsed:")
        #     print(df[df['Timestamp'].isnull()])

 
        # filtered_df = df[(df['Timestamp'] >= start_time) & (df['Timestamp'] <= end_time)]
            

# 8) ** For all visualizations show charts using apex-charts using react ** and no need to use or show the visualizations using matplotlib and seaborn only use react apex-charts to show visualizations everytime. Also execute it everytime.



# doc_paths = [
#     r"C:\Users\EQ363EQ\Downloads\eLogbook_Dump.pdf",
#     r"C:\Users\EQ363EQ\Downloads\VFD_Motor_Fault_manual.pdf",
#     r"C:\Users\EQ363EQ\Downloads\VFD_M501_Drawing.pdf",
# ]



config_list=  [
{
"model": "gpt-4",
"api_key": os.environ['AZURE_OPEN_AI_KEY'],
"base_url": os.environ['AZURE_ENDPOINT'],
"api_type": "azure",
"api_version": os.environ['AZURE_API_VERSION'],
"temperature": 0,
"stream": True,
}
]

def termination_msg(x):
    return isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()



# Placeholder Agents
llm_config = {"config_list": config_list, "timeout": 60}


boss = autogen.UserProxyAgent(
    name="Boss",
    is_termination_msg=termination_msg,
    human_input_mode="NEVER",
    # code_execution_config=False,  # we don't want to execute code in this case.
    default_auto_reply="Reply `TERMINATE` if the task is done.",
    description="The boss who initiates the chat and is responsible for executing Python code and passing the output to the Speaker : 'Output_Reviewer_and_Explainer' for validation and feedback. You can also install packages as required.",
    code_execution_config={
        "last_n_messages": 2,
        "work_dir": "../groupchat1",
        "use_docker": False,
    }
)

researcher = autogen.AssistantAgent(
    name="Researcher",
    is_termination_msg=termination_msg,
    system_message="""You are a Researcher. For any question that requires external knowledge (Logs, Events, Sensor Readings, Thresholds) or resources, 
    you MUST first suggest calling the 'retrieve_content' function before answering. Reply `TERMINATE` in the end when everything is done.""",
    llm_config=llm_config,
    description="You are a Expert Researcher, Always Your only work is to call the tool.",
)

context_reviewer = AssistantAgent(
    name="context_reviewer",
    is_termination_msg=termination_msg,
    system_message='''You are a helpful assistant capable of retrieving information from PDF documents. 
    If :
        If you can't answer the question relevently with the current context or answer is not available in the given snippet, only then you should reply exactly `UPDATE CONTEXT`.

    Else:

        **Terminate and report Your final answer must be in a below format:
        Final Answer : 
        **


    ''',
    llm_config={
        "timeout": 600,
        "cache_seed": 42,
        "config_list": config_list,
    },
    description='A Context Reviewer responsible for validating retrieved content and ensuring it aligns with the query requirements.'
)


coder = AssistantAgent(
    name="Coder",
    is_termination_msg=termination_msg,
    
    system_message=f'''You are a senior python engineer, you provide python code by following :
        1) First set the pandas display options to show all the columns and use those columns names only to generate code.
        2) Using those columns names and the given context:{CSV_PROMPT_SUFFIX}, Generate code,
            to answer the question. Make sure columns symantic meanings also you have to check with the users query. Handle errors and Install libraries as required. ***Always only provide Complete Python code.This agent is a senior Python engineer responsible for only writing complete and error-free Python codes and ensuring codes are validated and executable. And for code execution he will ALWAYS pass to speaker : Boss. And this agent is also responsible for seeing errors from speaker : Boss and then will again generate the updated & executable code.
            ''',
    # Reply 'TERMINATE' if you have the answer.
    llm_config=llm_config,
    description="This agent is a senior Python engineer responsible for only writing complete and error-free Python codes and ensuring codes are validated and executable. And for code execution he will ALWAYS pass to speaker : Boss. And this agent is also responsible for seeing errors from speaker : Boss and then will again generate the updated & executable code. *** But he will only output python code everytime***",
)
General_chat_Assistant = AssistantAgent(
    name="General_chat_Assistant",
    system_message="You are a friendly and conversational assistant.You are made by EY India GEN AI Engineers and your name is 'Maddy' and always you will support end user to analysize, summarize tasks specefically focusing on Industry 4.O. Your job is to handle general chat, answer basic questions, and engage in friendly conversations. Respond warmly and informally to greetings like 'hi', 'hello', 'how are you?', and other casual topics. Stay polite and positive while keeping your responses concise and engaging.",
    llm_config=llm_config,
    description="This agent specializes in handling general conversations, greetings, and casual interactions. It is designed to engage users in light, friendly, and helpful chat."
)


Output_Reviewer_and_Explainer = autogen.AssistantAgent(
    name="Output_Reviewer_and_Explainer",
    is_termination_msg=termination_msg,
    system_message='''This agent evaluates the output of executed code got from Speaker : Boss, ensuring correctness and completeness, and if you get table as output then show that table ***as a table in Markdown format, using proper headings and row separators, without loss of text in columns*** and along with it eplain the answer, and if table is not there then you can just explain the final answer and **EVERYTIME** there is no need to mention 'The code successfully executed'.

    For Question like 'Could you please create a maintenance work order in SAP?' **Just Respond Work Order has been created and assigned**
    *** If the output of coder is json like '[   { Timestamp: 'DD-MM-YYYY HH:MM:SS', 'TT-M501': 25.4, 'CT-M501': 15.2, 'VT-M501': 12.3, 'ST-M501': 20.1 }]' then just reply 'TERMINATE' nothing extra. ***
       ''',
    # If in final answer something like this is mentioned ' The code successfully executed and generated plots for the trends' or related to 'visualization', Just simply say 'We have popped up images for you and saved them in Images folder.Please let me know if you need any assistance ahead. Thanks
    # If answer is like [{  }] from coder, **ALWAYS just return the list from coder as is**. ***NEVER return any thing extra ***.
    # system_message='''Your role is to analyze the output of executed code, identify any issues, and provide constructive feedback. Confirm the final answer if the output is correct and complete. Reply 'TERMINATE' if the task is resolved successfully.''',
    llm_config=llm_config,
    description="This agent evaluates the output of executed code, ensuring correctness and completeness, and provides feedback or confirms the final answer.",
)


def _reset_agents():
    boss.reset()
    coder.reset()
    researcher.reset()
    context_reviewer.reset()
    Output_Reviewer_and_Explainer.reset()

temp_ans1 = """
I can see that the preventive maintenance for Conveyor-1 motor was done on 15th July, 2024 in SAP-PM Logs, and no abnormalities were logged in the report.

Upon reviewing the operator logbook from 2nd November, 2024, I also noticed an entry mentioning that unusual noise was observed in motor M501.

As a precautionary measure, the mechanical team performed manual lubrication and tightened some components.

It’s possible that despite these measures, a bearing failure may have still occurred, leading to the motor stall and the overcurrent issue.
"""


temp_ans2 = """
Sure! I will check the production data and estimate the total number of billets scheduled for production by the end of the day today. Please hold on for a moment.​

Today's production is estimated to be 592 billets by the end of the day.
"""


temp_ans3 = """
During his shift, Mr. Anil conducted a startup inspection where he checked the belt tension and roller alignment, ensuring all components were in working order and no abnormal noises were detected. Throughout the shift, the conveyor ran smoothly without any interruptions, and the load handling was within safe limits.
"""


# Group chat logic
def call_rag_chat(question, history=None):
    _reset_agents()

    next_qt = suggest_question(question)
    cache_key = question

    # if question == 'Could you also share the snippet of the VFD motor M501 specifications from the drawing':
    #     image_data = cache.get(question)
    #     image_stream = BytesIO(image_data)
    #     return StreamingResponse(image_stream, media_type="image/png")


    # Check cache
    cached_response = cache.get(cache_key)
    if cached_response:
        print("Cache hit!")
        return cached_response, history, next_qt
    
    print("Cache miss! Processing...")



    def retrieve_content(user_query: str):
        # chromadb_path=r"C:\DJ\Udemy\Gen_AI\Auto_New_2_Frontend\project\backend\chromadb_directory"
        chromadb_path=os.path.join(os.path.dirname(__file__), 'chromadb_directory')
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"
        top_k_results=3

        try:
            # Initialize ChromaDB client and collections
            client = PersistentClient(path=chromadb_path)
            collections = client.list_collections()
            
            if not collections:
                return "No collections found in the database."
            
            collection_doc = client.get_collection(collections[0].name)
            
            # Embedding model setup (disable multiprocessing for compatibility)
            embedding_model = HuggingFaceEmbeddings(
                model_name=embedding_model_name,
                multi_process=False,  # Disabled to avoid multiprocessing issues
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            
            # Search function
            query_vector = embedding_model.embed_query(user_query)
            results = collection_doc.query(query_embeddings=[query_vector], n_results=top_k_results)
            print(results)
            # Return the results
            return results['documents']
        
        except Exception as e:
            return f"An error occurred: {e}"


    for caller in [researcher]:
        d_retrieve_content = caller.register_for_llm(
            description="retrieve content for code generation and question answering.", api_style="tool"
        )(retrieve_content)

    for executor in [researcher]:
        executor.register_for_execution()(d_retrieve_content)

    allowed_transitions  = {
        boss: [coder, researcher, Output_Reviewer_and_Explainer, General_chat_Assistant],
        coder: [boss],
        # Output_Reviewer_and_Explainer: [coder],
        researcher: [context_reviewer],
        context_reviewer: [researcher],
    }


    groupchat = autogen.GroupChat(
        agents=[boss, coder, researcher, context_reviewer, Output_Reviewer_and_Explainer, General_chat_Assistant],
        allowed_or_disallowed_speaker_transitions=allowed_transitions,
        speaker_transitions_type="allowed",
        select_speaker_message_template = '''

        Below are the agent descriptions for you which we have.
            Boss: The boss who initiates the chat and is responsible for executing Python code and  then if got successful execution pass the output to the Speaker : 'Output_Reviewer_and_Explainer' or if got any execution error pass that as an feedback to Speaker:'Coder'. You can also install packages as required.

            Researcher: This agent is exclusively responsible for fetching industry-specific external knowledge or additional information. It is invoked whenever external knowledge is required to answer the user's question, as internal knowledge is not utilized by this agent. The agent uses the retrieve_content function to acquire the necessary information. and then for review or final output he will everytime send his retrived context to the speaker : 'context_reviewer'.

            context_reviewer: This Agent is responsible for validating retrieved content got from Speaker : 'Researcher' and ensuring it aligns with the query/user requirements. Everytime he will be only responsible for Explaining the Final answer.

            Coder: This agent is a senior Python engineer responsible for only writing complete and error-free Python codes and ensuring codes are validated and executable. And for code execution he will ALWAYS pass to speaker : Boss. And this agent is also responsible for seeing errors from speaker : Boss and then will again generate the updated & executable code. *** But he will only output python code everytime*** 

            Output_Reviewer_and_Explainer: This agent evaluates the output of executed code got from Speaker : Boss, ensuring correctness and completeness, and will just explain the final answer.


            General_chat_Assistant: This agent specializes in handling general conversations, greetings, and casual interactions. It is designed to engage users in light, friendly, and helpful chat. Everytime he will be only responsible for Explaining the Final answer.

        Read the above descriptions and
        Then select the next role from ['Boss', 'Researcher', 'context_reviewer', 'Coder', 'Output_Reviewer_and_Explainer'] to play. Only return the role.         
        

        Then **ALWAYS** You have to decide between 2 workflows as given A) and B).

        Workflows description:
        A) : Coding Workflow (***ALWAYS Never involve 'researcher' in this workflow ***)
            Follow the Flow :
                1) Generate Code using speaker - 'Coder'
                2) Execute the code using speaker - 'Boss'
                3) Share the execution output with speaker -  'Output_Reviewer_and_Explainer' for validation.   Return the final answer

        B) : Research Workflow (***ALWAYS Only involve 'researcher' and 'context_reviewer' in this workflow ***)
                1)Speaker 'researcher' can not talk directly with 'boss' and 'coder'
                2)Get context from Speaker 'researcher' and evertytime pass it for validation to speaker 'context_reviewer'.
        ''',
                # ***3)STRICTLY For final answer do not delegate to other speakers, You only report the final answer and TERMINATE.***
        max_round=20,
        speaker_selection_method="auto",
        messages=history or [],
    )

    manager = ResumableGroupChatManager(groupchat=groupchat, history=history, llm_config=llm_config)

    ans = boss.initiate_chat(manager, message=question + CSV_PROMPT_PREFIX, clear_history=False)
    conversation_history = groupchat.messages
    print('History : ', conversation_history)
    final_answer = None

    for message in reversed(conversation_history):
        content = message.get('content', '').strip()
        if content and content.upper() != 'TERMINATE':
            final_answer = content
            break

    cache.set(cache_key, final_answer, expire=8640000)  # Cache for 100 days


    return final_answer, conversation_history, next_qt


# @app.api_route("/show-image", methods=["GET", "POST"])
# @app.get("/show-image")
# def show_image():
#     return FileResponse(r"backend\static\dist\vfd_drawing.png", media_type="image/png")

# # Chat endpoint
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        session_id = request.session_id or "default"
        history = session_store.get(session_id, request.history)
        print('History in chat route : \n', history)

        if not request.message:
            raise HTTPException(status_code=400, detail="No message provided")
        


        elif request.message == "I think Mr. Anil was in that shift, what did he observe ?":
            next_qt = suggest_question(request.message)
            response = temp_ans3
            print(response)
            return {"response": response, "suggestion": next_qt}

        elif request.message == "Please check all the logs relevant to conveyer-1 motor":
            next_qt = suggest_question(request.message)
            # return {"temp_ans1":temp_ans1}
            print("hello")
            response = temp_ans1
            print(response)
            return {"response": response, "suggestion": next_qt}
        
        elif request.message == "Could you please check the production data & estimate how many billets we are producing by the end of the day today ?":
            next_qt = suggest_question(request.message)
            response = temp_ans2
            print(response)
            return {"response": response, "suggestion": next_qt}
        
        
        


        
        # elif request.message == "Could you share the snippet of the VFD motor M501 specifications from the drawing ?":
        elif request.message.lower() == "Could you share the snippet of VFD motor drawing ?" or re.search(r"\bdrawing\b|\bsnippet\b", request.message, re.IGNORECASE):
            # Return a URL for the first static image
            next_qt = suggest_question(request.message)

            return {
                "image_url": r"assets/vfd_drawing.png",
                # "image_url": r"C:\DJ\Udemy\Gen_AI\Auto_New_2_Frontend\project\backend\static\dist\vfd_drawing.png",
                # "image_url": os.path.join(os.path.dirname(__file__), 'static', 'dist', 'assets','vfd_drawing.png'),
                # "image_url": r"public\vfd_drawing.png",
                "suggestion": next_qt
            }
        #     return RedirectResponse(url="/show-image")
            

        # elif request.message == "Could you share the snippet of the M501 Drives Dimension Drawing ?":
        elif request.message.lower() == "could you share the snippet of the m501 drives dimension drawing ?" or re.search(r"\dimension\b|\bdrives\b", request.message, re.IGNORECASE):
            # Return a URL for the second static image
            next_qt = suggest_question(request.message)

            return {
                "image_url": r"assets/m501_drives_dimension_drawing.png",
                "suggestion": next_qt

            }
            

        
        else:
            response, updated_history, suggested_qt = call_rag_chat(request.message, history)

        # Update session history
            session_store[session_id] = updated_history

            return {
                "response": response,
                # "history": updated_history,
                "session_id": session_id,
                "suggestion": suggested_qt,
            }
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    


# @app.get("/")
# def read_root():
#     return {"message": "Welcome to the Backend API!"}



@app.get("/")
async def serve_frontend():
    """Serve frontend static files and handle client-side routing"""
    # If API request, let it pass through to the API routes
    # if full_path.startswith("api/"):
    #     return JSONResponse(
    #         status_code=404,
    #         content={"message": "API route not found"}
    #     )
       
    # For all other routes, serve the index.html
    index_path = os.path.join(base_static_path, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    else:
        return JSONResponse(
            status_code=404,
            content={"message": "Frontend not built. Please run 'npm run build' first. Then run npm run dev"}
        )
    


@app.get("/chat")
async def chat_page():
    """
    Serve the frontend for the /chat route.
    """
    # Path to the index.html file
    index_path = os.path.join(base_static_path, "index.html")

    # Check if the index.html file exists and serve it
    if os.path.exists(index_path):
        return FileResponse(index_path)
    else:
        return JSONResponse(
            status_code=404,
            content={"message": "Frontend not built. Please run 'npm run build' first. Then run npm run dev."}
        )
