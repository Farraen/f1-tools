import json
import re
import ast
import statistics
import pandas as pd
from openai import OpenAI


class engine:

    def __init__(self,client):

        self.client = client

        dict_table_columns = {
        "Track":"Name of the track for the race", "PU Failures":"Column of PU failures where the user can manually assign in the event of a PU failure","PU Actual":"Column of real PU allocation after a race has ended",
        "PU Projection":"Prediction of PU allocation made by a decision engine. This is project early of the season. Engineers should rely on this prediction for each races",
        "MinTemp": "Minimum track temperature during the race day",
        "MaxTemp": "Maximum track temperature during the race day",
        "Distance": "Total race distance for an F1 car to complete the race",
        "PowerLeft":"The PU power left after PU degradation at the end of the season",
        "PowerReduced":"Total PU power reduction after at the end of the season",
        "RUL":"The remaining useful life of the PU in percentage. 100percent is like new PU and 0percent is a failed PU",
        "DamageThisRace":"Is the amount of PU degradation after a race"
        }
        self.str_table_columns = json.dumps(dict_table_columns)


    def send_message_secondary(self,persona,prompt):

        persona = [{"role":"system", "content":prompt}]
        user_messages = [{"role":"user", "content":prompt}]
        user_messages.extend(persona)
        openai_response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": m["role"], "content": m["content"]} for m in user_messages])
        
        response_str = openai_response.choices[0].message.content

        return response_str

    def check_question(self,prompt):

        index = []
        for i in range(0,3):
            str1 = f"You are an f1 racing engineer and has racing dashboard waiting for request from the user. "
            str8 = f"Based on the user prompt '{prompt}', is it a specific process task? Check if it is about informing PU status or PU failures, then it is a specific process task and answer yes. Check if it is about reoptimising/restrategising/optimising PU allocation, then it is a specific process task and answer yes."
            persona = [{"role":"system", "content":str1}]
            prompt = str8
            response_str = self.send_message_secondary(persona,prompt)
            flag = response_str.lower().startswith("yes")  
            index.append(flag)

        flag = statistics.median(index)

        return flag


    def find_task(self,prompt):
        str1 = f"You are an f1 racing engineer and has racing dashboard waiting for request from the user. "

        dict_options = [
            "Initialise to fill in the PU projection/allocation table",
            "Assign or replace actual PU number for a race",
            "Restrategise/rerun/optimise PU projections/allocations",
            "failed PU"]
        str_options = json.dumps(dict_options)

        str8 = f"Based on the user prompt '{prompt}', find the closest item to the items in this list [{str_options}]. Put the results in a dictionary with clostest index in 'index' key and closest item in 'value' key. Just output the dictionary in string format."

        persona = [{"role":"system", "content":str1}]
        prompt = str8
        index = []
        for i in range(0,5):
            response_str = self.send_message_secondary(persona,prompt)
            try:
                response_str = ast.literal_eval(re.search('({.+})', response_str).group(0))
                flag = response_str['index']
                index.append(int(flag))
            except:
                pass

        index = statistics.median(index)

        return index

    def find_failed_pu(self,prompt):
        str5 = f"There are three power units that can be assigned for each race and they are identified as 1, 2 and 3."
        prompt = f"Based on the user prompt '{prompt}', which PU has failed? Do not include any explaination."

        persona = [{"role":"system", "content":str5}]
        response_str = self.send_message_secondary(persona,prompt)
        temp = re.findall(r'\d+', response_str)
        index = list(map(int, temp))

        prompt = f"Based on the user prompt '{prompt}', which race when the PU fail? Just answer the race index. Do not include any explaination."
        persona = [{"role":"system", "content":str5}]
        response_str = self.send_message_secondary(persona,prompt)
        temp = re.findall(r'\d+', response_str)
        race = list(map(int, temp))

        payload = [index,race]

        return payload

    def check_recommendations(self,prompt):
        str1 = f"You are an f1 racing engineer and has racing dashboard waiting for request from the user. "
        prompt = f"Based on the user prompt '{prompt}', did the user ask for recommendations or advice? Answer yes or no."
        persona = [{"role":"system", "content":str1}]
        index = []
        for i in range(0,3):
            response_str = self.send_message_secondary(persona,prompt)
            flag = response_str.lower().startswith("yes")  
            index.append(flag)
        flag = statistics.median(index)
        return flag


    def send_message_technical(self,prompt):

        df = pd.read_pickle("test.pkl", compression='infer')
        a = df.columns.values.tolist()
        b = df.values.tolist()
        b.insert(0, a)
        ystr = "["
        for row in b:
            s = '[' + ', '.join(str(x) for x in row) + '],'
            ystr = ystr + s
        df_str = ystr + ']'

        str1 = f"You are an F1 race engineer and a data scientist. Your role would be analysing race telemetry and find patterns, trends and anomalies. PU or power unit is the engine of an F1 car."
        str4 = f"There is a table called PU allocation table of F1 power units. There are 21 rows for each races with one power unit allocated for each race. These are the columns and description in a dictionary string format: {self.str_table_columns}. "
        str5 = f"There are three power units that can be assigned for each race and they are identified as 1, 2 and 3."
        str6 = f"The PU are selected for each of the race depending on their performance and durability. The objective of the selection is the maximise the vehicle performance throughout the season while keeping all PU survive until the last race of the season."
        str7 = f"The power unit is the engine of the F1 car and it has range of RUL or remaining useful life between 0% to 100%. Power unit or PU with RUL above 0% indicates that the PU is surviving. The power of the PU is in terms of kW. The higher the kW, the better the PU performance. "
        
        str2 = f"This is a power unit allocation table in a list format with the first row is the column names: '{df_str}'. "
        prompt = str2 + prompt

        persona = [{"role":"system", "content":str1+str4+str5+str6+str7}]
        response_str = self.send_message_secondary(persona,prompt)

        return response_str

    def update_table(self,prompt):


        df = pd.read_pickle("test.pkl", compression='infer')
        df = df[['Track', 'PU Failures', 'PU Actual', 'PU Projection']]
        df['Round'] = df.index +1 
        df['Race number'] = df.index +1 


        a = df.columns.values.tolist()
        b = df.values.tolist()
        b.insert(0, a)
        ystr = "["
        for row in b:
            s = '[' + ', '.join(str(x) for x in row) + '],'
            ystr = ystr + s
        df_str = ystr + ']'

        str1 = f"You are an F1 race engineer and a data scientist. Your role would be analysing race telemetry and find patterns, trends and anomalies. PU or power unit is the engine of an F1 car."
        str4 = f"There is a table called PU allocation table of F1 power units. There are 21 rows for each races with one power unit allocated for each race. These are the columns and description in a dictionary string format: {self.str_table_columns}. "
        str5 = f"There are three power units that can be assigned for each race and they are identified as 1, 2 and 3."
        str6 = f"The PU are selected for each of the race depending on their performance and durability. The objective of the selection is the maximise the vehicle performance throughout the season while keeping all PU survive until the last race of the season."
        str7 = f"The power unit is the engine of the F1 car and it has range of RUL or remaining useful life between 0% to 100%. Power unit or PU with RUL above 0% indicates that the PU is surviving. The power of the PU is in terms of kW. The higher the kW, the better the PU performance. "
        
        str2 = f"This is a power unit allocation table in a list format with the first row is the column names: '{df_str}'. "
        str3 = f"Update the table based on the user prompt '{prompt}'. Output the table string in between # and $. '"
        prompt = str2 + prompt + str3

        persona = [{"role":"system", "content":str1+str4+str5+str6+str7}]
        response_str = self.send_message_secondary(persona,prompt)

        return response_str