import streamlit as st

st.title("Conversation Outfit Recommender")
st.write("Type 'exit' to exit from chatbot")

from llama_index import (
    StorageContext, 
    load_index_from_storage, 
    LangchainEmbedding, 
    ServiceContext, 
    set_global_service_context, 
)
from llama_index.memory import ChatMemoryBuffer
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import openai
import json
import os

USER_DATA = {
    "age": "24",
    "gender": "male",
    "location": "Mumbai",
    "previous_orders": "Shirt, Trouser",
    "color_choices": "black, blue",
    "average_budget": "2000",
    "favorite_brands": "FTX, HERE&NOW, LOUIS PHILIPPE, PETER ENGLAND"
}

with open('./flipkart_data.json', 'r') as json_file:
    data = json.load(json_file)

with open('./key.json', 'r') as json_file:
    key = json.load(json_file)

os.environ["OPENAI_API_KEY"] = key.get("open_ai_key")
os.environ["TOKENIZERS_PARALLELISM"] = "False"
openai.api_key = key.get("open_ai_key")

embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2'))
service_context = ServiceContext.from_defaults(embed_model=embed_model)
set_global_service_context(service_context)

storage_context = StorageContext.from_defaults(persist_dir="./what_to_wear old")
fashion_data_index = load_index_from_storage(storage_context)


fashion_data_template = f"""
customer details:
{{Gender}}
{{Age}}
{{Location}}

suggestion asked: {{suggestion_asked}}
Output: 
"""

storage_context = StorageContext.from_defaults(persist_dir="./flipkart_products")
flipkart_products_index = load_index_from_storage(storage_context)


flipkart_data_template = f"""
Suggest me a {{engine1_output}} where
Vertical is {{vertical}}\n
Titles contain {{color}}\n
Keyspecs or product details contain {{material}}\n
Ratings.average is greater than 2.5\n
Customer details:
{{gender}}\n
{{age}}\n
{{location}}\n
{{previous_orders}}\n
{{color_choices}}\n
{{average_budget}}\n
{{favorite_brands}}\n
Answer: 
"""

memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

fashion_chat_engine = fashion_data_index.as_chat_engine(
    chat_mode="context",
    memory=memory,
    system_prompt="""
    You are an expert of Latest Fashion, suggesting a Trendy, Latest and 
    occasion-specific Fashion outfits as per customer Needs.
    Keep In mind the Gender, Age, Location, Favorite colors, 
    previously bought items and Customer's average budget 
    before suggesting the outfit. Your Output should be strictly an array of objects with fields: 
    "outfit name": eg: "Navy blue cotton shirt"
    "type": eg: "Shirt"
    "material": eg: "cotton"
    "color": eg: "blue
    You need to suggest only one clothing in one object. Multiple clothing, create multiple objects.
    No introduction or conclusion is required."""
)

flipkart_chat_engine = flipkart_products_index.as_chat_engine(
    chat_mode="context",
    memory=memory,
    system_prompt="You are suggesting clothes to the customer from your knowledge base, your response should strictly be in the format of a single object with fields, id, name, brand and its corresponding url. While suggesting the products, keep in mind the customer details. If the url does have https://flipkart.com, please add it. All the products should be from your knowledge base and never generate products by yourself, if product is not present then return empty object. Suggest only 1 product. If user is asking for another product, then suggest them a another product different from the previous one but from your knowledge only. Strictly If you don't have the requested product then return an empty object" ,
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Hello! How can I help you?"})

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if "count" not in st.session_state:
    st.session_state.count = 0

# React to user input
if user_input := st.chat_input("Suggest me an outfit for ...?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_input)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner('building your outfit ...'):

        if user_input == "exit":
            st.session_state.count = 0
            st.session_state.messages = []
            response = "Thank you for using our service. Have a nice day!"

        elif st.session_state.count == 0:
        
            prompt = fashion_data_template.format(
                Gender=USER_DATA.get("gender"),
                Age=USER_DATA.get("age"),
                Location=USER_DATA.get("location"),
                suggestion_asked=user_input
            )

            response = fashion_chat_engine.chat(prompt).response
            products = json.loads(response)

            flipkart_products = []

            for product in products:
                prompt = flipkart_data_template.format(
                    engine1_output=product.get("outfit name"),
                    vertical=product.get("type"),
                    material=product.get("material"),
                    color=product.get("color"),
                    gender=USER_DATA.get("gender"),
                    age=USER_DATA.get("age"),
                    location=USER_DATA.get("location"),
                    previous_orders=USER_DATA.get("previous_orders"),
                    color_choices=USER_DATA.get("color_choices"),
                    average_budget=USER_DATA.get("average_budget"),
                    favorite_brands=USER_DATA.get("favorite_brands")
                )
                response = flipkart_chat_engine.chat(prompt)
                response = json.loads(response.response)
                if response:
                    flipkart_products.append(response)

            matching_products = []

            for product in flipkart_products:
                for obj in data:
                    if product['id'] == obj['id']:
                        matching_products.append(product)
                        break 


            response = "Here are some suggestions for you: \n"
            for i, product in enumerate(matching_products):
                response += f"""{i+1}. {product['name']}
                    Id: {product['id']}
                    Brand: {product['brand']}
                    Url: [{product['url']}]({product['url']})\n\n"""
                
            st.session_state.count = st.session_state.count + 1
        
        else:
            response = flipkart_chat_engine.chat(user_input)
            response = json.loads(response.response)
            typeOfResponse = type(response)

            if typeOfResponse != list:
                response = f"""1. {response['name']}
                    Id: {response['id']}
                    Brand: {response['brand']}
                    Url: [{response['url']}]({response['url']})\n\n"""
                
            else:
                new_response = []
                for product in response:
                    for obj in data:
                        if product['id'] == obj['id']:
                            new_response.append(product)
                            break

                response = "Here are some suggestions for you: \n"
                for i, product in enumerate(new_response):
                    response += f"""{i+1}. {product['name']}
                        Id: {product['id']}
                        Brand: {product['brand']}
                        Url: [{product['url']}]({product['url']})\n\n"""

    st.success('Done!')

    if response == "Here are some suggestions for you: \n":
        response = "Sorry, currently my knowledge base doesn't have products to make the outfit : ("
        st.session_state.count = 0

    response = response
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

