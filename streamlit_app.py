import streamlit as st
import os
import logging
import csv
import traceback
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableBranch
from langchain_classic.memory import ChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()

OPENAI_API_KEY= "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"
PINECONE_API_KEY= "pcsk_6MNTdP_53FJ3DTSbrqLpEPDAQcx5LSKU8rwPv46aufVSQ6fiKBdAvoH26XeiUJPfEC9Q4Q"

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    st.error("ERROR: API keys not found in .env file.")
    st.info("Please create a .env file in the same folder with your OPENAI_API_KEY and PINECONE_API_KEY.")
    st.stop()

PINECONE_INDEX_NAME = "trinamix-website-index-openai-url"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
LLM_MODEL_NAME = "gpt-4o-mini"
RETRIEVED_CHUNKS_K = 8
FORM_TRIGGER_COUNT = 4
LEAD_FILE_NAME = 'leads.csv'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@st.cache_resource
def get_models_and_retriever():
    try:
        llm = ChatOpenAI(model_name=LLM_MODEL_NAME, temperature=0.1)
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)
        
        logger.info(f"Connecting to Pinecone index: {PINECONE_INDEX_NAME}...")
        vector_store = PineconeVectorStore(
            index_name=PINECONE_INDEX_NAME,
            embedding=embeddings
        )
        
        retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVED_CHUNKS_K})
        logger.info("Retriever and models loaded successfully.")
        return llm, retriever
    except Exception as e:
        logger.error(f"ERROR: Could not connect to Pinecone/OpenAI. {e}")
        st.error(f"Error initializing services: {e}")
        st.stop()

llm, retriever = get_models_and_retriever()

conversational_prompt = ChatPromptTemplate.from_template(
    """You are a polite conversational assistant. The user said something simple (like "Hi", "Thanks").
    Respond politely and briefly.
    CHAT HISTORY: {chat_history}
    USER INPUT: {question}
    YOUR POLITE RESPONSE:"""
)

rag_prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful and proactive assistant for the Trinamix website.
    Your goal is to answer the user's question and then anticipate their next logical query.
    Use the following pieces of context to answer the question at the end.
    The context provided already includes the 'Source URL' at the start of each piece of text.

    RULES:
    1. Answer from Context: Use the following pieces of context to answer. Answer based ONLY on the provided context. Do not use any outside knowledge.
    2. If the user's question is regarding prices of any products or services, respond with: "For pricing related queries, please contact our sales team at sales@trinamix.com or visit our Contact Us page: https://www.trinamix.com/contact-us/"
    3. If the user's question asks about job openings or careers or job roles, and you see a source URL containing '/careers/' in the context, please provide that URL as the place to find job openings, even if the text itself doesn't list specific jobs.
    4. Fallback: If you followed Rule 1 (tried to answer from context) but the context does not contain a relevant answer to the question, you MUST say: "For detailed information on that topic, please contact the sales team at sales@trinamix.com or visit the Contact Us page: https://www.trinamix.com/contact-us/"
    5. Proactive Follow-up: After providing a factual answer (not a fallback or conversational reply), add a helpful follow-up question to guide the user.
        - If you list items (like products), ask: "Would you like to know more about one of these in particular?"
        - If you define a product (like Documantra), ask: "Are you interested in its features or benefits?"
    6. Citation: When you provide an answer from the context (Rule 2), you MUST cite the 'Source URL:' from which the information came. Format the citation on a new line as: "Please refer the link for more information: [URL]"

    CONTEXT: {context}
    CHAT HISTORY: {chat_history}
    QUESTION: {question}
    ANSWER (with citation and follow-up):
    """
)

classifier_prompt = ChatPromptTemplate.from_template(
    """Given the user question, classify its intent into one of four categories:
    Your answer must be only either of the below two words:
    1. 'conversational': For simple chit-chat, greetings, or acknowledgements.
    2. 'general_retrieval': For all other factual questions (e.g., "what is Trinamix?").

    CHAT HISTORY: {chat_history}
    USER QUESTION: {question}
    Classification:
    """
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
    
def is_career_related(message):
    career_keywords = [
        'career', 'careers', 'job', 'jobs', 'hiring', 'position', 'positions',
        'employment', 'work', 'vacancy', 'vacancies', 'opening', 'openings',
        'opportunity', 'opportunities', 'apply', 'application', 'resume',
        'cv', 'recruit', 'recruiting', 'recruitment'
    ]
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in career_keywords)

def save_lead_to_csv(name, email):
    try:
        file_exists = os.path.isfile(LEAD_FILE_NAME)
        with open(LEAD_FILE_NAME, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Name', 'Email'])
            writer.writerow([name, email])
        return True, "Success! Your information has been saved."
    except Exception as e:
        logger.error(f"Failed to save lead: {e}")
        return False, f"Error: Could not save your information. {e}"

@st.cache_resource(show_spinner=False) 
def get_full_chain(_llm, _retriever):
    # This function is cached, so the chains are built only once
    output_parser = StrOutputParser()

    classifier_chain = (
        {"question": RunnablePassthrough(), "chat_history": RunnablePassthrough()}
        | classifier_prompt
        | _llm
        | output_parser
    )

    conversational_chain = (
        conversational_prompt
        | _llm
        | output_parser
    )
    
    context_chain = (
        RunnableLambda(lambda x: x["question"])
        | _retriever
        | format_docs
    )

    retrieval_chain = (
        {
            "context": context_chain, 
            "question": RunnableLambda(lambda x: x["question"]), 
            "chat_history": RunnableLambda(lambda x: x["chat_history"])
        }
        | rag_prompt
        | _llm
        | output_parser
    )
    
    branch = RunnableBranch(
        (RunnableLambda(lambda x: "conversational" in x["topic"]), conversational_chain),
        retrieval_chain  
    )

    full_chain = (
        {
            "topic": classifier_chain,
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"]
        }
        | branch
    )
    return full_chain
    
full_chain = get_full_chain(llm, retriever)


if "message_history" not in st.session_state:
    st.session_state.message_history = []
    st.session_state.message_history.append({"role": "assistant", "content": "Hi! How can I help you today?"})

if "user_message_count" not in st.session_state:
    st.session_state.user_message_count = 0

if "form_submitted" not in st.session_state:
    st.session_state.form_submitted = False
    
if "conversation_ended" not in st.session_state:
    st.session_state.conversation_ended = False


for message in st.session_state.message_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

show_form = (
    st.session_state.user_message_count >= FORM_TRIGGER_COUNT and 
    not st.session_state.form_submitted and 
    not st.session_state.conversation_ended
)

show_chat_input = (not st.session_state.conversation_ended and not show_form)


if show_form:
    st.info("Please provide your information to continue chatting.")
    
    with st.form(key="lead_form"):
        name = st.text_input("Your Name *")
        email = st.text_input("Your Email *")
        submit_button = st.form_submit_button("Submit")

    if submit_button:
        if not name.strip() or not email.strip() or '@' not in email:
            st.error("Please enter a valid name and email address.")
        else:
            success, message = save_lead_to_csv(name, email)
            if success:
                st.session_state.form_submitted = True
                st.success(f"Thank you, {name}! You can continue your chat.")
                # Add a message to the chat
                st.session_state.message_history.append({"role": "assistant", "content": f"Thanks, {name}! What else can I help you with?"})
                st.rerun() # Re-run to hide form and show chat input
            else:
                st.error(message)

if show_chat_input:
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        st.session_state.user_message_count += 1
        st.session_state.message_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        
        if st.session_state.user_message_count == FORM_TRIGGER_COUNT:
            user_messages = [msg["content"] for msg in st.session_state.message_history if msg["role"] == "user"]
            
            career_question_count = 0
            for msg_content in user_messages:
                if is_career_related(msg_content):
                    career_question_count += 1
            
            if career_question_count >= 3:
                end_message = "It seems you have several questions about careers. For all career-related inquiries, please visit our careers page. This chat session will now end."
                
                st.session_state.message_history.append({"role": "assistant", "content": end_message})
                with st.chat_message("assistant"):
                    st.write(end_message)
                
                st.session_state.conversation_ended = True
                st.rerun() 

 
        if not st.session_state.conversation_ended:
            with st.spinner("Thinking..."):
                try:
                    model_chat_history = []
                    for msg in st.session_state.message_history:
                        if msg["role"] == "user":
                            model_chat_history.append(HumanMessage(content=msg["content"]))
                        else:
                            model_chat_history.append(AIMessage(content=msg["content"]))

                    inputs = {"question": user_input, "chat_history": model_chat_history}
                    ai_response_raw = full_chain.invoke(inputs)
                    ai_response = ai_response_raw.strip() 

                    if "[END_CHAT]" in ai_response:
                        st.session_state.conversation_ended = True
                        ai_response = ai_response.replace("[END_CHAT]", "").strip()

                    if "[TRIGGER_FORM]" in ai_response:
                        st.session_state.user_message_count = FORM_TRIGGER_COUNT
                        ai_response = ai_response.replace("[TRIGGER_FORM]", "").strip()
                
                except Exception as e:
                    logger.error(f"Error invoking chain: {e}\n{traceback.format_exc()}")
                    ai_response = "I'm sorry, an error occurred. Please try again."

            st.session_state.message_history.append({"role": "assistant", "content": ai_response})
            with st.chat_message("assistant"):
                st.write(ai_response)
            
            st.rerun()
if st.session_state.conversation_ended:
    st.info("This chat session has ended. For further assistance, please visit our website.")
