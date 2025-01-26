import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import requests
import difflib
import re
from typing import Dict, Any
import json
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool, ToolException

# Load environment variables from .env file
load_dotenv()

# Custom CSS for dark theme and UI modifications
st.markdown(
    """
    <style>
    /* ----------------------------------------------------------
       1. Force black background in all high-level HTML containers
       ---------------------------------------------------------- */
    html, body {
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    .main,
    [data-testid="block-container"],
    [data-testid="stBlock"],
    [data-testid="stVerticalBlock"],
    [data-testid="stHorizontalBlock"] {
        background-color: #000000 !important;
        color: #ffffff !important;
    }

    /* ----------------------------------------------------------
       2. Streamlit's main UI, header, footer, and sidebar
       ---------------------------------------------------------- */
    .stApp {
        background-color: #000000 !important;
    }
    [data-testid="stAppViewContainer"] {
        background-color: #000000 !important;
    }
    [data-testid="stHeader"] {
        background-color: #000000 !important;
    }
    [data-testid="stToolbar"] {
        background-color: #000000 !important;
    }
    footer {
        background-color: #000000 !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #000000 !important;
    }
    [data-testid="stSidebarNav"] {
        background-color: #000000 !important;
    }

    /* ----------------------------------------------------------
       3. Font & Text Colors
       ---------------------------------------------------------- */
    .stMarkdown, p, span, div {
        color: #ffffff !important;
    }
    h1, h2, h3 {
        color: #ffffff !important;
    }

    /* ----------------------------------------------------------
       4. Chat Container & Message Bubbles
       ---------------------------------------------------------- */
    .stChatContainer {
        background-color: #000000 !important;
    }
    .stChatMessage {
        background-color: #1a1a1a !important;
    }
    .stChatMessageContent {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
    }
    /* Avatars */
    .stChatMessage > div:first-child {
        background-color: black !important;
        border: 2px solid black !important;
    }
    .stChatMessage > div:first-child > div {
        border: 2px solid black !important;
    }

    /* ----------------------------------------------------------
       5. Text Inputs (General)
       ---------------------------------------------------------- */
    .stTextInput > div > div > input {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
        caret-color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    input[type="password"] {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        caret-color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    input:focus {
        outline: none !important;
        caret-color: #ffffff !important;
        caret-animation: blink 1s infinite;
    }
    input::selection {
        background-color: rgba(255, 255, 255, 0.2) !important;
        color: #ffffff !important;
    }

    /* ----------------------------------------------------------
       6. Chat-Specific Input Fields
       ---------------------------------------------------------- */
    .stChatInputContainer {
        background-color: #1a1a1a !important;
    }
    /* The <textarea> for chat */
    textarea {
        color: #ffffff !important;
        caret-color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        opacity: 1 !important;
    }
    .stChatInput {
        color: #ffffff !important;
    }
    [data-testid="stChatInput"] {
        color: #ffffff !important;
        background-color: #1a1a1a !important;
    }
    [data-testid="stChatInput"] textarea {
        color: #ffffff !important;
        background-color: #1a1a1a !important;
        caret-color: #ffffff !important;
        border: 1px solid #333333 !important;
    }
    textarea:focus {
        outline: none !important;
        caret-color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        caret-animation: blink 1s infinite;
    }
    textarea::selection {
        background-color: rgba(255, 255, 255, 0.2) !important;
        color: #ffffff !important;
    }

    /* Cursor blink animation */
    @keyframes blink {
        0% { opacity: 1; }
        50% { opacity: 0; }
        100% { opacity: 1; }
    }

    /* ----------------------------------------------------------
       7. Buttons
       ---------------------------------------------------------- */
    .stButton > button {
        width: 100%;
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
    }

    /* The form submit button (sidebar "Enter" button) */
    [data-testid="stFormSubmitButton"] button {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
        width: 100% !important;
        font-size: 1rem !important;
        margin-top: 0.5rem !important;

        /* Ensure it's not disabled by CSS */
        pointer-events: auto !important;
        cursor: pointer !important;
        opacity: 1 !important;
    }

    /* The chat submit button */
    [data-testid="stChatInputSubmitButton"] {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
    }

    /* ----------------------------------------------------------
       8. Other UI Elements
       ---------------------------------------------------------- */
    .api-status {
        margin-top: 10px;
        padding: 10px;
        border-radius: 4px;
        background-color: #1a1a1a !important;
        text-align: center;
    }
    .stAlert {
        background-color: #1a1a1a !important;
        color: #ff4444 !important;
    }
    ::-webkit-scrollbar {
        background: #000000 !important;
        width: 10px;
    }
    ::-webkit-scrollbar-thumb {
        background: #333333 !important;
        border-radius: 5px;
    }
    div[data-testid="stDecoration"] {
        background-color: #000000 !important;
    }
    div[data-testid="stStatusWidget"] {
        background-color: #000000 !important;
    }
    iframe {
        background-color: #000000 !important;
    }

    /* ----------------------------------------------------------
       9. Additional Overriding for Bottom Regions
       ---------------------------------------------------------- */
    [data-testid="stBottom"] {
        background-color: #000000 !important;
    }
    [data-testid="stBottomBlockContainer"] {
        background-color: #000000 !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #000000 !important;
    }
    [data-testid="stVerticalBlock"] {
        background-color: #000000 !important;
    }
    [data-testid="stElementContainer"] {
        background-color: #000000 !important;
    }

    /* Example for any additional st-emotion-cache classes if needed */
    .st-emotion-cache-128upt6.ekr3hml3 {
        background-color: #000000 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize session state for chat history and API key status
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'api_key_active' not in st.session_state:
    st.session_state.api_key_active = False

def load_openai_key():
    """Load OpenAI API key from environment variable or user input"""
    try:
        return os.environ["OPENAI_API_KEY"]
    except KeyError:
        return None

def initialize_streamlit():
    """Initialize Streamlit app with title and sidebar"""
    # Main title and description
    st.title("🤖 Deo AI")
    st.write("Ask me anything about crypto, blockchain, or web3!")
    st.write("If you have a DAO proposal you want to optimize, please provide me the following:")
    st.write("1. The name of the DAO you're submitting the proposal to.")
    st.write("2. The proposal you want to optimize (I support proposals in 50+ languages and returns the optimized proposal to you in English).")
    st.write("3. The number of recent proposals to analyze for this DAO (If you are using our provided OpenAI API key, you can analyze up to 50 recent proposals. If you are using your OpenAI API key, if 50 proposals returns an error, then analyze up to 20 to 25 recent proposals. This is due to token rate limits set by OpenAI depending on the Tier level your OpenAI account is on).")

    # Sidebar title and API Key Management
    st.sidebar.title("🤖 Deo AI")
    
    default_key = load_openai_key()
    
    # Create a form in the sidebar
    with st.sidebar.form(key='api_key_form'):
        user_api_key = st.text_input(
            "Enter your OpenAI API key (if monthly free credits are exhausted):",
            type="password"
        )
        submit_button = st.form_submit_button("Enter")
        
        if submit_button and user_api_key:
            st.session_state.api_key_active = True
        elif submit_button:
            st.session_state.api_key_active = False
    
    # API Key status indicator (outside the form)
    if st.session_state.api_key_active:
        st.sidebar.markdown(
            """
            <div class='api-status' style='color: #00ff00;'>
                ✓ Custom API key active
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    return user_api_key if user_api_key and st.session_state.api_key_active else default_key

# This helper function escapes $ signs so they are not interpreted as LaTeX.
def sanitize_dollar_signs(text: str) -> str:
    """
    Convert every raw '$' into HTML entity '&#36;', so it displays
    as a literal dollar sign in the Streamlit UI rather than LaTeX.
    """
    return text.replace("$", "&#36;")

# Creating langgraph agent tool
@tool
def dao_proposal_optimizer(dao_name: str, initial_proposal: str, num_proposals: int = 25) -> str:
    """
    Optimizes a DAO proposal based on analysis of historical DAO data.
    
    Args:
        dao_name (str): Name of the DAO
        initial_proposal (str): The initial proposal to optimize
        num_proposals (int): Number of recent proposals to analyze
        
    Returns:
        str: A string containing both the DAO analysis (hidden from user) and
             the final optimized proposal in a labeled format.
        
    Raises:
        ToolException: If DAO is not found or other errors occur
    """
    
    def get_space_id() -> str:
        url = "https://hub.snapshot.org/graphql"
        query = """
        query($skip: Int!) {
            spaces(
                first: 1000,
                skip: $skip,
                where: {
                    verified: true
                }
            ) {
                id
                name
            }
        }
        """
        
        def clean_name(name: str) -> str:
            name = name.lower()
            name = re.sub(r'\b(dao|protocol|finance|v[0-9]+|governance)\b', '', name)
            name = ' '.join(name.split())
            return name
        
        def calculate_similarity(name1: str, name2: str, space_id: str) -> float:
            if name1.lower() == space_id.lower():
                return 1.0
            
            name1_clean = clean_name(name1)
            name2_clean = clean_name(name2)
            
            if name1_clean == name2_clean:
                return 1.0
                
            base_similarity = difflib.SequenceMatcher(None, name1_clean, name2_clean).ratio()
            
            if name1_clean in name2_clean or name2_clean in name1_clean:
                return base_similarity + 0.3
            
            length_ratio = min(len(name1_clean), len(name2_clean)) / max(len(name1_clean), len(name2_clean))
            if length_ratio < 0.5:
                return base_similarity * 0.5
                
            words1 = set(name1_clean.split())
            words2 = set(name2_clean.split())
            if words1 and words2:
                word_match_ratio = len(words1.intersection(words2)) / max(len(words1), len(words2))
                if len(words1.intersection(words2)) == 0:
                    return base_similarity * 0.3
            else:
                word_match_ratio = 0
            
            return min((base_similarity * 0.7) + (word_match_ratio * 0.3), 1.0)
        
        try:
            skip = 0
            all_spaces = []
            
            while True:
                response = requests.post(
                    url,
                    json={
                        "query": query,
                        "variables": {"skip": skip}
                    }
                )
                
                if response.status_code != 200:
                    raise ToolException(f"Snapshot API request failed: {response.text}")
                    
                data = response.json()
                spaces = data.get("data", {}).get("spaces", [])
                
                if not spaces:
                    break
                    
                all_spaces.extend(spaces)
                skip += 1000
            
            if not all_spaces:
                raise ToolException("No spaces found on Snapshot")
            
            matches = []
            for space in all_spaces:
                if not space.get('name') or not space.get('id'):
                    continue
                
                similarity = calculate_similarity(dao_name, space['name'], space['id'])
                matches.append((similarity, space['id'], space['name']))
            
            matches.sort(reverse=True)
            
            if matches and matches[0][0] > 0.6:
                return matches[0][1]
            else:
                raise ToolException(f"DAO '{dao_name}' not found on Snapshot platform")
                
        except ToolException:
            raise
        except Exception as e:
            raise ToolException(f"Error finding DAO space: {str(e)}")

    def get_dao_data(space_id: str) -> Dict[str, Any]:
        if space_id.startswith("'") and space_id.endswith("'"):
            space_id = space_id[1:-1]

        API_URL = "https://hub.snapshot.org/graphql"
        space_query = """
        query GetSpaceData($space_id: String!, $num_proposals: Int!) {
            space(id: $space_id) {
                id
                name
                about
                avatar
                network
                symbol
                strategies {
                    name
                    params
                }
                admins
                moderators
                members
                filters {
                    minScore
                    onlyMembers
                }
                plugins
            }
            proposals(
                first: $num_proposals,
                skip: 0,
                where: {
                    space: $space_id
                },
                orderBy: "created",
                orderDirection: desc
            ) {
                id
                title
                body
                choices
                start
                end
                snapshot
                state
                author
                created
                scores
                scores_total
            }
        }
        """
        
        try:
            response = requests.post(
                API_URL,
                json={
                    "query": space_query,
                    "variables": {
                        "space_id": space_id,
                        "num_proposals": num_proposals
                    }
                }
            )
            response.raise_for_status()
            data = response.json()
            
            if "data" in data and data["data"]["space"] is None:
                raise ToolException(f"DAO space '{space_id}' not found")
                
            if "errors" in data:
                raise ToolException(f"Error fetching DAO data: {data['errors']}")
                
            return {
                "space": data["data"]["space"],
                "proposals": data["data"]["proposals"]
            }
            
        except requests.exceptions.RequestException as e:
            raise ToolException(f"API request failed: {str(e)}")
        except Exception as e:
            raise ToolException(f"Error fetching DAO data: {str(e)}")

    def analyze_dao_data(dao_data: Dict[str, Any]) -> str:
        try:
            if not os.getenv("OPENAI_API_KEY"):
                raise ToolException("OpenAI API key not found in environment variables")
                
            client = OpenAI()
            
            space_info = json.dumps(dao_data["space"])
            proposals_info = json.dumps(dao_data["proposals"])
            
            analysis_prompt = f"""
            As a DAO governance proposal expert, analyze this DAO data obtained from the DAO governance platform Snapshot.
            
            DAO Info:
            {space_info}

            Key components in DAO Info:
            'id': Snapshot space ID for the DAO
            'name': The display name of the DAO
            'about': Contains voting information for the DAO
            'avatar': A link to the DAO's logo/avatar image
            'network': A number given by Snapshot representing which blockchain network the DAO is in
            'symbol': The governance token symbol
            'strategies': Defines how voting power is calculated containing 'name' which is the name of the strategy and 'params' which are the parameters including 'symbol' which is the governance token symbol, 'address' which is the token address and 'decimals' which is the number of decimal places the token can be divided into
            'admins': Contains the admin crypto wallet address
            'moderators': Contains who the moderators are
            'members': Contains who the members are
            'filters': Contains voting rules including 'minScore' which is the minimum voting power required and 'onlyMembers' which determines if voting is or is not restricted to members only
            'plugins': Contains configuration of additional plugins

            Historical Proposals:
            {proposals_info}

            Key components in Historical Proposals:
            'id': Unique identifier for the proposal
            'title': Title of the proposal
            'body': Full proposal text/content
            'choices': Array of voting options/choices
            'start': Start Unix format timestamp for voting
            'end': End Unix format timestamp for voting
            'snapshot': Block number for the snapshot
            'state': Current state of the proposal if it is still active or closed
            'author': Cryto wallet address of the proposal creator
            'created': Unix format timestamp when proposal was created
            'scores': Array of vote counts for each corresponding options/choices in 'choices'
            'scores_total': Total votes cast
            
            Based on this data, do a comprehensive analysis for each of the following points:
            1. Analyze DAO goals and community preferences
            2. Study successful vs failed proposals
            3. Analyze successful proposal titles and patterns
            4. Identify optimal proposal structure and language (eg. average proposal length, common sections used etc) of successful proposals
            5. Check for potential contradictions with other proposals
            6. Assess security risks and governance attack vectors
            7. Evaluate potential governance mistakes to avoid
            8. Consider harmonious proposal combinations
            9. Evaluate optimal voting duration of successful proposals

            Output:
            The comprehensive analysis done above of the DAO data. DO NOT mention examples of any protocols or projects in the output analysis.
            """
            
            analysis_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert in DAO (decentralized autonomous organization) governance proposals."},
                    {"role": "user", "content": analysis_prompt}
                ]
            )

            return analysis_response.choices[0].message.content
            
        except Exception as e:
            raise ToolException(f"Error analyzing DAO data: {str(e)}")

    def optimize_proposal(english_proposal: str, dao_data_analysis: str) -> str:
        try:
            client = OpenAI()
            
            optimize_prompt = f"""
            As a DAO governance proposal optimization expert, optimize the initial proposal below based on the DAO data analysis obtained from the DAO platform Snapshot to obtain a higher passing rate.

            Initial Proposal:
            {english_proposal}
            
            DAO Data Analysis:
            {dao_data_analysis}

            In addition, for the optimized proposal use clear, professional language that balances technical and non-technical understanding.

            Output:
            1. An optimized version of the initial proposal based on the DAO data analysis
            2. A bullet-point list of all changes and recommendations made

            Keep the output focused only on the optimized proposal and the list of changes/recommendations. Do not output anything else.
            """
            
            optimization_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert in DAO (decentralized autonomous organization) governance and proposal optimization."},
                    {"role": "user", "content": optimize_prompt}
                ]
            )
            
            return optimization_response.choices[0].message.content
            
        except Exception as e:
            raise ToolException(f"Error optimizing proposal: {str(e)}")

    try:
        space_id = get_space_id()
        dao_data = get_dao_data(space_id)
        dao_analysis = analyze_dao_data(dao_data)
        
        client = OpenAI()
        translation_prompt = f"""
        Detect the language of this text and if it's not English, translate it to English. 
        If this text is in English, then output the exact same text word for word:
        {initial_proposal}
        """
        translation_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a language detection and translation expert."},
                {"role": "user", "content": translation_prompt}
            ]
        )
        english_proposal = translation_response.choices[0].message.content

        optimized_text = optimize_proposal(english_proposal, dao_analysis)
           
        result_text = (
            f"DAO_ANALYSIS:\n{dao_analysis}\nEND_ANALYSIS\n\n"
            f"OPTIMIZED_PROPOSAL:\n{optimized_text}"
        )
        return result_text
        
    except ToolException as te:
        raise ToolException(str(te))
    except Exception as e:
        raise ToolException(f"Error in proposal optimization process: {str(e)}")

def convert_messages(messages):
    """
    Convert dictionary-based messages to LangChain message objects.
    """
    converted_messages = []
    
    message_types = {
        "system": SystemMessage,
        "user": HumanMessage,
        "assistant": AIMessage
    }
    
    for message in messages:
        role = message["role"]
        content = message["content"]
        
        if role in message_types:
            MessageClass = message_types[role]
            converted_message = MessageClass(content=content)
            converted_messages.append(converted_message)
            
    return converted_messages

def create_chat_completion(api_key, user_prompt, message_placeholder):
    """Create streaming chat completion using OpenAI API"""
    try:
        client = OpenAI(api_key=api_key)
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant who provides responses to user questions based on the context in "
                    "cryptocurrency, blockchain and web3 only. If you are asked to optimized a DAO proposal, use "
                    "the 'dao_proposal_optimizer' tool. If the user did not provided the three inputs (ie. the "
                    "name of the DAO to submit the proposal to, the DAO proposal to optimize and the number of "
                    "recent DAO proposals from Snapshot to analyze) required to use the 'dao_proposal_optimizer' "
                    "tool, then ask the user to provide any of these three required inputs which are missing. "
                    "If the 'dao_proposal_optimizer' tool is used, only output what is outputted from this tool."
                )
            }
        ]
        
        # Add previous messages
        messages.extend(st.session_state.messages)
        
        # Add current user prompt
        messages.append({
            "role": "user",
            "content": user_prompt
        })

        converted_messages = convert_messages(messages)
        
        model = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
        tools = [dao_proposal_optimizer]

        with st.spinner('Thinking...'):
            agent_executor = create_react_agent(model, tools)
            response = agent_executor.invoke({"messages": converted_messages})

            last_response = response["messages"][-1].content
        
            # If hidden analysis is present, separate from the final output
            if "DAO_ANALYSIS:" in last_response and "END_ANALYSIS" in last_response:
                # Extract the analysis
                analysis_part = last_response.split("DAO_ANALYSIS:")[1].split("END_ANALYSIS")[0].strip()
                # Append hidden system message with the analysis
                st.session_state.messages.append({
                    "role": "system", 
                    "content": analysis_part
                })
                
                # If there's an optimized proposal part, parse it out
                if "OPTIMIZED_PROPOSAL:" in last_response:
                    # Show only the optimized proposal portion to user
                    last_response = last_response.split("OPTIMIZED_PROPOSAL:")[1].strip()
                else:
                    # Fallback if something unexpected
                    last_response = "No optimized proposal found."

            # Here we escape dollar signs so they don't get interpreted as LaTeX.
            safe_last_response = sanitize_dollar_signs(last_response)
            
            accumulated_response = ""
            for char in safe_last_response:
                accumulated_response += char
                message_placeholder.markdown(sanitize_dollar_signs(accumulated_response) + "▌")

            message_placeholder.markdown(sanitize_dollar_signs(safe_last_response))

            return safe_last_response, None

    except Exception as e:
        error_str = str(e).lower()
        if "insufficient_quota" in error_str or "billing" in error_str:
            return None, "Monthly API credits exhausted. Please enter your OpenAI API key in the sidebar to continue using Deo AI."
        else:
            return None, f"An error occurred: {str(e)}"

def main():
    api_key = initialize_streamlit()
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'dao_analysis' not in st.session_state:
        st.session_state.dao_analysis = None
    
    # Display chat history with custom avatars
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user", avatar="🦖"):
                # Escape $ before displaying user messages
                display_text = sanitize_dollar_signs(message["content"])
                st.markdown(display_text)
        elif message["role"] == "assistant":
            with st.chat_message("assistant", avatar="🤖"):
                display_text = sanitize_dollar_signs(message["content"])
                st.markdown(display_text)
        else:
            # Skip system messages in the UI
            pass
    
    if prompt := st.chat_input("Type your message here..."):
        if not api_key:
            st.error("Please provide an OpenAI API key in the sidebar to use Deo AI.")
            return
        
        # Display user message with custom avatars
        with st.chat_message("user", avatar="🦖"):
            safe_prompt = sanitize_dollar_signs(prompt)
            st.markdown(safe_prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display assistant message with custom avatars
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            response, error = create_chat_completion(api_key, prompt, message_placeholder)
            
            if error:
                if "Monthly API credits exhausted" in error:
                    st.sidebar.error(error)
                else:
                    st.error(error)
            else:
                # Store the response in chat history
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
