import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import requests
from io import StringIO

# Try to import GROQ
try:
    from groq import Groq
    groq_available = True
except ImportError:
    groq_available = False
    st.warning("GROQ not available. Please install: pip install groq")

try:
    from langchain_groq import ChatGroq
    from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
    langchain_available = True
except ImportError:
    langchain_available = False
    st.warning("⚠️ LangChain not installed. AI Assistant will have limited capabilities.")
    st.info("For full AI code execution features, install: `pip install langchain-groq langchain-experimental`")
    

# Page setup
st.set_page_config(
    page_title="Beer AI Analytics", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #ff6b35 0%, #f7931e 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        border-left: 4px solid #ff6b35;
    }
    .chat-container {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border: 1px solid #dee2e6;
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 18px;
        border-radius: 18px;
        margin: 8px 0;
        margin-left: 50px;
        text-align: right;
    }
    .assistant-message {
        background-color: #e9ecef;
        color: #495057;
        padding: 12px 18px;
        border-radius: 18px;
        margin: 8px 0;
        margin-right: 50px;
        border-left: 3px solid #ff6b35;
    }
    .stButton > button {
        background: linear-gradient(90deg, #ff6b35 0%, #f7931e 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
    }
    .analytics-section {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    .ai-section {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-top: 3px solid #ff6b35;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None

# Function to load data from GitHub
@st.cache_data
def load_data_from_github():
    """Load sample sales data from GitHub repository"""
    # Replace with your actual GitHub raw file URL
    github_url = "https://raw.githubusercontent.com/Yuriiiii9/InsightDock/main/sample_data.csv"
    
    try:
        response = requests.get(github_url)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
            return df
        else:
            st.error(f"Could not load data from GitHub. Status code: {response.status_code}")
            st.stop()
    except Exception as e:
        st.error(f"Error loading data from GitHub: {e}")
        st.stop()

# Main header
st.markdown("""
<div class="main-header">
    <h1>🍺 Beer AI Analytics Dashboard</h1>
    <p>Intelligent Business Insights for Craft Brewery Operations</p>
</div>
""", unsafe_allow_html=True)

# Load data
if not st.session_state.data_loaded:
    with st.spinner("Loading sales data..."):
        df = load_data_from_github()
        st.session_state.df = df
        st.session_state.data_loaded = True
        st.success("✅ Data loaded successfully!")

df = st.session_state.df

# ==================== ANALYTICS DASHBOARD (TOP SECTION) ====================
st.markdown('<div class="analytics-section">', unsafe_allow_html=True)
st.header("📊 Business Intelligence Dashboard")

# Key metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_sales = df['Sales'].sum()
    st.markdown(f"""
    <div class="metric-card">
        <h3>${total_sales:,.0f}</h3>
        <p>Total Sales</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    total_orders = len(df)
    st.markdown(f"""
    <div class="metric-card">
        <h3>{total_orders:,}</h3>
        <p>Total Orders</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    unique_accounts = df['Account Name'].nunique()
    st.markdown(f"""
    <div class="metric-card">
        <h3>{unique_accounts}</h3>
        <p>Active Accounts</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    total_bottles = df['Total Bottles'].sum()
    st.markdown(f"""
    <div class="metric-card">
        <h3>{total_bottles:,.0f}</h3>
        <p>Bottles Sold</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Charts section
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Monthly Sales Trend")
    try:
        monthly_sales = df.groupby(['Year', 'Month'])['Sales'].sum().reset_index()
        monthly_sales['Date'] = pd.to_datetime(monthly_sales[['Year', 'Month']].assign(day=1))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(monthly_sales['Date'], monthly_sales['Sales'], 
                marker='o', linewidth=3, markersize=8, 
                color='#ff6b35', markerfacecolor='#f7931e')
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Sales ($)', fontsize=12)
        ax.set_title('Monthly Sales Performance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error creating sales trend: {e}")

with col2:
    st.subheader("🍺 Product Performance")
    try:
        product_sales = df.groupby('Product Line')['Sales'].sum().sort_values(ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(product_sales.index, product_sales.values, color='#ff6b35', alpha=0.8)
        ax.set_xlabel('Total Sales ($)', fontsize=12)
        ax.set_ylabel('Product Line', fontsize=12)
        ax.set_title('Sales by Product Line', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'${width:,.0f}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error creating product chart: {e}")

# Channel and Geographic Analysis
col1, col2 = st.columns(2)

with col1:
    st.subheader("🏪 Sales by Channel")
    try:
        channel_sales = df.groupby('Sales Channel Name')['Sales'].sum().sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = ['#ff6b35', '#f7931e', '#ffaa44', '#ffcc77', '#ffdd99']
        wedges, texts, autotexts = ax.pie(channel_sales.values, labels=channel_sales.index, 
                                         autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('Revenue Distribution by Sales Channel', fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error creating channel chart: {e}")

with col2:
    st.subheader("🌍 Geographic Distribution")
    try:
        province_sales = df.groupby('Province')['Sales'].sum().sort_values(ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(province_sales.index, province_sales.values, color='#f7931e', alpha=0.8)
        ax.set_xlabel('Total Sales ($)', fontsize=12)
        ax.set_ylabel('Province', fontsize=12)
        ax.set_title('Sales by Province', fontsize=14, fontweight='bold')
        
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'${width:,.0f}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error creating geographic chart: {e}")

st.markdown('</div>', unsafe_allow_html=True)

# ==================== AI ASSISTANT (BOTTOM SECTION) ====================
st.markdown('<div class="ai-section">', unsafe_allow_html=True)
st.header("🤖 AI Sales Intelligence Assistant")

if groq_available:
    groq_api_key = os.environ.get("GROQ_API_KEY", "")
    
    if groq_api_key:
        st.success("✅ AI Assistant is ready!")
        
        # Sample questions
        st.markdown("**💡 Try asking questions like:**")
        sample_questions = [
            "What are my top 3 performing products?",
            "Which sales channel generates the most revenue?", 
            "How did sales perform in Q4 compared to Q3?",
            "What's the average order value by province?",
            "Which accounts should I focus on for growth?"
        ]
        
        cols = st.columns(len(sample_questions))
        for i, question in enumerate(sample_questions):
            with cols[i]:
                if st.button(f"❓ {question}", key=f"sample_{i}"):
                    st.session_state.chat_messages.append({"role": "user", "content": question})
        
        # Display chat history
        st.markdown("### 💬 Conversation")
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_messages:
                if message["role"] == "user":
                    st.markdown(f'<div class="user-message">🧑‍💼 {message["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="assistant-message">🤖 {message["content"]}</div>', unsafe_allow_html=True)
        
        # Chat input
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input("Ask about your brewery's performance...", 
                                     placeholder="e.g., What are my best-selling products this year?",
                                     key="chat_input")
        with col2:
            send_button = st.button("Send 📤", type="primary")
            clear_chat = st.button("Clear 🗑️")
        
        if clear_chat:
            st.session_state.chat_messages = []
            st.rerun()
        
        if (send_button and user_input) or any([message for message in st.session_state.chat_messages if message["role"] == "user" and message["content"] not in [msg["content"] for msg in st.session_state.chat_messages[:-1] if msg["role"] == "user"]]):
            
            # Get the latest user message
            try:
                st.info("🔄 Using LangChain with Python code execution...")
                
                llm = ChatGroq(
                    groq_api_key=groq_api_key, 
                    model="llama3-8b-8192", 
                    temperature=0.1,
                    max_tokens=2000
                )
                
                # 更详细的提示
                enhanced_question = f"""
Please analyze the brewery sales data to answer: {user_input}

Steps:
1. First examine the data structure 
2. Perform calculations using pandas
3. Provide specific insights with numbers
4. Give actionable recommendations

Make sure to show your analysis process and provide a clear final answer.
"""
        
                agent = create_pandas_dataframe_agent(
                    llm, 
                    df, 
                    allow_dangerous_code=True,
                    verbose=True,
                    handle_parsing_errors=True,
                    max_iterations=3  # 限制迭代避免无限循环
                )
                
                st.write("🤖 Agent is thinking and coding...")
                result = agent.invoke({"input": enhanced_question})  # 使用invoke而不是run
                
                # 处理不同的返回格式
                if isinstance(result, dict):
                    response_text = result.get('output', str(result))
                else:
                    response_text = str(result)
                    
                st.write(f"📝 Raw result: {response_text[:500]}...")  # 显示原始结果
                
                if response_text and response_text.strip():
                    st.success("✅ LangChain analysis completed!")
                else:
                    st.warning("Empty result from LangChain")
                    response_text = None
                
            except Exception as e:
                st.error(f"LangChain failed: {e}")
                response_text = None
                                
            else:
                st.warning("🔑 GROQ_API_KEY not found in environment variables.")
                st.info("Please set your GROQ_API_KEY in the deployment settings.")

else:
    st.warning("📦 GROQ library not available.")
    st.code("pip install groq", language="bash")

# Footer
st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    🍺 <strong>Nonny Beer AI Analytics Dashboard</strong> | 
    Built with Streamlit & GROQ AI | 
    <em>Showcasing AI Product Management Excellence</em>
</div>
""", unsafe_allow_html=True)
