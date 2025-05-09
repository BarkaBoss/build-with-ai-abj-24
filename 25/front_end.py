import streamlit as st
from multi_agent import run_agents

st.set_page_config(page_title="Multi-Agent Shop Assistant", layout="centered")

st.title("🛒 Multi-Agent Shopping Assistant")
st.write("This app checks item availability and nutritional info using multiple AI agents.")

with st.form("purchase_form"):
    item = st.text_input("Enter item name", value="banana")
    quantity = st.number_input("Enter quantity", min_value=1, value=1)
    submitted = st.form_submit_button("Check Item")

if submitted:
    with st.spinner("🤖 Agents are working..."):
        convo, stock, nutrition = run_agents(item, quantity)

    st.subheader("🧠 User Intent (Conversation Agent)")
    st.info(convo)

    st.subheader("📦 Inventory Check (Inventory Agent)")
    st.success(stock)

    st.subheader("🥗 Nutrition Facts (Nutrition Agent)")
    st.write(nutrition)
