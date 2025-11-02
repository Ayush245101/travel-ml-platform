import os
import pandas as pd
import streamlit as st
import joblib
import altair as alt

st.set_page_config(page_title="Travel Recommender & Insights", layout="wide")

FLIGHT_CSV = os.getenv("FLIGHT_CSV", "data/flight.csv")
HOTEL_CSV = os.getenv("HOTEL_CSV", "data/hotal.csv")
USER_CSV = os.getenv("USER_CSV", "data/user.csv")
RECOMMENDER_ARTIFACT = os.getenv("RECOMMENDER_ARTIFACT", "models/recommender.pkl")

@st.cache_data
def load_data():
    flights = pd.read_csv(FLIGHT_CSV)
    hotels = pd.read_csv(HOTEL_CSV)
    users = pd.read_csv(USER_CSV)
    return flights, hotels, users

@st.cache_resource
def load_recommender():
    if os.path.exists(RECOMMENDER_ARTIFACT):
        return joblib.load(RECOMMENDER_ARTIFACT)
    return None

def main():
    st.title("Travel Recommendation and Insights")
    flights, hotels, users = load_data()
    artifact = load_recommender()

    st.sidebar.header("User Selection")
    user_ids = users["code"].tolist()
    selected_user = st.sidebar.selectbox("User code", options=user_ids)

    st.header("User Insights")
    uflights = flights[flights["userCode"] == selected_user]
    uhotels = hotels[hotels["userCode"] == selected_user]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Flights by Destination")
        top_to = uflights["to"].value_counts().head(10).reset_index()
        top_to.columns = ["destination", "count"]
        chart = alt.Chart(top_to).mark_bar().encode(x="destination", y="count")
        st.altair_chart(chart, use_container_width=True)
    with col2:
        st.subheader("Hotel Stays by Place")
        top_places = uhotels["place"].value_counts().head(10).reset_index()
        top_places.columns = ["place", "count"]
        chart2 = alt.Chart(top_places).mark_bar().encode(x="place", y="count")
        st.altair_chart(chart2, use_container_width=True)

    st.subheader("Spending Overview")
    flights_spend = uflights["price"].sum()
    hotels_spend = uhotels["total"].sum()
    st.metric("Total Flight Spend", f"${flights_spend:,.2f}")
    st.metric("Total Hotel Spend", f"${hotels_spend:,.2f}")

    st.header("Recommendations")
    if artifact is not None:
        from src.train_recommender import recommend_for_user
        recs = recommend_for_user(selected_user, artifact, k=5)
        st.dataframe(recs)
    else:
        st.warning("Recommender artifact not found. Run train_recommender.py first.")

if __name__ == "__main__":
    main()