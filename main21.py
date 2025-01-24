# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 00:45:04 2025

@author: raypa
"""

import streamlit as st
from datetime import datetime, timedelta
import praw
import os
import time
import pandas as pd
#from praw.exceptions import NotFound
from social_monitoring import display
from ydata_profiling import ProfileReport
from prawcore.exceptions import NotFound

st.set_page_config(page_title="Data Quality Monitoring Tool", layout="wide")

# Initialize session state for navigation
if "page" not in st.session_state:
    st.session_state.page = "Home"

# Define a function to switch pages
def set_page(page_name):
    st.session_state.page = page_name

# Home Page
if st.session_state.page == "Home":
    st.markdown("""
        <style>
            .stTextInput label {
                font-size: 25px;
                color:green;
                font-weight:bold;
            }
            .stTextInput input {
                height: 50px;
                font-size: 18px;
                padding: 20px;
                width: 100%;
            }
            .stDateInput input {
                font-size: 16px;
                height: 35px;
                padding: 10px;
                width: 250px;
            }
            .stDateInput label {
                font-weight:bold;
                font-size: 18px;
                color:green;
            }
            .stDateInput input:focus {
                border: 2px solid #ff6347;
            }
            .stNumberInput input {
                font-size: 16px;
                height: 35px;
                padding: 10px;
                width: 150px;
                border-radius: 5px;
            }
            .stNumberInput label {
                font-weight: bold;
                font-size: 18px;
                color:green;
            }
            .stNumberInput input:focus {
                border: 2px solid #32CD32;
            }
        </style>
    """, unsafe_allow_html=True)

    # Set up Reddit API access
    reddit = praw.Reddit(
        client_id='1UCDggbRZb-jQjY7OZdxEQ',        # Replace with your app's client ID
        client_secret='xkrfZSO9B20NNHGRBU26JCFg6zbNrg', # Replace with your app's client secret
        user_agent='TEST_TOOL1/u/One-Bee-8526'       # e.g., 'myBot v1.0'
    )

    # Function to fetch Reddit posts and comments
    def fetch_reddit_data(search_query, subreddit, start_date, end_date, limit=500):
        data = []
        start_timestamp = int(start_date.timestamp())
        end_timestamp = int(end_date.timestamp())

        try:
            for submission in reddit.subreddit(subreddit).search(search_query, sort='new', limit=limit):
                try:
                    if not (start_timestamp <= submission.created_utc <= end_timestamp):
                        continue
                    post_date = datetime.utcfromtimestamp(submission.created_utc).strftime('%Y-%m-%d %H:%M:%S')
                    submission.comments.replace_more(limit=0)
                    for comment in submission.comments.list():
                        try:
                            comment_date = datetime.utcfromtimestamp(comment.created_utc).strftime('%Y-%m-%d %H:%M:%S')
                            data.append({
                                'Post ID': submission.id,
                                'Post Title': submission.title,
                                'Post URL': submission.url,
                                'Post Score': submission.score,
                                'Posted by': str(submission.author),
                                'Post Date': post_date,
                                'NSFW': submission.over_18,
                                'Number of Comments': submission.num_comments,
                                'Comment ID': comment.id,
                                'Comment Author': str(comment.author),
                                'Comment Score': comment.score,
                                'Comment Body': comment.body,
                                'Comment Date': comment_date
                            })
                        except Exception:
                            st.warning("Error fetching comment data.")
                            continue
                    time.sleep(1)
                except Exception:
                    st.warning("Error fetching post data.")
                    continue
        except NotFound:
            st.error("Error 404: Subreddit not found or inaccessible. Please check the subreddit name.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
        return data

    with st.expander("", expanded=True):
        col1, col2 = st.columns([2, 1])  # Adjust columns: 2 for left, 1 for right

        with col1:
            # Streamlit app UI
            st.markdown("""
                <h1 style='text-align: center; color: blue; font-weight: bold; font-family: "Courier New", monospace;'>
                RedditIQ:</h1> <h4 style= color: black;> Social Data Quality Monitoring Tool with Reddit Post
                </h4>
            """, unsafe_allow_html=True)
            st.write("Fetch Reddit posts and comments based on a search query, subreddit, and date range.")
            search_query = st.text_input("Enter a Search Query:", value="", placeholder="Type a query to search for posts")

            # Input fields
            subreddit = st.text_input("Enter a Subreddit (e.g., 'all', 'worldnews'):", value="")

            # Date inputs
            start_date = st.date_input("Start Date:", value=datetime.now() - timedelta(days=15))
            end_date = st.date_input("End Date:", value=datetime.now())

            if start_date > end_date:
                st.error("Start date cannot be after end date.")
            else:
                start_date = datetime.combine(start_date, datetime.min.time())
                end_date = datetime.combine(end_date, datetime.max.time())

            limit = st.number_input("Number of Posts to Fetch:", min_value=1, max_value=500, value=20)

            if "reddit_data" not in st.session_state:
                st.session_state.reddit_data = pd.DataFrame()

            # Fetch data on button click
            if st.button("Fetch Data"):
                if not search_query.strip() or not subreddit.strip():
                    st.error("Both 'Search Query' and 'Subreddit' fields are required and cannot be empty.")      
                else:
                    with st.spinner("Fetching data..."):
                        data = fetch_reddit_data(search_query, subreddit, start_date, end_date, limit)
                        if data:
                            st.success("Data fetched successfully!")
                            df = pd.DataFrame(data)

                            # Display the data
                            st.write(f"Showing {len(df)} records:")

                            # Downloadable CSV
                            csv = df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download data as CSV",
                                data=csv,
                                file_name=f'reddit_posts_comments_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                                mime='text/csv',
                            )
                            save_path = 'C:/py/reddit/reddit_data_25.csv'
                            os.makedirs('./saved_data', exist_ok=True)
                            df.to_csv(save_path, index=False)
                            st.success("Data saved")

                            # Save the data to session state
                            st.session_state["reddit_data"] = df
                        else:
                            st.warning("No data fetched. Try a different query, subreddit, or date range.")
            
            st.button("Social Data Monitoring", on_click=set_page, args=("social",))
           
        with col2:
            st.image("dd.jpg", caption="", use_column_width=True)

elif st.session_state.page == "social":
    if st.session_state.reddit_data.empty:
        st.markdown('<p style="color:red; font-weight:bold;">No data available.</p>', unsafe_allow_html=True)
        st.button("Go to Home", on_click=set_page, args=("Home",))
    else:
        display()
        st.button("Go to Home", on_click=set_page, args=("Home",))
