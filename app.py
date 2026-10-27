import html
import re
from urllib.parse import quote
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------
# CINESCOPE AI — MODERN INTERACTIVE FRONT END
# ---------------------------------------------------------

BASE_DIR = Path(__file__).parent

st.set_page_config(
    page_title="CineScope AI | Movie Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_bundle():
    return joblib.load(BASE_DIR / "models_bundle.joblib")


@st.cache_data
def load_csv(filename, modified_time=None):
    path = BASE_DIR / filename
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


bundle = load_bundle()
models = bundle["models"]
vectorizer = bundle["vectorizer"]
catalog = load_csv("movie_catalog.csv", (BASE_DIR / "movie_catalog.csv").stat().st_mtime)
reviews = load_csv("review_library.csv")
training_data = load_csv("movie_reviews.csv")
metrics = pd.DataFrame(bundle.get("metrics", []))


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------
# GLOBAL VISUAL STYLE
# ---------------------------------------------------------

st.markdown("""
<style>
:root {
    --bg: #0b1020;
    --panel: #141d32;
    --panel-light: #1d2942;
    --border: #34415f;
    --text: #f4f7ff;
    --muted: #bbc7df;
    --purple: #a78bfa;
    --pink: #f472b6;
    --blue: #60a5fa;
    --green: #4ade80;
}

[data-testid="stAppViewContainer"] {
    background:
      radial-gradient(ellipse at 85% 0%, #27204b 0%, transparent 34%),
      linear-gradient(160deg, #0b1020 0%, #10182a 65%, #111629 100%);
    color: var(--text);
}

[data-testid="stHeader"] {
    background: rgba(11, 16, 32, 0.88);
}

[data-testid="stSidebar"] {
    background: linear-gradient(175deg, #151b31, #101625);
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * {
    color: #e8edff;
}

.block-container {
    max-width: 1550px;
    padding: 2rem 2.6rem 4rem;
}

h1, h2, h3, h4 {
    color: var(--text) !important;
    letter-spacing: -0.35px;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stCaptionContainer"] {
    color: var(--muted);
}

[data-testid="stMetric"] {
    background: linear-gradient(140deg, #1c2942, #151d32);
    padding: 18px 20px;
    border: 1px solid var(--border);
    border-radius: 16px;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.13);
}

[data-testid="stMetricLabel"] {
    color: #c4cee5 !important;
}

[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-weight: 750;
}

[data-testid="stMetricDelta"] {
    color: #8ce5b0 !important;
}

.stButton > button {
    border-radius: 12px;
    min-height: 42px;
    border: 1px solid #6558a3;
    color: #ffffff;
    background: linear-gradient(105deg, #6d46b7, #4d66b5);
    font-weight: 650;
    transition: all 0.18s ease;
}

.stButton > button:hover {
    border-color: #c4b5fd;
    background: linear-gradient(105deg, #8359d5, #527bd7);
    transform: translateY(-1px);
}

.stDownloadButton > button {
    border-radius: 11px;
    border: 1px solid #526786;
    background: #192640;
    color: white;
}

.stTextInput input,
.stTextArea textarea {
    color: #ffffff !important;
    background: #10182a !important;
    border: 1px solid #415174 !important;
    border-radius: 11px !important;
}

[data-baseweb="select"] > div {
    background: #151f34;
    border-color: #415174;
    border-radius: 10px;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(23, 33, 55, 0.68);
    border-color: #34415f;
    border-radius: 16px;
}

div[data-testid="stImage"] img {
    border-radius: 12px;
}

div[data-testid="stPlotlyChart"] {
    background: rgba(18, 27, 46, 0.72);
    border: 1px solid #34415f;
    border-radius: 15px;
    padding: 8px;
}

.hero {
    position: relative;
    overflow: hidden;
    padding: 34px 36px;
    border-radius: 24px;
    margin: 8px 0 25px;
    border: 1px solid #514879;
    background:
      radial-gradient(circle at 92% 20%, rgba(244,114,182,.22), transparent 24%),
      linear-gradient(115deg, #40205f 0%, #24274c 50%, #14233c 100%);
    box-shadow: 0 15px 45px rgba(0,0,0,.20);
    animation: riseIn .5s ease-out;
}

.hero h1 {
    font-size: clamp(32px, 4vw, 48px);
    line-height: 1.12;
    margin: 12px 0 12px;
    color: white !important;
}

.hero p {
    color: #e1e7fb !important;
    font-size: 16px;
    max-width: 760px;
}

.eyebrow {
    text-transform: uppercase;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.8px;
    color: #e5d7ff;
}

.pill {
    display: inline-block;
    padding: 6px 11px;
    margin: 5px 5px 0 0;
    background: rgba(255,255,255,.09);
    border: 1px solid rgba(255,255,255,.19);
    color: #f5edff;
    border-radius: 100px;
    font-size: 12px;
}

.section-note {
    color: #c2cde3;
    font-size: 14px;
    margin-top: -7px;
    margin-bottom: 20px;
}

.result-card {
    padding: 23px;
    border-radius: 17px;
    background: linear-gradient(135deg, #1d2b46, #161e32);
    border: 1px solid #3c4d70;
    margin-top: 14px;
    animation: riseIn .28s ease-out;
}

.result-positive {
    border-left: 5px solid #4ade80;
}

.result-negative {
    border-left: 5px solid #fb7185;
}

.result-title {
    font-size: 27px;
    font-weight: 800;
    margin: 8px 0;
    color: #ffffff;
}

.small-muted {
    font-size: 13px;
    color: #bbc7df;
}

@keyframes riseIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

@media (max-width: 700px) {
    .block-container {
        padding: 1rem 1rem 3rem;
    }
    .hero {
        padding: 23px 20px;
    }
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------

pages = [
    "Home",
    "Movie Explorer",
    "Movie Details",
    "Sentiment Analyzer",
    "Review Library",
    "Sentiment Dashboard",
    "Model Lab",
    "Bulk Analysis",
    "About Project"
]

st.sidebar.markdown(
    """
    <div style="padding:8px 0 18px">
      <div style="font-size:28px;font-weight:850">
        🎬 CineScope AI
      </div>
      <div style="color:#b8c5df;font-size:13px;margin-top:5px">
        Movie Review Intelligence
      </div>
      <div style="margin-top:12px">
        <span class="pill">4 ML models</span>
        <span class="pill">Interactive lab</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

requested_movie = st.query_params.get("movie", "")

# Select the detail page when a poster link supplies a movie title.
if requested_movie:
    st.session_state["nav_page"] = "Movie Details"
elif st.session_state.get("nav_page") == "Movie Details":
    st.session_state["nav_page"] = "Movie Explorer"

page = st.sidebar.radio(
    "EXPLORE",
    pages,
    key="nav_page"
)

st.sidebar.divider()
st.sidebar.markdown("**Project status**")
st.sidebar.success("Models loaded")
st.sidebar.caption("Dataset: English movie reviews")
st.sidebar.caption("Sentiment task: Positive / Negative")
st.sidebar.caption("Academic demonstration project")


# ---------------------------------------------------------
# SHARED COMPONENTS
# ---------------------------------------------------------

def section_header(eyebrow, title, description):
    st.markdown(
        f"""
        <div style="margin:5px 0 24px">
          <div class="eyebrow">{html.escape(eyebrow)}</div>
          <h1 style="margin:8px 0 12px">{html.escape(title)}</h1>
          <div class="section-note">{html.escape(description)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_movie_cards(data, columns_per_row=3):
    if data.empty:
        st.info("No movies match your search.")
        return

    st.caption("🎬 Select a movie poster to open its details and reviews.")

    for start in range(0, len(data), columns_per_row):
        batch = data.iloc[start:start + columns_per_row]
        cols = st.columns(columns_per_row)

        for col, (_, movie) in zip(cols, batch.iterrows()):
            title = str(movie.get("title", "Unknown movie"))
            safe_title = html.escape(title)
            movie_link = quote(title, safe="")

            poster = movie.get("poster_url", "")
            language = html.escape(str(movie.get("language", "")))
            year = html.escape(str(movie.get("year", "")))
            genre = html.escape(str(movie.get("genre", "")))

            if pd.notna(poster) and str(poster).startswith("http"):
                poster_html = f"""
                    <img class="cine-poster"
                    src="{html.escape(str(poster), quote=True)}"
                    alt="{safe_title} movie poster">
                """
            else:
                poster_html = """
                    <div class="cine-placeholder">
                    🎬<br>Poster unavailable
                    </div>
                """

            with col:
                st.html(f"""
                <style>
                .cine-card {{
                    overflow:hidden;
                    border:1px solid #394968;
                    border-radius:15px;
                    background:#151e32;
                    transition:transform .2s,border-color .2s;
                    margin-bottom:8px;
                }}
                .cine-card:hover {{
                    transform:translateY(-4px);
                    border-color:#b29bff;
                }}
                .cine-poster-link {{
                    display:block;
                    text-decoration:none !important;
                    color:white !important;
                }}
                .cine-poster-wrap {{
                    height:285px;
                    position:relative;
                    overflow:hidden;
                    background:#202a42;
                }}
                .cine-poster {{
                    height:100%;
                    width:100%;
                    display:block;
                    object-fit:cover;
                }}
                .cine-placeholder {{
                    height:285px;
                    display:flex;
                    justify-content:center;
                    align-items:center;
                    flex-direction:column;
                    color:white;
                }}
                .cine-poster-title {{
                    position:absolute;
                    bottom:0;
                    left:0;
                    right:0;
                    padding:34px 10px 12px;
                    background:linear-gradient(
                        transparent,rgba(4,8,20,.97)
                    );
                    font-size:15px;
                    font-weight:800;
                    color:white;
                }}
                .cine-info {{
                    padding:12px;
                    color:#f4f7ff;
                    font-family:Arial,sans-serif;
                }}
                .cine-info-title {{
                    font-size:15px;
                    font-weight:750;
                }}
                .cine-info-sub {{
                    color:#c4cee5;
                    font-size:12px;
                    margin-top:5px;
                }}
                .cine-info-genre {{
                    color:#c4b5fd;
                    font-size:11px;
                    margin-top:6px;
                }}
                .cine-view {{
                    color:#d6c8ff;
                    font-size:10px;
                    font-weight:800;
                    letter-spacing:.7px;
                    margin-top:13px;
                }}
                </style>
                <a class="cine-poster-link" href="?movie={movie_link}">
                  <article class="cine-card">
                    <div class="cine-poster-wrap">
                      {poster_html}
                      <div class="cine-poster-title">{safe_title}</div>
                    </div>
                    <div class="cine-info">
                      <div class="cine-info-title">{safe_title}</div>
                      <div class="cine-info-sub">{language} · {year}</div>
                      <div class="cine-info-genre">{genre}</div>
                      <div class="cine-view">VIEW MOVIE & REVIEWS ↗</div>
                    </div>
                  </article>
                </a>
                """)


def show_movie_sentiment_comparison(movie_title):
    st.markdown("## 📊 Sentiment Comparison for This Movie")
    st.caption(
        "Only reviews associated with the selected movie are included."
    )

    sentiment_rows = []

    # Source 1: fictional demonstration reviews and their intended tone.
    demo_reviews = reviews[
        reviews["movie"].astype(str).str.casefold()
        == str(movie_title).casefold()
    ]

    if "example_tone" in demo_reviews.columns:
        for _, row in demo_reviews.iterrows():
            label = str(row.get("example_tone", "")).strip().title()

            if label in ["Positive", "Negative"]:
                sentiment_rows.append({
                    "Source": "Demonstration review tone",
                    "Sentiment": label
                })

    # Source 2: user-submitted reviews classified by the selected model.
    user_file = BASE_DIR / "user_reviews.csv"

    if user_file.exists():
        user_reviews = pd.read_csv(user_file)

        if (
            "movie" in user_reviews.columns
            and "predicted_sentiment" in user_reviews.columns
        ):
            selected_reviews = user_reviews[
                user_reviews["movie"].astype(str).str.casefold()
                == str(movie_title).casefold()
            ]

            for _, row in selected_reviews.iterrows():
                label = str(
                    row.get("predicted_sentiment", "")
                ).strip().title()

                if label in ["Positive", "Negative"]:
                    sentiment_rows.append({
                        "Source": "User reviews (AI prediction)",
                        "Sentiment": label
                    })

    if not sentiment_rows:
        st.info(
            "No labelled reviews are available for this movie yet. "
            "Write a review below to start building its sentiment results."
        )
        return

    sentiment_df = pd.DataFrame(sentiment_rows)

    # Ensure positive and negative categories both appear in the chart.
    source_names = [
        "Demonstration review tone",
        "User reviews (AI prediction)"
    ]

    count_df = (
        sentiment_df.groupby(["Source", "Sentiment"])
        .size()
        .rename("Review count")
        .reset_index()
    )

    full_index = pd.MultiIndex.from_product(
        [source_names, ["Positive", "Negative"]],
        names=["Source", "Sentiment"]
    )

    count_df = (
        count_df.set_index(["Source", "Sentiment"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    # Display counts for the selected movie.
    demo_counts = (
        count_df[count_df["Source"] == "Demonstration review tone"]
        .set_index("Sentiment")["Review count"]
    )

    user_counts = (
        count_df[count_df["Source"] == "User reviews (AI prediction)"]
        .set_index("Sentiment")["Review count"]
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📝 Demonstration reviews")
        st.metric(
            "😊 Positive examples",
            int(demo_counts.get("Positive", 0))
        )
        st.metric(
            "😞 Negative examples",
            int(demo_counts.get("Negative", 0))
        )

    with col2:
        st.markdown("### 🤖 User reviews")
        st.metric(
            "😊 AI-predicted positive",
            int(user_counts.get("Positive", 0))
        )
        st.metric(
            "😞 AI-predicted negative",
            int(user_counts.get("Negative", 0))
        )

    # Create an interactive comparison chart.
    fig = px.bar(
        count_df,
        x="Sentiment",
        y="Review count",
        color="Source",
        barmode="group",
        text="Review count",
        title=f"Positive vs Negative Reviews — {movie_title}",
        color_discrete_map={
            "Demonstration review tone": "#a78bfa",
            "User reviews (AI prediction)": "#4ade80"
        },
        category_orders={
            "Sentiment": ["Positive", "Negative"]
        }
    )

    fig.update_traces(textposition="outside")

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#eef2ff",
        xaxis_title="Sentiment",
        yaxis_title="Number of reviews",
        legend_title="Review source",
        height=440
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="movie_sentiment_chart"
    )

    st.caption(
        "Demonstration review tones are intended example labels. "
        "User-review sentiment is predicted by the selected ML model "
        "and is not independently verified."
    )

def show_movie_detail(title):
    matches = catalog[
        catalog["title"].astype(str).str.casefold()
        == str(title).casefold()
    ]

    if matches.empty:
        st.error("This movie could not be found in the catalogue.")
        if st.button("← Back to Movie Explorer"):
            st.query_params.clear()
            st.rerun()
        return

    movie = matches.iloc[0]
    actual_title = str(movie["title"])
    language = str(movie.get("language", ""))
    movie_year = str(movie.get("year", ""))
    genre = str(movie.get("genre", ""))
    poster = movie.get("poster_url", "")

    # Dedicated movie detail header
    st.markdown("## 🎬 Movie Details")

    back_col, title_col = st.columns([1, 5])
    with back_col:
        if st.button("← Back", key="movie_details_back"):
            st.query_params.clear()
            st.rerun()

    with title_col:
        st.caption("MOVIE INFORMATION · REVIEWS · YOUR OPINION")

    poster_col, info_col = st.columns([1, 2])

    with poster_col:
        if pd.notna(poster) and str(poster).startswith("http"):
            st.image(poster, use_container_width=True)
        else:
            st.markdown("🎬 Poster unavailable")

    with info_col:
        st.markdown(f"# {html.escape(actual_title)}")
        st.markdown(
            f"**{html.escape(language)}** · "
            f"**{html.escape(movie_year)}**"
        )
        st.markdown(
            f'<span class="pill">{html.escape(genre)}</span>',
            unsafe_allow_html=True
        )
        st.write("")
        st.markdown(
            "Read the available reviews or add your own rating and review."
        )
        st.info(
            "AI sentiment is a prediction from the selected model. "
            "It is not a verified audience rating."
        )

    st.divider()
    show_movie_sentiment_comparison(actual_title)
    st.divider()

    # Show the current sample reviews
    st.markdown("## 💬 Existing Reviews")

    sample_reviews = reviews[
        reviews["movie"].astype(str).str.casefold()
        == actual_title.casefold()
    ]

    if sample_reviews.empty:
        st.info("No sample reviews are available for this movie yet.")
    else:
        for _, item in sample_reviews.iterrows():
            tone = str(item.get("example_tone", "Sample"))

            with st.container(border=True):
                st.markdown(f"**{html.escape(tone)} demonstration review**")
                st.write(str(item["review"]))

        st.caption(
            "These are fictional demonstration reviews, not verified "
            "audience reviews."
        )

    # User-submitted reviews are stored separately from training data.
    review_file = BASE_DIR / "user_reviews.csv"
    safe_key = re.sub(r"[^a-zA-Z0-9]", "_", actual_title)

    st.divider()
    st.markdown("## ✍️ Write Your Review")
    st.caption("Rate the film, write your thoughts, and analyze your review.")

    with st.form(key=f"review_form_{safe_key}"):
        reviewer_name = st.text_input(
            "Your name (optional)",
            placeholder="Guest"
        )

        rating = st.select_slider(
            "Your movie rating",
            options=[1, 2, 3, 4, 5],
            value=5,
            format_func=lambda x: "★" * x + "☆" * (5 - x) + f"  ({x}/5)"
        )

        written_review = st.text_area(
            "Your review",
            height=150,
            placeholder="What did you think about the story, acting, music..."
        )

        model_name = st.selectbox(
            "Choose the sentiment model",
            list(models.keys()),
            index=(
                list(models.keys()).index("Linear SVM")
                if "Linear SVM" in models else 0
            )
        )

        submitted = st.form_submit_button(
            "🚀 Submit Review & Analyze Sentiment",
            type="primary",
            use_container_width=True
        )

    if submitted:
        cleaned_review = clean_text(written_review)

        if not cleaned_review:
            st.warning("Please write a review before submitting.")
        else:
            features = vectorizer.transform([cleaned_review])
            prediction = models[model_name].predict(features)[0]
            predicted_sentiment = sentiment_label(prediction)

            new_review = pd.DataFrame([{
                "movie": actual_title,
                "language": language,
                "reviewer": reviewer_name.strip() or "Guest",
                "rating": int(rating),
                "review": written_review.strip(),
                "predicted_sentiment": predicted_sentiment,
                "model_used": model_name,
                "source": "User-submitted"
            }])

            if review_file.exists():
                old_reviews = pd.read_csv(review_file)
                saved_reviews = pd.concat(
                    [old_reviews, new_review],
                    ignore_index=True
                )
            else:
                saved_reviews = new_review

            saved_reviews.to_csv(review_file, index=False)

            st.success("Your review has been saved successfully!")

            if predicted_sentiment == "Positive":
                st.success(
                    f"😊 Predicted sentiment: POSITIVE ({model_name})"
                )
            else:
                st.error(
                    f"😞 Predicted sentiment: NEGATIVE ({model_name})"
                )

    # Display previously submitted reviews for this movie
    st.divider()
    st.markdown("## 🌟 Community Reviews")

    if review_file.exists():
        all_user_reviews = pd.read_csv(review_file)
        movie_reviews = all_user_reviews[
            all_user_reviews["movie"].astype(str).str.casefold()
            == actual_title.casefold()
        ]

        if movie_reviews.empty:
            st.info("No user reviews submitted for this movie yet.")
        else:
            for _, item in movie_reviews.iloc[::-1].iterrows():
                try:
                    stars = max(0, min(5, int(item.get("rating", 0))))
                except (ValueError, TypeError):
                    stars = 0

                with st.container(border=True):
                    st.markdown(
                        f"**{html.escape(str(item.get('reviewer', 'Guest')))}**"
                    )
                    st.markdown(
                        f'<span style="color:#fbbf24">'
                        f'{"★" * stars}{"☆" * (5-stars)}</span>',
                        unsafe_allow_html=True
                    )
                    st.write(str(item.get("review", "")))
                    st.caption(
                        "AI-predicted sentiment: "
                        + str(item.get("predicted_sentiment", "Not analyzed"))
                    )

    st.caption(
        "Submitted reviews are stored in the current application "
        "environment and are not published to a public database."
    )


def sentiment_label(label):
    return "Positive" if str(label).lower() in ("pos", "positive") else "Negative"


def make_chart_theme(fig, height=420):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e7edff"),
        height=height,
        margin=dict(l=25, r=25, t=65, b=35),
        legend_title_text=""
    )
    return fig


if catalog.empty or reviews.empty:
    st.error(
        "The movie catalogue or review library is missing. "
        "Check the CSV files in the project folder."
    )
    st.stop()


# ---------------------------------------------------------
# HOME
# ---------------------------------------------------------

if page == "Movie Details":
    show_movie_detail(requested_movie)

elif page == "Home":
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">YOUR CINEMA, WITH AN INTELLIGENT TWIST</div>
          <h1>Every review tells a story.<br>Let AI read between the lines.</h1>
          <p>
            Discover South Indian cinema, explore movie reviews, and compare
            four machine learning algorithms in one interactive experience.
          </p>
          <span class="pill">🎞 Movie discovery</span>
          <span class="pill">🧠 AI sentiment analysis</span>
          <span class="pill">📊 Model performance</span>
          <span class="pill">📂 Bulk review analysis</span>
        </div>
        """,
        unsafe_allow_html=True
    )


    # Quick movie search on the homepage
    st.markdown("## 🔎 Find your movie")

    home_query = st.text_input(
        "Search by movie name",
        placeholder="Try RRR, Pushpa, Jailer 2, Premalu...",
        key="home_movie_name_search"
    )

    if home_query.strip():
        matching_movies = catalog[
            catalog["title"].astype(str).str.contains(
                re.escape(home_query.strip()),
                case=False,
                na=False
            )
        ]

        if matching_movies.empty:
            st.info(
                "No matching movie found. Try another title."
            )
        else:
            st.success(
                f"Found {len(matching_movies)} matching movie(s)."
            )
            show_movie_cards(
                matching_movies,
                columns_per_row=4
            )

    st.write("")

    best_name = bundle.get("best_model", "Linear SVM")
    best_accuracy = None

    if not metrics.empty and "Accuracy" in metrics.columns:
        best_row = metrics.sort_values("Accuracy", ascending=False).iloc[0]
        best_name = str(best_row.get("Model", best_name))
        best_accuracy = float(best_row["Accuracy"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎥 Movie catalogue", len(catalog))
    c2.metric("📝 Training reviews", f"{len(training_data):,}")
    c3.metric("🧠 ML algorithms", len(models))
    c4.metric(
        "🏆 Best test accuracy",
        f"{best_accuracy:.2%}" if best_accuracy is not None else "N/A"
    )

    st.write("")
    st.markdown("## 🍿 Spotlight: South Indian cinema")
    st.markdown(
        '<div class="section-note">Choose a film to explore the catalogue.</div>',
        unsafe_allow_html=True
    )
    show_movie_cards(catalog.sort_values("year", ascending=False).head(10), columns_per_row=5)

    st.write("")
    st.markdown("## ✨ What would you like to do?")

    a, b, c = st.columns(3)

    with a:
        with st.container(border=True):
            st.markdown("### 🧠 Sentiment Studio")
            st.write("Try a review and see how the algorithms interpret it.")
            if st.button("Analyze a review →", key="home_analyzer", use_container_width=True):
                st.session_state["nav_page"] = "Sentiment Analyzer"
                st.rerun()

    with b:
        with st.container(border=True):
            st.markdown("### 📊 Model Lab")
            st.write("Compare accuracy, precision, recall, and F1-score.")
            if st.button("Explore model results →", key="home_models", use_container_width=True):
                st.session_state["nav_page"] = "Model Lab"
                st.rerun()

    with c:
        with st.container(border=True):
            st.markdown("### 📂 Bulk Analysis")
            st.write("Upload a review CSV and export predictions.")
            if st.button("Analyze a CSV →", key="home_bulk", use_container_width=True):
                st.session_state["nav_page"] = "Bulk Analysis"
                st.rerun()

    st.info(
        "The catalogue contains South Indian films, but the current models "
        "were trained on English reviews. Regional-language understanding "
        "requires suitable Tamil, Telugu, Malayalam, or Kannada training data."
    )


# ---------------------------------------------------------
# MOVIE EXPLORER
# ---------------------------------------------------------

elif page == "Movie Explorer":
    section_header(
        "DISCOVER",
        "Movie Explorer",
        "Browse films, filter by language, and find your next movie."
    )

    left, right = st.columns([2, 1])
    with left:
        query = st.text_input(
            "🔎 Search titles",
            placeholder="Search RRR, Vikram, Premalu..."
        )
    with right:
        languages = sorted(catalog["language"].dropna().unique().tolist())
        selected_languages = st.multiselect(
            "Film industry / language",
            languages,
            default=languages
        )

    genres = sorted(catalog["genre"].dropna().astype(str).unique().tolist())
    chosen_genres = st.multiselect(
        "Filter by genre",
        genres,
        default=genres
    )

    years = pd.to_numeric(catalog["year"], errors="coerce").dropna()

    filtered = catalog.copy()
    filtered = filtered[filtered["language"].isin(selected_languages)]
    filtered = filtered[filtered["genre"].astype(str).isin(chosen_genres)]

    if query.strip():
        filtered = filtered[
            filtered["title"].astype(str).str.contains(
                query.strip(), case=False, na=False
            )
        ]

    if not years.empty:
        year_range = st.slider(
            "Release year range",
            int(years.min()),
            int(years.max()),
            (int(years.min()), int(years.max()))
        )
        filtered_years = pd.to_numeric(filtered["year"], errors="coerce")
        filtered = filtered[
            filtered_years.between(year_range[0], year_range[1])
        ]

    st.caption(f"Showing {len(filtered)} of {len(catalog)} films")
    show_movie_cards(filtered, columns_per_row=3)

    if not filtered.empty:
        st.divider()
        selected_title = st.selectbox(
            "Explore sample reviews for a movie",
            filtered["title"].astype(str).tolist()
        )

        selected_reviews = reviews[reviews["movie"] == selected_title]

        if selected_reviews.empty:
            st.info("No demonstration reviews have been added for this film yet.")
        else:
            for _, row in selected_reviews.iterrows():
                with st.container(border=True):
                    st.markdown(
                        f"**Demonstration tone: {html.escape(str(row.get('example_tone', 'Unlabelled')))}**"
                    )
                    st.write(str(row["review"]))

            st.caption(
                "These are fictional demonstration reviews, not verified "
                "audience reviews."
            )


# ---------------------------------------------------------
# SENTIMENT ANALYZER
# ---------------------------------------------------------

elif page == "Sentiment Analyzer":
    section_header(
        "THE SENTIMENT STUDIO",
        "What does the review feel like?",
        "Enter text, choose an algorithm, and compare the model predictions."
    )

    POSITIVE_EXAMPLE = (
        "The acting was fantastic, the music was excellent, "
        "and I enjoyed every minute of the movie."
    )
    NEGATIVE_EXAMPLE = (
        "The story was boring, the screenplay was weak, "
        "and the movie was disappointing."
    )

    def load_positive_review():
        st.session_state["review_text_input"] = POSITIVE_EXAMPLE

    def load_negative_review():
        st.session_state["review_text_input"] = NEGATIVE_EXAMPLE

    with st.container(border=True):
        st.markdown("### ✍️ Compose your review")

        col1, col2 = st.columns(2)

        with col1:
            selected_movie = st.selectbox(
                "Movie context",
                ["General review"] + catalog["title"].astype(str).tolist()
            )

        with col2:
            model_name = st.selectbox(
                "Choose your ML algorithm",
                list(models.keys()),
                index=(
                    list(models.keys()).index("Linear SVM")
                    if "Linear SVM" in models else 0
                )
            )

        sample1, sample2, sample3 = st.columns([1, 1, 2])

        with sample1:
            st.button(
                "😊 Try positive example",
                on_click=load_positive_review,
                use_container_width=True
            )

        with sample2:
            st.button(
                "😞 Try negative example",
                on_click=load_negative_review,
                use_container_width=True
            )

        with sample3:
            st.caption(
                "Examples are editable. Write your own English review below."
            )

        review_text = st.text_area(
            "Your movie review",
            height=170,
            placeholder="Describe the acting, story, music, screenplay...",
            key="review_text_input"
        )

        words = len(review_text.split()) if review_text.strip() else 0
        st.caption(f"📝 {words} words · {len(review_text)} characters")

        analyze_clicked = st.button(
            "✨ Analyze sentiment",
            type="primary",
            use_container_width=True
        )

    if analyze_clicked:
        cleaned = clean_text(review_text)

        if not cleaned:
            st.warning("Enter a review containing English words first.")
        else:
            features = vectorizer.transform([cleaned])

            all_predictions = {
                name: model.predict(features)[0]
                for name, model in models.items()
            }

            selected_prediction = all_predictions[model_name]
            sentiment = sentiment_label(selected_prediction)
            agree_count = sum(
                sentiment_label(pred) == sentiment
                for pred in all_predictions.values()
            )

            positive = sentiment == "Positive"
            result_class = "result-positive" if positive else "result-negative"
            color = "#4ade80" if positive else "#fb7185"
            emoji = "😊" if positive else "😞"

            st.markdown(
                f"""
                <div class="result-card {result_class}">
                  <div class="eyebrow">PREDICTION RESULT</div>
                  <div class="result-title" style="color:{color}">
                    {emoji} {sentiment}
                  </div>
                  <p style="color:#e1e7fb">
                    Selected algorithm: <b>{html.escape(model_name)}</b>
                  </p>
                  <p style="color:#c4d0e8">
                    Movie context: {html.escape(selected_movie)}
                  </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            m1, m2, m3 = st.columns(3)
            m1.metric("Selected model", model_name)
            m2.metric("Word count", words)
            m3.metric("Model agreement", f"{agree_count} / {len(models)}")

            st.markdown("### 🔬 How did the other models respond?")

            comparison_cols = st.columns(len(all_predictions))

            for col, (name, prediction) in zip(
                comparison_cols, all_predictions.items()
            ):
                with col:
                    label = sentiment_label(prediction)
                    icon = "✅" if label == sentiment else "↔️"
                    with st.container(border=True):
                        st.markdown(f"**{name}**")
                        st.markdown(
                            f"### {icon} {label}"
                        )

            st.caption(
                "Model agreement is the number of algorithms that predicted "
                "the selected model's class. It is not a probability or "
                "a guarantee that the prediction is correct."
            )


# ---------------------------------------------------------
# REVIEW LIBRARY
# ---------------------------------------------------------

elif page == "Review Library":
    section_header(
        "COMMUNITY WALL",
        "Review Library",
        "Browse, search, and filter the demonstration movie reviews."
    )

    st.warning(
        "The current review library contains fictional sample reviews "
        "for interface demonstration; these are not verified audience opinions."
    )

    c1, c2 = st.columns([2, 1])

    movie_options = ["All movies"] + sorted(
        reviews["movie"].dropna().astype(str).unique().tolist()
    )
    tone_options = ["All tones"] + sorted(
        reviews["example_tone"].dropna().astype(str).unique().tolist()
    )

    with c1:
        selected_movie = st.selectbox("Filter by movie", movie_options)

    with c2:
        selected_tone = st.selectbox("Example tone", tone_options)

    search = st.text_input("🔎 Search within review text")

    filtered = reviews.copy()

    if selected_movie != "All movies":
        filtered = filtered[filtered["movie"] == selected_movie]

    if selected_tone != "All tones":
        filtered = filtered[filtered["example_tone"] == selected_tone]

    if search.strip():
        filtered = filtered[
            filtered["review"].astype(str).str.contains(
                search.strip(), case=False, na=False
            )
        ]

    st.caption(f"{len(filtered)} matching review(s)")

    if filtered.empty:
        st.info("No reviews match your search. Try another filter.")
    else:
        for _, row in filtered.iterrows():
            tone = str(row.get("example_tone", "Unlabelled"))
            accent = "#4ade80" if tone == "Positive" else "#fb7185"

            with st.container(border=True):
                left, right = st.columns([5, 1])

                with left:
                    st.markdown(f"### 🎬 {row['movie']}")
                    st.write(str(row["review"]))
                    st.caption(str(row.get("language", "")))

                with right:
                    st.markdown(
                        f'<span class="pill" style="border-color:{accent}">'
                        f'{html.escape(tone)}</span>',
                        unsafe_allow_html=True
                    )


# ---------------------------------------------------------
# MODEL LAB
# ---------------------------------------------------------

elif page == "Model Lab":
    section_header(
        "THE ML ARENA",
        "Model Performance Lab",
        "Compare all four algorithms using your held-out test set."
    )

    if metrics.empty or "Accuracy" not in metrics.columns:
        st.warning("Model metrics could not be loaded.")
    else:
        metrics = metrics.sort_values(
            "Accuracy", ascending=False
        ).reset_index(drop=True)

        if "Model" not in metrics.columns:
            st.error("The metrics file does not contain model names.")
            st.stop()

        best = metrics.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("🏆 Best model", str(best["Model"]))
        c2.metric("Test accuracy", f"{float(best['Accuracy']):.2%}")
        c3.metric("Models compared", len(metrics))

        display_columns = [
            c for c in [
                "Accuracy", "Precision", "Recall", "F1-score"
            ] if c in metrics.columns
        ]

        metric_choice = st.selectbox(
            "Choose the metric to rank",
            display_columns,
            index=display_columns.index("Accuracy")
        )

        chart_data = metrics.sort_values(
            metric_choice, ascending=True
        ).copy()

        fig = px.bar(
            chart_data,
            x=metric_choice,
            y="Model",
            orientation="h",
            color=metric_choice,
            text=chart_data[metric_choice].map(lambda x: f"{x:.2%}"),
            color_continuous_scale=["#5b6fc5", "#a78bfa", "#f472b6"],
            title=f"{metric_choice} by algorithm"
        )
        fig.update_xaxes(tickformat=".0%", range=[0, 1])
        fig.update_traces(textposition="outside")
        make_chart_theme(fig, 400)
        st.plotly_chart(fig, use_container_width=True)

        score_columns = [
            c for c in ["Precision", "Recall", "F1-score"]
            if c in metrics.columns
        ]

        if score_columns:
            melted = metrics.melt(
                id_vars="Model",
                value_vars=score_columns,
                var_name="Metric",
                value_name="Score"
            )

            fig2 = px.bar(
                melted,
                x="Model",
                y="Score",
                color="Metric",
                barmode="group",
                title="Precision, Recall and F1-score",
                color_discrete_sequence=[
                    "#60a5fa", "#4ade80", "#f472b6"
                ]
            )
            fig2.update_yaxes(tickformat=".0%", range=[0, 1])
            make_chart_theme(fig2, 450)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### 📋 Detailed comparison")

        formatted = metrics.copy()
        for column in display_columns:
            formatted[column] = formatted[column].map(
                lambda value: f"{value:.2%}"
            )

        st.dataframe(
            formatted,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Download model comparison",
            data=metrics.to_csv(index=False).encode("utf-8"),
            file_name="cinescope_model_comparison.csv",
            mime="text/csv"
        )

        # Rebuild the same held-out split to show a confusion matrix.
        if (
            "cleaned_review" in training_data.columns
            and "sentiment" in training_data.columns
        ):
            st.markdown("### 🎯 Confusion matrix")

            matrix_model_name = st.selectbox(
                "Choose an algorithm for the confusion matrix",
                list(models.keys())
            )

            X_test_text, _, y_test, _ = train_test_split(
                training_data["cleaned_review"].fillna("").astype(str),
                training_data["sentiment"],
                test_size=0.20,
                random_state=42,
                stratify=training_data["sentiment"]
            )

            # Same fixed test partition as the original training notebook.
            X_test_matrix = vectorizer.transform(X_test_text)
            matrix_predictions = models[matrix_model_name].predict(
                X_test_matrix
            )

            cm = confusion_matrix(
                y_test, matrix_predictions, labels=["neg", "pos"]
            )

            cm_fig = px.imshow(
                cm,
                x=["Predicted negative", "Predicted positive"],
                y=["Actual negative", "Actual positive"],
                text_auto=True,
                aspect="auto",
                color_continuous_scale=["#18243a", "#60a5fa"],
                title=f"{matrix_model_name}: confusion matrix"
            )
            make_chart_theme(cm_fig, 380)
            st.plotly_chart(cm_fig, use_container_width=True)

        st.caption(
            "Evaluation scores describe the current test set; performance "
            "on new or regional-language reviews may be different."
        )


# ---------------------------------------------------------
# BULK ANALYSIS
# ---------------------------------------------------------

elif page == "Bulk Analysis":
    section_header(
        "BATCH PROCESSING",
        "Bulk Review Analyzer",
        "Analyze many reviews at once and export predictions to CSV."
    )

    st.markdown(
        """
        <div class="result-card">
          <h3>📂 Upload your review dataset</h3>
          <p>Use a CSV file with a column containing English-language reviews.</p>
          <p class="small-muted">Tip: a column named <b>review</b> is automatically detected.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"]
    )

    model_name = st.selectbox(
        "Algorithm for batch predictions",
        list(models.keys())
    )

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)

            if input_df.empty or len(input_df.columns) == 0:
                st.error("The uploaded CSV is empty.")
            else:
                st.markdown("### Preview")
                st.dataframe(
                    input_df.head(8),
                    use_container_width=True,
                    hide_index=True
                )

                columns = input_df.columns.tolist()
                default_index = next(
                    (
                        i for i, col in enumerate(columns)
                        if col.lower() == "review"
                    ),
                    0
                )

                review_column = st.selectbox(
                    "Select the review text column",
                    columns,
                    index=default_index
                )

                if st.button(
                    "🚀 Analyze all reviews",
                    type="primary",
                    use_container_width=True
                ):
                    original = input_df[review_column].fillna("").astype(str)
                    cleaned = [clean_text(text) for text in original]

                    valid_positions = [
                        i for i, value in enumerate(cleaned) if value
                    ]

                    predictions = ["Not analyzed"] * len(input_df)

                    if valid_positions:
                        matrix = vectorizer.transform(
                            [cleaned[i] for i in valid_positions]
                        )
                        labels = models[model_name].predict(matrix)

                        for i, label in zip(valid_positions, labels):
                            predictions[i] = sentiment_label(label)

                    output_df = input_df.copy()
                    output_df["predicted_sentiment"] = predictions

                    positive_count = predictions.count("Positive")
                    negative_count = predictions.count("Negative")

                    a, b, c = st.columns(3)
                    a.metric("Reviews processed", len(valid_positions))
                    b.metric("Predicted positive", positive_count)
                    c.metric("Predicted negative", negative_count)

                    st.markdown("### Results")
                    st.dataframe(
                        output_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    st.download_button(
                        "⬇️ Download predictions CSV",
                        data=output_df.to_csv(index=False).encode("utf-8"),
                        file_name="cinescope_predictions.csv",
                        mime="text/csv"
                    )

        except Exception as exc:
            st.error(f"Unable to read or analyze this CSV: {exc}")


# ---------------------------------------------------------
# ABOUT
# ---------------------------------------------------------


elif page == "Sentiment Dashboard":
    section_header(
        "AUDIENCE SENTIMENT",
        "Positive vs Negative Reviews",
        "Compare sentiment counts, explore movie-level patterns, "
        "and download your analysis."
    )

    st.markdown("""
    <div class="hero">
      <div class="eyebrow">SENTIMENT INTELLIGENCE</div>
      <h1>What are reviewers saying?</h1>
      <p>
        Explore how many reviews are positive or negative.
        Switch between labelled training data, demonstration reviews,
        and reviews submitted through CineScope AI.
      </p>
    </div>
    """, unsafe_allow_html=True)

    source = st.selectbox(
        "Choose which reviews to analyze",
        [
            "Training dataset (known labels)",
            "User-submitted reviews (AI predictions)",
            "Demonstration reviews (fictional examples)"
        ],
        key="sentiment_dashboard_source"
    )

    # Select one source, keeping different types of labels separate.
    view = pd.DataFrame(columns=["Movie", "Sentiment", "Review"])
    source_description = ""

    if source == "Training dataset (known labels)":
        if (
            not training_data.empty
            and "sentiment" in training_data.columns
        ):
            labels = (
                training_data["sentiment"]
                .astype(str)
                .str.lower()
                .map({
                    "pos": "Positive",
                    "positive": "Positive",
                    "neg": "Negative",
                    "negative": "Negative"
                })
            )

            view = pd.DataFrame({
                "Sentiment": labels,
                "Review": (
                    training_data["review"].astype(str)
                    if "review" in training_data.columns
                    else ""
                )
            }).dropna(subset=["Sentiment"])

            source_description = (
                "These counts use the dataset's supplied sentiment labels. "
                "The dataset does not identify each review's South Indian "
                "movie title, so this source cannot produce movie-by-movie "
                "comparisons."
            )

    elif source == "User-submitted reviews (AI predictions)":
        user_file = BASE_DIR / "user_reviews.csv"

        if user_file.exists():
            user_df = pd.read_csv(user_file)

            if "predicted_sentiment" in user_df.columns:
                view = pd.DataFrame({
                    "Movie": (
                        user_df["movie"].astype(str)
                        if "movie" in user_df.columns
                        else "Unknown"
                    ),
                    "Sentiment": (
                        user_df["predicted_sentiment"]
                        .astype(str).str.strip().str.title()
                    ),
                    "Review": (
                        user_df["review"].astype(str)
                        if "review" in user_df.columns
                        else ""
                    )
                })

                source_description = (
                    "These labels are AI predictions, not independently "
                    "verified sentiment labels."
                )
            else:
                st.warning(
                    "The user review file does not contain a "
                    "predicted_sentiment column."
                )
        else:
            st.info(
                "No user-submitted reviews have been saved yet. "
                "Click a movie poster and submit a review first."
            )

    else:
        if (
            not reviews.empty
            and "example_tone" in reviews.columns
        ):
            view = pd.DataFrame({
                "Movie": reviews["movie"].astype(str),
                "Sentiment": (
                    reviews["example_tone"]
                    .astype(str).str.strip().str.title()
                ),
                "Review": reviews["review"].astype(str)
            })

            source_description = (
                "These are fictional demonstration reviews labelled "
                "by their intended example tone. They are not verified "
                "audience reviews or model predictions."
            )

    # Keep only the binary sentiment categories.
    if not view.empty:
        view["Sentiment"] = (
            view["Sentiment"].astype(str).str.strip().str.title()
        )

        view = view[
            view["Sentiment"].isin(["Positive", "Negative"])
        ].copy()

    if not view.empty:
        # Apply a movie filter where movie names are available.
        if "Movie" in view.columns:
            movie_options = ["All movies"] + sorted(
                view["Movie"].dropna().astype(str).unique().tolist()
            )

            selected_movie = st.selectbox(
                "Filter by movie",
                movie_options,
                key="sentiment_dashboard_movie"
            )

            if selected_movie != "All movies":
                view = view[view["Movie"] == selected_movie]

        total = len(view)

        counts = (
            view["Sentiment"]
            .value_counts()
            .reindex(["Positive", "Negative"], fill_value=0)
        )

        positive_count = int(counts["Positive"])
        negative_count = int(counts["Negative"])

        positive_pct = positive_count / total if total else 0
        negative_pct = negative_count / total if total else 0

        st.markdown("### 📌 Sentiment summary")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "😊 Positive reviews",
            f"{positive_count:,}",
            f"{positive_pct:.1%} of selected reviews"
        )

        c2.metric(
            "😞 Negative reviews",
            f"{negative_count:,}",
            f"{negative_pct:.1%} of selected reviews"
        )

        c3.metric(
            "📝 Total reviews",
            f"{total:,}"
        )

        st.markdown("### 📊 Overall comparison")

        counts_df = counts.rename_axis("Sentiment").reset_index(
            name="Number of reviews"
        )

        left, right = st.columns(2)

        with left:
            fig_bar = px.bar(
                counts_df,
                x="Sentiment",
                y="Number of reviews",
                color="Sentiment",
                text="Number of reviews",
                title="Positive vs Negative Count",
                color_discrete_map={
                    "Positive": "#4ade80",
                    "Negative": "#fb7185"
                }
            )
            fig_bar.update_traces(textposition="outside")
            fig_bar.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#eef2ff",
                showlegend=False,
                height=420
            )
            st.plotly_chart(
                fig_bar, use_container_width=True
            )

        with right:
            pie_df = counts_df[counts_df["Number of reviews"] > 0]

            if not pie_df.empty:
                fig_pie = px.pie(
                    pie_df,
                    names="Sentiment",
                    values="Number of reviews",
                    hole=0.58,
                    title="Sentiment Share",
                    color="Sentiment",
                    color_discrete_map={
                        "Positive": "#4ade80",
                        "Negative": "#fb7185"
                    }
                )
                fig_pie.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#eef2ff",
                    height=420
                )
                st.plotly_chart(
                    fig_pie, use_container_width=True
                )

        # Movie-level comparison for sources that have a movie column.
        if "Movie" in view.columns:
            st.markdown("### 🎬 Compare sentiment by movie")

            if view["Movie"].nunique() > 1:
                max_movies = min(20, view["Movie"].nunique())

                number_to_show = st.slider(
                    "Number of movies to display",
                    min_value=1,
                    max_value=max_movies,
                    value=min(10, max_movies),
                    key="sentiment_dashboard_movie_count"
                )
            else:
                number_to_show = 1

            comparison = pd.crosstab(
                view["Movie"], view["Sentiment"]
            ).reindex(
                columns=["Positive", "Negative"],
                fill_value=0
            )

            comparison["Total"] = comparison.sum(axis=1)

            comparison = comparison.sort_values(
                "Total", ascending=False
            ).head(number_to_show)

            comparison_long = (
                comparison
                .drop(columns="Total")
                .reset_index()
                .melt(
                    id_vars="Movie",
                    var_name="Sentiment",
                    value_name="Number of reviews"
                )
            )

            movie_fig = px.bar(
                comparison_long,
                x="Movie",
                y="Number of reviews",
                color="Sentiment",
                barmode="group",
                title="Positive and Negative Reviews by Movie",
                color_discrete_map={
                    "Positive": "#4ade80",
                    "Negative": "#fb7185"
                }
            )

            movie_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#eef2ff",
                height=480,
                xaxis_tickangle=-35
            )

            st.plotly_chart(
                movie_fig, use_container_width=True
            )

            st.markdown("### 📋 Movie sentiment counts")
            st.dataframe(
                comparison.reset_index(),
                use_container_width=True,
                hide_index=True
            )

        st.markdown("### ⬇️ Export analysis")

        export_csv = view.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download selected review data as CSV",
            data=export_csv,
            file_name="cinescope_sentiment_analysis.csv",
            mime="text/csv",
            key="sentiment_dashboard_download"
        )

        st.caption(source_description)

    else:
        st.info(
            "There are no positive or negative reviews to compare "
            "for this selection yet."
        )

    st.warning(
        "Interpret the results according to the selected source: "
        "training labels, fictional demonstration labels, and AI-predicted "
        "labels are different and should not be combined as if they were "
        "verified audience opinions."
    )


elif page == "About Project":
    section_header(
        "PROJECT NOTEBOOK",
        "About CineScope AI",
        "The objective, methodology, algorithms, and current limitations."
    )

    left, right = st.columns([3, 2])

    with left:
        with st.container(border=True):
            st.markdown("### 🎯 Problem statement")
            st.write(
                "Reading large numbers of movie reviews manually can be "
                "time-consuming. CineScope AI demonstrates how supervised "
                "machine learning can classify English movie reviews as "
                "positive or negative."
            )

            st.markdown("### 🧪 Methodology")
            st.markdown("""
            1. Load and inspect labelled movie reviews.
            2. Clean the text.
            3. Convert text into numerical features with TF-IDF.
            4. Split the dataset into training and test subsets.
            5. Train four supervised classifiers.
            6. Evaluate the models using accuracy, precision, recall, and F1-score.
            7. Integrate the trained models into an interactive application.
            """)

    with right:
        with st.container(border=True):
            st.markdown("### 🧠 Algorithms")
            for name in models:
                st.markdown(f"✅ {name}")

            st.markdown("### 🧰 Technologies")
            for tool in [
                "Python", "Pandas", "Scikit-learn",
                "TF-IDF", "Streamlit", "Plotly"
            ]:
                st.markdown(f"• {tool}")


    st.markdown("### Credits")
    st.markdown(
        '<a href="https://www.themoviedb.org/" target="_blank">'
        '<img src="https://www.themoviedb.org/assets/2/v4/logos/'
        'v2/blue_long_2-9665a76b1ae401a510ec1e0ca40ddcb3b0cfe45f1d51'
        'b77a308fea0845885648.svg" width="145" alt="TMDB logo"></a>',
        unsafe_allow_html=True
    )
    st.write("Movie posters and metadata provided by TMDB.")
    st.write(
        "This product uses the TMDB API but is not endorsed or certified by TMDB."
    )

    st.markdown("### ⚠️ Limitations and responsible interpretation")

    st.warning(
        "The current model predicts positive or negative sentiment only. "
        "It was trained on English reviews and is not validated for Tamil, "
        "Telugu, Malayalam, or Kannada. The movie catalogue's demonstration "
        "reviews are fictional. Predictions may be incorrect, particularly "
        "for sarcasm, mixed opinions, or unfamiliar vocabulary."
    )

    st.caption(
        "For a research or academic submission, report the actual evaluation "
        "results and describe the dataset and validation procedure accurately."
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()
st.markdown(
    """
    <div style="text-align:center;padding:8px 0 15px">
      <div style="font-weight:750;color:#e3e9ff">🎬 CineScope AI</div>
      <div class="small-muted">
        Movie discovery · Machine learning · Interactive analytics
      </div>
    </div>
    """,
    unsafe_allow_html=True
)
