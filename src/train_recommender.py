import argparse
import os
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from src.utils import setup_logging, read_csv_with_date, ensure_columns

logger = setup_logging("train_recommender")

REQ_FLIGHT = ["userCode", "to", "date"]
REQ_HOTEL = ["userCode", "name", "place", "price", "days", "total", "date"]

def build_recommender(flights: pd.DataFrame, hotels: pd.DataFrame):
    hotel_pop = hotels.groupby(["name", "place"]).size().reset_index(name="pop")
    hotel_stats = hotels.groupby(["name", "place"]).agg(
        avg_price=("price", "mean"),
        avg_days=("days", "mean")
    ).reset_index()
    hotels_feat = hotel_pop.merge(hotel_stats, on=["name", "place"], how="left")

    user_place_hist = hotels.groupby(["userCode", "place"]).size().unstack(fill_value=0)
    user_to_hist = flights.groupby(["userCode", "to"]).size().unstack(fill_value=0)

    common_places = set(user_place_hist.columns).union(set(user_to_hist.columns))
    user_pref = pd.DataFrame(index=sorted(set(user_place_hist.index).union(user_to_hist.index)))
    for p in common_places:
        user_pref[p] = user_place_hist.get(p, 0) + user_to_hist.get(p, 0) * 0.5

    user_pref = user_pref.div(user_pref.sum(axis=1).replace(0, 1), axis=0)

    hotel_vectors = []
    hotel_index = []
    places = sorted(common_places)
    for _, row in hotels_feat.iterrows():
        vec = [1 if row["place"] == p else 0 for p in places]
        vec.append(row["avg_price"])
        hotel_vectors.append(vec)
        hotel_index.append((row["name"], row["place"]))
    hotel_matrix = pd.DataFrame(hotel_vectors, columns=places + ["avg_price"], index=pd.MultiIndex.from_tuples(hotel_index))
    if not hotel_matrix["avg_price"].std() == 0:
        hotel_matrix["avg_price"] = (hotel_matrix["avg_price"] - hotel_matrix["avg_price"].mean()) / (hotel_matrix["avg_price"].std())

    artifact = {
        "hotels_feat": hotels_feat,
        "user_pref": user_pref,
        "places": places,
        "hotel_matrix": hotel_matrix
    }
    return artifact

def recommend_for_user(user_code: int, artifact, k: int = 5):
    user_pref = artifact["user_pref"]
    hotel_matrix = artifact["hotel_matrix"]
    hotels_feat = artifact["hotels_feat"]

    if user_code not in user_pref.index:
        top = hotels_feat.sort_values("pop", ascending=False).head(k)
        return top[["name", "place", "avg_price", "avg_days", "pop"]].reset_index(drop=True)

    places = artifact["places"]
    uvec = [user_pref.loc[user_code].get(p, 0) for p in places]
    uvec.append(0)
    sims = cosine_similarity([uvec], hotel_matrix.values)[0]
    hotels_feat = hotels_feat.copy()
    hotels_feat["score"] = sims
    recs = hotels_feat.sort_values(["score", "pop"], ascending=[False, False]).head(k)
    return recs[["name", "place", "avg_price", "avg_days", "pop", "score"]].reset_index(drop=True)

def main(args):
    flights = read_csv_with_date(args.flight_path, date_cols=["date"])
    hotels = read_csv_with_date(args.hotel_path, date_cols=["date"])
    ensure_columns(flights, REQ_FLIGHT)
    ensure_columns(hotels, REQ_HOTEL)

    artifact = build_recommender(flights, hotels)
    os.makedirs("models", exist_ok=True)
    joblib.dump(artifact, args.output_path)
    logger.info(f"Recommender artifact saved to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flight_path", required=True)
    parser.add_argument("--hotel_path", required=True)
    parser.add_argument("--user_path", required=False)
    parser.add_argument("--output_path", default="models/recommender.pkl")
    args = parser.parse_args()
    main(args)