import streamlit as st
from transformers import pipeline
import pickle
import os
import pandas as pd
import seaborn as sns
import ast
import string
import re

st.set_page_config(
	page_title="Offer Recommender",
	layout="wide"
)

pipe = pipeline(task="zero-shot-classification", model="facebook/bart-large-mnli")
dire = "DS_NLP_search_data"

@st.cache_data
def get_processed_offers():
	processed_offers = pd.read_csv(os.path.join(dire, "processed_offers.csv"))
	processed_offers["CATEGORY"] = processed_offers["CATEGORY"].map(ast.literal_eval)
	return processed_offers

@st.cache_data
def get_categories_data():
	cats = pd.read_csv(os.path.join(dire, "categories.csv"))
	return cats

@st.cache_data
def get_offers_data():
	offers = pd.read_csv(os.path.join(dire, "offer_retailer.csv"))
	return offers

@st.cache_data
def get_categories(cats_):
	categories = list(cats_["IS_CHILD_CATEGORY_TO"].unique())
	for x in ["Mature"]:
		if x in categories:
			categories.remove(x)
	return categories

def get_confidence_charts(results):
	df = (
		pd.DataFrame({"Category": results["labels"], "probability": results["scores"]})
		.sort_values(by="probability", ascending=False)
		.reset_index(drop=True)
	)

	df.index += 1

	# Add styling
	cmGreen = sns.light_palette("blue", as_cmap=True)
	cmRed = sns.light_palette("red", as_cmap=True)
	df = df.style.background_gradient(
		cmap=cmGreen,
		subset=[
			"probability",
		],
	)


	format_dictionary = {
		"Score": "{:.1%}",
	}

	df = df.format(format_dictionary)

	return df


def check_in_offer(search_str, offer_rets):
	offers = []
	# print(offer_rets)
	for i in range(len(offer_rets)):
		offer_str = offer_rets.iloc[i]["OFFER"]
		# print(offer_str)
		parsed_str = offer_str.lower().translate(str.maketrans('', '', string.punctuation))
		parsed_str = re.sub('[^a-zA-Z0-9 \n\.]', '', parsed_str)
		# print(parsed_str)
		if search_str.lower() in parsed_str.split(" "):
		  offers.append(offer_str)
	df = pd.DataFrame({"OFFER":offers})
	# print(df)
	return df

def is_retailer(search_str, threshold=0.5):
	processed_search_str = search_str.lower().capitalize()
	labels = pipe(processed_search_str,
	  candidate_labels=["brand", "retailer", "item"],
	)

	return labels["labels"][0] == "retailer" and labels["scores"][0] > threshold

@st.cache
def get_prod_categories():
	retail_mapping = {}
	for retailer in list(offer_rets["RETAILER"].unique()):
		query_direct_retail = complete_df[complete_df["RETAILER"] == retailer]
		prod_cats = query_direct_retail["PRODUCT_CATEGORY"].unique()
		retail_mapping[retailer] = prod_cats
	return retail_mapping


def get_most_overlap(retailer, offer_rets, retail_mapping, top_n=3):
	overlaps = {}

	for key, value in retail_mapping.items():
		if key != retailer.upper():
			overlap = set(value).intersection(set(retail_mapping[retailer.upper()]))
			overlaps[key] = len(overlap)

	sorted_overlaps = dict(sorted(overlaps.items(), key=lambda x:x[1], reverse=True))

	related_retailers =  list({k:sorted_overlaps[k] for k in list(sorted_overlaps)[:top_n]}.keys())
	offers = list(offer_rets[offer_rets["RETAILER"].isin(related_retailers)]["OFFER"])
	df = pd.DataFrame({"OFFERS": offers})
	return df

def perform_cat_inference(search_str, categories, cats, processed_offers):
	labels = pipe(search_str,
		candidate_labels=categories,
	)
	print(labels)
	# labels = [l for i, l in enumerate(labels["labels"]) if labels["scores"][i] > 0.20]
	filtered_cats = list(cats[cats["IS_CHILD_CATEGORY_TO"].isin(labels["labels"][:3])]["PRODUCT_CATEGORY"].unique())
	labels_2 = pipe(search_str,
		candidate_labels=filtered_cats,
	)
	print(labels_2)
	top_labels = labels_2["labels"][:3]



	print(top_labels)
	offers = processed_offers[processed_offers["CATEGORY"].apply(lambda x: bool(set(x) & set(top_labels)))]["OFFER"].reset_index()

	return offers, labels, labels_2

def main():
	col_1, col_2, col_3 = st.columns(3)
	search_str = col_2.text_input("Enter a retailer, brand, or category").capitalize()
	processed_offers = get_processed_offers()
	cats = get_categories_data()
	offer_rets = get_offers_data()
	categories = get_categories(cats)
	# retail_mapping = get_prod_categories()

	if col_2.button("Search", type="primary"):
		retail = is_retailer(search_str)
		direct_offers = check_in_offer(search_str, offer_rets)

		if retail:
			col_2.table(direct_offers)
			# related_offers = get_most_overlap(retailer, offer_rets, retail_mapping, top_n=3)
			# col_2.table(related_offers[~related_offers["OFFER"].isin(list(direct_offers["OFFER"]))])

		else:
			col_2.table(direct_offers)
			related_offers, labels_1, labels_2 = perform_cat_inference(search_str, categories, cats, processed_offers) 
			
			col_2.table(related_offers[~related_offers["OFFER"].isin(list(direct_offers["OFFER"]))])
			col_2.table(pd.DataFrame({"labels": labels_1["labels"][:5], "scores": labels_1["scores"][:5]}))
			col_2.table(pd.DataFrame({"labels": labels_2["labels"][:5], "scores": labels_2["scores"][:5]}))
			# df = get_confidence_charts(labels_2)
			# st.table(df)
if __name__ == "__main__":

	main()

