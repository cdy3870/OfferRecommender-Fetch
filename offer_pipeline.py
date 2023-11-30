import streamlit as st
from transformers import pipeline
import pickle
import os
import pandas as pd
import ast
import string
import re
from sentence_transformers import SentenceTransformer, util

st.set_page_config(
	page_title="Offer Recommender",
	layout="wide"
)

# Download and cache models
pipe = pipeline(task="zero-shot-classification", model="valhalla/distilbart-mnli-12-3")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Directory of csv files
dire = "DS_NLP_search_data"

# Use Streamlit caching to load data once
@st.cache_data
def get_processed_offers():
	'''
	Load processed offers from exploration notebook and cache

	Returns:
		processed_offers (pd.DataFrame) :  zero-shot categorized offers
	'''
	processed_offers = pd.read_csv(os.path.join(dire, "processed_offers.csv"))
	processed_offers["CATEGORY"] = processed_offers["CATEGORY"].map(ast.literal_eval)

	return processed_offers


@st.cache_data
def get_categories_data():
	'''
	Load raw category data and cache

	Returns:
		cats (pd.DataFrame) :  raw category data
	'''

	cats = pd.read_csv(os.path.join(dire, "categories.csv"))

	return cats


@st.cache_data
def get_offers_data():
	'''
	Load raw offfers data and cache

	Returns:
		cats (pd.DataFrame) :  raw offers data
	'''

	offers = pd.read_csv(os.path.join(dire, "offer_retailer.csv"))

	return offers


@st.cache_data
def get_categories(cats_):
	'''
	Extract, load categories and cache

	Parameters:
		cats_ (pd.DataFrame) : raw categories data

	Returns:
		categories (List) :  child categories
	'''

	categories = list(cats_["IS_CHILD_CATEGORY_TO"].unique())
	for x in ["Mature"]:
		if x in categories:
			categories.remove(x)

	return categories


def check_in_offer(search_str, offer_rets):
	'''
	Determine if the input text is directly in the offer with basic string matching

	Parameters:
		search_str (string) : user text input
		offer_rets (pd.DataFrame) : raw offer data

	Returns:
		df (pd.DataFrame) :  offers with text input
	'''

	offers = []
	for i in range(len(offer_rets)):
		offer_str = offer_rets.iloc[i]["OFFER"]
		parsed_str = offer_str.lower().translate(str.maketrans('', '', string.punctuation))
		parsed_str = re.sub('[^a-zA-Z0-9 \n\.]', '', parsed_str)

		if search_str.lower() in parsed_str.split(" "):
		  offers.append(offer_str)
	df = pd.DataFrame({"OFFER":offers})

	return df


def is_retailer(search_str, threshold=0.5):
	'''
	Determine if the text input is highly likely to be a retailer

	Parameters:
		search_str (string) : user text input
		threshold (int) : probability threshold

	Returns:
		is_ret (boolean) :  true if retailer, false otherwise
	'''

	processed_search_str = search_str.lower().capitalize()
	labels = pipe(processed_search_str,
	  candidate_labels=["brand", "retailer", "item"],
	)

	is_ret = labels["labels"][0] == "retailer" and labels["scores"][0] > threshold

	return is_ret


def perform_cat_inference(search_str, categories, cats, processed_offers):
	'''
	Perform zero shot learning twice and return the offers relevant to the child categories

	Parameters:
		search_str (string) : user text input
		categories (pd.DataFrame) : list of categories
		cats (pd.DataFrame) : raw category data
		processed_offers (pd.DataFrame) : processed_offer_data

	Returns:
		offers (pd.DataFrame) : relevant offers  
		labels (dict) : parent categories and their probability scores
		labels_2 (dict) : child categories and their probability scores
	'''

	labels = pipe(search_str,
		candidate_labels=categories,
	)
	# labels = [l for i, l in enumerate(labels["labels"]) if labels["scores"][i] > 0.20]
	filtered_cats = list(cats[cats["IS_CHILD_CATEGORY_TO"].isin(labels["labels"][:3])]["PRODUCT_CATEGORY"].unique())
	labels_2 = pipe(search_str,
		candidate_labels=filtered_cats,
	)
	top_labels = labels_2["labels"][:3]
	offers = processed_offers[processed_offers["CATEGORY"].apply(lambda x: bool(set(x) & set(top_labels)))]["OFFER"].reset_index()

	return offers, labels, labels_2


def sort_by_similarity(search_str, related_offers):
	'''
	Use sentence embeddings to evaluate the similarity of relevant offers to the text input

	Parameters:
		search_str (string) : user text input
		related_offers (pd.DataFrame) : relevant offers discovered by zero shot learning

	Returns:
		df (pd.DataFrame) : relevant offers and their similiarity scores  
	'''

	temp_dict = {}
	embedding_1 = model.encode(search_str, convert_to_tensor=True)

	for offer in list(related_offers["OFFER"]):
		embedding_2 = model.encode(offer, convert_to_tensor=True)

		temp_dict[offer] = float(util.pytorch_cos_sim(embedding_1, embedding_2))

	sorted_dict = dict(sorted(temp_dict.items(), key=lambda x : x[1], reverse=True))
	df = pd.DataFrame({"OFFER":list(sorted_dict.keys())[:20], "scores":list(sorted_dict.values())[:20]})

	return df


def main():
	# Load and cache data
	col_1, col_2, col_3 = st.columns(3)
	search_str = col_1.text_input("Enter a retailer, brand, or category").capitalize()
	processed_offers = get_processed_offers()
	cats = get_categories_data()
	offer_rets = get_offers_data()
	categories = get_categories(cats)

	if col_1.button("Search", type="primary"):
		# Check offers where the text is directly in it
		retail = is_retailer(search_str)
		direct_offers = check_in_offer(search_str, offer_rets)
		col_2.write("Directly related offers")

		if len(direct_offers) == 0:
			col_2.write("None found")
		else:
			col_2.table(direct_offers)

		if retail:
			# If retail, we directly compare every offer using sentence embeddings
			related_offers = offer_rets[~offer_rets["OFFER"].isin(list(direct_offers["OFFER"]))]
		else:
			# Otherwise, we use zero shot learning with processed offers to narrow down our search
			related_offers, labels_1, labels_2 = perform_cat_inference(search_str, categories, cats, processed_offers) 
			related_offers = related_offers[~related_offers["OFFER"].isin(list(direct_offers["OFFER"]))]

			col_2.write("Parent categories probabilities")
			col_2.table(pd.DataFrame({"labels": labels_1["labels"][:5], "scores": labels_1["scores"][:5]}))
			col_2.write("Child categories probabilities")
			col_2.table(pd.DataFrame({"labels": labels_2["labels"][:5], "scores": labels_2["scores"][:5]}))
		
		col_2.write("Other related offers")
		sorted_offers = sort_by_similarity(search_str, related_offers)

		if len(sorted_offers) == 0:
			col_2.write("None found")
		else:
			col_2.table(sorted_offers)

if __name__ == "__main__":
	main()

