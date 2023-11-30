# Offer/Deal Recommender

To solve this problem, I first broke it down into its simplest form and determined the inputs and outputs. The input is a text string that is meant for a category, brand, or a retailer. The first step is to analyze and process the data. The problem would be simple if we could associate every offer to a category, meaning we join based on offer-brand and brand-category. However, there is a contingency that brands may be associated with different categories. For example, Barilla can fall under the category of red sauce and dry pasta, which works out but can also fall under the category of chips in the data which doesn’t make much sense. I processed the data in the following way.

## Data Processing
1. Every offer that has the same retailer and brand is likely just a general offer for that particular store that applies to every good in that store. For example “Spend $10 at CVS” falls under medicine & treatments, skin care, and more. We assign all of these categories to that offer.
2. Every offer that does not have the same retailer and brand is specific to a product brand. There are instances where every corresponding category applies. For example, “Beyond Meat Plant-Based products, spend $25” falls under all of its categories (plant-based meat, frozen plant-based, packaged meat). However, a deal on "GOYA Coconut Water" falls under water, rice & grains, sauces & marinades, etc. Obviously we only want to fall under "water". 
    * We deal with this using a zero-shot learning model from HuggingFace. Zero-shot learning is an NLP technique that is capable of categorizing text even when it has not been trained on specific provided labels
    * We iterate through each of these offers and categorize it according to its most likely labels (> 0.20 probability)
3. We then concatenate that these tables so that we have a list of offers with its corresponding categor(ies)

After this processing, we perform the following control flow logic using the same zero-shot learning model and a sentence embedding model.

## Pipeline
1. Search for the text input directly in the offer and return this list of offers
2. Determine if the text input is highly likely to be a retailer, otherwise default to brand/category inference
    * We use the zero shot learning model with the labels "brand", "retailer", "category". If there is a > 0.40 score of it being a retailer. Then we continue as follows:
    * Extract a sentence embedding using a pre-trained embedding model for each offer and compare with the retailer text input, sort and take the top 20
    * Something that I didn't do but could have added is to first find other retailers that have a high overlap with types of goods sold, then we narrow the search down before comparing with each offer

3. If not highly likely to be a retailer, then we continue with brand/category inference. The rationale behind this separation is that we can leverage the human-labeled categories that help us have a more refined search when provided a brand or category. The same doesn't really apply for retailers that a have broad range of categories. 
    * We classify the text as one of the 22 parent categories using the zero-shot learning model
    * We filter based on the ones that have greater than 0.20 score
    * We find the child categories that are associated with this filtered list of parent categories
    * We classify the text again according to this reduced set of child categories
    * Extract a sentence embedding using a pre-trained embedding model for each offer and compare with the retailer text input, sort and take the top 20 
4. Return the corresponding offers

## Assumptions
1. One assumption made is that the user will not try to fool the system by using strings that are not real words, contain numbers, or are sentences. Although the model is still probabilistic and will output something that it thinks it is closest to. 

2. Another assumption is that we only need to provide the offer and not the corresponding retailer and brand. For example, one offer may be present for multiple brands or categories so it would ideally appear for both. However, I made the assumption that we only care about relationship between the offer and the text input.  

## Tradeoffs
1. Allowing for open-ended inputs is a tradeoff of flexibility over more refined results. By not keeping a specific set of brands or categories that is only found in the data, we can allow the model to generalize. 

2. Another tradeoff I made is to remove the need for the user to specify if the input is a retailer, brand, or category. If this is specified, then the search can be refined and we also may not accidentally misclassify something as a retailer or not a retailer. However, again we obtain flexibility and generalization this way. Using terms like "beef" and "steak" may lead to similar offers, but may not be actually found in the data. 

3. Lastly, I made the tradeoff of speed over performance in the case of using a smaller zero-shot learning model. The performance difference is negligble from online research that performed various experiments on zero-shot learning models. 


## Conclusion
I believe this pipeline would work much better if there were more offers associated with the provided categories. For example, if we use "Huggies" as our input, we see that the model correctly finds the subcategories, "Diapering", "Potty Training", and "Baby Safety", but there are no offers that are associated with the brands of these categories. Therefore, it defaults to other categories that aren't super relevant.

## Requirements and Instructions
This app is hosted on HuggingFace Spaces: https://huggingface.co/spaces/cdy3870/Fetch_App. It takes a minute to load both the models but is cached afterwards. Unfortunately the free cpu they provide is quite slow for inferencing so I would suggest running locally. Inferencing is still a bit slow locally but is obviously device independent. If hosted on services where a GPU is enabled, the app would be much more efficient.

1. Install the requirements in a virtual environment

```
pip install -r requirements.txt
```


2. Run the following command

```
streamlit run offer_pipeline.py
```

3. The HuggingFace models take a minute to download, but it is cached after downloading. 

4. Enter in a term (pepsi, target, gummy bears for example). The enter button is not implemented so you need to press the search button. 
