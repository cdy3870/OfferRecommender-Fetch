# Offer/Deal Recommender

To solve this problem, I first broke it down into its simplest form and determined the inputs and outputs. The input is a text string that is meant for a category, brand, or a retailer. The first step is to analyze and process the data. The problem would be simple if we could associate every offer to a category, meaning we join based on offer-brand and brand-category. However, there is a contingency that brands may be associated with different categories. For example, Barilla can fall under the category of red sauce and dry pasta, which works out but can also fall under the category of chips in the data which doesn’t make much sense. I processed the data in the following way.

## Data Processing
1. Every offer that has the same retailer and brand is likely just a general offer for that particular store that applies to every good in that store. For example “Spend $10 at CVS” falls under medicine & treatments, skin care, and more. We assign all of these categories to that offer.
2. Every offer that does not have the same retailer and brand is specific to a product brand. There are instances where every corresponding category applies. For example, “Beyond Meat Plant-Based products, spend $25” falls under all of its categories (plant-based meat, frozen plant-based, packaged meat). However, a deal on "GOYA Coconut Water" falls under water, rice & grains, sauces & marinades, etc. Obviously we only want to fall under "water". 
    * We deal with this using a zero-shot learning model from HuggingFace. Zero-shot learning is an NLP technique that is capable of categorizing text even when it has not been trained on specific provided labels
    * We iterate through each of these offers and categorize it according to its most likely labels (> 0.20 probability)
3. We then concatenate that these tables so that we have a list of offers with its corresponding categor(ies)

After this processing, we perform the following control flow logic using the same zero-shot learning model.

## Pipeline
1. Search for the text input directly in the offer and return this list of offers
2. Determine if the text input is highly likely to be a retailer, otherwise default to brand/category inference
    * We use the zero shot learning model with the labels "brand", "retailer", "category". If there is a > 0.40 score of it being a retailer. Then we continue as follows:
    * Create a mapping of all the categories that a retailer falls under
    * Find the retailers that have the most overlap in the types of goods sold
3. If not highly likely to be a retailer, then we continue with brand/category inference
    * We classify the text as one of the 22 parent categories using the zero-shot learning model
    * We filter based on the ones that have greater than 0.20 score
    * We find the child categories that are associated with this filtered list of parent categories
    * We classify the text again according to this reduced set of child categories
4. Return the corresponding offers

## Assumptions and tradeoffs
One assumption made is that the user will not try to fool the system by using strings that are not real words, contain numbers, or are sentences. Although the model is still probabilistic and will output something that it thinks it is closest to. Allowing for open-ended inputs is a tradeoff of flexibility over more refined results. By not keeping a specific set of brands or categories that is only found in the data, we can allow the model to generalize. Another tradeoff I made is to remove the need for the user to specify if the input is a retailer, brand, or category. If this is specified, then the search can be refined and we also may not accidentally misclassify something as a retailer or not a retailer. However, again we obtain flexibility and generalization this way. Using terms like "beef" and "steak" may lead to similar offers, but may not be actually found in the data. 


## Requirements and Instructions
