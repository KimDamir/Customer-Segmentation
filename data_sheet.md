# Datasheet 

## Motivation

- **Purpose**

  The dataset is used to create a model for customer segmentation to recommend three tiers of products (Tier 1, Tier 2, Tier 3) based on demographic and purchasing data. 

- **Why was the dataset created?**

  The dataset was created to understand customer behavior and personalize product offerings, ultimately improving marketing strategies.

- **Who created the dataset? Who funded the creation of the dataset?**

  The dataset was originally collected by 'Business Analytics' Using SAS Enterprise Guide and SAS Enterprise Miner.
   

## Composition

- **What do the instances that comprise the dataset represent**

  The dataset contains records for 2,240 customers.
 
- **Features**

  Demographics: Age, Education, Marital_Status, Income, Children
  Customer Activity: Recency, Number of Web Purchases, Number of Store Purchases
  Customer Activity: Recency, Number of Web Purchases, Number of Store Purchases
  Response to Campaigns: AcceptedCmp3, AcceptedCmp4

- **Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctor–patient confidentiality, data that includes the content of individuals’ non-public communications)?**

  No, the dataset does not contain confidential information. The data appear to have been collected in a controlled research setting with likely participant consent.

- **Is the data synthetic or real?** 

  The dataset contains real customer data.


## Collection process

- **How was the data acquired?**

  The data was collected through a customer relationship management system, combining demographic information with purchase histories, campaign responses, and recency data.
  
- **Who collected it?**

  The data was collected by the marketing team of the company running the marketing campaigns.
  
- **Is the data static or dynamic?**

  The dataset is static and represents a snapshot of customer behavior during the campaign period.


## Preprocessing/cleaning/labelling
  
- **Missing values**

  Missing values in Income were replaced with the median.

- **Categorical encoding**

  Categorical variables like Education and Marital_Status were label encoded.

- **Scaling**

  Numerical features like Income and Total Children were scaled using StandardScaler.

- **Feature engineering**

  New features, such as Customer_Since (customer tenure) and Total_Children, were created from existing features.

 
## Uses

- **Targeted marketing campaigns**

  The data is used to segment customers for personalized product recommendations.
  
- **Customer lifetime value prediction**

  Can be used to predict high-value customers based on their spending and demographic attributes.
  

## Limitations

- **Bias**

  The data might be biased toward customers who responded to campaigns, excluding customers who didn’t interact. Certain features (e.g., Income) could introduce bias based on socioeconomic status, influencing product recommendations disproportionately.

- **Temporal limitations**

  The data may not represent customer behavior over time, as it is static.


## Distribution

- **How has the dataset already been distributed?**

  The dataset is publicly available online through Kaggle’s website for academic and research purposes.
  
- **Is it subject to any copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?**

  Not applicable.


## Maintenance

- **Who maintains the dataset?**

  Not maintained.

