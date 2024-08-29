import csv
import re
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from itertools import combinations
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import streamlit as st
import io
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import calendar
import pandas as pd
import datetime
from itertools import islice

def load_and_process_data(uploaded_file):
    data = []

    # Read lines from uploaded_file and decode each line
    decoded_file = [line.decode('utf-8') for line in uploaded_file.readlines()]
    csv_reader = csv.reader(decoded_file)

    for row in csv_reader:
        data.append(row)

    headers = data[0]
    data = data[1:]

    # Convert price to float
    for row in data:
        try:
            row[4] = float(row[4])
        except ValueError:
            row[4] = 0.0

        try:
            row[5] = float(re.sub('[^0-9.]', '', row[5]))
        except ValueError:
            row[5] = 0.0  # Or another placeholder value

    # Using label encoders to convert string data into numeric for clustering
    brand_encoder = LabelEncoder()
    date_encoder = LabelEncoder()

    brands = [row[2] for row in data]
    dates = [row[6] for row in data]

    encoded_brands = brand_encoder.fit_transform(brands)
    encoded_dates = date_encoder.fit_transform(dates)

    for idx, row in enumerate(data):
        row[2] = encoded_brands[idx]  # Encoded brand
        row[6] = encoded_dates[idx]  # Encoded date

    return data, headers,brand_encoder, date_encoder

def visualize_data(data, headers,brand_encoder, date_encoder):
    def apply_kmeans_on_single_column(data, column_idx, encoder=None, n_clusters=4):
        # Extract the relevant column data
        X = np.array([[row[column_idx]] for row in data])
        
        # If the data is continuous, it's a good idea to scale it
        if isinstance(X[0][0], (int, float)) and column_idx not in [5, 6]:  # Exclude special columns from scaling
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X
        
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X_scaled)
        labels = kmeans.labels_
        
        # Calculate the silhouette score
        silhouette_avg = silhouette_score(X_scaled, labels)
        st.text(f"Silhouette Score for column {headers[column_idx]}: {silhouette_avg:.2f}")

        
        # Visualization
        plt.figure(figsize=(10, 6))
        plt.scatter(X, [i for i in range(len(X))], c=kmeans.labels_, cmap='rainbow')
        
        # Get centroid colors
        centroid_colors = [plt.cm.rainbow(label/n_clusters) for label in range(n_clusters)]
        
        centroid_scatter = plt.scatter(kmeans.cluster_centers_, [i for i in range(n_clusters)], s=200, c=centroid_colors, marker='X')
        
        # If an encoder is provided and it's a label encoded column, reverse the encoded values and set legend
        if encoder:
            original_values = [encoder.inverse_transform([int(center)])[0] for center in np.round(kmeans.cluster_centers_).flatten()]
            # Create a legend for encoded columns
            legend_handles = [plt.Line2D([0], [0], marker='X', color='w', label=value, markersize=10, markerfacecolor=centroid_colors[i]) for i, value in enumerate(original_values)]
            plt.legend(handles=legend_handles, title=headers[column_idx])
        else:
            # Create a legend for other columns with centroids
            legend_handles = [plt.Line2D([0], [0], marker='X', color='w', label=f"Centroid {i+1}", markersize=10, markerfacecolor=centroid_colors[i]) for i in range(n_clusters)]
            plt.legend(handles=legend_handles, title=headers[column_idx])
        
        # Special x-labels for price and date columns
        if column_idx == 5:
            plt.xlabel("Price")
            flattened_X = X.flatten()
            plt.xticks(ticks=np.linspace(min(flattened_X), max(flattened_X), n_clusters), labels=[f"{tick:.2f}" for tick in np.linspace(min(flattened_X), max(flattened_X), n_clusters)])
        elif column_idx == 6:
            plt.xlabel("Date")
            flattened_X = X.flatten()
            plt.xticks(ticks=np.linspace(min(flattened_X), max(flattened_X), n_clusters), labels=[encoder.inverse_transform([int(tick)])[0] for tick in np.linspace(min(flattened_X), max(flattened_X), n_clusters)])
        else:
            plt.xlabel(headers[column_idx])
        
        plt.ylabel('Data Points Index')
        plt.title(f'KMeans Clustering on {headers[column_idx]}')
        plt.show()
        st.pyplot(plt.gcf())
        return kmeans.cluster_centers_
 
    # Apply KMeans and capture the centroids
    centroids_for_brand = apply_kmeans_on_single_column(data, 2, brand_encoder)
    
    with st.expander("Quantity Centroid Details"):
        centroids_for_quantity  = apply_kmeans_on_single_column(data, 4)
        
        # Display the meaning of the centroids for the quantity
        quantity_centroid_meanings = [
            "Very low quantity (rare or seldom purchased items).",
            "Very high quantity (major bulk or special event purchases).",
            "Low to average quantity (commonly purchased items).",
            "Above average quantity (bulk or frequently purchased items)."
            
        ]

        for i, center in enumerate((centroids_for_quantity)):
            st.text(f"Centroid {i+1}: {quantity_centroid_meanings[i]} (Value: {center[0]:.2f})")
            
    with st.expander("Price Centroid Details"):
        centroids_for_price = apply_kmeans_on_single_column(data, 5)
        
        # Display the meaning of the centroids for the price
        price_centroid_meanings = [
            
            "Middle ground between lowest and average-priced items.",
            "Higher price range (more premium but not the most expensive).",
            "Highest price range (most premium or expensive).",
            "Lowest price range (most affordable)."
            
        ]

        for i, center in enumerate((centroids_for_price)):
            st.text(f"Centroid {i+1}: {price_centroid_meanings[i]} (Value: {center[0]:.2f})")

        
    centroids_for_date = apply_kmeans_on_single_column(data, 6, date_encoder)

    




    #clustering two clumn together
    # Custom function to apply KMeans and calculate silhouette score
    def apply_kmeans_and_evaluate(data, columns, encoders=None, n_clusters=3):
        col_indices = [headers.index(col) for col in columns]
        X = np.array([[row[i] for i in col_indices] for row in data])
        
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
        labels = kmeans.labels_
        
        # Calculate the silhouette score
        silhouette_avg = silhouette_score(X, labels)
        st.write(f"Silhouette Score for columns {columns}: {silhouette_avg:.2f}")

        
        # Plotting the clusters
        plt.figure(figsize=(10, 6))
        plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='rainbow')
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='black', marker='X')
        
        # Adjust labels for BRAND vs DATE
        if columns == ["BRAND", "DATE"]:
            unique_brands = sorted(list(set(encoders[0].inverse_transform(X[:, 0].astype(int)))))
            dates = sorted(list(set(encoders[1].inverse_transform(X[:, 1].astype(int)))))
            min_date = dates[0]
            max_date = dates[-1]
            mid_date = dates[len(dates)//2]
            plt.xticks(ticks=range(len(unique_brands)), labels=unique_brands)
            plt.yticks(ticks=[0, len(dates)//2, len(dates)-1], labels=[min_date, mid_date, max_date])
        elif columns == ["ITEM", "PRICE"]:
            total_items = len(set([row[headers.index("ITEM")] for row in data]))
            plt.xticks(ticks=range(0, total_items, total_items//3))
        
        plt.xlabel(columns[0])
        plt.ylabel(columns[1])
        
        # Create a legend for cluster centers
        legend_handles = [plt.Line2D([0], [0], marker='X', color='w', label=f"Consumers {i+1}", markersize=10, markerfacecolor='black') for i in range(n_clusters)]
        plt.legend(handles=legend_handles, title="Group of Consumers")
        
        plt.title(f'KMeans Clustering: {columns[0]} vs {columns[1]}')
        plt.show()
        st.pyplot(plt.gcf())

    # 1. Brand and Date
    apply_kmeans_and_evaluate(data, ["BRAND", "DATE"], encoders=[brand_encoder, date_encoder])

    # 2. Price and Quantity
    apply_kmeans_and_evaluate(data, ["PRICE", "QUANTITY"])

    product_encoder = LabelEncoder()
    encoded_products = product_encoder.fit_transform([row[3] for row in data])

    for idx, row in enumerate(data):
        row[3] = encoded_products[idx]  # Encoded product

    # Apply k-means clustering on 'Product' and 'Month'
    apply_kmeans_and_evaluate(data, ["ITEM", "PRICE"])
    
    
 



def display_association_results(uploaded_file):
    text_data = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
    data = list(islice(csv.reader(text_data), 1, None))

    month_name = {
        '01': 'January',
        '02': 'February',
        '03': 'March',
        '04': 'April',
        '05': 'May',
        '06': 'June',
        '07': 'July',
        '08': 'August',
        '09': 'September',
        '10': 'October',
        '11': 'November',
        '12': 'December'
    }

    # Grouping products by transactions and month
    monthly_transactions = defaultdict(lambda: defaultdict(set))
    
    for row in data:
        month = row[6].split("/")[0]
        transaction_id = row[1]
        item = row[3]
        monthly_transactions[month][transaction_id].add(item)

    # Initializing itemset_support for all months combined
    all_itemset_support = {}
    all_transactions = defaultdict(set)
    
    # For each month in the data
    for month, transactions in sorted(monthly_transactions.items()):
        st.subheader(f"Month: {month_name[month]}")
        
        # Count occurrences of individual items for the month
        item_counts = Counter(item for items in transactions.values() for item in items)

        # Calculate support for each item for the month
        total_transactions = len(transactions)
        item_support = {item: count/total_transactions for item, count in item_counts.items()}

        # Sorting the items by support and get the top 5
        sorted_items = sorted(item_support.items(), key=lambda x: x[1], reverse=True)[:5]

        # Plotting the top 5 items with highest support for the month
        items, supports = zip(*sorted_items)
        plt.figure(figsize=(10, 6))
        plt.barh(items, supports, color='lightblue')
        plt.xlabel('Support')
        plt.ylabel('Items')
        plt.title(f'Top 5 Items with Highest Support for Month {month_name[month]}')
        plt.gca().invert_yaxis()
        st.pyplot(plt.gcf())
        
        # Combining transactions for all months
        all_transactions.update(transactions)

    # Count occurrences of individual items
    item_counts = Counter(item for items in all_transactions.values() for item in items)

    # Calculate support for each item
    total_transactions = len(all_transactions)
    itemset_support = {frozenset([item]): count/total_transactions for item, count in item_counts.items()}

    # Count itemsets
    itemset_counts = Counter()
    max_itemset_size = 2
    for items in all_transactions.values():
        if len(items) < 2:
            continue
        for L in range(2, min(len(items), max_itemset_size) + 1):
            for subset in combinations(items, L):
                itemset = frozenset(subset)
                itemset_counts[itemset] += 1

    # Calculate support for each itemset
    itemset_support.update({itemset: count/total_transactions for itemset, count in itemset_counts.items()})

    # Sorting the itemsets by support
    sorted_itemsets = sorted(itemset_support.items(), key=lambda x: x[1], reverse=True)

    # Extract 2-itemsets
    sorted_2_itemsets = [(itemset, support) for itemset, support in sorted_itemsets if len(itemset) == 2][:10]

    # Calculate Association Rules
    association_rules = []
    for itemset, support in itemset_support.items():
        if len(itemset) == 2:
            for item in itemset:
                antecedent = frozenset([item])
                consequent = itemset - antecedent

                antecedent_support = itemset_support.get(antecedent, 0)
                confidence = support / antecedent_support
                association_rules.append((antecedent, consequent, confidence))

    # Sort rules by support and then confidence
    sorted_rules = sorted(association_rules, key=lambda x: (itemset_support[x[0] | x[1]], x[2]), reverse=True)

    # Displaying the results in Streamlit
    st.subheader("Top Items with Support:")
    top_items_data = [{"Itemset": ', '.join(itemset), "Support": support} for itemset, support in sorted_itemsets[:10]]
    st.table(top_items_data)


    st.subheader("Pairs with High Support and High Confidence:")
    pairs_data = [{"Antecedent": ', '.join(antecedent), 
                "Consequent": ', '.join(consequent), 
                "Support": itemset_support[antecedent | consequent],
                "Confidence": confidence} for antecedent, consequent, confidence in sorted_rules[:10]]
    st.table(pairs_data)
    
    sorted_pairs_data = sorted(pairs_data, key=lambda x: x['Confidence'], reverse=True)

    # Extract pairs labels and their confidence values for the top 10 pairs
    pairs_labels = [f"{pair['Antecedent']} -> {pair['Consequent']}" for pair in sorted_pairs_data[:10]]
    pairs_confidences = [pair['Confidence'] for pair in sorted_pairs_data[:10]]

    # Create the horizontal bar chart using matplotlib
    plt.figure(figsize=(10, 6))
    plt.barh(pairs_labels, pairs_confidences, color='lightblue')
    plt.xlabel('Confidence')
    plt.ylabel('Pairs')
    plt.gca().invert_yaxis()  # to have the pair with the highest confidence at the top
    plt.title('Top 10 Pairs by Highest Confidence')
    st.pyplot(plt.gcf())
    


    

    # Plotting
    if sorted_2_itemsets:
        labels = [', '.join(itemset) for itemset, _ in sorted_2_itemsets]
        supports = [support for _, support in sorted_2_itemsets]

        plt.figure(figsize=(10, 6))
        plt.barh(labels, supports, color='lightblue')
        plt.xlabel('Support')
        plt.ylabel('Itemsets')
        plt.title('Top 10 2-Itemsets with Highest Support')
        plt.gca().invert_yaxis()
        st.pyplot(plt.gcf())
    else:
        st.write("No 2-itemsets available for plotting!")




def run_random_forest_regression(uploaded_file):
    text_data = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
    
    # Read the CSV from the decoded content
    data = [row for row in csv.reader(text_data)]
    header = data[0]  
    df = pd.DataFrame(data[1:], columns=header)  
    
    # Convert DATE to datetime format and extract month names
    df['DATE'] = pd.to_datetime(df['DATE'])
    df['Month_Name'] = df['DATE'].dt.strftime('%B')
    df['BRAND'] = df['BRAND'].str.strip().str.upper()

    # Brands in the data
    brands = df['BRAND'].unique()
    month_order = sorted(df['Month_Name'].unique().tolist(), key=lambda x: datetime.datetime.strptime(x, '%B'))

    # Set up plotting for brand popularity
    fig, ax = plt.subplots(figsize=(12, 6))

    # Evaluation scores
    evaluation_scores = {}

    for brand in brands:
        brand_data = df[df['BRAND'] == brand]
        monthly_sales = brand_data.groupby('Month_Name').size().reindex(month_order, fill_value=0).reset_index(name='Occurrences')
        months = np.array(range(1, len(monthly_sales) + 1)).reshape(-1, 1)
        
        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(months, monthly_sales['Occurrences'], test_size=0.3, random_state=42)
        
        # Using a Random Forest Regressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # Evaluate the model
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        evaluation_scores[brand] = (mae, mse, r2)
        
        # Predict the occurrences for the next 3 months
        future_months = np.array(range(len(monthly_sales) + 1, len(monthly_sales) + 4)).reshape(-1, 1)
        future_predictions = model.predict(future_months)
        
        # Plot the data
        all_data = list(monthly_sales['Occurrences']) + list(future_predictions)
        all_months = month_order + [calendar.month_name[(month_order.index(month_order[-1]) + 1 + i) % 12 + 1] for i in range(3)]
        ax.plot(all_months, all_data, marker='o', label=brand)
        ax.scatter(month_order, monthly_sales['Occurrences'], color='gray')  # actual data points
  
    # Formatting the plot for brand popularity
    ax.set_xlabel('Month')
    ax.set_ylabel('Number of Occurrences')
    ax.set_title('Predicted Brand Popularity in Upcoming Months')
    ax.legend()
    ax.set_ylim(bottom=0)
    ax.grid(True)
    st.pyplot(fig)  # Display the brand popularity plot in Streamlit

    # Plotting the evaluation scores
    evaluation_df = pd.DataFrame(evaluation_scores, index=['MAE', 'MSE', 'R2']).T
    st.markdown(f'<style>table {{font-size: 20px;}}</style>', unsafe_allow_html=True)
    st.table(evaluation_df)









