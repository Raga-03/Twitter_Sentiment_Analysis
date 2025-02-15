import joblib
from tkinter import *
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import nltk
import seaborn as sns
import re
from tkinter import *
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import nltk
import seaborn as sns
import re
import pickle
global sentiment_model, vectorizer
vectorizer = None

global train_df, test_df, words_freq

# Upload Function for training dataset
def upload_train():
    global train_df, vectorizer
    train_filename = filedialog.askopenfilename(initialdir="dataset", title="Select Training Dataset")
    train_pathlabel.config(text=train_filename)
    train_df = pd.read_csv(train_filename)
    text.delete('1.0', END)
    text.insert(END, 'Training Dataset loaded\n')
    text.insert(END, "Training Dataset Size: " + str(train_df.shape) + "\n")
    text.insert(END, train_df.head().to_string() + "\n")

    # Initialize and fit the vectorizer
    vectorizer = CountVectorizer(stop_words='english')
    vectorizer.fit(train_df['tweet'])  # Fit on the 'tweet' column


# Upload Function for test dataset
def upload_test():
    global test_df
    test_filename = filedialog.askopenfilename(initialdir="dataset", title="Select Test Dataset")
    test_pathlabel.config(text=test_filename)
    test_df = pd.read_csv(test_filename)
    text.insert(END, '\nTest Dataset loaded\n')
    text.insert(END, "Test Dataset Size: " + str(test_df.shape) + "\n")
    text.insert(END, test_df.head().to_string() + "\n")

# Function to display the bar graph for label value counts
def show_graph():
    global train_df
    if train_df is not None:
        if 'label' in train_df.columns:
            plt.figure(figsize=(6, 4))
            train_df['label'].value_counts().plot.bar(color='pink')
            plt.title("Label Value Counts")
            plt.ylabel("Count")
            plt.xlabel("Label")
            plt.show()
        else:
            messagebox.showerror("Error", "The 'label' column is not found in the training dataset")
    else:
        messagebox.showerror("Error", "Please upload the training dataset first")

# Function to display tweet length distribution
def show_tweet_length_distribution():
    global train_df, test_df
    if train_df is not None and test_df is not None:
        if 'tweet' in train_df.columns and 'tweet' in test_df.columns:
            plt.figure(figsize=(6, 4))
            train_df['tweet'].str.len().plot.hist(color='pink', alpha=0.5, label="Train", bins=50)
            test_df['tweet'].str.len().plot.hist(color='orange', alpha=0.5, label="Test", bins=50)
            plt.title("Distribution of Tweet Lengths")
            plt.xlabel("Tweet Length")
            plt.ylabel("Frequency")
            plt.legend()
            plt.show()
        else:
            messagebox.showerror("Error", "'tweet' column is missing in the datasets")
    else:
        messagebox.showerror("Error", "Please upload both the training and test datasets")

# Function to add tweet length column and display the first 10 rows of the training dataset
def add_length_column():
    global train_df, test_df
    if train_df is not None and test_df is not None:
        if 'tweet' in train_df.columns and 'tweet' in test_df.columns:
            train_df['len'] = train_df['tweet'].str.len()
            test_df['len'] = test_df['tweet'].str.len()
            text.delete('1.0', END)
            text.insert(END, 'Tweet length column added to both datasets\n')
            text.insert(END, 'First 10 rows of the training dataset with length column:\n')
            text.insert(END, train_df.head(10).to_string() + "\n")
        else:
            messagebox.showerror("Error", "'tweet' column is missing in the datasets")
    else:
        messagebox.showerror("Error", "Please upload both the training and test datasets")

# Function to display the most frequently occurring words in the training dataset
def show_most_frequent_words():
    global train_df, words_freq
    if train_df is not None:
        if 'tweet' in train_df.columns:
            # CountVectorizer to find word frequencies
            cv = CountVectorizer(stop_words='english')
            words = cv.fit_transform(train_df['tweet'])

            sum_words = words.sum(axis=0)
            words_freq = [(word, sum_words[0, i]) for word, i in cv.vocabulary_.items()]
            words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

            frequency = pd.DataFrame(words_freq, columns=['word', 'freq'])
            
            plt.figure(figsize=(15, 7))
            frequency.head(30).plot(x='word', y='freq', kind='bar', color='blue')
            plt.title("Most Frequently Occurring Words - Top 30")
            plt.xlabel("Word")
            plt.ylabel("Frequency")
            plt.xticks(rotation=90)
            plt.show()
        else:
            messagebox.showerror("Error", "'tweet' column is missing in the training dataset")
    else:
        messagebox.showerror("Error", "Please upload the training dataset first")


# Function to display the neutral word cloud
def show_neutral_word_cloud():
    global train_df
    if train_df is not None:
        if 'tweet' in train_df.columns and 'label' in train_df.columns:
            neutral_words = ' '.join([text for text in train_df['tweet'][train_df['label'] == 0]])
            wordcloud = WordCloud(width=800, height=500, random_state=0, max_font_size=110).generate(neutral_words)

            plt.figure(figsize=(10, 7))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis('off')
            plt.title('The Neutral Words')
            plt.show(block=False)  # Non-blocking plt.show to avoid interfering with the Tkinter window
        else:
            messagebox.showerror("Error", "'tweet' or 'label' column is missing in the training dataset")
    else:
        messagebox.showerror("Error", "Please upload the training dataset first")


# Main window setup
main = tk.Tk()
main.title("Twitter Sentiment Analysis") 
main.geometry("1600x1500")

font = ('times', 16, 'bold')
title = Label(main, text='Twitter Sentiment Analysis', font=("times"))
title.config(bg='Dark Blue', fg='white')
title.config(font=font)           
title.config(height=3, width=145)
title.place(x=0, y=5)

# Text area to display outputs
font1 = ('times', 12, 'bold')
text = Text(main, height=20, width=180)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50, y=120)
text.config(font=font1)

# Button to upload training dataset
font1 = ('times', 13, 'bold')
uploadTrainButton = Button(main, text="Upload Training Dataset", command=upload_train)
uploadTrainButton.place(x=50, y=550)
uploadTrainButton.config(font=font1) 

# Button to upload test dataset
uploadTestButton = Button(main, text="Upload Test Dataset", command=upload_test)
uploadTestButton.place(x=50, y=600)
uploadTestButton.config(font=font1) 




# Labels to display paths of uploaded datasets
train_pathlabel = Label(main)
train_pathlabel.config(bg='DarkOrange1', fg='white')  
train_pathlabel.config(font=font1)           
train_pathlabel.place(x=330, y=550)

test_pathlabel = Label(main)
test_pathlabel.config(bg='DarkOrange1', fg='white')  
test_pathlabel.config(font=font1)           
test_pathlabel.place(x=330, y=600)



# Function to extract and display hashtags
def show_hashtags():
    global train_df
    if train_df is not None:
        if 'tweet' in train_df.columns and 'label' in train_df.columns:
            # Hashtag extraction function
            def hashtag_extract(x):
                hashtags = []
                for i in x:
                    ht = re.findall(r"#(\w+)", i)
                    hashtags.append(ht)
                return hashtags
            
            # Extract hashtags from non-racist/sexist tweets
            HT_regular = hashtag_extract(train_df['tweet'][train_df['label'] == 0])

            # Extract hashtags from racist/sexist tweets
            HT_negative = hashtag_extract(train_df['tweet'][train_df['label'] == 1])

            # Unnest lists
            HT_regular = sum(HT_regular, [])
            HT_negative = sum(HT_negative, [])

            # Frequency distribution of regular hashtags
            a = nltk.FreqDist(HT_regular)
            d = pd.DataFrame({'Hashtag': list(a.keys()), 'Count': list(a.values())})

            # Selecting top 20 most frequent hashtags
            d = d.nlargest(columns="Count", n=20)

            # Display the bar plot for hashtags
            plt.figure(figsize=(16, 5))
            ax = sns.barplot(data=d, x="Hashtag", y="Count")
            ax.set(ylabel='Count')
            plt.title("Top 20 Most Frequent Hashtags")
            plt.xticks(rotation=45)
            plt.show()
        else:
            messagebox.showerror("Error", "'tweet' or 'label' column is missing in the training dataset")
    else:
        messagebox.showerror("Error", "Please upload the training dataset first")




# Function to extract and display hashtags from negative (racist/sexist) tweets
def show_negative_hashtags():
    global train_df
    if train_df is not None:
        if 'tweet' in train_df.columns and 'label' in train_df.columns:
            def hashtag_extract(x):
                hashtags = []
                for i in x:
                    ht = re.findall(r"#(\w+)", i)
                    hashtags.append(ht)
                return hashtags

            HT_negative = hashtag_extract(train_df['tweet'][train_df['label'] == 1])
            HT_negative = sum(HT_negative, [])
            a = nltk.FreqDist(HT_negative)
            d = pd.DataFrame({'Hashtag': list(a.keys()), 'Count': list(a.values())})

            d = d.nlargest(columns="Count", n=20)
            plt.figure(figsize=(16, 5))
            ax = sns.barplot(data=d, x="Hashtag", y="Count")
            ax.set(ylabel='Count')
            plt.title("Top 20 Most Frequent Hashtags (Negative)")
            plt.xticks(rotation=45)
            plt.show()
        else:
            messagebox.showerror("Error", "'tweet' or 'label' column is missing in the training dataset")
    else:
        messagebox.showerror("Error", "Please upload the training dataset first")



# Button to show the label value counts graph
showGraphButton = Button(main, text="Show Label Value Counts", command=show_graph)
showGraphButton.place(x=50, y=650)
showGraphButton.config(font=font1) 

# Button to show tweet length distribution
showTweetLengthButton = Button(main, text="Show Tweet Length Distribution", command=show_tweet_length_distribution)
showTweetLengthButton.place(x=350, y=650)
showTweetLengthButton.config(font=font1) 

# Button to add tweet length column and display first 10 rows
addLengthColumnButton = Button(main, text="Add Length Column", command=add_length_column)
addLengthColumnButton.place(x=650, y=650)
addLengthColumnButton.config(font=font1)

# Button to show most frequently occurring words
showFrequentWordsButton = Button(main, text="Show Most Frequent Words", command=show_most_frequent_words)
showFrequentWordsButton.place(x=950, y=650)
showFrequentWordsButton.config(font=font1)

# Button to show neutral word cloud
showWordCloudButton = Button(main, text="Show Neutral Word Cloud", command=show_neutral_word_cloud)
showWordCloudButton.place(x=50, y=700)
showWordCloudButton.config(font=font1)

# Button to extract and display hashtags
showHashtagsButton = Button(main, text="Show Hashtags", command=show_hashtags)
showHashtagsButton.place(x=350, y=700)
showHashtagsButton.config(font=font1)

# Button to extract and display negative hashtags
showNegativeHashtagsButton = Button(main, text="Show Negative Hashtags", command=show_negative_hashtags)
showNegativeHashtagsButton.place(x=650, y=700)
showNegativeHashtagsButton.config(font=font1)

# Sample word lists for sentiment prediction
positive_words = ["good", "happy", "love", "amazing", "great", "wonderful", "awesome", "fantastic", "positive", "joyful", "excellent", "outstanding", "satisfied", "pleased", "blessed", "hopeful", "excited", "inspiring", "motivated", "healthy", "supportive", "bright", "uplifting", "achieved", "delighted", "grateful", "cheerful", "incredible", "successful", "brilliant"]
negative_words = ["bad", "sad", "hate", "terrible","fuck","Bloody","suck","Damn","Hell","rape","dick", "awful", "horrible", "worst", "disappointed", "angry", "frustrated", "negative", "upset", "miserable", "depressed", "annoyed", "pessimistic", "regret", "painful", "hurt", "failure", "disgusting", "worthless", "unhappy", "dismal", "gloomy", "dreadful", "unfortunate", "angst", "helpless", "stressed"]
neutral_words = ["ok", "fine", "average", "normal", "acceptable", "neutral", "mediocre", "balanced", "alright", "so-so", "decent", "indifferent", "standard", "basic", "unsure", "unaffected", "content", "calm", "steady", "undecided", "ambivalent", "modest", "subdued", "stable", "unconcerned", "unremarkable", "unimportant", "quiet", "relaxed", "indifferent", "unmoved"]


# Function for sentiment prediction
def sentiment_prediction():
    def submit_input():
        # Get the input message from the text entry
        message = input_message.get("1.0", "end-1c").strip()

        if not message:
            messagebox.showerror("Error", "Please enter a message.")
            return
        
        # Tokenize the message into words
        message_words = re.findall(r'\w+', message.lower())

        # Count the occurrences of positive, negative, and neutral words
        positive_count = sum(1 for word in message_words if word in positive_words)
        negative_count = sum(1 for word in message_words if word in negative_words)
        neutral_count = sum(1 for word in message_words if word in neutral_words)

        # Determine sentiment based on the word counts
        if positive_count > negative_count and positive_count > neutral_count:
            sentiment = "Positive"
        elif negative_count > positive_count and negative_count > neutral_count:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        # Display the sentiment prediction result
        text.delete('1.0', END)
        text.insert(END, f"Predicted Sentiment: {sentiment}\n")

    # Create a new window for sentiment prediction input
    sentiment_window = Toplevel(main)
    sentiment_window.title("Sentiment Prediction")
    sentiment_window.geometry("400x300")

    # Label for instructions
    label = Label(sentiment_window, text="Enter your message:", font=("times", 14))
    label.pack(pady=10)

    # Text area to input message
    input_message = Text(sentiment_window, height=4, width=40, font=("times", 12))
    input_message.pack(pady=10)

    # Submit button to predict sentiment
    submit_button = Button(sentiment_window, text="Submit", font=("times", 12), command=submit_input)
    submit_button.pack(pady=10)

# Add a button to open the sentiment prediction window
sentimentPredictionButton = Button(main, text="Sentiment Prediction", command=sentiment_prediction)
sentimentPredictionButton.place(x=50, y=750)
sentimentPredictionButton.config(font=font1)
 

main.config(bg='#F08080')
main.mainloop()