import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from ucimlrepo import fetch_ucirepo 
from sklearn.metrics import accuracy_score
import string
from sklearn.preprocessing import LabelEncoder

class forest_pred():
    #constructor
    def __init__(self):
        self.forest = RandomForestClassifier(max_depth=4)

    def pred(self):
        wine = fetch_ucirepo(id=109) 
        X= wine.data.features
        y = wine.data.targets
        #print(f"dataset size {X.shape}")
        #print(f"number of classes: {len(np.unique(y))}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        #flatten the array, random forest classifier 
        self.forest.fit(X_train, y_train.to_numpy().reshape(-1))
        y_pred = self.forest.predict(X_test)
        y_train_pred = self.forest.predict(X_train)

        acc = accuracy_score(y_test, y_pred)
        train_acc = accuracy_score(y_train, y_train_pred)

        print(f"testing accuracy: {acc}") 
        print(f"training accuracy: {train_acc}")
        print(f"depth of the tree: {self.forest.max_depth}")

# model = forest_pred()
# model.pred()


class forest_gen():

    #constructor 
    def __init__(self, path):
        self.path = path
        self.forest = RandomForestClassifier()
        self.tokens = []
        self.encoder = LabelEncoder()
        
    @staticmethod
    def preprocess(text):

        def to_lower(text):
            return text.lower()
        
        #remove brackets and their content
        def remove_brack(text):
            out = ""    #output string without content in brackets
            count = 0   #count of opened brackets
            i = 0       #index of char in text
            while i < len(text):
                char = text[i]
                #if we run into an opening bracket we increase count, to flag that we are inside of a bracket 
                if char == "[":
                    count +=1
                #if we run into a closing bracket we increase the count to flag we ended one whole bracket
                elif char == "]":
                    count -=1
                    #if the count is 0, we are out of the whole bracket, and  we continue to not add the closing bracket char to the output
                    if count == 0:
                        i+=1
                        continue
                #if the count is 0 and char is not a closing bracket, we append the char to the output 
                if count == 0:
                    out += char               
                i +=1
            return out
        
        def remove_punct(text):
            text = text.translate(str.maketrans("", "", string.punctuation))
            text = text.replace('“', '').replace('”', '')  
            return text.strip()

        text = to_lower(text)
        text = remove_brack(text)
        text = remove_punct(text)
        return text

    #map every two words to one 
    def pair_words(self, seq_len = 2):

        #read the text file, preprocess, and split to tokens
        with open(self.path, 'r', encoding='utf-8') as file:
            text = file.read()
            text = self.preprocess(text)
            self.tokens = text.split()

        #encode the tokens to integers to be able to input them to randomForestClasifier
        encoded = self.encoder.fit_transform(self.tokens)

        X, y = [], []
        
        #map every 2 words to 1 
        for i in range(len(encoded) - seq_len):
            X.append(encoded[i:i + seq_len])
            y.append(encoded[i + seq_len])

        return np.array(X), np.array(y)
    
    def train_model(self):

        X, y = self.pair_words()
        #split the data to train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle= True)
        #fit the training data into the forest model
        self.forest.fit(X_train, y_train)
    
    #generate a sentence with length of 10 words
    def gen_words(self, starting_words, sentence_len = 10):
        
        #encode the input words with the encoder
        start_words_encoded = list(self.encoder.transform(starting_words))

        #generate a word every round of the loop, until completing the desired length
        for _ in range(sentence_len):
            
            #take the last two encoded words from the input array and reshape them to be a 1d array so we could use predict_proba
            X_input = np.array(start_words_encoded[-2:]).reshape(1, -1)

            #generate probability distribution for the next word
            prob_dist = self.forest.predict_proba(X_input)[0]
            
            #generate a list of all possible outputs 
            possible_words = np.arange(len(prob_dist))

            #find the encoding of the next word randomly based on the probability distribution 
            next_word_encoded =np.random.choice(possible_words, p = prob_dist)

            #append the encoded next word to the list of words we already have
            start_words_encoded.append(next_word_encoded)

        #decode the words to generate a sentence 
        generated = self.encoder.inverse_transform(start_words_encoded)

        #return the final decoded sentence
        return generated

#forest_model = forest_gen("C:\\Users\\Owner\\OneDrive\\Desktop\\CSEN166\\LAB2\\onefishtwofish.txt")
#forest_model = forest_gen("C:\\Users\\Owner\\OneDrive\\Desktop\\CSEN166\\LAB2\\chap2.txt")
forest_model = forest_gen("C:\\Users\\Owner\\OneDrive\\Desktop\\CSEN166\\LAB2\\chaps1_2_3.txt")
forest_model.train_model()
start_words = ['i', 'do']
words = forest_model.gen_words(start_words)
print(" ".join(words))