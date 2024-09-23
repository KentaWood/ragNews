#!/bin/python3

from cgitb import text
from imghdr import tests
import json
from typing import List
import logging
import ragnews
import argparse

class RAGEvaluator:
	def __init__(self):
		'''
        Initialize the RAGClassifier. The __init__ function should take
        an input that specifies the valid labels to predict.
        The class should be a "predictor" following the scikit-learn interface.
        You will have to use the ragnews.rag function internally to perform the prediction.
        '''
		pass

	def predict(self, maskedtext: str, valid_labels: set()):
		'''
		>>> model = RAGEvaluator()
		>>> model.predict('[MASK0] is the democratic nominee.')
		['Harris']
		>>> model.predict('[MASK0] is the democratic presidential nominee, and [MASK1] is the republican nominee.')
		['Harris','Trump']
		>>> model.predict('hi')
		[]
		'''
		#think about:
		# calling ragnews.run llm function directly
		textprompt = f'''
		You are a reliable assistant tasked with predicting the missing words in the masked text based solely on the given information.
		Each masked text also know as tokens follows the format [MASK0], [MASK1], etc.
		Explain your reasoning briefly in no more than 50 words. If the answer is a person, provide only their last name.
		If the given no tokens say say ""
		Only ever give the answer as the predicted words with a space in between and no other pireces of context,explainations or even comments
  
		I'm going to provide you a sentence, and that sentence will have a masked token inside of it that will look like [MASK0] or [MASK1] or [MASK2] and so on.
		And your job is to tell me what the value of that masked token was.
		Valid values include: {valid_labels}

		Example:
		Therefore [MASK0] is Sarah and [MASK1] is John.

		Answer:
		Sarah John
  
		Example:
		Hi I like to ski
  
		Answer:

		Example:
		[MASK0] is the democratic presidential nominee, and [MASK1] is the republican nominee.
  
		Answer:
		Harris Trump

  		'''
		textprompt += "\n" + maskedtext
		# print(textprompt)
  
		db = ragnews.ArticleDB('ragnews.db')
        
  
		output = ragnews.rag(textprompt, db)
		
		return output.split()


	def accuracy(self, file_Json):
		#the valid labels for the test set
		labels = set()
		
		# list of the [{masked_text: text} , {masks: answers}] 
		tests = []
		
  
		with open(file_Json, 'r') as f:
			for line in f:
				dp = json.loads(line)
				labels.update(dp['masks'])
				tests.append(dp)
				
			# tests = [json.loads(line) for line in f]
		
		print(labels)
  
		

		# Initialize counters for total masks and correct predictions
		total_masks = 0
		corrects = 0

		# For debugging, limit the number of tests (remove or adjust as needed)
		# tests = tests[:5]
		
		print(tests)
		for test in tests:
			# Get the predicted masks from the model
			pred = self.predict(test['masked_text'],valid_labels=labels)

		
			actual_masks = test['masks']  # Actual masks from the file
			
			# Count the total number of masks
			total_masks += len(actual_masks)

			# Compare predicted and actual masks
			for i in range(len(actual_masks)):
				if i < len(pred) and pred[i] == actual_masks[i]:
					corrects += 1

			# Debugging output
			print('Predicted:', pred)
			print('Actual:', actual_masks)

		# Return the accuracy as the number of correct predictions divided by total masks
		return corrects / total_masks if total_masks > 0 else 0

     
# focusing on the ragnews
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Evaluate the model accuracy using a data file")
    parser.add_argument('--data_file', type=str, required=True, help="Path to the data file (JSON format)")
    args = parser.parse_args()

    
    model = RAGEvaluator()

    # Call the accuracy function and pass the data file
    print( "Accuracy of the is:",model.accuracy(file_Json=args.data_file))
    
    