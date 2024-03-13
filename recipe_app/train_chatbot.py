import torch
from transformers import BertTokenizer, BertForMaskedLM, DataCollatorWithPadding
from transformers import TrainingArguments, Trainer, default_data_collator
import os
import pandas as pd
import pprint
import isodate
import re
import ast
from datasets import Dataset
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def convert_iso_duration(duration_str):
    try:
      duration = isodate.parse_duration(duration_str)
    
      days = duration.days
      hours = duration.seconds // 3600
      minutes = (duration.seconds % 3600) // 60
      
      components = []
      if days:
          components.append(f"{days} day{'s' if days != 1 else ''}")
      if hours:
          components.append(f"{hours} hour{'s' if hours != 1 else ''}")
      if minutes:
          components.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
      
      return ", ".join(components)
    except Exception as e:
      return None

def replace_multi_spaces(s):
  return re.sub("\s\s+" , " ", s)

def parseList(row):
  try:
    ast.literal_eval(row['steps'])
    ast.literal_eval(row['ingredients_raw_str'])
    return True
  except Exception as e:
    return False

# # Tokenize the data
def tokenize_data(row):
    new_line  = '\n'
    bullet_point = '\u2022'
    prep_time = convert_iso_duration(row['PrepTime'])
    cook_time = convert_iso_duration(row['CookTime'])
    total_time = convert_iso_duration(row['TotalTime'])
    prep_time_text = f"Prep time: {prep_time} " if prep_time != None else ""
    cook_time_text = f"Cooking time: {cook_time}, " if cook_time != None else ""
    total_time_text = f"Total time: {total_time}." if total_time != None else ""
    display_time_section = prep_time_text != "" or cook_time_text != "" or total_time_text != ""
    time_section = prep_time_text + cook_time_text + total_time_text + '\n\n' if display_time_section else ""
    try:
      steps = ast.literal_eval(row['steps'])
    except Exception as e:
      steps = []
    try:
      ingredients_list = ast.literal_eval(row['ingredients_raw_str'])
    except Exception as e:
      ingredients_list = []
    # Combine relevant fields into a single text input
    text = f"The dish that we are going to prepare is {row['name']}. {row['description']}\n\nNumber of servings this recipe makes is {row['servings']}. With each serving equivalent to {row['serving_size']}.\n\n{time_section}Nutritional value:\nTotal calories: {row['Calories']}\nFat content: {row['FatContent']}g\nSaturated fat content: {row['SaturatedFatContent']}g\nCholesterol content: {row['CholesterolContent']}mg\nSodium content: {row['SodiumContent']}mg\nCarbohydrate content: {row['CarbohydrateContent']}g\nSugar content: {row['SugarContent']}g\nFiber content: {row['FiberContent']}g\nProtein content: {row['ProteinContent']}g\n\nHere's a basic recipe for {row['name']}:\n\nIngredients:\n{bullet_point} {f'{new_line}{bullet_point} '.join(map(replace_multi_spaces, ingredients_list))}\n\nInstructions:\n{bullet_point} {f'{new_line}{bullet_point} '.join(map(replace_multi_spaces, steps))}"
    result = tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
    return result

cwd = os.getcwd()
df = pd.read_parquet(cwd + '/recipe_app/data/recipe_and_nutrition.parquet')
dataset = Dataset.from_pandas(df)
tokenized_dataset = dataset.map(tokenize_data)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load pre-trained BERT model
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
train_test_split_datasets = dataset.train_test_split(test_size=0.15, seed=42)
train_dataset = train_test_split_datasets['train']
val_dataset = train_test_split_datasets['test']
# Define training arguments
training_args = TrainingArguments(
    output_dir='./finetuned_model',
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Create a Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model('./finetuned_model')