import ollama
import os

model = "llama3.2"
path_in_file = "./data/games.txt"
path_out_file = "./data/games_categorized.txt"

if not os.path.exists(path_in_file):
    print("Input file not found")
    exit()

games = []
with open(path_in_file, 'r') as file:
    for line in file:
        games.append(line.strip())

prompt = f"""
You are a game expert.
PARAMETER temperature: 0.1

here are a list of games:
{games}

1. Categorize the games above into genres:
2. Sort the games by genre and alphabetically within each genre
3. Return the games in the following format:
## game_genre
* game_name_1
* game_name_2

"""

try:
    response = ollama.generate(model=model, prompt=prompt)
    if response:

        if os.path.exists(path_out_file):
            os.remove(path_out_file)

        generated_text = response.get('response', '')
        print(generated_text)
        with open(path_out_file, 'w') as file:
            file.write(generated_text.strip())

        print("Games categorized and saved to", path_out_file)
    else:
        print(f"Error: {response.status_code}")

except Exception as e:
    print(f"Error: {e}")
    exit()


