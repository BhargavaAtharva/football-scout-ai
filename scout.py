import pandas as pd 
import os
from groq import Groq
import json
from sentence_transformers import SentenceTransformer
import chromadb

df = pd.read_csv('/Users/ath/aiml/football-scout-ai/player_stats.csv', encoding='latin-1')
df['player'] = df['player'].str.encode('latin-1').str.decode('utf-8', errors='replace')
chunks=[]
for row in df.itertuples(index=False):
    chunks.append(f"name: {row.player}, country: {row.country}, age: {row.age}, club: {row.club}, ball control: {row.ball_control}, dribbliing: {row.dribbling}, crossing: {row.crossing}, acceleration: {row.acceleration}, agility: {row.agility}, heading: {row.heading}, shot power: {row.shot_power}, finishing: {row.finishing}, penalties: {row.penalties}, gk positioning: {row.gk_positioning}, gk handling: {row.gk_handling}, gk reflexes: {row.gk_reflexes}, value: {row.value}")

client_groq = Groq(api_key=os.environ.get("GROQ_API_KEY"))

model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings= model.encode(chunks)

print("embeddings complete")
print(embeddings.shape)
print(f"First embedding (first 5 numbers): {embeddings[0][:5]}")

client=chromadb.Client()
collection=client.create_collection("footballers")

batch_size = 500
for i in range(0, len(chunks), batch_size):
    batch_chunks = chunks[i:i+batch_size]
    batch_embeddings = embeddings[i:i+batch_size].tolist()
    batch_ids = [str(j) for j in range(i, i+len(batch_chunks))]
    
    collection.add(
        documents=batch_chunks,
        embeddings=batch_embeddings,
        ids=batch_ids
    )
    print(f"Stored batch {i//batch_size + 1}")

print(f"collection count= {collection.count()}")
def classify_query(query) :
    response= client_groq.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role":"system",
            "content":"""classify the query that will be passed to you in either 2 categories:
            one category is the STAT query category which means the query asked is based on stats for example the highest,lowest,best,worst searching question which can be calculated based on a numerical stat
            the other category is a semantic query which basically means that query given is a descriptive query which cannot necessarily be decided by fetching certain stat of a playyer for example compare messi vs ronaldo or who is a better midfielder or what is the best strength of a certain player
            return in json only format for example {"category":"stat", "column":"dribbling"} or {"category":"semantic"}
            for stat type of query the column should only be only from these player,country,age,club,ball_control,dribbling,crossing,acceleration,agility,heading,shot_power,finishing,penalties,gk_positioning,gk_handling,gk_reflexes,value"""
        },
        {
            "role":"user",
            "content":query
        }
    ]
    
    )
    return json.loads(response.choices[0].message.content)

def stat_query(column):
    result=df.nlargest(5,column)[['player','club',column]]
    return result.to_string(index=False)


def rag_query(query):
    query_embedding=model.encode([query]).tolist()

    result=collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )

    

    context = "\n".join(result['documents'][0])

    response = client_groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "Answer the question using only the context provided. Be concise."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            }   
        ]   
    )
    return (f"\nAnswer: {response.choices[0].message.content}")


def handle_query(query):
    result=classify_query(query)
    if(result["category"]=="stat"):
        return stat_query(result["column"])
    else:
        return rag_query(query)

print(handle_query("who has the best dribbling?"))
print(handle_query("find me a creative midfielder who can press hard"))