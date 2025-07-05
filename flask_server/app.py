from flask import Flask, request, jsonify
import requests
import time
import json
import re
from huggingface_hub import InferenceClient
from openai import OpenAI
from groq import Groq
from dotenv import load_dotenv
import os
from groq import Groq
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import random
load_dotenv()

app = Flask(__name__)

@app.route('/receive_data', methods=['POST'])
def receive_data():
    sentence_sets = request.json
    print(f"Received data: {sentence_sets}")
    results = []
    for sentence_set in sentence_sets:
      prompt = f"""
      Compare the following sentences and give a similarity score between 0 (completely different) and 1 (identical):

      Sentence 1: {sentence_set["source_sentence"]}
      Sentence 2: {sentence_set["user_sentences"]}

      Give only the score.
      Respond ONLY in the following JSON format:
      {{
        "similarity_score": float
      }}
      """
      client = Groq(api_key=os.getenv("GROQ_API_KEY"))
      try:
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
            max_completion_tokens=3901,
            top_p=1,
            stream=False,
            response_format={"type": "json_object"},
            stop=None,
        )
        message_content = completion.choices[0].message.content
        parsed_json = json.loads(message_content)  # Converts string to dict
        similarity_score = float(parsed_json["similarity_score"])  # Now it's accessible
      except requests.exceptions.HTTPError as http_err:
          print("HTTP error occurred:", http_err)
          return json.dumps({"error": "HTTP error"})

      accuracy = f"{similarity_score * 100:.2f}%"  # Convert to percentage and format to two decimal places
      
      result_set = {
          "your_answer": sentence_set["source_sentence"],
          "right_answer": sentence_set["user_sentences"],
          "accuracy": accuracy
      }
      results.append(result_set)

    response_data = {'message': 'Data received successfully', 'results': results}
    return jsonify(response_data)

def get_embedding_get_retriever():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(
    "faiss_index_book",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    return retriever

def get_summary(topic_content):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    try:
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            messages = [
            {
                "role": "system",
                "content": "You are an AI assistant that generates recommendations and responses in natural, human-like language. You must avoid generating any code, focusing purely on textual explanations and advice."
            },
            {
                "role": "user",
                "content": f"""You are a summarizer. Summarize the following blog content into a clear, concise summary within 100 words. 
            The summary should capture only the most important technical ideas and key points, avoiding fluff or repetition. 
            This summary will be used to retrieve relevant content from a book, so ensure it's fact-focused and topic-aligned.

            BLOG CONTENT:
            {topic_content}

            SUMMARY (100 words max):"""
                }
            ],
            temperature=0.4,
            top_p=1,
            stream=False,
            stop=None,
          )
        output = completion.choices[0].message.content
    except requests.exceptions.HTTPError as http_err:
          print("HTTP error occurred:", http_err)
          return json.dumps({"error": "HTTP error"})

    return output


mcq_prompt = PromptTemplate(
  input_variables=["context"],
  template="""
      You are an expert quiz generator. Your task is to generate exactly two high-quality, diverse multiple-choice questions (MCQs) based only on the context provided below.

      ### Context:
      {context}

      ### Guidelines:
      - Only use information from the context. Do not use external knowledge.
      - Cover different subtopics or aspects from the context.
      - Avoid overlapping or similar questions.
      - Avoid clustering the correct answers (e.g., don't always make it the first option).
      - All options must be reasonable and not obviously incorrect.
      - Each question must have exactly 4 options.
      - Provide the **index** (0-3) of the correct option.

      ### Output Format (JSON only):
      {{
        "mcq": [
          {{
            "question": "Sample question text?",
            "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
            "correct": 2
          }},
          ...
        ]
      }}
      Respond with a--
      **valid JSON object only take care about each comma, brackets** 
      **No additional comments this is very strict instruction** 
      """
      )


descriptive_prompt = PromptTemplate(
          input_variables=["context"],
          template="""
      You are an expert examiner. Based on the context below, generate exactly two descriptive questions and their accurate, concise answers.

      ### Context:
      {context}

      ### Guidelines:
      - Only use information from the context. Do not hallucinate or invent facts.
      - Questions should test conceptual understanding, reasoning, or explanation.
      - Ensure diversity in question scope and difficulty.
      - Keep answers clear, factual, and directly supported by the context.
      - Avoid repetition and generalizations.
      - generate question such that answer length vary between 0 to 50 words

      ### Output Format (JSON only):
      {{
        "descriptive": [
          {{
            "question": "What is XYZ?",
            "answer": "XYZ is defined as..."
          }},
          ...
        ]
      }}
      Respond with a--
      **valid JSON object only take care about each comma, brackets** 
      **No additional comments this is very strict instruction** 
      """
      )
def get_questions(retriever, summary, llm):
    chunks = retriever.invoke(summary)

    # Make sure it's a list
    if isinstance(chunks, dict):
        chunks = [chunks]

    # Step 2: Collect all MCQs and Descriptive Q&As
    all_mcqs = []
    all_desc = []

    for chunk in chunks:
        context = chunk.page_content

        # Prepare prompts
        final_prompt_mcq = mcq_prompt.invoke({"context": context})
        final_prompt_des = descriptive_prompt.invoke({"context": context})

        # Generate answers
        answer1 = llm.invoke(final_prompt_mcq).content
        answer2 = llm.invoke(final_prompt_des).content
        print(answer1)
        # print("*"*100)
        print(answer2)
        # Parse the JSON responses (must be valid JSON strings)
        try:
            mcq_data = json.loads(answer1)["mcq"]
            desc_data = json.loads(answer2)["descriptive"]

            all_mcqs.extend(mcq_data)     # 2 mcqs per chunk
            all_desc.extend(desc_data)    # 2 descriptive per chunk
        except Exception as e:
            print("❌ Error parsing JSON:", e)

    # Step 3: Shuffle and select 5 of each
    print(len(all_mcqs))
    random.shuffle(all_mcqs)
    random.shuffle(all_desc)

    final_mcqs = all_mcqs[:5]
    final_desc = all_desc[:5]
    return final_mcqs,final_desc



def generate_quiz( topic_content):
    retriever = get_embedding_get_retriever()
    # Get summary of the topic content
    summary = get_summary(topic_content)
    llm = ChatGroq(model_name="llama3-8b-8192")
    mcq,desc=get_questions(retriever, summary, llm)
    def generate_quiz_array():
      quiz_array = {
          "mcq": [],
          "descriptive": []
      }

      for item in mcq:
          quiz_array["mcq"].append({
              "question": item["question"],
              "options": item["options"],
              "correct": str(item["correct"])  # converting index to string
          })

      for item in desc:
          quiz_array["descriptive"].append({
              "question": item["question"],
              "answer": item["answer"]
          })

      return quiz_array
    return generate_quiz_array()

@app.route('/receive_quiz_topic', methods=['POST'])
def receive_quiz_topic():
    """Endpoint to receive a quiz topic and generate questions."""
    try:
        topic = request.json.get("topicTitle", "")
        topic_content = request.json.get("topicContent", "")
        if not topic:
            return jsonify({"error": "Topic is required"}), 400

        # Generate the quiz
        quizArray = generate_quiz(topic_content)
        return jsonify(quizArray), 200
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "An error occurred while generating the quiz."}), 500
    

def  gen_prompt_for_improvement():
    prompt = """
You are an AI assistant that generates natural language sentences for users to improve their learning on an educational platform where quizzes are taken on various topics. Your task is to generate personalized feedback for the user based **exclusively** on the topics and scores provided in the `progress_on_quiz` variable.

`progress_on_quiz`: ${ new_array }

### Goal:
Provide personalized recommendations in simple natural language sentences in short bullet points.

### Instructions:
1. Analyze the topics and scores in the `progress_on_quiz`
2. For each topic in the input:
   - Assess the user's performance based on their score.
   - Provide concise, actionable feedback directly related to the topic and score in natural language sentences.
3. Ensure the feedback includes:
   - A summary of the user's performance for the topic.
   - Specific suggestions to improve their understanding.
   - Optional resource recommendations relevant to the topic.
4. **Do not** write code or reference topics or scores not included in the input.
5. Output only plain text in the format specified below.

### Output Format:
For each topic:
- Topic: topicTitle , Score: quizScore
  - Action Plan: personalized feedback based on the score, it should be of 200 words or atleast 180, focusing on what the user can do to improve.
  - Resource: optional resource suggestion for further learning,These resources should be correct web urls or youtube links.

### Example Output:
- Topic: Time Complexity (Score: 45)
  - Action Plan: Review basics of Big O notation and practice complexity analysis.it should be of 200 words or atleast 180.
  - Resource: [Beginner's Guide to Time Complexity (YouTube)], These resources should be correct web urls or youtube links.

- Topic: Sorting Algorithms (Score: 75)
  - Action Plan: Explore advanced techniques like radix sort and external sorting. it should be of 200 words or atleast 180,
  - Resource: [LeetCode Sorting Section],These resources should be correct web urls or youtube links.

### Note:
- Stick strictly to the topics and scores provided in `progress_on_quiz`.
"""
    return prompt


# for evaluating quiz using LLM
def get_model_response_improvement(prompt):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    try:
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            messages = [
                {
                  "role": "system",
                  "content": "You are an AI assistant that generates recommendations and responses in natural, human-like language. You must avoid generating any code, focusing purely on textual explanations and advice."
                },
          { "role": "user", "content": prompt },
        ],
            temperature=0,
            max_completion_tokens=3901,
            top_p=1,
            stream=False,
            stop=None,
        )
        output = completion.choices[0].message.content
    except requests.exceptions.HTTPError as http_err:
          print("HTTP error occurred:", http_err)
          return json.dumps({"error": "HTTP error"})

    return output


@app.route('/receive_for_improvement', methods=['POST'])
def receive_for_improvement():
    """Endpoint to receive progress_on_quiz."""
    try:
        progress_on_quiz = request.json.get("progress_on_quiz", "")
        if not progress_on_quiz:
            return jsonify({"error": "Progress_on_quiz is required"}), 400

        # Generate the quiz
        new_array = [{k: v for k, v in item.items() if k != "topicId"} for item in progress_on_quiz]
        prompt = gen_prompt_for_improvement().replace("${ new_array }" , str(new_array))
        response = get_model_response_improvement(prompt)

        return jsonify(response), 200
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "An error occurred while processing progress."}), 500
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
