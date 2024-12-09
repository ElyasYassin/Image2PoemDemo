from flask import Flask, request, jsonify, render_template, send_file
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoProcessor, BlipForConditionalGeneration
import torch
from PIL import Image
from parler_tts import ParlerTTSForConditionalGeneration
import soundfile as sf
import os

app = Flask(__name__)

# We'll first load the models that were trained
blip_processor = AutoProcessor.from_pretrained("models/finetuned_blip_captioning_model_new")
blip_model = BlipForConditionalGeneration.from_pretrained("models/finetuned_blip_captioning_model_new")

poem_model = GPT2LMHeadModel.from_pretrained("models/gpt2_haiku_model/checkpoint-4227")
poem_tokenizer = GPT2Tokenizer.from_pretrained("models/gpt2_haiku_model/checkpoint-4227")

# In case we have a GPU, we'll move the models to that
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
blip_model.to(device)
poem_model.to(device)

# Create a path for the temporary audio
TEMP_AUDIO_PATH = "temp_audio"
os.makedirs(TEMP_AUDIO_PATH, exist_ok=True)

# The generated poem is seperated by '/' for each line and '$' at the end of the poem
def split_haiku(haiku):
    haiku = haiku.split('$')[0]
    lines = [line.strip() for line in haiku.split('/')]
    return lines

# We'll generate the text to speech audio here
def generate_tts_audio(haiku_text, output_path):
    tts_model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler_tts_mini_v0.1").to(device)
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler_tts_mini_v0.1")

    description = "A female speaker with a slightly low-pitched, very expressive voice delivers her words at a normal pace in a poetic but very slow manner with proper pauses while speaking inside a confined space with very clear audio"

    input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(haiku_text, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        generation = tts_model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    audio_arr = generation.cpu().numpy().squeeze()

    sf.write(output_path, audio_arr, tts_model.config.sampling_rate)

# Basic route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# POST request since we're inputting an image
@app.route('/generate-poem', methods=['POST'])
def generate_poem():
    # We'll take the image and save it as uploaded_image.jpg
    image_file = request.files['image']
    image = Image.open(image_file).convert("RGB")
    saved_image_path = os.path.join(TEMP_AUDIO_PATH, "uploaded_image.jpg")

    # Get the first line of the poem from the fine-tuned blip model
    blip_inputs = blip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        generated_ids = blip_model.generate(**blip_inputs, max_length=10, num_beams=5, no_repeat_ngram_size=2)
    caption = blip_processor.batch_decode(generated_ids)[0]

    # Generate the rest of the poem using the fine-tuned GPT-2 Model
    poem_inputs = poem_tokenizer.encode(caption, return_tensors='pt').to(device)
    with torch.no_grad():
        poem_output = poem_model.generate(poem_inputs, max_length=150, num_return_sequences=1, no_repeat_ngram_size=1, temperature=1, top_k=5)
    haiku = poem_tokenizer.decode(poem_output[0], skip_special_tokens=True)
    
    # Split the poem and display it
    haiku_lines = split_haiku(haiku)
    # Create the formatting for a proper display of the poem
    formatted_haiku = "\n".join(haiku_lines)

    audio_file_path = os.path.join(TEMP_AUDIO_PATH, "parler_tts_out.wav")
    generate_tts_audio(formatted_haiku, audio_file_path)

    # Display poem, image and audio
    return jsonify({
        'poem': formatted_haiku,
        'audio_url': f'/audio/{os.path.basename(audio_file_path)}',
        'image_url': f'/static/{os.path.basename(saved_image_path)}'  
    })

# Serve the audio file
@app.route('/audio/<filename>')
def serve_audio(filename):
    return send_file(os.path.join(TEMP_AUDIO_PATH, filename), as_attachment=False)

if __name__ == '__main__':
    app.run(debug=True)