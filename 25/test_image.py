#pip install pillow
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import base64

client = genai.Client(
    api_key='AIzaSyA6_LkslJ6VUQdZp4HeQfGmRei_2jQJlxI'
)

contents = ('Hi, can you create an image of akara '
            'in a glass bowl with a lid, '
            'on table?')

response = client.models.generate_content(
    model="gemini-2.0-flash-preview-image-generation",
    contents=contents,
    config=types.GenerateContentConfig(
      response_modalities=['TEXT', 'IMAGE']
    )
)

for part in response.candidates[0].content.parts:
  if part.text is not None:
    print(part.text)
  elif part.inline_data is not None:
    image = Image.open(BytesIO((part.inline_data.data)))
    image.save('gemini-gen-image.png')
    image.show()