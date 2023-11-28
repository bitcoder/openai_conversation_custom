import base64


def compose_payload(image_path: str, prompt: str) -> dict:
    """
    Composes a payload dictionary with a base64 encoded image and a text prompt for the GPT-4 Vision model.

    Args:
        image_path (str): The image path to encode and include in the payload.
        prompt (str): The prompt text to accompany the image in the payload.

    Returns:
        dict: A dictionary structured as a payload for the GPT-4 Vision model, including the model name,
              an array of messages each containing a role and content with text and the base64 encoded image,
              and the maximum number of tokens to generate.
    """

    # encode a base64 image from the image path
    with open(image_path, "rb") as image_file:
         encoded_string = base64.b64encode(image_file.read())
         base64_image = encoded_string.decode('utf-8')

    return {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 600
    }

def compose_text_payload(prompt: str) -> dict:
    """
    Composes a payload dictionary with a base64 encoded image and a text prompt the standard text model.

    Args:
        prompt (str): The prompt text to accompany the image in the payload.

    Returns:
        dict: A dictionary structured as a payload for the GPT-4 Vision model, including the model name,
              an array of messages each containing a role and content with text and the base64 encoded image,
              and the maximum number of tokens to generate.
    """
    return {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        "max_tokens": 600
    }
